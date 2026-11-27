"""Fingerprint this run's local predictions, checkpoints, measured models and metrics."""

import argparse
import hashlib
import os
from pathlib import Path

from kwater_lab.phase2_panel import PROJECT, RUN, file_hash, read, write


SKIP = {
    "bridge",
    "venv",
    "uv-cache",
    "hf",
    "temp",
    "cuda-cache",
    "torch-cache",
    "fusion-input",
    "sam-extra",
    "sam-kernel",
    "sam-triton-cache",
    "sam-inductor-cache",
    "__pycache__",
}


def build(index=None):
    out = RUN / "artifact-receipts.json"
    known = {}
    download = RUN / "download.log"
    log = (
        download.read_bytes().decode("utf8", errors="replace")
        if download.exists()
        else ""
    )
    if (RUN / "public-sources.json").exists():
        for f in read(RUN / "public-sources.json")["Qwen/Qwen3.5-4B"]["files"]:
            if (
                f.get("lfs", {}).get("sha256")
                and f"verified {f['rfilename']} {f['size']}" in log
            ):
                known[RUN / "qwen35" / f["rfilename"]] = (f["size"], f["lfs"]["sha256"])
    meta = RUN / "regions/models/receipts.json"
    if meta.exists():
        for group, m in read(meta).items():
            for f in m["files"]:
                if "lfs" in f:
                    known[RUN / "regions/models" / group / f["rfilename"]] = (
                        f["size"],
                        f["lfs"]["sha256"],
                    )
    package = RUN / "regions/package.json"
    if package.exists():
        for k, v in read(package)["separate_zip_increments"].items():
            known[RUN / "regions/package-increments" / (k + ".zip")] = (
                v["bytes"],
                v["sha256"],
            )
    for directory in (RUN / "large-models").glob("*"):
        if not directory.is_dir():
            continue
        complete = directory / "complete.json"
        meta = directory / "source.json"
        if complete.is_file():
            done = read(complete)
            if "weight" in done and "sha256" in done:
                weight = directory / done["weight"]
                if (
                    weight.is_file()
                    and weight.stat().st_mtime_ns <= complete.stat().st_mtime_ns
                ):
                    known[weight] = (done["bytes"], done["sha256"])
            elif meta.is_file() and done.get("revision") == read(meta).get("revision"):
                spec = read(meta)
                if done.get("verified_files") == len(spec["files"]):
                    for item in spec["files"]:
                        weight = directory / item["rfilename"]
                        if (
                            "lfs" in item
                            and weight.is_file()
                            and weight.stat().st_mtime_ns <= complete.stat().st_mtime_ns
                        ):
                            known[weight] = (item["size"], item["lfs"]["sha256"])
        for receipt in (directory / "transfers").glob("*.json"):
            part = read(receipt)
            if "weight" in part and "sha256" in part:
                weight = directory / part["weight"]
                if (
                    weight.is_file()
                    and weight.stat().st_mtime_ns <= receipt.stat().st_mtime_ns
                ):
                    known[weight] = (part["bytes"], part["sha256"])
    previous = {}
    if out.exists():
        previous = {r["path"]: r for r in read(out)["artifacts"]}
    items = []
    for root, dirs, files in os.walk(RUN, followlinks=False):
        base = Path(root)
        dirs[:] = [
            d
            for d in dirs
            if d not in SKIP
            and not (base / d).is_symlink()
            and not (base / d).is_junction()
        ]
        for name in sorted(files):
            p = base / name
            if (
                p == out
                or p == RUN / "review-progress.json"
                or p.suffix.lower() in (".partial", ".tmp", ".lock", ".pyc")
                or p.is_symlink()
                or not p.suffix
            ):
                continue
            stat = p.stat()
            rel = p.relative_to(PROJECT).as_posix()
            earlier = previous.get(rel)
            if p in known and known[p][0] == stat.st_size:
                sha = known[p][1]
                origin = "verified local fetch or ZIP receipt"
            elif earlier and (earlier["bytes"], earlier["mtime_ns"]) == (
                stat.st_size,
                stat.st_mtime_ns,
            ):
                sha = earlier["sha256"]
                origin = earlier["verification"]
            else:
                sha = file_hash(p)
                origin = "direct SHA-256"
            if p.stat().st_mtime_ns != stat.st_mtime_ns:
                raise ValueError(f"Changed during hashing: {p}")
            items.append(
                dict(
                    path=rel,
                    bytes=stat.st_size,
                    sha256=sha,
                    mtime_ns=stat.st_mtime_ns,
                    verification=origin,
                )
            )
    items.sort(key=lambda r: r["path"])
    digest = hashlib.sha256(
        "".join(r["path"] + " " + r["sha256"] + "\n" for r in items).encode()
    ).hexdigest()
    write(
        out,
        {
            "run": RUN.name,
            "source_project": str(PROJECT),
            "files": len(items),
            "bytes": sum(r["bytes"] for r in items),
            "path_hashes_sha256": digest,
            "artifacts": items,
        },
    )
    if index:
        current = read(index)
        records = {r["path"]: r for r in current["artifacts"]}
        for r in items:
            records[r["path"]] = {k: r[k] for k in ("path", "bytes", "sha256")}
        rel = out.relative_to(PROJECT).as_posix()
        records[rel] = dict(path=rel, bytes=out.stat().st_size, sha256=file_hash(out))
        current.update(
            files=len(records),
            total_bytes=sum(r["bytes"] for r in records.values()),
            artifacts=[records[k] for k in sorted(records)],
        )
        write(index, current)
    print("MEASURED_ARTIFACTS", len(items), digest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    build(ap.parse_args().index)
