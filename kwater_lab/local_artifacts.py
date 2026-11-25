"""Index local-only competition artifacts by relative name, byte length and SHA-256."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from urllib.parse import urlparse

from .train_head import save_atomic


LOCAL_REPORTS = (
    "clip_probe.json", "visual_probe.json", "taxonomy_followup.json", "prototype.json",
    "prevention_replay.json", "public_law_replay.json", "public_law_replay_review.json",
    "train_audit_full.json", "train_audit_reviewed.json",
)
TEMPORARY_SUFFIXES = {".pyc", ".copying", ".partial", ".part", ".crdownload", ".tmp", ".assembling"}


def is_link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0)
                                               & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def aihub_package(path, data):
    return (path.suffix.lower() == ".zip" and path.parent.name == "packages"
            and path.parent.parent.parent == data / "external/aihub")


def eligible_files(root):
    data = root / "data"
    if not data.is_dir() or is_link(data):
        raise ValueError("Expected a local data directory")
    for base, dirs, files in os.walk(data, followlinks=False):
        base = Path(base)
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not is_link(base / d))
        for name in files:
            p = base / name
            if p.is_relative_to(data / "external/aihub") and name in {"transfer.log", "transfer-status.json"}:
                continue
            if aihub_package(p, data) and not p.with_suffix(".zip.json").is_file():
                continue
            if p.suffix.lower() not in TEMPORARY_SUFFIXES and not is_link(p) and p.is_file():
                yield p
    for name in LOCAL_REPORTS:
        p = root / "results" / name
        if p.is_file() and not is_link(p):
            yield p
    for p in (root / "results").glob("*/submission.jsonl"):
        if not is_link(p.parent) and not is_link(p):
            yield p


def derived_output(path, root):
    parts = path.relative_to(root).parts
    if parts[0] == "results":
        return True
    if parts[1] == "external":
        return path.parent.name != "packages" and path.suffix.lower() in {".json", ".jsonl", ".csv", ".npz", ".txt"}
    return parts[1] not in {"packages", "baseline", "local-kit", "train", "encoders"}


def inventory(root, previous=None, *, refresh_derived=False):
    recorded = {r["path"]: r for r in previous["artifacts"]} if previous else {}
    rows = []
    for p in sorted(eligible_files(root)):
        initial = p.stat()
        size = initial.st_size
        name = p.relative_to(root).as_posix()
        existing = recorded.pop(name, None)
        if existing is not None and not (refresh_derived and derived_output(p, root)):
            if size != existing["bytes"]:
                raise ValueError(f"Previously recorded file changed size; run a full index: {name}")
            rows.append(existing)
            continue
        if previous and aihub_package(p, root / "data"):
            receipt = json.loads(p.with_suffix(".zip.json").read_text(encoding="utf-8"))
            checksum = receipt["sha256"]
            if (receipt["bytes"] != size or not re.fullmatch(r"[a-f0-9]{64}", checksum)
                    or receipt["dataset"] != p.parent.parent.name or receipt["zip_members"] < 1
                    or p.name != f"{receipt['filekey']}-{receipt['name']}"):
                raise ValueError(f"Completed package and its download receipt differ: {name}")
        elif (previous and name == "data/external/floodnet/FloodNet.zip"
              and p.with_suffix(".zip.json").is_file() and initial.st_mtime_ns <= p.with_suffix(".zip.json").stat().st_mtime_ns):
            receipt = json.loads(p.with_suffix(".zip.json").read_text(encoding="utf-8"))
            checksum = receipt["sha256"]
            source = urlparse(receipt["source_url"])
            if (receipt["bytes"] != size or not re.fullmatch("[a-f0-9]{64}", checksum)
                    or source.scheme != "https" or source.hostname != "www.dropbox.com"):
                raise ValueError("FloodNet package differs from its acquisition receipt")
        else:
            with p.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        final = p.stat()
        if (final.st_size, final.st_mtime_ns) != (size, initial.st_mtime_ns):
            raise RuntimeError(f"File changed during hash: {p.name}")
        rows.append({"path": name, "bytes": size, "sha256": checksum})
    if recorded:
        raise ValueError("Previously recorded files are missing. Inspect the changes before refreshing the index.")
    return {"algorithm": "SHA-256", "files": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
            "scope": "Completed project data files and local-only research outputs. In-progress downloads, live transfer status and environment caches are excluded.",
            "artifacts": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Rehash and compare without changing the existing index")
    modes.add_argument("--append-new", action="store_true",
                       help="Hash new outputs only; keep existing digests, checking their presence and size. Use --check to reverify bytes.")
    modes.add_argument("--refresh-derived", action="store_true",
                       help="Rehash local research outputs, including same-sized revisions, and add newly verified inputs.")
    args = parser.parse_args()
    root = args.root.resolve()
    dest = root / "results/local_artifacts.json"
    previous = json.loads(dest.read_text(encoding="utf-8")) if args.append_new or args.refresh_derived else None
    report = inventory(root, previous, refresh_derived=args.refresh_derived)
    if args.check:
        if report != json.loads(dest.read_text(encoding="utf-8")):
            raise SystemExit("Local inventory differs from the recorded commit")
    else:
        save_atomic(dest, lambda f: f.write((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()))
    print(f"{report['files']} local artifacts; {report['total_bytes']} bytes")


if __name__ == "__main__":
    main()
