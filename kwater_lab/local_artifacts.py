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

# LFS lengths and hashes at the exact revisions used by the local verification runs.
MODEL_RECEIPTS = {
    "qwen3.5-4b-q4/Qwen3.5-4B-Q4_K_M.gguf": (
        "vlm_candidate_public.json", "e87f176479d0855a907a41277aca2f8ee7a09523", "weights_sha256",
        "Qwen3.5-4B-Q4_K_M.gguf", 2740937888, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
    "qwen3.5-4b-q4/mmproj-F16.gguf": (
        "vlm_candidate_public.json", "e87f176479d0855a907a41277aca2f8ee7a09523", "weights_sha256",
        "mmproj-F16.gguf", 672423616, "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"),
    "siglip2-base-p16-384/vision_model_quantized.onnx": (
        "encoder_pilot_384.json", "99f5d40afb5331b81eecee4bb02c360fbb9cdf8a", "asset_digests",
        "onnx/vision_model_quantized.onnx", 95720693, "9f354d80bbb46b319f46f9e9d0e24ae523e3eb5921aac3d34979d74dbd0db6cd"),
}


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
        if base.parent == data / "phase2-runs":
            # Phase 2 keeps the offline runtime and hard-linked input view beside the measurements.
            transient = {"venv","bridge","uv-cache","cuda-cache","torch-cache","hf","temp","fusion-input"}
            dirs[:] = [d for d in dirs if d not in transient]
        for name in files:
            p = base / name
            if p.is_relative_to(data / "external") and p.suffix.lower() == ".lock":
                continue
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


def submitted_digest(path, root, attributes):
    if path.parent != root / "data/submissions" or path.suffix.lower() != ".zip":
        return None
    accepted = set()
    for record in (root / "results").glob("public_submission_*.json"):
        if attributes.st_mtime_ns > record.stat().st_mtime_ns:
            continue
        receipt = json.loads(record.read_text(encoding="utf-8"))
        if receipt.get("submitted_archive") != path.name:
            continue
        checksum = receipt.get("sha256", "")
        if (receipt.get("bytes") != attributes.st_size or not re.fullmatch(r"[a-f0-9]{64}", checksum)
                or not receipt.get("submitted_at_kst") or not receipt.get("receipt_source")):
            raise ValueError(f"Submitted ZIP and acceptance record differ: {path.name}")
        accepted.add(checksum)
    if len(accepted) > 1:
        raise ValueError(f"Conflicting accepted ZIP fingerprints: {path.name}")
    return next(iter(accepted), None)


def model_digest(path, root, attributes):
    folder = root / "data/encoders"
    if not path.is_relative_to(folder):
        return None
    spec = MODEL_RECEIPTS.get(path.relative_to(folder).as_posix())
    if spec is None:
        return None
    receipt_name, revision, field, key, size, checksum = spec
    record = root / "results" / receipt_name
    if not record.is_file() or attributes.st_mtime_ns > record.stat().st_mtime_ns:
        return None
    receipt = json.loads(record.read_text(encoding="utf-8"))
    if attributes.st_size != size or receipt.get("revision") != revision or receipt.get(field, {}).get(key) != checksum:
        raise ValueError(f"Local model differs from verified revision: {path.name}")
    return checksum


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
        accepted = (submitted_digest(p, root, initial) or model_digest(p, root, initial)) if previous else None
        if accepted:
            checksum = accepted
        elif previous and aihub_package(p, root / "data"):
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
            try:
                with p.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as exc:
                raise OSError(f"Cannot hash local file {name}: {exc}") from exc
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
