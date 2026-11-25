"""Index local-only competition artifacts by relative name, byte length and SHA-256."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from .train_head import save_atomic


LOCAL_REPORTS = (
    "clip_probe.json", "visual_probe.json", "taxonomy_followup.json", "prototype.json",
    "prevention_replay.json", "public_law_replay.json", "public_law_replay_review.json",
    "train_audit_full.json", "train_audit_reviewed.json",
)


def is_link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0)
                                               & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def eligible_files(root):
    data = root / "data"
    if not data.is_dir() or is_link(data):
        raise ValueError("Expected a local data directory")
    for base, dirs, files in os.walk(data, followlinks=False):
        base = Path(base)
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not is_link(base / d))
        for name in files:
            p = base / name
            if p.suffix not in (".pyc", ".copying") and not is_link(p) and p.is_file():
                yield p
    for name in LOCAL_REPORTS:
        p = root / "results" / name
        if p.is_file() and not is_link(p):
            yield p
    for p in (root / "results").glob("*/submission.jsonl"):
        if not is_link(p.parent) and not is_link(p):
            yield p


def inventory(root):
    rows = []
    for p in sorted(eligible_files(root)):
        size = p.stat().st_size
        with p.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if p.stat().st_size != size:
            raise RuntimeError(f"Size changed during hash: {p.name}")
        rows.append({"path": p.relative_to(root).as_posix(), "bytes": size, "sha256": checksum})
    return {"algorithm": "SHA-256", "files": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
            "scope": "Project data and local-only research outputs. Environment caches are excluded.",
            "artifacts": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Compare without changing the existing index")
    args = parser.parse_args()
    root = args.root.resolve()
    dest = root / "results/local_artifacts.json"
    report = inventory(root)
    if args.check:
        if report != json.loads(dest.read_text(encoding="utf-8")):
            raise SystemExit("Local inventory differs from the recorded commit")
    else:
        save_atomic(dest, lambda f: f.write((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()))
    print(f"{report['files']} local artifacts; {report['total_bytes']} bytes")


if __name__ == "__main__":
    main()
