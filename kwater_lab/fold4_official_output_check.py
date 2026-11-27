"""Check the three Fold 4 prediction files against the official output kit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .metrics import read_index
from .offline_archive_audit import run_output_checker

ARMS = ("clean_8code_baseline", "v3_11code_standalone", "v3_hybrid_margin_verified")
DEFAULT_PREDICTIONS = Path("data/local-eval-predictions/fold4-v3-benchmark")
DEFAULT_MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
EXPECTED_ROWS = 175
EXPECTED_GROUPS = 39


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def check_fold4(manifest, predictions, kit, workspace, python):
    rows = [r for r in json.loads(manifest.read_bytes()) if r["split"] == "dev" and r["dev_fold"] == 4]
    mapping = {
        r["sample_id"]: hashlib.sha256(f"kwater/fold4-official-check/{r['sample_id']}".encode()).hexdigest()[:32]
        for r in rows
    }
    if len(rows) != EXPECTED_ROWS or len(mapping) != EXPECTED_ROWS or len(set(mapping.values())) != EXPECTED_ROWS:
        raise ValueError("Expected exactly 175 unique Fold 4 photos and archive IDs")
    if len({r["group_id"] for r in rows}) != EXPECTED_GROUPS:
        raise ValueError("Expected exactly 39 Fold 4 groups")

    originals = {}
    hashes = {}
    for arm in ARMS:
        path = predictions / f"{arm}.jsonl"
        before = sha256_file(path)
        index = read_index(path, lambda row: (row["image_id"], row), limit_size=True)
        if before != sha256_file(path):
            raise ValueError(f"{arm}: prediction file changed during preflight")
        if index.keys() != mapping.keys():
            raise ValueError(f"{arm}: expected exact Fold 4 ID coverage, got {len(index.keys() & mapping.keys())}/{EXPECTED_ROWS}")
        originals[arm] = index
        hashes[arm] = before

    workspace.mkdir(parents=True, exist_ok=False)
    inputs = workspace / "input"
    inputs.mkdir()
    write_jsonl(inputs / "manifest.jsonl", ({"image_id": value} for value in mapping.values()))
    (workspace / "id-map-local.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    results = {}
    for arm, index in originals.items():
        output = workspace / arm
        output.mkdir()
        submission = output / "submission.jsonl"
        write_jsonl(submission, ({**index[sid], "image_id": hex_id} for sid, hex_id in mapping.items()))
        check = run_output_checker(kit, inputs, output, python)
        results[arm] = {
            **check,
            "passed": check["all_rows_valid"] and check["valid_rows"] == EXPECTED_ROWS,
            "original_jsonl_sha256": hashes[arm],
            "mapped_jsonl_sha256": sha256_file(submission),
        }
    return {
        "photos": EXPECTED_ROWS,
        "groups": EXPECTED_GROUPS,
        "manifest_sha256": sha256_file(manifest),
        "official_checker_sha256": sha256_file(kit / "check_output.py"),
        "official_schema_sha256": sha256_file(kit / "prediction.schema.json"),
        "arms": results,
        "all_arms_passed": all(r["passed"] for r in results.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--kit", type=Path, default=Path("data/local-kit/kit"))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checker-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if not args.workspace.resolve().is_relative_to(Path("data").resolve()):
        parser.error("--workspace must be inside the local data directory")
    if args.out.exists():
        parser.error("--out must be a new aggregate file")
    summary = check_fold4(args.manifest, args.predictions, args.kit, args.workspace, args.checker_python)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return 0 if summary["all_arms_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
