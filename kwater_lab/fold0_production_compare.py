"""Stage official fold-0 inputs and compare full-coverage production outputs locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .metrics import decode_target, evaluate, paired_bootstrap, read_index


EXPECTED_FOLD0_PHOTOS = 216


def fold_rows(path: Path) -> list[dict]:
    rows = [r for r in json.loads(path.read_text(encoding="utf-8")) if r["split"] == "dev" and r["dev_fold"] == 0]
    if len(rows) != EXPECTED_FOLD0_PHOTOS or len({r["sample_id"] for r in rows}) != EXPECTED_FOLD0_PHOTOS:
        raise ValueError("This benchmark requires exactly 216 unique fold-0 photos")
    return rows


def image_id(sample_id: str) -> str:
    return hashlib.sha256(f"kwater/fold0-production/{sample_id}".encode()).hexdigest()[:32]


def prepare_inputs(rows: list[dict], train_dir: Path, output: Path) -> None:
    sources = [train_dir / r["image"] for r in rows]
    if not rows or not all(p.is_file() for p in sources):
        raise ValueError("Fold 0 has no inputs or is missing image files")
    for row, source in zip(rows, sources, strict=True):
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError("Training image differs from its audited fold manifest hash")
    (output / "images").mkdir(parents=True, exist_ok=False)
    for row, source in zip(rows, sources, strict=True):
        os.link(source, output / "images" / f"{image_id(row['sample_id'])}{source.suffix}")
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps({"image_id": image_id(r["sample_id"])}) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )


def compare(rows: list[dict], baseline_path: Path, candidate_path: Path) -> dict:
    baseline_anon = read_index(baseline_path, decode_target, limit_size=True)
    candidate = read_index(candidate_path, lambda r: decode_target(r, local_ids=True), limit_size=True)
    baseline_ids = {image_id(r["sample_id"]): r["sample_id"] for r in rows}
    if baseline_anon.keys() != baseline_ids.keys():
        raise ValueError(f"Baseline covers {len(baseline_anon.keys() & baseline_ids.keys())}/{len(rows)} expected images")
    baseline = {baseline_ids[k]: v for k, v in baseline_anon.items()}
    if baseline.keys() != candidate.keys():
        raise ValueError(f"Candidate covers {len(candidate.keys() & baseline.keys())}/{len(rows)} expected images")
    gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    groups = {r["sample_id"]: r["group_id"] for r in rows}
    return {
        "protocol": "Fold 0, exact image coverage. Official production LoRA used official training data and is an exposed reference.",
        "photos": len(rows),
        "previous_local_exposure_photos": sum(bool(r.get("exposed_prior")) for r in rows),
        "official_production_lora": evaluate(gold, baseline, groups),
        "candidate": evaluate(gold, candidate, groups),
        "candidate_vs_official_production_lora": paired_bootstrap(gold, baseline, candidate, groups, resamples=2000, seed=42),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_dev.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--train-dir", type=Path, default=Path("data/train"))
    prepare.add_argument("--output", type=Path, required=True)
    evaluate_parser = commands.add_parser("compare")
    evaluate_parser.add_argument("--baseline", type=Path, required=True)
    evaluate_parser.add_argument("--candidate", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = fold_rows(args.manifest)
    if args.command == "prepare":
        prepare_inputs(rows, args.train_dir, args.output)
        print(f"Staged {len(rows)} photos")
    else:
        result = compare(rows, args.baseline, args.candidate)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
