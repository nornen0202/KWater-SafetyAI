"""Count local inputs for Linux fold-0 inference and a separate corrected training run."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from kwater_lab.fold0_official_resume import read_inputs
from kwater_lab.fold0_production_compare import fold_rows, image_id


def plan(root: Path, verify_inputs: bool) -> dict:
    data = root / "data"
    dev_manifest = data / "clean-split-v1/split_manifest_dev.json"
    dev = json.loads(dev_manifest.read_text(encoding="utf-8"))
    fold0 = fold_rows(dev_manifest)
    training = [r for r in dev if r["split"] == "dev" and r["dev_fold"] != 0]
    if not training or {r["group_id"] for r in training} & {r["group_id"] for r in fold0}:
        raise ValueError("Expected disjoint fold-0 training and evaluation incident groups")
    archive = data / "packages/kwater-baseline.zip"
    with ZipFile(archive) as bundle:
        full_bytes = sum(i.file_size for i in bundle.infolist())
        model_bytes = sum(i.file_size for i in bundle.infolist() if i.filename.startswith("base_model/"))
    sample = data / "local-kit/sample-input"
    official_fold = data / "fold0-production-input"
    sample_inputs = read_inputs(sample)
    fold_inputs = read_inputs(official_fold)
    for row, item in zip(fold0, fold_inputs, strict=True):
        if item["sample_id"] != image_id(row["sample_id"]):
            raise ValueError("Fold-0 input order or image ID differs from the pinned development manifest")
        if verify_inputs:
            with (official_fold / item["image"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError("Fold-0 image bytes differ from the pinned manifest")

    def input_size(path, rows):
        return (path / "manifest.jsonl").stat().st_size + sum((path / item["image"]).stat().st_size for item in rows)

    return {
        "development_manifest_sha256": hashlib.sha256(dev_manifest.read_bytes()).hexdigest(),
        "official_zip_bytes": archive.stat().st_size,
        "official_zip_extracted_bytes": full_bytes,
        "base_model_extracted_bytes": model_bytes,
        "official_kit_zip_bytes": (data / "packages/kwater-local-kit.zip").stat().st_size,
        "official_smoke": {"images": len(sample_inputs), "input_bytes": input_size(sample, sample_inputs)},
        "official_fold0": {"images": len(fold_inputs), "input_bytes": input_size(official_fold, fold_inputs),
                           "input_manifest_sha256": hashlib.sha256((official_fold / "manifest.jsonl").read_bytes()).hexdigest(),
                           "image_sha256_verified": verify_inputs},
        "isolated_corrected_fold0_training": {"images": len(training), "image_bytes":
            sum((data / "train" / row["image"]).stat().st_size for row in training), "evaluation_group_overlap": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(plan(args.root, args.verify_inputs), indent=2))


if __name__ == "__main__":
    main()
