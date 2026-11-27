"""Build the completed Fold 4 eight-code comparison adapter on the official runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from zipfile import ZIP_STORED, ZipFile

from .metrics import unique_keys
from .offline_archive_audit import entries

MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
LABELS = Path("data/train/train.jsonl")
ORIGINAL_ZIP = Path("data/packages/kwater-baseline.zip")
ADAPTER = Path("data/v3-baseline8-fold4/adapter")
PREFIX = "adapter/"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_keys)


def dump(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()


def case_bank(manifest, labels, official):
    scope = {row["sample_id"] for row in manifest if row["split"] == "dev" and row["dev_fold"] != 4}
    fold = {row["sample_id"] for row in manifest if row["split"] == "dev" and row["dev_fold"] == 4}
    if (len(scope), len(fold)) != (791, 175) or scope & fold:
        raise ValueError("The clean Fold 4 manifest is required")
    valid_labels = {row["sample_id"] for row in labels if row["sample_id"] in scope}
    cases = [row for row in official if row["sample_id"] in valid_labels]
    if len(valid_labels) != 791 or len(cases) != 790 or len({row["sample_id"] for row in cases}) != 790:
        raise ValueError("Expected exactly 790 official representable training cases")
    return cases


def one_replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected official runtime shape around {old}")
    return text.replace(old, new)


def build(original, adapter, manifest, output):
    summary = read(adapter.parent / "training_summary.json")
    adapter_weights = adapter / "adapter_model.safetensors"
    weight_sha = digest(adapter_weights)
    if (weight_sha != summary["adapter_safetensors_sha256"] or summary["optimizer_steps"] != 100
        or summary["epochs"] != 2 or summary["val_fold"] != 4 or summary["train_photos"] != 791 or summary["curriculum_examples"] != 790):
        raise ValueError("Expected the completed 100-step clean Fold 4 eight-code adapter")
    labels = [json.loads(line, object_pairs_hook=unique_keys) for line in LABELS.read_bytes().splitlines() if line]
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    if not output.parent.is_dir():
        raise ValueError("The archive output directory must exist")

    with ZipFile(original) as source:
        entries(source)  # reject unsafe members, collisions, encryption and unexpected ZIP sizes
        main = source.read("main.py").decode()
        main = one_replace(main, 'os.environ.get("KW_BATCH", "16")', 'os.environ.get("KW_BATCH", "4")')
        main = one_replace(main, "max_new_tokens=900", "max_new_tokens=620")
        compile(main, "main.py", "exec")
        cases = case_bank(read(manifest), labels, [json.loads(line) for line in source.read("assets/train.jsonl").splitlines() if line])
        bank = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in cases)
        assets = json.loads(source.read("assets.json"))
        model_asset, data_asset = assets["assets"][1:3]
        model_asset.update(name="Fold 4 대조용 위험 판정 LoRA (8종, 2에폭)",
                           source="kwater_lab/v3_turnaround.py --step train-8code-fold4, 완료 100스텝",
                           revision=f"sha256:{weight_sha}")
        data_asset.update(source="대회 배포 kwater-train.zip과 동봉 주최자 법령집",
                          revision=f"개발 Fold 0–3의 8종 변환 가능 790장; manifest sha256:{digest(manifest)}",
                          use="공식 배포 사진만으로 LoRA 학습; 같은 790 사례의 조문 검색. 사진, 평가 행, 외부 데이터 미포함")
        patches = {"assets.json": dump(assets), "assets/train.jsonl": bank, "main.py": main.encode()}
        files = {PREFIX + name: adapter / name for name in ("adapter_config.json", "adapter_model.safetensors")}
        with partial.open("xb") as staging:
            try:
                with ZipFile(staging, "w", compression=ZIP_STORED, allowZip64=True) as dest:
                    for entry in source.infolist():
                        name = entry.filename
                        if name.startswith(PREFIX):
                            continue
                        if name in patches:
                            dest.writestr(name, patches[name])
                        else:
                            with source.open(entry) as src, dest.open(name, "w", force_zip64=True) as dst:
                                shutil.copyfileobj(src, dst, length=1024 * 1024)
                    for name, path in files.items():
                        dest.write(path, name)
                staging.flush()
                if staging.tell() > 20_000_000_000:
                    raise ValueError("Candidate exceeds the 20 GB archive limit")
            except BaseException:
                staging.close()
                partial.unlink(missing_ok=True)
                raise
    os.rename(partial, output)
    return {"archive_bytes": output.stat().st_size, "adapter_sha256": weight_sha,
            "manifest_sha256": digest(manifest), "law_cases": len(cases),
            "law_cases_sha256": hashlib.sha256(bank).hexdigest(), "batch_size": 4, "max_new_tokens": 620}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL_ZIP)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.original, args.adapter, args.manifest, args.zip), indent=2))


if __name__ == "__main__":
    main()
