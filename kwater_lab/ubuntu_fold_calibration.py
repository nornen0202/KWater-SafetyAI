"""Seal Fold2, lock one shared threshold, then evaluate Fold3 from one new adapter.

Run: .venv/bin/python -m kwater_lab.ubuntu_fold_calibration preflight|run
All per-photo data stays under data/. Only aggregate reports may be exported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from . import ubuntu_calibration_protocol as protocol
from . import ubuntu_inner_verifier_eval as verifier
from .ubuntu_calibration_stats import compare
from .ubuntu_focused_eval import write_once

SOURCE_FILES: Final = (
    "ubuntu_fold_calibration.py",
    "ubuntu_calibration_protocol.py",
    "ubuntu_calibration_stats.py",
    "ubuntu_calibration_train.py",
    "ubuntu_calibration_receipt.py",
    "ubuntu_focused_curriculum.py",
    "ubuntu_focused_eval.py",
    "ubuntu_inner_verifier_eval.py",
    "ubuntu_roi_readout.py",
    "ubuntu_image_alignment.py",
    "vlm_11code.py",
    "metrics.py",
)


@dataclass(frozen=True, slots=True)
class Paths:
    root: Path
    training: Path
    output: Path
    report: Path

    @property
    def base(self) -> Path:
        return self.root / "data/baseline/base_model"

    @property
    def adapter(self) -> Path:
        return self.training / "training/checkpoint-12"

    @property
    def gold(self) -> Path:
        return self.root / "data/train/train.jsonl"


def private_output(paths: Paths) -> None:
    """Require local restricted cache storage and a separate aggregate destination."""
    if not paths.output.resolve().is_relative_to((paths.root / "data").resolve()):
        raise verifier.EvaluationError(
            "Predictions, receipts and locks must stay under private data/"
        )


def preflight(paths: Paths) -> None:
    """Check real metadata/images on CPU without opening gold labels or any model."""
    private_output(paths)
    cohorts = protocol.load_cohorts(paths.root)
    report = {
        "scope": "CPU metadata-only preflight; no gold targets accessed, no GPU/model load",
        "protocol": protocol.PROTOCOL,
        "protocol_sha256": verifier.stable_hash(protocol.PROTOCOL),
        "manifest_sha256": verifier.MANIFEST_SHA,
        "photos_by_fold": {str(fold): len(photos) for fold, photos in cohorts.items()},
        "groups_by_fold": {
            str(fold): len({p.group_id for p in photos})
            for fold, photos in cohorts.items()
        },
        "conditions_by_fold": {
            str(fold): len(photos) * len(verifier.CODES)
            for fold, photos in cohorts.items()
            if fold in (2, 3)
        },
        "protected_identity_overlaps": 0,
        "all_image_hashes_match": True,
        "checkpoint_complete": all(
            (paths.adapter / name).is_file()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ),
        "training_receipt_required_before_prediction": True,
    }
    write_once(paths.report, report)


def initialize(paths: Paths) -> Path:
    """Validate completed Fold1 training and freeze one adapter/runtime identity."""
    import torch
    from torch.version import cuda

    from .ubuntu_calibration_receipt import verify_training_receipt
    from .ubuntu_focused_curriculum import decision_token_ids
    from .vlm_11code import load_qwen_processor

    private_output(paths)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    verify_training_receipt(paths.root, paths.training)
    receipt = protocol.read_json(paths.training / "training-verification.json")
    if (
        receipt["folds"] != [1]
        or receipt["optimizer_steps"] != 12
        or receipt["examples"] != 96
    ):
        raise verifier.EvaluationError(
            "Receipt does not prove the prespecified Fold1-only training"
        )
    cohorts = protocol.load_cohorts(paths.root)
    processor = load_qwen_processor(paths.base, verifier.MAX_PIXELS)
    adapter = {
        name: verifier.digest(paths.adapter / name)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    if any(
        receipt["checkpoint_files"].get(name) != sha for name, sha in adapter.items()
    ):
        raise verifier.EvaluationError("Training receipt checkpoint differs")
    identity: protocol.Identity = {
        "schema": protocol.SCHEMA,
        "protocol_sha256": verifier.stable_hash(protocol.PROTOCOL),
        "manifest_sha256": verifier.MANIFEST_SHA,
        "gold_sha256": verifier.GOLD_SHA,
        "training_receipt_sha256": verifier.digest(
            paths.training / "training-verification.json"
        ),
        "cohorts": {
            str(fold): protocol.cohort_records(photos)
            for fold, photos in cohorts.items()
        },
        "source": {
            name: verifier.digest(Path(__file__).with_name(name))
            for name in SOURCE_FILES
        },
        "runtime": {
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "bitsandbytes", "Pillow")
            },
            "cuda": str(cuda),
            "device": torch.cuda.get_device_name(0),
            "precision": "NF4 double quant bf16; existing inner12 loader visual/lm_head exclusions",
        },
        "adapter": adapter,
        "base": {
            path.name: verifier.digest(path)
            for path in sorted(paths.base.iterdir())
            if path.is_file()
        },
        "prompt_sha256": verifier.digest(
            paths.root / "kwater_lab/system_prompt_11code.txt"
        ),
        "processor_sha256": verifier.stable_hash(
            processor.image_processor.to_json_string()
        ),
        "decision_ids": list(decision_token_ids(processor.tokenizer)),
    }
    if identity["base"] != receipt["base_files"]:
        raise verifier.EvaluationError(
            "Official base files differ from training receipt"
        )
    run = paths.output / verifier.stable_hash(identity)[:24]
    write_once(run / "protocol.json", protocol.PROTOCOL)
    write_once(run / "identity.json", identity)
    return run


def predict_fold(paths: Paths, run: Path, fold: int) -> None:
    """Resume keyed atomic single-photo caches; Fold3 requires a preexisting lock."""
    identity = protocol.verify_identity(run)
    private_output(paths)
    stage = run / f"fold{fold}"
    if fold == 3:
        protocol.verify_lock(run)
        write_once(
            stage / "start.json",
            {"threshold_lock_sha256": verifier.digest(run / "threshold-lock.json")},
        )
    elif fold != 2:
        raise verifier.EvaluationError("Inference is restricted to Fold2 and Fold3")
    if (stage / "seal.json").exists():
        protocol.verify_stage(run, fold)
        return
    for name, expected in identity["adapter"].items():
        if verifier.digest(paths.adapter / name) != expected:
            raise verifier.EvaluationError("Frozen adapter changed before inference")
    for name, expected in identity["base"].items():
        if verifier.digest(paths.base / name) != expected:
            raise verifier.EvaluationError("Frozen base changed before inference")
    if (
        verifier.digest(paths.root / "kwater_lab/system_prompt_11code.txt")
        != identity["prompt_sha256"]
    ):
        raise verifier.EvaluationError("Frozen prompt changed before inference")

    import torch

    from .ubuntu_focused_curriculum import Decision, Encoder, decision_token_ids
    from .vlm_11code import load_qwen_model, load_qwen_processor

    processor = load_qwen_processor(paths.base, verifier.MAX_PIXELS)
    token_ids = decision_token_ids(processor.tokenizer)
    if (
        list(token_ids) != identity["decision_ids"]
        or verifier.stable_hash(processor.image_processor.to_json_string())
        != identity["processor_sha256"]
    ):
        raise verifier.EvaluationError("Frozen processor changed before inference")
    encoder = Encoder(
        processor,
        paths.root,
        (paths.root / "kwater_lab/system_prompt_11code.txt").read_text(),
    )
    fingerprint = verifier.stable_hash(identity)
    records: list[verifier.Margin] = []
    model = None
    photos = protocol.photos_for(identity, fold)
    for number, photo in enumerate(photos, 1):
        key = verifier.stable_hash((fingerprint, fold, asdict(photo)))
        cache = stage / "cache" / f"{key}.json"
        rows = verifier.load_cached_photo(cache, key, photo)
        if rows is None:
            if model is None:
                torch.manual_seed(verifier.SEED)
                model = load_qwen_model(
                    paths.base, use_4bit=True, adapter_dir=paths.adapter
                ).eval()
            fresh: list[verifier.Margin] = []
            for code in verifier.CODES:
                example = Decision(
                    photo.sample_id,
                    photo.image,
                    photo.group_id,
                    fold,
                    code,
                    False,
                    "normal",
                )
                batch = encoder.encode(example)
                tensor_hashes = {
                    name: {
                        "sha256": hashlib.sha256(
                            tensor.contiguous().numpy().tobytes()
                        ).hexdigest(),
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                    }
                    for name, tensor in batch.items()
                }
                input_sha = verifier.stable_hash((tensor_hashes, list(token_ids)))
                with torch.inference_mode():
                    logits = (
                        model(**batch.to("cuda"), logits_to_keep=1, use_cache=False)
                        .logits[0, -1]
                        .float()
                    )
                    margin = float((logits[token_ids[0]] - logits[token_ids[1]]).item())
                fresh.append(verifier.Margin(photo.sample_id, code, margin, input_sha))
                del batch, logits
            rows = verifier.parse_margins([row.to_record() for row in fresh], (photo,))
            payload = [row.to_record() for row in rows]
            write_once(
                cache,
                {
                    "cache_key": key,
                    "rows": payload,
                    "rows_sha256": verifier.stable_hash(payload),
                },
            )
        records.extend(rows)
        print(f"fold{fold} {number}/{len(photos)} photos cached", flush=True)
    write_once(stage / "predictions.json", [row.to_record() for row in records])
    protocol.seal_stage(run, fold)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def score(run: Path, gold: Path, report_path: Path) -> None:
    """Produce aggregate-only paired statistics after both seals and the lock verify."""
    lock = protocol.verify_lock(run)
    photos = protocol.verify_stage(run, 3)
    labels = protocol.load_labels(run, 3, gold)
    rows = verifier.parse_margins(
        protocol.read_json(run / "fold3/predictions.json"), photos
    )
    report = {
        **compare(photos, labels, rows, lock["selection"]["threshold"]),
        "calibration": lock["selection"],
        "protocol": protocol.PROTOCOL,
        "identity_sha256": verifier.digest(run / "identity.json"),
        "threshold_lock_sha256": verifier.digest(run / "threshold-lock.json"),
        "fold2_seal_sha256": verifier.digest(run / "fold2/seal.json"),
        "fold3_seal_sha256": verifier.digest(run / "fold3/seal.json"),
        "training_receipt_sha256": protocol.verify_identity(run)[
            "training_receipt_sha256"
        ],
        "severity4_definition": "Severity4 candidate-code photos; complete miss means no true severity4 candidate detected; no predicted severity grading",
    }
    write_once(report_path, report)


def main() -> None:
    """Run the fixed sequence; scoring alone cannot create or tune a threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "score"))
    parser.add_argument("--root", type=Path, default=verifier.ROOT)
    parser.add_argument(
        "--training",
        type=Path,
        default=verifier.ROOT / "data/ubuntu3050-runs/qlora-fold1-calibration",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=verifier.ROOT / "data/ubuntu3050-runs/fold1-calibration-eval",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    paths = Paths(args.root, args.training, args.out, args.report)
    match args.command:
        case "preflight":
            preflight(paths)
        case "run":
            run = initialize(paths)
            predict_fold(paths, run, 2)
            protocol.fit_threshold(run, paths.gold)
            predict_fold(paths, run, 3)
            score(run, paths.gold, paths.report)
            print(f"SEALED_RUN={run}", flush=True)
        case "score":
            if args.run is None:
                raise verifier.EvaluationError("score requires --run")
            score(args.run, paths.gold, paths.report)
    print(f"AGGREGATE_REPORT={paths.report}", flush=True)


if __name__ == "__main__":
    main()
