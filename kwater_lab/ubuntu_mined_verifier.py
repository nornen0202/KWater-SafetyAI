"""Paired inner-fold spatial hard negatives; local CPU preparation or fresh QLoRA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Final, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

from .metrics import CODES, unique_keys
from .spatial_probe import Row, _fit_scores, digest, encoder_fingerprint, read_cached
from .vlm_11code import (
    KOR_CODE_NAMES,
    build_training_curriculum,
    train_multimodal_qlora,
    verify_user_text,
)

SEED: Final = 20261127
MANIFEST: Final = Path("data/ubuntu3050-runs/qlora-inner12-train.json")
MANIFEST_SHA256: Final = "9897bff9009b78f95f2c58fe50c30c4a18775887dbdd6854c6c012c905d436ac"
DEFAULT_OUTPUT: Final = Path("data/ubuntu3050-runs/qlora-inner12-mined-verifier")


class ManifestRow(TypedDict):
    sample_id: str
    image: str
    sha256: str
    group_id: str
    dev_fold: int
    split: str
    codes: list[str]


class Example(TypedDict):
    sample_id: str
    image: str
    user_text: str
    assistant_text: str
    augment: bool
    kind: str


@dataclass(frozen=True, slots=True)
class MiningBoundaryError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def validate_rows(rows: Sequence[Row]) -> None:
    """Reject protected folds and identities shared by the two mining sides."""
    if {row.fold for row in rows} != {1, 2}:
        raise MiningBoundaryError("Mining requires exactly inner folds 1 and 2")
    if len({row.sample_id for row in rows}) != len(rows):
        raise MiningBoundaryError("Duplicate training sample identity")
    if any(not row.codes <= set(CODES) for row in rows):
        raise MiningBoundaryError("Unknown training code")
    for field in ("group_id", "sha256", "image"):
        left = {getattr(row, field) for row in rows if row.fold == 1}
        right = {getattr(row, field) for row in rows if row.fold == 2}
        if left & right:
            raise MiningBoundaryError(f"Cross-fold {field} overlap")


def mine_codes(rows: Sequence[Row], features: NDArray[np.float32]) -> dict[str, str]:
    """Choose the best absent code using a top-three-region head fit on the other fold."""
    validate_rows(rows)
    if features.ndim != 3 or features.shape[:2] != (len(rows), 16) or not np.isfinite(features).all():
        raise MiningBoundaryError("Expected one finite 16-region grid per training image")
    chosen: dict[str, str] = {}
    for held_fold in (1, 2):
        fit = np.asarray([row.fold != held_fold for row in rows])
        fit_rows = [row for row in rows if row.fold != held_fold]
        held_rows = [row for row in rows if row.fold == held_fold]
        codes = sorted({code for row in fit_rows for code in row.codes})
        if not codes:
            raise MiningBoundaryError("The opposite fold has no eligible positive code")
        labels = np.asarray([[code in row.codes for code in codes] for row in fit_rows], dtype=np.float32)
        scores = _fit_scores(
            features[fit], labels, features[~fit], regional=True, top_k=3,
            seed=SEED + held_fold, steps=160, device_name="cpu",
        )
        for row, values in zip(held_rows, scores, strict=True):
            eligible = [index for index, code in enumerate(codes) if code not in row.codes]
            if not eligible:
                raise MiningBoundaryError("No fit-supported absent code for a training image")
            best = max(eligible, key=lambda index: float(values[index]))
            chosen[row.sample_id] = codes[best]
    return chosen


def replace_hard_negatives(
    baseline: Sequence[Example], rows: Sequence[Row], choices: dict[str, str],
) -> list[Example]:
    """Keep every image, slot, kind, and non-hard example unchanged."""
    by_id = {row.sample_id: row for row in rows}
    curriculum: list[Example] = []
    changed = 0
    for example in baseline:
        if example["kind"] != "verify_neg_hard":
            curriculum.append(example.copy())
            continue
        sample_id, old_code = example["sample_id"].rsplit("__ver_neg_", 1)
        row = by_id[sample_id]
        code = choices[sample_id]
        if code not in CODES or code in row.codes or example["image"] != row.image:
            raise MiningBoundaryError("Mined negative must keep its image and use an absent code")
        answer = json.dumps(
            {
                "decision": "reject",
                "reason": (
                    f"사진에서 {KOR_CODE_NAMES[code]}에 해당하는 "
                    "직접적인 불안전 상태는 확인되지 않는다."
                ),
            },
            ensure_ascii=False, separators=(",", ":"),
        )
        curriculum.append({
            **example, "sample_id": f"{sample_id}__ver_neg_{code}",
            "user_text": verify_user_text(code), "assistant_text": answer,
        })
        changed += code != old_code
    if changed == 0:
        raise MiningBoundaryError("Mining changed zero hard-negative codes")
    return curriculum


def prepare(root: Path, output: Path) -> tuple[list[ManifestRow], list[Example], dict[str, str | int]]:
    """Load only approved inner-fold supervision and matched cached base features."""
    manifest_path = root / MANIFEST
    if digest(manifest_path) != MANIFEST_SHA256:
        raise MiningBoundaryError("Inner-fold training manifest SHA-256 differs")
    manifest: list[ManifestRow] = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if len(manifest) != 374 or any(item["split"] != "dev" for item in manifest):
        raise MiningBoundaryError("Expected 374 approved development rows")
    rows = [Row(item["sample_id"], item["image"], item["sha256"], item["group_id"],
                int(item["dev_fold"]), frozenset(item["codes"])) for item in manifest]
    validate_rows(rows)
    expected_ids = {row.sample_id for row in rows}
    raw = {}
    with (root / "data/train/train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line, object_pairs_hook=unique_keys)
                if item["sample_id"] in expected_ids:
                    if item["sample_id"] in raw:
                        raise MiningBoundaryError("Duplicate raw training label identity")
                    raw[item["sample_id"]] = item
    if set(raw) != expected_ids:
        raise MiningBoundaryError("Approved training labels are missing")
    for row in rows:
        if set(raw[row.sample_id]["target"].get("hazards", {})) != row.codes:
            raise MiningBoundaryError("Raw labels differ from the approved inner-fold manifest")
    baseline: list[Example] = build_training_curriculum(manifest, raw, seed=SEED)
    hard_count = sum(item["kind"] == "verify_neg_hard" for item in baseline)
    if len(baseline) != 500 or hard_count != 18:
        raise MiningBoundaryError("Expected the matched 500-example baseline with 18 hard negatives")
    encoder = encoder_fingerprint(root / "data/baseline/base_model", 262144, "bf16")
    grids = []
    for row in rows:
        if digest(root / "data/train" / row.image) != row.sha256:
            raise MiningBoundaryError("Training image differs from its approved hash")
        grid = read_cached(row, root / ".runtime/spatial-qwen3vl-cache", encoder)
        if grid is None:
            raise MiningBoundaryError("Matching 262144-pixel base feature cache is missing")
        grids.append(grid)
    curriculum = replace_hard_negatives(baseline, rows, mine_codes(rows, np.stack(grids)))
    changed = {
        index for index, (before, after) in enumerate(zip(baseline, curriculum, strict=True)) if before != after
    }
    ordered = list(range(len(curriculum)))
    random.Random(SEED + 1000).shuffle(ordered)
    first_changed_step = min(ordered.index(index) // 8 + 1 for index in changed)
    report: dict[str, str | int] = {
        "arm": "other-inner-fold-top3-region-mined-negative",
        "scope": "Training curriculum preparation only; no advantage or Fold3 outcome measured",
        "manifest_sha256": MANIFEST_SHA256, "encoder_sha256": encoder,
        "supervision_folds": "1,2 only; Fold0/Fold3/Fold4 excluded",
        "fit_steps_per_head": 160, "fit_device": "cpu", "regional_top_k": 3,
        "mining_max_pixels": 262144, "training_max_pixels": 131072,
        "feature_regions": 16, "feature_dimension": int(grids[0].shape[-1]),
        "fold1_photos": sum(row.fold == 1 for row in rows),
        "fold2_photos": sum(row.fold == 2 for row in rows),
        "training_photos": len(rows), "curriculum_examples": len(curriculum),
        "hard_negative_slots": 18, "changed_code_slots": len(changed),
        "scheduled_examples": 96, "changed_slots_in_first_96": len(changed & set(ordered[:96])),
        "hard_slots_in_first_96": sum(
            curriculum[index]["kind"] == "verify_neg_hard" for index in ordered[:96]
        ),
        "first_changed_optimizer_step": first_changed_step,
        "first_changed_checkpoint_step": ((first_changed_step + 5) // 6) * 6,
        "seed": SEED, "epochs": 1, "grad_accum": 8, "stop_after_steps": 12,
        "checkpoint_steps": 6, "val_fold": 3,
        "baseline_curriculum_sha256": hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
        "mined_curriculum_sha256": hashlib.sha256(json.dumps(curriculum, sort_keys=True).encode()).hexdigest(),
        "training_labels_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
        "mining_source_sha256": digest(Path(__file__)),
        "trainer_source_sha256": digest(root / "kwater_lab/vlm_11code.py"),
        "prompt_sha256": digest(root / "kwater_lab/system_prompt_11code.txt"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "curriculum-local.json").write_text(
        json.dumps(curriculum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return manifest, curriculum, report


def main() -> None:
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in {"prep", "train"}:
        raise SystemExit("Usage: python -m kwater_lab.ubuntu_mined_verifier prep|train [local-output-directory]")
    root = Path.cwd()
    output = (Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_OUTPUT).resolve()
    if not output.is_relative_to((root / "data").resolve()):
        raise MiningBoundaryError("Curriculum and checkpoints must remain under local data/")
    train_dir = output / "training"
    if sys.argv[1] == "train" and train_dir.exists():
        raise MiningBoundaryError("Fresh independent training requires a new training directory")
    manifest, curriculum, report = prepare(root, output)
    if sys.argv[1] == "prep":
        print(json.dumps(report, sort_keys=True))
        return
    if report["changed_slots_in_first_96"] == 0:
        raise MiningBoundaryError(
            "No changed example within the paired 12-step budget; "
            f"earliest changed checkpoint is {report['first_changed_checkpoint_step']}"
        )

    def build_matched(rows, raw, seed):
        labels_sha = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        if seed != SEED or rows != manifest or labels_sha != report["training_labels_sha256"]:
            raise MiningBoundaryError("Trainer supervision differs from the prepared inner-fold experiment")
        return curriculum

    def observe_initial(value: str) -> None:
        (output / "initialization-local.json").write_text(
            json.dumps({
                "initial_trainable_lora_sha256": value,
                "initialization": "fresh official base; no adapter resume",
            }) + "\n",
            encoding="utf-8",
        )

    train_multimodal_qlora(
        root=root, manifest_path=root / MANIFEST, prompt_path=root / "kwater_lab/system_prompt_11code.txt",
        out_dir=train_dir, val_fold=3, epochs=1, grad_accum=8, max_pixels=131072, seed=SEED,
        checkpoint_steps=6, stop_after_steps=12, curriculum_builder=build_matched,
        experiment_identity=report, initialization_observer=observe_initial,
    )


if __name__ == "__main__":
    main()
