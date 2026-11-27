"""Fold1-only preparation and shared focused QLoRA training for calibration."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final, Literal

from .metrics import CODES as ALL_CODES
from .metrics import unique_keys
from .spatial_probe import Row, digest
from .ubuntu_focused_curriculum import (
    CODES,
    CONTROL_HORIZON,
    CONTROL_WARMUP,
    GRAD_ACCUM,
    LR,
    MAX_PIXELS,
    PREFIX,
    SEED,
    STEPS,
    Decision,
    Encoder,
    FocusedError,
    decision_token_ids,
)
from .ubuntu_inner_verifier_eval import MANIFEST_SHA
from .vlm_11code import load_qwen_processor

DEFAULT_OUTPUT: Final = Path("data/ubuntu3050-runs/qlora-fold1-calibration")
CANONICAL: Final = Path("data/clean-split-v1/split_manifest_dev.json")
Kind = Literal["positive", "normal", "other_hazard"]
QUOTAS: Final[tuple[tuple[Kind, int], ...]] = (
    ("positive", 12),
    ("normal", 10),
    ("other_hazard", 10),
)


def validate_rows(rows: Sequence[Row]) -> None:
    """Reject cross-fold, duplicate-identity or unknown-code inputs before sampling."""
    if not rows or {row.fold for row in rows} != {1}:
        raise FocusedError("Calibration training requires Fold1 only")
    for field in ("sample_id", "image", "sha256"):
        if len({getattr(row, field) for row in rows}) != len(rows):
            raise FocusedError(f"Duplicate Fold1 {field}")
    if any(not row.codes <= set(ALL_CODES) for row in rows):
        raise FocusedError("Unknown Fold1 training code")


def load_training_rows(root: Path) -> tuple[Row, ...]:
    """Pin Fold1 membership and exclude every protected identity before raw gold use."""
    path = root / CANONICAL
    if digest(path) != MANIFEST_SHA:
        raise FocusedError("Canonical manifest SHA-256 differs")
    manifest = json.loads(path.read_text(), object_pairs_hook=unique_keys)
    approved = [r for r in manifest if r["split"] == "dev" and r["dev_fold"] == 1]
    if len(approved) != 177:
        raise FocusedError("Expected exactly 177 canonical Fold1 photos")
    rows = tuple(
        Row(
            r["sample_id"],
            r["image"],
            r["sha256"],
            r["group_id"],
            r["dev_fold"],
            frozenset(r["codes"]),
        )
        for r in approved
    )
    validate_rows(rows)
    protected = [r for r in manifest if r not in approved]
    for field in ("sample_id", "image", "sha256", "group_id"):
        if {getattr(r, field) for r in rows} & {r[field] for r in protected}:
            raise FocusedError(f"Protected-fold {field} overlaps Fold1")
    expected = {r.sample_id: r.codes for r in rows}
    seen: set[str] = set()
    with (root / "data/train/train.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line, object_pairs_hook=unique_keys)
            sample_id = item["sample_id"]
            if sample_id not in expected:
                continue
            if (
                sample_id in seen
                or set(item["target"]["hazards"]) != expected[sample_id]
            ):
                raise FocusedError("Duplicated Fold1 raw label or code mismatch")
            seen.add(sample_id)
    if seen != set(expected):
        raise FocusedError("Missing Fold1 raw labels")
    return rows


def select_curriculum(rows: Sequence[Row]) -> tuple[Decision, ...]:
    """Fill scarce positives first, preferring globally underrepresented groups."""
    validate_rows(rows)
    rng = random.Random(SEED)
    selected: list[Decision] = []
    used: set[str] = set()
    group_use: Counter[str] = Counter()
    codes = sorted(CODES, key=lambda c: (sum(c in r.codes for r in rows), c))
    for kind, count in QUOTAS:
        for code in codes:
            eligible = [
                r
                for r in sorted(rows, key=lambda r: r.sample_id)
                if r.sample_id not in used
                and (
                    (kind == "positive" and code in r.codes)
                    or (kind == "normal" and not r.codes)
                    or (
                        kind == "other_hazard" and bool(r.codes) and code not in r.codes
                    )
                )
            ]
            if len(eligible) < count:
                raise FocusedError(
                    f"Cannot fill {code}/{kind} with distinct photographs"
                )
            rng.shuffle(eligible)
            stratum_use: Counter[str] = Counter()
            for _ in range(count):
                row = min(
                    eligible,
                    key=lambda r: (stratum_use[r.group_id], group_use[r.group_id]),
                )
                eligible.remove(row)
                used.add(row.sample_id)
                stratum_use[row.group_id] += 1
                group_use[row.group_id] += 1
                selected.append(
                    Decision(
                        row.sample_id,
                        row.image,
                        row.group_id,
                        row.fold,
                        code,
                        kind == "positive",
                        kind,
                    )
                )
    random.Random(SEED + 1000).shuffle(selected)
    return tuple(selected)


def prepare(
    root: Path, output: Path
) -> tuple[tuple[Decision, ...], Encoder, tuple[int, int]]:
    """Validate and encode all 96 decisions on CPU, writing only private local artifacts."""
    if not output.resolve().is_relative_to((root / "data").resolve()):
        raise FocusedError("Private calibration output must remain under local data/")
    rows = load_training_rows(root)
    curriculum = select_curriculum(rows)
    base = root / "data/baseline/base_model"
    processor = load_qwen_processor(base, MAX_PIXELS)
    encoder = Encoder(
        processor, root, (root / "kwater_lab/system_prompt_11code.txt").read_text()
    )
    token_ids = decision_token_ids(processor.tokenizer)
    by_id = {r.sample_id: r for r in rows}
    image_root = (root / "data/train").resolve()
    prefix_ids = processor.tokenizer.encode(PREFIX, add_special_tokens=False)
    encoded = []
    for example in curriculum:
        path = (image_root / example.image).resolve()
        if (
            not path.is_relative_to(image_root)
            or digest(path) != by_id[example.sample_id].sha256
        ):
            raise FocusedError("Selected image path or SHA-256 differs")
        batch = encoder.encode(example)
        if batch["input_ids"][0, -len(prefix_ids) :].tolist() != prefix_ids:
            raise FocusedError(
                "Encoded image/chat input does not end at exact JSON prefix"
            )
        encoded.append(
            {
                "sample_id": example.sample_id,
                "code": example.code,
                "image_sha256": by_id[example.sample_id].sha256,
                "target_token_id": token_ids[0 if example.accepted else 1],
                "input_tokens": int(batch["input_ids"].shape[-1]),
                "input_sha256": hashlib.sha256(
                    batch["input_ids"].numpy().tobytes()
                ).hexdigest(),
            }
        )
    serialized = [asdict(x) for x in curriculum]
    report = {
        "schema": "fold1-calibration-decision-v1",
        "status": "cpu-prepared-gpu-unrun",
        "canonical_manifest_sha256": MANIFEST_SHA,
        "folds": [1],
        "eligible_photos": len(rows),
        "examples": len(curriculum),
        "unique_photos": len({x.sample_id for x in curriculum}),
        "unique_groups": len({x.group_id for x in curriculum}),
        "counts": dict(Counter(f"{x.code}/{x.negative_kind}" for x in curriculum)),
        "groups_per_stratum": {
            f"{code}/{kind}": len(
                {
                    x.group_id
                    for x in curriculum
                    if x.code == code and x.negative_kind == kind
                }
            )
            for code in CODES
            for kind, _ in QUOTAS
        },
        "protected_overlap_counts": {
            field: 0 for field in ("sample_id", "image", "sha256", "group_id")
        },
        "fold_counts": dict(Counter(x.fold for x in curriculum)),
        "seed": SEED,
        "max_pixels": MAX_PIXELS,
        "grad_accum": GRAD_ACCUM,
        "optimizer_steps": STEPS,
        "lr": LR,
        "vision_lr_ratio": 0.65,
        "scheduler_horizon": CONTROL_HORIZON,
        "scheduler_warmup": CONTROL_WARMUP,
        "supervised_tokens_per_example": 1,
        "decision_prefix": PREFIX,
        "decision_token_ids": token_ids,
        "curriculum_sha256": hashlib.sha256(
            json.dumps(serialized, sort_keys=True).encode()
        ).hexdigest(),
        "sources": {
            name: digest(root / "kwater_lab" / name)
            for name in (
                "ubuntu_calibration_train.py",
                "ubuntu_focused_verifier.py",
                "ubuntu_focused_curriculum.py",
                "vlm_11code.py",
                "ubuntu_inner_verifier_eval.py",
                "system_prompt_11code.txt",
            )
        },
        "base_files": {
            path.name: digest(path) for path in sorted(base.iterdir()) if path.is_file()
        },
        "comparison_limits": "Fold1-only training; no Fold2/Fold3 inference. Uses fresh focused LoRA recipe and first 12 control-schedule updates.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("curriculum-local.json", serialized),
        ("encoding-local.json", encoded),
        ("preparation.json", report),
    ):
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    return curriculum, encoder, token_ids


def main() -> None:
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in {"prep", "train", "resume"}:
        raise SystemExit(
            "Usage: python -m kwater_lab.ubuntu_calibration_train prep|train|resume [local-output-directory]"
        )
    root = Path.cwd()
    output = (Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_OUTPUT).resolve()
    if sys.argv[1] in {"train", "resume"}:
        from .ubuntu_focused_verifier import train

        train(root, output, resume=sys.argv[1] == "resume", preparer=prepare)
    else:
        curriculum, _, tokens = prepare(root, output)
        print(
            json.dumps(
                {
                    "prepared_examples": len(curriculum),
                    "decision_token_ids": tokens,
                    "receipt": str(output / "preparation.json"),
                }
            )
        )


if __name__ == "__main__":
    main()
