"""CPU preparation of the approved 96-photo conditional decision experiment."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict

from PIL import Image, ImageOps
from transformers import (
    BatchFeature,
    PreTrainedTokenizerBase,
)

from .metrics import unique_keys
from .spatial_probe import Row, digest
from .ubuntu_inner_verifier_eval import CODES, MANIFEST_SHA, PREFIX
from .ubuntu_mined_verifier import MANIFEST, MANIFEST_SHA256, SEED, validate_rows
from .vlm_11code import (
    load_qwen_processor,
    verify_user_text,
)

MAX_PIXELS: Final = 131072
GRAD_ACCUM: Final = 8
STEPS: Final = 12
CONTROL_HORIZON: Final = 63
CONTROL_WARMUP: Final = 4
LR: Final = 1.4e-4


@dataclass(frozen=True, slots=True)
class FocusedError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Decision:
    sample_id: str
    image: str
    group_id: str
    fold: int
    code: str
    accepted: bool
    negative_kind: Literal["positive", "normal", "other_hazard"]


class Content(TypedDict, total=False):
    type: str
    text: str


class Message(TypedDict):
    role: str
    content: str | list[Content]


class Processor(Protocol):
    """Runtime processor surface; generic Transformers kwargs omit these Qwen options."""

    def apply_chat_template(
        self, messages: list[Message], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def __call__(
        self, *, text: list[str], images: list[Image.Image], return_tensors: str
    ) -> BatchFeature: ...

    def save_pretrained(self, path: str) -> list[str]: ...


def load_training_rows(root: Path) -> tuple[Row, ...]:
    """Check canonical membership and protected group identities before raw labels."""
    canonical_path = root / "data/clean-split-v1/split_manifest_dev.json"
    if (
        digest(root / MANIFEST) != MANIFEST_SHA256
        or digest(canonical_path) != MANIFEST_SHA
    ):
        raise FocusedError("Canonical or inner-fold manifest SHA-256 differs")
    manifest = json.loads((root / MANIFEST).read_text(), object_pairs_hook=unique_keys)
    canonical = json.loads(canonical_path.read_text(), object_pairs_hook=unique_keys)
    approved = [
        item
        for item in canonical
        if item["split"] == "dev" and item["dev_fold"] in (1, 2)
    ]
    if len(manifest) != 374 or manifest != approved:
        raise FocusedError("Training rows differ from the canonical folds 1+2 subset")
    rows = tuple(
        Row(
            item["sample_id"],
            item["image"],
            item["sha256"],
            item["group_id"],
            item["dev_fold"],
            frozenset(item["codes"]),
        )
        for item in manifest
    )
    validate_rows(rows)
    protected = [item for item in canonical if item["dev_fold"] not in (1, 2)]
    for field in ("sample_id", "image", "sha256", "group_id"):
        if {getattr(row, field) for row in rows} & {item[field] for item in protected}:
            raise FocusedError(f"Protected-fold {field} overlaps training")
    expected = {row.sample_id: row for row in rows}
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
                or set(item["target"]["hazards"]) != expected[sample_id].codes
            ):
                raise FocusedError("Duplicated raw label or raw code/manifest mismatch")
            seen.add(sample_id)
    if seen != set(expected):
        raise FocusedError("Missing raw labels for approved training rows")
    return rows


def select_curriculum(rows: Sequence[Row]) -> tuple[Decision, ...]:
    """Sample scarce positives first, then group-balanced negatives, without photo reuse."""
    validate_rows(rows)
    rng = random.Random(SEED)
    selected: list[Decision] = []
    used: set[str] = set()
    strata: list[tuple[str, Literal["positive", "normal", "other_hazard"], int]] = [
        (code, "positive", 16)
        for code in sorted(CODES, key=lambda c: sum(c in r.codes for r in rows))
    ]
    negative_kinds: tuple[Literal["normal", "other_hazard"], ...] = (
        "normal",
        "other_hazard",
    )
    for kind in negative_kinds:
        for code in CODES:
            strata.append((code, kind, 8))
    for code, kind, count in strata:
        eligible = [
            row
            for row in rows
            if row.sample_id not in used
            and (
                (kind == "positive" and code in row.codes)
                or (kind == "normal" and not row.codes)
                or (
                    kind == "other_hazard" and bool(row.codes) and code not in row.codes
                )
            )
        ]
        if len(eligible) < count:
            raise FocusedError(f"Cannot fill {code}/{kind} with distinct photographs")
        groups: dict[str, list[Row]] = {}
        for row in sorted(eligible, key=lambda r: r.sample_id):
            groups.setdefault(row.group_id, []).append(row)
        group_order = sorted(groups)
        rng.shuffle(group_order)
        for pool in groups.values():
            rng.shuffle(pool)
        taken = 0
        while taken < count:
            for group in group_order:
                if groups[group] and taken < count:
                    row = groups[group].pop()
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
                    used.add(row.sample_id)
                    taken += 1
    random.Random(SEED + 1000).shuffle(selected)
    return tuple(selected)


def decision_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    """Require exact one-token continuations of the evaluation JSON prefix."""
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    result = []
    for decision in ("accept", "reject"):
        tokens = tokenizer.encode(PREFIX + decision, add_special_tokens=False)
        if tokens[:-1] != prefix or len(tokens) != len(prefix) + 1:
            raise FocusedError(
                "Decision does not extend exact evaluation prefix by one token"
            )
        result.append(tokens[-1])
    return result[0], result[1]


@dataclass(frozen=True, slots=True)
class Encoder:
    processor: Processor
    root: Path
    system: str

    def encode(self, example: Decision) -> BatchFeature:
        """Encode the same EXIF-corrected full image and prompt as the fixed evaluator."""
        with Image.open(self.root / "data/train" / example.image) as source:
            ImageOps.exif_transpose(source, in_place=True)
            image = source.convert("RGB")
        messages: list[Message] = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": verify_user_text(example.code)},
                ],
            },
        ]
        text = (
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            + PREFIX
        )
        return self.processor(text=[text], images=[image], return_tensors="pt")


def prepare(
    root: Path, output: Path
) -> tuple[tuple[Decision, ...], Encoder, tuple[int, int]]:
    """CPU-only validation and image encoding of every scheduled decision."""
    if not output.resolve().is_relative_to((root / "data").resolve()):
        raise FocusedError(
            "Private curriculum and checkpoints must remain under local data/"
        )
    rows = load_training_rows(root)
    curriculum = select_curriculum(rows)
    base = root / "data/baseline/base_model"
    processor = load_qwen_processor(base, MAX_PIXELS)
    encoder = Encoder(
        processor, root, (root / "kwater_lab/system_prompt_11code.txt").read_text()
    )
    token_ids = decision_token_ids(processor.tokenizer)
    by_id = {row.sample_id: row for row in rows}
    encoded = []
    image_root = (root / "data/train").resolve()
    prefix_ids = processor.tokenizer.encode(PREFIX, add_special_tokens=False)
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
                "target_token_id": token_ids[0 if example.accepted else 1],
                "input_tokens": int(batch["input_ids"].shape[-1]),
                "input_sha256": hashlib.sha256(
                    batch["input_ids"].numpy().tobytes()
                ).hexdigest(),
            }
        )
    report = {
        "schema": "inner12-focused-decision-v1",
        "status": "cpu-prepared-gpu-unrun",
        "manifest_sha256": MANIFEST_SHA256,
        "canonical_manifest_sha256": MANIFEST_SHA,
        "folds": [1, 2],
        "examples": len(curriculum),
        "unique_photos": len({x.sample_id for x in curriculum}),
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
            for kind in ("positive", "normal", "other_hazard")
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
        "lr_policy": "control 63-step cosine schedule, first 12 updates only",
        "supervised_tokens_per_example": 1,
        "decision_prefix": PREFIX,
        "decision_token_ids": token_ids,
        "curriculum_sha256": hashlib.sha256(
            json.dumps([asdict(x) for x in curriculum], sort_keys=True).encode()
        ).hexdigest(),
        "sources": {
            name: digest(root / "kwater_lab" / name)
            for name in (
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
        "comparison_limits": "Changes sampling and objective jointly; no discarded preflight backward (control had one), so dropout paths differ. No Fold3 predictions or outcomes used.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("curriculum-local.json", [asdict(x) for x in curriculum]),
        ("encoding-local.json", encoded),
        ("preparation.json", report),
    ):
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    return curriculum, encoder, token_ids
