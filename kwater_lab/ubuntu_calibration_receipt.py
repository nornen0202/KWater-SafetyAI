"""Verify the executed Fold1 training against its sealed CPU preparation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final
from zipfile import ZipFile

import torch
from safetensors.torch import load_file

from .metrics import unique_keys
from .spatial_probe import digest
from .ubuntu_calibration_train import load_training_rows, select_curriculum
from .ubuntu_focused_curriculum import GRAD_ACCUM, LR, STEPS, Decision, FocusedError
from .ubuntu_focused_verifier import StepRecord
from .ubuntu_inner_verifier_eval import MANIFEST_SHA
from .ubuntu_roi_readout import write_json

ARCHIVE_SHA: Final = "be91eb3a745dc6e1f305237b44842106707164f4174940439b246e4abb385aab"


@dataclass(frozen=True, slots=True)
class TrainingReceipt:
    schema: str
    canonical_manifest_sha256: str
    preparation_sha256: str
    curriculum_sha256: str
    folds: tuple[int, ...]
    optimizer_steps: int
    examples: int
    unique_photos: int
    unique_groups: int
    checkpoint_files: Mapping[str, str]
    base_files: Mapping[str, str]
    archive_sha256: str
    initial_trainable_lora_sha256: str
    sources: Mapping[str, str]


def verify_step_records(
    curriculum: Sequence[Decision], records: Sequence[StepRecord]
) -> None:
    """Require exactly the prescribed 12 complete updates and applied control learning rates."""
    if len(records) != STEPS or len(curriculum) != STEPS * GRAD_ACCUM:
        raise FocusedError("Expected exactly 12 updates and 96 curriculum examples")
    for index, record in enumerate(records):
        group = curriculum[index * GRAD_ACCUM : (index + 1) * GRAD_ACCUM]
        expected_lr = (
            LR * index / 4
            if index < 4
            else LR * 0.5 * (1 + math.cos(math.pi * (index - 4) / 59))
        )
        if (
            record.get("step") != index + 1
            or record.get("examples") != GRAD_ACCUM
            or record.get("supervised_tokens") != GRAD_ACCUM
            or record.get("sample_ids") != [x.sample_id for x in group]
            or record.get("accept") != sum(x.accepted for x in group)
            or record.get("reject") != sum(not x.accepted for x in group)
        ):
            raise FocusedError(
                "Executed step identities or supervision differ from curriculum"
            )
        rates = record.get("applied_lr", [])
        if len(rates) != 2 or not all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
            for a, b in zip(rates, (expected_lr, expected_lr * 0.65), strict=True)
        ):
            raise FocusedError(
                "Applied learning rates differ from first 12 control updates"
            )
        if not all(
            math.isfinite(record.get(name, math.nan)) for name in ("loss", "grad_norm")
        ):
            raise FocusedError("Missing or nonfinite loss/gradient record")


def verify_official_base(root: Path, expected: Mapping[str, str]) -> None:
    """Compare every actual base file with bytes in the hash-pinned official archive."""
    archive_path = root / "data/packages/kwater-baseline.zip"
    if digest(archive_path) != ARCHIVE_SHA:
        raise FocusedError("Official baseline archive SHA-256 differs")
    base = root / "data/baseline/base_model"
    actual = {p.name: digest(p) for p in sorted(base.iterdir()) if p.is_file()}
    if actual != expected:
        raise FocusedError("Current base files differ from prepared base")
    with ZipFile(archive_path) as archive:
        members = [
            m
            for m in archive.infolist()
            if m.filename.startswith("base_model/") and not m.is_dir()
        ]
        if {Path(m.filename).name for m in members} != set(actual):
            raise FocusedError("Official archive and actual base file sets differ")
        for member in members:
            with archive.open(member) as stream:
                member_hash = hashlib.sha256()
                while block := stream.read(1024 * 1024):
                    member_hash.update(block)
                if member_hash.hexdigest() != actual[Path(member.filename).name]:
                    raise FocusedError("Base file differs from official archive member")


def verify_training_receipt(root: Path, output: Path) -> TrainingReceipt:
    """Fail closed unless preparation, step12 state, fresh initialization and base agree."""
    if not output.resolve().is_relative_to((root / "data").resolve()):
        raise FocusedError("Training receipt must remain under private data/")
    preparation = output / "preparation.json"
    prep = json.loads(preparation.read_text(), object_pairs_hook=unique_keys)
    if prep["canonical_manifest_sha256"] != MANIFEST_SHA or prep["folds"] != [1]:
        raise FocusedError("Preparation is not the canonical Fold1 experiment")
    for name, expected in prep["sources"].items():
        if digest(root / "kwater_lab" / name) != expected:
            raise FocusedError(f"Training source changed: {name}")
    rows = load_training_rows(root)
    curriculum = select_curriculum(rows)
    serialized = [asdict(x) for x in curriculum]
    actual = json.loads(
        (output / "curriculum-local.json").read_text(), object_pairs_hook=unique_keys
    )
    curriculum_sha = hashlib.sha256(
        json.dumps(serialized, sort_keys=True).encode()
    ).hexdigest()
    if actual != serialized or prep["curriculum_sha256"] != curriculum_sha:
        raise FocusedError(
            "Prepared curriculum differs from deterministic Fold1 selection"
        )
    by_id = {r.sample_id: r for r in rows}
    image_root = (root / "data/train").resolve()
    for example in curriculum:
        path = (image_root / example.image).resolve()
        if (
            not path.is_relative_to(image_root)
            or digest(path) != by_id[example.sample_id].sha256
        ):
            raise FocusedError("Executed Fold1 image path or SHA-256 differs")
    checkpoint = output / "training/checkpoint-12"
    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=True
    )
    if state["step"] != STEPS or state["preparation_sha256"] != digest(preparation):
        raise FocusedError("Checkpoint12 does not bind the current preparation")
    records = [
        json.loads(line, object_pairs_hook=unique_keys)
        for line in (output / "training/steps.jsonl").read_text().splitlines()
    ]
    if state["records"] != records:
        raise FocusedError("Durable checkpoint and external step log differ")
    verify_step_records(curriculum, records)
    initial = json.loads(
        (output / "initialization-local.json").read_text(),
        object_pairs_hook=unique_keys,
    )
    reference = (
        root
        / "data/ubuntu3050-runs/qlora-inner12-mined-verifier/initialization-local.json"
    )
    if (
        initial["reference_receipt_sha256"] != digest(reference)
        or initial["initial_trainable_lora_sha256"]
        != json.loads(reference.read_text())["initial_trainable_lora_sha256"]
    ):
        raise FocusedError("Fresh initialization reference changed")
    tensors = load_file(str(output / "training/checkpoint-0/adapter_model.safetensors"))
    initial_hash = hashlib.sha256()
    for saved_name, tensor in sorted(tensors.items()):
        name = saved_name.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        initial_hash.update(
            repr((name, tuple(tensor.shape), str(tensor.dtype))).encode()
        )
        initial_hash.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    if initial_hash.hexdigest() != initial["initial_trainable_lora_sha256"]:
        raise FocusedError(
            "Checkpoint0 tensors differ from recorded fresh initialization"
        )
    verify_official_base(root, prep["base_files"])
    receipt = TrainingReceipt(
        "fold1-calibration-training-verification-v1",
        MANIFEST_SHA,
        digest(preparation),
        curriculum_sha,
        (1,),
        STEPS,
        len(curriculum),
        len({x.sample_id for x in curriculum}),
        len({x.group_id for x in curriculum}),
        {p.name: digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
        prep["base_files"],
        ARCHIVE_SHA,
        initial_hash.hexdigest(),
        {**prep["sources"], "ubuntu_calibration_receipt.py": digest(Path(__file__))},
    )
    write_json(output / "training-verification.json", asdict(receipt))
    return receipt
