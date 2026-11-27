"""Label-blind cohorts, immutable prediction seals and calibration lock contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Final, TypedDict

from . import ubuntu_inner_verifier_eval as verifier
from .metrics import unique_keys
from .ubuntu_calibration_stats import Selection, select_threshold
from .ubuntu_focused_eval import write_once
from .ubuntu_image_alignment import Labels

SCHEMA: Final = "fold1-focused-fold2-threshold-fold3-development-v1"
COUNTS: Final = {1: 177, 2: 197, 3: 201}
PROTOCOL: Final = {
    "schema": SCHEMA,
    "codes": list(verifier.CODES),
    "max_pixels": 131072,
    "threshold_rule": "accept iff margin > threshold; ties reject",
    "calibration": "Fold2 only; one shared scalar; maximize fixed 3-code MacroF1",
    "constraint": "normal-photo false alarms <= floor(n_normal / 10)",
    "normal": "no gold hazards across all codes",
    "candidates": "unique finite observed Fold2 margins; maximum also means all reject",
    "ties": "fewer normal-photo false alarms, then higher threshold",
    "fold3": "one new Fold1 checkpoint; compare zero and locked threshold",
    "bootstrap": "2000 paired whole metadata-group draws, seed 20261127",
    "sensitivity": "exclude largest actual metadata group; size ties sorted group ID; no retuning",
    "scope": "development-only; folds previously exposed; no independent event guarantee",
}


class Identity(TypedDict):
    schema: str
    protocol_sha256: str
    manifest_sha256: str
    gold_sha256: str
    training_receipt_sha256: str
    cohorts: dict[str, list[verifier.PhotoRecord]]
    source: dict[str, str]
    runtime: dict[str, str]
    adapter: dict[str, str]
    base: dict[str, str]
    prompt_sha256: str
    processor_sha256: str
    decision_ids: list[int]


class Lock(TypedDict):
    identity_sha256: str
    fold2_seal_sha256: str
    protocol_sha256: str
    selection: Selection
    selection_sha256: str


def read_json(path: Path):
    """Reject duplicate object keys in every persisted protocol artifact."""
    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def load_cohorts(root: Path) -> dict[int, tuple[verifier.Photo, ...]]:
    """Verify canonical metadata and image bytes without accessing label fields."""
    manifest = root / "data/clean-split-v1/split_manifest_dev.json"
    if verifier.digest(manifest) != verifier.MANIFEST_SHA:
        raise verifier.EvaluationError("Canonical manifest checksum differs")
    records = read_json(manifest)
    cohorts = {}
    for fold, count in COUNTS.items():
        selected = [r for r in records if r["split"] == "dev" and r["dev_fold"] == fold]
        photos = tuple(
            verifier.Photo(
                *(r[field] for field in ("sample_id", "image", "sha256", "group_id"))
            )
            for r in selected
        )
        if len(photos) != count:
            raise verifier.EvaluationError(f"Fold{fold} size differs from {count}")
        protected = [r for r in records if r["dev_fold"] != fold]
        for field in ("sample_id", "image", "sha256", "group_id"):
            values = {getattr(photo, field) for photo in photos}
            if values & {r[field] for r in protected}:
                raise verifier.EvaluationError(f"Protected fold overlap: {field}")
            if field != "group_id" and len(values) != count:
                raise verifier.EvaluationError(f"Duplicate cohort {field}")
        image_root = (root / "data/train").resolve()
        for photo in photos:
            image = (image_root / photo.image).resolve()
            if (
                not image.is_relative_to(image_root)
                or verifier.digest(image) != photo.sha256
            ):
                raise verifier.EvaluationError(
                    "Image content/path differs from canonical manifest"
                )
        cohorts[fold] = photos
    return cohorts


def verify_identity(run: Path) -> Identity:
    """Reject protocol/source drift before reading caches, locks or labels."""
    identity: Identity = read_json(run / "identity.json")
    if (
        identity["schema"] != SCHEMA
        or identity["protocol_sha256"] != verifier.stable_hash(PROTOCOL)
        or identity["manifest_sha256"] != verifier.MANIFEST_SHA
        or read_json(run / "protocol.json") != PROTOCOL
    ):
        raise verifier.EvaluationError("Frozen protocol differs")
    for name, expected in identity["source"].items():
        if verifier.digest(Path(__file__).with_name(name)) != expected:
            raise verifier.EvaluationError("Frozen source changed")
    return identity


def photos_for(identity: Identity, fold: int) -> tuple[verifier.Photo, ...]:
    """Parse the exact fixed ordered cohort from a frozen run."""
    photos = tuple(verifier.Photo(**row) for row in identity["cohorts"][str(fold)])
    if (
        len(photos) != COUNTS[fold]
        or len({p.sample_id for p in photos}) != COUNTS[fold]
    ):
        raise verifier.EvaluationError("Frozen cohort count or uniqueness differs")
    return photos


def verify_stage(run: Path, fold: int) -> tuple[verifier.Photo, ...]:
    """Verify complete predictions and checksums before opening any stage labels."""
    identity = verify_identity(run)
    stage = run / f"fold{fold}"
    seal = read_json(stage / "seal.json")
    if seal["identity_sha256"] != verifier.digest(run / "identity.json"):
        raise verifier.EvaluationError("Sealed identity changed")
    if seal["predictions_sha256"] != verifier.digest(stage / "predictions.json"):
        raise verifier.EvaluationError("Sealed predictions changed")
    photos = photos_for(identity, fold)
    verifier.parse_margins(read_json(stage / "predictions.json"), photos)
    if seal["conditions"] != COUNTS[fold] * len(verifier.CODES):
        raise verifier.EvaluationError("Incomplete sealed conditions")
    if fold == 3:
        verify_lock(run)
        if seal["threshold_lock_sha256"] != verifier.digest(
            run / "threshold-lock.json"
        ):
            raise verifier.EvaluationError("Fold3 threshold lock changed")
        if (
            read_json(stage / "start.json")["threshold_lock_sha256"]
            != seal["threshold_lock_sha256"]
        ):
            raise verifier.EvaluationError("Fold3 start lock changed")
    return photos


def seal_stage(run: Path, fold: int) -> None:
    """Publish a seal only after all expected photo/code rows are present."""
    identity = verify_identity(run)
    photos = photos_for(identity, fold)
    stage = run / f"fold{fold}"
    verifier.parse_margins(read_json(stage / "predictions.json"), photos)
    lock_sha = None
    if fold == 3:
        verify_lock(run)
        lock_sha = verifier.digest(run / "threshold-lock.json")
        if read_json(stage / "start.json")["threshold_lock_sha256"] != lock_sha:
            raise verifier.EvaluationError(
                "Fold3 began under a different threshold lock"
            )
    write_once(
        stage / "seal.json",
        {
            "identity_sha256": verifier.digest(run / "identity.json"),
            "predictions_sha256": verifier.digest(stage / "predictions.json"),
            "conditions": len(photos) * len(verifier.CODES),
            "threshold_lock_sha256": lock_sha,
        },
    )


def load_labels(run: Path, fold: int, gold_path: Path) -> Labels:
    """Access selected targets only after verifying the corresponding complete seal."""
    photos = verify_stage(run, fold)
    identity = verify_identity(run)
    if verifier.digest(gold_path) != identity["gold_sha256"]:
        raise verifier.EvaluationError("Gold checksum differs")
    selected = {photo.sample_id for photo in photos}
    hazards, severe = {}, {}
    with gold_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line, object_pairs_hook=unique_keys)
            sample_id = record["sample_id"]
            if sample_id not in selected:
                continue
            if sample_id in hazards:
                raise verifier.EvaluationError("Duplicate selected gold")
            target = record["target"]["hazards"]
            hazards[sample_id] = frozenset(target)
            severe[sample_id] = frozenset(
                code
                for code, hazard in target.items()
                if code in verifier.CODES
                and int(hazard["risk_level"]["severity"]["score"]) == 4
            )
    if set(hazards) != selected:
        raise verifier.EvaluationError("Gold coverage differs")
    return Labels(
        tuple(hazards[p.sample_id] for p in photos),
        tuple(severe[p.sample_id] for p in photos),
    )


def fit_threshold(run: Path, gold_path: Path) -> Lock:
    """Publish one immutable scalar lock; never tune after any Fold3 work starts."""
    photos = verify_stage(run, 2)
    if (run / "threshold-lock.json").exists():
        return verify_lock(run)
    if (run / "fold3").exists():
        raise verifier.EvaluationError("Cannot fit after Fold3 work exists")
    labels = load_labels(run, 2, gold_path)
    rows = verifier.parse_margins(read_json(run / "fold2/predictions.json"), photos)
    selection = select_threshold(labels, rows)
    lock: Lock = {
        "identity_sha256": verifier.digest(run / "identity.json"),
        "fold2_seal_sha256": verifier.digest(run / "fold2/seal.json"),
        "protocol_sha256": verifier.stable_hash(PROTOCOL),
        "selection": selection,
        "selection_sha256": verifier.stable_hash(selection),
    }
    write_once(run / "threshold-lock.json", lock)
    return lock


def verify_lock(run: Path) -> Lock:
    """Bind the threshold to the sealed Fold2 margins and frozen protocol."""
    verify_stage(run, 2)
    lock: Lock = read_json(run / "threshold-lock.json")
    if (
        lock["identity_sha256"] != verifier.digest(run / "identity.json")
        or lock["fold2_seal_sha256"] != verifier.digest(run / "fold2/seal.json")
        or lock["protocol_sha256"] != verifier.stable_hash(PROTOCOL)
    ):
        raise verifier.EvaluationError("Threshold lock provenance differs")
    selection = lock["selection"]
    if verifier.stable_hash(selection) != lock["selection_sha256"]:
        raise verifier.EvaluationError("Threshold lock selection checksum changed")
    if (
        not math.isfinite(selection["threshold"])
        or selection["false_alarm_cap"] != selection["normal_photos"] // 10
        or selection["normal_false_alarm_photos"] > selection["false_alarm_cap"]
    ):
        raise verifier.EvaluationError("Locked threshold exceeds photo false-alarm cap")
    candidates = {row["margin"] for row in read_json(run / "fold2/predictions.json")}
    if selection["threshold"] not in candidates:
        raise verifier.EvaluationError("Locked threshold is not a frozen candidate")
    return lock


def cohort_records(photos: tuple[verifier.Photo, ...]) -> list[verifier.PhotoRecord]:
    """Serialize label-free metadata for the immutable protocol identity."""
    return [
        {
            "sample_id": photo.sample_id,
            "image": photo.image,
            "sha256": photo.sha256,
            "group_id": photo.group_id,
        }
        for photo in photos
    ]
