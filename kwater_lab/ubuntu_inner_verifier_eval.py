"""Paired Fold3 checkpoint-12 verifier audit: seal both prediction arms before gold.

Run: .venv/bin/python -m kwater_lab.ubuntu_inner_verifier_eval run
The fixed three-code conditional decision is not full JSON generation or leaderboard scoring.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

from PIL import Image, ImageOps

from .hybrid_pipeline import severity4_miss_stats
from .metrics import unique_keys
from .spatial_probe import digest
from .ubuntu_roi_readout import (
    CODES,
    PREFIX,
    classification_metrics,
    stable_hash,
    write_json,
)

ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST_SHA: Final = "3864848fe26b951ec070710499d2694078b17c76ffd2aaf38d312b99ad3b2f60"
GOLD_SHA: Final = "ab427fed4615b32627b76001e27124e99ae9202c855ba4170916ebe4fdc4de55"
ARMS: Final = ("control", "mined")
PHOTO_COUNT: Final = 201
MAX_PIXELS: Final = 131072
SEED: Final = 20261127


@dataclass(frozen=True, slots=True)
class EvaluationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Photo:
    sample_id: str
    image: str
    sha256: str
    group_id: str


@dataclass(frozen=True, slots=True)
class Margin:
    sample_id: str
    code: str
    margin: float
    input_sha256: str

    def to_record(self) -> MarginRecord:
        return {
            "sample_id": self.sample_id,
            "code": self.code,
            "margin": self.margin,
            "input_sha256": self.input_sha256,
        }


class PhotoRecord(TypedDict):
    sample_id: str
    image: str
    sha256: str
    group_id: str


class MarginRecord(TypedDict):
    sample_id: str
    code: str
    margin: float
    input_sha256: str


class SealedIdentity(TypedDict):
    photos: list[PhotoRecord]
    codes: list[str]
    gold_sha256: str
    manifest_sha256: str
    max_pixels: int
    threshold: float


def load_cohort(root: Path) -> tuple[Photo, ...]:
    """Read canonical split metadata only; label fields are never accessed here."""
    manifest = root / "data/clean-split-v1/split_manifest_dev.json"
    if digest(manifest) != MANIFEST_SHA:
        raise EvaluationError("Canonical development manifest SHA-256 differs")
    rows = json.loads(manifest.read_text(), object_pairs_hook=unique_keys)
    selected = [row for row in rows if row["split"] == "dev" and row["dev_fold"] == 3]
    photos = tuple(
        Photo(*(row[key] for key in ("sample_id", "image", "sha256", "group_id")))
        for row in selected
    )
    if len(photos) != PHOTO_COUNT:
        raise EvaluationError("Expected exactly 201 Fold3 photos")
    for field in ("sample_id", "sha256", "image"):
        if len({getattr(photo, field) for photo in photos}) != PHOTO_COUNT:
            raise EvaluationError(f"Duplicate Fold3 {field}")
    fit_groups = {row["group_id"] for row in rows if row["dev_fold"] in (1, 2)}
    if fit_groups & {photo.group_id for photo in photos}:
        raise EvaluationError("Fold3 event overlaps inner training folds")
    image_root = (root / "data/train").resolve()
    for photo in photos:
        path = (image_root / photo.image).resolve()
        if not path.is_relative_to(image_root) or digest(path) != photo.sha256:
            raise EvaluationError("Fold3 image path or content differs")
    return photos


def parse_margins(
    payload: list[MarginRecord], photos: tuple[Photo, ...]
) -> tuple[Margin, ...]:
    """Require exact ordered photo/code coverage and finite numeric margins."""
    rows = tuple(Margin(**item) for item in payload)
    expected = [(photo.sample_id, code) for photo in photos for code in CODES]
    if [(row.sample_id, row.code) for row in rows] != expected:
        raise EvaluationError(
            "Incomplete, duplicated, or reordered photo/code coverage"
        )
    for row in rows:
        if (
            type(row.margin) not in (int, float)
            or not math.isfinite(row.margin)
            or len(row.input_sha256) != 64
        ):
            raise EvaluationError("Invalid cached margin or input fingerprint")
    return rows


def load_cached_photo(path: Path, key: str, photo: Photo) -> tuple[Margin, ...] | None:
    """Reuse only a complete finite photo block with its exact protocol key."""
    if not path.exists():
        return None
    cached = json.loads(path.read_text(), object_pairs_hook=unique_keys)
    if cached["cache_key"] != key:
        raise EvaluationError("Cached protocol identity changed")
    return parse_margins(cached["rows"], (photo,))


def seal_run(run: Path, photos: tuple[Photo, ...]) -> None:
    """Seal both complete arms together, after checking identical encoded inputs."""
    if len(photos) != PHOTO_COUNT:
        raise EvaluationError("A paired seal requires all 201 Fold3 photos")
    arms = [
        parse_margins(json.loads((run / arm / "predictions.json").read_text()), photos)
        for arm in ARMS
    ]
    if [row.input_sha256 for row in arms[0]] != [row.input_sha256 for row in arms[1]]:
        raise EvaluationError("Encoded inputs differ between the two arms")
    write_json(
        run / "seal.json",
        {
            "identity_sha256": digest(run / "identity.json"),
            "predictions_sha256": {
                arm: digest(run / arm / "predictions.json") for arm in ARMS
            },
            "photos": PHOTO_COUNT,
            "conditions_per_arm": PHOTO_COUNT * len(CODES),
        },
    )


def verify_run(
    run: Path,
) -> tuple[SealedIdentity, tuple[Photo, ...], dict[str, tuple[Margin, ...]]]:
    """Validate a complete paired seal before a caller may open gold."""
    seal = json.loads((run / "seal.json").read_text(), object_pairs_hook=unique_keys)
    if digest(run / "identity.json") != seal["identity_sha256"]:
        raise EvaluationError("Sealed identity changed")
    identity: SealedIdentity = json.loads(
        (run / "identity.json").read_text(), object_pairs_hook=unique_keys
    )
    photos = tuple(Photo(**row) for row in identity["photos"])
    if (
        identity["manifest_sha256"] != MANIFEST_SHA
        or identity["max_pixels"] != MAX_PIXELS
        or identity["threshold"] != 0.0
    ):
        raise EvaluationError("Sealed canonical protocol differs")
    if len(photos) != PHOTO_COUNT or identity["codes"] != list(CODES):
        raise EvaluationError("Sealed cohort or candidates differ")
    if len({photo.sample_id for photo in photos}) != PHOTO_COUNT:
        raise EvaluationError("Duplicate sealed photo identity")
    arms = {}
    for arm in ARMS:
        path = run / arm / "predictions.json"
        if digest(path) != seal["predictions_sha256"][arm]:
            raise EvaluationError("Sealed predictions changed")
        arms[arm] = parse_margins(json.loads(path.read_text()), photos)
    if [row.input_sha256 for row in arms["control"]] != [
        row.input_sha256 for row in arms["mined"]
    ]:
        raise EvaluationError("Sealed encoded inputs differ between arms")
    return identity, photos, arms


def predict(args: argparse.Namespace) -> Path:
    """Resume image/arm caches under an immutable joint protocol fingerprint."""
    import torch

    from .vlm_11code import load_qwen_model, load_qwen_processor, verify_user_text

    if not args.out.resolve().is_relative_to((ROOT / "data").resolve()):
        raise EvaluationError(
            "Private predictions and caches must remain under local data/"
        )
    adapters = {"control": args.control, "mined": args.mined}
    for adapter in adapters.values():
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adapter / name).is_file():
                raise EvaluationError(
                    f"Checkpoint-12 is not complete: {adapter / name}"
                )
    photos = load_cohort(ROOT)
    processor = load_qwen_processor(args.base, MAX_PIXELS)
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    decision_ids = []
    for decision in ("accept", "reject"):
        tokens = tokenizer.encode(PREFIX + decision, add_special_tokens=False)
        if len(tokens) != len(prefix_ids) + 1 or tokens[:-1] != prefix_ids:
            raise EvaluationError(
                "Decision must extend the exact JSON prefix by one token"
            )
        decision_ids.append(tokens[-1])
    system = args.prompt.read_text()
    identity = {
        "schema": "inner12-paired-fold3-verifier-v1",
        "manifest_sha256": MANIFEST_SHA,
        "gold_sha256": GOLD_SHA,
        "photos": [asdict(photo) for photo in photos],
        "codes": list(CODES),
        "max_pixels": MAX_PIXELS,
        "batch_size": 1,
        "seed": SEED,
        "precision": "NF4 double quant, bf16 compute; existing loader visual/lm_head exclusions",
        "decision_prefix": PREFIX,
        "decision_token_ids": decision_ids,
        "threshold": 0.0,
        "prompt_sha256": digest(args.prompt),
        "user_text": {code: verify_user_text(code) for code in CODES},
        "processor": json.loads(processor.image_processor.to_json_string()),
        "base_files": {
            path.name: digest(path)
            for path in sorted(args.base.iterdir())
            if path.is_file()
        },
        "adapters": {
            arm: {
                name: digest(adapter / name)
                for name in ("adapter_config.json", "adapter_model.safetensors")
            }
            for arm, adapter in adapters.items()
        },
        "source": {
            path.name: digest(path)
            for path in (Path(__file__), Path(__file__).with_name("vlm_11code.py"))
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "Pillow")
        },
    }
    fingerprint = stable_hash(identity)
    run = args.out / fingerprint[:24]
    write_json(run / "identity.json", identity)
    if (run / "seal.json").exists():
        verify_run(run)
        return run
    for arm, adapter in adapters.items():
        model = None
        records: list[Margin] = []
        for number, photo in enumerate(photos, 1):
            cache_key = stable_hash((fingerprint, arm, asdict(photo)))
            cache = run / arm / "cache" / f"{cache_key}.json"
            rows = load_cached_photo(cache, cache_key, photo)
            if rows is None:
                if model is None:
                    torch.manual_seed(SEED)
                    model = load_qwen_model(
                        args.base, use_4bit=True, adapter_dir=adapter
                    ).eval()
                with Image.open(ROOT / "data/train" / photo.image) as source:
                    ImageOps.exif_transpose(source, in_place=True)
                    image = source.convert("RGB")
                new_rows: list[Margin] = []
                for code in CODES:
                    messages = [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": verify_user_text(code)},
                            ],
                        },
                    ]
                    text = (
                        processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        + PREFIX
                    )
                    batch = processor(text=[text], images=[image], return_tensors="pt")
                    tensor_hashes = {
                        key: {
                            "sha256": hashlib.sha256(
                                tensor.contiguous().numpy().tobytes()
                            ).hexdigest(),
                            "dtype": str(tensor.dtype),
                            "shape": list(tensor.shape),
                        }
                        for key, tensor in batch.items()
                    }
                    input_sha = stable_hash((tensor_hashes, decision_ids))
                    with torch.inference_mode():
                        logits = (
                            model(**batch.to("cuda"), logits_to_keep=1, use_cache=False)
                            .logits[0, -1]
                            .float()
                        )
                        margin = float(
                            (logits[decision_ids[0]] - logits[decision_ids[1]]).item()
                        )
                    new_rows.append(Margin(photo.sample_id, code, margin, input_sha))
                    del batch, logits
                rows = parse_margins([row.to_record() for row in new_rows], (photo,))
                write_json(
                    cache,
                    {"cache_key": cache_key, "rows": [asdict(row) for row in rows]},
                )
            records.extend(rows)
            print(f"{arm} {number}/{PHOTO_COUNT} photos cached", flush=True)
        write_json(run / arm / "predictions.json", [asdict(row) for row in records])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    seal_run(run, photos)
    return run


def score(run: Path, gold_path: Path, report_path: Path) -> None:
    """Open gold only after both arms are sealed, then emit aggregates only."""
    identity, photos, arms = verify_run(run)
    if digest(gold_path) != identity["gold_sha256"]:
        raise EvaluationError("Gold source differs from the frozen protocol")
    selected = {photo.sample_id for photo in photos}
    gold = {}
    with gold_path.open() as stream:
        for line in stream:
            item = json.loads(line, object_pairs_hook=unique_keys)
            if item["sample_id"] in selected:
                if item["sample_id"] in gold:
                    raise EvaluationError("Duplicate selected gold sample")
                gold[item["sample_id"]] = item
    if set(gold) != selected:
        raise EvaluationError("Gold coverage differs from sealed photos")
    ids = [photo.sample_id for photo in photos]
    gold_codes = [frozenset(gold[sid]["target"]["hazards"]) for sid in ids]
    totals = {}
    decisions = {}
    for arm in ARMS:
        predicted = {sid: set() for sid in ids}
        for row in arms[arm]:
            if row.margin > 0.0:
                predicted[row.sample_id].add(row.code)
        metrics = classification_metrics(
            gold_codes, [frozenset(predicted[sid]) for sid in ids]
        )
        divisor = 2 * metrics["tp"] + metrics["fp"] + metrics["fn"]
        serious_all = severity4_miss_stats(
            gold, {sid: frozenset(predicted[sid]) for sid in ids}
        )
        if serious_all["sev4_instances"] == 0:
            serious_all["sev4_instance_recall"] = None
        serious_candidates = {
            sid: frozenset(
                code
                for code, hazard in gold[sid]["target"]["hazards"].items()
                if code in CODES and int(hazard["risk_level"]["severity"]["score"]) == 4
            )
            for sid in ids
        }
        serious_support = sum(len(codes) for codes in serious_candidates.values())
        serious_hits = sum(
            len(codes & predicted[sid]) for sid, codes in serious_candidates.items()
        )
        totals[arm] = {
            **metrics,
            "pooled_f1": 2 * metrics["tp"] / divisor if divisor else 0.0,
            "serious4_candidate_codes_primary": {
                "instances": serious_support,
                "hits": serious_hits,
                "misses": serious_support - serious_hits,
                "recall": serious_hits / serious_support if serious_support else None,
            },
            "serious4_all_codes_secondary": serious_all,
        }
        decisions[arm] = predicted
    changed = {
        code: {
            "reject_to_accept": sum(
                code not in decisions["control"][sid]
                and code in decisions["mined"][sid]
                for sid in ids
            ),
            "accept_to_reject": sum(
                code in decisions["control"][sid]
                and code not in decisions["mined"][sid]
                for sid in ids
            ),
        }
        for code in CODES
    }
    write_json(
        report_path,
        {
            "scope": (
                "Fixed Fold3 conditional 3-code verifier audit; "
                "development-only, not leaderboard or full generated JSON."
            ),
            "protocol_fingerprint": stable_hash(identity),
            "seal": json.loads((run / "seal.json").read_text()),
            "photos": len(photos),
            "candidate_codes": list(CODES),
            "max_pixels": MAX_PIXELS,
            "threshold": "accept iff conditional accept minus reject logit > 0; zero ties reject",
            "normal_definition": "No gold hazard across all codes; 3-code predictions only",
            "serious4_limitation": (
                "Primary denominator: severity-4 instances in the three fixed candidate codes. "
                "Secondary all-code denominator includes unproposed codes as misses; no predicted severity grading."
            ),
            "arms": totals,
            "changed_decisions": changed,
            "changed_photos": sum(
                decisions["control"][sid] != decisions["mined"][sid] for sid in ids
            ),
            "pooled_f1_delta_mined_minus_control": totals["mined"]["pooled_f1"]
            - totals["control"]["pooled_f1"],
            "training_comparison": (
                "Two checkpoint-12 adapters; matching training budgets/initialization "
                "must be checked in training receipts."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "score", "run"))
    parser.add_argument("--base", type=Path, default=ROOT / "data/baseline/base_model")
    parser.add_argument(
        "--control",
        type=Path,
        default=ROOT / "data/ubuntu3050-runs/qlora-inner12-base-control/checkpoint-12",
    )
    parser.add_argument(
        "--mined",
        type=Path,
        default=ROOT
        / "data/ubuntu3050-runs/qlora-inner12-mined-verifier/training/checkpoint-12",
    )
    parser.add_argument(
        "--prompt", type=Path, default=ROOT / "kwater_lab/system_prompt_11code.txt"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/ubuntu3050-runs/inner12-paired-eval"
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/train/train.jsonl")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "results/ubuntu_inner12_paired_verifier.json",
    )
    args = parser.parse_args()
    match args.command:
        case "predict":
            print(f"SEALED_RUN={predict(args)}", flush=True)
        case "run":
            run = predict(args)
            score(run, args.gold, args.report)
            print(f"AGGREGATE_REPORT={args.report}", flush=True)
        case "score":
            if args.run is None:
                raise EvaluationError(
                    "score requires --run with a complete paired seal"
                )
            score(args.run, args.gold, args.report)
            print(f"AGGREGATE_REPORT={args.report}", flush=True)


if __name__ == "__main__":
    main()
