"""Compare fixed step-12 focused training to an already sealed mixed control.

Run with --source pointing to a complete inner12 paired seal. Only the new focused
adapter runs inference; all mixed margins are reused without modification.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

from . import ubuntu_inner_verifier_eval as verifier
from .metrics import unique_keys
from .ubuntu_image_alignment import Labels
from .ubuntu_roi_readout import PREFIX, classification_metrics

REPLICATIONS: Final = 2000
SEED: Final = 20261127
ADAPTER_FILES: Final = ("adapter_config.json", "adapter_model.safetensors")
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


class Identity(verifier.SealedIdentity):
    schema: str
    batch_size: int
    seed: int
    precision: str
    cuda: str
    device: str
    decision_prefix: str
    decision_token_ids: list[int]
    prompt_sha256: str
    user_text: dict[str, str]
    processor: dict[str, Json]
    base_files: dict[str, str]
    adapters: dict[str, dict[str, str]]
    source: dict[str, str]
    packages: dict[str, str]


@dataclass(frozen=True, slots=True)
class Paths:
    source: Path
    out: Path
    base: Path
    control: Path
    focused: Path
    prompt: Path


def read_identity(path: Path) -> Identity:
    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def runtime_identity(paths: Paths) -> Identity:
    """Reconstruct predictor identity before any inference can use a drifted key."""
    import torch
    from torch.version import cuda

    from .vlm_11code import load_qwen_processor, verify_user_text

    if paths.control.name != "checkpoint-12" or paths.focused.name != "checkpoint-12":
        raise verifier.EvaluationError(
            "Only prespecified checkpoint-12 adapters are eligible"
        )
    processor = load_qwen_processor(paths.base, verifier.MAX_PIXELS)
    prefix = processor.tokenizer.encode(PREFIX, add_special_tokens=False)
    decision_ids = []
    for decision in ("accept", "reject"):
        tokens = processor.tokenizer.encode(PREFIX + decision, add_special_tokens=False)
        if len(tokens) != len(prefix) + 1 or tokens[:-1] != prefix:
            raise verifier.EvaluationError("Decision token contract differs")
        decision_ids.append(tokens[-1])
    photo_records: list[verifier.PhotoRecord] = [
        {
            "sample_id": photo.sample_id,
            "image": photo.image,
            "sha256": photo.sha256,
            "group_id": photo.group_id,
        }
        for photo in verifier.load_cohort(verifier.ROOT)
    ]
    return {
        "schema": "inner12-paired-fold3-verifier-v1",
        "manifest_sha256": verifier.MANIFEST_SHA,
        "gold_sha256": verifier.GOLD_SHA,
        "photos": photo_records,
        "codes": list(verifier.CODES),
        "max_pixels": verifier.MAX_PIXELS,
        "batch_size": 1,
        "seed": SEED,
        "precision": "NF4 double quant, bf16 compute; existing loader visual/lm_head exclusions",
        "cuda": str(cuda),
        "device": torch.cuda.get_device_name(0),
        "decision_prefix": PREFIX,
        "decision_token_ids": decision_ids,
        "threshold": 0.0,
        "prompt_sha256": verifier.digest(paths.prompt),
        "user_text": {code: verify_user_text(code) for code in verifier.CODES},
        "processor": json.loads(processor.image_processor.to_json_string()),
        "base_files": {
            path.name: verifier.digest(path)
            for path in sorted(paths.base.iterdir())
            if path.is_file()
        },
        "adapters": {
            arm: {name: verifier.digest(adapter / name) for name in ADAPTER_FILES}
            for arm, adapter in (("control", paths.control), ("mined", paths.focused))
        },
        "source": {
            path.name: verifier.digest(path)
            for path in (
                Path(verifier.__file__),
                Path(verifier.__file__).with_name("vlm_11code.py"),
            )
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "Pillow")
        },
    }


def require_replacement(source: Identity, expected: Identity) -> None:
    """Only the mined-adapter digests may differ from the sealed source protocol."""
    baseline = json.loads(json.dumps(source))
    baseline["adapters"]["mined"] = expected["adapters"]["mined"]
    if baseline != expected:
        raise verifier.EvaluationError(
            "Runtime protocol drift; refusing mixed inference or gold"
        )
    if source["adapters"]["mined"] == expected["adapters"]["mined"]:
        raise verifier.EvaluationError(
            "Focused adapter must differ from the original mined adapter"
        )


def write_once[T](path: Path, payload: T) -> None:
    """Resume identical files only; an unexpected file is never overwritten."""
    if path.exists():
        if json.loads(path.read_text(), object_pairs_hook=unique_keys) != payload:
            raise verifier.EvaluationError(
                f"Refusing to overwrite differing artifact: {path.name}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=".stage-", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(stream.name, path)
        except FileExistsError:
            if json.loads(path.read_text(), object_pairs_hook=unique_keys) != payload:
                raise verifier.EvaluationError(
                    f"Concurrent differing artifact: {path.name}"
                ) from None


def prepare_cache(source: Path, out: Path, expected: Identity) -> Path:
    """Copy 603 verified mixed margins into exact expected per-photo cache keys."""
    _, photos, arms = verifier.verify_run(source)
    original = read_identity(source / "identity.json")
    require_replacement(original, expected)
    fingerprint = verifier.stable_hash(expected)
    run = out / fingerprint[:24]
    if run.resolve() == source.resolve():
        raise verifier.EvaluationError(
            "Focused run must have a separate output identity"
        )
    write_once(run / "identity.json", expected)
    width = len(verifier.CODES)
    for i, photo in enumerate(photos):
        key = verifier.stable_hash((fingerprint, "control", asdict(photo)))
        rows = [row.to_record() for row in arms["control"][i * width : (i + 1) * width]]
        write_once(
            run / "control/cache" / f"{key}.json",
            {
                "cache_key": key,
                "rows": rows,
                "rows_sha256": verifier.stable_hash(rows),
            },
        )
    write_once(
        run / "mixed-cache-provenance.json",
        {
            "source_seal_sha256": verifier.digest(source / "seal.json"),
            "source_identity_sha256": verifier.digest(source / "identity.json"),
            "source_predictions_sha256": verifier.digest(
                source / "control/predictions.json"
            ),
            "expected_fingerprint": fingerprint,
            "copied_conditions": len(arms["control"]),
            "implementation_sha256": verifier.digest(Path(__file__)),
        },
    )
    return run


def verify_comparison(
    source: Path, run: Path
) -> tuple[tuple[verifier.Photo, ...], dict[str, tuple[verifier.Margin, ...]]]:
    """Require unchanged reused margins and paired encoded inputs before any gold."""
    _, photos, original = verifier.verify_run(source)
    _, focused_photos, focused = verifier.verify_run(run)
    require_replacement(
        read_identity(source / "identity.json"), read_identity(run / "identity.json")
    )
    if photos != focused_photos or original["control"] != focused["control"]:
        raise verifier.EvaluationError(
            "Mixed control margins or cohort changed during reuse"
        )
    if [row.input_sha256 for row in original["control"]] != [
        row.input_sha256 for row in focused["mined"]
    ]:
        raise verifier.EvaluationError(
            "Focused encoded inputs differ from sealed mixed control"
        )
    return photos, focused


def predict(paths: Paths) -> Path:
    """Run the existing inference implementation only after exact cache reuse guards."""
    if not paths.out.resolve().is_relative_to((verifier.ROOT / "data").resolve()):
        raise verifier.EvaluationError(
            "Private focused predictions must remain under data/"
        )
    verifier.verify_run(paths.source)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    expected = runtime_identity(paths)
    run = prepare_cache(paths.source, paths.out, expected)
    actual = verifier.predict(
        argparse.Namespace(
            base=paths.base,
            control=paths.control,
            mined=paths.focused,
            prompt=paths.prompt,
            out=paths.out,
        )
    )
    if (
        actual.resolve() != run.resolve()
        or read_identity(actual / "identity.json") != expected
    ):
        raise verifier.EvaluationError(
            "Predictor returned a different protocol fingerprint"
        )
    verify_comparison(paths.source, run)
    return run


class Bootstrap(TypedDict):
    statistic: str
    delta_focused_minus_mixed: float
    resamples: int
    seed: int
    groups: int
    percentile_95: list[float]
    limitation: str


class Sensitivity(TypedDict):
    scope: str
    retained_photos: int
    excluded_photos: int
    mixed_fixed_3code_macro_f1: float
    focused_fixed_3code_macro_f1: float
    delta_focused_minus_mixed: float


class FocusedReport(TypedDict):
    scope: str
    photos: int
    candidate_codes: list[str]
    arms: dict[str, dict[str, Json]]
    paired_group_bootstrap: Bootstrap
    largest_group_excluded_secondary: Sensitivity
    threshold: str
    normal_definition: str
    severe_definition: str
    source_seal_sha256: str
    focused_seal_sha256: str
    reused_mixed_predictions_sha256: str
    focused_predictions_sha256: str
    implementation_sha256: str


def largest_group_sensitivity(
    photos: tuple[verifier.Photo, ...],
    labels: Labels,
    predictions: tuple[list[frozenset[str]], list[frozenset[str]]],
) -> Sensitivity:
    """Drop the largest metadata group without selecting thresholds or checkpoints."""
    groups: dict[str, int] = {}
    for photo in photos:
        groups[photo.group_id] = groups.get(photo.group_id, 0) + 1
    largest = min(groups, key=lambda key: (-groups[key], key))
    indices = [i for i, photo in enumerate(photos) if photo.group_id != largest]
    if not indices:
        raise verifier.EvaluationError(
            "Largest-group sensitivity needs more than one group"
        )
    gold = [labels.hazards[i] for i in indices]
    values = [
        classification_metrics(gold, [arm[i] for i in indices])["fixed_3code_macro_f1"]
        for arm in predictions
    ]
    return {
        "scope": "Secondary unretuned sensitivity; primary retains all photos. Size ties use sorted group key.",
        "retained_photos": len(indices),
        "excluded_photos": groups[largest],
        "mixed_fixed_3code_macro_f1": values[0],
        "focused_fixed_3code_macro_f1": values[1],
        "delta_focused_minus_mixed": values[1] - values[0],
    }


def paired_bootstrap(
    photos: tuple[verifier.Photo, ...],
    labels: Labels,
    predictions: tuple[list[frozenset[str]], list[frozenset[str]]],
) -> Bootstrap:
    """Resample whole manifest groups with replacement, using the same draw in both arms."""
    groups: dict[str, list[int]] = {}
    for i, photo in enumerate(photos):
        groups.setdefault(photo.group_id, []).append(i)
    ordered = [groups[key] for key in sorted(groups)]
    generator = random.Random(SEED)
    deltas = []
    for _ in range(REPLICATIONS):
        indices = [i for _ in ordered for i in generator.choice(ordered)]
        gold = [labels.hazards[i] for i in indices]
        values = [
            classification_metrics(gold, [arm[i] for i in indices])[
                "fixed_3code_macro_f1"
            ]
            for arm in predictions
        ]
        deltas.append(values[1] - values[0])
    measured = [
        classification_metrics(list(labels.hazards), arm)["fixed_3code_macro_f1"]
        for arm in predictions
    ]
    return {
        "statistic": "fixed_3code_macro_f1",
        "delta_focused_minus_mixed": measured[1] - measured[0],
        "resamples": REPLICATIONS,
        "seed": SEED,
        "groups": len(groups),
        "percentile_95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "limitation": "Conditional development-set group bootstrap; metadata groups are not independently verified events.",
    }


def score(source: Path, run: Path, gold_path: Path) -> FocusedReport:
    """Use existing score only after checking complete seals and exact mixed reuse."""
    photos, arms = verify_comparison(source, run)
    intermediate = run / "focused-verifier-score.json"
    verifier.score(run, gold_path, intermediate)
    selected = {photo.sample_id for photo in photos}
    gold = {}
    with gold_path.open() as stream:
        for line in stream:
            item = json.loads(line, object_pairs_hook=unique_keys)
            if item["sample_id"] in selected:
                if item["sample_id"] in gold:
                    raise verifier.EvaluationError("Duplicate gold photo")
                gold[item["sample_id"]] = item["target"]["hazards"]
    if set(gold) != selected:
        raise verifier.EvaluationError("Gold coverage differs from paired cohort")
    labels = Labels(
        tuple(frozenset(gold[p.sample_id]) for p in photos),
        tuple(
            frozenset(
                code
                for code, hazard in gold[p.sample_id].items()
                if code in verifier.CODES
                and int(hazard["risk_level"]["severity"]["score"]) == 4
            )
            for p in photos
        ),
    )
    predictions: list[list[frozenset[str]]] = []
    summary = json.loads(intermediate.read_text(), object_pairs_hook=unique_keys)
    renamed: dict[str, dict[str, Json]] = {}
    for arm, name in (("control", "mixed"), ("mined", "focused")):
        by_id: dict[str, set[str]] = {photo.sample_id: set() for photo in photos}
        for row in arms[arm]:
            if row.margin > 0:
                by_id[row.sample_id].add(row.code)
        predicted = [frozenset(by_id[p.sample_id]) for p in photos]
        predictions.append(predicted)
        severe_photos = sum(bool(codes) for codes in labels.severe)
        complete_misses = sum(
            bool(g) and not bool(g & p)
            for g, p in zip(labels.severe, predicted, strict=True)
        )
        renamed[name] = {
            **summary["arms"][arm],
            "severity4_candidate_photos": severe_photos,
            "severity4_candidate_complete_miss_photos": complete_misses,
            "severity4_candidate_complete_miss_rate": complete_misses / severe_photos
            if severe_photos
            else None,
        }
    return {
        "scope": "Fixed checkpoint-12 mixed versus focused, same 201 development photos; no checkpoint selection.",
        "photos": len(photos),
        "candidate_codes": list(verifier.CODES),
        "arms": renamed,
        "paired_group_bootstrap": paired_bootstrap(
            photos, labels, (predictions[0], predictions[1])
        ),
        "largest_group_excluded_secondary": largest_group_sensitivity(
            photos, labels, (predictions[0], predictions[1])
        ),
        "threshold": "Accept iff sealed margin > 0; zero ties reject.",
        "normal_definition": "No gold hazard of any code; predictions restricted to three fixed candidates.",
        "severe_definition": "Candidate severity-4 photos; complete miss means none of their candidate severity-4 codes accepted.",
        "source_seal_sha256": verifier.digest(source / "seal.json"),
        "focused_seal_sha256": verifier.digest(run / "seal.json"),
        "reused_mixed_predictions_sha256": verifier.digest(
            source / "control/predictions.json"
        ),
        "focused_predictions_sha256": verifier.digest(run / "mined/predictions.json"),
        "implementation_sha256": verifier.digest(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "score"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--focused", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=verifier.ROOT / "data/ubuntu3050-runs/focused12-paired-eval",
    )
    parser.add_argument(
        "--base", type=Path, default=verifier.ROOT / "data/baseline/base_model"
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=verifier.ROOT
        / "data/ubuntu3050-runs/qlora-inner12-base-control/checkpoint-12",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=verifier.ROOT / "kwater_lab/system_prompt_11code.txt",
    )
    parser.add_argument(
        "--gold", type=Path, default=verifier.ROOT / "data/train/train.jsonl"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.focused is None:
            raise verifier.EvaluationError(
                "run requires a completed --focused checkpoint-12"
            )
        run = predict(
            Paths(
                args.source,
                args.out,
                args.base,
                args.control,
                args.focused,
                args.prompt,
            )
        )
    else:
        if args.run is None:
            raise verifier.EvaluationError("score requires an already sealed --run")
        run = args.run
    verifier.write_json(args.report, score(args.source, run, args.gold))
    print(f"AGGREGATE_REPORT={args.report}")


if __name__ == "__main__":
    main()
