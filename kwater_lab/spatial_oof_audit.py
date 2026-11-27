"""Cache-only paired OOF audit of frozen Qwen global and top-3 regional heads.

Run with the project environment: python -m kwater_lab.spatial_oof_audit --help.
No image extraction, threshold selection, or sealed-set evaluation is performed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from .metrics import CODES
from .spatial_probe import (
    ALLOWED_FOLDS,
    Row,
    _fit_scores,
    digest,
    encoder_fingerprint,
    load_rows,
    read_cached,
)

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class AuditReport(TypedDict):
    scope: str
    folds: list[int]
    photos: int
    excluded: str
    fit_steps: int
    seed: int
    fit_device: str
    top_k: int
    parameter_budget: str
    bootstrap: dict[str, JsonValue]
    negative_control: dict[str, JsonValue]
    normal_control: dict[str, JsonValue]
    per_code_definition: str
    multilabel_note: str
    observed: dict[str, JsonValue]
    train_label_shuffle: dict[str, JsonValue]
    per_fold: list[dict[str, JsonValue]]
    inputs: NotRequired[dict[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class AuditInputError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class FitConfig:
    steps: int = 160
    seed: int = 20261127
    device: str = "cpu"
    bootstrap_resamples: int = 2000


@dataclass(frozen=True, slots=True)
class Dataset:
    rows: tuple[Row, ...]
    features: NDArray[np.float32]
    labels: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Predictions:
    global_top1: NDArray[np.int64]
    regional_top1: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Contingency:
    both_hit: int
    global_only: int
    regional_only: int
    neither_hit: int


@dataclass(frozen=True, slots=True)
class CodeChange:
    code: str
    support: int
    global_hits: int
    regional_hits: int
    gains: int
    losses: int


@dataclass(frozen=True, slots=True)
class ClusterInterval:
    lower: float | None
    upper: float | None
    valid_resamples: int
    rejected_zero_denominator_draws: int
    event_groups: int
    seed: int


@dataclass(frozen=True, slots=True)
class PairedSummary:
    abnormal_photos: int
    global_hits: int
    regional_hits: int
    global_recall_at_1: float | None
    regional_recall_at_1: float | None
    delta_regional_minus_global: float | None
    contingency: Contingency
    cluster_percentile_95_ci: ClusterInterval
    per_code: tuple[CodeChange, ...]
    normal_photos: int
    normal_global_proposals: dict[str, int]
    normal_regional_proposals: dict[str, int]


def load_dataset(rows: list[Row], cache_dir: Path, encoder_id: str) -> Dataset:
    """Reject duplicate sources and cross-fold events before accessing cached grids."""
    if {row.fold for row in rows} != set(ALLOWED_FOLDS):
        raise AuditInputError("expected exactly development folds 1, 2, 3")
    for name, values in (
        ("sample IDs", [row.sample_id for row in rows]),
        ("source hashes", [row.sha256 for row in rows]),
    ):
        if len(values) != len(set(values)):
            raise AuditInputError(f"duplicate {name} in selected rows")
    group_folds: dict[str, int] = {}
    for row in rows:
        if not row.group_id or not row.codes.issubset(CODES):
            raise AuditInputError("empty event group or unknown risk code")
        if group_folds.setdefault(row.group_id, row.fold) != row.fold:
            raise AuditInputError("incident group crosses fit/validation folds")
    grids = [read_cached(row, cache_dir, encoder_id) for row in rows]
    if any(grid is None for grid in grids):
        raise AuditInputError(
            "all selected rows require a matching frozen feature cache"
        )
    features = np.stack([grid for grid in grids if grid is not None])
    labels = np.asarray(
        [[code in row.codes for code in CODES] for row in rows], dtype=bool
    )
    return Dataset(tuple(rows), features, labels)


def paired_interval(
    rows: tuple[Row, ...], outcomes: NDArray[np.int64], config: FitConfig
) -> ClusterInterval:
    """Bootstrap complete events within folds, pooling photo-weighted hit deltas.

    Columns are paired hit delta and abnormal-photo denominator. Normal-only
    groups remain in the sampling frame. Empty-denominator draws are retried.
    This conditions on already-fitted OOF heads; it does not refit models.
    """
    totals: dict[tuple[int, str], NDArray[np.int64]] = {}
    for row, outcome in zip(rows, outcomes, strict=True):
        key = (row.fold, row.group_id)
        totals[key] = totals.get(key, np.zeros(2, dtype=np.int64)) + outcome
    if outcomes[:, 1].sum() == 0:
        return ClusterInterval(None, None, 0, 0, len(totals), config.seed)
    strata = [
        np.stack([totals[key] for key in sorted(totals) if key[0] == fold])
        for fold in sorted({row.fold for row in rows})
    ]
    generator = np.random.default_rng(config.seed)
    deltas: list[float] = []
    rejected = 0
    while len(deltas) < config.bootstrap_resamples:
        sampled = np.zeros(2, dtype=np.int64)
        for groups in strata:
            sampled += groups[generator.integers(len(groups), size=len(groups))].sum(
                axis=0
            )
        if sampled[1] == 0:
            rejected += 1
            if rejected > 20 * config.bootstrap_resamples:
                raise AuditInputError(
                    "too many empty-denominator cluster bootstrap draws"
                )
            continue
        deltas.append(float(sampled[0] / sampled[1]))
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return ClusterInterval(
        float(lower), float(upper), len(deltas), rejected, len(totals), config.seed
    )


def summarize(
    dataset: Dataset, predictions: Predictions, config: FitConfig
) -> PairedSummary:
    """Count paired outcomes on abnormal photos; normals have proposals, no FPR."""
    labels = dataset.labels
    abnormal = labels.any(axis=1)
    normal = ~abnormal
    indices = np.arange(len(labels))
    global_hit = labels[indices, predictions.global_top1]
    regional_hit = labels[indices, predictions.regional_top1]
    denominator = int(abnormal.sum())
    global_hits, regional_hits = int(global_hit.sum()), int(regional_hit.sum())
    outcomes = np.column_stack(
        (
            regional_hit.astype(np.int64) - global_hit.astype(np.int64),
            abnormal.astype(np.int64),
        )
    )
    per_code: list[CodeChange] = []
    for column, code in enumerate(CODES):
        global_code = labels[:, column] & (predictions.global_top1 == column)
        regional_code = labels[:, column] & (predictions.regional_top1 == column)
        per_code.append(
            CodeChange(
                code,
                int(labels[:, column].sum()),
                int(global_code.sum()),
                int(regional_code.sum()),
                int((regional_code & ~global_code).sum()),
                int((global_code & ~regional_code).sum()),
            )
        )
    return PairedSummary(
        denominator,
        global_hits,
        regional_hits,
        global_hits / denominator if denominator else None,
        regional_hits / denominator if denominator else None,
        (regional_hits - global_hits) / denominator if denominator else None,
        Contingency(
            int((global_hit & regional_hit).sum()),
            int((global_hit & ~regional_hit).sum()),
            int((regional_hit & ~global_hit).sum()),
            int((abnormal & ~global_hit & ~regional_hit).sum()),
        ),
        paired_interval(dataset.rows, outcomes, config),
        tuple(per_code),
        int(normal.sum()),
        {
            code: int((normal & (predictions.global_top1 == col)).sum())
            for col, code in enumerate(CODES)
        },
        {
            code: int((normal & (predictions.regional_top1 == col)).sum())
            for col, code in enumerate(CODES)
        },
    )


def shuffle_training_labels(
    labels: NDArray[np.float32], seed: int
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Permute complete training label vectors without changing their source."""
    permutation = np.random.default_rng(seed).permutation(len(labels))
    return labels[permutation], permutation


def evaluate(dataset: Dataset, config: FitConfig) -> AuditReport:
    """Fit matched heads and one train-label shuffle control in each held-out fold."""
    if config.steps < 1 or config.bootstrap_resamples < 1:
        raise AuditInputError("fit steps and bootstrap resamples must be positive")
    predictions = {
        control: Predictions(
            np.full(len(dataset.rows), -1, dtype=np.int64),
            np.full(len(dataset.rows), -1, dtype=np.int64),
        )
        for control in ("observed", "train_label_shuffle")
    }
    folds: list[dict[str, JsonValue]] = []
    for fold in ALLOWED_FOLDS:
        valid = np.asarray([row.fold == fold for row in dataset.rows])
        train = ~valid
        train_labels = dataset.labels[train].astype(np.float32)
        eligible = np.flatnonzero(train_labels.any(axis=0))
        if not len(eligible):
            raise AuditInputError(f"fold {fold} training split contains no risk labels")
        shuffled, permutation = shuffle_training_labels(
            train_labels, config.seed + fold
        )
        fold_record: dict[str, JsonValue] = {
            "fold": fold,
            "fit_photos": int(train.sum()),
            "validation_photos": int(valid.sum()),
            "fit_groups": len(
                {
                    row.group_id
                    for row, use in zip(dataset.rows, train, strict=True)
                    if use
                }
            ),
            "validation_groups": len(
                {
                    row.group_id
                    for row, use in zip(dataset.rows, valid, strict=True)
                    if use
                }
            ),
            "ranking_eligible_codes_from_fit_only": [
                CODES[column] for column in eligible
            ],
            "shuffle_seed": config.seed + fold,
            "shuffle_unchanged_label_vectors": int(
                np.all(shuffled == train_labels, axis=1).sum()
            ),
            "shuffle_fixed_row_assignments": int(
                (permutation == np.arange(len(permutation))).sum()
            ),
        }
        valid_data = Dataset(
            tuple(row for row in dataset.rows if row.fold == fold),
            dataset.features[valid],
            dataset.labels[valid],
        )
        for control, fitting_labels in (
            ("observed", train_labels),
            ("train_label_shuffle", shuffled),
        ):
            prediction = predictions[control]
            for regional, target in (
                (False, prediction.global_top1),
                (True, prediction.regional_top1),
            ):
                scores = _fit_scores(
                    dataset.features[train],
                    fitting_labels,
                    dataset.features[valid],
                    regional=regional,
                    top_k=3,
                    seed=config.seed + fold,
                    steps=config.steps,
                    device_name=config.device,
                )
                if not np.isfinite(scores).all():
                    raise AuditInputError("non-finite fitted score")
                target[valid] = eligible[scores[:, eligible].argmax(axis=1)]
            fold_record[control] = asdict(
                summarize(
                    valid_data,
                    Predictions(
                        prediction.global_top1[valid], prediction.regional_top1[valid]
                    ),
                    config,
                )
            )
        folds.append(fold_record)
    return {
        "scope": "Development OOF frozen-base feature audit; not an independent final performance estimate.",
        "folds": list(ALLOWED_FOLDS),
        "photos": len(dataset.rows),
        "excluded": "Fold0, Fold4, and sealed/holdout rows",
        "fit_steps": config.steps,
        "seed": config.seed,
        "fit_device": config.device,
        "top_k": 3,
        "parameter_budget": "Each head has D*C+C parameters; identical optimizer and fit steps.",
        "bootstrap": {
            "method": "2000 by default: paired event-cluster percentile bootstrap stratified by validation fold",
            "requested_resamples": config.bootstrap_resamples,
            "estimand": "Pooled abnormal-photo recall@1 difference, regional minus global",
            "conditions_on": "Fixed OOF predictions; excludes model retraining and model-selection uncertainty",
        },
        "negative_control": {
            "method": "One deterministic permutation of complete training label vectors per fold",
            "preserves": (
                "All 16-token image blocks, cross-token covariance, "
                "label-vector marginals, fit-only eligibility"
            ),
            "limitation": "Diagnostic single shuffle, not a permutation significance test or proof of spatial geometry",
        },
        "normal_control": {
            "classifier_threshold": None,
            "false_positive_rate": None,
            "interpretation": "Top-1 always proposes a risk code; normal histograms are not classification FPR.",
        },
        "per_code_definition": "Support is true-label photos; gains/losses are changes in correct code-specific top-1.",
        "multilabel_note": (
            "Per-code switches can occur with both photo-level heads correct; "
            "counts need not equal contingency gains."
        ),
        "observed": asdict(summarize(dataset, predictions["observed"], config)),
        "train_label_shuffle": asdict(
            summarize(dataset, predictions["train_label_shuffle"], config)
        ),
        "per_fold": folds,
    }


def main() -> None:
    # Match the existing research CLIs without modifying the shared environment.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--summary", type=Path, default=Path("results/clean_split_v1_summary.json")
    )
    parser.add_argument(
        "--base-model", type=Path, default=Path("data/baseline/base_model")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".runtime/spatial-qwen3vl-cache")
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--cache-precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--expected-rows", type=int, default=575)
    parser.add_argument(
        "--encoder-id",
        help="Explicit cache identity for synthetic tests; bypasses base fingerprint computation",
    )
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    rows = load_rows(args.manifest, args.summary)
    if len(rows) != args.expected_rows:
        raise AuditInputError(
            f"expected {args.expected_rows} selected rows, found {len(rows)}"
        )
    encoder_id = args.encoder_id or encoder_fingerprint(
        args.base_model, args.max_pixels, args.cache_precision
    )
    dataset = load_dataset(rows, args.cache_dir, encoder_id)
    report = evaluate(
        dataset, FitConfig(args.steps, args.seed, args.device, args.bootstrap_resamples)
    )
    report["inputs"] = {
        "manifest_sha256": digest(args.manifest),
        "summary_sha256": digest(args.summary),
        "encoder_sha256": encoder_id,
        "encoder_identity_explicit_override": args.encoder_id is not None,
        "max_pixels": args.max_pixels,
        "cache_precision": args.cache_precision,
        "cache_directory": str(args.cache_dir),
        "new_images_extracted": 0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.report)
    print(f"wrote aggregate-only paired OOF audit: {args.report}")


if __name__ == "__main__":
    main()
