"""One global operating threshold and paired development-only statistics."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import TypedDict

from . import ubuntu_focused_eval as paired
from .ubuntu_image_alignment import Labels, auc
from .ubuntu_inner_verifier_eval import CODES, EvaluationError, Margin, Photo
from .ubuntu_roi_readout import Metrics, classification_metrics


class Selection(TypedDict):
    threshold: float
    fixed_3code_macro_f1: float
    normal_photos: int
    normal_false_alarm_photos: int
    false_alarm_cap: int
    candidate_count: int
    all_reject_threshold: float


class OperatingMetrics(Metrics):
    severity4_candidate_photos: int
    severity4_candidate_complete_miss_photos: int
    severity4_candidate_instances: int
    severity4_candidate_missed_instances: int


class Change(TypedDict):
    recovered_true_positives: int
    lost_true_positives: int
    added_false_positives: int
    removed_false_positives: int


class Bootstrap(TypedDict):
    statistic: str
    delta_calibrated_minus_zero: float
    resamples: int
    seed: int
    groups: int
    percentile_95: list[float]
    limitation: str


class Sensitivity(TypedDict):
    scope: str
    retained_photos: int
    excluded_photos: int
    zero_fixed_3code_macro_f1: float
    calibrated_fixed_3code_macro_f1: float
    delta_calibrated_minus_zero: float


class Comparison(TypedDict):
    scope: str
    arms: dict[str, OperatingMetrics]
    changes_by_code: dict[str, Change]
    paired_group_bootstrap: Bootstrap
    largest_group_excluded_secondary: Sensitivity
    secondary_threshold_free_auc: dict[str, float | None]
    actual_metadata_groups: int
    largest_group_photos: int


def decisions(rows: tuple[Margin, ...], threshold: float) -> list[frozenset[str]]:
    """Use the existing strict margin comparison; equality rejects."""
    return [
        frozenset(
            row.code
            for row in rows[start : start + len(CODES)]
            if row.margin > threshold
        )
        for start in range(0, len(rows), len(CODES))
    ]


def exact_macro(
    gold: tuple[frozenset[str], ...], predicted: list[frozenset[str]]
) -> Fraction:
    """Exact rational F1 makes selection ties independent of floating-point summation."""
    total = Fraction()
    for code in CODES:
        tp = sum(code in g and code in p for g, p in zip(gold, predicted, strict=True))
        fp = sum(
            code not in g and code in p for g, p in zip(gold, predicted, strict=True)
        )
        fn = sum(
            code in g and code not in p for g, p in zip(gold, predicted, strict=True)
        )
        denominator = 2 * tp + fp + fn
        total += Fraction(2 * tp, denominator) if denominator else Fraction()
    return total / len(CODES)


def select_threshold(labels: Labels, rows: tuple[Margin, ...]) -> Selection:
    """Maximize fixed-three-code F1 under the normal-photo cap, with frozen ties."""
    if not rows or len(rows) != len(labels.hazards) * len(CODES):
        raise EvaluationError("Threshold selection requires a complete nonempty cohort")
    normals = sum(not codes for codes in labels.hazards)
    if not normals:
        raise EvaluationError("Normal-photo calibration requires at least one normal")
    cap = normals // 10
    candidates = sorted({row.margin for row in rows})
    eligible: list[tuple[Fraction, int, float]] = []
    for threshold in candidates:
        predicted = decisions(rows, threshold)
        alarms = sum(
            not g and bool(p) for g, p in zip(labels.hazards, predicted, strict=True)
        )
        if alarms <= cap:
            eligible.append(
                (exact_macro(labels.hazards, predicted), -alarms, threshold)
            )
    metric, negative_alarms, threshold = max(eligible)
    return {
        "threshold": threshold,
        "fixed_3code_macro_f1": float(metric),
        "normal_photos": normals,
        "normal_false_alarm_photos": -negative_alarms,
        "false_alarm_cap": cap,
        "candidate_count": len(candidates),
        "all_reject_threshold": candidates[-1],
    }


def operating_metrics(
    labels: Labels, predicted: list[frozenset[str]]
) -> OperatingMetrics:
    """Normal means no hazard in any code; severe misses use the fixed candidate codes."""
    return {
        **classification_metrics(list(labels.hazards), predicted),
        "severity4_candidate_photos": sum(bool(row) for row in labels.severe),
        "severity4_candidate_complete_miss_photos": sum(
            bool(g) and not bool(g & p)
            for g, p in zip(labels.severe, predicted, strict=True)
        ),
        "severity4_candidate_instances": sum(map(len, labels.severe)),
        "severity4_candidate_missed_instances": sum(
            len(g - p) for g, p in zip(labels.severe, predicted, strict=True)
        ),
    }


def compare(
    photos: tuple[Photo, ...],
    labels: Labels,
    rows: tuple[Margin, ...],
    threshold: float,
) -> Comparison:
    """Compare fixed zero with one locked threshold using identical sealed margins."""
    baseline, selected = decisions(rows, 0.0), decisions(rows, threshold)
    changes: dict[str, Change] = {}
    for code in CODES:
        triples = list(zip(labels.hazards, baseline, selected, strict=True))
        changes[code] = {
            "recovered_true_positives": sum(
                code in g and code not in b and code in s for g, b, s in triples
            ),
            "lost_true_positives": sum(
                code in g and code in b and code not in s for g, b, s in triples
            ),
            "added_false_positives": sum(
                code not in g and code not in b and code in s for g, b, s in triples
            ),
            "removed_false_positives": sum(
                code not in g and code in b and code not in s for g, b, s in triples
            ),
        }
    groups = Counter(photo.group_id for photo in photos)
    aucs = {
        code: auc(
            [code in g for g in labels.hazards],
            [row.margin for row in rows if row.code == code],
        )
        for code in CODES
    }
    defined = [value for value in aucs.values() if value is not None]
    aucs["macro_defined_codes"] = sum(defined) / len(defined) if defined else None
    bootstrap = paired.paired_bootstrap(photos, labels, (baseline, selected))
    sensitivity = paired.largest_group_sensitivity(photos, labels, (baseline, selected))
    return {
        "scope": "Development-only conditional three-code operating-point evaluation; not independent test or full JSON performance.",
        "arms": {
            "fixed_zero": operating_metrics(labels, baseline),
            "calibrated": operating_metrics(labels, selected),
        },
        "changes_by_code": changes,
        "paired_group_bootstrap": {
            "statistic": bootstrap["statistic"],
            "delta_calibrated_minus_zero": bootstrap["delta_focused_minus_mixed"],
            "resamples": bootstrap["resamples"],
            "seed": bootstrap["seed"],
            "groups": bootstrap["groups"],
            "percentile_95": bootstrap["percentile_95"],
            "limitation": bootstrap["limitation"],
        },
        "largest_group_excluded_secondary": {
            "scope": sensitivity["scope"],
            "retained_photos": sensitivity["retained_photos"],
            "excluded_photos": sensitivity["excluded_photos"],
            "zero_fixed_3code_macro_f1": sensitivity["mixed_fixed_3code_macro_f1"],
            "calibrated_fixed_3code_macro_f1": sensitivity[
                "focused_fixed_3code_macro_f1"
            ],
            "delta_calibrated_minus_zero": sensitivity["delta_focused_minus_mixed"],
        },
        "secondary_threshold_free_auc": aucs,
        "actual_metadata_groups": len(groups),
        "largest_group_photos": max(groups.values()),
    }
