"""CPU-only cross-group image reassignment diagnostic of sealed paired margins.

Run metadata-only preflight before scoring: python -m kwater_lab.ubuntu_image_alignment
preflight --plan data/image-alignment/plan.json --report <aggregate.json>.
No inference, prompt changes, threshold tuning, or independent evidence is produced.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Final, TypedDict

from .metrics import unique_keys
from .ubuntu_inner_verifier_eval import (
    ARMS,
    CODES,
    GOLD_SHA,
    ROOT,
    EvaluationError,
    Margin,
    Photo,
    digest,
    load_cohort,
    stable_hash,
    verify_run,
    write_json,
)
from .vlm_11code import verify_user_text

SEED: Final = 20261127
REPLICATIONS: Final = 256


class MappingPlan(TypedDict):
    schema: str
    seed: int
    replications: int
    cohort_sha256: str
    donor_indices: list[list[int]]


@dataclass(frozen=True, slots=True)
class Labels:
    hazards: tuple[frozenset[str], ...]
    severe: tuple[frozenset[str], ...]


class Metrics(TypedDict):
    per_code_auc: dict[str, float | None]
    macro_auc: float | None
    auc_eligible_codes: int
    tp: int
    fp: int
    fn: int
    tn: int
    pooled_f1: float
    fixed_code_macro_f1: float
    normal_photos: int
    normal_false_alarm_photos: int
    normal_false_alarm_rate: float | None
    severe_instances: int
    severe_misses: int
    severe_recall: float | None
    approvals_per_code: dict[str, int]


def validate_mapping(photos: tuple[Photo, ...], mapping: list[int]) -> None:
    """Require every donor once, with neither self nor same-group assignment."""
    if sorted(mapping) != list(range(len(photos))):
        raise EvaluationError("Donors must form a complete bijection")
    if any(photos[i].group_id == photos[j].group_id for i, j in enumerate(mapping)):
        raise EvaluationError("Donor assignment crosses neither self nor group")


def build_plan(photos: tuple[Photo, ...]) -> MappingPlan:
    """Sample constrained rotations without reading predictions or labels.

    Shuffle group blocks and members; rotate by a uniform integer between largest
    block size and N minus that size. This is not uniform over all derangements.
    The entire cohort is eligible iff no group contains more than half its photos.
    """
    groups: dict[str, list[int]] = {}
    for index, photo in enumerate(photos):
        groups.setdefault(photo.group_id, []).append(index)
    largest = max(map(len, groups.values()), default=0)
    if not photos or largest * 2 > len(photos):
        raise EvaluationError("Whole-cohort cross-group bijection is infeasible")
    generator = random.Random(SEED)
    mappings = []
    for _ in range(REPLICATIONS):
        keys = sorted(groups)
        generator.shuffle(keys)
        ordered = []
        for key in keys:
            members = groups[key].copy()
            generator.shuffle(members)
            ordered.extend(members)
        offset = generator.randint(largest, len(photos) - largest)
        mapping = [0] * len(photos)
        for position, recipient in enumerate(ordered):
            mapping[recipient] = ordered[(position + offset) % len(photos)]
        validate_mapping(photos, mapping)
        mappings.append(mapping)
    return {
        "schema": "sealed-image-alignment-v1",
        "seed": SEED,
        "replications": REPLICATIONS,
        "cohort_sha256": stable_hash(
            [(p.sample_id, p.sha256, p.group_id) for p in photos]
        ),
        "donor_indices": mappings,
    }


def auc(labels: list[bool], scores: list[float]) -> float | None:
    """Mann-Whitney AUROC; ties receive half credit, absent classes are undefined."""
    positive = [score for label, score in zip(labels, scores, strict=True) if label]
    negative = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (
        len(positive) * len(negative)
    )


def evaluate(labels: Labels, scores: tuple[tuple[float, ...], ...]) -> Metrics:
    """Fixed-zero candidate-code operating metrics; normal uses all gold codes."""
    predicted = [
        frozenset(c for c, m in zip(CODES, row, strict=True) if m > 0) for row in scores
    ]
    per_code = {
        code: auc([code in row for row in labels.hazards], [row[k] for row in scores])
        for k, code in enumerate(CODES)
    }
    defined = [value for value in per_code.values() if value is not None]
    counts = []
    for code in CODES:
        pairs = list(zip(labels.hazards, predicted, strict=True))
        counts.append(
            (
                sum(code in g and code in p for g, p in pairs),
                sum(code not in g and code in p for g, p in pairs),
                sum(code in g and code not in p for g, p in pairs),
            )
        )
    tp, fp, fn = (sum(row[k] for row in counts) for k in range(3))
    normals = sum(not row for row in labels.hazards)
    false_alarms = sum(
        not g and bool(p) for g, p in zip(labels.hazards, predicted, strict=True)
    )
    severe = sum(map(len, labels.severe))
    misses = sum(len(g - p) for g, p in zip(labels.severe, predicted, strict=True))
    return {
        "per_code_auc": per_code,
        "macro_auc": mean(defined) if defined else None,
        "auc_eligible_codes": len(defined),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": len(scores) * len(CODES) - tp - fp - fn,
        "pooled_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fixed_code_macro_f1": mean(
            2 * t / (2 * t + f + n) if 2 * t + f + n else 0.0 for t, f, n in counts
        ),
        "normal_photos": normals,
        "normal_false_alarm_photos": false_alarms,
        "normal_false_alarm_rate": false_alarms / normals if normals else None,
        "severe_instances": severe,
        "severe_misses": misses,
        "severe_recall": (severe - misses) / severe if severe else None,
        "approvals_per_code": {
            code: sum(code in p for p in predicted) for code in CODES
        },
    }


class Range(TypedDict):
    mean: float
    minimum: float
    maximum: float


class ArmReport(TypedDict):
    correct_image: Metrics
    donor_diagnostic_ranges: dict[str, Range]
    donor_per_code_auc_ranges: dict[str, Range]
    correct_minus_donor: dict[str, Range]
    paired_margin_concordance_gold_positive: Range | None
    marginal_approvals_preserved_all_replications: bool


class AlignmentReport(TypedDict):
    source_sha256: dict[str, str]
    plan_sha256: str
    photos: int
    arms: dict[str, ArmReport]
    seed: int
    replications: int
    scope: str
    sampling: str
    range_interpretation: str
    group_limitation: str
    auc_definition: str
    concordance_definition: str
    severe_denominator: str
    normal_definition: str
    threshold: str


def spread(values: list[float]) -> Range:
    return {"mean": mean(values), "minimum": min(values), "maximum": max(values)}


def compare(labels: Labels, rows: tuple[Margin, ...], plan: MappingPlan) -> ArmReport:
    """Use exact sealed margins for donors, preserving each code's marginal scores."""
    width = len(CODES)
    scores = tuple(
        tuple(row.margin for row in rows[i : i + width])
        for i in range(0, len(rows), width)
    )
    correct = evaluate(labels, scores)
    shuffled: list[Metrics] = []
    concordances: list[float] = []
    positives = [
        (i, k)
        for i, gold in enumerate(labels.hazards)
        for k, code in enumerate(CODES)
        if code in gold
    ]
    for mapping in plan["donor_indices"]:
        donors = tuple(scores[index] for index in mapping)
        result = evaluate(labels, donors)
        if result["approvals_per_code"] != correct["approvals_per_code"]:
            raise EvaluationError("Reassignment changed marginal approvals")
        shuffled.append(result)
        if positives:
            concordances.append(
                mean(
                    (scores[i][k] > donors[i][k]) + 0.5 * (scores[i][k] == donors[i][k])
                    for i, k in positives
                )
            )
    fields = (
        "macro_auc",
        "pooled_f1",
        "fixed_code_macro_f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "normal_false_alarm_photos",
        "normal_false_alarm_rate",
        "severe_misses",
        "severe_recall",
    )
    ranges = {
        field: spread(
            [value for result in shuffled if (value := result[field]) is not None]
        )
        for field in fields
        if correct[field] is not None
    }
    return {
        "correct_image": correct,
        "donor_diagnostic_ranges": ranges,
        "donor_per_code_auc_ranges": {
            code: spread(
                [
                    value
                    for result in shuffled
                    if (value := result["per_code_auc"][code]) is not None
                ]
            )
            for code in CODES
            if correct["per_code_auc"][code] is not None
        },
        "correct_minus_donor": {
            field: spread(
                [
                    original - value
                    for result in shuffled
                    if (value := result[field]) is not None
                ]
            )
            for field in ("macro_auc", "pooled_f1", "fixed_code_macro_f1")
            if (original := correct[field]) is not None
        },
        "paired_margin_concordance_gold_positive": spread(concordances)
        if concordances
        else None,
        "marginal_approvals_preserved_all_replications": True,
    }


def score(run: Path, plan_path: Path, gold_path: Path) -> AlignmentReport:
    """Validate the whole sealed source and frozen mapping before opening gold."""
    identity, photos, arms = verify_run(run)
    if photos != load_cohort(ROOT):
        raise EvaluationError("Sealed photos differ from canonical metadata")
    raw_identity = json.loads(
        (run / "identity.json").read_text(), object_pairs_hook=unique_keys
    )
    if raw_identity["user_text"] != {code: verify_user_text(code) for code in CODES}:
        raise EvaluationError("Image-independent candidate prompt differs")
    for name, expected in raw_identity["source"].items():
        if (
            name not in ("ubuntu_inner_verifier_eval.py", "vlm_11code.py")
            or digest(Path(__file__).with_name(name)) != expected
        ):
            raise EvaluationError("Sealed inference source changed")
    if set(raw_identity["source"]) != {
        "ubuntu_inner_verifier_eval.py",
        "vlm_11code.py",
    }:
        raise EvaluationError("Incomplete inference source provenance")
    if raw_identity["prompt_sha256"] != digest(
        Path(__file__).with_name("system_prompt_11code.txt")
    ):
        raise EvaluationError("Sealed system prompt differs")
    plan: MappingPlan = json.loads(plan_path.read_text(), object_pairs_hook=unique_keys)
    if plan != build_plan(photos):
        raise EvaluationError("Frozen metadata-only mapping differs")
    source_hashes = {
        name: digest(run / name)
        for name in (
            "seal.json",
            "identity.json",
            "control/predictions.json",
            "mined/predictions.json",
        )
    }
    if identity["gold_sha256"] != GOLD_SHA or digest(gold_path) != GOLD_SHA:
        raise EvaluationError("Gold source differs from canonical protocol")
    selected = {photo.sample_id for photo in photos}
    gold = {}
    with gold_path.open() as stream:
        for line in stream:
            item = json.loads(line, object_pairs_hook=unique_keys)
            if item["sample_id"] in selected:
                if item["sample_id"] in gold:
                    raise EvaluationError("Duplicate selected gold sample")
                gold[item["sample_id"]] = item["target"]["hazards"]
    if set(gold) != selected:
        raise EvaluationError("Gold coverage differs from sealed photos")
    labels = Labels(
        tuple(frozenset(gold[p.sample_id]) for p in photos),
        tuple(
            frozenset(
                code
                for code, risk in gold[p.sample_id].items()
                if code in CODES and int(risk["risk_level"]["severity"]["score"]) == 4
            )
            for p in photos
        ),
    )
    return {
        "source_sha256": source_hashes,
        "plan_sha256": digest(plan_path),
        "photos": len(photos),
        "arms": {arm: compare(labels, arms[arm], plan) for arm in ARMS},
        "seed": SEED,
        "replications": REPLICATIONS,
        "scope": "Development-only image-match diagnostic; no independent performance evidence.",
        "sampling": "Same randomized group-block rotations for both arms; not uniform derangements.",
        "range_interpretation": "Finite constrained-permutation diagnostic range, NOT confidence interval.",
        "group_limitation": "Manifest groups are exclusion proxies, not independently verified events.",
        "auc_definition": "Unweighted mean of within-code AUROC with both classes; ties half credit.",
        "concordance_definition": "Correct versus donor margin within gold-positive photo/code pairs; NOT AUROC.",
        "severe_denominator": "Gold severity-4 instances only within eligible fixed candidate codes.",
        "normal_definition": "No gold hazards of any code; false alarms use fixed candidate codes.",
        "threshold": "Accept iff sealed margin > 0; zero ties reject.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "score"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gold", type=Path, default=ROOT / "data/train/train.jsonl")
    args = parser.parse_args()
    if not args.plan.resolve().is_relative_to((ROOT / "data").resolve()):
        raise EvaluationError("Private mapping must remain under local data/")
    if args.command == "preflight":
        photos = load_cohort(ROOT)
        plan = build_plan(photos)
        if args.plan.exists() and json.loads(args.plan.read_text()) != plan:
            raise EvaluationError("Existing mapping differs; refusing replacement")
        write_json(args.plan, plan)
        groups = Counter(photo.group_id for photo in photos)
        report = {
            "status": "metadata_only_preflight_pass",
            "gold_opened": False,
            "predictions_opened": False,
            "photos": len(photos),
            "eligible_photos": len(photos),
            "groups": len(groups),
            "largest_group": max(groups.values()),
            "seed": SEED,
            "replications": REPLICATIONS,
            "all_mappings_bijective_cross_group": True,
            "plan_sha256": digest(args.plan),
            "cohort_sha256": plan["cohort_sha256"],
        }
    else:
        if args.run is None:
            raise EvaluationError("Scoring requires an already sealed --run")
        report = score(args.run, args.plan, args.gold)
    write_json(args.report, report)
    print(f"AGGREGATE_REPORT={args.report}")


if __name__ == "__main__":
    main()
