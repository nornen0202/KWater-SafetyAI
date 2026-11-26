"""Compare Fold 0 predictions on the same whole-group visual exclusion mask."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .fold0_production_compare import fold_rows, image_id
from .law_reranker_11code import read_selected_cases
from .metrics import CODES, decode_target, evaluate, paired_bootstrap, read_index


MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
LABELS = Path("data/train/train.jsonl")
EXCLUSIONS = Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")
COMMON = tuple(code for code in CODES if code not in ("drowning_inundation", "confined_space", "thermal_weather"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prediction_map(path, expected, *, baseline=False):
    anon_ids = {image_id(sid): sid for sid in expected}
    prediction = read_index(path, lambda row: decode_target(row, local_ids=not baseline), limit_size=True)
    if prediction.keys() == anon_ids.keys():
        return {anon_ids[key]: codes for key, codes in prediction.items()}
    if not baseline and prediction.keys() == expected:
        return prediction
    raise ValueError(f"{'Baseline' if baseline else 'Alternative'} must cover all {len(expected)} original photos using one ID scheme")


def validate_mask(manifest_path, exclusions_path):
    exclusion = json.loads(exclusions_path.read_text(encoding="utf-8"))
    if exclusion.get("canonical_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Exclusion list and canonical development manifest SHA-256 differ")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = defaultdict(set)
    for row in manifest:
        if row["split"] != "dev":
            raise ValueError("Expected a development-only manifest")
        groups[row["group_id"]].add(row["dev_fold"])

    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Group crosses canonical folds")
    excluded = set()
    pairs = set()
    review_numbers = set()
    for link in exclusion["confirmed_cross_group_links"]:
        training, evaluation = link["train_group_id"], link["fold0_group_id"]
        if groups.get(evaluation) != {0} or groups.get(training) not in ({1}, {2}, {3}, {4}):
            raise ValueError("Confirmed link must join a Fold 0 group and a training group")
        pair = (training, evaluation)
        number = link["review_number"]
        if pair in pairs or number in review_numbers:
            raise ValueError("Repeated group link or review number")
        pairs.add(pair)
        review_numbers.add(number)
        excluded.add(evaluation)
    if not excluded:
        raise ValueError("At least one confirmed Fold 0 group required")
    return excluded, len(pairs), exclusion["review_status"]


def arm_metrics(gold, predicted, groups, severe_gold):
    result = evaluate(gold, predicted, groups)
    per_code = result["per_code"]
    result["fixed_8_common_macro_f1"] = mean(per_code[c]["f1"] or 0.0 for c in COMMON)
    result["prediction_only_active_codes"] = [c for c in CODES if not per_code[c]["support"] and per_code[c]["predicted"]]
    severe = {sid: severe_gold[sid] for sid in gold if severe_gold[sid]}
    result["severity4"] = {
        "photos": len(severe),
        "groups": len({groups[sid] for sid in severe}),
        "complete_misses": sum(not (codes & predicted[sid]) for sid, codes in severe.items()),
    }
    return result


def compare(manifest_path, exclusions_path, labels_path, baseline_path, candidate_path, hybrid_path=None):
    rows = fold_rows(manifest_path)
    expected = {row["sample_id"] for row in rows}
    all_groups = {row["group_id"] for row in rows}
    if len(all_groups) != 36:
        raise ValueError("The primary Fold 0 benchmark requires 36 original groups")
    inputs = {"official_baseline": baseline_path, "candidate": candidate_path}
    if hybrid_path is not None:
        inputs["hybrid"] = hybrid_path
    # Validate original, exact coverage for every arm before an exclusion can hide a missing row.
    predictions = {arm: prediction_map(path, expected, baseline=(arm == "official_baseline")) for arm, path in inputs.items()}
    exclusion, links, review_status = validate_mask(manifest_path, exclusions_path)
    primary_groups = {row["sample_id"]: row["group_id"] for row in rows}
    primary_gold = {row["sample_id"]: frozenset(row["codes"]) for row in rows}
    groups = {sid: group for sid, group in primary_groups.items() if group not in exclusion}
    gold = {sid: primary_gold[sid] for sid in groups}
    severe_gold = {}
    for row in read_selected_cases(labels_path, expected):
        target = row["target"]
        if primary_gold[row["sample_id"]] != set(target["hazards"]):
            raise ValueError("Canonical manifest label differs from official label")
        severe_gold[row["sample_id"]] = {
            code for code, item in target["hazards"].items() if item["risk_level"]["severity"]["score"] == 4
        }
    if severe_gold.keys() != primary_gold.keys():
        raise ValueError("Incomplete severity data for the primary cohort")
    group_sizes = Counter(primary_groups.values())
    largest_group = min(group_sizes, key=lambda group: (-group_sizes[group], group))
    largest_ids = {sid for sid, group in primary_groups.items() if group == largest_group}
    largest_subset_groups = {sid: primary_groups[sid] for sid in largest_ids}
    largest_gold = {sid: primary_gold[sid] for sid in largest_ids}
    kept = {arm: {sid: pred[sid] for sid in groups} for arm, pred in predictions.items()}
    comparisons = [("official_baseline", "candidate")]
    if hybrid_path is not None:
        comparisons.extend((("official_baseline", "hybrid"), ("candidate", "hybrid")))
    return {
        "scope": "Secondary sensitivity only. Same whole-group mask for every fully validated primary prediction file.",
        "canonical_manifest_sha256": sha256(manifest_path),
        "exclusions_sha256": sha256(exclusions_path),
        "official_labels_sha256": sha256(labels_path),
        "review_status": review_status,
        "prediction_sha256": {arm: sha256(path) for arm, path in inputs.items()},
        "primary_photos": len(rows), "primary_groups": len(all_groups),
        "confirmed_cross_group_links": links,
        "excluded_photos": len(rows) - len(gold), "excluded_groups": len(exclusion),
        "retained_photos": len(gold), "retained_groups": len(set(groups.values())),
        "largest_primary_group_diagnostic": {
            "photos": len(largest_ids), "groups": 1,
            "primary_photo_share": len(largest_ids) / len(rows),
            "other_photos": len(rows) - len(largest_ids), "other_groups": len(all_groups) - 1,
            "other_primary_photo_share": 1 - len(largest_ids) / len(rows),
            "scope": "Descriptive evaluation of one inferred group; no group confidence interval.",
            "metrics": {
                arm: arm_metrics(largest_gold, {sid: pred[sid] for sid in largest_ids}, largest_subset_groups, severe_gold)
                for arm, pred in predictions.items()
            },
        },
        "metrics": {arm: arm_metrics(gold, pred, groups, severe_gold) for arm, pred in kept.items()},
        "paired_group_bootstrap": {
            f"{a}_vs_{b}": paired_bootstrap(gold, kept[a], kept[b], groups, resamples=2000, seed=42)
            for a, b in comparisons
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--exclusions", type=Path, default=EXCLUSIONS)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.manifest, args.exclusions, args.labels, args.baseline, args.candidate, args.hybrid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
