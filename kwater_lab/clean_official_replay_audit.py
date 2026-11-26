"""Development replay of one out-of-fold specialist code added to official output."""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from .clean_development_audit import GEMINI, LABELS, MANIFEST, PROTOCOL, comparison, severity4
from .fold0_production_compare import image_id
from .metrics import CODES, decode_target, read_index


REPLAY = Path("data/phase2-runs/p38-20261126-49bb82ce/release-replay")
COHORTS = ("prototype18", "fresh64")
THRESHOLD = 0.5


def cohort_predictions(directory, allowed):
    paths = sorted(directory.glob("*.json"))
    predictions = {}
    for path in paths:
        sid = unquote(path.stem)
        if sid not in allowed:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["sample_id"] != sid or sid in predictions:
            raise ValueError("replay has a sample mismatch")
        _, codes = decode_target(raw["outputs"]["pristine"])
        predictions[sid] = codes
    return predictions, len(paths)


def score_production_fold0(baseline_path, candidate_path, ids, folds, probs, labels, severe, groups):
    fold0 = folds == 0
    folded = [sid for sid, keep in zip(ids, fold0) if keep]
    official_ids = {image_id(sid): sid for sid in folded}
    official = read_index(baseline_path, decode_target, limit_size=True)
    if official.keys() != official_ids.keys():
        raise ValueError(f"baseline must cover all {len(folded)} fold-0 photos with anonymous IDs")
    baseline = {official_ids[key]: value for key, value in official.items()}
    result = {"neural_fixed05_proposal_replay": summarize(baseline, ids, probs, labels, severe, groups)}
    if candidate_path is not None:
        predictions = read_index(candidate_path, lambda row: decode_target(row, local_ids=True), limit_size=True)
        if predictions.keys() == official_ids.keys():
            predictions = {official_ids[key]: value for key, value in predictions.items()}
        if predictions.keys() != baseline.keys():
            raise ValueError(f"candidate must cover all {len(folded)} fold-0 photos")
        base_codes = np.array([[code in baseline.get(sid, ()) for code in CODES] for sid in ids])
        candidate_codes = np.array([[code in predictions.get(sid, ()) for code in CODES] for sid in ids])
        result["baseline_vs_frozen_candidate"] = comparison(labels, base_codes, candidate_codes, groups, fold0, severe)
    return result


def run(root, baseline_path=None, candidate_path=None):
    rows = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("expected development manifest")
    ids = [row["sample_id"] for row in rows]
    groups = np.array([row["group_id"] for row in rows])
    folds = np.array([row["dev_fold"] for row in rows])
    labels = np.array([[code in row["codes"] for code in CODES] + [row["status"] == "hazard"] for row in rows])
    ids_set = set(ids)
    severe = severity4(root / LABELS, ids)
    with np.load(root / GEMINI, allow_pickle=False) as archive:
        if (archive["ids"].tolist() != ids or not np.array_equal(archive["y_codes"], labels[:, :-1])
                or not np.array_equal(archive["folds"], folds)):
            raise ValueError("probability archive disagrees with development manifest")
        probs = archive["code_probs"]
    if baseline_path is not None:
        return score_production_fold0(baseline_path, candidate_path, ids, folds, probs, labels, severe, groups)
    by_cohort = {}
    candidate_counts = {}
    for cohort in COHORTS:
        by_cohort[cohort], candidate_counts[cohort] = cohort_predictions(root / REPLAY / cohort, ids_set)
    combined = by_cohort[COHORTS[0]].copy()
    overlap = set(combined) & set(by_cohort[COHORTS[1]])
    if any(combined[sid] != by_cohort[COHORTS[1]][sid] for sid in overlap):
        raise ValueError("official baseline differs on cohort-overlapping rows")
    combined.update(by_cohort[COHORTS[1]])
    group_lookup = dict(zip(ids, groups))
    cohort_group_overlap = set(group_lookup[sid] for sid in by_cohort[COHORTS[0]]) & {
        group_lookup[sid] for sid in by_cohort[COHORTS[1]]
    }
    return {
        "policy": "Add at most one original outer-OOF specialist code with probability >=.50. Keep every baseline code and fill at most three slots.",
        "selected_replay_rows": candidate_counts,
        "outside_development_excluded": {key: candidate_counts[key] - len(value) for key, value in by_cohort.items()},
        "cohort_photo_overlap_within_development": len(overlap),
        "cohort_group_overlap_within_development": len(cohort_group_overlap),
        "results": {key: summarize(data, ids, probs, labels, severe, groups)
                    for key, data in {**by_cohort, "union_unique": combined}.items()},
    }


def summarize(data, ids, probs, labels, severe, groups):
    mask = np.array([sid in data for sid in ids])
    if not mask.any():
        return None
    gold = labels[:, :-1]
    baseline = np.array([[code in data.get(sid, ()) for code in CODES] for sid in ids])
    extra = np.zeros_like(baseline)
    for i in np.flatnonzero(mask & (baseline.sum(axis=1) < 3)):
        choices = np.where(baseline[i], -np.inf, probs[i])
        best = int(np.argmax(choices))
        if choices[best] >= THRESHOLD:
            extra[i, best] = True
    oracle = extra & gold
    new_negative = extra.any(axis=1) & ~labels[:, -1] & ~baseline.any(axis=1)
    negative = mask & ~labels[:, -1]
    n_new = int(extra.sum())
    correct = int(oracle.sum())
    true_positive = int((baseline & gold)[mask].sum())
    total_gold_and_pred = int(gold[mask].sum() + baseline[mask].sum())
    gross_remaining = int(gold[mask].sum()) - true_positive
    baseline_false_alarms = int(baseline[negative].any(axis=1).sum())
    by_code = {}
    for j, code in enumerate(CODES):
        if not extra[:, j].any():
            continue
        denominator = int(baseline[mask, j].sum() + gold[mask, j].sum())
        tp = int((baseline[mask, j] & gold[mask, j]).sum())
        by_code[code] = {
            "support": int(gold[mask, j].sum()), "baseline_predicted": int(baseline[mask, j].sum()),
            "proposed": int(extra[:, j].sum()), "correct": int(oracle[:, j].sum()),
            "strict_code_f1_improvement_accept_precision_gt": tp / denominator if denominator else None,
        }
    group_keys = sorted(set(groups[mask]))
    group_proposed = np.array([extra[groups == group].sum() for group in group_keys])
    group_correct = np.array([oracle[groups == group].sum() for group in group_keys])
    n_groups = len(group_keys)
    draws = np.random.default_rng(PROTOCOL["bootstrap_seed"]).multinomial(
        n_groups, np.full(n_groups, 1 / n_groups), size=PROTOCOL["bootstrap_resamples"]
    )
    proposed_draw = draws @ group_proposed
    valid_draws = proposed_draw > 0
    correct_draw = draws[valid_draws] @ group_correct
    precision_interval = (np.quantile(correct_draw / proposed_draw[valid_draws], [0.025, 0.975]).tolist()
                          if valid_draws.any() else None)
    return {
        "baseline_vs_automatic": comparison(labels, baseline, baseline | extra, groups, mask, severe),
        "baseline_vs_oracle": comparison(labels, baseline, baseline | oracle, groups, mask, severe),
        "baseline_correct_hazards": true_positive,
        "baseline_missed_hazards": gross_remaining,
        "proposed": n_new, "recoverable_missed_hazards": correct,
        "incorrect_additions": n_new - correct,
        "incorrect_additions_on_newly_false_alarming_photos": int(new_negative.sum()),
        "candidate_accept_precision_if_all_accepted": correct / n_new if n_new else None,
        "candidate_accept_precision_group_bootstrap_95_percent": precision_interval,
        "strict_micro_f1_improvement_accept_precision_gt": true_positive / total_gold_and_pred if total_gold_and_pred else None,
        "new_false_alarm_budget_under_twenty_percent": math.floor(0.20 * negative.sum()) - baseline_false_alarms,
        "by_code": by_code,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--production-fold0-baseline", type=Path)
    parser.add_argument("--production-fold0-candidate", type=Path)
    args = parser.parse_args()
    if args.production_fold0_candidate and not args.production_fold0_baseline:
        parser.error("a production candidate requires the paired official baseline")
    if args.production_fold0_baseline and args.output is None:
        parser.error("choose an output path for the production fold-0 comparison")
    output = args.output or Path("results/clean-official-replay-specialist-audit.json")
    result = run(args.root, args.production_fold0_baseline, args.production_fold0_candidate)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
