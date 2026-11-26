"""Test a pre-specified joint false-alarm / severity-4 constraint in neural inner OOF."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .clean_development_audit import GEMINI, LABELS, MANIFEST, comparison, point, severity4, sha256, top3
from .clean_neural_calibration_audit import run as neural_inner_oof
from .clean_specialist import apply_calibrated_policy, fit_nested_calibration
from .metrics import CODES
from .train_head import save_atomic


PLAN = {
    "false_alarm_cap": 0.20,
    "severity4_limit": "at most the fixed 0.50 policy's severe4 complete misses on inner OOF",
    "code_thresholds": "fitted candidate plus 29 common values from .20 to .76; original rare-code suppression preserved",
    "scene_thresholds": "19 values from .28 to .64 plus the fitted value",
    "strong_override_margins": [0.16, 1.0],
    "ranking": "max active macro F1, then fewer severe4 misses, then fewer false alarms, then stable grid order",
    "infeasible": "keep fixed 0.50 for decision sensitivity; flag that no joint inner policy was found",
    "outer_evaluation": "all clean development rows; separate pass checks on each outer fold and whole pooled cohort",
    "epochs": 55,
    "seeds": [42, 1042, 2042],
}


def selection(code_scores, scene_scores, labels, groups, severe):
    fitted = fit_nested_calibration(code_scores, scene_scores, labels[:, :-1], groups)
    temps = np.array(fitted["temperatures"])
    clipped = code_scores.clip(1e-6, 1 - 1e-6)
    calibrated = 1 / (1 + np.exp(-np.log(clipped / (1 - clipped)) / temps))
    fitted_thresholds = np.array(fitted["thresholds"])
    fixed = top3(code_scores, np.full(len(CODES), 0.50))
    clean = ~labels[:, -1]
    severe_photos = severe.any(axis=1)
    severity_limit = int((severe_photos & ~(fixed & severe).any(axis=1)).sum())
    false_alarm_limit = int(clean.sum()) // 5
    code_grids = [fitted_thresholds]
    rare = np.array([code in fitted["suppressed_autonomous_codes"] for code in CODES])
    for threshold in np.linspace(0.20, 0.76, 29).round(4):
        code_grids.append(np.where(rare, fitted_thresholds, threshold))
    scene_grid = sorted(set(np.linspace(0.28, 0.64, 19).round(4)) | {fitted["scene_threshold"]})
    best = None
    stats = {
        "inner_photos": len(labels),
        "inner_no_visible": int(clean.sum()),
        "inner_severe4_photos": int(severe_photos.sum()),
        "inner_severe4_complete_miss_limit": severity_limit,
        "inner_no_visible_false_alarm_count_limit": false_alarm_limit,
        "inner_fixed05_no_visible_false_alarms": int(fixed[clean].any(axis=1).sum()),
        "attempted": 0, "false_alarm_feasible": 0, "jointly_feasible": 0,
        "min_severe4_misses_under_false_alarm_cap": None,
        "min_false_alarms_under_severe4_cap": None,
    }
    for thresholds in code_grids:
        eligible = top3(calibrated, thresholds)
        margins = (calibrated - thresholds).max(axis=1)
        for scene in scene_grid:
            for override in PLAN["strong_override_margins"]:
                gate = (scene_scores >= scene) | (margins >= override)
                prediction = eligible & gate[:, None]
                false_alarms = int(prediction[clean].any(axis=1).sum())
                severe_misses = int((severe_photos & ~(prediction & severe).any(axis=1)).sum())
                stats["attempted"] += 1
                if severe_misses <= severity_limit:
                    key = "min_false_alarms_under_severe4_cap"
                    stats[key] = min(stats[key], false_alarms) if stats[key] is not None else false_alarms
                if false_alarms > false_alarm_limit:
                    continue
                stats["false_alarm_feasible"] += 1
                key = "min_severe4_misses_under_false_alarm_cap"
                stats[key] = min(stats[key], severe_misses) if stats[key] is not None else severe_misses
                if severe_misses > severity_limit:
                    continue
                stats["jointly_feasible"] += 1
                f1 = point(labels[:, :-1], prediction)["macro_f1"]
                rank = (f1, -severe_misses, -false_alarms)
                if best is None or rank > best[0]:
                    policy = {
                        **fitted, "thresholds": thresholds.tolist(),
                        "thresholds_by_code": dict(zip(CODES, thresholds.tolist())),
                        "scene_threshold": float(scene), "strong_override_margin": override,
                    }
                    best = (rank, policy)
    if best is None:
        return {"search": stats, "policy": None, "selected_inner": None}
    rank, policy = best
    return {"search": stats, "policy": policy,
            "selected_inner": {"macro_f1": rank[0], "severe4_misses": -rank[1], "false_alarms": -rank[2]}}


def pass_checks(result):
    after = result["b"]
    severe = result["severity_4"]
    has_no_visible = after["no_visible"] > 0
    return {
        "false_alarm_cap_met": after["no_visible_false_alarms"] <= after["no_visible"] // 5 if has_no_visible else None,
        "severity4_nonincrease_met": severe["complete_miss_b"] <= severe["complete_miss_a"] if severe["photos"] else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/clean-neural-joint-constraint-audit.json"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("set one to four CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.root
    rows = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("expected development rows only")
    ids = [row["sample_id"] for row in rows]
    folds = np.array([row["dev_fold"] for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    labels = np.array([[code in row["codes"] for code in CODES] + [row["status"] == "hazard"] for row in rows])
    severe = severity4(root / LABELS, ids)
    with np.load(root / GEMINI, allow_pickle=False) as archive:
        if (archive["ids"].tolist() != ids or not np.array_equal(archive["folds"], folds)
                or not np.array_equal(archive["y_codes"], labels[:, :-1])):
            raise ValueError("probability archive disagrees with development manifest")
        code_scores, scene_scores = archive["code_probs"], archive["scene_probs"]
    fixed = top3(code_scores, np.full(len(CODES), 0.50))
    predicted = fixed.copy()
    evaluated = np.zeros(len(rows), dtype=bool)
    by_fold = {}
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        cache = root / f"data/clean-specialist-v1/strict-inner-oof/fold-{fold}.npz"
        inner_scores, inner_scene = neural_inner_oof(
            root, int(fold), PLAN["epochs"], PLAN["seeds"], cache=cache, inner_only=True,
        )
        selected = selection(inner_scores, inner_scene, labels[train], groups[train], severe[train])
        if selected["policy"] is not None:
            codes = apply_calibrated_policy(code_scores[test], scene_scores[test], selected["policy"])
            predicted[test] = np.array([[code in row for code in CODES] for row in codes])
            evaluated |= test
        result = comparison(labels, fixed, predicted, groups, test, severe)
        selected["outer_result"] = result
        selected["outer_pass_checks"] = pass_checks(result) if selected["policy"] is not None else None
        by_fold[str(int(fold))] = selected
        print(f"strict outer={fold} joint_inner_candidates={selected['search']['jointly_feasible']}", flush=True)
    full = comparison(labels, fixed, predicted, groups, np.ones(len(rows), bool), severe)
    feasible_only = comparison(labels, fixed, predicted, groups, evaluated, severe) if evaluated.any() else None
    result = {
        "plan": PLAN,
        "outer_probability_sha256": sha256(root / GEMINI),
        "by_fold": by_fold,
        "outer_photos_with_selected_joint_policy": int(evaluated.sum()),
        "pooled_eligible_folds_fixed05_vs_joint": feasible_only,
        "pooled_eligible_folds_pass_checks": pass_checks(feasible_only) if feasible_only else None,
        "pooled_all_folds_fixed05_vs_joint_with_fixed05_fallback": full,
        "pooled_all_folds_pass_checks": pass_checks(full),
    }
    output = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    save_atomic(args.output, lambda stream: stream.write(output))


if __name__ == "__main__":
    main()
