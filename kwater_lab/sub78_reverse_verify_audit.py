"""Estimate a fixed singleton reverse-verification gate's observable Fold 4 limits."""
from collections import Counter
from pathlib import Path

from .fold4_code_diagnosis import digest, fold4, project_prediction
from .fold4_same_verifier import read, save
from .metrics import CODES, evaluate, read_index

ARCHIVE = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
POLICY = Path("data/clean-specialist-v1/calibration-audit-folds/fold-4.json")
DRAFT = Path("data/vlm-scale-seoul-8b-fold4-20261128/eval1/predictions.jsonl")
COMPARISON = Path("results/paired-qwen-4b-8b-fold4-20261128.json")
OUTPUT = Path("results/sub78-reverse-verify-candidate-limits-20261128.json")


def candidate_stats(groups, gold, severe, draft, eligible):
    singleton = {sid for sid in groups if len(draft[sid]) == 1}
    filtered = singleton & eligible
    baseline = evaluate(gold, draft, groups)
    baseline_severe_misses = sum(bool(severe[sid]) and not (draft[sid] & severe[sid]) for sid in groups)
    false_safe = {sid for sid in filtered if not gold[sid]}
    wrong_code = {sid for sid in filtered if gold[sid] and not (draft[sid] & gold[sid])}
    correct_code = {sid for sid in filtered if draft[sid] & gold[sid]}
    severe_hit = {sid for sid in correct_code if draft[sid] & severe[sid]}
    normal_by_group = Counter(groups[sid] for sid in groups if not gold[sid])
    false_alarms_by_group = Counter(groups[sid] for sid in groups if not gold[sid] and draft[sid])

    def tally(ids):
        return {"photos": len(ids), "groups": len({groups[sid] for sid in ids})}

    def remove(ids):
        pruned = {sid: set() if sid in ids else draft[sid] for sid in groups}
        measure = evaluate(gold, pruned, groups)
        return {"macro_f1": measure["macro_f1"], "macro_f1_delta": measure["macro_f1"] - baseline["macro_f1"],
                "normal_false_alarms": measure["no_hazard_false_alarms"],
                "severity4_completely_missed": sum(bool(severe[sid]) and not (pruned[sid] & severe[sid]) for sid in groups)}

    return {
        "photos": len(groups), "groups": len(set(groups.values())),
        "baseline_macro_f1": baseline["macro_f1"],
        "normal_photos": baseline["no_hazard_photos"],
        "normal_photo_groups": len(normal_by_group),
        "normal_false_alarms": baseline["no_hazard_false_alarms"],
        "normal_false_alarm_groups": len(false_alarms_by_group),
        "normal_false_alarm_group_sizes": [
            {"normal_photos": normal, "false_alarms": fp}
            for fp, normal in sorted(((fp, normal_by_group[g]) for g, fp in false_alarms_by_group.items()), reverse=True)
        ],
        "normal_false_alarms_with_singleton_draft": tally({sid for sid in singleton if not gold[sid]}),
        "severity4_photos": sum(bool(severe[sid]) for sid in groups),
        "severity4_groups": len({groups[sid] for sid in groups if severe[sid]}),
        "baseline_severity4_complete_misses": baseline_severe_misses,
        "eligible_singleton_drafts": tally(filtered),
        "eligible_false_alarms_in_normal_photos": tally(false_safe),
        "eligible_wrong_draft_code_in_hazard_photos": tally(wrong_code),
        "eligible_correct_singleton_draft_code_at_risk": tally(correct_code),
        "eligible_severity4_correct_draft_code_at_risk": tally(severe_hit),
        "oracle_prune_only_false_codes": remove(false_safe | wrong_code),
        "counterfactual_prune_all_eligible": remove(filtered),
    }


def main():
    import numpy as np

    panels, gold, severity = fold4()
    policy = read(POLICY)
    sha = {p.name: digest(p) for p in (ARCHIVE, POLICY, DRAFT, COMPARISON)}
    if policy["outer_fold"] != 4 or policy["outer_probability_sha256"] != sha[ARCHIVE.name]:
        raise ValueError("Specialist scores are not the frozen outer Fold 4 archive")
    if read(COMPARISON)["prediction_sha256"]["8b_epoch1"] != sha[DRAFT.name]:
        raise ValueError("Draft does not match the exact 8B epoch-1 paired evaluation")
    draft = read_index(DRAFT, project_prediction)
    if draft.keys() != gold.keys():
        raise ValueError("8B predictions do not cover the exact Fold 4 cohort")
    with np.load(ARCHIVE, allow_pickle=False) as source:
        ids = source["ids"].tolist()
        codes, scenes = source["code_probs"], source["scene_probs"]
        folds = source["folds"]
    if (len(set(ids)) != len(ids) or set(sid for sid, fold in zip(ids, folds, strict=True) if fold == 4) != gold.keys()
            or codes.shape != (len(ids), len(CODES)) or scenes.shape != (len(ids),)
            or not np.isfinite(codes).all() or not np.isfinite(scenes).all()):
        raise ValueError("Specialist rows or scores do not align")
    calibrated = policy["correct_inner_oof_policy"]
    threshold = calibrated["scene_threshold"]
    score_index = {sid: i for i, sid in enumerate(ids)}
    low_scene = {sid for sid in draft if scenes[score_index[sid]] < threshold}
    low_both = set()
    for sid in low_scene:
        if len(draft[sid]) != 1:
            continue
        c = CODES.index(next(iter(draft[sid])))
        p = float(np.clip(codes[score_index[sid], c], 1e-6, 1 - 1e-6))
        tempered = 1 / (1 + np.exp(-np.log(p / (1 - p)) / calibrated["temperatures"][c]))
        if tempered < calibrated["thresholds"][c]:
            low_both.add(sid)
    views = {name: {condition: candidate_stats(groups, {sid: gold[sid] for sid in groups}, severity,
                                               {sid: draft[sid] for sid in groups}, eligible)
                    for condition, eligible in (("all_singletons_oracle", set(draft)),
                                                ("existing_inner_scene_threshold", low_scene),
                                                ("existing_inner_scene_and_code_thresholds", low_both))}
             for name, groups in panels.items()}
    save(OUTPUT, {"scope": "CPU candidate support and oracle bounds. No actual 8B reverse-verifier decisions or runtime.",
                  "candidate_rules": {"both_rules_use_singleton_8b_draft_only": True, "inner_scene_strict_below": threshold,
                                      "inner_code_strict_below_by_code": calibrated["thresholds_by_code"],
                                      "inner_temperatures": calibrated["temperatures"],
                                      "threshold_note": "Values came from the existing inner OOF standalone specialist policy; they were not optimized for deletion or against the outer Fold 4 labels."},
                  "input_sha256": sha, "panels": views})


if __name__ == "__main__":
    main()
