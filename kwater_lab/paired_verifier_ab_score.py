"""Score both completed A/B verifier arms against the same official Fold0 anchor."""

from __future__ import annotations

import json
from pathlib import Path
import random
from statistics import mean

from .audit import digest
from .fold0_group_sensitivity import EXCLUSIONS, MANIFEST, arm_metrics, validate_mask
from .fold0_production_compare import fold_rows
from .law_reranker_11code import read_selected_cases
from .metrics import check_text, paired_bootstrap, percentile, reject_constant, unique_keys
from .official_anchored_verifier import OFFICIAL, add_verified_records, check_official_output, official_records
from .paired_verifier_ab import LABELS, PLAN, PROBE, read, validate_completed_arm, verification_identity, write
from .train_head import save_atomic
from .vlm_11code import parse_verified_hazard


def decision(raw, code):
    if parse_verified_hazard(raw, code) is not None:
        return "accept"
    try:
        parsed = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant)
        if isinstance(parsed, dict) and parsed.keys() == {"decision", "reason"} and parsed["decision"] == "reject":
            value = check_text(parsed["reason"], "reason", 480)
            value.encode("utf-8")
            if any(char.isalnum() for char in value):
                return "reject"
    except (TypeError, ValueError, RecursionError):
        pass
    return "malformed"


def read_verifications(plan, workspace, arm, adapter):
    path = workspace / arm / "raw-verifier-local.json"
    cache = read(path)
    identity = verification_identity(plan, adapter)
    if (cache["fingerprint"]["batch_size"] != 1 or cache["fingerprint"]["candidates"] != plan["candidates"]
            or any(cache["fingerprint"]["inputs"].get(k) != v for k, v in identity.items())):
        raise ValueError("Raw verifier cache does not match this frozen probe and adapter")
    rows = cache["results"]
    keys = ("sample_id", "code", "image")
    if [tuple(r[key] for key in keys) for r in rows] != [tuple(r[key] for key in keys) for r in plan["candidates"]]:
        raise ValueError("Both verifier arms must supply every frozen photo and code in the same order")
    runtime = cache["fingerprint"]["inputs"]["evaluation_runtime"]
    if runtime.get("precision") != "BF16" or not cache["fingerprint"]["inputs"].get("device_model_key"):
        raise ValueError("Frozen BF16 runtime provenance is required for each A/B cache")
    decoded = {}
    for row in rows:
        raw = row["raw_verify_text"]
        outcome = decision(raw, row["code"])
        hazard = parse_verified_hazard(raw, row["code"]) if outcome == "accept" else None
        if (row["accepted"] is not (hazard is not None) or row["hazard"] != hazard):
            raise ValueError("Cached decisions disagree with the shared strict parser")
        decoded[row["sample_id"]] = {**row, "verdict": outcome}
    return decoded, {"cache_sha256": digest(path), "adapter_sha256": identity["adapter_sha256"], "runtime": runtime,
                     "verification_seconds": cache["verification_seconds"]}


def decision_ci(probes, verdict_a, verdict_b, gold, groups):
    cluster = {}
    for sid, code in probes.items():
        answer = "accept" if code in gold[sid] else "reject"
        cluster.setdefault(groups[sid], []).append(int(verdict_b[sid] == answer) - int(verdict_a[sid] == answer))
    members = [cluster[key] for key in sorted(cluster)]
    if len(members) < 2:
        return None
    rng = random.Random(42)
    draws = sorted(mean([item for group in rng.choices(members, k=len(members)) for item in group]) for _ in range(2000))
    return [percentile(draws, p) for p in (0.025, 0.975)]


def scored_arm(probes, verdict, predictions, gold, groups, severity):
    positives = sum(code in gold[sid] for sid, code in probes.items())
    negatives = len(probes) - positives
    stats = {"annotated_candidates": positives, "unannotated_candidates": negatives, "valid_accept": 0,
             "valid_reject": 0, "malformed": 0, "correct_accept": 0, "incorrect_accept": 0, "correct_reject": 0}
    for sid, code in probes.items():
        result = verdict[sid]
        in_gold = code in gold[sid]
        if result == "accept":
            stats["valid_accept"] += 1
            stats["correct_accept" if in_gold else "incorrect_accept"] += 1
        elif result == "reject":
            stats["valid_reject"] += 1
            stats["correct_reject"] += not in_gold
        else:
            stats["malformed"] += 1
    ratio = lambda num, den: num / den if den else None
    stats["decision_accuracy"] = ratio(stats["correct_accept"] + stats["correct_reject"], len(probes))
    stats["accepted_precision"] = ratio(stats["correct_accept"], stats["valid_accept"])
    stats["accept_recall"] = ratio(stats["correct_accept"], positives)
    stats["invalid_accept_rate_on_unannotated"] = ratio(stats["incorrect_accept"], negatives)
    stats["ambiguous_or_invalid_model_responses"] = stats["malformed"]
    stats["accepted_visually_ambiguous_ground_truth"] = None
    return {"verifier": stats, "official_anchored": augmented_metrics(gold, predictions, groups, severity)}


def focused_code_decisions(probes, verdicts, gold):
    treatments = {"fall_height", "struck_by_lifting"}
    results = {}
    for name, use_treatment in (("fall_height_and_struck_by_lifting", True), ("other_codes", False)):
        selected = {sid: code for sid, code in probes.items() if (code in treatments) == use_treatment}
        positives = {sid for sid, code in selected.items() if code in gold[sid]}
        by_arm = {}
        accepted = {}
        for arm, decisions in verdicts.items():
            accepted[arm] = {sid for sid in selected if decisions[sid] == "accept"}
            by_arm[arm] = {"accepted_tp": len(accepted[arm] & positives), "accepted_fp": len(accepted[arm] - positives),
                           "valid_rejects": sum(decisions[sid] == "reject" for sid in selected),
                           "malformed": sum(decisions[sid] == "malformed" for sid in selected)}
        a_only, b_only = accepted["A"] - accepted["B"], accepted["B"] - accepted["A"]
        a, b = verdicts["A"], verdicts["B"]
        results[name] = {
            "proposals": len(selected), "official_annotated_positives": len(positives), "arms": by_arm,
            "same_valid_decision": sum(a[sid] == b[sid] and a[sid] in {"accept", "reject"} for sid in selected),
            "both_malformed": sum(a[sid] == b[sid] == "malformed" for sid in selected),
            "acceptance_switches": {"annotated_A_only": len(a_only & positives), "annotated_B_only": len(b_only & positives),
                                    "unannotated_A_only": len(a_only - positives), "unannotated_B_only": len(b_only - positives)},
        }
    return results


def augmented_metrics(gold, predicted, groups, severity):
    result = arm_metrics(gold, predicted, groups, severity)
    result["severity4"]["code_instances"] = sum(len(severity[sid]) for sid in gold)
    result["severity4"]["code_instances_missed"] = sum(len(severity[sid] - predicted[sid]) for sid in gold)
    return result


def write_checked_predictions(workspace, arm, baseline, result):
    source = OFFICIAL / "output/submission.jsonl"
    original = {json.loads(line)["image_id"]: line for line in source.read_bytes().splitlines(keepends=True)}
    output = workspace / arm / "official-anchored" / "output/submission.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    unchanged = 0
    serialized = []
    for sid, row in result.items():
        if row == baseline[sid]:
            unchanged += 1
            serialized.append(original[row["image_id"]])
        else:
            serialized.append((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    save_atomic(output, lambda stream: stream.write(b"".join(serialized)))
    check_official_output(output, output.parent.parent / "checker.log")
    return {"sha256": digest(output), "official_checker_valid_rows": len(result), "unchanged_rows_verbatim": unchanged,
            "official_hazard_items_preserved": sum(len(row["hazards"]) for row in baseline.values())}


def score(plan, workspace, output):
    checkpoints = {arm: validate_completed_arm(plan, workspace, arm) for arm in ("A", "B")}
    if checkpoints["A"][1] != checkpoints["B"][1]:
        raise ValueError("Actual optimizer learning-rate grids differ across arms")
    if len(plan["candidates"]) != 102 or plan["candidates"] != read(PROBE)["proposals"]:
        raise ValueError("Expected the original frozen 102 proposals")
    decoded, witnesses = {}, {}
    for arm, (adapter, _) in checkpoints.items():
        decoded[arm], witnesses[arm] = read_verifications(plan, workspace, arm, adapter)
    if witnesses["A"]["runtime"] != witnesses["B"]["runtime"]:
        raise ValueError("A/B verifier hardware or software differs")
    verdicts = {arm: {sid: row["verdict"] for sid, row in values.items()} for arm, values in decoded.items()}
    rows = fold_rows(MANIFEST)
    records = official_records(rows)
    excluded, _, _ = validate_mask(MANIFEST, EXCLUSIONS)
    official = {sid: frozenset(r["hazards"]) for sid, r in records.items()}
    proposals = {r["sample_id"]: r["code"] for r in plan["candidates"]}
    if any(code in official[sid] or len(official[sid]) >= 3 for sid, code in proposals.items()):
        raise ValueError("Frozen probe would overwrite an existing hazard")
    gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    severity = {r["sample_id"]: {c for c, h in r["target"]["hazards"].items() if h["risk_level"]["severity"]["score"] == 4}
                for r in read_selected_cases(LABELS, set(gold))}
    if severity.keys() != gold.keys():
        raise ValueError("The official severity labels do not cover the primary cohort")
    law_ids = plan["common_law_train_ids"]
    if not law_ids or law_ids & gold.keys():
        raise ValueError("Training case bank overlaps the evaluation cohort")
    for arm, values in decoded.items():
        records_for_arm = add_verified_records(records, list(values.values()), law_train_ids=law_ids)
        witnesses[arm]["output"] = write_checked_predictions(workspace, arm, records, records_for_arm)
        decoded[arm] = {sid: frozenset(record["hazards"]) for sid, record in records_for_arm.items()}
    results = {}
    for name, selected in (("primary_frozen_fold0", rows),
                           ("confirmed_cross_group_exclusion", [r for r in rows if r["group_id"] not in excluded])):
        groups = {r["sample_id"]: r["group_id"] for r in selected}
        actual = {sid: gold[sid] for sid in groups}
        before = {sid: official[sid] for sid in groups}
        probes = {sid: proposals[sid] for sid in groups if sid in proposals}
        preds, stats = {}, {}
        for arm in ("A", "B"):
            preds[arm] = {sid: decoded[arm][sid] for sid in groups}
            stats[arm] = scored_arm(probes, verdicts[arm], preds[arm], actual, groups, severity)
        ci = decision_ci(probes, verdicts["A"], verdicts["B"], actual, groups)
        a, b = (stats[k]["verifier"] for k in ("A", "B"))
        delta = b["decision_accuracy"] - a["decision_accuracy"] if probes else None
        results[name] = {"photos": len(groups), "groups": len(set(groups.values())), "frozen_probes": len(probes),
                         "official_base": augmented_metrics(actual, before, groups, severity), "arms": stats,
                         "supplemental_by_treatment_code": focused_code_decisions(probes, verdicts, actual),
                         "paired_decision_accuracy_delta": delta, "paired_decision_accuracy_group_ci": ci,
                         "decision_support_groups": len({groups[sid] for sid in probes}),
                         "paired_anchored_macro_f1": paired_bootstrap(actual, preds["A"], preds["B"], groups),
                         "paired_official_vs_A_macro_f1": paired_bootstrap(actual, before, preds["A"], groups),
                         "paired_official_vs_B_macro_f1": paired_bootstrap(actual, before, preds["B"], groups)}
    primary = results["primary_frozen_fold0"]
    a, b = (primary["arms"][k] for k in ("A", "B"))
    ci = primary["paired_decision_accuracy_group_ci"]
    fpr = b["official_anchored"]["no_hazard_false_alarm_rate"]
    checks = {
        "decision_accuracy_improvement": primary["paired_decision_accuracy_delta"] >= PLAN["primary_min_accuracy_delta"],
        "positive_decision_group_interval": bool(ci and ci[0] > 0),
        "incorrect_accepts": b["verifier"]["incorrect_accept"] - a["verifier"]["incorrect_accept"] <= PLAN["max_incorrect_accept_increase"],
        "correct_accepts": a["verifier"]["correct_accept"] - b["verifier"]["correct_accept"] <= PLAN["max_correct_accept_decrease"],
        "official_anchored_false_alarm_rate": fpr is not None and fpr <= PLAN["max_anchored_false_alarm_rate"],
        "severity4_complete_misses": (b["official_anchored"]["severity4"]["complete_misses"]
                                      - a["official_anchored"]["severity4"]["complete_misses"] <= PLAN["max_severity4_miss_increase"]),
    }
    receipt = {"scope": "Same frozen 102-proposal conditional verifier experiment; no new primary generation or proposal mining",
               "visual_ground_truth": "Official hazard-code annotations. No independent ambiguity labels for the 102 proposals; malformed model replies are unresolved and not counted as correct rejects.",
               "scorer_sha256": digest(Path(__file__)),
               "official_output_sha256": digest(OFFICIAL / "output/submission.jsonl"),
               "shared_new_law_case_source_photos": len(law_ids), "arm_witnesses": witnesses,
               "pilot_only": plan["composition"]["pilot_only"],
               "promotion_judgment": "withheld_exploratory_small_treatment_sample" if plan["composition"]["pilot_only"] else "independent_confirmation_required",
               "identity": plan["identity"], "pre_registered_decision_rule": PLAN, "cohorts": results,
               "primary_rule_checks": checks, "primary_meets_rule": all(checks.values())}
    write(output, receipt)
