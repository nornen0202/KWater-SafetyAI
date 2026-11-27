"""Freeze the actual Fold 4 specialist proposals for both v3 checkpoints without importing PyTorch."""
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .fold4_crosssplit_audit import DEV, LOCAL, V3_EVAL
from .metrics import CODES, decode_target, read_index
from .train_head import save_atomic

SPLIT = Path("data/clean-split-v1/split_manifest_full_sealed.json")
INNER = Path("data/clean-specialist-v1/strict-inner-oof/fold-4.npz")
OUTER = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
CALIBRATION = Path("results/clean-neural-calibration-pooled.json")
MARGIN_GATE = Path("data/v3-11code-fold4/margin_gate_calibration.json")
ORIGINAL_REPORT = Path("results/fold4_v3_clean_benchmark.json")
FROZEN = V3_EVAL.parent / "fold4-same-verifier/proposals.json"
AGGREGATE = Path("results/fold4-v3-frozen-verifier-proposals.json")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def extract(path, names, context):
    module = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError(f"Missing one or more required source functions in {path}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), context)


def freeze():
    hybrid_source = Path("kwater_lab/hybrid_pipeline.py")
    specialist_source = Path("kwater_lab/clean_specialist.py")
    scope = {"np": np, "CODES": CODES, "NUM_CODES": len(CODES), "Any": Any}
    extract(specialist_source, {"apply_calibrated_policy"}, scope)
    extract(hybrid_source, {"compute_calib_code_precisions", "usable_draft_hazards", "select_booster_candidates"}, scope)
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    ids = [r["sample_id"] for r in rows]
    if len(ids) != 175 or len(set(ids)) != 175 or len({r["group_id"] for r in rows}) != 39:
        raise ValueError("Unexpected evaluation cohort")
    train = {r["sample_id"]: r for r in load(SPLIT) if r["split"] == "dev" and r["dev_fold"] != 4}
    calib = load(CALIBRATION)["calibration_by_fold"]["4"]
    with np.load(INNER, allow_pickle=False) as inner, np.load(OUTER, allow_pickle=False) as outer:
        train_order = inner["train_ids"].tolist()
        if len(train_order) != 791 or set(train_order) != train.keys():
            raise ValueError("Unexpected strict inner OOF scope")
        train_gold = np.asarray([[code in train[sid]["codes"] for code in CODES] for sid in train_order], dtype=np.float32)
        code_precision = scope["compute_calib_code_precisions"](inner["code_scores"], inner["scene_scores"], train_gold, calib)
        positions = {sid: i for i, sid in enumerate(outer["ids"].tolist())}
        indices = [positions[sid] for sid in ids]
        code_probs, scene_probs = outer["code_probs"][indices], outer["scene_probs"][indices]
    proposals = {}
    inputs = [DEV, SPLIT, INNER, OUTER, CALIBRATION, MARGIN_GATE, ORIGINAL_REPORT, hybrid_source, specialist_source]
    for epoch in ("epoch1", "epoch2"):
        raw_path = V3_EVAL / f"v3_{epoch}_raw_175.jsonl"
        cache_path = V3_EVAL / f"v3_{epoch}_infer_cache.json"
        cache = load(cache_path)
        outputs = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
        if [row["sample_id"] for row in outputs] != ids or list(cache["outputs"]) != ids:
            raise ValueError(f"{epoch}: raw order differs from canonical IDs")
        if any(set(row) != set(cache["outputs"][sid]) | {"image"}
               or any(cache["outputs"][sid][k] != v for k, v in row.items() if k != "image")
               or row["image"] != r["image"]
               for row, r, sid in zip(outputs, rows, ids, strict=True)):
            raise ValueError(f"{epoch}: raw rows differ from original cache or manifest")
        if cache["identity"]["samples"] != [[r["sample_id"], r["image"]] for r in rows]:
            raise ValueError(f"{epoch}: original cache ordering differs from manifest")
        inputs.extend([raw_path, cache_path])
        proposals[epoch] = scope["select_booster_candidates"](
            rows, outputs, code_probs, scene_probs, calib, code_precisions=code_precision,
            margin_boost_when_hazard=0.04, margin_boost_when_empty=0.12, max_candidates_per_photo=1,
        )
    e2_report = load(ORIGINAL_REPORT)["arms"]["v3_hybrid_1token_margin_verified"]
    if len(proposals["epoch2"]) != e2_report["raw_specialist_proposals"]:
        raise ValueError("Did not reproduce the actual regular epoch-2 proposal count")
    stand = V3_EVAL / "v3_11code_standalone.jsonl"
    hybrid = V3_EVAL / "v3_hybrid_margin_verified.jsonl"
    def decode(path):
        result = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
        if result.keys() != set(ids):
            raise ValueError(f"Unexpected exact coverage in {path}")
        return result
    before, after = decode(stand), decode(hybrid)
    inputs.extend([stand, hybrid])
    if any(not before[sid] <= after[sid] for sid in ids):
        raise ValueError("Existing hybrid removed published codes")
    final_additions = {(sid, code) for sid in ids for code in after[sid] - before[sid]}
    keyed = {epoch: {(c["sample_id"], c["code"]): c for c in items} for epoch, items in proposals.items()}
    if len(final_additions) != e2_report["final_accepted_additions"] or not final_additions <= keyed["epoch2"].keys():
        raise ValueError("Actual additions are inconsistent with reproduced 2ep proposals")
    union = [keyed["epoch2"].get(k, keyed["epoch1"].get(k))
             for k in sorted(keyed["epoch1"].keys() | keyed["epoch2"].keys(), key=lambda k: (ids.index(k[0]), k[1]))]
    common = keyed["epoch1"].keys() & keyed["epoch2"].keys()
    if any(keyed["epoch1"][k]["spec_cal_prob"] != keyed["epoch2"][k]["spec_cal_prob"] for k in common):
        raise ValueError("Specialist probability changed on a common candidate")
    hashes = {str(path.as_posix()): digest(path) for path in inputs}
    payload = {"input_sha256": hashes, "fixed_epoch2_gate": load(MARGIN_GATE), **proposals,
               "union_for_scoring_image_and_code_only": union}
    save_atomic(FROZEN, lambda f: f.write((json.dumps(payload, indent=2) + "\n").encode()))
    group = {r["sample_id"]: r["group_id"] for r in rows}
    gold = {r["sample_id"]: set(r["codes"]) for r in rows}
    mask_path = LOCAL / "fold4-group-exclusions-local.json"
    if load(mask_path)["canonical_manifest_sha256"] != digest(DEV):
        raise ValueError("Sensitivity mask differs from the frozen development manifest")
    excluded = {r["fold4_group_id"] for r in load(mask_path)["confirmed_cross_group_links"]}
    panels = {}
    retained = {sid for sid in ids if group[sid] not in excluded}
    sizes = Counter(group[sid] for sid in retained)
    largest = min(sizes, key=lambda g: (-sizes[g], g))
    for name, cohort in (("primary_175", set(ids)), ("sensitivity_107", retained),
                         ("exploratory_58", {sid for sid in retained if group[sid] != largest})):
        epoch1, epoch2 = ({k for k in keyed[e] if k[0] in cohort} for e in ("epoch1", "epoch2"))
        shared = epoch1 & epoch2
        panel = {"photos": len(cohort), "groups": len({group[sid] for sid in cohort}),
                 "shared_image_code": len(shared),
                 "shared_candidate_empty_context_changed": sum(
                     keyed["epoch1"][k]["was_vlm_empty"] != keyed["epoch2"][k]["was_vlm_empty"] for k in shared
                 ),
                 "only_epoch1": len(epoch1 - epoch2), "only_epoch2": len(epoch2 - epoch1),
                 "shared_epoch2_final_additions": len(final_additions & shared)}
        for epoch, subset in (("epoch1", epoch1), ("epoch2", epoch2)):
            panel[epoch] = {"candidates": len(subset), "gold_code_present": sum(code in gold[sid] for sid, code in subset),
                            "gold_code_absent": sum(code not in gold[sid] for sid, code in subset),
                            "gold_no_visible_hazard": sum(not gold[sid] for sid, _ in subset)}
        panels[name] = panel
    report = {
        "scope": "Proposal identity replay only. Actual epoch2 final additions checked; all other margin and verifier outcomes unobserved by this CPU replay.",
        "input_sha256": hashes | {mask_path.as_posix(): digest(mask_path)},
        "frozen_candidate_sha256": digest(FROZEN),
        "epoch2_reported_proposals": e2_report["raw_specialist_proposals"],
        "epoch2_observed_final_additions": len(final_additions),
        "union_candidates": len(union), "panels": panels,
    }
    save_atomic(AGGREGATE, lambda f: f.write((json.dumps(report, indent=2) + "\n").encode()))
    return report


def conditional_bounds():
    """Enumerate unknown 1ep additions under deterministic fixed 2ep verifier outcomes."""
    from itertools import product

    from .fold0_group_sensitivity import arm_metrics
    from .law_reranker_11code import read_selected_cases

    proposed = load(FROZEN)
    gate = proposed["fixed_epoch2_gate"]
    frozen_hash = digest(FROZEN)
    if frozen_hash != load(AGGREGATE)["frozen_candidate_sha256"]:
        raise ValueError("Frozen proposals differ from the exact replay")
    mask_path = LOCAL / "fold4-group-exclusions-local.json"
    mask = load(mask_path)
    if mask["canonical_manifest_sha256"] != digest(DEV):
        raise ValueError("Sensitivity mask differs from the fixed development manifest")
    exclusions = {r["fold4_group_id"] for r in mask["confirmed_cross_group_links"]}
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    all_gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    groups = {r["sample_id"]: r["group_id"] for r in rows}
    ids = set(all_gold)
    labels = Path("data/train/train.jsonl")
    loaded_gold = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(labels, ids)}
    if loaded_gold.keys() != ids or any(set(loaded_gold[sid]) != all_gold[sid] for sid in ids):
        raise ValueError("Exact Fold 4 labels are required")
    severe = {sid: {code for code, row in codes.items() if row["risk_level"]["severity"]["score"] == 4}
              for sid, codes in loaded_gold.items()}
    relative = {"baseline8": "clean_8code_baseline.jsonl", "epoch1": "v3_epoch1_11code_standalone.jsonl",
                "epoch2": "v3_11code_standalone.jsonl", "epoch2_hybrid": "v3_hybrid_margin_verified.jsonl"}
    paths = {name: V3_EVAL / value for name, value in relative.items()}
    predictions = {name: read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
                   for name, path in paths.items()}
    if any(rows.keys() != ids for rows in predictions.values()):
        raise ValueError("Every source must cover exactly the 175 evaluation photos")
    e2_final = {(sid, code) for sid in ids for code in predictions["epoch2_hybrid"][sid] - predictions["epoch2"][sid]}
    e1, e2 = ({(c["sample_id"], c["code"]): c for c in proposed[epoch]} for epoch in ("epoch1", "epoch2"))
    if not e2_final <= e2.keys():
        raise ValueError("Epoch-2 final output differs from its reproduced proposal context")
    fixed_accept, fixed_reject, unknown = set(), set(), set()
    for key, candidate in e1.items():
        if key not in e2:
            unknown.add(key)
            continue
        threshold = gate["tau_when_empty" if candidate["was_vlm_empty"] else "tau_when_hazard"]
        prior_threshold = gate["tau_when_empty" if e2[key]["was_vlm_empty"] else "tau_when_hazard"]
        if key in e2_final and threshold <= prior_threshold:
            fixed_accept.add(key)
        elif key not in e2_final and threshold >= prior_threshold:
            fixed_reject.add(key)
        else:
            unknown.add(key)
    if any(code in predictions["epoch1"][sid] for sid, code in e1):
        raise ValueError("Proposal must introduce at most one previously absent code per photo")
    if len(unknown) > 16:
        raise ValueError("Too many uncertain additions for this small exact enumeration")
    retained = {sid: g for sid, g in groups.items() if g not in exclusions}
    size = Counter(retained.values())
    largest = min(size, key=lambda g: (-size[g], g))
    cohorts = {"primary_175": groups, "sensitivity_107": retained,
               "exploratory_58": {sid: g for sid, g in retained.items() if g != largest}}
    panels = {}
    for name, cohort in cohorts.items():
        gold = {sid: all_gold[sid] for sid in cohort}
        base = arm_metrics(gold, {sid: predictions["baseline8"][sid] for sid in cohort}, cohort, severe)
        kinds = {"must_accept": fixed_accept, "must_reject": fixed_reject, "unknown": unknown}
        panels[name] = {
            "photos": len(cohort), "groups": len(set(cohort.values())),
            "candidate_partitions": {label: sum(sid in cohort for sid, _ in items) for label, items in kinds.items()},
            "baseline8_f1": base["macro_f1"], "normal_photos": base["no_hazard_photos"],
            "severity4_photos": base["severity4"]["photos"], "baseline8_severity4_misses": base["severity4"]["complete_misses"],
        }
    checks = {name: Counter() for name in cohorts}
    observations = {name: [] for name in cohorts}
    joint = 0
    unknown_order = sorted(unknown)
    for bits in product((False, True), repeat=len(unknown_order)):
        accepted = fixed_accept | {key for key, flag in zip(unknown_order, bits, strict=True) if flag}
        pred = dict(predictions["epoch1"])
        for sid, code in accepted:
            pred[sid] = pred[sid] | {code}
        passed = {}
        for name, cohort in cohorts.items():
            gold = {sid: all_gold[sid] for sid in cohort}
            metrics = arm_metrics(gold, {sid: pred[sid] for sid in cohort}, cohort, severe)
            item = panels[name]
            f1 = metrics["macro_f1"]
            fa = metrics["no_hazard_false_alarms"]
            missed = metrics["severity4"]["complete_misses"]
            observations[name].append((f1, fa, missed))
            conditions = (f1 - item["baseline8_f1"] >= .03,
                          fa * 5 <= item["normal_photos"],
                          missed <= item["baseline8_severity4_misses"])
            passed[name] = all(conditions)
            checks[name].update({key: flag for key, flag in zip(
                ("f1_gain_ge_003", "normal_fpr_le_20pct", "severity4_not_increased", "all"),
                (*conditions, passed[name]), strict=True)})
        joint += passed["primary_175"] and passed["sensitivity_107"]
    for name, metrics in observations.items():
        panels[name]["bounds"] = {key: [min(values), max(values)] for key, values in zip(
            ("macro_f1", "normal_false_alarms", "severity4_complete_misses"), zip(*metrics), strict=True)}
        panels[name]["assignments_passing"] = checks[name]
    result = {
        "scope": "Conditional exact Boolean enumeration, not an inference measurement. Assumes equal 2ep margin and verification decisions for identical photo/code prompts regardless of batching; threshold changes only the margin gate. The original standalone predictions are always preserved.",
        "frozen_candidate_sha256": frozen_hash,
        "input_prediction_sha256": {name: digest(path) for name, path in paths.items()},
        "sensitivity_mask_sha256": digest(mask_path),
        "labels_sha256": digest(labels),
        "unknown_assignments": 2 ** len(unknown),
        "assignments_passing_joint_175_and_107": joint,
        "panels": panels,
    }
    path = Path("results/fold4-v3-fixed-verifier-conditional-bounds.json")
    save_atomic(path, lambda f: f.write((json.dumps(result, indent=2) + "\n").encode()))
    return result

def score_completed():
    """Verify a completed GPU cache and score only outputs tied to that cache."""
    import math
    import re
    import sys
    from tempfile import TemporaryDirectory

    from .fold4_crosssplit_audit import _score_panels
    from .fold4_official_output_check import check_fold4
    from .metrics import check_text, reject_constant, unique_keys

    work = FROZEN.parent
    receipt_path = work / "completed.json"
    cache_path = work / "fixed_verifier_cache.json"
    bundle, receipt, cache = map(load, (FROZEN, receipt_path, cache_path))
    identity = cache["identity"]
    gate = bundle["fixed_epoch2_gate"]
    original = load(ORIGINAL_REPORT)
    if (identity["proposal_sha256"] != digest(FROZEN) or identity["manifest_sha256"] != digest(DEV)
            or identity["gate_sha256"] != digest(MARGIN_GATE) or identity["adapter_sha256"] != original["v3_adapter_sha256"]
            or gate != load(MARGIN_GATE) or gate != original["arms"]["v3_hybrid_1token_margin_verified"]["margin_gate"]):
        raise ValueError("GPU cache did not use the frozen 2ep adapter and gate")
    epoch2_key = load(V3_EVAL / "v3_epoch2_infer_cache.json")["identity"]["model_key"]
    if epoch2_key != f"bf16:{identity['adapter_sha256']}:{identity['prompt_sha256']}:px393216":
        raise ValueError("GPU verifier differs from the original BF16 checkpoint/prompt")
    if receipt["epoch2_exact_prior_hybrid_rows"] != 175:
        raise ValueError("GPU replay did not reproduce prior 2ep hybrid")
    paths = {"epoch1": V3_EVAL / "v3_epoch1_11code_standalone.jsonl",
             "epoch2": V3_EVAL / "v3_11code_standalone.jsonl",
             "epoch1_fixed2": work / "epoch1_fixed_epoch2_verifier.jsonl",
             "epoch2_fixed2": work / "epoch2_fixed_epoch2_verifier.jsonl"}
    prior = V3_EVAL / "v3_hybrid_margin_verified.jsonl"
    def checked(path):
        return read_index(path, lambda row: (decode_target(row, local_ids=True)[0], row), limit_size=True)
    if checked(prior) != checked(paths["epoch2_fixed2"]):
        raise ValueError("Frozen epoch-2 output differs from original hybrid")
    code_set = {name: {sid: set(row["hazards"]) for sid, row in checked(path).items()}
                for name, path in paths.items()}
    context = {"Any": Any, "CODES": CODES, "re": re, "json": json, "unique_keys": unique_keys,
               "reject_constant": reject_constant, "check_text": check_text}
    extract(Path("kwater_lab/vlm_11code.py"),
            {"validate_vlm_11code_obj", "verified_hazard", "parse_verified_hazard"}, context)
    gate_context = {"math": math}
    extract(Path("kwater_lab/fold4_same_verifier.py"), {"passes"}, gate_context)
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    gold = {r["sample_id"]: set(r["codes"]) for r in rows}
    groups = {r["sample_id"]: r["group_id"] for r in rows}
    mask = load(LOCAL / "fold4-group-exclusions-local.json")
    excluded = {link["fold4_group_id"] for link in mask["confirmed_cross_group_links"]}
    retained = {sid: g for sid, g in groups.items() if g not in excluded}
    sizes = Counter(retained.values())
    largest = min(sizes, key=lambda group: (-sizes[group], group))
    cohorts = {"primary_175": groups, "sensitivity_107": retained,
               "exploratory_58": {sid: g for sid, g in retained.items() if g != largest}}
    union_keys = {f"{c['sample_id']}|{c['code']}" for e in ("epoch1", "epoch2") for c in bundle[e]}
    if cache["margins"].keys() != union_keys or not all(math.isfinite(v) for v in cache["margins"].values()):
        raise ValueError("GPU cache does not contain one finite margin for each union candidate")
    generation_keys = set()
    panels = {name: {} for name in cohorts}
    for epoch in ("epoch1", "epoch2"):
        model_pred = code_set[epoch]
        key_pred = code_set[epoch + "_fixed2"]
        expected_additions = set()
        seen, passed = [], []
        for item in bundle[epoch]:
            sid, code = item["sample_id"], item["code"]
            key = f"{sid}|{code}"
            seen.append((sid, code))
            if gate_context["passes"](item, cache["margins"][key], gate):
                generation_keys.add(key)
                passed.append((sid, code))
                if context["parse_verified_hazard"](cache["generations"][key], code) is not None:
                    expected_additions.add((sid, code))
        if any(not model_pred[sid] <= key_pred[sid] for sid in gold):
            raise ValueError(f"{epoch}: fixed verifier output removed a draft code")
        actual = {(sid, code) for sid in gold for code in key_pred[sid] - model_pred[sid]}
        arm = receipt[epoch]
        if (actual != expected_additions or (len(seen), len(passed), len(actual)) != (
                arm["proposed"], arm["passed"], arm["accepted"])
                or arm["output_sha256"] != digest(paths[epoch + "_fixed2"])):
            raise ValueError(f"{epoch}: receipt, strict verifier cache decisions and rendered additions differ")
        for name, cohort in cohorts.items():
            subset = [(sid, code) for sid, code in actual if sid in cohort]
            panels[name][epoch] = {
                "proposed": sum(sid in cohort for sid, _ in seen),
                "margin_passed": sum(sid in cohort for sid, _ in passed),
                "accepted": len(subset),
                "added_tp_codes": sum(code in gold[sid] for sid, code in subset),
                "added_fp_codes": sum(code not in gold[sid] for sid, code in subset),
                "new_normal_false_alarm_photos": sum(not gold[sid] and not model_pred[sid] for sid, _ in subset),
            }
    if cache["generations"].keys() != generation_keys:
        raise ValueError("GPU completion must have exactly the union of passed verifier generations")
    out = Path("results/fold4-v3-fixed-verifier-paired-code-evaluation.json")
    baseline = V3_EVAL / "clean_8code_baseline.jsonl"
    with TemporaryDirectory(prefix="fixed-verifier-official-", dir=work) as scratch:
        official = check_fold4(DEV, None, Path("data/local-kit/kit"), Path(scratch) / "check",
                               Path(sys.executable), arm_files={name: paths[name] for name in ("epoch1_fixed2", "epoch2_fixed2")})
    if not official["all_arms_passed"]:
        raise ValueError("Frozen verifier outputs must pass both official checks")
    res = _score_panels(LOCAL, {"legacy8": baseline, **paths}, None, "shared_epoch2_verifier_and_margin_gate",
                        score_exploratory=True)
    res["official_checker"] = official
    res["frozen_verifier"] = {"identity": identity, "policy": gate,
                              "receipt_sha256": digest(receipt_path), "cache_sha256": digest(cache_path),
                              "epoch2_exact_prior_hybrid_rows": 175, "union_candidates": len(union_keys),
                              "candidate_pipeline_by_panel": panels}
    save_atomic(out, lambda f: f.write((json.dumps(res, indent=2) + "\n").encode()))
    return res

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "bounds", "score"), nargs="?", default="freeze")
    {"freeze": freeze, "bounds": conditional_bounds, "score": score_completed}[parser.parse_args().stage]()
