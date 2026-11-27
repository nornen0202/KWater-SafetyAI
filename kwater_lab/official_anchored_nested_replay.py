"""Replay one already-fitted Fold 0 inner-OOF policy without model or GPU imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from .fold0_production_compare import fold_rows, image_id
from .law_reranker_11code import read_selected_cases
from .metrics import CODES, check_text, decode_target, evaluate, read_index, reject_constant, unique_keys


MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
LABELS = Path("data/train/train.jsonl")
MASK = Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")
OOF = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
INNER = Path("data/clean-specialist-v1/strict-inner-oof/fold-0.npz")
POLICY = Path("results/clean-neural-calibration-fold0.json")
DIRECTORY = Path("data/local-eval-predictions/fold0")
FROZEN = DIRECTORY / "v2-official-anchored/frozen_proposals.json"
HISTORIC = DIRECTORY / "v2-official-anchored/anchored_verifier_cache.json"
FOLLOWUP = DIRECTORY / "fold0-qlora-v2-118-bf16/specialist_verification_cache.json"
OFFICIAL = Path("data/fold0-qlora-v2-118-official/output/submission.jsonl")
OFFICIAL_REPORT = Path("results/fold0_qlora_v2_118_official.json")
VERIFIED_REPORT = Path("results/v2_official_anchored_verified.json")
PROMPT = Path("kwater_lab/system_prompt_11code.txt")
OUTPUT = Path("results/v2_official_anchored_true_nested_replay.json")
V2 = DIRECTORY / "fold0-qlora-v2-118-bf16/ft_vlm_11code.jsonl"
V2_REPORT = Path("results/fold0_qlora_v2_118_bf16_vlm.json")
BLIND = Path("data/fold0-ai-blind-review-20261127")
LEGACY_RECIPE = Path("data/baseline/assets/train.jsonl")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_functions():
    """Reuse source array and parser functions without importing the GPU modules."""
    namespace = {"np": np, "CODES": CODES, "NUM_CODES": len(CODES), "Any": Any, "json": json, "re": re,
                 "unique_keys": unique_keys, "reject_constant": reject_constant, "check_text": check_text}
    sources = {
        "clean_specialist": ("apply_calibrated_policy",),
        "hybrid_pipeline": ("compute_calib_code_precisions", "usable_draft_hazards", "select_booster_candidates"),
        "vlm_11code": ("validate_vlm_11code_obj", "verified_hazard", "parse_verified_hazard"),
    }
    for module, names in sources.items():
        path = Path(__file__).with_name(module + ".py")
        parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(selected) != len(names) or any(node.decorator_list for node in selected):
            raise ValueError("The source definitions no longer match the CPU replay")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def key(row):
    return row["sample_id"], row["image"], row["code"]


def cached_decisions(anchored, followup, manifest, parse):
    configurations = [obj["fingerprint"] for obj in (anchored, followup)]
    configs = [item["inputs"] for item in configurations]
    invariants = ("model_key", "prompt_sha256", "processor_size", "max_new_tokens")
    if (any(cfg["batch_size"] != 1 for cfg in configurations)
            or any(configs[0][field] != configs[1][field] for field in invariants)
            or configs[0]["prompt_sha256"] != digest(PROMPT)
            or configs[0]["ft_verifier_cache_sha256"] != digest(FOLLOWUP)
            or configs[0]["frozen_proposals_sha256"] != digest(FROZEN)):
        raise ValueError("Recorded cached responses do not share one verifier identity")
    result = {}
    raw_by_key = {}
    for source, receipt, config in zip(("anchored", "prior_followup"), (anchored, followup), configs, strict=True):
        candidates = receipt["fingerprint"]["candidates"]
        responses = receipt["results"]
        if [key(row) for row in candidates] != [key(row) for row in responses] or len({key(r) for r in responses}) != len(responses):
            raise ValueError("Cache has missing, reordered or duplicate responses")
        for row in responses:
            photo, image, code = key(row)
            if manifest[photo]["image"] != image or manifest[photo]["sha256"] != config["selected_images_sha256"].get(image):
                raise ValueError("Cache image bytes differ from the canonical image identity")
            accepted = parse(row["raw_verify_text"], code) is not None
            if accepted != row["accepted"]:
                raise ValueError("Strict reparse differs from historically recorded acceptance")
            if key(row) in raw_by_key and raw_by_key[key(row)] != row["raw_verify_text"]:
                raise ValueError("Shared historic cache entries must contain identical raw completion text")
            raw_by_key[key(row)] = row["raw_verify_text"]
            result.setdefault(key(row), (accepted, source))
    return result, {k: configs[0][k] for k in invariants}


def scores(gold, groups, baseline, proposals, add, severity):
    result = {sid: set(codes) for sid, codes in baseline.items()}
    for sid, image, code in proposals:
        if (sid, image, code) in add:
            result[sid].add(code)
    out = evaluate(gold, result, groups)
    return {**{k: out[k] for k in ("macro_f1", "active_codes", "no_hazard_false_alarms", "no_hazard_photos")},
            "severity4_complete_misses": sum(bool(severity[sid]) and not (severity[sid] & result[sid]) for sid in gold),
            "severity4_photos": sum(bool(severity[sid]) for sid in gold)}


def summarize(rows, official, fixed, nested, cache, severity):
    ids = {row["sample_id"] for row in rows}
    gold = {row["sample_id"]: set(row["codes"]) for row in rows}
    groups = {row["sample_id"]: row["group_id"] for row in rows}
    baseline = {sid: official[sid] for sid in ids}
    a, b = ({key(item) for item in prop if item["sample_id"] in ids} for prop in (fixed, nested))
    a_photos = {pair[0] for pair in a}
    b_photos = {pair[0] for pair in b}
    true = lambda pair: pair[2] in gold[pair[0]]
    fixed_accepted = {pair for pair in a if cache[pair][0]}
    nested_known = b & cache.keys()
    nested_accepted = {pair for pair in nested_known if cache[pair][0]}
    nested_unknown = b - nested_known
    unknown_gold = {pair for pair in nested_unknown if true(pair)}

    def measure(proposals, additions):
        return scores(gold, groups, baseline, proposals, additions, severity)

    def bounds(known):
        unknown = sorted(b - known)
        if len(unknown) > 12:
            raise ValueError("This frozen replay expects a small exact scenario enumeration")
        accepted = {pair for pair in known if cache[pair][0]}
        scenarios = [measure(b, accepted | set(selected)) for k in range(len(unknown) + 1)
                     for selected in combinations(unknown, k)]
        return {"scenarios": len(scenarios), "unknown": len(unknown),
                "ranges": {field: [min(s[field] for s in scenarios), max(s[field] for s in scenarios)]
                           for field in ("macro_f1", "no_hazard_false_alarms", "severity4_complete_misses")}}

    def stats(items):
        accepted = {pair for pair in items if pair in cache and cache[pair][0]}
        unknown = items - cache.keys()
        return {"candidates": len(items), "annotated_positive": sum(map(true, items)),
                "unannotated": sum(not true(pair) for pair in items),
                "confirmed_accepted_tp": sum(map(true, accepted)),
                "confirmed_accepted_fp": sum(not true(pair) for pair in accepted),
                "unknown_annotated_positive": sum(map(true, unknown)),
                "unknown_unannotated": sum(not true(pair) for pair in unknown)}

    return {
        "photos": len(rows), "groups": len(set(groups.values())),
        "fixed": stats(a), "nested": stats(b),
        "same_photo_same_code": stats(a & b),
        "removed_without_replacement": stats({pair for pair in a - b if pair[0] not in b_photos}),
        "old_code_of_replacement": stats({pair for pair in a - b if pair[0] in b_photos}),
        "new_code_of_replacement": stats({pair for pair in b - a if pair[0] in a_photos}),
        "new_photo": stats({pair for pair in b - a if pair[0] not in a_photos}),
        "nested_matching_cache_by_origin": dict(Counter(cache[pair][1] for pair in nested_known)),
        "fixed_measured": measure(a, fixed_accepted),
        "nested_measured_subset_unknowns_deferred": measure(b, nested_accepted),
        "nested_lower_bound_unknown_false_only": measure(b, nested_accepted | (nested_unknown - unknown_gold)),
        "nested_upper_bound_unknown_true_only": measure(b, nested_accepted | unknown_gold),
        "nested_label_oracle_all_proposals": measure(b, {pair for pair in b if true(pair)}),
        "exhaustive_existing_anchor_cache_only": bounds(a & b),
        "exhaustive_all_exact_matched_caches": bounds(nested_known),
    }


def ancillary_audit(manifest, rows):
    if read(V2_REPORT)["formatted_output_sha256"] != digest(V2):
        raise ValueError("BF16 output does not match the measured local receipt")
    predictions = read_index(V2, lambda row: (decode_target(row, local_ids=True)[0], row), limit_size=True)
    local_ids = {row["sample_id"] for row in rows}
    if predictions.keys() != local_ids:
        raise ValueError("BF16 output does not cover Fold 0")
    predicted_hazards = [hazard for row in predictions.values() for hazard in row["hazards"].values()]
    if read(BLIND / "judgements-lock.json")["judgements_sha256"] != digest(BLIND / "judgements-blind.json"):
        raise ValueError("Expected the locked blind audit")
    if read(BLIND / "packet-provenance.json")["candidate_output_sha256"] != digest(V2):
        raise ValueError("Blind packet corresponds to another arm")
    panel = []
    keys = read(BLIND / "sealed-key.json")
    for case, mapping in keys.items():
        letter, = (arm for arm, name in mapping.items() if name == "candidate")
        panel.extend(read(BLIND / "blind" / f"{case}.json")[letter]["hazards"].values())
    with LEGACY_RECIPE.open(encoding="utf-8") as stream:
        recipe = [json.loads(line, object_pairs_hook=unique_keys) for line in stream if line.strip()]
    recipes = {r["sample_id"] for r in recipe}
    return {
        "v2_bf16_output_risk_photos": sum(bool(row["hazards"]) for row in predictions.values()),
        "v2_bf16_hazard_items": len(predicted_hazards),
        "v2_bf16_refs_per_hazard_histogram": dict(Counter(len(h.get("references", [])) for h in predicted_hazards)),
        "blind_photos": len(keys), "blind_candidate_hazards": len(panel),
        "blind_candidate_refs_per_hazard_histogram": dict(Counter(len(h.get("references", [])) for h in panel)),
        "dev_only_selector_population_for_val_fold_minus_one": sum(r["split"] == "dev" for r in manifest),
        "measured_v2_val_fold": read(Path("data/vlm-11code-fold0-v2/training_summary.json"))["val_fold"],
        "measured_v2_train_photos": read(Path("data/vlm-11code-fold0-v2/training_summary.json"))["train_photos"],
        "bundled_legacy_training_recipe_rows": len(recipe),
        "legacy_recipe_fold0_id_intersection": len(local_ids & recipes),
        "recipe_scope": "Artifact membership only; does not establish actual training IDs of the released checkpoint.",
    }


def run():
    funcs = cpu_functions()
    manifest = read(MANIFEST)
    by_id = {row["sample_id"]: row for row in manifest}
    rows = fold_rows(MANIFEST)
    frozen = read(FROZEN)
    historic, followup = read(HISTORIC), read(FOLLOWUP)
    anchored = read(VERIFIED_REPORT)
    report = read(OFFICIAL_REPORT)
    if (anchored["proposal_file_sha256"] != digest(FROZEN) or report["output_sha256"] != digest(OFFICIAL)
            or report["official_checker_valid_rows"] != 216
            or frozen["fingerprint"]["official_predictions_sha256"] != digest(OFFICIAL)
            or frozen["fingerprint"]["oof_scores_sha256"] != digest(OOF)
            or frozen["fingerprint"]["manifest_sha256"] != digest(MANIFEST)):
        raise ValueError("Frozen anchor artifacts changed")
    official_ids = {image_id(row["sample_id"]): row["sample_id"] for row in rows}
    official = read_index(OFFICIAL, decode_target, limit_size=True)
    if official.keys() != official_ids.keys():
        raise ValueError("Expected complete official coverage")
    official = {official_ids[anon]: value for anon, value in official.items()}
    calibration = read(POLICY)
    if calibration["outer_fold"] != 0 or calibration["outer_probability_sha256"] != digest(OOF):
        raise ValueError("Expected the independently fitted Fold 0 calibration")
    calibration = calibration["correct_inner_oof_policy"]
    with np.load(INNER, allow_pickle=False) as arc:
        inner_ids = arc["train_ids"].tolist()
        fp = json.loads(str(arc["fingerprint"].item()))
        inputs = {k.replace("\\", "/"): v for k, v in fp["inputs"].items()}
        if (fp["outer"] != 0 or fp["epochs"] != 55 or fp["seeds"] != [42, 1042, 2042]
                or inputs[MANIFEST.as_posix()] != digest(MANIFEST) or inputs[LABELS.as_posix()] != digest(LABELS)
                or len(inner_ids) != len(set(inner_ids)) or set(inner_ids) != {r["sample_id"] for r in manifest if r["dev_fold"] != 0}):
            raise ValueError("Precision source is not the frozen outer-training-only group OOF cache")
        y = np.array([[code in by_id[sid]["codes"] for code in CODES] for sid in inner_ids])
        precision = funcs["compute_calib_code_precisions"](arc["code_scores"], arc["scene_scores"], y, calibration)
    with np.load(OOF, allow_pickle=False) as arc:
        index = {sid: i for i, sid in enumerate(arc["ids"].tolist())}
        outer = [index[r["sample_id"]] for r in rows]
        if not np.all(arc["folds"][outer] == 0):
            raise ValueError("Expected fold zero predictions")
        probs, scenes = arc["code_probs"][outer], arc["scene_probs"][outer]
    drafts = [{"parsed": {"status": "hazard" if official[r["sample_id"]] else "no_visible_hazard",
                          "hazards": [{"code": code} for code in official[r["sample_id"]]]}} for r in rows]
    fixed_policy = {"temperatures": [1.0] * len(CODES), "thresholds": [0.5] * len(CODES), "scene_threshold": 0.5}
    select = funcs["select_booster_candidates"]
    fixed = select(rows, drafts, probs, scenes, fixed_policy, max_candidates_per_photo=1)
    if [key(row) for row in fixed] != [key(row) for row in frozen["proposals"]]:
        raise ValueError("Fixed branch failed to reproduce all existing proposals")
    nested = select(rows, drafts, probs, scenes, calibration, code_precisions=precision, max_candidates_per_photo=1)
    cached, verifier = cached_decisions(historic, followup, by_id, funcs["parse_verified_hazard"])
    for sid, image, code in {key(row) for row in fixed + nested} & cached.keys():
        if digest(Path("data/train") / image) != by_id[sid]["sha256"]:
            raise ValueError("Current photo bytes do not match the stored verifier input")
    mask = read(MASK)
    if mask["canonical_manifest_sha256"] != digest(MANIFEST) or frozen["fingerprint"]["group_mask_sha256"] != digest(MASK):
        raise ValueError("Expected the previously frozen group sensitivity mask")
    excluded = {item["fold0_group_id"] for item in mask["confirmed_cross_group_links"]}
    severity = {r["sample_id"]: {code for code, hazard in r["target"]["hazards"].items()
                               if hazard["risk_level"]["severity"]["score"] == 4}
                for r in read_selected_cases(LABELS, {row["sample_id"] for row in rows})}
    if set(severity) != {row["sample_id"] for row in rows}:
        raise ValueError("Severity labels differ from Fold 0")
    return {
        "scope": "One locked CPU proposal replay, no verifier generation or policy fitting. Historical cache identities only.",
        "input_sha256": {path.as_posix(): digest(path) for path in (MANIFEST, MASK, POLICY, INNER, OOF, FROZEN, HISTORIC, FOLLOWUP)},
        "inner_calibration_rows": len(inner_ids), "inner_precision_by_code": precision,
        "cache_identity": verifier,
        "ancillary_artifact_audit": ancillary_audit(manifest, rows),
        "cohorts": {"primary": summarize(rows, official, fixed, nested, cached, severity),
                    "confirmed_cross_group_exclusion": summarize([r for r in rows if r["group_id"] not in excluded],
                                                                  official, fixed, nested, cached, severity)},
    }


if __name__ == "__main__":
    import sys
    result = run()
    assert "torch" not in sys.modules
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
