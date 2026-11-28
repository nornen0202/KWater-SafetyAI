"""CPU replay of verifier reject-prefix stopping and Sub 7/8 evidence."""
import ast
from collections import Counter
from hashlib import sha256
from importlib.metadata import distribution
import json
from pathlib import Path
import re
from typing import Any

from .metrics import CODES, check_text, reject_constant, unique_keys

DATA = Path("data/local-eval-predictions")
RESULTS = Path("results")
BASE = Path("data/baseline/base_model")
REJECT = re.compile(r'^\s*\{\s*"decision"\s*:\s*"reject"')


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


class LocalTokenizer:
    def __init__(self):
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_file(str(BASE / "tokenizer.json"))

    def encode(self, text, *, add_special_tokens):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        assert not clean_up_tokenization_spaces
        return self.backend.decode(tokens, skip_special_tokens=skip_special_tokens)


def load_strict_parser():
    """Execute the production parser definitions without loading model libraries."""
    path = Path(__file__).with_name("vlm_11code.py")
    names = {"parse_verified_hazard", "verified_hazard", "validate_vlm_11code_obj"}
    all_nodes = ast.parse(path.read_text(encoding="utf-8"))
    source = ast.Module(body=[node for node in all_nodes.body
                              if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    if {node.name for node in source.body} != names:
        raise ValueError("Missing strict parser definition")
    scope = dict(json=json, re=re, Any=Any, CODES=CODES, check_text=check_text,
                 reject_constant=reject_constant, unique_keys=unique_keys)
    exec(compile(source, str(path), "exec"), scope)
    staged_path = Path("data/submissions/safeground_v3_full1045_hybrid_dir/main.py")
    staged_parsers = {node.name: ast.dump(node) for node in ast.parse(staged_path.read_text(encoding="utf-8")).body
                      if isinstance(node, ast.FunctionDef) and node.name in names}
    source_parsers = {node.name: ast.dump(node) for node in source.body}
    return scope["parse_verified_hazard"], hash_file(path), staged_parsers == source_parsers


def accepted_equivalence(rows, tokenizer, parser, batch_size, clamp=280):
    """Estimate forward iterations under dense, fixed-size greedy batches."""
    row_count = Counter()
    decisions = Counter()
    token_counts = Counter()
    batches = Counter()
    for group in rows:
        for start in range(0, len(group), batch_size):
            chunk = group[start:start+batch_size]
            original, shortened = [], []
            for code, text in chunk:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                base = min(clamp, len(tokens)+1)
                parsed = parser(text, code)
                row_count["rows"] += 1
                row_count["strict_accepted"] += parsed is not None
                try:
                    obj = json.loads(text, object_pairs_hook=unique_keys)
                    decision = obj.get("decision") if isinstance(obj, dict) else None
                except (TypeError, ValueError, RecursionError):
                    decision = None
                decisions["explicit_reject" if decision == "reject" else "accept_seen" if decision == "accept"
                          else "other_or_malformed"] += 1
                row_count["accept_decision_failed_strict_parser"] += decision == "accept" and parsed is None
                stop = next(((i, tokenizer.decode(tokens[:i], skip_special_tokens=True, clean_up_tokenization_spaces=False))
                             for i in range(1, min(6, len(tokens))+1)
                             if REJECT.match(tokenizer.decode(tokens[:i], skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=False))), None)
                cut = stop[0] if stop else base
                row_count["canonical_reject_prefix_within_six"] += stop is not None
                row_count["strict_object_changes_anchored"] += stop is not None and parser(stop[1], code) != parsed
                row_count["strict_object_changes_broad_substring"] += (
                    '"reject"' in tokenizer.decode(tokens[:6], clean_up_tokenization_spaces=False)
                    and parsed is not None
                )
                original.append(base)
                shortened.append(cut)
                token_counts["baseline_nonpad"] += base
                token_counts["early_stop_nonpad"] += cut
            before, after = max(original), max(shortened)
            batches["batches"] += 1
            batches["shorter_batches"] += after < before
            batches["single_row_or_all_rows_detectable"] += all(a > b for a, b in zip(original, shortened))
            batches["baseline_forward_iterations"] += before
            batches["early_stop_forward_iterations"] += after
            batches["baseline_dense_token_slots"] += before * len(chunk)
            batches["early_stop_dense_token_slots"] += after * len(chunk)
    return {"rows": dict(row_count), "raw_decision_counts": dict(decisions),
            "reencoded_token_estimate": dict(token_counts), "dense_batch_estimate": dict(batches)}


def evaluate():
    from .fold4_same_verifier import candidate_key, passes, prioritized_union

    tokenizer = LocalTokenizer()
    parse_verified_hazard, parser_sha, staged_parser_matches = load_strict_parser()
    cache_path = DATA / "fold4-same-verifier/fixed_verifier_cache.json"
    proposal_path = DATA / "fold4-same-verifier/proposals.json"
    fixed = read(cache_path)
    props = read(proposal_path)
    gate = read(Path("data/v3-11code-fold4/margin_gate_calibration.json"))
    if any(hash_file(path) != fixed["identity"][key] for key, path in
           [("proposal_sha256", proposal_path), ("gate_sha256", Path("data/v3-11code-fold4/margin_gate_calibration.json"))]):
        raise ValueError("Fixed verifier cache identity mismatch")
    groups = prioritized_union(*[[c for c in props[e] if passes(c, fixed["margins"][candidate_key(c)], gate)]
                                 for e in ("epoch2", "epoch1")])
    if [candidate_key(c) for group in groups for c in group] != list(fixed["generations"]):
        raise ValueError("Fixed verifier cache order differs from actual epoch-2 and novel epoch-1 generation batches")
    panels = {"fold4_fixed_epoch2_bf16_actual_batches": {
        "sha256": hash_file(cache_path),
        **accepted_equivalence([[(c["code"], fixed["generations"][candidate_key(c)]) for c in group]
                               for group in groups], tokenizer, parse_verified_hazard, 2)}}
    for key, suffix in [
        ("fold0_v2_specialist", "fold0/fold0-qlora-v2-118-bf16/specialist_verification_cache.json"),
        ("fold0_v2_official_anchored", "fold0/v2-official-anchored/anchored_verifier_cache.json"),
    ]:
        path = DATA / suffix
        rows = read(path)["results"]
        panels[key] = {"sha256": hash_file(path), "batching_note": "Singleton counterfactual only; cache order may combine reused contexts.",
                       **accepted_equivalence([[(r["code"], r["raw_verify_text"]) for r in rows]],
                                              tokenizer, parse_verified_hazard, 1, clamp=320)}
    transformers = distribution("transformers")
    utils = transformers.locate_file("transformers/generation/utils.py").read_text(encoding="utf-8")
    mixin = next(node for node in ast.parse(utils).body
                 if isinstance(node, ast.ClassDef) and node.name == "GenerationMixin")
    sampling_method = next(node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == "_sample")
    gen_src = ast.get_source_segment(utils, sampling_method)
    return {"transformers_version": transformers.version, "tokenizer_sha256": hash_file(BASE / "tokenizer.json"),
            "strict_parser_source_sha256": parser_sha,
            "deployed_stage_parser_ast_matches": staged_parser_matches,
            "tokenizer_ids": {s: tokenizer.encode(s, add_special_tokens=False)
                              for s in ('accept', 'reject', '{"decision":"', '{"decision":"reject"')},
            "generation_configuration": {k: read(BASE / "generation_config.json")[k] for k in ("eos_token_id", "pad_token_id")},
            "installed_greedy_implementation_checks": {s: s in gen_src for s in
                ("unfinished_sequences", "stopping_criteria(input_ids, scores)", "pad_token_id", "torch.argmax(next_token_scores")},
            "limits": ["Raw caches preserve decoded text, not original token IDs or times per row/batch.",
                       "Re-encoding can alter token boundaries; add one estimated EOS step within the original cap.",
                       "Batch max models dense decode iterations only; it cannot produce a wall-clock forecast.",
                       "Prefix check begins after the common padded input width and only commits after the full JSON string value.",
                       "Forced one-token margin prefix is a different conditional path from free greedy JSON generation.",
                       "Strict parser equivalence is an offline replay; no early-stop deployment or GPU timing A/B executed."],
            "replays": panels}


def benchmark_evidence():
    scale = read(RESULTS / "seoul-l4-roadmap-experiments-20261128.json")
    clock = scale["full1045_seventh_public200"]
    telemetry = clock["runtime_phases"]
    directory = Path("data/local-reports/seoul-seventh-200-20261128")
    snapshot = sorted(path.name for path in directory.iterdir() if path.is_file())
    cohorts = read(RESULTS / "paired-qwen-4b-8b-fold4-20261128.json")["panels"]
    dev = read(Path("data/clean-split-v1/split_manifest_dev.json"))
    fold4 = [row for row in dev if row["split"] == "dev" and row["dev_fold"] == 4]
    mask = read(Path("data/fold4-crosssplit-visual-review/fold4-group-exclusions-local.json"))
    excluded = {r["fold4_group_id"] for r in mask["confirmed_cross_group_links"]}
    scale_data = Path("data/vlm-scale-seoul-8b-fold4-20261128")
    caches = {
        "4b_epoch1": DATA / "fold4-v3-benchmark/v3_epoch1_infer_cache.json",
        "4b_epoch2": DATA / "fold4-v3-benchmark/v3_epoch2_infer_cache.json",
        "8b_epoch1": scale_data / "eval1/cache.json",
        "8b_epoch2": scale_data / "eval2/cache.json",
    }
    raw_by_arm = {arm: read(path)["outputs"] for arm, path in caches.items()}
    if any(set(raw) != {r["sample_id"] for r in fold4} for raw in raw_by_arm.values()):
        raise ValueError("Abstention raw cache does not cover exactly the 175 evaluation photos")
    cohort_rows = {}
    for label, cohort in cohorts.items():
        hazard_ids = [r["sample_id"] for r in fold4 if r["codes"] and
                      (label == "primary_175" or r["group_id"] not in excluded)]
        cohort_rows[label] = {
            arm: {"hazard_predicted_empty": m["hazard_completely_missed"], "hazard_photos": m["hazard_photos"],
                  "severity4_completely_missed": m["severity4"]["complete_misses"],
                  "severity4_photos": m["severity4"]["photos"],
                  "raw_explicit_no_visible_hazard": sum(
                      (raw_by_arm[arm][sid]["parsed"] or {}).get("status") == "no_visible_hazard" for sid in hazard_ids),
                  "raw_parse_failed": sum(raw_by_arm[arm][sid]["parsed"] is None for sid in hazard_ids),
                  "raw_internal_schema_error": sum(bool(raw_by_arm[arm][sid]["vlm_schema_errors"]) for sid in hazard_ids)}
            for arm, m in cohort["arm_metrics"].items()
        }
    fold2 = [row for row in dev if row["split"] == "dev" and row["dev_fold"] == 2]
    disjoint = not ({row["group_id"] for row in fold2} & {row["group_id"] for row in fold4})
    eligible = [row for row in dev if row["split"] == "dev" and row["dev_fold"] != 2]
    ledger = read(RESULTS / "v3_hybrid_deployment_ledger.json")["production_full1045_hybrid_zip_ledger"]
    spend = scale["spend_estimate_krw"]
    resources = scale["resources"]
    rate = scale["cost_basis"]["planning_krw_per_usd"] * (1 + scale["cost_basis"]["vat_fraction"])
    estimate_sum = sum(r["planning_krw_including_vat_before_egress"] for r in resources)
    return {"seoul_full1045_200": {"retained_artifact_basenames": snapshot,
                "passed_to_verifier": telemetry["margin_gate_passed_candidates"],
                "accepted": telemetry["verified_accepted_candidates"],
                "not_accepted": telemetry["margin_gate_passed_candidates"]-telemetry["verified_accepted_candidates"],
                "explicit_reject_identifiable": False, "candidate_raw_generations_retained": False,
                "total_kit_seconds": clock["kit_clock"]["total_seconds"],
                "verify_generate_seconds": telemetry["verify_generate_seconds"],
                "requested_145s_fraction_of_verify_phase": 145/telemetry["verify_generate_seconds"],
                "measured_early_stop_seconds_or_output_parity": None},
            "increased_abstention": cohort_rows,
            "training": {model: {"epoch_weighted_token_losses": [r["weighted_token_loss"] for r in fit["history"]],
                                 "curriculum_breakdown": fit["curriculum_breakdown"],
                                 "optimizer_steps": fit["optimizer_steps"], "outer_exclusion_fold": fit["val_fold"]}
                         for model, fit in {"4b": read(Path("data/v3-11code-fold4/training_summary.json")),
                                             "8b": read(scale_data / "fit/training_summary.json")}.items()},
            "causal_claim": "Mixed inspect/verification curriculum and additional training steps co-occur. No matched inspection-only arm or task-gradient intervention to attribute pass-to-pass changes to reject targets.",
            "fold2": {"photos": len(fold2), "groups": len({r["group_id"] for r in fold2}),
                      "normal": sum(not r["codes"] for r in fold2), "fold2_fold4_manifest_groups_disjoint": disjoint,
                      "fold2_outer_training_dev_photos": len(eligible),
                      "independent_of_existing_fold4_or_full1045_model": False,
                      "required": "New Fold-2-excluded weights, prospectively locked recipe, exclusions applied to all modules/case banks and grouped inner fit/exposure audit."},
            "production_calibration": ledger["training_and_calibration_scope"]["margin_gate_exposure_audit"],
            "spend_basis": spend,
            "spend_crosscheck": {
                "actual_scenario_arithmetic_krw": [estimate_sum, estimate_sum + 450 + 207,
                                                  estimate_sum + 450 + 1500],
                "additional_approved_ceiling_krw": 3500,
                "additional_approved_ceiling_plus_conservative_estimate_krw": spend[
                    "including_1500_krw_total_network_and_miscellaneous_allowance"] + 3500,
                "one_hour_g2_8_spot_compute_only_krw": round(resources[1]["machine_hourly_usd"] * rate),
                "forty_five_minutes_g2_16_ondemand_compute_only_krw": round(resources[2]["machine_hourly_usd"] * rate * .75),
                "future_plan_note": "The approved additional 3,500 KRW ceiling is within the 15,000 KRW total and excluded from prior spend. The 2,700 KRW roadmap figure is a planning estimate."}}


def main():
    report = {**evaluate(), **benchmark_evidence()}
    (RESULTS / "sub78-reject-runtime-audit-20261128.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
