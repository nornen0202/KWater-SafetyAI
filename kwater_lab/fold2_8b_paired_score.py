"""Score recovered Fold 2 8B outputs against the exact existing 4B arm, locally."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import mean

from .fold0_group_sensitivity import COMMON, arm_metrics
from .fold2_8b_baseline import (
    BASE_REVISION, CONTROL, EVAL_SEGMENTS, RUN, STEPS, checkpoint_state, order_digest, write_json,
)
from .fold2_v3_epoch1 import LABELS, LAWS, MASK, PROMPT, split
from .fold4_same_verifier import digest, read
from .law_reranker_11code import read_selected_cases
from .metrics import CODES, counts, decode_target, paired_bootstrap, percentile, ratio, read_index
from .sub78_public200_compare import match_changes

RESULT = Path("results/fold2-matched-4b-8b-epoch1-20261128.json")
CONTROL_RESULT = Path("results/fold2-v3-4b-epoch1-20261128.json")
EXPOSURE_RESULT = Path("results/fold2-v3-4b-epoch1-exposure19-replay-20261128.json")


def shared_f1(gold, a, b, ids):
    left, right = counts(ids, gold, a), counts(ids, gold, b)
    codes = [c for c in CODES if left[0][c] or left[1][c] or right[1][c]]

    def score(arm, subset):
        actual, predicted, hit = arm
        return mean(ratio(2*hit[c], actual[c] + predicted[c]) or 0.0 for c in subset) if subset else None

    return codes, [(score(arm, codes), score(arm, COMMON)) for arm in (left, right)]


def score_panel(ids, groups, gold, a, b, severe_gold):
    gold, a, b, groups = ({sid: mapping[sid] for sid in ids} for mapping in (gold, a, b, groups))
    safe = [sid for sid in ids if not gold[sid]]
    severe = [sid for sid in ids if severe_gold[sid]]
    multi = [sid for sid in ids if len(gold[sid]) > 1]
    slip_negative = [sid for sid in ids if "slip_trip" not in gold[sid]]
    ordinary = {label: arm_metrics(gold, arm, groups, severe_gold)
                for label, arm in (("4b", a), ("8b", b))}
    for label, arm in (("4b", a), ("8b", b)):
        true_codes = sum(len(gold[sid]) for sid in multi)
        matches = sum(len(gold[sid] & arm[sid]) for sid in multi)
        ordinary[label]["multi_hazard"] = {"photos": len(multi), "target_codes": true_codes, "correct_codes": matches,
                                           "micro_recall": ratio(matches, true_codes),
                                           "exact_set_matches": sum(gold[sid] == arm[sid] for sid in multi)}
        slip_fp = sum("slip_trip" in arm[sid] for sid in slip_negative)
        ordinary[label]["slip_trip_on_absent_target"] = {"photos": len(slip_negative), "false_positives": slip_fp,
                                                          "false_positive_rate": ratio(slip_fp, len(slip_negative))}
    codes, point = shared_f1(gold, a, b, ids)
    members = defaultdict(list)
    for sid in sorted(ids):
        members[groups[sid]].append(sid)
    sites = sorted(members)
    rng = random.Random(42)
    active_deltas, common_deltas = [], []
    for _ in range(2000):
        sampled = [sid for site in rng.choices(sites, k=len(sites)) for sid in members[site]]
        _, ((a_active, a_common), (b_active, b_common)) = shared_f1(gold, a, b, sampled)
        if a_active is not None:
            active_deltas.append(b_active - a_active)
        common_deltas.append(b_common - a_common)

    def interval(deltas, column):
        deltas.sort()
        return {"delta_8b_minus_4b": point[1][column] - point[0][column] if codes or column else None,
                "defined_resamples": len(deltas),
                "percentile_95_ci": [percentile(deltas, q) for q in (0.025, 0.975)] if deltas else None}

    return {
        "photos": len(ids), "manifest_groups": len(sites), "metrics": ordinary,
        "normal_false_alarm_gate": {"rate": 0.18, "maximum_photos": math.floor(0.18 * len(safe))},
        "shared_active_codes": codes, "shared_active_macro_f1": dict(zip(("4b", "8b"), (x[0] for x in point))),
        "paired_group_bootstrap": {"resamples": 2000, "seed": 42,
                                   "active_per_arm": paired_bootstrap(gold, a, b, groups, resamples=2000, seed=42),
                                   "supplemental_shared_active": interval(active_deltas, 0),
                                   "fixed_8": interval(common_deltas, 1)},
        "paired_outcomes": {
            "correct_normal": match_changes({sid for sid in safe if not a[sid]},
                                            {sid for sid in safe if not b[sid]}, len(safe)),
            "severity4_any_correct": match_changes({sid for sid in severe if a[sid] & severe_gold[sid]},
                                                  {sid for sid in severe if b[sid] & severe_gold[sid]}, len(severe)),
        },
    }


def score(prepared, fit, evaluated, result):
    from .v3_turnaround import ActionSupportedLawRetriever, format_v3_record

    recipe = read(prepared / "transfer-receipt.json")["recipe"]
    train, outer = split()
    ids = [row["sample_id"] for row in outer]
    if (recipe["training"] != read(CONTROL / "frozen-recipe.json")
            or order_digest(outer) != recipe["eval_order_sha256"]
            or digest(PROMPT) != recipe["training"]["prompt_sha256"]
            or digest(LAWS) != recipe["training"]["law_corpus_sha256"]
            or digest(LABELS) != recipe["training"]["label_source_sha256"]):
        raise ValueError("Local scoring sources changed since preflight")
    adapter = fit / "epoch-1"
    state = checkpoint_state(adapter, STEPS, recipe)
    step_log = [json.loads(line) for line in (fit / "steps.jsonl").read_bytes().splitlines()]
    if step_log != state["records"]:
        raise ValueError("Recovered training log and checkpoint state differ")
    records = state["records"]
    del state
    discrete_keys = ("examples", "supervised_tokens", "decision_tokens")
    for actual, historic in zip(records, recipe["epoch1_step_controls_4b"], strict=True):
        if ([actual[k] for k in discrete_keys] != historic[:3] or
                not math.isclose(actual["applied_lr"], historic[3], rel_tol=1e-14, abs_tol=0.0)):
            raise ValueError("Ordered training supervision or learning rate differs from original 4B")
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    receipt = read(evaluated / "inference-receipt.json")
    expected_cache_key = f"fold2-8b:{BASE_REVISION}:{adapter_sha}:{digest(PROMPT)}:bf16:px393216"
    if (receipt["adapter_sha256"] != adapter_sha or receipt["cache_key"] != expected_cache_key
            or receipt["eval_order_sha256"] != recipe["eval_order_sha256"]):
        raise ValueError("Inference and recovered step-67 adapter identities differ")
    control_cache = CONTROL / "epoch1-bf16-raw-cache.json"
    old_inference = read(CONTROL_RESULT)["inference"]
    basis = read(control_cache)["identity"]
    batch_witness = read(Path("results/fold2-v3-4b-epoch1-batch-witness-20261128.json"))
    if (digest(control_cache) != batch_witness["raw_cache_sha256"]
            or batch_witness["predictions_sha256"] != old_inference["prediction_sha256"]
            or batch_witness["batch_transitions"] != [
                {"rows_already_complete": start, "effective_batch_size": batch}
                for start, _, batch in EVAL_SEGMENTS
            ]):
        raise ValueError("Original 4B cache or batch witness differs from the frozen evaluation")
    raw_outputs = {}
    for segment, (start, end, batch) in zip(receipt["segments"], EVAL_SEGMENTS, strict=True):
        path = evaluated / f"raw-{start}-{end}.json"
        cache = read(path)
        identity = cache["identity"]
        samples = basis["samples"][start:end]
        stats = segment["stats"]
        if ((segment["start"], segment["end"], digest(path)) != (start, end, segment["raw_sha256"])
                or identity["model_key"] != expected_cache_key
                or identity["max_new_tokens"] != 620
                or identity["batch_size"] != batch
                or identity["samples"] != samples
                or identity["image_sha256"] != {name: basis["image_sha256"][name] for _, name in samples}
                or any(identity[key] != basis[key] for key in ("prefix_sha256", "processor_size", "max_new_tokens"))
                or cache["effective_batch_size"] != batch
                or stats["photos"] != end-start
                or stats["initial_batch_size"] != batch or stats["final_effective_batch_size"] != batch
                or stats["strict_vlm_schema_success"] != sum(not row["vlm_schema_errors"] for row in cache["outputs"].values())
                or stats["raw_json_parse_success"] != sum(row["parse_error"] is None and row["parsed"] is not None
                                                          for row in cache["outputs"].values())
                or list(cache["outputs"]) != ids[start:end]
                or any(row["sample_id"] != sid for sid, row in cache["outputs"].items())):
            raise ValueError("Recovered inference cache failed ordered coverage or batch verification")
        raw_outputs.update(cache["outputs"])

    train_ids = {row["sample_id"] for row in train}
    train_cases = {row["sample_id"]: row for row in read_selected_cases(LABELS, train_ids)}
    if train_cases.keys() != train_ids:
        raise ValueError("Training-only case bank is incomplete")
    laws = [json.loads(line) for line in LAWS.read_bytes().splitlines() if line.strip()]
    ranker = ActionSupportedLawRetriever(laws, [train_cases[row["sample_id"]] for row in train])
    formatted = [format_v3_record(sid, raw_outputs[sid]["parsed"], ranker) for sid in ids]
    prediction_path = evaluated / "epoch1-bf16-predictions.jsonl"
    pending = prediction_path.with_suffix(".part")
    pending.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in formatted).encode())
    pending.replace(prediction_path)
    candidate = read_index(prediction_path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    control_path = CONTROL / "epoch1-bf16-predictions.jsonl"
    control = read_index(control_path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    if (list(control) != ids or list(candidate) != ids or
            digest(control_path) != old_inference["prediction_sha256"]):
        raise ValueError("Paired scoring requires exact coverage and immutable 4B controls")

    truth = {row["sample_id"]: row["target"]["hazards"] for row in read_selected_cases(LABELS, set(ids))}
    gold = {row["sample_id"]: set(row["codes"]) for row in outer}
    if truth.keys() != gold.keys() or any(set(truth[sid]) != codes for sid, codes in gold.items()):
        raise ValueError("Official labels differ from the frozen manifest")
    severe = {sid: {code for code, h in labels.items() if h["risk_level"]["severity"]["score"] == 4}
              for sid, labels in truth.items()}
    groups = {row["sample_id"]: row["group_id"] for row in outer}
    sensitivity = set(read(MASK)["prospective_mask_retained_ids"])
    stricter = read(CONTROL / "exposure19-reconstructed-ids-local.json")["ordered_ids"]
    exposure_sha = sha256(json.dumps(stricter, separators=(",", ":")).encode()).hexdigest()
    if (len(set(stricter)) != 19 or not set(stricter) <= sensitivity or
            exposure_sha != read(EXPOSURE_RESULT)["reconstructed_ordered_ids_sha256"]):
        raise ValueError("Reconstructed exploratory exposure mask changed")
    panels = {"primary_197": ids, "sensitivity_55": [sid for sid in ids if sid in sensitivity],
              "exploratory_19": stricter}
    write_json(result, {
        "scope": "Fold 2 photograph-held-out paired scale comparison. Known cross-site links exist in the 197-photo "
                 "canonical split. The 55-photo sensitivity retains prior exploratory exposure. The 19-photo "
                 "reconstruction is exploratory. Training systems are Windows RTX 3060 and Linux L4.",
        "recipe": recipe, "generated": receipt, "4b_predictions_sha256": digest(control_path),
        "8b_predictions_sha256": digest(prediction_path), "8b_steps_sha256": digest(fit / "steps.jsonl"),
        "ordered_training_step_controls_match": True, "exploratory_mask_sha256": exposure_sha,
        "panels": {name: score_panel(members, groups, gold, control, candidate, severe)
                   for name, members in panels.items()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=RUN)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--evaluated", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.evaluated.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Recovered raw and formatted predictions must stay in data/")
    score(args.prepared, args.fit, args.evaluated, args.result)
