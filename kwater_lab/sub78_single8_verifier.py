"""Freeze, measure, and score the 52-proposal same-base 8B verifier comparison."""

import argparse
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import time

from .fold0_group_sensitivity import arm_metrics
from .fold4_same_verifier import candidate_key, passes
from .metrics import decode_target, paired_bootstrap, read_index
from .sub78_verifier_audit import (
    CALIBRATION, DEV, INNER, LOCAL, MARGIN_GATE, OUTER, PROMPT, RUN, SPLIT, V3_EVAL,
    digest, load, non_accept_state, read_selected_cases, select_8b_fold4_candidates, write,
)

BASE = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
FIT = RUN.parent / "fit"
WORK = Path("data/local-eval-predictions/sub78-single8-verifier")
ARMS = ("epoch-1", "epoch-2")
PRECISIONS = ("bf16", "nf4")
ADAPTERS = ("adapter_config.json", "adapter_model.safetensors")
WEIGHT_HASH = {
    "epoch-1": "8269bc13e8c211ce1414f749da663c0c243bb93038d16e3891086cf30c44509b",
    "epoch-2": "2d501955745c8a0239ce91dfa5abf39782585f1fc10b909524ae41af78c745a8",
}
BASE_CONFIG = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
MASK = LOCAL / "fold4-group-exclusions-local.json"
TRAIN = Path("data/train")
FIXED_4B = Path("results/sub78-8b-verifier-audit-20261128.json")
SIXTH = V3_EVAL / "v3_epoch1_11code_standalone.jsonl"
RECIPE = {"pixels": 393216, "batch": 2, "max_new_tokens": 280, "seed": 42,
          "generation": "greedy for every proposal", "decision_tokens": [10330, 46996]}


def cohort():
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    if len(rows) != 175 or len({r["sample_id"] for r in rows}) != 175 or len({r["group_id"] for r in rows}) != 39:
        raise ValueError("Canonical Fold 4 cohort differs")
    return rows


def indices(path, ids):
    predictions = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    if list(predictions) != ids:
        raise ValueError(f"Prediction order or coverage differs: {path}")
    return predictions


def freeze(work):
    rows = cohort()
    ids = [r["sample_id"] for r in rows]
    draft = load(RUN / "cache.json")
    receipt = load(RUN / "generation.json")
    identity = draft["identity"]
    expected_key = f"bf16:{BASE_CONFIG}:{WEIGHT_HASH['epoch-1']}:{digest(PROMPT)}:px393216"
    if (
        list(draft["outputs"]) != ids
        or identity["samples"] != [[r["sample_id"], r["image"]] for r in rows]
        or identity["model_key"] != expected_key
        or receipt["photos"] != 175
        or receipt["adapter_sha256"] != WEIGHT_HASH["epoch-1"]
        or receipt["base_config_sha256"] != BASE_CONFIG
        or any(identity["image_sha256"].get(r["image"]) != r["sha256"] for r in rows)
    ):
        raise ValueError("8B BF16 draft provenance or order differs")
    candidates, _, _ = select_8b_fold4_candidates(rows, draft["outputs"])
    pred = indices(RUN / "predictions.jsonl", ids)
    if len(candidates) != 52 or len({c["sample_id"] for c in candidates}) != 52:
        raise ValueError("Expected 52 ordered unique proposals")
    for candidate in candidates:
        row = rows[candidate["row_idx"]]
        image = TRAIN / candidate["image"]
        if (
            (candidate["sample_id"], candidate["image"]) != (row["sample_id"], row["image"])
            or not image.resolve().is_relative_to(TRAIN.resolve())
            or digest(image) != row["sha256"]
            or candidate["code"] in pred[candidate["sample_id"]]
        ):
            raise ValueError("Candidate image, index or novel code differs")
        candidate["image_sha256"] = row["sha256"]
    shards = sorted(set(load(BASE / "model.safetensors.index.json")["weight_map"].values()))
    if len(shards) != 4 or any(Path(filename).name != filename for filename in shards):
        raise ValueError("Unexpected base shard index")
    model_files = shards + [
        "chat_template.json", "config.json", "generation_config.json", "merges.txt",
        "model.safetensors.index.json", "preprocessor_config.json", "tokenizer_config.json",
        "tokenizer.json", "video_preprocessor_config.json", "vocab.json",
    ]
    files = [
        DEV, SPLIT, INNER, OUTER, CALIBRATION, PROMPT, MARGIN_GATE, MASK, SIXTH,
        RUN / "cache.json", RUN / "generation.json", RUN / "predictions.jsonl", FIXED_4B,
        FIT / "training_summary.json", BASE / "source.json",
        Path(__file__).resolve().relative_to(Path.cwd().resolve()),
        *[Path("kwater_lab") / name for name in (
            "sub78_verifier_audit.py", "fold4_same_verifier.py", "clean_specialist.py",
            "hybrid_pipeline.py", "v3_turnaround.py", "vlm_11code.py",
            "fold0_group_sensitivity.py", "metrics.py",
        )],
        *[FIT / arm / name for arm in ARMS for name in ADAPTERS],
        *[BASE / name for name in model_files],
    ]
    # The existing byte ledger and original LFS pins allow a cheap CPU freeze.
    # Measure rehashes the actual model bytes before loading any weights.
    published = {entry["rfilename"]: entry for entry in load(BASE / "source.json")["files"]}
    ledger = {entry["path"]: entry for entry in load(Path("results/local_artifacts.json"))["artifacts"]}
    hashes = {}
    for path in files:
        name = path.as_posix()
        if path.suffix != ".safetensors":
            hashes[name] = digest(path)
            continue
        recorded = ledger[name]
        if path.stat().st_size != recorded["bytes"]:
            raise ValueError(f"Recorded model byte length differs: {name}")
        if path.parent == BASE:
            pinned = published[path.name]["lfs"]
            if recorded["sha256"] != pinned["sha256"] or recorded["bytes"] != pinned["size"]:
                raise ValueError(f"Recorded base shard provenance differs: {name}")
        hashes[name] = recorded["sha256"]
    if hashes[(BASE / "config.json").as_posix()] != BASE_CONFIG or any(
        hashes[(FIT / arm / ADAPTERS[1]).as_posix()] != WEIGHT_HASH[arm] for arm in ARMS
    ):
        raise ValueError("8B checkpoint SHA mismatch")
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / "manifest.json"
    plan = {"schema": 1, "source_sha256": hashes, "draft_identity": expected_key,
            "recipe": RECIPE, "gate": load(MARGIN_GATE), "candidates": candidates}
    if manifest.exists() and load(manifest) != plan:
        raise ValueError("Refusing to replace an existing divergent freeze")
    write(manifest, plan)
    return {"candidate_count": len(candidates), "draft_count": len(rows),
            "empty_draft_proposal_count": sum(c["was_vlm_empty"] for c in candidates),
            "base_shard_bytes": sum((BASE / name).stat().st_size for name in shards),
            "manifest_sha256": digest(manifest), "adapter_sha256": WEIGHT_HASH,
            "candidate_code_counts": dict(Counter(c["code"] for c in candidates))}


def plan_from_disk(work, *, model_weights):
    path = work / "manifest.json"
    plan = load(path)
    if plan["recipe"] != RECIPE or plan["gate"] != load(MARGIN_GATE):
        raise ValueError("Frozen policy differs")
    for name, expected in plan["source_sha256"].items():
        if not model_weights and Path(name).suffix == ".safetensors":
            continue
        if digest(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for row in plan["candidates"]:
        image = TRAIN / row["image"]
        if not image.resolve().is_relative_to(TRAIN.resolve()) or digest(image) != row["image_sha256"]:
            raise ValueError("Frozen candidate picture changed")
    return plan, digest(path)


def response_path(work, precision, arm):
    return work / f"{precision}-{arm}.json"


def measure(work, precision, manifest_sha):
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("-1", ""):
        raise RuntimeError("GPU execution requires visible CUDA")
    if digest(work / "manifest.json") != manifest_sha:
        raise ValueError("Use the explicitly pinned manifest SHA-256")
    plan, fingerprint = plan_from_disk(work, model_weights=True)
    from .v3_turnaround import score_verifier_1token_margins
    from .vlm_11code import load_qwen_4bit_or_bf16, seed_all, verify_specialist_candidates_with_vlm
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required for both verifier precision conditions")
    paths = {arm: response_path(work, precision, arm) for arm in ARMS}
    saved = {arm: load(path) for arm, path in paths.items() if path.exists()}
    props = torch.cuda.get_device_properties(0)
    runtime = {
        "torch": torch.__version__, "cuda": torch.version.cuda, "os": os.name, "gpu": props.name,
        "gpu_total_bytes": props.total_memory,
        **{name: version(name) for name in ("transformers", "peft", "accelerate")},
    }
    try:
        runtime["bitsandbytes"] = version("bitsandbytes")
    except PackageNotFoundError:
        if precision == "nf4":
            raise
        runtime["bitsandbytes"] = None
    candidates = plan["candidates"]
    for arm, response in saved.items():
        check_response(response, candidates, fingerprint, precision, arm)
        if response["runtime"] != runtime:
            raise ValueError("Existing arm runtime differs from the resumed environment")
    if len(saved) == 2:
        return
    seed_all(RECIPE["seed"])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model, processor = load_qwen_4bit_or_bf16(BASE, RECIPE["pixels"], precision == "nf4", FIT / ARMS[0])
    torch.cuda.synchronize()
    if bool(getattr(model, "is_loaded_in_4bit", False)) != (precision == "nf4"):
        raise ValueError("The loaded model does not match the requested precision")
    context = {"base_first_adapter_load_seconds": time.perf_counter() - start}
    token_ids = [processor.tokenizer.encode(word, add_special_tokens=False) for word in ("accept", "reject")]
    if token_ids != [[x] for x in RECIPE["decision_tokens"]]:
        raise ValueError("Actual decision tokens differ from the frozen template audit")
    prompt = PROMPT.read_text("utf-8")
    for arm in ARMS:
        if arm in saved:
            continue
        start = time.perf_counter()
        if arm == ARMS[1]:
            model.load_adapter(str(FIT / arm), adapter_name=arm, is_trainable=False)
            model.set_adapter(arm)
        actual_adapter = "default" if arm == ARMS[0] else arm
        if model.active_adapter != actual_adapter:
            raise ValueError("Wrong adapter active for the verifier")
        torch.cuda.synchronize()
        switch_time = time.perf_counter() - start
        start = time.perf_counter()
        margins = score_verifier_1token_margins(model, processor, candidates, TRAIN, prompt, RECIPE["batch"])
        torch.cuda.synchronize()
        margin_time = time.perf_counter() - start
        start = time.perf_counter()
        generated = verify_specialist_candidates_with_vlm(
            model, processor, candidates, TRAIN, prompt, RECIPE["batch"], RECIPE["max_new_tokens"],
        )
        torch.cuda.synchronize()
        response = {
            "manifest_sha256": fingerprint, "precision": precision, "arm": arm,
            "adapter_sha256": WEIGHT_HASH[arm], "runtime": runtime,
            "telemetry": {**context, "adapter_switch_seconds": switch_time, "margin_seconds": margin_time,
                          "generation_seconds": time.perf_counter() - start,
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            "rows": [{"key": candidate_key(r), "margin": m, "raw": r["raw_verify_text"]}
                     for r, m in zip(generated, margins, strict=True)],
        }
        check_response(response, candidates, fingerprint, precision, arm)
        write(paths[arm], response)


def check_response(response, candidates, fingerprint, precision, arm):
    if (
        response["manifest_sha256"] != fingerprint
        or response["precision"] != precision
        or response["arm"] != arm
        or response["adapter_sha256"] != WEIGHT_HASH[arm]
        or [row["key"] for row in response["rows"]] != [candidate_key(c) for c in candidates]
        or any(type(row["margin"]) not in (int, float) or not math.isfinite(row["margin"])
               or not isinstance(row["raw"], str) for row in response["rows"])
    ):
        raise ValueError("Invalid verifier response provenance, ordered coverage or raw payload")


def score(work, precision, manifest_sha):
    plan, fingerprint = plan_from_disk(work, model_weights=False)
    if fingerprint != manifest_sha:
        raise ValueError("Use the explicitly pinned manifest SHA-256")
    candidates = plan["candidates"]
    responses = {arm: load(response_path(work, precision, arm)) for arm in ARMS}
    for arm, response in responses.items():
        check_response(response, candidates, fingerprint, precision, arm)
    if responses[ARMS[0]]["runtime"] != responses[ARMS[1]]["runtime"]:
        raise ValueError("The two arms must use the same recorded runtime")
    rows = cohort()
    ids = [row["sample_id"] for row in rows]
    groups = {row["sample_id"]: row["group_id"] for row in rows}
    gold = {row["sample_id"]: set(row["codes"]) for row in rows}
    actual = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(TRAIN / "train.jsonl", set(ids))}
    if set(actual) != set(ids) or any(set(actual[sid]) != gold[sid] for sid in ids):
        raise ValueError("Official annotations differ from the canonical fold")
    severe = {
        sid: {code for code, value in hazards.items() if value["risk_level"]["severity"]["score"] == 4}
        for sid, hazards in actual.items()
    }
    mask = load(MASK)
    if mask["canonical_manifest_sha256"] != digest(DEV):
        raise ValueError("Sensitivity mask targets another fold manifest")
    excluded = {item["fold4_group_id"] for item in mask["confirmed_cross_group_links"]}
    retain = {sid: group for sid, group in groups.items() if group not in excluded}
    sizes = Counter(retain.values())
    largest = min(sizes, key=lambda group: (-sizes[group], group))
    cohorts = {"primary_175": groups, "sensitivity_107": retain,
               "exploratory_58": {sid: group for sid, group in retain.items() if group != largest}}
    if [len(c) for c in cohorts.values()] != [175, 107, 58]:
        raise ValueError("Wrong panel sizes")
    draft = indices(RUN / "predictions.jsonl", ids)
    sixth = indices(SIXTH, ids)
    _, scope, _ = select_8b_fold4_candidates(rows, load(RUN / "cache.json")["outputs"])
    parse = scope["parse_verified_hazard"]
    pred = {}
    candidate_states = {}
    for arm, response in responses.items():
        by_id = {}
        pred[arm] = {sid: set(codes) for sid, codes in draft.items()}
        for item, measured in zip(candidates, response["rows"], strict=True):
            sid, code = item["sample_id"], item["code"]
            gate = passes(item, measured["margin"], plan["gate"])
            hazard = parse(measured["raw"], code)
            generation = "accepted" if hazard is not None else non_accept_state(measured["raw"])
            by_id[sid] = (gate, generation)
            if gate and hazard is not None:
                pred[arm][sid].add(code)
        candidate_states[arm] = by_id
    fixed_4b = load(FIXED_4B)
    panels = {}
    for name, subset in cohorts.items():
        y = {sid: gold[sid] for sid in subset}
        views = {arm: {sid: p[sid] for sid in subset} for arm, p in {
            "sixth": sixth, "eight_b_standalone": draft, **pred,
        }.items()}
        metrics = {arm: arm_metrics(y, p, subset, severe) for arm, p in views.items()}
        reference = metrics["sixth"]
        decisions = {}
        for arm in ARMS:
            seen = [state for sid, state in candidate_states[arm].items() if sid in subset]
            additions = [(sid, code) for sid in subset for code in views[arm][sid] - views["eight_b_standalone"][sid]]
            m = metrics[arm]
            decisions[arm] = {
                "proposals": len(seen),
                "raw_generation": dict(Counter(state for _, state in seen)),
                "margin_pass": sum(gate for gate, _ in seen),
                "accepted_after_gate": sum(gate and state == "accepted" for gate, state in seen),
                "accepted_gold_true": sum(code in y[sid] for sid, code in additions),
                "accepted_gold_false": sum(code not in y[sid] for sid, code in additions),
                "passed_gate_generation": dict(Counter(state for gate, state in seen if gate)),
                "passes_existing_gate": {
                    "delta_f1_ge_003": m["macro_f1"] - reference["macro_f1"] >= 0.03,
                    "normal_fpr_le_20pct": 5 * m["no_hazard_false_alarms"] <= m["no_hazard_photos"],
                    "severity4_nonincrease": m["severity4"]["complete_misses"] <= reference["severity4"]["complete_misses"],
                },
                "next_phase_targets": {
                    "f1_ge_065": m["macro_f1"] >= 0.65,
                    "normal_fpr_le_18pct": 100 * m["no_hazard_false_alarms"] <= 18 * m["no_hazard_photos"],
                },
            }
        panels[name] = {
            "photos": len(subset), "groups": len(set(subset.values())), "metrics": metrics,
            "decisions": decisions,
            "fixed_4b_reference": fixed_4b["panels"][name]["measured_missing_merged"],
            "paired": {f"{b}_vs_{a}": paired_bootstrap(y, views[a], views[b], subset) for a, b in (
                (ARMS[0], ARMS[1]), ("eight_b_standalone", ARMS[0]), ("eight_b_standalone", ARMS[1]),
                ("sixth", ARMS[0]), ("sixth", ARMS[1]),
            )},
        }
    return {
        "scope": "Fixed Fold 4 proposal replay. The common frozen 4B gate is transferred unchanged; 8B margins are not calibrated or used to select the policy. Prior 4B data contain two hardware settings.",
        "draft_precision": "bf16", "verifier_precision": precision, "manifest_sha256": fingerprint,
        "runtime": responses[ARMS[0]]["runtime"],
        "adapter_sha256": WEIGHT_HASH,
        "gold_sha256": digest(TRAIN / "train.jsonl"),
        "raw_response_sha256": {arm: digest(response_path(work, precision, arm)) for arm in ARMS},
        "telemetry": {arm: value["telemetry"] for arm, value in responses.items()},
        "panels": panels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "measure", "score"))
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--precision", choices=PRECISIONS)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze(args.work)
    else:
        if args.precision is None or args.manifest_sha256 is None:
            parser.error("measure and score require --precision and --manifest-sha256")
        if args.stage == "measure":
            measure(args.work, args.precision, args.manifest_sha256)
            return
        result = score(args.work, args.precision, args.manifest_sha256)
    if args.output is not None:
        write(args.output, result)
    else:
        from pprint import pprint
        pprint(result, sort_dicts=False)


if __name__ == "__main__":
    main()
