"""Audit the 8B epoch-1 draft with cached frozen 4B epoch-2 verifier decisions."""

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any

import numpy as np

from .fold0_group_sensitivity import arm_metrics
from .fold4_crosssplit_audit import LOCAL
from .fold4_frozen_verifier import (
    DEV,
    SPLIT,
    INNER,
    OUTER,
    CALIBRATION,
    MARGIN_GATE,
    FROZEN,
    V3_EVAL,
    extract,
    load,
    digest,
)
from .law_reranker_11code import read_selected_cases
from .metrics import (
    CODES,
    check_text,
    decode_target,
    paired_bootstrap,
    read_index,
    reject_constant,
    unique_keys,
)
from .train_head import save_atomic

RUN = Path("data/vlm-scale-seoul-8b-fold4-20261128/eval1")
WORK = Path("data/local-eval-predictions/sub78-fixed-verifier")
OUT = Path("results/sub78-8b-verifier-audit-20261128.json")
ADAPTER = Path("data/v3-11code-fold4/adapter")
PROMPT = Path("kwater_lab/system_prompt_v3.txt")


def write(path, value):
    save_atomic(
        path,
        lambda stream: stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()),
    )


def read_seoul_response(path, frozen_candidates, parse, passes):
    manifest_path = WORK / "seoul-packet/input.json"
    projection = load(manifest_path)
    key_map = load(WORK / "seoul-local-key-map.json")
    response = load(path)
    environment = response["environment"]
    input_hash = digest(manifest_path)
    if (
        key_map["input_json_sha256"] != input_hash
        or response["input_sha256"] != input_hash
        or projection["frozen_local_manifest_sha256"] != digest(WORK / "unmeasured.json")
        or response["recipe"] != projection["recipe"]
        or response["gate"] != projection["gate"]
        or response["base_sha256_verified"] != projection["base_sha256"]
        or response["adapter_sha256"] != projection["packet_files"]["adapter/adapter_model.safetensors"]
        or response["system_prompt_sha256"] != projection["packet_files"]["system_prompt.txt"]
        or "L4" not in response["gpu"]
        or environment["os"] != "Linux"
        or environment["dtype"] != "torch.bfloat16"
        or not environment["torch_build"]
        or not environment["cuda_build"]
    ):
        raise ValueError("Returned response provenance differs from the frozen Seoul request")
    candidates = projection["candidates"]
    actual = response["rows"]
    if len(candidates) != len(frozen_candidates) or len(actual) != len(candidates):
        raise ValueError("Incomplete frozen Seoul measurements")
    accepted = {}
    decisions = Counter()
    for projected, original, measured in zip(candidates, frozen_candidates, actual, strict=True):
        key = projected["sample_id"]
        mapped = key_map["keys"][key]
        if (
            measured["key"] != key
            or measured["code"] != projected["code"]
            or mapped["sample_id"] != original["sample_id"]
            or mapped["code"] != original["code"]
            or mapped["image_sha256"] != original["image_sha256"]
            or projection["packet_files"][projected["image"]] != original["image_sha256"]
            or any(
                projected[name] != original[name]
                for name in ("code", "spec_cal_prob", "was_vlm_empty", "spec_margin")
            )
            or type(measured["margin"]) not in (int, float)
            or not math.isfinite(measured["margin"])
        ):
            raise ValueError("Returned candidate mapping, gate context or margin differs")
        margin_pass = passes(projected, measured["margin"], projection["gate"])
        raw = measured["raw_verify_text"]
        if margin_pass != isinstance(raw, str) or (not margin_pass and raw is not None):
            raise ValueError("Raw verifier result must exist exactly for margin passers")
        if not margin_pass:
            decisions["requirement_not_met"] += 1
        elif parse(raw, projected["code"]) is not None:
            accepted[original["sample_id"]] = original["code"]
            decisions["accepted"] += 1
        else:
            decisions[non_accept_state(raw)] += 1
    return accepted, {
        "scope": "Five fresh L4 requests only; earlier accepted/rejected evidence remains the original Windows 4B cache.",
        "response_sha256": digest(path),
        "packet_input_sha256": input_hash,
        "runtime_versions": response["runtime_versions"],
        "gpu": response["gpu"],
        "environment": environment,
        "decisions": dict(decisions),
        "measurements": len(actual),
    }


def non_accept_state(raw):
    try:
        value = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError):
        return "invalid_generation"
    if (
        isinstance(value, dict)
        and value.keys() == {"decision", "reason"}
        and value["decision"] == "reject"
        and isinstance(value["reason"], str)
        and value["reason"].strip()
    ):
        return "rejected"
    return "invalid_generation"


def select_8b_fold4_candidates(rows, draft):
    """Replay the original strict-inner OOF selector on ordered 8B draft outputs."""
    ids = [row["sample_id"] for row in rows]
    scope = {
        "np": np,
        "CODES": CODES,
        "NUM_CODES": len(CODES),
        "Any": Any,
        "math": math,
        "json": json,
        "re": re,
        "check_text": check_text,
        "reject_constant": reject_constant,
        "unique_keys": unique_keys,
    }
    extract(Path("kwater_lab/clean_specialist.py"), {"apply_calibrated_policy"}, scope)
    extract(
        Path("kwater_lab/hybrid_pipeline.py"),
        {
            "compute_calib_code_precisions",
            "usable_draft_hazards",
            "select_booster_candidates",
        },
        scope,
    )
    extract(Path("kwater_lab/fold4_same_verifier.py"), {"passes"}, scope)
    extract(
        Path("kwater_lab/vlm_11code.py"),
        {"parse_verified_hazard", "verified_hazard", "validate_vlm_11code_obj"},
        scope,
    )
    train = {r["sample_id"]: r for r in load(SPLIT) if r["split"] == "dev" and r["dev_fold"] != 4}
    cfg = load(CALIBRATION)["calibration_by_fold"]["4"]
    with (
        np.load(INNER, allow_pickle=False) as inner,
        np.load(OUTER, allow_pickle=False) as outer,
    ):
        order = inner["train_ids"].tolist()
        if len(order) != 791 or set(order) != train.keys():
            raise ValueError("Expected exact 791-photo inner OOF training pool")
        y = np.asarray(
            [[c in train[sid]["codes"] for c in CODES] for sid in order],
            dtype=np.float32,
        )
        prec = scope["compute_calib_code_precisions"](inner["code_scores"], inner["scene_scores"], y, cfg)
        ix = {sid: i for i, sid in enumerate(outer["ids"].tolist())}
        take = [ix[sid] for sid in ids]
        cand = scope["select_booster_candidates"](
            rows,
            list(draft.values()),
            outer["code_probs"][take],
            outer["scene_probs"][take],
            cfg,
            code_precisions=prec,
            margin_boost_when_hazard=0.04,
            margin_boost_when_empty=0.12,
            max_candidates_per_photo=1,
        )
    return cand, scope, prec


def audit(response_path=None):
    w = FROZEN.parent
    cache = load(w / "fixed_verifier_cache.json")
    e = load(RUN / "cache.json")
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    ids = [r["sample_id"] for r in rows]
    if list(e["outputs"]) != ids or e["identity"]["samples"] != [[r["sample_id"], r["image"]] for r in rows]:
        raise ValueError("8B drafts differ from the ordered evaluation cohort")
    if cache["identity"]["proposal_sha256"] != digest(FROZEN) or cache["identity"]["gate_sha256"] != digest(
        MARGIN_GATE
    ):
        raise ValueError("Cache does not match the frozen proposal and margin gate")
    cand, scope, prec = select_8b_fold4_candidates(rows, e["outputs"])
    gold = {r["sample_id"]: set(r["codes"]) for r in rows}
    gate = load(MARGIN_GATE)
    dec = []
    for c in cand:
        key = c["sample_id"] + "|" + c["code"]
        margin = cache["margins"].get(key)
        raw = cache["generations"].get(key)
        if margin is None:
            state = "missing_margin"
        elif not scope["passes"](c, margin, gate):
            state = "requirement_not_met"
        elif raw is None:
            state = "missing_generation"
        elif scope["parse_verified_hazard"](raw, c["code"]) is not None:
            state = "accepted"
        else:
            state = non_accept_state(raw)
        dec.append((c, state, margin, raw))
    ident = cache["identity"]
    if any(digest(Path("kwater_lab") / name) != expected for name, expected in ident["source"].items()):
        raise ValueError("Current verifier source differs from the cached runtime")
    if len(cand) != len({c["sample_id"] for c in cand}):
        raise ValueError("Expected at most one candidate per canonical photo")
    if (
        ident["manifest_sha256"] != digest(DEV)
        or ident["prompt_sha256"] != digest(PROMPT)
        or ident["recipe"] != ["BF16", 393216, 2, 280]
        or ident["adapter_sha256"] != load(Path("results/fold4_v3_clean_benchmark.json"))["v3_adapter_sha256"]
        or ident["adapter_sha256"] != digest(ADAPTER / "adapter_model.safetensors")
        or ident["adapter_config_sha256"] != digest(ADAPTER / "adapter_config.json")
        or load(FROZEN)["fixed_epoch2_gate"] != gate
    ):
        raise ValueError("Frozen 4B runtime identity differs")
    generation = load(RUN / "generation.json")
    model_key = f"bf16:{generation['base_config_sha256']}:{generation['adapter_sha256']}:{ident['prompt_sha256']}:px393216"
    eight_adapter = RUN.parent / "fit/epoch-1/adapter_model.safetensors"
    if (
        digest(eight_adapter) != generation["adapter_sha256"]
        or e["identity"]["model_key"] != model_key
        or generation["photos"] != 175
    ):
        raise ValueError("8B draft cache does not match its generation receipt")
    if len(ids) != 175 or len(set(ids)) != 175 or len({r["group_id"] for r in rows}) != 39:
        raise ValueError("Unexpected canonical cohort")
    image_by_id = {r["sample_id"]: r for r in rows}
    if any(e["identity"]["image_sha256"].get(r["image"]) != r["sha256"] for r in rows):
        raise ValueError("8B cache image hashes differ from canonical Fold 4")
    original_proposals = load(FROZEN)
    measured = {
        f"{c['sample_id']}|{c['code']}": c
        for epoch in ("epoch1", "epoch2")
        for c in original_proposals[epoch]
    }
    if set(measured) != cache["margins"].keys():
        raise ValueError("Unexpected frozen margin cache keys")
    for c, state, margin, raw in dec:
        k = f"{c['sample_id']}|{c['code']}"
        if k in measured and (
            measured[k]["image"] != c["image"]
            or measured[k]["spec_cal_prob"] != c["spec_cal_prob"]
            or not math.isfinite(margin)
        ):
            raise ValueError("Cached proposal differs in image or specialist probability")
    six = V3_EVAL / "v3_epoch1_11code_standalone.jsonl"
    eight_pred = RUN / "predictions.jsonl"

    def pred(path):
        p = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
        if list(p) != ids:
            raise ValueError("Standalone predictions must preserve all canonical IDs in order")
        return p

    previous, standalone = pred(six), pred(eight_pred)
    labels = Path("data/train/train.jsonl")
    actual = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(labels, set(ids))}
    if set(actual) != set(ids) or any(set(actual[sid]) != gold[sid] for sid in ids):
        raise ValueError("Official annotations differ from manifest")
    severe = {
        sid: {code for code, value in hazards.items() if value["risk_level"]["severity"]["score"] == 4}
        for sid, hazards in actual.items()
    }
    fixed = {c["sample_id"]: c["code"] for c, state, *_ in dec if state == "accepted"}
    unknown = [c for c, state, *_ in dec if state in ("missing_margin", "missing_generation")]
    if any(c["code"] in standalone[c["sample_id"]] for c in cand):
        raise ValueError("A proposal would not add a new standalone code")
    if len(unknown) > 10:
        raise ValueError("Unexpectedly many unmeasured candidates")
    mask_path = LOCAL / "fold4-group-exclusions-local.json"
    mask = load(mask_path)
    if mask["canonical_manifest_sha256"] != digest(DEV):
        raise ValueError("Frozen mask identity differs")
    excluded = {r["fold4_group_id"] for r in mask["confirmed_cross_group_links"]}
    groups = {sid: r["group_id"] for sid, r in image_by_id.items()}
    retain = {sid: g for sid, g in groups.items() if g not in excluded}
    sizes = Counter(retain.values())
    largest = min(sizes, key=lambda g: (-sizes[g], g))
    cohorts = {
        "primary_175": groups,
        "sensitivity_107": retain,
        "exploratory_58": {sid: g for sid, g in retain.items() if g != largest},
    }
    if [len(c) for c in cohorts.values()] != [175, 107, 58]:
        raise ValueError("Unexpected panel sizes")
    decisions = [(c["sample_id"], c["code"]) for c in unknown]
    cases = []
    for bits in product((False, True), repeat=len(decisions)):
        additions = fixed | {sid: code for (sid, code), on in zip(decisions, bits, strict=True) if on}
        cases.append(
            {
                sid: codes | ({additions[sid]} if sid in additions else set())
                for sid, codes in standalone.items()
            }
        )

    def select(value, cohort):
        return {sid: value[sid] for sid in cohort}

    panels = {}
    joint_pass = [True] * len(cases)
    for name, cohort in cohorts.items():
        y = select(gold, cohort)
        six_codes, draft_codes = select(previous, cohort), select(standalone, cohort)
        reference = arm_metrics(y, six_codes, cohort, severe)
        draft = arm_metrics(y, draft_codes, cohort, severe)
        obs = [arm_metrics(y, select(p, cohort), cohort, severe) for p in cases]
        active_ref = [c for c in CODES if reference["per_code"][c]["f1"] is not None]

        def fixed_f1(m):
            return mean(m["per_code"][c]["f1"] or 0 for c in active_ref)

        def values(metrics):
            return (
                metrics["macro_f1"],
                fixed_f1(metrics),
                metrics["no_hazard_false_alarms"],
                metrics["severity4"]["complete_misses"],
            )

        meas = list(map(values, obs))
        limits = {
            "normal_max": reference["no_hazard_photos"] // 5,
            "severity4_max": reference["severity4"]["complete_misses"],
            "macro_f1_delta_min": 0.03,
        }
        flags = [
            (
                f1 - reference["macro_f1"] >= 0.03,
                fa <= limits["normal_max"],
                sev <= limits["severity4_max"],
            )
            for f1, _, fa, sev in meas
        ]
        if name != "exploratory_58":
            joint_pass = [before and all(bits) for before, bits in zip(joint_pass, flags, strict=True)]
        exploratory_target = {
            "macro_f1_ge_065": sum(f1 >= 0.65 for f1, _, _, _ in meas),
            "normal_fpr_le_18pct": sum(
                100 * fa <= 18 * reference["no_hazard_photos"] for _, _, fa, _ in meas
            ),
        }
        passing = {
            "f1_gain": sum(x[0] for x in flags),
            "normal_fpr": sum(x[1] for x in flags),
            "severity4": sum(x[2] for x in flags),
            "all": sum(all(x) for x in flags),
        }
        fields = (
            "macro_f1",
            "macro_f1_fixed_sixth_codes",
            "normal_false_alarms",
            "severity4_complete_misses",
        )
        panels[name] = {
            "photos": len(cohort),
            "groups": len(set(cohort.values())),
            "candidate_outcomes": dict(Counter(state for c, state, *_ in dec if c["sample_id"] in cohort)),
            "sixth_active_codes": active_ref,
            "all_assignments_same_active_codes_as_sixth": all(
                [c for c in CODES if m["per_code"][c]["f1"] is not None] == active_ref for m in obs
            ),
            "sixth": reference,
            "eight_b_standalone": draft,
            "unmeasured_omitted": obs[0],
            "unmeasured_omitted_vs_sixth": paired_bootstrap(y, six_codes, select(cases[0], cohort), cohort),
            "unmeasured_omitted_vs_eight_b": paired_bootstrap(
                y, draft_codes, select(cases[0], cohort), cohort
            ),
            "exact_assignment_ranges": {f: [min(v), max(v)] for f, v in zip(fields, zip(*meas), strict=True)},
            "observed_gate_count_over_assignments": passing,
            "separate_next_phase_target_count_over_assignments": exploratory_target,
            "gate": limits,
        }
    sources = [
        DEV,
        SPLIT,
        INNER,
        OUTER,
        CALIBRATION,
        FROZEN,
        MARGIN_GATE,
        PROMPT,
        mask_path,
        labels,
        six,
        eight_pred,
        RUN / "cache.json",
        RUN / "generation.json",
        w / "fixed_verifier_cache.json",
    ]
    hashes = {p.as_posix(): digest(p) for p in sources}
    frozen_input = {
        "source_sha256": hashes,
        "verifier_identity": ident,
        "gate": gate,
        "unknown": [{**c, "image_sha256": image_by_id[c["sample_id"]]["sha256"]} for c in unknown],
    }
    if response_path is None:
        write(WORK / "unmeasured.json", frozen_input)
    elif load(WORK / "unmeasured.json") != frozen_input:
        raise ValueError("Do not merge responses into changed frozen candidates")
    result = {
        "scope": "CPU replay combining actual 8B epoch-1 drafts and measured fixed 4B epoch-2 verifier responses. Unknown outcomes are enumerated, not GPU predictions.",
        "input_sha256": hashes,
        "code_precision_inner_oof_791": prec,
        "fixed_verifier_identity": ident,
        "eight_b_generation_identity": model_key,
        "eight_b_verifier_used": False,
        "eight_b_draft_was_trained_on_mixed_curriculum": True,
        "gate": gate,
        "proposal_policy": {"hazard_margin_boost": 0.04, "empty_margin_boost": 0.12, "max_per_photo": 1},
        "cached_margins_use_current_8b_draft_empty_status_for_gate": True,
        "unmeasured_requirements": "Each lacks a fixed 4B margin; only margin passers need fresh 4B verifier generation.",
        "bounds_note": "Omitting all missing candidates is a scenario. The exact F1 minimum may differ.",
        "cache_replay_is_not_a_new_complete_hybrid_generation": True,
        "candidate_count": len(cand),
        "cached_margin_count": sum(m is not None for _, _, m, _ in dec),
        "candidate_outcomes": dict(Counter(state for _, state, *_ in dec)),
        "candidate_gold_positive_by_outcome": {
            k: sum(c["code"] in gold[c["sample_id"]] for c, s, *_ in dec if s == k)
            for k in sorted({s for _, s, *_ in dec})
        },
        "unmeasured_count": len(unknown),
        "unmeasured_frozen_manifest_sha256": digest(WORK / "unmeasured.json"),
        "possible_boolean_assignments": len(cases),
        "assignments_passing_joint_175_and_107": sum(joint_pass),
        "panels": panels,
    }
    if response_path is not None:
        addition, receipt = read_seoul_response(
            response_path, frozen_input["unknown"], scope["parse_verified_hazard"], scope["passes"]
        )
        measured = {
            sid: codes | ({addition[sid]} if sid in addition else set()) for sid, codes in cases[0].items()
        }
        for name, cohort in cohorts.items():
            metrics = arm_metrics(select(gold, cohort), select(measured, cohort), cohort, severe)
            panel = result["panels"][name]
            panel["measured_missing_merged"] = metrics
            panel["measured_missing_merged_gates"] = {
                "macro_f1_gain_ge_003": metrics["macro_f1"] - panel["sixth"]["macro_f1"] >= 0.03,
                "normal_fpr_le_20pct": 5 * metrics["no_hazard_false_alarms"] <= metrics["no_hazard_photos"],
                "severity4_not_increased": (
                    metrics["severity4"]["complete_misses"] <= panel["sixth"]["severity4"]["complete_misses"]
                ),
                "next_phase_f1_ge_065": metrics["macro_f1"] >= 0.65,
                "next_phase_normal_fpr_le_18pct": (
                    100 * metrics["no_hazard_false_alarms"] <= 18 * metrics["no_hazard_photos"]
                ),
            }
            result["panels"][name]["measured_missing_merged_vs_sixth"] = paired_bootstrap(
                select(gold, cohort),
                select(previous, cohort),
                select(measured, cohort),
                cohort,
            )
        result["seoul_supplement"] = receipt
    write(OUT, result)
    return result


def measure_missing():
    """Run only the frozen missing requests on the original 4B GPU holder."""
    from importlib.metadata import version

    manifest_path = WORK / "unmeasured.json"
    frozen = load(manifest_path)
    for path, expected in frozen["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    identity = frozen["verifier_identity"]
    if any(version(package) != expected for package, expected in identity["runtime"].items()):
        raise ValueError("Use the original fixed verifier runtime")
    if (
        digest(ADAPTER / "adapter_model.safetensors") != identity["adapter_sha256"]
        or digest(ADAPTER / "adapter_config.json") != identity["adapter_config_sha256"]
        or digest(PROMPT) != identity["prompt_sha256"]
    ):
        raise ValueError("Use the same final 4B verifier adapter/configuration and prompt")
    if any(digest(Path("kwater_lab") / name) != sha for name, sha in identity["source"].items()):
        raise ValueError("Verifier source differs from the frozen cache")
    destination = WORK / "measured_missing.json"
    if destination.exists():
        raise FileExistsError("Preserve the prior independent measurements")
    image_root = Path("data/train")
    candidates = frozen["unknown"]
    for row in candidates:
        path = image_root / row["image"]
        if not path.resolve().is_relative_to(image_root.resolve()) or digest(path) != row["image_sha256"]:
            raise ValueError("Candidate image differs from the frozen manifest")
    from .fold4_same_verifier import passes
    from .v3_turnaround import _resolve_base_model_dir, score_verifier_1token_margins
    from .vlm_11code import (
        load_qwen_4bit_or_bf16,
        seed_all,
        verify_specialist_candidates_with_vlm,
    )

    base = _resolve_base_model_dir(Path.cwd())
    base_stats = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in base.iterdir() if p.is_file()}
    if base_stats != identity["base_files"]:
        raise ValueError("Local base file metadata differs from the cached fixed verifier")
    precision, pixels, batch, cap = identity["recipe"]
    if precision != "BF16":
        raise ValueError("This frozen verifier uses BF16")
    seed_all(42)
    model, proc = load_qwen_4bit_or_bf16(base, pixels, False, ADAPTER)
    prompt = PROMPT.read_text("utf-8")
    margins = score_verifier_1token_margins(model, proc, candidates, image_root, prompt, batch_size=batch)
    passed = [
        row for row, margin in zip(candidates, margins, strict=True) if passes(row, margin, frozen["gate"])
    ]
    generated = verify_specialist_candidates_with_vlm(
        model,
        proc,
        passed,
        image_root,
        prompt,
        batch_size=batch,
        max_new_tokens=cap,
    )

    def key(row):
        return f"{row['sample_id']}|{row['code']}"

    write(
        destination,
        {
            "frozen_input_sha256": digest(manifest_path),
            "verifier_identity": identity,
            "batch_layout": "Independent batches containing only previously unmeasured requests.",
            "margins": {key(row): margin for row, margin in zip(candidates, margins, strict=True)},
            "generations": {key(row): row["raw_verify_text"] for row in generated},
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("audit", "measure-missing"), nargs="?", default="audit")
    cli.add_argument("--seoul-response", type=Path)
    args = cli.parse_args()
    if args.stage == "audit":
        audit(args.seoul_response)
    else:
        if args.seoul_response is not None:
            cli.error("--seoul-response applies to audit only")
        measure_missing()
