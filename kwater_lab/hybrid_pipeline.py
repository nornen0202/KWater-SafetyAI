"""End-to-end Fold-0 & Sealed-Holdout Evaluator for Standalone VLM, Vision Specialist, and Hybrid Verifier.

Strictly separates and records:
  1. Raw VLM generation parse rate & strict schema validity rate.
  2. Official competition schema (`prediction.schema.json` / `decode_target`) validity rate.
  3. Ground-truth Active-Code Macro-F1, quantitative points (/70), per-code TP/FP/FN,
     no-hazard false alarm rate, complete hazard miss rate, severity-4 miss rate,
     and 2,000-resample incident-group paired bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any

from jsonschema import Draft202012Validator
import numpy as np
import torch

from .audit import digest
from .clean_specialist import SpatialDualSpecialistHead, apply_calibrated_policy, predict_probs
from .fold0_production_compare import EXPECTED_FOLD0_PHOTOS, generation_conditions, image_id
from .law_reranker_11code import Native11LawRetriever, read_selected_cases
from .metrics import CODES, decode_target, evaluate, paired_bootstrap, read_index
from .train_head import save_atomic
from .verifier_cache import verify_with_cache
from .vlm_11code import (
    CANDIDATE_MAX_PIXELS,
    INSPECT_MAX_NEW_TOKENS,
    VERIFY_MAX_NEW_TOKENS,
    format_competition_record,
    infer_batched_vlm,
    load_qwen_model,
    load_qwen_processor,
    verify_specialist_candidates_with_vlm,
)


@dataclass(frozen=True)
class Fold0Followup:
    primary_summary: Path
    official_predictions: Path
    official_summary: Path
    paired_summary: Path


def verify_followup_pair(paths, eval_ids, primary_prediction_path):
    paired = json.loads(paths.paired_summary.read_text(encoding="utf-8"))
    conditions = generation_conditions(paths.official_summary, paths.primary_summary,
                                       paths.official_predictions, primary_prediction_path)
    official = json.loads(paths.official_summary.read_text(encoding="utf-8"))
    if (paired.get("photos") != EXPECTED_FOLD0_PHOTOS or len(eval_ids) != EXPECTED_FOLD0_PHOTOS
        or official.get("official_checker_valid_rows") != EXPECTED_FOLD0_PHOTOS
        or paired.get("generation_conditions") != conditions):
        raise ValueError("Finish the full, fingerprinted 216-photo paired comparison before verification")
    baseline = read_index(paths.official_predictions, decode_target, limit_size=True)
    keys = {image_id(sid): sid for sid in eval_ids}
    if baseline.keys() != keys.keys():
        raise ValueError("Official comparison predictions do not match the verification cohort")
    return {keys[anon]: codes for anon, codes in baseline.items()}, conditions


def count_added_predictions(gold, before, after):
    if gold.keys() != before.keys() or before.keys() != after.keys():
        raise ValueError("Verification must cover the same photos as the primary prediction")
    true_positives = 0
    false_positives = 0
    new_false_alarm_photos = 0
    for sid, prior in before.items():
        if not prior <= after[sid]:
            raise ValueError("Verification removed a primary hazard prediction")
        additions = after[sid] - prior
        true_positives += len(additions & gold[sid])
        false_positives += len(additions - gold[sid])
        new_false_alarm_photos += int(not gold[sid] and not prior and bool(additions))
    return {"new_true_positive_codes": true_positives, "new_false_positive_codes": false_positives,
            "new_false_alarm_photos": new_false_alarm_photos}


def severity4_miss_stats(gold_rows_by_id: dict[str, dict], preds_codes: dict[str, frozenset[str]]) -> dict[str, Any]:
    """Track recall specifically on fatal/catastrophic (severity_score == 4) hazards."""
    sev4_photos = 0
    sev4_photo_completely_missed = 0
    sev4_photo_without_any_prediction = 0
    sev4_instances = 0
    sev4_instance_missed = 0

    for sid, pr in preds_codes.items():
        t = gold_rows_by_id[sid]["target"]
        if t["status"] != "hazard":
            continue
        s4_codes = [
            c for c, h in t["hazards"].items()
            if int(h["risk_level"]["severity"]["score"]) == 4
        ]
        if s4_codes:
            sev4_photos += 1
            if not pr.intersection(s4_codes):
                sev4_photo_completely_missed += 1
            if not pr:
                sev4_photo_without_any_prediction += 1
            for c in s4_codes:
                sev4_instances += 1
                if c not in pr:
                    sev4_instance_missed += 1

    return {
        "sev4_photos": sev4_photos,
        "sev4_photo_completely_missed": sev4_photo_completely_missed,
        "sev4_photo_without_any_prediction": sev4_photo_without_any_prediction,
        "sev4_instances": sev4_instances,
        "sev4_instance_missed": sev4_instance_missed,
        "sev4_instance_recall": round(1.0 - sev4_instance_missed / max(sev4_instances, 1), 4),
    }


def compute_calib_code_precisions(
    oof_code_probs: np.ndarray,
    oof_scene_probs: np.ndarray,
    y_codes: np.ndarray,
    calib: dict[str, Any],
) -> dict[str, float]:
    """Compute leak-free per-code precision on the calibration OOF split."""
    preds = apply_calibrated_policy(oof_code_probs, oof_scene_probs, calib)
    prec: dict[str, float] = {}
    for c_idx, code in enumerate(CODES):
        tp = sum((y_codes[i, c_idx] > 0.5) and (code in preds[i]) for i in range(len(preds)))
        pr = sum(code in preds[i] for i in range(len(preds)))
        prec[code] = float(tp / pr) if pr >= 3 else 0.0
    return prec


def usable_draft_hazards(draft: dict | None) -> dict[str, dict]:
    """Recover the usable hazards recognized by the submission formatter."""
    if not isinstance(draft, dict) or not isinstance(draft.get("hazards"), list):
        return {}
    if draft.get("status", draft.get("scene_state")) not in (None, "hazard"):
        return {}
    by_code = {}
    for hazard in draft["hazards"]:
        if isinstance(hazard, dict):
            code = str(hazard.get("code") or "").strip()
            if code in CODES and code not in by_code:
                by_code[code] = hazard
    return by_code


def select_booster_candidates(
    eval_rows: list[dict],
    vlm_outputs: list[dict[str, Any]],
    spec_code_probs: np.ndarray,
    spec_scene_probs: np.ndarray,
    calib: dict[str, Any],
    code_precisions: dict[str, float] | None = None,
    margin_boost_when_hazard: float = 0.04,
    margin_boost_when_empty: float = 0.12,
    max_candidates_per_photo: int = 2,
) -> list[dict[str, Any]]:
    """Propose missed hazard codes from high-precision Dual Vision Specialist classes for VLM verification."""
    if len(eval_rows) != len(vlm_outputs) or spec_code_probs.shape != (len(eval_rows), len(CODES)) or spec_scene_probs.shape != (len(eval_rows),):
        raise ValueError("Specialist inputs and VLM outputs must align with evaluation rows")
    temps = np.array(calib["temperatures"], dtype=np.float64)
    logits = np.log(np.clip(spec_code_probs, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(spec_code_probs, 1e-6, 1.0 - 1e-6)))
    cal_probs = 1.0 / (1.0 + np.exp(-logits / temps[None, :]))
    taus = np.array(calib["thresholds"], dtype=np.float64)
    tau_scene = float(calib["scene_threshold"])

    candidates: list[dict[str, Any]] = []
    for i, (r, v_out) in enumerate(zip(eval_rows, vlm_outputs)):
        existing_codes = usable_draft_hazards(v_out.get("parsed"))
        if len(existing_codes) >= 3:
            continue

        is_vlm_empty = not existing_codes
        base_req_margin = margin_boost_when_empty if is_vlm_empty else margin_boost_when_hazard
        if is_vlm_empty and spec_scene_probs[i] < tau_scene + 0.08:
            continue

        margins = cal_probs[i] - taus
        eligible_c = []
        for c in np.argsort(-margins):
            code_name = CODES[c]
            if code_name in existing_codes:
                continue
            # Require stronger margin if calibration-split precision is below 0.50 (e.g., struck_by_lifting)
            extra_pen = 0.0
            if code_precisions is not None:
                p_c = code_precisions.get(code_name, 0.60)
                if p_c < 0.35:
                    continue
                if p_c < 0.55:
                    extra_pen = 0.08
            if margins[c] >= base_req_margin + extra_pen:
                eligible_c.append(c)

        slots = min(max_candidates_per_photo, 3 - len(existing_codes))
        for c in eligible_c[:slots]:
            candidates.append({
                "row_idx": i,
                "sample_id": r["sample_id"],
                "image": r["image"],
                "code": CODES[c],
                "spec_cal_prob": round(float(cal_probs[i, c]), 4),
                "spec_margin": round(float(margins[c]), 4),
                "was_vlm_empty": is_vlm_empty,
            })
    return candidates


def merge_verified_candidates(
    vlm_outputs: list[dict[str, Any]],
    verified: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = copy.deepcopy(vlm_outputs)
    by_idx: dict[int, list[dict[str, Any]]] = {}
    for v in verified:
        if v.get("accepted") is True and isinstance(v.get("hazard"), dict) and v["hazard"].get("code") == v.get("code"):
            by_idx.setdefault(int(v["row_idx"]), []).append(v["hazard"])

    for idx, new_hazards in by_idx.items():
        cur = merged[idx].get("parsed")
        if not isinstance(cur, dict):
            cur = {"scene_note": "현장 작업 구간에서 안전 위험요인이 확인된다.", "status": "hazard", "hazards": []}
        hazards = usable_draft_hazards(cur)
        had_draft_hazards = bool(hazards)
        for nh in new_hazards:
            if nh.get("code") in CODES and nh["code"] not in hazards and len(hazards) < 3:
                hazards[nh["code"]] = nh
        if hazards:
            if not had_draft_hazards:
                cur["scene_note"] = next(iter(hazards.values()))["evidence"]
            cur["status"] = "hazard"
            cur["hazards"] = list(hazards.values())[:3]
        merged[idx]["parsed"] = cur
    return merged


def validate_official_records(records, rows, validator):
    """Use temporary hex IDs for the official schema and keep local labels for metrics."""
    predictions = {}
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        image_id, codes = decode_target(record, local_ids=True)
        if image_id != row["sample_id"] or image_id in predictions:
            raise ValueError("Formatted predictions must match the ordered evaluation IDs")
        validator.validate({**record, "image_id": f"{index:032x}"})
        predictions[image_id] = codes
    return predictions


def evaluate_split(
    root: Path,
    manifest_path: Path,
    prompt_path: Path,
    specialist_dir: Path,
    adapter_dir: Path,
    out_summary_path: Path,
    split_mode: str = "fold0",  # "fold0" or "holdout"
    run_zeroshot_base: bool = True,
    max_pixels_infer: int = CANDIDATE_MAX_PIXELS,
    prediction_dir: Path | None = None,
    use_4bit: bool = True,
    batch_size: int = 4,
    run_verifier: bool = True,
    followup: Fold0Followup | None = None,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if followup and (split_mode != "fold0" or not run_verifier or out_summary_path.resolve() == followup.primary_summary.resolve()):
        raise ValueError("Use a separate Fold0 verification report for the paired follow-up")
    expected_fold = {"fold0": 0, "holdout": -1}.get(split_mode)
    if expected_fold is None:
        raise ValueError(f"Unknown split_mode: {split_mode}")
    training_summary = json.loads((adapter_dir.parent / "training_summary.json").read_text(encoding="utf-8"))
    adapter_sha256 = digest(adapter_dir / "adapter_model.safetensors")
    if (training_summary.get("val_fold") != expected_fold
        or training_summary.get("adapter_safetensors_sha256") != adapter_sha256
        or not training_summary.get("epochs")
        or len(training_summary.get("history", ())) != training_summary["epochs"]):
        raise ValueError("Completed adapter training provenance must match the evaluation split and weights")
    training_prompt = training_summary.get("configuration", {}).get("prompt_sha256")
    if training_prompt is not None and training_prompt != digest(prompt_path):
        raise ValueError("Evaluation prompt differs from the adapter training prompt")
    local_pred_dir = (prediction_dir or root / "data/local-eval-predictions" / split_mode).resolve()
    if not local_pred_dir.is_relative_to((root / "data").resolve()):
        raise ValueError("Raw and formatted evaluation outputs must remain under data/")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL evaluation requires a CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    laws = [
        json.loads(line)
        for line in (root / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if split_mode == "fold0":
        eval_rows = [r for r in manifest if r["split"] == "dev" and int(r["dev_fold"]) == 0]
        train_rows = [r for r in manifest if r["split"] == "dev" and int(r["dev_fold"]) != 0]
    elif split_mode == "holdout":
        eval_rows = [r for r in manifest if r["split"] == "holdout"]
        train_rows = [r for r in manifest if r["split"] == "dev"]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")
    if not eval_rows or len(train_rows) != training_summary.get("train_photos"):
        raise ValueError("Evaluation or training population differs from the adapter provenance")
    raw_train_by_id = {r["sample_id"]: r for r in read_selected_cases(
        root / "data/train/train.jsonl", {r["sample_id"] for r in train_rows + eval_rows}
    )}
    train_law_rows = [raw_train_by_id[r["sample_id"]] for r in train_rows]

    law_retriever = Native11LawRetriever(laws, train_law_rows)
    validator = Draft202012Validator(json.loads(
        (root / "data/local-kit/kit/prediction.schema.json").read_text(encoding="utf-8")
    ))
    gold_codes = {r["sample_id"]: frozenset(r["codes"]) for r in eval_rows}
    group_map = {r["sample_id"]: r["group_id"] for r in eval_rows}
    if followup:
        official_preds, generation = verify_followup_pair(followup, gold_codes, local_pred_dir / "ft_vlm_11code.jsonl")
        primary_summary = json.loads(followup.primary_summary.read_text(encoding="utf-8"))
        if primary_summary["manifest_sha256"] != digest(manifest_path) or primary_summary["adapter_sha256"] != adapter_sha256:
            raise ValueError("Follow-up manifest or adapter differs from the completed primary run")

    # The supplied fold-0 calibration uses OOF models trained on fold 0. Use fixed
    # thresholds for fold 0 until there is calibration built wholly outside the fold.
    feat_arc = np.load(specialist_dir / "features_multiview_2176d.npz", allow_pickle=False)
    feat_id_to_idx = {sid: i for i, sid in enumerate(feat_arc["ids"].tolist())}
    x_eval = feat_arc["features"][[feat_id_to_idx[r["sample_id"]] for r in eval_rows]]

    spec_ckpt = torch.load(specialist_dir / "specialist_5fold_ensemble.pt", map_location="cpu", weights_only=True)
    if tuple(spec_ckpt["codes"]) != CODES:
        raise ValueError("Specialist code order differs from the competition")
    if split_mode == "fold0":
        state_list = spec_ckpt["fold_models"]["0"]
        spec_calib = {"temperatures": [1.0] * len(CODES), "thresholds": [0.5] * len(CODES), "scene_threshold": 0.5}
        code_precs = None
    else:
        state_list = [st for f in range(5) for st in spec_ckpt["fold_models"][str(f)]]
        spec_calib = spec_ckpt["full_dev_calibration"]
        with np.load(specialist_dir / "dev_oof_predictions.npz", allow_pickle=False) as oof_arc:
            code_precs = compute_calib_code_precisions(
                oof_arc["code_probs"], oof_arc["scene_probs"], oof_arc["y_codes"], spec_calib
            )

    spec_models = []
    for st in state_list:
        m = SpatialDualSpecialistHead().to(device)
        m.load_state_dict(st)
        m.eval()
        spec_models.append(m)
    spec_code_p, spec_scene_p = predict_probs(spec_models, x_eval, device)
    del spec_models
    torch.cuda.empty_cache()

    spec_pred_list = apply_calibrated_policy(spec_code_p, spec_scene_p, spec_calib)
    spec_preds = {r["sample_id"]: pr for r, pr in zip(eval_rows, spec_pred_list)}
    spec_eval = evaluate(gold_codes, spec_preds, group_map)
    spec_sev4 = severity4_miss_stats(raw_train_by_id, spec_preds)

    system_prompt = prompt_path.read_text(encoding="utf-8")
    image_root = root / "data/train"
    base_model_dir = root / "data/baseline/base_model"
    base_cache_parts = {}
    for path in sorted(base_model_dir.iterdir()):
        if path.is_file():
            if path.suffix == ".safetensors":
                details = path.stat()
                base_cache_parts[path.name] = [details.st_size, details.st_mtime_ns]
            else:
                base_cache_parts[path.name] = digest(path)
    inference_packages = ("torch", "transformers", "peft") + (("bitsandbytes",) if use_4bit else ())
    cache_base_identity = {
        "base": base_cache_parts,
        "device": torch.cuda.get_device_name(device),
        "packages": {p: importlib.metadata.version(p) for p in inference_packages},
        "torch_cuda_version": torch.version.cuda,
        "inference_source_sha256": digest(Path(__file__).with_name("vlm_11code.py")),
        "use_4bit": use_4bit,
    }

    def cache_key(adapter_sha=None):
        identity = {**cache_base_identity, "adapter_sha256": adapter_sha}
        if adapter_sha:
            identity["adapter_config_sha256"] = digest(adapter_dir / "adapter_config.json")
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    results_out: dict[str, Any] = {
        "split_mode": split_mode,
        "photos": len(eval_rows),
        "groups": len(set(group_map.values())),
        "manifest_sha256": digest(manifest_path),
        "adapter_sha256": adapter_sha256,
        "qlora_inference_weight_mode": "NF4 language / BF16 vision" if use_4bit else "BF16 base",
        "vlm_batch_size": batch_size,
        "vlm_max_pixels": max_pixels_infer,
        "inspect_max_new_tokens": INSPECT_MAX_NEW_TOKENS,
        "verify_max_new_tokens": VERIFY_MAX_NEW_TOKENS,
        "specialist_standalone": {
            "calibration": "unfitted_fixed_0.5" if split_mode == "fold0" else "development_oof",
            "metrics": spec_eval,
            "severity4": spec_sev4,
        },
    }
    local_pred_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions(filename, rows, *, raw=False):
        save_atomic(
            local_pred_dir / filename,
            lambda f: f.write("".join(json.dumps(x, ensure_ascii=raw) + "\n" for x in rows).encode("utf-8")),
        )

    def write_summary():
        results_out["elapsed_seconds"] = round(time.perf_counter() - eval_start, 2)
        results_out["peak_gpu_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_atomic(out_summary_path, lambda f: f.write((json.dumps(results_out, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))

    # Optional Zero-Shot Base Qwen3-VL-4B-Instruct control (100% leak-free base VLM benchmark)
    processor = load_qwen_processor(base_model_dir, max_pixels_infer)
    zs_preds: dict[str, frozenset[str]] | None = None
    zero_cache_path = local_pred_dir / "zeroshot_vlm_raw_cache.json"

    def disabled_adapter_zero_key():
        return hashlib.sha256(f"{cache_key(adapter_sha256)}:disabled-adapter".encode()).hexdigest()

    def evaluate_zero_shot(model_factory, *, disabled_adapter=False):
        print(f"Running Zero-Shot Base Qwen3-VL-4B-Instruct on {split_mode} (n={len(eval_rows)})...", flush=True)
        zs_raw, zs_gen = infer_batched_vlm(
            None, processor, eval_rows, image_root, system_prompt, batch_size=batch_size,
            cache_path=zero_cache_path, cache_key=disabled_adapter_zero_key() if disabled_adapter else cache_key(),
            model_factory=model_factory,
        )
        torch.cuda.empty_cache()

        zs_recs = [format_competition_record(r["sample_id"], o["parsed"], law_retriever) for r, o in zip(eval_rows, zs_raw)]
        save_predictions("zeroshot_raw.jsonl", zs_raw, raw=True)
        save_predictions("zeroshot_formatted.jsonl", zs_recs)
        zero_predictions = validate_official_records(zs_recs, eval_rows, validator)
        results_out["zeroshot_base_vlm_11code"] = {
            "generation": zs_gen,
            "kit_schema_valid_rate": 1.0,
            "metrics": evaluate(gold_codes, zero_predictions, group_map),
            "severity4": severity4_miss_stats(raw_train_by_id, zero_predictions),
        }
        return zero_predictions

    def zero_shot_model():
        return load_qwen_model(base_model_dir, use_4bit=use_4bit)

    if run_zeroshot_base and not followup:
        zs_preds = evaluate_zero_shot(zero_shot_model)

    # Fine-tuned 11-Code Multimodal LoRA VLM + Hybrid Visual Candidate Verification
    print(f"Running Fine-Tuned 11-Code Qwen3-VL-4B LoRA ({adapter_dir}) on {split_mode} (n={len(eval_rows)})...", flush=True)
    @cache
    def finetuned_model():
        return load_qwen_model(base_model_dir, use_4bit=use_4bit, adapter_dir=adapter_dir)

    def missing_primary_cache():
        raise ValueError("The full, matching primary 216-photo generation cache is required for the follow-up")

    ft_raw, ft_gen = infer_batched_vlm(
        None, processor, eval_rows, image_root, system_prompt, batch_size=batch_size,
        cache_path=local_pred_dir / "ft_vlm_raw_cache.json", cache_key=cache_key(adapter_sha256),
        model_factory=missing_primary_cache if followup else finetuned_model,
    )

    ft_recs = [format_competition_record(r["sample_id"], o["parsed"], law_retriever) for r, o in zip(eval_rows, ft_raw)]
    if followup:
        saved = [json.loads(line) for line in (local_pred_dir / "ft_vlm_11code.jsonl").read_text(encoding="utf-8").splitlines()]
        if saved != ft_recs:
            raise ValueError("Formatted primary predictions differ from their cached generations")
        results_out["generation_conditions"] = generation
    else:
        save_predictions("ft_vlm_raw.jsonl", ft_raw, raw=True)
        save_predictions("ft_vlm_11code.jsonl", ft_recs)
    results_out["formatted_output_sha256"] = digest(local_pred_dir / "ft_vlm_11code.jsonl")
    ft_preds = validate_official_records(ft_recs, eval_rows, validator)
    ft_eval = evaluate(gold_codes, ft_preds, group_map)
    ft_sev4 = severity4_miss_stats(raw_train_by_id, ft_preds)
    results_out["finetuned_vlm_11code_standalone"] = {
        "generation": ft_gen,
        "kit_schema_valid_rate": 1.0,
        "metrics": ft_eval,
        "severity4": ft_sev4,
    }
    comparisons = {
        "ft_vlm_vs_specialist": paired_bootstrap(gold_codes, spec_preds, ft_preds, group_map, resamples=2000, seed=42),
    }
    if zs_preds is not None:
        comparisons["ft_vlm_vs_zeroshot_base"] = paired_bootstrap(
            gold_codes, zs_preds, ft_preds, group_map, resamples=2000, seed=42
        )
    results_out["paired_bootstrap_comparisons"] = comparisons
    results_out["hybrid_verification_status"] = "pending" if run_verifier else "skipped"
    if not followup:
        write_summary()
    if not run_verifier:
        finetuned_model.cache_clear()
        return results_out

    # Propose high-confidence missed hazards from Specialist -> VLM Visual Verification
    t_ver = time.perf_counter()
    max_candidates = 1 if split_mode == "fold0" else 2
    cands = select_booster_candidates(
        eval_rows, ft_raw, spec_code_p, spec_scene_p, spec_calib, code_precisions=code_precs,
        max_candidates_per_photo=max_candidates,
    )
    ver_identity = {
        "model_key": cache_key(adapter_sha256), "manifest_sha256": digest(manifest_path),
        "primary_sha256": results_out["formatted_output_sha256"], "prompt_sha256": digest(prompt_path),
        "policy_source_sha256": digest(Path(__file__)), "specialist_sha256": digest(specialist_dir / "specialist_5fold_ensemble.pt"),
        "selected_images_sha256": {c["image"]: digest(image_root / c["image"]) for c in cands},
        "processor_size": dict(processor.image_processor.size), "max_new_tokens": VERIFY_MAX_NEW_TOKENS,
    }
    verified, ver_cache = verify_with_cache(cands, lambda batch: verify_specialist_candidates_with_vlm(
        finetuned_model(), processor, batch, image_root, system_prompt, batch_size=batch_size
    ), local_pred_dir / "specialist_verification_cache.json", ver_identity, batch_size)
    save_predictions("specialist_verification_raw.jsonl", verified, raw=True)
    ver_sec = round(time.perf_counter() - t_ver, 2)

    hyb_raw = merge_verified_candidates(ft_raw, verified)
    hyb_recs = [format_competition_record(r["sample_id"], o["parsed"], law_retriever) for r, o in zip(eval_rows, hyb_raw)]
    save_predictions("hybrid_vlm_verified_specialist.jsonl", hyb_recs)
    results_out["hybrid_output_sha256"] = digest(local_pred_dir / "hybrid_vlm_verified_specialist.jsonl")
    hyb_preds = validate_official_records(hyb_recs, eval_rows, validator)
    hyb_eval = evaluate(gold_codes, hyb_preds, group_map)
    hyb_sev4 = severity4_miss_stats(raw_train_by_id, hyb_preds)

    results_out["hybrid_verification_status"] = "completed"
    results_out["hybrid_vlm_plus_verified_specialist"] = {
        "proposed_candidates": len(cands),
        "max_proposals_per_photo": max_candidates,
        "vlm_accepted_candidates": sum(bool(v.get("accepted")) for v in verified),
        "vlm_rejected_candidates": sum(not bool(v.get("accepted")) for v in verified),
        "verification_seconds": ver_sec,
        **ver_cache,
        **count_added_predictions(gold_codes, ft_preds, hyb_preds),
        "kit_schema_valid_rate": 1.0,
        "metrics": hyb_eval,
        "severity4": hyb_sev4,
    }
    comparisons.update({
        "hybrid_vs_ft_vlm": paired_bootstrap(gold_codes, ft_preds, hyb_preds, group_map, resamples=2000, seed=42),
        "hybrid_vs_specialist": paired_bootstrap(gold_codes, spec_preds, hyb_preds, group_map, resamples=2000, seed=42),
    })
    if followup:
        comparisons["hybrid_vs_official_lora"] = paired_bootstrap(
            gold_codes, official_preds, hyb_preds, group_map, resamples=2000, seed=42
        )
        results_out["official_production_lora"] = {"metrics": evaluate(gold_codes, official_preds, group_map),
                                                   "severity4": severity4_miss_stats(raw_train_by_id, official_preds)}
        results_out["followup_complete"] = not run_zeroshot_base
        write_summary()
        if run_zeroshot_base:
            if zero_cache_path.exists():
                prior_zero_key = json.loads(zero_cache_path.read_text(encoding="utf-8"))["identity"]["model_key"]
                reuse_resident = prior_zero_key == disabled_adapter_zero_key()
            else:
                reuse_resident = bool(finetuned_model.cache_info().currsize)
            if reuse_resident:
                with torch.inference_mode(), finetuned_model().disable_adapter():
                    zs_preds = evaluate_zero_shot(finetuned_model, disabled_adapter=True)
                results_out["zeroshot_weight_source"] = "resident base with PEFT adapter disabled"
            else:
                finetuned_model.cache_clear()
                torch.cuda.empty_cache()
                zs_preds = evaluate_zero_shot(zero_shot_model)
                results_out["zeroshot_weight_source"] = "standalone base or complete zero-shot cache"
            comparisons["ft_vlm_vs_zeroshot_base"] = paired_bootstrap(
                gold_codes, zs_preds, ft_preds, group_map, resamples=2000, seed=42
            )
            results_out["followup_complete"] = True
    finetuned_model.cache_clear()
    torch.cuda.empty_cache()
    if zs_preds is not None:
        comparisons["hybrid_vs_zeroshot_base"] = paired_bootstrap(
            gold_codes, zs_preds, hyb_preds, group_map, resamples=2000, seed=42
        )

    write_summary()
    return results_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["fold0", "holdout"], default="fold0")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompt", type=Path, default=Path("kwater_lab/system_prompt_11code.txt"))
    parser.add_argument("--specialist-dir", type=Path, default=Path("data/clean-specialist-v1"))
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--bf16", action="store_true", help="Use the submission entrypoint's unquantized BF16 base weights")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-verifier", action="store_true")
    parser.add_argument("--paired-followup", nargs=4, type=Path,
                        metavar=("PRIMARY_SUMMARY", "OFFICIAL_PREDICTIONS", "OFFICIAL_SUMMARY", "PAIRED_SUMMARY"))
    args = parser.parse_args()

    manifest = args.manifest or args.root / "data/clean-split-v1" / (
        "split_manifest_dev.json" if args.split == "fold0" else "split_manifest_full_sealed.json"
    )
    res = evaluate_split(
        root=args.root.resolve(),
        manifest_path=manifest.resolve(),
        prompt_path=args.prompt.resolve(),
        specialist_dir=args.specialist_dir.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
        out_summary_path=args.out.resolve(),
        split_mode=args.split,
        run_zeroshot_base=not args.skip_zeroshot,
        prediction_dir=args.prediction_dir,
        use_4bit=not args.bf16,
        batch_size=args.batch_size,
        run_verifier=not args.skip_verifier,
        followup=Fold0Followup(*args.paired_followup) if args.paired_followup else None,
    )
    ft_m = res["finetuned_vlm_11code_standalone"]["metrics"]
    sp_m = res["specialist_standalone"]["metrics"]
    hybrid = res.get("hybrid_vlm_plus_verified_specialist")
    hybrid_text = f" | Hybrid F1={hybrid['metrics']['macro_f1']:.4f}" if hybrid else ""
    print(
        f"EVAL_{args.split.upper()}_DONE | Specialist F1={sp_m['macro_f1']:.4f} ({sp_m['quantitative_points_out_of_70']:.2f}pt) "
        f"| FT-VLM-11 F1={ft_m['macro_f1']:.4f} ({ft_m['quantitative_points_out_of_70']:.2f}pt) "
        f"{hybrid_text}",
        flush=True,
    )


if __name__ == "__main__":
    main()
