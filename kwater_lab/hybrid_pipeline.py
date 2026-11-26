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
from .law_reranker_11code import Native11LawRetriever, read_selected_cases
from .metrics import CODES, decode_target, evaluate, paired_bootstrap
from .train_head import save_atomic
from .vlm_11code import (
    format_competition_record,
    infer_batched_vlm,
    load_qwen_4bit_or_bf16,
    verify_specialist_candidates_with_vlm,
)


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
        if v.get("accepted") and isinstance(v.get("hazard"), dict):
            by_idx.setdefault(int(v["row_idx"]), []).append(v["hazard"])

    for idx, new_hazards in by_idx.items():
        cur = merged[idx].get("parsed")
        if not isinstance(cur, dict):
            cur = {"scene_note": "현장 작업 구간에서 안전 위험요인이 확인된다.", "status": "hazard", "hazards": []}
        hazards = usable_draft_hazards(cur)
        for nh in new_hazards:
            if nh.get("code") in CODES and nh["code"] not in hazards and len(hazards) < 3:
                hazards[nh["code"]] = nh
        if hazards:
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
    max_pixels_infer: int = 512 * 32 * 32,
    prediction_dir: Path | None = None,
    use_4bit: bool = True,
    batch_size: int = 4,
    run_verifier: bool = True,
) -> dict[str, Any]:
    eval_start = time.perf_counter()
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
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
    cache_base_identity = {
        "base": base_cache_parts,
        "device": torch.cuda.get_device_name(device),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        "inference_source_sha256": digest(Path(infer_batched_vlm.__code__.co_filename)),
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
        "specialist_standalone": {
            "calibration": "unfitted_fixed_0.5" if split_mode == "fold0" else "development_oof",
            "metrics": spec_eval,
            "severity4": spec_sev4,
        },
    }
    local_pred_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions(filename, rows):
        save_atomic(
            local_pred_dir / filename,
            lambda f: f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows).encode("utf-8")),
        )

    def write_summary():
        results_out["elapsed_seconds"] = round(time.perf_counter() - eval_start, 2)
        results_out["peak_gpu_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        out_summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_atomic(out_summary_path, lambda f: f.write((json.dumps(results_out, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))

    # Optional Zero-Shot Base Qwen3-VL-4B-Instruct control (100% leak-free base VLM benchmark)
    zs_preds: dict[str, frozenset[str]] | None = None
    if run_zeroshot_base:
        print(f"Running Zero-Shot Base Qwen3-VL-4B-Instruct on {split_mode} (n={len(eval_rows)})...", flush=True)
        zs_model, zs_proc = load_qwen_4bit_or_bf16(base_model_dir, max_pixels=max_pixels_infer, use_4bit=use_4bit, adapter_dir=None)
        zs_raw, zs_gen = infer_batched_vlm(
            zs_model, zs_proc, eval_rows, image_root, system_prompt, batch_size=batch_size,
            cache_path=local_pred_dir / "zeroshot_vlm_raw_cache.json", cache_key=cache_key(),
        )
        del zs_model, zs_proc
        torch.cuda.empty_cache()

        zs_recs = [format_competition_record(r["sample_id"], o["parsed"], law_retriever) for r, o in zip(eval_rows, zs_raw)]
        save_predictions("zeroshot_raw.jsonl", zs_raw)
        save_predictions("zeroshot_formatted.jsonl", zs_recs)
        zs_preds = validate_official_records(zs_recs, eval_rows, validator)
        results_out["zeroshot_base_vlm_11code"] = {
            "generation": zs_gen,
            "kit_schema_valid_rate": 1.0,
            "metrics": evaluate(gold_codes, zs_preds, group_map),
            "severity4": severity4_miss_stats(raw_train_by_id, zs_preds),
        }

    # Fine-tuned 11-Code Multimodal LoRA VLM + Hybrid Visual Candidate Verification
    print(f"Running Fine-Tuned 11-Code Qwen3-VL-4B LoRA ({adapter_dir}) on {split_mode} (n={len(eval_rows)})...", flush=True)
    ft_model, ft_proc = load_qwen_4bit_or_bf16(base_model_dir, max_pixels=max_pixels_infer, use_4bit=use_4bit, adapter_dir=adapter_dir)
    ft_raw, ft_gen = infer_batched_vlm(
        ft_model, ft_proc, eval_rows, image_root, system_prompt, batch_size=batch_size,
        cache_path=local_pred_dir / "ft_vlm_raw_cache.json", cache_key=cache_key(adapter_sha256),
    )

    ft_recs = [format_competition_record(r["sample_id"], o["parsed"], law_retriever) for r, o in zip(eval_rows, ft_raw)]
    save_predictions("ft_vlm_raw.jsonl", ft_raw)
    save_predictions("ft_vlm_11code.jsonl", ft_recs)
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
    write_summary()
    if not run_verifier:
        return results_out

    # Propose high-confidence missed hazards from Specialist -> VLM Visual Verification
    t_ver = time.perf_counter()
    cands = select_booster_candidates(
        eval_rows, ft_raw, spec_code_p, spec_scene_p, spec_calib, code_precisions=code_precs
    )
    verified = verify_specialist_candidates_with_vlm(ft_model, ft_proc, cands, image_root, system_prompt, batch_size=batch_size)
    save_predictions("specialist_verification_raw.jsonl", verified)
    ver_sec = round(time.perf_counter() - t_ver, 2)
    del ft_model, ft_proc
    torch.cuda.empty_cache()

    hyb_raw = merge_verified_candidates(ft_raw, verified)
    hyb_recs = [format_competition_record(r["sample_id"], o["parsed"], law_retriever) for r, o in zip(eval_rows, hyb_raw)]
    save_predictions("hybrid_vlm_verified_specialist.jsonl", hyb_recs)
    hyb_preds = validate_official_records(hyb_recs, eval_rows, validator)
    hyb_eval = evaluate(gold_codes, hyb_preds, group_map)
    hyb_sev4 = severity4_miss_stats(raw_train_by_id, hyb_preds)

    results_out["hybrid_verification_status"] = "completed"
    results_out["hybrid_vlm_plus_verified_specialist"] = {
        "proposed_candidates": len(cands),
        "vlm_accepted_candidates": sum(bool(v.get("accepted")) for v in verified),
        "vlm_rejected_candidates": sum(not bool(v.get("accepted")) for v in verified),
        "verification_seconds": ver_sec,
        "kit_schema_valid_rate": 1.0,
        "metrics": hyb_eval,
        "severity4": hyb_sev4,
    }
    comparisons.update({
        "hybrid_vs_ft_vlm": paired_bootstrap(gold_codes, ft_preds, hyb_preds, group_map, resamples=2000, seed=42),
        "hybrid_vs_specialist": paired_bootstrap(gold_codes, spec_preds, hyb_preds, group_map, resamples=2000, seed=42),
    })
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
