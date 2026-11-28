"""Self-contained competition submission builder & ZIP packager (`safeground_v5` / `v3` engine).

Stages local artifacts and builds the archive for the official local kit:
  - Matches adapter training prompt (`system_prompt_v3.txt` or `system_prompt_11code.txt`) & `max_pixels` by SHA-256.
  - Bundles 1-token log-odds margin verification (`margin_gate_calibration.json`) when hybrid mode is enabled.
  - Bundles `ActionSupportedLawRetriever`, `prevention_law_guard`, `ULTRA_RARE_EVIDENCE_CUES`, and `format_v3_record`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import torch

from . import clean_specialist, law_reranker_11code, metrics, prevention_law_guard, v3_turnaround, vlm_11code
from .audit import digest
from .hybrid_pipeline import merge_verified_candidates, select_booster_candidates, usable_draft_hazards
from .law_reranker_11code import export_compact_case_bank, read_selected_cases
from .metrics import CODES, LIKELIHOOD, RISK_BANDS, SEVERITY
from .train_head import save_atomic


MAIN_PY_SOURCE = r'''"""SafeGround v5 (v3 Engine): Native 11-Code Multimodal Qwen3-VL-4B LoRA + 1-Token Margin Verified Dual-Vision Specialist."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import log, sqrt
import math
import os
from pathlib import Path
import random
import re
import time
import unicodedata
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent

# {{SHARED_RUNTIME}}


@torch.inference_mode()
def run_vision_specialist(image_paths: list[Path], spec_dir: Path, device: torch.device):
    from transformers import Dinov2Model, SiglipVisionModel

    sig_model = SiglipVisionModel.from_pretrained(
        str(spec_dir / "sig_vision"), dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    dino_model = Dinov2Model.from_pretrained(
        str(spec_dir / "dino_vision"), dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    checkpoint = torch.load(spec_dir / "specialist_5fold_ensemble.pt", map_location="cpu", weights_only=True)
    if tuple(checkpoint["codes"]) != CODES:
        raise ValueError("Specialist code order differs from the competition")
    heads = []
    for fold in range(5):
        for state in checkpoint["fold_models"][str(fold)]:
            head = SpatialDualSpecialistHead().to(device)
            head.load_state_dict(state)
            heads.append(head.eval())

    codes = np.empty((len(image_paths), len(CODES)), dtype=np.float64)
    scenes = np.empty(len(image_paths), dtype=np.float64)
    chunk_size = 8
    for start in range(0, len(image_paths), chunk_size):
        pictures = []
        for path in image_paths[start : start + chunk_size]:
            with Image.open(path) as image:
                pictures.append(ImageOps.exif_transpose(image).convert("RGB"))
        features = extract_batch_7view_features(pictures, sig_model, dino_model, device, crop_batch_size=24)
        codes[start : start + len(pictures)], scenes[start : start + len(pictures)] = predict_probs(heads, features, device)
    del sig_model, dino_model, heads
    torch.cuda.empty_cache()
    return codes, scenes, checkpoint["full_dev_calibration"]


def is_cuda_oom(error: RuntimeError) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    first_line = str(error).partition("\n")[0].strip().lower()
    return first_line.rstrip(".") in {"cuda error: out of memory", "cuda out of memory"} or first_line.startswith(
        "cuda out of memory. tried to allocate "
    )


def load_vlm(base_dir: Path, adapter_dir: Path, max_pixels: int):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(str(base_dir), min_pixels=128 * 32 * 32, max_pixels=max_pixels, local_files_only=True)
    proc.image_processor.size = {"shortest_edge": 128 * 32 * 32, "longest_edge": max_pixels}
    for attr, size in (("min_pixels", 128 * 32 * 32), ("max_pixels", max_pixels)):
        if hasattr(proc.image_processor, attr):
            setattr(proc.image_processor, attr, size)
    proc.tokenizer.padding_side = "left"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base_dir), local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, proc


@torch.inference_mode()
def score_candidates_1token_margin(
    model: Any, proc: Any, candidates: list[dict[str, Any]], prompt: str, batch_sz: int
) -> tuple[list[float], int, int]:
    if not candidates:
        return [], max(1, int(batch_sz)), 0
    acc_ids = proc.tokenizer.encode("accept", add_special_tokens=False)
    rej_ids = proc.tokenizer.encode("reject", add_special_tokens=False)
    if len(acc_ids) != 1 or len(rej_ids) != 1:
        raise RuntimeError(f"Expected single-token accept/reject, got {acc_ids} vs {rej_ids}")
    acc_tok, rej_tok = acc_ids[0], rej_ids[0]
    margins: list[float] = []
    pos = 0
    eff_bs = max(1, int(batch_sz))
    backoffs = 0
    while pos < len(candidates):
        chunk = candidates[pos : pos + eff_bs]
        txts, pics = [], []
        for candidate in chunk:
            vm = [{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verify_user_text(candidate["code"])}]}]
            txts.append(proc.apply_chat_template(vm, tokenize=False, add_generation_prompt=True) + '{"decision":"')
            with Image.open(candidate["image"]) as im:
                pics.append(ImageOps.exif_transpose(im).convert("RGB"))
        vb = None
        oom = False
        try:
            vb = proc(text=txts, images=pics, padding=True, return_tensors="pt").to("cuda")
            out = model(**vb, logits_to_keep=1, use_cache=False)
        except RuntimeError as error:
            if len(chunk) == 1 or not is_cuda_oom(error):
                raise
            oom = True
        if oom:
            del vb
            torch.cuda.empty_cache()
            eff_bs = max(1, len(chunk) // 2)
            backoffs += 1
            print(f"verify_margin_batch_backoff {eff_bs} after {pos} scored candidates", flush=True)
            continue
        last_logits = out.logits[:, -1, :].float()
        diff = (last_logits[:, acc_tok] - last_logits[:, rej_tok]).cpu().tolist()
        margins.extend(float(x) for x in diff)
        del vb, out, last_logits
        pos += len(chunk)
    return margins, eff_bs, backoffs


def main() -> None:
    t_total0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    seed_all(int(os.environ.get("KWATER_SEED", "42")))
    cfg = json.loads((ROOT / "pipeline_config.json").read_text(encoding="utf-8"))
    use_hybrid = bool(cfg.get("enable_hybrid_booster", True))
    margin_gate = validate_margin_gate_config(cfg.get("margin_gate")) if use_hybrid else None
    max_cands_per_photo = int(cfg.get("max_candidates_per_photo", 1))
    if max_cands_per_photo < 1 or max_cands_per_photo > 3:
        raise ValueError(f"Invalid max_candidates_per_photo: {max_cands_per_photo}")

    manifest_lines = [l for l in (args.input_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest_rows = [json.loads(l) for l in manifest_lines]
    img_by_stem = {p.stem: p for p in (args.input_dir / "images").iterdir() if p.is_file()}
    image_paths = [
        (args.input_dir / r["image"]) if "image" in r and (args.input_dir / str(r["image"])).is_file() else img_by_stem[r["image_id"]]
        for r in manifest_rows
    ]
    image_ids = [r["image_id"] for r in manifest_rows]

    device = torch.device("cuda:0")
    spec_code_p, spec_scene_p, spec_calib = (None, None, None)
    specialist_seconds = 0.0
    if use_hybrid:
        t_sp0 = time.perf_counter()
        spec_code_p, spec_scene_p, spec_calib = run_vision_specialist(image_paths, ROOT / "specialist", device)
        specialist_seconds = round(time.perf_counter() - t_sp0, 3)

    prompt_file = cfg.get("prompt_filename", "system_prompt_11code.txt")
    prompt = (ROOT / prompt_file).read_text(encoding="utf-8")
    max_pixels = int(cfg["max_pixels_infer"])
    batch_sz = max(1, int(os.environ.get("KW_BATCH", str(cfg["batch_size"]))))
    verify_batch_sz = max(1, int(os.environ.get("KW_VERIFY_BATCH", str(cfg.get("verify_batch_size", 2)))))
    verify_max_new_tokens = int(cfg.get("verify_max_new_tokens", VERIFY_MAX_NEW_TOKENS))
    if verify_max_new_tokens < 1:
        raise ValueError(f"Invalid verify_max_new_tokens: {verify_max_new_tokens}")

    t_load0 = time.perf_counter()
    model, proc = load_vlm(ROOT / "base_model", ROOT / "adapter", max_pixels=max_pixels)
    vlm_load_seconds = round(time.perf_counter() - t_load0, 3)

    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]}]
    prefix = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    vlm_outputs = []
    effective_batch_sz = batch_sz
    inspect_backoffs = 0
    raw_specialist_candidates = 0
    margin_gate_passed_candidates = 0
    verified_accepted_candidates = 0
    margin_gate_seconds = 0.0
    verify_generate_seconds = 0.0
    verify_margin_eff_bs = verify_batch_sz
    verify_gen_eff_bs = verify_batch_sz
    verify_backoffs = 0

    with torch.inference_mode():
        t_ins0 = time.perf_counter()
        pos = 0
        while pos < len(image_paths):
            sub_p = image_paths[pos : pos + effective_batch_sz]
            pics = []
            for p in sub_p:
                with Image.open(p) as im:
                    pics.append(ImageOps.exif_transpose(im).convert("RGB"))
            batch = None
            oom = False
            try:
                batch = proc(text=[prefix] * len(sub_p), images=pics, padding=True, return_tensors="pt").to("cuda")
                gen = model.generate(**batch, do_sample=False, max_new_tokens=INSPECT_MAX_NEW_TOKENS, use_cache=True)
            except RuntimeError as error:
                if len(sub_p) == 1 or not is_cuda_oom(error):
                    raise
                oom = True
            if oom:
                del batch
                torch.cuda.empty_cache()
                effective_batch_sz = max(1, len(sub_p) // 2)
                inspect_backoffs += 1
                print(f"vlm_batch_backoff {effective_batch_sz} after {pos} completed images", flush=True)
                continue
            in_len = batch["input_ids"].shape[-1]
            for li in range(len(sub_p)):
                txt = proc.decode(gen[li, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                parsed, parse_error = parse_json_loose(txt)
                vlm_outputs.append({"parsed": parsed, "parse_error": parse_error})
            del batch, gen
            pos += len(sub_p)
        inspect_seconds = round(time.perf_counter() - t_ins0, 3)
        print(f"vlm_generation_done initial_batch={batch_sz} final_effective_batch={effective_batch_sz}", flush=True)

        # Hybrid Specialist -> Mandatory 1-Token Calibrated Margin Gate -> VLM Visual Verification pass
        if use_hybrid:
            if spec_code_p is None or spec_calib is None or margin_gate is None:
                raise RuntimeError("Hybrid booster requires specialist predictions and a calibrated 1-token margin_gate")
            eval_rows = [{"sample_id": i, "image": p} for i, p in zip(image_ids, image_paths, strict=True)]
            candidates = select_booster_candidates(
                eval_rows, vlm_outputs, spec_code_p, spec_scene_p, spec_calib,
                code_precisions=cfg["specialist_oof_code_precisions"],
                margin_boost_when_hazard=cfg["margin_boost_when_hazard"],
                margin_boost_when_empty=cfg["margin_boost_when_empty"],
                max_candidates_per_photo=max_cands_per_photo,
            )
            raw_specialist_candidates = len(candidates)
            t_mg0 = time.perf_counter()
            vlm_margins, verify_margin_eff_bs, mg_backoffs = score_candidates_1token_margin(
                model, proc, candidates, prompt, verify_batch_sz
            )
            verify_backoffs += mg_backoffs
            beta = float(margin_gate["beta"])
            tau_haz = float(margin_gate["tau_when_hazard"])
            tau_emp = float(margin_gate["tau_when_empty"])
            gated_candidates = []
            for c, vm in zip(candidates, vlm_margins, strict=True):
                sp = min(max(float(c["spec_cal_prob"]), 1e-4), 1.0 - 1e-4)
                joint = vm + beta * math.log(sp / (1.0 - sp))
                req = tau_emp if c["was_vlm_empty"] else tau_haz
                if joint >= req:
                    gated_candidates.append(c)
            margin_gate_seconds = round(time.perf_counter() - t_mg0, 3)
            margin_gate_passed_candidates = len(gated_candidates)
            print(f"margin gate passed {len(gated_candidates)}/{len(candidates)} specialist candidates", flush=True)

            t_vg0 = time.perf_counter()
            accepted = []
            vpos = 0
            while vpos < len(gated_candidates):
                chunk = gated_candidates[vpos : vpos + verify_gen_eff_bs]
                v_txts, v_pics = [], []
                for candidate in chunk:
                    vm = [{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verify_user_text(candidate["code"])}]}]
                    v_txts.append(proc.apply_chat_template(vm, tokenize=False, add_generation_prompt=True))
                    with Image.open(candidate["image"]) as im:
                        v_pics.append(ImageOps.exif_transpose(im).convert("RGB"))
                vb = None
                voom = False
                try:
                    vb = proc(text=v_txts, images=v_pics, padding=True, return_tensors="pt").to("cuda")
                    vgen = model.generate(**vb, do_sample=False, max_new_tokens=verify_max_new_tokens, use_cache=True)
                except RuntimeError as error:
                    if len(chunk) == 1 or not is_cuda_oom(error):
                        raise
                    voom = True
                if voom:
                    del vb
                    torch.cuda.empty_cache()
                    verify_gen_eff_bs = max(1, len(chunk) // 2)
                    verify_backoffs += 1
                    print(f"verify_gen_batch_backoff {verify_gen_eff_bs} after {vpos} verified candidates", flush=True)
                    continue
                vin_len = vb["input_ids"].shape[-1]
                for li, candidate in enumerate(chunk):
                    vtxt = proc.decode(vgen[li, vin_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    hazard = parse_verified_hazard(vtxt, candidate["code"])
                    if hazard is not None:
                        accepted.append({**candidate, "accepted": True, "hazard": hazard})
                del vb, vgen
                vpos += len(chunk)
            verify_generate_seconds = round(time.perf_counter() - t_vg0, 3)
            verified_accepted_candidates = len(accepted)
            print(f"specialist accepted {len(accepted)}/{len(gated_candidates)} verified candidates", flush=True)
            vlm_outputs = merge_verified_candidates(vlm_outputs, accepted)

    t_post0 = time.perf_counter()
    law_rows = [json.loads(l) for l in (ROOT / "assets" / "laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    case_rows = [json.loads(l) for l in (ROOT / "assets" / "cases_11code.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    use_v3_formatter = bool(cfg.get("use_v3_formatter", True))
    reranker = ActionSupportedLawRetriever(law_rows, case_rows) if use_v3_formatter else Native11LawRetriever(law_rows, case_rows)
    row_formatter = format_v3_record if use_v3_formatter else format_competition_record
    parsed_count = sum(item["parse_error"] is None for item in vlm_outputs)
    print(f"raw VLM parsed {parsed_count}/{len(image_ids)} images", flush=True)

    out_lines = [
        json.dumps(row_formatter(iid, item["parsed"], reranker), ensure_ascii=False, separators=(",", ":")) + "\n"
        for iid, item in zip(image_ids, vlm_outputs, strict=True)
    ]
    out_bytes = "".join(out_lines).encode("utf-8")
    output_sha256 = hashlib.sha256(out_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "submission.jsonl").write_bytes(out_bytes)
    postprocess_seconds = round(time.perf_counter() - t_post0, 3)
    total_seconds = round(time.perf_counter() - t_total0, 3)
    telemetry = {
        "input_photos": len(image_ids),
        "enable_hybrid_booster": use_hybrid,
        "max_candidates_per_photo": max_cands_per_photo if use_hybrid else 0,
        "inspect_initial_batch_size": batch_sz,
        "inspect_final_effective_batch_size": effective_batch_sz,
        "inspect_backoff_count": inspect_backoffs,
        "verify_initial_batch_size": verify_batch_sz if use_hybrid else 0,
        "verify_margin_effective_batch_size": verify_margin_eff_bs if use_hybrid else 0,
        "verify_gen_effective_batch_size": verify_gen_eff_bs if use_hybrid else 0,
        "verify_max_new_tokens": verify_max_new_tokens if use_hybrid else 0,
        "verify_backoff_count": verify_backoffs,
        "raw_vlm_parsed_count": parsed_count,
        "raw_specialist_candidates": raw_specialist_candidates,
        "margin_gate_passed_candidates": margin_gate_passed_candidates,
        "verified_accepted_candidates": verified_accepted_candidates,
        "specialist_seconds": specialist_seconds,
        "vlm_load_seconds": vlm_load_seconds,
        "inspect_seconds": inspect_seconds,
        "margin_gate_seconds": margin_gate_seconds,
        "verify_generate_seconds": verify_generate_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_seconds": total_seconds,
        "sec_per_photo": round(total_seconds / max(len(image_ids), 1), 3),
        "output_submission_sha256": output_sha256,
    }
    print("MAIN_PY_TELEMETRY " + json.dumps(telemetry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
'''


PRODUCTION_VERIFY_MAX_NEW_TOKENS = 280  # Matched to Fold-4 benchmark & fold4_same_verifier


def validate_margin_gate_config(gate: Any) -> dict[str, Any]:
    """Require a finite calibrated 1-token log-odds margin gate for hybrid deployment."""
    if not isinstance(gate, dict):
        raise ValueError("Hybrid deployment requires a calibrated margin_gate dictionary")
    for key in ("beta", "tau_when_hazard", "tau_when_empty"):
        val = gate.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"Invalid or missing margin_gate[{key!r}]: {val!r}")
    if float(gate["beta"]) <= 0.0:
        raise ValueError(f"margin_gate['beta'] must be positive, got {gate['beta']!r}")
    return gate


def render_entrypoint() -> str:
    """Bundle audited shared logic without importing training modules at inference time."""
    constants = {
        "CODES": CODES,
        "LIKELIHOOD": LIKELIHOOD,
        "SEVERITY": SEVERITY,
        "RISK_BANDS": RISK_BANDS,
        "KOR_CODE_NAMES": vlm_11code.KOR_CODE_NAMES,
        "INSPECT_USER_TEXT": vlm_11code.INSPECT_USER_TEXT,
        "INSPECT_MAX_NEW_TOKENS": vlm_11code.INSPECT_MAX_NEW_TOKENS,
        "VERIFY_MAX_NEW_TOKENS": PRODUCTION_VERIFY_MAX_NEW_TOKENS,
        "STATUTORY_CODE_PRIORS": law_reranker_11code.STATUTORY_CODE_PRIORS,
        "_STOP": sorted(law_reranker_11code._STOP),
        "SIG_DIM": clean_specialist.SIG_DIM,
        "DINO_DIM": clean_specialist.DINO_DIM,
        "FEAT_DIM": clean_specialist.FEAT_DIM,
        "NUM_VIEWS": clean_specialist.NUM_VIEWS,
        "NUM_CODES": len(CODES),
    }
    parts = [f"{name} = {value!r}" for name, value in constants.items()]
    parts.append("_STOP = frozenset(_STOP)")
    parts.append(f"HEX_ID = re.compile({metrics.HEX_ID.pattern!r})")
    for name in ("_ART_RE", "_WORD_RE", "_AMEND_RE"):
        parts.append(f"{name} = re.compile({getattr(law_reranker_11code, name).pattern!r})")

    # Guard regexes and statutory support rules from prevention_law_guard & v3_turnaround
    for name in ("_CLAUSE", "_DENIAL", "_CONTACT", "_CUT_BYPASS", "_CONTACT_PROHIBITION", "_EXPLICIT_FIRST", "_FIRE_POSITION", "_AWAY"):
        parts.append(f"{name} = re.compile({getattr(prevention_law_guard, name).pattern!r})")
    parts.append(f"_CUT = {prevention_law_guard._CUT!r}")
    parts.extend(
        inspect.getsource(cls_def).rstrip()
        for cls_def in (prevention_law_guard.SupportRule, prevention_law_guard.OrderCheck, prevention_law_guard.RerankDecision)
    )
    rules_src = ",\n    ".join(
        f"SupportRule({r.name!r}, {r.law!r}, {r.record!r}, {r.action!r}, {r.scope!r})"
        for r in prevention_law_guard.RULES
    )
    parts.append(f"RULES = (\n    {rules_src},\n)")

    cues_src = ",\n    ".join(
        f"(re.compile({pat.pattern!r}), {boosts!r})" for pat, boosts in v3_turnaround.ACTION_ARTICLE_CUES
    )
    parts.append(f"ACTION_ARTICLE_CUES = (\n    {cues_src},\n)")
    rare_src = ",\n    ".join(
        f"{k!r}: re.compile({pat.pattern!r})" for k, pat in v3_turnaround.ULTRA_RARE_EVIDENCE_CUES.items()
    )
    parts.append(f"ULTRA_RARE_EVIDENCE_CUES = {{\n    {rare_src},\n}}")

    definitions = (
        metrics.unique_keys, metrics.reject_constant, metrics.check_keys, metrics.check_text,
        metrics.check_level, metrics.decode_target,
        law_reranker_11code.article_to_osh_id, law_reranker_11code.extract_terms,
        law_reranker_11code.hazard_query_terms, law_reranker_11code._TFIDFIndex,
        law_reranker_11code.Native11LawRetriever,
        prevention_law_guard._text, prevention_law_guard.check_unsafe_action_order,
        prevention_law_guard.build_active_records, prevention_law_guard.action_support,
        prevention_law_guard.rerank_top1_by_action_support,
        vlm_11code.seed_all, vlm_11code.verify_user_text, vlm_11code.sanitize_text,
        vlm_11code.parse_json_loose, vlm_11code.validate_vlm_11code_obj,
        vlm_11code.verified_hazard, vlm_11code.parse_verified_hazard, vlm_11code.format_competition_record,
        clean_specialist.make_spatial_crops, clean_specialist._pil_list_to_sig_tensor,
        clean_specialist._pil_list_to_dino_tensor, clean_specialist.extract_batch_7view_features,
        clean_specialist.SpatialDualSpecialistHead, clean_specialist.predict_probs,
        usable_draft_hazards, merge_verified_candidates, select_booster_candidates,
        validate_margin_gate_config,
    )
    parts.extend(inspect.getsource(definition).rstrip() for definition in definitions)

    # Append ActionSupportedLawRetriever (replacing relative imports with already-inlined globals),
    # _repair_unsafe_prevention_order, and format_v3_record
    aslr_src = inspect.getsource(v3_turnaround.ActionSupportedLawRetriever).rstrip()
    aslr_src = aslr_src.replace("from .law_reranker_11code import article_to_osh_id", "pass")
    aslr_src = aslr_src.replace("from .law_reranker_11code import STATUTORY_CODE_PRIORS, hazard_query_terms", "pass")
    parts.append(aslr_src)
    parts.append(inspect.getsource(v3_turnaround._repair_unsafe_prevention_order).rstrip())
    parts.append(inspect.getsource(v3_turnaround.format_v3_record).rstrip())

    source = MAIN_PY_SOURCE.replace("# {{SHARED_RUNTIME}}", "\n\n".join(parts))
    compile(source, "main.py", "exec")
    return source


ZIP_LIMIT_BYTES = 20_000_000_000


def _resolve_prompt_file(root: Path, expected_sha: str | None, explicit_prompt: Path | None = None) -> Path:
    candidates = [explicit_prompt] if explicit_prompt is not None else [
        root / "kwater_lab/system_prompt_v3.txt",
        root / "kwater_lab/system_prompt_11code.txt",
    ]
    for p in candidates:
        if p is not None and p.is_file() and (expected_sha is None or digest(p) == expected_sha):
            return p
    raise ValueError("Bundled prompt differs from the adapter training prompt")


def _resolve_oof_archive(specialist_ckpt: Path) -> Path:
    for name in ("oof_1045_predictions.npz", "dev_oof_predictions.npz"):
        cand = specialist_ckpt.parent / name
        if cand.is_file():
            return cand
    return specialist_ckpt.parent / "dev_oof_predictions.npz"


def validated_release_inputs(
    root: Path,
    adapter_dir: Path,
    specialist_ckpt: Path,
    enable_hybrid: bool,
    prompt_path: Path | None = None,
    allow_fold_hybrid: bool = False,
    base_dir: Path | None = None,
):
    """Reject incomplete adapters and identify the exact case-bank training population."""
    from safetensors import safe_open

    summary_path = adapter_dir.parent / "training_summary.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    for path in (adapter_weights, adapter_config_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed native 11-code training artifact required: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    version = summary.get("version")
    examples = summary.get("curriculum_examples")
    accumulation = summary.get("grad_accum")
    if not (type(examples) is int and examples > 0 and type(accumulation) is int and accumulation > 0):
        raise ValueError("A completed native 11-code training summary is required")
    steps_per_epoch = (examples + accumulation - 1) // accumulation

    if version in {"vlm-11code-multimodal-qlora-v1", "vlm-11code-multimodal-qlora-v2"}:
        epochs = summary.get("epochs")
        if (type(epochs) is not int or epochs <= 0
            or len(summary.get("history", ())) != epochs
            or summary.get("optimizer_steps") != epochs * steps_per_epoch):
            raise ValueError("A completed native 11-code training summary is required")
    elif version == "vlm-11code-multimodal-qlora-v2-epoch-boundary":
        cfg_epochs = summary.get("configured_epochs")
        comp_epochs = summary.get("completed_epochs")
        cursor = summary.get("resume_cursor") or {}
        if (type(cfg_epochs) is not int or type(comp_epochs) is not int
            or not (1 <= comp_epochs < cfg_epochs)
            or summary.get("configuration", {}).get("epochs") != cfg_epochs
            or len(summary.get("history", ())) != comp_epochs
            or summary.get("optimizer_steps") != comp_epochs * steps_per_epoch
            or summary.get("scheduled_total_steps") != cfg_epochs * steps_per_epoch
            or cursor != {"next_epoch": comp_epochs + 1, "next_offset": 0, "global_step": comp_epochs * steps_per_epoch}):
            raise ValueError("Invalid epoch-boundary training summary metadata")
    else:
        raise ValueError("A completed native 11-code training summary is required")
    if summary.get("adapter_safetensors_sha256") != digest(adapter_weights):
        raise ValueError("Adapter weights do not match the completed training summary")

    training_prompt = summary.get("configuration", {}).get("prompt_sha256")
    resolved_prompt = _resolve_prompt_file(root, training_prompt, prompt_path)

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("A PEFT LoRA adapter is required")
    with safe_open(adapter_weights, framework="pt", device="cpu") as weights:
        keys = weights.keys()
        if not all(any(family in key for key in keys) for family in ("language_model.layers", "visual.merger", "visual.deepstack_merger_list")):
            raise ValueError("Adapter is missing language or visual merger weights")

    full1045_manifest = adapter_dir.parent / "manifest_all_1045_train.json"
    validation_fold = summary["val_fold"]
    if full1045_manifest.is_file() and validation_fold == -1:
        manifest = json.loads(full1045_manifest.read_text(encoding="utf-8"))
        train_ids = {row["sample_id"] for row in manifest if row["split"] == "dev"}
    else:
        manifest_path = root / "data/clean-split-v1/split_manifest_dev.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = {
            row["sample_id"] for row in manifest if row["split"] == "dev" and row["dev_fold"] != validation_fold
        }
    if validation_fold not in {-1, 0, 1, 2, 3, 4} or len(train_ids) != summary.get("train_photos"):
        raise ValueError("Adapter training population differs from the release manifest")
    if enable_hybrid and validation_fold != -1 and not allow_fold_hybrid:
        raise ValueError("Production specialist ensemble requires a development-wide adapter; use --vlm-only or --allow-fold-hybrid")
    if enable_hybrid:
        oof_arc_path = _resolve_oof_archive(specialist_ckpt)
        gate_file = adapter_dir.parent / "margin_gate_calibration.json"
        for path in (specialist_ckpt, oof_arc_path, gate_file):
            if not path.is_file():
                raise FileNotFoundError(f"Hybrid artifact required: {path}")
        validate_margin_gate_config(json.loads(gate_file.read_text(encoding="utf-8")))

    base = base_dir or root / "data/baseline/base_model"
    training_base = summary.get("configuration", {}).get("base_config_sha256")
    if training_base is not None and training_base != digest(base / "config.json"):
        raise ValueError("Bundled base model config differs from the training config")
    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for name in set(index["weight_map"].values()):
        with safe_open(base / name, framework="pt", device="cpu") as shard:
            expected_keys = {key for key, filename in index["weight_map"].items() if filename == name}
            if set(shard.keys()) != expected_keys:
                raise ValueError(f"Base shard does not agree with the tensor index: {name}")
    wheelhouse = root / "data/baseline/wheelhouse"
    wheels = {f.name.split("-")[0].replace("_", "-"): f for f in wheelhouse.glob("*.whl")}
    for line in (root / "data/baseline/requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        package, version_and_hash = line.split("==", 1)
        version, expected_hash = version_and_hash.split(" --hash=sha256:", 1)
        wheel = wheels[package.replace("_", "-")]
        with wheel.open("rb") as stream:
            if wheel.name.split("-")[1] != version or hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                raise ValueError(f"Offline wheel differs from its lock: {wheel.name}")
    return summary, train_ids, resolved_prompt


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _describe_training_and_calibration_scope(
    root: Path,
    adapter_dir: Path,
    summary: dict[str, Any],
    train_ids: set[str],
    resolved_prompt: Path,
    enable_hybrid: bool,
    specialist_ckpt: Path,
    oof_path: Path,
    gate_file: Path,
    margin_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    val_fold = int(summary["val_fold"])
    provenance_hashes: dict[str, str | None] = {
        "adapter_safetensors_sha256": str(summary["adapter_safetensors_sha256"]),
        "adapter_config_sha256": digest(adapter_dir / "adapter_config.json"),
        "prompt_sha256": digest(resolved_prompt),
        "curriculum_sha256": summary.get("configuration", {}).get("curriculum_sha256"),
    }
    scope: dict[str, Any] = {
        "vlm_validation_fold": val_fold,
        "vlm_base_inspect_train_photos": len(train_ids),
        "vlm_curriculum_examples": int(summary["curriculum_examples"]),
        "law_case_bank_train_photos": len(train_ids),
        "fold4_is_independent_holdout": val_fold == 4 and not enable_hybrid,
        "fold4_evaluation_guardrail": (
            "In-sample for full-1045 weights; Fold-4 metrics on these weights MUST NOT be treated as an independent score."
            if val_fold == -1
            else "Fold 4 (175 photos, 39 groups) excluded from VLM training."
        ),
        "provenance_sha256": provenance_hashes,
    }
    if not enable_hybrid:
        return scope

    import numpy as np

    ckpt_meta = torch.load(specialist_ckpt, map_location="cpu", weights_only=True)
    with np.load(oof_path, allow_pickle=False) as oof_arc:
        spec_ids = [str(x) for x in oof_arc["ids"].tolist()]
        oof_scores_arr = oof_arc["code_probs"]
    spec_models_n = sum(len(v) for v in ckpt_meta["fold_models"].values())

    inner_pairs_file = adapter_dir.parent / "inner_margin_cal_pairs.json"
    inner_pairs = json.loads(inner_pairs_file.read_text(encoding="utf-8")) if inner_pairs_file.is_file() else []
    cal_pair_keys = {(str(p["orig_id"]), str(p["code"])) for p in inner_pairs if "orig_id" in p and "code" in p}
    cal_photo_ids = {str(p["orig_id"]) for p in inner_pairs if "orig_id" in p}
    cal_group_ids = {str(p["group_id"]) for p in inner_pairs if "group_id" in p}
    cal_in_inspect_n = len(cal_photo_ids & train_ids)

    # Reconstruct exact verify_* supervision pairs to separate pair disjointness from photo/group exposure
    verify_pair_overlap_n: int | None = None
    verify_photo_overlap_n: int | None = None
    verify_group_overlap_n: int | None = None
    full1045_manifest = adapter_dir.parent / "manifest_all_1045_train.json"
    manifest_src = full1045_manifest if (full1045_manifest.is_file() and val_fold == -1) else (root / "data/clean-split-v1/split_manifest_dev.json")
    raw_train_jsonl = root / "data/train/train.jsonl"
    if inner_pairs and manifest_src.is_file() and raw_train_jsonl.is_file():
        manifest_rows = json.loads(manifest_src.read_text(encoding="utf-8"))
        split_rows = (
            [r for r in manifest_rows if r["split"] == "dev"]
            if val_fold == -1
            else [r for r in manifest_rows if r["split"] == "dev" and int(r["dev_fold"]) != val_fold]
        )
        oof_for_cur_ids = spec_ids
        oof_for_cur_scores = oof_scores_arr
        if val_fold == 4 and (root / "data/clean-specialist-v1/strict-inner-oof/fold-4.npz").is_file():
            with np.load(root / "data/clean-specialist-v1/strict-inner-oof/fold-4.npz", allow_pickle=False) as f4_in:
                oof_for_cur_ids = [str(x) for x in f4_in["train_ids"].tolist()]
                oof_for_cur_scores = f4_in["code_scores"]
        raw_by_id = {r["sample_id"]: r for r in read_selected_cases(raw_train_jsonl, {r["sample_id"] for r in split_rows})}
        seed_cfg = int(summary.get("configuration", {}).get("seed", 42))
        rebuilt_cur, _ = v3_turnaround.build_v3_curriculum_and_inner_val(
            split_rows, raw_by_id, oof_for_cur_scores, oof_for_cur_ids, seed=seed_cfg
        )
        grp_by_sid = {r["sample_id"]: str(r["group_id"]) for r in split_rows}
        ver_pairs: set[tuple[str, str]] = set()
        ver_photos: set[str] = set()
        ver_groups: set[str] = set()
        for item in rebuilt_cur:
            if str(item.get("kind", "")).startswith("verify_"):
                sid_part, _, tag_part = str(item["sample_id"]).partition("__ver_")
                code_part = tag_part.split("_", 1)[1] if "_" in tag_part else ""
                ver_pairs.add((sid_part, code_part))
                ver_photos.add(sid_part)
                if sid_part in grp_by_sid:
                    ver_groups.add(grp_by_sid[sid_part])
        verify_pair_overlap_n = len(cal_pair_keys & ver_pairs)
        verify_photo_overlap_n = len(cal_photo_ids & ver_photos)
        verify_group_overlap_n = len(cal_group_ids & ver_groups)

    provenance_hashes.update({
        "margin_gate_calibration_sha256": digest(gate_file),
        "inner_margin_cal_pairs_sha256": digest(inner_pairs_file) if inner_pairs_file.is_file() else None,
        "specialist_ckpt_sha256": digest(specialist_ckpt),
        "specialist_oof_sha256": digest(oof_path),
    })

    scope.update({
        "specialist_checkpoint_spec": str(ckpt_meta.get("spec", "clean-spec-v1")),
        "specialist_train_photos": len(spec_ids),
        "specialist_ensemble_models": spec_models_n,
        "specialist_oof_archive": oof_path.name,
        "specialist_calibration_method": (
            f"Single calibration fit (temperatures, code thresholds, scene threshold) and per-code precisions "
            f"computed on all {len(spec_ids)} pooled 5-fold x 3-seed incident-group OOF predictions ({oof_path.name}); "
            f"not a separate inner calibration split; inference averages all {spec_models_n} heads."
        ),
        "margin_gate_required": True,
        "margin_gate_exposure_audit": {
            "inner_pairs_n": int((margin_gate or {}).get("inner_pairs_n", len(inner_pairs))),
            "unique_photos_n": len(cal_photo_ids) if cal_photo_ids else None,
            "unique_groups_n": len(cal_group_ids) if cal_group_ids else None,
            "exact_pair_disjoint_from_verify_supervision": (verify_pair_overlap_n == 0) if verify_pair_overlap_n is not None else True,
            "verify_supervision_exact_pair_overlap_count": verify_pair_overlap_n,
            "verify_supervision_photo_overlap_count": verify_photo_overlap_n,
            "verify_supervision_group_overlap_count": verify_group_overlap_n,
            "base_inspect_photo_overlap_count": cal_in_inspect_n if cal_photo_ids else None,
            "all_calibration_photos_used_in_base_inspect_training": True,
        },
        "fold4_is_independent_holdout": False,
    })
    if val_fold == 4 and len(spec_ids) == 966:
        scope["fold4_evaluation_guardrail"] = (
            "Bundled 5-fold specialist checkpoint covers 966 dev photos (including Fold 4 in 4/5 folds); "
            "clean Fold-4 benchmark used Fold-4 OOF specialist + Fold-0..3 inner calibration instead."
        )
    return scope


def build_submission_package(
    root: Path,
    adapter_dir: Path,
    specialist_ckpt: Path,
    dest_dir: Path,
    zip_path: Path | None,
    enable_hybrid: bool = True,
    batch_size: int = 4,
    verify_batch_size: int = 2,
    verify_max_new_tokens: int = PRODUCTION_VERIFY_MAX_NEW_TOKENS,
    max_candidates_per_photo: int = 1,
    prompt_path: Path | None = None,
    allow_fold_hybrid: bool = False,
    research_8b_base_dir: Path | None = None,
) -> dict:
    if batch_size < 1 or verify_batch_size < 1 or verify_max_new_tokens < 1:
        raise ValueError("Batch sizes and verify_max_new_tokens must be positive")
    if max_candidates_per_photo < 1 or max_candidates_per_photo > 3:
        raise ValueError("max_candidates_per_photo must be in 1..3")
    dest_dir = dest_dir.resolve()
    zip_path = zip_path.resolve() if zip_path is not None else None
    if dest_dir.exists() or (zip_path is not None and zip_path.exists()):
        raise FileExistsError("Use a fresh staging directory and archive path")
    if zip_path is not None and zip_path.is_relative_to(dest_dir):
        raise ValueError("Archive path must be outside the staging directory")
    base_dir = root / "data/baseline/base_model"
    base_name = "Qwen3-VL-4B-Instruct"
    base_revision = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    if research_8b_base_dir is not None:
        if enable_hybrid:
            raise ValueError("The research 8B base requires --vlm-only")
        base_dir = research_8b_base_dir.resolve()
        base_name = "Qwen3-VL-8B-Instruct"
        base_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
        source = json.loads((base_dir / "source.json").read_bytes())
        if (source["repo"], source["revision"]) != (f"Qwen/{base_name}", base_revision):
            raise ValueError("Expected the pinned BF16 Qwen3-VL-8B-Instruct source")
        base_file_names = {
            "chat_template.json", "config.json", "generation_config.json", "merges.txt",
            "model.safetensors.index.json", "preprocessor_config.json", "tokenizer_config.json",
            "tokenizer.json", "video_preprocessor_config.json", "vocab.json",
        }
        index = json.loads((base_dir / "model.safetensors.index.json").read_bytes())
        shards = {f"model-{number:05d}-of-00004.safetensors" for number in range(1, 5)}
        if set(index["weight_map"].values()) != shards:
            raise ValueError("Expected four official 8B shards")
        base_file_names.update(shards)
        source_sizes = {item["rfilename"]: item["size"] for item in source["files"]}
        for name in base_file_names:
            if (base_dir / name).stat().st_size != source_sizes.get(name):
                raise ValueError(f"Research base file size differs from pinned asset: {name}")
    else:
        base_file_names = {path.name for path in base_dir.iterdir() if path.is_file()}

    summary, train_ids, resolved_prompt = validated_release_inputs(
        root, adapter_dir, specialist_ckpt, enable_hybrid, prompt_path=prompt_path,
        allow_fold_hybrid=allow_fold_hybrid, base_dir=base_dir,
    )
    if research_8b_base_dir is not None and (
        summary["val_fold"] != 4 or len(train_ids) != 791
        or summary.get("completed_epochs") != 1 or summary.get("optimizer_steps") != 69
        or summary["configuration"].get("base_config_sha256")
        != "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
    ):
        raise ValueError("Expected the measured Fold 4 8B first-epoch adapter")
    dest_dir.mkdir(parents=True, exist_ok=False)

    oof_code_precs = {}
    oof_path = _resolve_oof_archive(specialist_ckpt)
    if enable_hybrid and oof_path.is_file() and specialist_ckpt.is_file():
        import numpy as np
        from .hybrid_pipeline import compute_calib_code_precisions
        ckpt_tmp = torch.load(specialist_ckpt, map_location="cpu", weights_only=True)
        if "oof_code_precisions" in ckpt_tmp:
            oof_code_precs = ckpt_tmp["oof_code_precisions"]
        else:
            oof_arc = np.load(oof_path, allow_pickle=False)
            oof_code_precs = {
                k: round(v, 4)
                for k, v in compute_calib_code_precisions(
                    oof_arc["code_probs"], oof_arc["scene_probs"], oof_arc["y_codes"], ckpt_tmp["full_dev_calibration"]
                ).items()
            }

    margin_gate = None
    gate_file = adapter_dir.parent / "margin_gate_calibration.json"
    if enable_hybrid:
        margin_gate = validate_margin_gate_config(json.loads(gate_file.read_text(encoding="utf-8")))

    scope_info = _describe_training_and_calibration_scope(
        root=root,
        adapter_dir=adapter_dir,
        summary=summary,
        train_ids=train_ids,
        resolved_prompt=resolved_prompt,
        enable_hybrid=enable_hybrid,
        specialist_ckpt=specialist_ckpt,
        oof_path=oof_path,
        gate_file=gate_file,
        margin_gate=margin_gate,
    )

    # 1. Write main.py, prompt file, pipeline_config.json
    entrypoint = render_entrypoint()
    if research_8b_base_dir is not None:
        entrypoint = entrypoint.replace("Qwen3-VL-4B LoRA + 1-Token Margin Verified Dual-Vision Specialist",
                                        "Qwen3-VL-8B Standalone LoRA", 1)
    (dest_dir / "main.py").write_text(entrypoint, encoding="utf-8", newline="\n")
    prompt_filename = resolved_prompt.name
    shutil.copy2(resolved_prompt, dest_dir / prompt_filename)
    max_pixels_infer = int(summary.get("max_pixels") or v3_turnaround.TRAIN_EVAL_MAX_PIXELS)
    save_atomic(
        dest_dir / "pipeline_config.json",
        lambda f: f.write(
            (
                json.dumps(
                    {
                        "enable_hybrid_booster": enable_hybrid,
                        "use_v3_formatter": True,
                        "prompt_filename": prompt_filename,
                        "max_pixels_infer": max_pixels_infer,
                        "batch_size": batch_size,
                        "verify_batch_size": verify_batch_size,
                        "verify_max_new_tokens": verify_max_new_tokens,
                        "max_candidates_per_photo": max_candidates_per_photo,
                        "margin_boost_when_hazard": 0.04,
                        "margin_boost_when_empty": 0.12,
                        "margin_gate": margin_gate,
                        "specialist_oof_code_precisions": oof_code_precs,
                        "training_and_calibration_scope": scope_info,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
        ),
    )

    # 2. Copy requirements.lock and wheelhouse/ from official verified baseline
    link_or_copy(root / "data/baseline/requirements.lock", dest_dir / "requirements.lock")
    for wf in (root / "data/baseline/wheelhouse").iterdir():
        if wf.is_file():
            link_or_copy(wf, dest_dir / "wheelhouse" / wf.name)

    # 3. Link unmodified BF16 Qwen3-VL-4B-Instruct base_model/ and fine-tuned 11-code adapter/
    for name in sorted(base_file_names):
        link_or_copy(base_dir / name, dest_dir / "base_model" / name)
    for af in adapter_dir.iterdir():
        if af.is_file() and af.name in ("adapter_config.json", "adapter_model.safetensors"):
            link_or_copy(af, dest_dir / "adapter" / af.name)

    # 4. Export compact FP16 vision-only towers + specialist heads
    if enable_hybrid:
        pre_sig = root / "data/v3-staged-vision-towers/sig_vision"
        pre_dino = root / "data/v3-staged-vision-towers/dino_vision"
        if pre_sig.is_dir() and pre_dino.is_dir():
            for f_s in pre_sig.iterdir():
                if f_s.is_file():
                    link_or_copy(f_s, dest_dir / "specialist/sig_vision" / f_s.name)
            for f_d in pre_dino.iterdir():
                if f_d.is_file():
                    link_or_copy(f_d, dest_dir / "specialist/dino_vision" / f_d.name)
        else:
            from transformers import Dinov2Model, SiglipVisionModel

            sig_src = root / "data/phase01-runs/phase01-20261126-0513-2155e8fa/models/sig"
            dino_src = root / "data/phase01-runs/phase01-20261126-0513-2155e8fa/models/dino"
            sig_m = SiglipVisionModel.from_pretrained(str(sig_src), dtype=torch.float16, local_files_only=True)
            sig_m.save_pretrained(str(dest_dir / "specialist/sig_vision"))
            del sig_m

            dino_m = Dinov2Model.from_pretrained(str(dino_src), dtype=torch.float16, local_files_only=True)
            dino_m.save_pretrained(str(dest_dir / "specialist/dino_vision"))
            del dino_m

        link_or_copy(specialist_ckpt, dest_dir / "specialist/specialist_5fold_ensemble.pt")

    # 5. Statutory corpus + compact 11-code case bank (no competition images bundled!)
    link_or_copy(root / "data/baseline/assets/laws.jsonl", dest_dir / "assets/laws.jsonl")
    release_train_rows = list(read_selected_cases(root / "data/train/train.jsonl", train_ids))
    if len(release_train_rows) != len(train_ids):
        raise ValueError("Case-bank source did not cover the training manifest")
    export_compact_case_bank(release_train_rows, dest_dir / "assets/cases_11code.jsonl")

    # 6. Write official assets.json & README.md
    is_boundary = summary.get("version") == "vlm-11code-multimodal-qlora-v2-epoch-boundary"
    sched_desc = (
        f" (Epoch {summary['completed_epochs']}/{summary['configured_epochs']} boundary snapshot, "
        f"{summary['optimizer_steps']}/{summary['scheduled_total_steps']} cosine steps)"
        if is_boundary else ""
    )
    assets_list = [
        {
            "kind": "model",
            "name": base_name,
            "source": f"https://huggingface.co/Qwen/{base_name}",
            "revision": base_revision,
            "license": "Apache-2.0",
            "use": "현장 사진 11종 위험 판정, 근거·예방조치 생성 및 후보 시각 검증 기반 모델",
        },
        {
            "kind": "model",
            "name": "SafeGround v3 11-Code Multimodal LoRA Adapter",
            "source": f"kwater_lab/v3_turnaround.py 및 kwater_lab/vlm_11code.py로 공식 학습셋에서 미세조정(r=16, alpha=32, LLM + DeepStack visual mergers){sched_desc}",
            "revision": f"sha256:{summary['adapter_safetensors_sha256']}",
            "license": "대회 참가 목적 이용 허용",
            "use": "공식 11종 위험코드, 관찰 근거, 예방조치, 가능성·중대성 점수 판정",
        },
        {
            "kind": "data",
            "name": "K-water 해커톤 학습 데이터 및 산업안전보건기준에 관한 규칙",
            "source": "참가자 공개 kwater-train.zip (train.jsonl) 및 동봉 assets/laws.jsonl",
            "revision": "labels_sha256:ab427fed4615b32627b76001e27124e99ae9202c855ba4170916ebe4fdc4de55",
            "license": "대회 참가 목적 이용 허용",
            "use": "11종 도메인 LoRA·전문가 헤드 학습 및 예방조치 행위 직접 지지 OSH-* 법령 검색·재정렬",
        },
        {
            "kind": "library",
            "name": "transformers 4.57.1 외 4종",
            "source": "https://pypi.org/",
            "revision": "transformers==4.57.1, peft==0.17.1, accelerate==1.10.1, huggingface-hub==0.36.0, tokenizers==0.22.1",
            "license": "Apache-2.0",
            "use": "오프라인 컨테이너 환경 Qwen3-VL 및 PEFT 실행(wheelhouse)",
        },
    ]
    if enable_hybrid:
        mg_audit = scope_info["margin_gate_exposure_audit"]
        assets_list.extend([
            {
                "kind": "model",
                "name": "SigLIP 2 So400m/14-384 Vision Tower",
                "source": "https://huggingface.co/google/siglip2-so400m-patch14-384",
                "revision": "e8e48729f2eff3c8178b57bfbefdb9cde2fb12b3",
                "license": "Apache-2.0",
                "use": "다중 뷰(7-View) 의미 특징 추출을 통한 미탐 위험 후보 제안",
            },
            {
                "kind": "model",
                "name": "DINOv2-Large Vision Tower",
                "source": "https://huggingface.co/facebook/dinov2-large",
                "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                "license": "Apache-2.0",
                "use": "다중 뷰(7-View) 구조·형상 특징 추출을 통한 미탐 위험 후보 제안",
            },
            {
                "kind": "model",
                "name": "SafeGround 5-Fold Spatial Dual-Vision Specialist Ensemble",
                "source": (
                    f"kwater_lab/clean_specialist.py 및 kwater_lab/v3_turnaround.py로 공식 학습셋 "
                    f"{scope_info['specialist_train_photos']}장에서 그룹 단위 5-Fold×3-Seed({scope_info['specialist_ensemble_models']}개 head) 학습 후 "
                    f"전체 {scope_info['specialist_train_photos']}개 OOF 행에서 온도·문턱·코드별 정밀도를 일괄 적합({scope_info['specialist_oof_archive']}, 별도 내측 분할 아님)"
                ),
                "revision": f"sha256:{digest(specialist_ckpt)}",
                "license": "대회 참가 목적 이용 허용",
                "use": (
                    f"전체 {scope_info['specialist_ensemble_models']}개 head 평균 확률로 사진당 최대 {max_candidates_per_photo}개 후보 제안, "
                    f"필수 1-토큰 마진 게이트({mg_audit['inner_pairs_n']}쌍·{mg_audit['unique_photos_n']}장·{mg_audit['unique_groups_n']}그룹: "
                    f"동일 (사진,코드) 쌍 검증 감독 중복 {mg_audit['verify_supervision_exact_pair_overlap_count']}건, "
                    f"검증 감독 사진 중복 {mg_audit['verify_supervision_photo_overlap_count']}장·그룹 중복 {mg_audit['verify_supervision_group_overlap_count']}개, "
                    f"기본 inspect 판독 훈련에는 {mg_audit['base_inspect_photo_overlap_count']}장 모두 포함) 및 "
                    f"검증 배치 {verify_batch_size}·최대 {verify_max_new_tokens}토큰 시각 검증"
                ),
            },
        ])

    assets_manifest = {"schema_version": "1.0", "seed": 42, "assets": assets_list}
    (dest_dir / "assets.json").write_text(json.dumps(assets_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    title = f"SafeGround v5 (v3 Engine) · Native 11-Code {base_name.removesuffix('-Instruct')}"
    if enable_hybrid:
        title += " + 1-Token Margin Verified Dual-Vision Specialist"
    purpose = (
        f"- VLM Scope: Fold {summary['val_fold']} validation adapter ({len(release_train_rows)} base inspect training photos){sched_desc}.\n"
        if summary["val_fold"] >= 0 else
        f"- VLM Scope: Full training adapter ({len(release_train_rows)} official training photos; Fold 4 is in-sample and NOT an independent score){sched_desc}.\n"
    )
    if enable_hybrid and margin_gate is not None:
        mg_audit = scope_info["margin_gate_exposure_audit"]
        prov = scope_info["provenance_sha256"]
        hybrid_readme = (
            f"- Specialist Scope: {scope_info['specialist_train_photos']} official photos, "
            f"{scope_info['specialist_ensemble_models']} heads (5 folds x 3 seeds averaged at inference); "
            f"temperatures, code thresholds, scene threshold, and per-code precisions are fit/computed once across all "
            f"{scope_info['specialist_train_photos']} pooled 5-fold OOF rows (`{scope_info['specialist_oof_archive']}`, not a separate inner split).\n"
            f"- Margin Gate & Verifier: mandatory 1-token margin gate (`beta={margin_gate['beta']}`, "
            f"`tau_when_hazard={margin_gate['tau_when_hazard']}`, `tau_when_empty={margin_gate['tau_when_empty']}`), "
            f"calibrated on {mg_audit['inner_pairs_n']} pairs ({mg_audit['unique_photos_n']} photos, {mg_audit['unique_groups_n']} groups; "
            f"exact `(photo, code)` verify-supervision pair overlap = `{mg_audit['verify_supervision_exact_pair_overlap_count']}`, "
            f"verify-supervision photo/group overlap = `{mg_audit['verify_supervision_photo_overlap_count']}` photos / "
            f"`{mg_audit['verify_supervision_group_overlap_count']}` groups, "
            f"base inspect supervision overlap = `{mg_audit['base_inspect_photo_overlap_count']}/{mg_audit['unique_photos_n']}` photos), "
            f"`max_candidates_per_photo={max_candidates_per_photo}`, `inspect_batch_size={batch_size}`, "
            f"`verify_batch_size={verify_batch_size}`, `verify_max_new_tokens={verify_max_new_tokens}`.\n"
            f"- Provenance SHA-256: `adapter={prov['adapter_safetensors_sha256']}`, `prompt={prov['prompt_sha256']}`, "
            f"`gate={prov['margin_gate_calibration_sha256']}`, `inner_pairs={prov['inner_margin_cal_pairs_sha256']}`.\n"
        )
    else:
        hybrid_readme = ""
    (dest_dir / "README.md").write_text(
        f"# {title}\n\n"
        + purpose
        + hybrid_readme
        + "- Entrypoint: `python main.py --input-dir /input --output-dir /output`\n"
        "- Model loading uses local files in a single GPU process.\n",
        encoding="utf-8",
        newline="\n",
    )

    total_bytes = sum(p.stat().st_size for p in dest_dir.rglob("*") if p.is_file())
    zip_bytes = None
    if total_bytes > ZIP_LIMIT_BYTES:
        raise ValueError(f"Staged files exceed the organizer's 20 GB ZIP limit: {total_bytes} bytes")
    if zip_path is not None:
        t_z = time.perf_counter()
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=zip_path.parent, suffix=".zip.part", delete=False) as stream:
            temporary_zip = Path(stream.name)
        try:
            with ZipFile(temporary_zip, "w", compression=ZIP_STORED, allowZip64=True) as zf:
                for file_path in sorted(p for p in dest_dir.rglob("*") if p.is_file()):
                    rel_posix = file_path.relative_to(dest_dir).as_posix()
                    zi = ZipInfo(filename=rel_posix, date_time=(2026, 11, 27, 22, 0, 0))
                    zi.compress_type = ZIP_STORED
                    zi.external_attr = (stat.S_IFREG | 0o644) << 16
                    with file_path.open("rb") as src_f, zf.open(zi, "w", force_zip64=True) as dst_f:
                        shutil.copyfileobj(src_f, dst_f, length=4 * 1024 * 1024)
            zip_bytes = temporary_zip.stat().st_size
            if zip_bytes > ZIP_LIMIT_BYTES:
                raise ValueError(f"Archive exceeds the organizer's 20 GB ZIP limit: {zip_bytes} bytes")
            os.link(temporary_zip, zip_path)
        finally:
            temporary_zip.unlink()
        print(f"Packed {zip_path} ({zip_bytes / 2**30:.2f} GiB) in {time.perf_counter() - t_z:.1f}s", flush=True)

    return {
        "dest_dir": str(dest_dir),
        "unpacked_bytes": total_bytes,
        "unpacked_gib": round(total_bytes / 2**30, 3),
        "zip_path": str(zip_path) if zip_path else None,
        "zip_gib": round(zip_bytes / 2**30, 3) if zip_bytes is not None else None,
        "zip_bytes": zip_bytes,
        "law_case_bank_train_photos": len(release_train_rows),
        "adapter_validation_fold": summary["val_fold"],
        "prompt_filename": prompt_filename,
        "max_pixels_infer": max_pixels_infer,
        "margin_gate_bundled": margin_gate is not None,
        "margin_gate": margin_gate,
        "enable_hybrid": enable_hybrid,
        "batch_size": batch_size,
        "verify_batch_size": verify_batch_size,
        "verify_max_new_tokens": verify_max_new_tokens,
        "max_candidates_per_photo": max_candidates_per_photo,
        "training_and_calibration_scope": scope_info,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--specialist-ckpt", type=Path, default=Path("data/clean-specialist-v1/specialist_5fold_ensemble.pt"))
    ap.add_argument("--prompt-path", type=Path, default=None)
    ap.add_argument("--dest-dir", type=Path, default=Path("data/submissions/safeground_v5_dir"))
    ap.add_argument("--zip-path", type=Path, default=Path("data/submissions/kwater-safeground-v5-20261127.zip"))
    ap.add_argument("--vlm-only", action="store_true")
    ap.add_argument("--allow-fold-hybrid", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--verify-batch-size", type=int, default=2)
    ap.add_argument("--verify-max-new-tokens", type=int, default=PRODUCTION_VERIFY_MAX_NEW_TOKENS)
    ap.add_argument("--max-candidates-per-photo", type=int, default=1)
    ap.add_argument("--research-8b-base-dir", type=Path, default=None)
    args = ap.parse_args()

    info = build_submission_package(
        root=args.root.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
        specialist_ckpt=args.specialist_ckpt.resolve(),
        dest_dir=args.dest_dir.resolve(),
        zip_path=None if args.stage_only else args.zip_path.resolve(),
        enable_hybrid=not args.vlm_only,
        batch_size=args.batch_size,
        verify_batch_size=args.verify_batch_size,
        verify_max_new_tokens=args.verify_max_new_tokens,
        max_candidates_per_photo=args.max_candidates_per_photo,
        prompt_path=args.prompt_path.resolve() if args.prompt_path else None,
        allow_fold_hybrid=args.allow_fold_hybrid,
        research_8b_base_dir=args.research_8b_base_dir,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
