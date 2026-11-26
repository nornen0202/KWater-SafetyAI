"""Self-contained competition submission builder & ZIP packager (`safeground_v5`).

Stages local artifacts and builds the archive for the official local kit.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import torch

from . import clean_specialist, law_reranker_11code, metrics, vlm_11code
from .hybrid_pipeline import merge_verified_candidates, select_booster_candidates, usable_draft_hazards
from .law_reranker_11code import export_compact_case_bank, read_selected_cases
from .audit import digest
from .metrics import CODES, LIKELIHOOD, RISK_BANDS, SEVERITY
from .train_head import save_atomic


MAIN_PY_SOURCE = r'''"""SafeGround v5: Native 11-Code Multimodal Qwen3-VL-4B LoRA + Verified Dual-Vision Specialist."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from functools import lru_cache
import json
from math import log, sqrt
import math
import os
from pathlib import Path
import random
import re
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    seed_all(int(os.environ.get("KWATER_SEED", "42")))
    cfg = json.loads((ROOT / "pipeline_config.json").read_text(encoding="utf-8"))
    manifest_lines = [l for l in (args.input_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    manifest_rows = [json.loads(l) for l in manifest_lines]
    img_by_stem = {p.stem: p for p in (args.input_dir / "images").iterdir() if p.is_file()}
    image_paths = [
        (args.input_dir / r["image"]) if "image" in r and (args.input_dir / str(r["image"])).is_file() else img_by_stem[r["image_id"]]
        for r in manifest_rows
    ]
    image_ids = [r["image_id"] for r in manifest_rows]

    device = torch.device("cuda:0")
    use_hybrid = bool(cfg.get("enable_hybrid_booster", True))
    spec_code_p, spec_scene_p, spec_calib = (None, None, None)
    if use_hybrid:
        spec_code_p, spec_scene_p, spec_calib = run_vision_specialist(image_paths, ROOT / "specialist", device)

    prompt = (ROOT / "system_prompt_11code.txt").read_text(encoding="utf-8")
    max_pixels = cfg["max_pixels_infer"]
    batch_sz = cfg["batch_size"]
    model, proc = load_vlm(ROOT / "base_model", ROOT / "adapter", max_pixels=max_pixels)

    msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]}]
    prefix = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    vlm_outputs = []
    with torch.inference_mode():
        for s in range(0, len(image_paths), batch_sz):
            sub_p = image_paths[s : s + batch_sz]
            pics = []
            for p in sub_p:
                with Image.open(p) as im:
                    pics.append(ImageOps.exif_transpose(im).convert("RGB"))
            batch = proc(text=[prefix] * len(sub_p), images=pics, padding=True, return_tensors="pt").to("cuda")
            gen = model.generate(**batch, do_sample=False, max_new_tokens=INSPECT_MAX_NEW_TOKENS, use_cache=True)
            in_len = batch["input_ids"].shape[-1]
            for li in range(len(sub_p)):
                txt = proc.decode(gen[li, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                parsed, parse_error = parse_json_loose(txt)
                vlm_outputs.append({"parsed": parsed, "parse_error": parse_error})
            del batch, gen

        # Hybrid Specialist -> VLM Visual Verification pass
        if use_hybrid and spec_code_p is not None and spec_calib is not None:
            eval_rows = [{"sample_id": i, "image": p} for i, p in zip(image_ids, image_paths, strict=True)]
            candidates = select_booster_candidates(
                eval_rows, vlm_outputs, spec_code_p, spec_scene_p, spec_calib,
                code_precisions=cfg["specialist_oof_code_precisions"],
                margin_boost_when_hazard=cfg["margin_boost_when_hazard"],
                margin_boost_when_empty=cfg["margin_boost_when_empty"],
            )
            accepted = []
            for cs in range(0, len(candidates), batch_sz):
                chunk = candidates[cs : cs + batch_sz]
                v_txts, v_pics = [], []
                for candidate in chunk:
                    vm = [{"role": "system", "content": prompt}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verify_user_text(candidate["code"])}]}]
                    v_txts.append(proc.apply_chat_template(vm, tokenize=False, add_generation_prompt=True))
                    with Image.open(candidate["image"]) as im:
                        v_pics.append(ImageOps.exif_transpose(im).convert("RGB"))
                vb = proc(text=v_txts, images=v_pics, padding=True, return_tensors="pt").to("cuda")
                vgen = model.generate(**vb, do_sample=False, max_new_tokens=VERIFY_MAX_NEW_TOKENS, use_cache=True)
                vin_len = vb["input_ids"].shape[-1]
                for li, candidate in enumerate(chunk):
                    vtxt = proc.decode(vgen[li, vin_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    hazard = parse_verified_hazard(vtxt, candidate["code"])
                    if hazard is not None:
                        accepted.append({**candidate, "accepted": True, "hazard": hazard})
                del vb, vgen
            print(f"specialist accepted {len(accepted)}/{len(candidates)} verified candidates", flush=True)
            vlm_outputs = merge_verified_candidates(vlm_outputs, accepted)

    law_rows = [json.loads(l) for l in (ROOT / "assets" / "laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    case_rows = [json.loads(l) for l in (ROOT / "assets" / "cases_11code.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    reranker = Native11LawRetriever(law_rows, case_rows)
    parsed_count = sum(item["parse_error"] is None for item in vlm_outputs)
    print(f"raw VLM parsed {parsed_count}/{len(image_ids)} images", flush=True)

    out_lines = [
        json.dumps(format_competition_record(iid, item["parsed"], reranker), ensure_ascii=False, separators=(",", ":")) + "\n"
        for iid, item in zip(image_ids, vlm_outputs, strict=True)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Write directly to submission.jsonl (no leftover temp files in output_dir!)
    (args.output_dir / "submission.jsonl").write_bytes("".join(out_lines).encode("utf-8"))


if __name__ == "__main__":
    main()
'''


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
        "VERIFY_MAX_NEW_TOKENS": vlm_11code.VERIFY_MAX_NEW_TOKENS,
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
    definitions = (
        metrics.unique_keys, metrics.reject_constant, metrics.check_keys, metrics.check_text,
        metrics.check_level, metrics.decode_target,
        law_reranker_11code.article_to_osh_id, law_reranker_11code.extract_terms,
        law_reranker_11code.hazard_query_terms, law_reranker_11code._TFIDFIndex,
        law_reranker_11code.Native11LawRetriever,
        vlm_11code.seed_all, vlm_11code.verify_user_text, vlm_11code.sanitize_text,
        vlm_11code.parse_json_loose, vlm_11code.validate_vlm_11code_obj,
        vlm_11code.verified_hazard, vlm_11code.parse_verified_hazard, vlm_11code.format_competition_record,
        clean_specialist.make_spatial_crops, clean_specialist._pil_list_to_sig_tensor,
        clean_specialist._pil_list_to_dino_tensor, clean_specialist.extract_batch_7view_features,
        clean_specialist.SpatialDualSpecialistHead, clean_specialist.predict_probs,
        usable_draft_hazards, merge_verified_candidates, select_booster_candidates,
    )
    parts.extend(inspect.getsource(definition).rstrip() for definition in definitions)
    source = MAIN_PY_SOURCE.replace("# {{SHARED_RUNTIME}}", "\n\n".join(parts))
    compile(source, "main.py", "exec")
    return source


ZIP_LIMIT_BYTES = 20_000_000_000


def validated_release_inputs(root: Path, adapter_dir: Path, specialist_ckpt: Path, enable_hybrid: bool):
    """Reject incomplete adapters and identify the exact case-bank training population."""
    from safetensors import safe_open

    summary_path = adapter_dir.parent / "training_summary.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_config_path = adapter_dir / "adapter_config.json"
    for path in (adapter_weights, adapter_config_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed native 11-code training artifact required: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("version") != "vlm-11code-multimodal-qlora-v1" or not summary.get("epochs") or len(summary.get("history", ())) != summary["epochs"]:
        raise ValueError("A completed native 11-code training summary is required")
    if summary.get("adapter_safetensors_sha256") != digest(adapter_weights):
        raise ValueError("Adapter weights do not match the completed training summary")
    training_prompt = summary.get("configuration", {}).get("prompt_sha256")
    if training_prompt is not None and training_prompt != digest(root / "kwater_lab/system_prompt_11code.txt"):
        raise ValueError("Bundled prompt differs from the adapter training prompt")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("A PEFT LoRA adapter is required")
    with safe_open(adapter_weights, framework="pt", device="cpu") as weights:
        keys = weights.keys()
        if not all(any(family in key for key in keys) for family in ("language_model.layers", "visual.merger", "visual.deepstack_merger_list")):
            raise ValueError("Adapter is missing language or visual merger weights")

    manifest_path = root / "data/clean-split-v1/split_manifest_dev.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_fold = summary["val_fold"]
    train_ids = {
        row["sample_id"] for row in manifest if row["split"] == "dev" and row["dev_fold"] != validation_fold
    }
    if validation_fold not in {-1, 0, 1, 2, 3, 4} or len(train_ids) != summary.get("train_photos"):
        raise ValueError("Adapter training population differs from the release manifest")
    if enable_hybrid and validation_fold != -1:
        raise ValueError("Production specialist ensemble requires a development-wide adapter; use --vlm-only for fold adapters")
    if enable_hybrid:
        for path in (specialist_ckpt, specialist_ckpt.parent / "dev_oof_predictions.npz"):
            if not path.is_file():
                raise FileNotFoundError(f"Hybrid artifact required: {path}")
    base = root / "data/baseline/base_model"
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
    return summary, train_ids


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_submission_package(
    root: Path,
    adapter_dir: Path,
    specialist_ckpt: Path,
    dest_dir: Path,
    zip_path: Path | None,
    enable_hybrid: bool = True,
    batch_size: int = 4,
) -> dict:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    dest_dir = dest_dir.resolve()
    zip_path = zip_path.resolve() if zip_path is not None else None
    if dest_dir.exists() or (zip_path is not None and zip_path.exists()):
        raise FileExistsError("Use a fresh staging directory and archive path")
    if zip_path is not None and zip_path.is_relative_to(dest_dir):
        raise ValueError("Archive path must be outside the staging directory")
    summary, train_ids = validated_release_inputs(root, adapter_dir, specialist_ckpt, enable_hybrid)
    dest_dir.mkdir(parents=True, exist_ok=False)

    # Compute full-dev OOF code precisions if specialist oof file is available
    oof_code_precs = {}
    oof_path = specialist_ckpt.parent / "dev_oof_predictions.npz"
    if enable_hybrid and oof_path.is_file() and specialist_ckpt.is_file():
        import numpy as np
        from .hybrid_pipeline import compute_calib_code_precisions
        ckpt_tmp = torch.load(specialist_ckpt, map_location="cpu", weights_only=True)
        oof_arc = np.load(oof_path, allow_pickle=False)
        oof_code_precs = {
            k: round(v, 4)
            for k, v in compute_calib_code_precisions(
                oof_arc["code_probs"], oof_arc["scene_probs"], oof_arc["y_codes"], ckpt_tmp["full_dev_calibration"]
            ).items()
        }

    # 1. Write main.py, system_prompt_11code.txt, pipeline_config.json
    (dest_dir / "main.py").write_text(render_entrypoint(), encoding="utf-8", newline="\n")
    shutil.copy2(root / "kwater_lab/system_prompt_11code.txt", dest_dir / "system_prompt_11code.txt")
    save_atomic(
        dest_dir / "pipeline_config.json",
        lambda f: f.write(
            (
                json.dumps(
                    {
                        "enable_hybrid_booster": enable_hybrid,
                        "max_pixels_infer": vlm_11code.CANDIDATE_MAX_PIXELS,
                        "batch_size": batch_size,
                        "margin_boost_when_hazard": 0.04,
                        "margin_boost_when_empty": 0.12,
                        "specialist_oof_code_precisions": oof_code_precs,
                    },
                    indent=2,
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
    for bf in (root / "data/baseline/base_model").iterdir():
        if bf.is_file():
            link_or_copy(bf, dest_dir / "base_model" / bf.name)
    for af in adapter_dir.iterdir():
        if af.is_file() and af.name in ("adapter_config.json", "adapter_model.safetensors"):
            link_or_copy(af, dest_dir / "adapter" / af.name)

    # 4. Export compact FP16 vision-only towers (stripping unused 2.8GB SigLIP text tower!) + specialist heads
    if enable_hybrid:
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
    assets_list = [
        {
            "kind": "model",
            "name": "Qwen3-VL-4B-Instruct",
            "source": "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "license": "Apache-2.0",
            "use": "현장 사진 11종 위험 판정, 근거·예방조치 생성 및 후보 시각 검증 기반 모델",
        },
        {
            "kind": "model",
            "name": "SafeGround 11-Code Multimodal LoRA Adapter",
            "source": "kwater_lab/vlm_11code.py로 공식 학습셋에서 직접 미세조정(r=16, alpha=32, LLM + DeepStack visual mergers)",
            "revision": f"sha256:{summary['adapter_safetensors_sha256']}",
            "license": "대회 참가 목적 이용 허용",
            "use": "공식 11종 위험코드, 관찰 근거, 예방조치, 가능성·중대성 점수 및 후보 검증 판정",
        },
        {
            "kind": "data",
            "name": "K-water 해커톤 학습 데이터 및 산업안전보건기준에 관한 규칙",
            "source": "참가자 공개 kwater-train.zip (train.jsonl) 및 동봉 assets/laws.jsonl",
            "revision": "labels_sha256:ab427fed4615b32627b76001e27124e99ae9202c855ba4170916ebe4fdc4de55",
            "license": "대회 참가 목적 이용 허용",
            "use": "11종 도메인 LoRA·전문가 헤드 학습 및 조문 단위 OSH-* 법령 검색·재정렬",
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
                "source": "kwater_lab/clean_specialist.py로 공식 학습셋에서 5-Fold 앙상블 학습",
                "revision": "clean-specialist-v1",
                "license": "대회 참가 목적 이용 허용",
                "use": "11종 위험코드 다중 뷰 공간 어텐션 확률 산출 및 VLM 검증 후보 제안",
            },
        ])

    assets_manifest = {"schema_version": "1.0", "seed": 42, "assets": assets_list}
    (dest_dir / "assets.json").write_text(json.dumps(assets_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (dest_dir / "README.md").write_text(
        "# SafeGround v5 (Native 11-Code Multimodal Qwen3-VL-4B + Verified Dual-Vision Specialist)\n\n"
        "- Entrypoint: `python main.py --input-dir /input --output-dir /output`\n"
        "- 100% offline, single-process GPU execution (no network or localhost sockets).\n",
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
                    zi = ZipInfo(filename=rel_posix, date_time=(2026, 11, 26, 23, 30, 0))
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
        "unpacked_gib": round(total_bytes / 2**30, 3),
        "zip_path": str(zip_path) if zip_path else None,
        "zip_gib": round(zip_bytes / 2**30, 3) if zip_bytes is not None else None,
        "zip_bytes": zip_bytes,
        "law_case_bank_train_photos": len(release_train_rows),
        "adapter_validation_fold": summary["val_fold"],
        "enable_hybrid": enable_hybrid,
        "batch_size": batch_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--specialist-ckpt", type=Path, default=Path("data/clean-specialist-v1/specialist_5fold_ensemble.pt"))
    ap.add_argument("--dest-dir", type=Path, default=Path("data/submissions/safeground_v5_dir"))
    ap.add_argument("--zip-path", type=Path, default=Path("data/submissions/kwater-safeground-v5-20261127.zip"))
    ap.add_argument("--vlm-only", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    info = build_submission_package(
        root=args.root.resolve(),
        adapter_dir=args.adapter_dir.resolve(),
        specialist_ckpt=args.specialist_ckpt.resolve(),
        dest_dir=args.dest_dir.resolve(),
        zip_path=None if args.stage_only else args.zip_path.resolve(),
        enable_hybrid=not args.vlm_only,
        batch_size=args.batch_size,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
