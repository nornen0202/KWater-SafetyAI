"""Native 11-Code Qwen3-VL-4B Multimodal QLoRA Trainer, Evaluator, and Visual Candidate Verifier.

Key innovations over the 42.04 official baseline:
  1. Preserves 100% of all 11 official competition hazard codes and 100% of training hazards
     (fixing the 3 dropped codes and the distorted `other -> collision_contact` definition).
  2. Multimodal LoRA (`r=16, alpha=32`) jointly adapts BOTH `language_model` attention/MLP layers
     AND `visual.merger` + `visual.deepstack_merger_list.{0,1,2}` vision-to-LLM bridges at higher
     spatial resolution (`448*32*32` train / `512*32*32` infer vs baseline `384*32*32`).
  3. Decision-token weighted cross-entropy (4.5x weight on `status`, `decision`, `code`,
     `likelihood_score`, `severity_score` spans; 1.0x on Korean observation/rationale prose).
  4. Jointly trains a supervised Visual Candidate Verification mode (`accept` with grounded
     `evidence`/`prevention` vs `reject`) so high-confidence candidates from the Dual Vision
     Specialist are visually verified by the VLM before acceptance.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F

from .audit import digest
from .law_reranker_11code import Native11LawRetriever
from .metrics import CODES, LIKELIHOOD, RISK_BANDS, SEVERITY, check_text, decode_target, evaluate, paired_bootstrap, reject_constant, unique_keys
from .train_head import save_atomic


KOR_CODE_NAMES: dict[str, str] = {
    "fall_height": "추락(고소·단부·개구부·사다리)",
    "slip_trip": "전도(미끄러짐·걸려넘어짐·통로장애물)",
    "caught_in_machine": "끼임(기계 회전체·물림점·구동부)",
    "struck_by_lifting": "물체에 맞음·깔림(인양물·낙하·적재물전도)",
    "electrical_energy": "감전·아크(피복손상·노출충전부·분전반파손)",
    "drowning_inundation": "익사·수몰(수로·집수정·침수·수변방호미비)",
    "fire_explosion": "화재·폭발(용접불티·가연물·가스용기)",
    "confined_space": "밀폐공간 질식(맨홀·탱크·피트·환기미비)",
    "chemical_exposure": "유해물질·분진·흄 노출(화학물질·시멘트분진·무방호)",
    "thermal_weather": "고온·한랭·기상 위험(고열물·폭염·한파·강풍)",
    "other": "기타 위험(비계·거푸집·사면 불안정, 중장비·차량 접촉, 출입통제 미비, 돌출철근 등)",
}

MINORITY_CODES = frozenset({
    "chemical_exposure",
    "fire_explosion",
    "electrical_energy",
    "caught_in_machine",
    "drowning_inundation",
    "confined_space",
    "thermal_weather",
})

INSPECT_USER_TEXT = "이 현장사진을 평가하고 지정된 JSON 하나만 출력한다."
CANDIDATE_MAX_PIXELS = 512 * 32 * 32
INSPECT_MAX_NEW_TOKENS = 620
VERIFY_MAX_NEW_TOKENS = 320
DECISION_SPAN_RE = re.compile(
    r'"(?:status|decision|code)"\s*:\s*"[^"]+"|"(?:likelihood_score|severity_score)"\s*:\s*\d+'
)
LORA_TARGET_REGEX = (
    r".*(language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))|"
    r"visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12])$"
)


def verify_user_text(code: str) -> str:
    return (
        f"후보 위험코드 검증: 이 현장사진에서 [{code}] ({KOR_CODE_NAMES[code]}) 위험이 "
        f"실제 불안전 상태로 눈으로 확인되는지 판정하고 지정된 JSON 하나만 출력한다. "
        f'형식: 확인되면 {{"decision":"accept","hazard":{{"code":"{code}","likelihood_score":3,"severity_score":3,'
        f'"evidence":"...","prevention":"..."}}}}, 확인되지 않으면 {{"decision":"reject","reason":"..."}}'
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_text(text: Any, limit: int, fallback: str) -> str:
    s = re.sub(r"[\x00-\x1f\x7f-\x9f\u2028\u2029\ud800-\udfff\ufeff]+", " ", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = fallback
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "."
    return s


def gold_row_to_vlm_target(row: dict) -> dict[str, Any]:
    """Convert official train.jsonl row into concise observation-first 11-code JSON target."""
    t = row["target"]
    notes = row.get("notes") or {}
    if t["status"] == "no_visible_hazard":
        s_ev = sanitize_text(t.get("scene_evidence") or notes.get("scene_note"), 220, "사진에서 확인되는 위험요인이 없다.")
        return {"scene_note": s_ev, "status": "no_visible_hazard", "hazards": []}

    s_note = sanitize_text(notes.get("scene_note"), 200, "현장 작업 구간에서 안전 위험요인이 관찰된다.")
    hz_items = []
    for code, h in t["hazards"].items():
        lv = int(h["risk_level"]["likelihood"]["score"])
        sv = int(h["risk_level"]["severity"]["score"])
        hz_items.append({
            "code": code,
            "likelihood_score": lv,
            "severity_score": sv,
            "evidence": sanitize_text(h["evidence"], 220, "사진에서 불안전 상태가 확인된다."),
            "prevention": sanitize_text(h["prevention"], 420, "해당 위험원 방호 및 안전조치를 실시한다."),
        })
    hz_items.sort(key=lambda x: (-(x["likelihood_score"] * x["severity_score"]), -x["severity_score"], CODES.index(x["code"])))
    return {"scene_note": s_note, "status": "hazard", "hazards": hz_items[:3]}


def parse_json_loose(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    def load(text: str) -> Any:
        parsed = json.loads(text, object_pairs_hook=unique_keys)
        json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return parsed

    s = str(raw_text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        obj = load(s)
        return (obj, None) if isinstance(obj, dict) else (None, "not_dict")
    except ValueError as err:
        # Try extracting outermost JSON braces if trailing token truncation or prefix occurred
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last > first:
            try:
                obj = load(s[first : last + 1])
                if isinstance(obj, dict):
                    return obj, None
            except ValueError:
                pass
        return None, str(err)


def validate_vlm_11code_obj(obj: dict[str, Any] | None) -> list[str]:
    if not isinstance(obj, dict):
        return ["output is not a dict"]
    errs: list[str] = []
    status = obj.get("status", obj.get("scene_state"))
    if status not in ("hazard", "no_visible_hazard"):
        errs.append(f"invalid status: {status!r}")
    hazards = obj.get("hazards")
    if not isinstance(hazards, list):
        return errs + ["hazards must be a list"]
    if status == "hazard" and not (1 <= len(hazards) <= 3):
        errs.append("hazard requires 1..3 items")
    if status == "no_visible_hazard" and len(hazards) != 0:
        errs.append("no_visible_hazard requires empty hazards")
    seen = set()
    for i, h in enumerate(hazards):
        if not isinstance(h, dict):
            errs.append(f"hazards[{i}] not dict")
            continue
        code = h.get("code")
        if not isinstance(code, str) or code not in CODES:
            errs.append(f"invalid code: {code!r}")
        elif code in seen:
            errs.append(f"duplicate code: {code}")
        else:
            seen.add(code)
        lv = h.get("likelihood_score")
        sv = h.get("severity_score")
        if type(lv) is not int or not (1 <= lv <= 5):
            errs.append(f"invalid likelihood_score: {lv!r}")
        if type(sv) is not int or not (1 <= sv <= 4):
            errs.append(f"invalid severity_score: {sv!r}")
        if not isinstance(h.get("evidence"), str) or not h["evidence"].strip():
            errs.append(f"empty evidence at {i}")
        if not isinstance(h.get("prevention"), str) or not h["prevention"].strip():
            errs.append(f"empty prevention at {i}")
    return errs


def verified_hazard(obj: dict[str, Any] | None, code: str) -> dict[str, Any] | None:
    if not isinstance(obj, dict) or obj.keys() != {"decision", "hazard"} or obj["decision"] != "accept":
        return None
    hazard = obj.get("hazard")
    if (not isinstance(hazard, dict)
            or hazard.keys() != {"code", "likelihood_score", "severity_score", "evidence", "prevention"}
            or hazard["code"] != code):
        return None
    if validate_vlm_11code_obj({"status": "hazard", "hazards": [hazard]}):
        return None
    try:
        check_text(hazard["evidence"], "verification evidence", 240).encode("utf-8")
        check_text(hazard["prevention"], "verification prevention", 480).encode("utf-8")
    except ValueError:
        return None
    if not all(any(char.isalnum() for char in hazard[key]) for key in ("evidence", "prevention")):
        return None
    evidence = hazard["evidence"].strip()
    denies_risk = re.search(
        r"(?:위험(?:요인|요소|상황)?|불안전\s*상태)(?:은|는|이|가|도)?\s*"
        r"(?:(?:확인|관찰|발견)되지\s*않|(?:확인|관찰|발견)할\s*수\s*없|보이지\s*않|없(?:다|음|습니다))"
        r"|\bno\s+(?:visible\s+)?(?:hazard|risk)(?:\s+(?:is\s+)?(?:visible|observed|detected))?\s*[.!]?$",
        evidence, flags=re.IGNORECASE,
    )
    if denies_risk:
        return None
    return {
        "code": code,
        "likelihood_score": hazard["likelihood_score"],
        "severity_score": hazard["severity_score"],
        "evidence": hazard["evidence"].strip(),
        "prevention": hazard["prevention"].strip(),
    }


def parse_verified_hazard(text: str, code: str) -> dict[str, Any] | None:
    """Accept only the complete verifier JSON object specified in the training prompt."""
    try:
        obj = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except (TypeError, ValueError, RecursionError):
        return None
    return verified_hazard(obj, code)


def format_competition_record(
    image_id: str,
    vlm_obj: dict[str, Any] | None,
    law_retriever: Native11LawRetriever,
) -> dict[str, Any]:
    """Deterministic schema-guaranteed converter to official v1.0 competition row."""
    scene_note = sanitize_text((vlm_obj or {}).get("scene_note"), 240, "판독 결과에 현장 관찰 설명이 없습니다.")
    raw_status = (vlm_obj or {}).get("status", (vlm_obj or {}).get("scene_state"))
    raw_hazards = (vlm_obj or {}).get("hazards") if isinstance((vlm_obj or {}).get("hazards"), list) else []

    hazards_out: dict[str, Any] = {}
    if raw_status == "hazard" or (raw_status is None and raw_hazards):
        for h in raw_hazards:
            if not isinstance(h, dict):
                continue
            code = str(h.get("code") or "").strip()
            if code not in CODES or code in hazards_out or len(hazards_out) >= 3:
                continue
            lv_raw = h.get("likelihood_score", 3)
            sv_raw = h.get("severity_score", 3)
            lv = lv_raw if type(lv_raw) is int and 1 <= lv_raw <= 5 else 3
            sv = sv_raw if type(sv_raw) is int and 1 <= sv_raw <= 4 else 3
            prod = lv * sv
            r_grade = next(g for top, g in RISK_BANDS if prod <= top)

            ev = sanitize_text(h.get("evidence"), 240, f"현장 내 {KOR_CODE_NAMES[code]} 관련 불안전 상태가 확인된다.")
            pr = sanitize_text(h.get("prevention"), 480, f"{KOR_CODE_NAMES[code]} 예방을 위한 방호조치 및 작업 전 안전점검을 실시한다.")
            refs = law_retriever.select_osh_references(code, ev, pr, scene_note, max_refs=2)

            entry: dict[str, Any] = {
                "evidence": ev,
                "prevention": pr,
                "risk_level": {
                    "likelihood": {"grade": LIKELIHOOD[lv - 1], "score": lv},
                    "severity": {"grade": SEVERITY[sv - 1], "score": sv},
                    "risk": {"grade": r_grade, "score": prod},
                },
            }
            if refs:
                entry["references"] = refs
            hazards_out[code] = entry

    if hazards_out:
        record = {
            "schema_version": "1.0",
            "image_id": image_id,
            "status": "hazard",
            "hazards": hazards_out,
        }
    else:
        if raw_status != "no_visible_hazard" or raw_hazards:
            scene_note = "사진 판독 결과에서 위험 유무를 확정할 수 없습니다."
        record = {
            "schema_version": "1.0",
            "image_id": image_id,
            "status": "no_visible_hazard",
            "hazards": {},
            "scene_evidence": scene_note,
        }

    # Final strict check via metrics.decode_target
    decode_target(record, local_ids=not re.fullmatch(r"[0-9a-f]{32}", image_id))
    return record


def build_training_curriculum(
    train_split_rows: list[dict],
    raw_train_by_id: dict[str, dict],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Construct primary 11-code inspection + minority oversampling + supervised visual verification curriculum."""
    rng = random.Random(seed)
    curriculum: list[dict[str, Any]] = []

    # 1. Primary 11-code scene inspection examples (100% of training split) + 1x oversampling on minority classes
    for m_row in train_split_rows:
        sid = m_row["sample_id"]
        g_row = raw_train_by_id[sid]
        target_json = json.dumps(gold_row_to_vlm_target(g_row), ensure_ascii=False, separators=(",", ":"))
        curriculum.append({
            "sample_id": sid,
            "image": m_row["image"],
            "user_text": INSPECT_USER_TEXT,
            "assistant_text": target_json,
            "augment": False,
            "kind": "inspect",
        })
        if set(m_row["codes"]) & MINORITY_CODES:
            curriculum.append({
                "sample_id": sid + "__min_aug",
                "image": m_row["image"],
                "user_text": INSPECT_USER_TEXT,
                "assistant_text": target_json,
                "augment": True,
                "kind": "inspect_minority",
            })

    # 2. Supervised verification pairs. Reject examples cover both confusing hazards and safe scenes.
    pos_candidates = []
    hard_neg_candidates = []
    safe_neg_candidates = []
    confusable_map = {
        "slip_trip": ["electrical_energy", "fall_height", "other"],
        "fall_height": ["other", "struck_by_lifting", "slip_trip"],
        "other": ["fall_height", "struck_by_lifting", "caught_in_machine"],
        "struck_by_lifting": ["caught_in_machine", "other", "fall_height"],
        "electrical_energy": ["fire_explosion", "slip_trip"],
        "fire_explosion": ["chemical_exposure", "electrical_energy"],
    }

    for m_row in train_split_rows:
        sid = m_row["sample_id"]
        g_row = raw_train_by_id[sid]
        t = g_row["target"]
        present = set(m_row["codes"])

        if t["status"] == "hazard":
            for code, h in t["hazards"].items():
                ans = json.dumps(
                    {
                        "decision": "accept",
                        "hazard": {
                            "code": code,
                            "likelihood_score": int(h["risk_level"]["likelihood"]["score"]),
                            "severity_score": int(h["risk_level"]["severity"]["score"]),
                            "evidence": sanitize_text(h["evidence"], 220, "불안전 상태 확인."),
                            "prevention": sanitize_text(h["prevention"], 420, "안전조치 실시."),
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                weight_prio = 2 if (code in MINORITY_CODES or len(present) >= 2) else 1
                pos_candidates.append((weight_prio, sid, m_row["image"], code, ans))

            # Confusable absent code on a hazard scene
            first_c = next(iter(t["hazards"]))
            for absent_c in confusable_map.get(first_c, ["fall_height", "slip_trip", "other"]):
                if absent_c not in present:
                    rej = json.dumps(
                        {"decision": "reject", "reason": f"사진에서 {KOR_CODE_NAMES[absent_c]}에 해당하는 직접적인 불안전 상태는 확인되지 않는다."},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    hard_neg_candidates.append((sid, m_row["image"], absent_c, rej))
                    break
        else:
            # Safe / no_visible_hazard scene tested against plausible false-alarm codes
            s_ev = sanitize_text(t.get("scene_evidence"), 180, "안전상태가 유지되어 위험이 확인되지 않는다.")
            cand_c = rng.choice(["fall_height", "slip_trip", "other", "struck_by_lifting", "electrical_energy"])
            rej = json.dumps(
                {"decision": "reject", "reason": s_ev},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            safe_neg_candidates.append((sid, m_row["image"], cand_c, rej))

    rng.shuffle(pos_candidates)
    rng.shuffle(hard_neg_candidates)
    rng.shuffle(safe_neg_candidates)
    pos_candidates.sort(key=lambda x: -x[0])
    rejects = [(item, "verify_neg_hard") for item in hard_neg_candidates[:18]]
    rejects += [(item, "verify_neg_safe") for item in safe_neg_candidates[: 35 - len(rejects)]]
    rejects += [(item, "verify_neg_hard") for item in hard_neg_candidates[18 : 18 + 35 - len(rejects)]]

    for _, sid, img_rel, c_code, ans in pos_candidates[:35]:
        curriculum.append({
            "sample_id": f"{sid}__ver_pos_{c_code}",
            "image": img_rel,
            "user_text": verify_user_text(c_code),
            "assistant_text": ans,
            "augment": False,
            "kind": "verify_pos",
        })
    for (sid, img_rel, c_code, ans), kind in rejects:
        curriculum.append({
            "sample_id": f"{sid}__ver_neg_{c_code}",
            "image": img_rel,
            "user_text": verify_user_text(c_code),
            "augment": False,
            "assistant_text": ans,
            "kind": kind,
        })

    return curriculum


def load_qwen_processor(base_model_dir: Path, max_pixels: int):
    from transformers import AutoProcessor

    min_px = 128 * 32 * 32
    processor = AutoProcessor.from_pretrained(
        str(base_model_dir), min_pixels=min_px, max_pixels=max_pixels, local_files_only=True
    )
    processor.image_processor.size = {"shortest_edge": min_px, "longest_edge": max_pixels}
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = min_px
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = max_pixels
    return processor


def load_qwen_model(base_model_dir: Path, use_4bit: bool = True, adapter_dir: Path | None = None):
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if use_4bit:
        # Keep the compact ViT + DeepStack mergers in crisp native BF16 while 4-bit NF4 quantizing the 3.7B LLM
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model_dir),
            quantization_config=bnb_cfg,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base_model_dir),
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map={"": 0},
            attn_implementation="sdpa",
        )

    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model


def load_qwen_4bit_or_bf16(
    base_model_dir: Path,
    max_pixels: int,
    use_4bit: bool = True,
    adapter_dir: Path | None = None,
):
    processor = load_qwen_processor(base_model_dir, max_pixels)
    return load_qwen_model(base_model_dir, use_4bit, adapter_dir), processor


def encode_weighted_example(
    processor: Any,
    item: dict[str, Any],
    image_root: Path,
    system_prompt: str,
    rng: random.Random,
    decision_weight: float = 4.5,
) -> dict[str, torch.Tensor]:
    with Image.open(image_root / item["image"]) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    if item.get("augment"):
        # Geometry and legible markings must agree with the original text target.
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.90, 1.10))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.90, 1.12))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["user_text"]}]},
    ]
    prefix_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_batch = processor(text=[prefix_str], images=[img], return_tensors="pt")

    ans_str = item["assistant_text"]
    messages_full = messages + [{"role": "assistant", "content": ans_str}]
    full_str = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
    full_batch = processor(text=[full_str], images=[img], return_tensors="pt")

    n_pref = prefix_batch["input_ids"].shape[-1]
    if not torch.equal(full_batch["input_ids"][:, :n_pref], prefix_batch["input_ids"]):
        raise RuntimeError(f"Assistant prefix tokenization changed for {item['sample_id']}")
    labels = full_batch["input_ids"].clone()
    labels[:, :n_pref] = -100

    # Build per-token weights (4.5x on status/decision/code/likelihood_score/severity_score spans)
    weights = torch.zeros_like(labels, dtype=torch.float32)
    weights[:, n_pref:] = 1.0

    spans = [m.span() for m in DECISION_SPAN_RE.finditer(ans_str)]
    if spans:
        tok_ans = processor.tokenizer(ans_str, add_special_tokens=False, return_offsets_mapping=True)
        answer_ids = tok_ans["input_ids"]
        if full_batch["input_ids"][0, n_pref : n_pref + len(answer_ids)].tolist() != answer_ids:
            raise RuntimeError(f"Assistant answer tokenization changed for {item['sample_id']}")
        for k, (cs, ce) in enumerate(tok_ans["offset_mapping"]):
            if ce > cs and any(cs < se and ce > ss for ss, se in spans):
                weights[0, n_pref + k] = decision_weight

    full_batch["labels"] = labels
    full_batch["token_weights"] = weights
    full_batch["n_prefix"] = torch.tensor(n_pref, dtype=torch.long)
    return full_batch


def forward_and_weighted_assistant_loss(
    model: Any,
    batch: dict[str, torch.Tensor],
    token_weights: torch.Tensor,
    n_prefix: int,
) -> tuple[torch.Tensor, int]:
    """Pop labels before forward and use logits_to_keep so lm_head + FP32 CE run ONLY on assistant tokens (~0.35 GiB vs 5.0 GiB)."""
    labels = batch.pop("labels")
    seq_len = int(batch["input_ids"].shape[-1])
    # Token at position t >= n_prefix is predicted by logit at position t - 1 -> last (seq_len - n_prefix + 1) positions
    n_keep = max(2, seq_len - n_prefix + 1)
    out = model(**batch, logits_to_keep=n_keep, use_cache=False)
    kept_logits = out.logits  # (1, n_keep, V)
    shift_logits = kept_logits[:, :-1, :].float()
    shift_labels = labels[:, -n_keep + 1 :].contiguous()
    shift_weights = token_weights[:, -n_keep + 1 :].contiguous().float()

    active_mask = shift_labels != -100
    n_sup = int(active_mask.sum().item())
    if n_sup == 0:
        raise RuntimeError("No supervised assistant tokens in batch")

    per_tok_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    eff_weights = shift_weights * active_mask.float()
    loss = (per_tok_ce * eff_weights).sum() / eff_weights.sum().clamp(min=1.0)
    return loss, n_sup


def train_multimodal_qlora(
    root: Path,
    manifest_path: Path,
    prompt_path: Path,
    out_dir: Path,
    val_fold: int = 0,
    epochs: int = 2,
    grad_accum: int = 16,
    lr: float = 1.4e-4,
    max_pixels: int = 448 * 32 * 32,
    seed: int = 42,
    preflight_only: bool = False,
    checkpoint_steps: int = 5,
    resume_from: Path | None = None,
    stop_after_steps: int | None = None,
    curriculum_builder=None,
    experiment_identity: dict | None = None,
    initialization_observer=None,
) -> dict[str, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    if epochs < 1 or grad_accum < 1 or checkpoint_steps < 1:
        raise ValueError("epochs, grad_accum, and checkpoint_steps must be positive")
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    if resume_from is not None and (preflight_only or resume_from.parent != out_dir):
        raise ValueError("Resume requires train mode and a checkpoint inside out_dir")
    if not preflight_only and resume_from is None and (out_dir / "steps.jsonl").exists():
        raise FileExistsError(f"Training log already exists in {out_dir}. Resume or use a new directory.")

    seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if val_fold >= 0:
        train_split_rows = [r for r in manifest if r["split"] == "dev" and int(r["dev_fold"]) != val_fold]
    else:
        train_split_rows = [r for r in manifest if r["split"] == "dev"]

    train_ids = {row["sample_id"] for row in train_split_rows}
    raw_train_by_id = {}
    with (root / "data/train/train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line, object_pairs_hook=unique_keys)
                if row["sample_id"] in train_ids:
                    raw_train_by_id[row["sample_id"]] = row

    system_prompt = prompt_path.read_text(encoding="utf-8")
    build_curriculum = curriculum_builder or build_training_curriculum
    curriculum = build_curriculum(train_split_rows, raw_train_by_id, seed=seed)
    kind_counts = Counter(x["kind"] for x in curriculum)
    if not curriculum:
        raise ValueError("The training split has no examples")
    steps_per_epoch = math.ceil(len(curriculum) / grad_accum)
    total_steps = steps_per_epoch * epochs
    base_model_dir = root / "data/baseline/base_model"
    configuration = {
        "val_fold": val_fold,
        "epochs": epochs,
        "grad_accum": grad_accum,
        "lr": lr,
        "max_pixels": max_pixels,
        "seed": seed,
        "curriculum_sha256": hashlib.sha256(json.dumps(curriculum, sort_keys=True).encode()).hexdigest(),
        "prompt_sha256": digest(prompt_path),
        "base_config_sha256": digest(base_model_dir / "config.json"),
    }
    if experiment_identity is not None:
        configuration["experiment"] = experiment_identity
    resume = None
    if resume_from is not None:
        resume = torch.load(resume_from / "training_state.pt", map_location="cpu", weights_only=True)
        if resume["configuration"] != configuration:
            raise ValueError("Training configuration differs from the checkpoint")
        if stop_after_steps is not None and stop_after_steps <= resume["global_step"]:
            raise ValueError("stop_after_steps must exceed the checkpoint's global step")
        for path in out_dir.iterdir():
            match = re.fullmatch(r"(checkpoint|epoch)-(\d+)", path.name)
            if match and (path / "training_state.pt").exists():
                saved_step = int(match[2]) * (steps_per_epoch if match[1] == "epoch" else 1)
                if saved_step > resume["global_step"]:
                    raise ValueError(f"Found a newer completed checkpoint: {path}")
    print(f"Built training curriculum (val_fold={val_fold}): total={len(curriculum)} kinds={dict(kind_counts)}", flush=True)

    model, proc = load_qwen_4bit_or_bf16(base_model_dir, max_pixels=max_pixels, use_4bit=True)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_REGEX,
    )
    if resume_from is None:
        model = get_peft_model(model, lora_cfg)
    else:
        model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if resume is None and initialization_observer is not None:
        initial = hashlib.sha256()
        for name, tensor in sorted(trainable, key=lambda entry: entry[0]):
            initial.update(repr((name, tuple(tensor.shape), str(tensor.dtype))).encode())
            initial.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        initialization_observer(initial.hexdigest())
    target_names = {n.split(".lora_")[0] for n, _ in trainable if ".lora_" in n}
    valid_targets = all(".lora_" in n and re.fullmatch(LORA_TARGET_REGEX, n.split(".lora_")[0]) for n, _ in trainable)
    expected_vision = {"merger"} | {
        f"deepstack_merger_list.{idx}" for idx, _ in enumerate(model.base_model.model.model.visual.deepstack_merger_list)
    }
    expected_vision_targets = {f"{merger}.linear_fc{layer}" for merger in expected_vision for layer in (1, 2)}
    vision_targets = {n.split(".visual.", 1)[1] for n in target_names if ".visual." in n}
    if not valid_targets or vision_targets != expected_vision_targets or not any(".language_model." in n for n in target_names):
        raise RuntimeError(f"Unexpected trainable layers; vision targets: {sorted(vision_targets)}")
    n_trainable = sum(p.numel() for _, p in trainable)
    print(f"Multimodal QLoRA active: trainable_params={n_trainable:,} (LLM+DeepStack mergers verified!)", flush=True)

    # Separate slightly lower LR for visual mergers vs LLM adapters for smooth multimodal alignment
    vis_params = [p for n, p in trainable if "visual." in n]
    llm_params = [p for n, p in trainable if "visual." not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": llm_params, "lr": lr, "weight_decay": 0.01},
            {"params": vis_params, "lr": lr * 0.65, "weight_decay": 0.01},
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(2, math.ceil(total_steps * 0.05)), total_steps)

    image_root = root / "data/train"
    rng = random.Random(seed)

    if resume is None:
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        sample_item = max(curriculum[:80], key=lambda x: len(x["assistant_text"]))
        batch = encode_weighted_example(proc, sample_item, image_root, system_prompt, rng).to("cuda")
        t_w = batch.pop("token_weights")
        n_pref = int(batch.pop("n_prefix").item())
        loss, n_sup = forward_and_weighted_assistant_loss(model, batch, t_w, n_pref)
        loss.backward()
        llm_grad_ok = any(p.grad is not None and bool((p.grad != 0).any()) for p in llm_params)
        merger_gradients = {
            merger: any(
                p.grad is not None and bool((p.grad != 0).any())
                for n, p in trainable if f".visual.{merger}." in n
            )
            for merger in sorted(expected_vision)
        }
        vis_grad_ok = all(merger_gradients.values())
        if not (llm_grad_ok and vis_grad_ok):
            raise RuntimeError(f"Missing gradients: llm={llm_grad_ok}, vision={merger_gradients}")
        torch.cuda.synchronize()
        pf_stats = {
            "status": "ok",
            "sample_id": sample_item["sample_id"],
            "seq_tokens": int(batch["input_ids"].shape[-1]),
            "supervised_tokens": n_sup,
            "high_weight_decision_tokens": int((t_w > 1.5).sum().item()),
            "loss": round(float(loss.detach().item()), 4),
            "trainable_params": n_trainable,
            "llm_grad_ok": llm_grad_ok,
            "vis_merger_grad_ok": vis_grad_ok,
            "merger_gradients": merger_gradients,
            "seconds": round(time.perf_counter() - t0, 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        del batch, t_w, loss
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        save_atomic(out_dir / "preflight.json", lambda f: f.write((json.dumps(pf_stats, indent=2) + "\n").encode("utf-8")))
        print("PREFLIGHT_OK " + json.dumps(pf_stats), flush=True)
        if preflight_only:
            return pf_stats

    start_epoch = 1
    history: list[dict[str, Any]] = []
    global_step = 0
    records: list[dict[str, Any]] = []
    steps_path = out_dir / "steps.jsonl"
    elapsed = 0.0
    t_all = time.perf_counter()
    if resume is not None:
        optimizer.load_state_dict(resume.pop("optimizer"))
        scheduler.load_state_dict(resume["scheduler"])
        torch.set_rng_state(resume["cpu_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"])
        start_epoch = resume["next_epoch"]
        global_step = resume["global_step"]
        history = resume["history"]
        records = resume["records"]
        elapsed = resume["elapsed"]
        save_atomic(steps_path, lambda f: f.write("".join(json.dumps(rec) + "\n" for rec in records).encode()))
        print(f"RESUME_OK step={global_step}/{total_steps}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        ep_rng = random.Random(seed + epoch * 1000)
        ordered = curriculum.copy()
        ep_rng.shuffle(ordered)

        ep_t0 = time.perf_counter()
        ep_loss_sum = 0.0
        ep_tok_sum = 0

        for b_start in range(0, len(ordered), grad_accum):
            step_t0 = time.perf_counter()
            group = ordered[b_start : b_start + grad_accum]
            if resume is not None and epoch == start_epoch and b_start < resume["next_offset"]:
                if b_start + grad_accum >= resume["next_offset"]:
                    ep_rng.setstate(resume["epoch_rng"])
                    ep_loss_sum = resume["epoch_loss_sum"]
                    ep_tok_sum = resume["epoch_tok_sum"]
                continue
            optimizer.zero_grad(set_to_none=True)
            grp_loss = 0.0
            step_supervised = step_decision_tokens = 0

            for item in group:
                batch = encode_weighted_example(proc, item, image_root, system_prompt, ep_rng).to("cuda")
                t_w = batch.pop("token_weights")
                n_pref = int(batch.pop("n_prefix").item())
                loss, n_sup = forward_and_weighted_assistant_loss(model, batch, t_w, n_pref)
                if experiment_identity is not None:
                    step_supervised += n_sup
                    step_decision_tokens += int((t_w > 1.5).sum().item())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss on {item['sample_id']}")
                (loss / len(group)).backward()
                lv = float(loss.detach().item())
                ep_loss_sum += lv * n_sup
                ep_tok_sum += n_sup
                grp_loss += lv / len(group)
                del batch, t_w, loss

            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True).item())
            applied_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            global_step += 1

            step_rec = {
                "epoch": epoch,
                "step": global_step,
                "total_steps": total_steps,
                "loss": round(grp_loss, 4),
                "applied_lr": applied_lr,
                "lr": round(scheduler.get_last_lr()[0], 7),
                "grad_norm": round(grad_norm, 4),
                "seconds": round(time.perf_counter() - step_t0, 2),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "res_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            }
            if experiment_identity is not None:
                step_rec.update(examples=len(group), supervised_tokens=step_supervised, decision_tokens=step_decision_tokens)
            records.append(step_rec)
            with steps_path.open("a", encoding="utf-8") as sf:
                sf.write(json.dumps(step_rec) + "\n")
            if global_step <= 3 or global_step % 5 == 0 or global_step == total_steps:
                print("train_step " + json.dumps(step_rec), flush=True)

            next_offset = b_start + grad_accum
            epoch_done = next_offset >= len(ordered)
            stop_now = global_step == stop_after_steps and global_step < total_steps
            if epoch_done or stop_now or global_step == 2 or global_step % checkpoint_steps == 0:
                ckpt_dir = out_dir / (f"epoch-{epoch}" if epoch_done else f"checkpoint-{global_step}")
                if epoch_done:
                    ep_entry = {
                        "epoch": epoch,
                        "weighted_token_loss": round(ep_loss_sum / max(ep_tok_sum, 1), 4),
                        "epoch_seconds": round(time.perf_counter() - ep_t0, 1),
                        "checkpoint": str(ckpt_dir),
                        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    }
                    history.append(ep_entry)
                state = {
                    "configuration": configuration,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "epoch_rng": ep_rng.getstate(),
                    "epoch_loss_sum": ep_loss_sum,
                    "epoch_tok_sum": ep_tok_sum,
                    "next_epoch": epoch + int(epoch_done),
                    "next_offset": 0 if epoch_done else next_offset,
                    "global_step": global_step,
                    "history": history,
                    "records": records,
                    "elapsed": elapsed + time.perf_counter() - t_all,
                }
                with TemporaryDirectory(dir=out_dir, prefix=".checkpoint-") as temp:
                    staged = Path(temp) / "checkpoint"
                    model.save_pretrained(str(staged))
                    proc.save_pretrained(str(staged))
                    torch.save(state, staged / "training_state.pt")
                    staged.rename(ckpt_dir)
                print(f"CHECKPOINT_OK step={global_step} path={ckpt_dir}", flush=True)
                if epoch_done:
                    save_atomic(out_dir / "history.json", lambda f: f.write((json.dumps(history, indent=2) + "\n").encode("utf-8")))
                    print("EPOCH_DONE " + json.dumps(ep_entry), flush=True)
                if stop_now:
                    return {"status": "paused", "optimizer_steps": global_step, "checkpoint": str(ckpt_dir)}

    final_adapter = out_dir / "adapter"
    if not final_adapter.exists():
        with TemporaryDirectory(dir=out_dir, prefix=".adapter-") as temp:
            staged = Path(temp) / "adapter"
            shutil.copytree(history[-1]["checkpoint"], staged, ignore=shutil.ignore_patterns("training_state.pt"))
            staged.rename(final_adapter)
    if digest(final_adapter / "adapter_model.safetensors") != digest(Path(history[-1]["checkpoint"]) / "adapter_model.safetensors"):
        raise RuntimeError(f"Existing final adapter differs from the last epoch in {out_dir}")

    summary = {
        "version": "vlm-11code-multimodal-qlora-v2",
        "configuration": configuration,
        "val_fold": val_fold,
        "train_photos": len(train_split_rows),
        "curriculum_examples": len(curriculum),
        "curriculum_breakdown": dict(kind_counts),
        "epochs": epochs,
        "grad_accum": grad_accum,
        "lr": lr,
        "max_pixels": max_pixels,
        "trainable_params": n_trainable,
        "optimizer_steps": global_step,
        "total_seconds": round(elapsed + time.perf_counter() - t_all, 1),
        "history": history,
        "final_adapter": str(final_adapter),
        "adapter_safetensors_sha256": digest(final_adapter / "adapter_model.safetensors"),
    }
    save_atomic(out_dir / "training_summary.json", lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    print("TRAIN_COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


@torch.inference_mode()
def infer_batched_vlm(
    model: Any,
    proc: Any,
    eval_rows: list[dict],
    image_root: Path,
    system_prompt: str,
    batch_size: int = 4,
    max_new_tokens: int = INSPECT_MAX_NEW_TOKENS,
    cache_path: Path | None = None,
    cache_key: str | None = None,
    model_factory=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Batch size and max_new_tokens must be positive")
    if (model is None) == (model_factory is None):
        raise ValueError("Supply a model or a lazy model factory")
    proc.tokenizer.padding_side = "left"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]},
    ]
    prefix = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    ids = {r["sample_id"]: r["image"] for r in eval_rows}
    if len(ids) != len(eval_rows):
        raise ValueError("Inference requires unique sample IDs")
    outputs: dict[str, dict[str, Any]] = {}
    previous_seconds = 0.0
    effective_batch_size = batch_size
    if cache_path is not None:
        if not cache_key:
            raise ValueError("cache_key must identify model weights and precision")
        cache_identity = {
            "version": 2,
            "image_root": str(image_root.resolve()),
            "samples": [[sample_id, name] for sample_id, name in ids.items()],
            "image_sha256": {name: digest(image_root / name) for name in sorted(set(ids.values()))},
            "model_key": cache_key,
            "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "processor_size": dict(proc.image_processor.size),
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
        }
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
            if cache["identity"] != cache_identity:
                raise ValueError(f"Inference settings differ from the raw cache: {cache_path}")
            outputs = cache["outputs"]
            if list(outputs) != list(ids)[:len(outputs)] or any(key != row["sample_id"] for key, row in outputs.items()):
                raise ValueError(f"Unexpected raw cache sample IDs: {cache_path}")
            previous_seconds = cache["seconds"]
            effective_batch_size = cache.get("effective_batch_size", batch_size)
            if not 1 <= effective_batch_size <= batch_size:
                raise ValueError(f"Invalid raw cache batch size: {cache_path}")
            print(f"  vlm_cache {len(outputs)}/{len(eval_rows)}", flush=True)
    reused_rows = len(outputs)
    pending = eval_rows[reused_rows:]
    if pending:
        if model is None:
            model = model_factory()
        model.eval()
    t0 = time.perf_counter()

    position = 0
    while position < len(pending):
        chunk = pending[position : position + effective_batch_size]
        pics = []
        for r in chunk:
            with Image.open(image_root / r["image"]) as im:
                pics.append(ImageOps.exif_transpose(im).convert("RGB"))
        inputs = None
        oom = False
        try:
            inputs = proc(text=[prefix] * len(chunk), images=pics, padding=True, return_tensors="pt").to("cuda")
            gen_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            oom = True
        if oom:
            del inputs
            torch.cuda.empty_cache()
            effective_batch_size = max(1, len(chunk) // 2)
            print(f"  vlm_batch_backoff {effective_batch_size} after {len(outputs)} completed images", flush=True)
            continue
        in_len = inputs["input_ids"].shape[-1]
        for local_i, r in enumerate(chunk):
            new_toks = gen_ids[local_i, in_len:]
            # Count non-pad generated tokens
            pad_id = proc.tokenizer.pad_token_id
            eff_len = int((new_toks != pad_id).sum().item()) if pad_id is not None else len(new_toks)
            txt = proc.decode(new_toks, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            obj, p_err = parse_json_loose(txt)
            if p_err is None and obj is not None:
                s_errs = validate_vlm_11code_obj(obj)
            else:
                s_errs = [p_err or "parse_failed"]
            outputs[r["sample_id"]] = {
                "sample_id": r["sample_id"],
                "raw_text": txt,
                "parsed": obj,
                "parse_error": p_err,
                "vlm_schema_errors": s_errs,
                "token_cap_hit": eff_len >= max_new_tokens,
            }
        del inputs, gen_ids
        position += len(chunk)
        elapsed = previous_seconds + time.perf_counter() - t0
        if cache_path is not None:
            result = {
                "identity": cache_identity,
                "outputs": outputs,
                "seconds": elapsed,
                "effective_batch_size": effective_batch_size,
            }
            save_atomic(cache_path, lambda f: f.write(json.dumps(result).encode("utf-8")))
        if len(outputs) % 24 == 0 or len(outputs) == len(eval_rows):
            print(f"  vlm_infer {len(outputs)}/{len(eval_rows)} elapsed={elapsed:.1f}s", flush=True)

    if outputs.keys() != ids.keys():
        raise RuntimeError("VLM inference did not cover all requested images")
    parse_ok = sum(row["parse_error"] is None and row["parsed"] is not None for row in outputs.values())
    vlm_schema_ok = sum(not row["vlm_schema_errors"] for row in outputs.values())
    elapsed = previous_seconds + time.perf_counter() - t0
    gen_stats = {
        "photos": len(eval_rows),
        "raw_json_parse_success": parse_ok,
        "raw_json_parse_rate": round(parse_ok / max(len(eval_rows), 1), 4),
        "strict_vlm_schema_success": vlm_schema_ok,
        "strict_vlm_schema_rate": round(vlm_schema_ok / max(len(eval_rows), 1), 4),
        "token_cap_hits": sum(row["token_cap_hit"] for row in outputs.values()),
        "cache_reused_rows": reused_rows,
        "initial_batch_size": batch_size,
        "final_effective_batch_size": effective_batch_size,
        "total_seconds": round(elapsed, 1),
        "sec_per_photo": round(elapsed / max(len(eval_rows), 1), 3),
    }
    return [outputs[r["sample_id"]] for r in eval_rows], gen_stats


@torch.inference_mode()
def verify_specialist_candidates_with_vlm(
    model: Any,
    proc: Any,
    candidates: list[dict[str, Any]],
    image_root: Path,
    system_prompt: str,
    batch_size: int = 4,
    max_new_tokens: int = VERIFY_MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    """Run supervised VLM visual verification on high-confidence extra hazard candidates."""
    if not candidates:
        return []
    model.eval()
    proc.tokenizer.padding_side = "left"
    results: list[dict[str, Any]] = []

    for s in range(0, len(candidates), batch_size):
        chunk = candidates[s : s + batch_size]
        texts = []
        pics = []
        for c in chunk:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verify_user_text(c["code"])}]},
            ]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            with Image.open(image_root / c["image"]) as im:
                pics.append(ImageOps.exif_transpose(im).convert("RGB"))
        inputs = proc(text=texts, images=pics, padding=True, return_tensors="pt").to("cuda")
        gen_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        in_len = inputs["input_ids"].shape[-1]
        for local_i, c in enumerate(chunk):
            txt = proc.decode(gen_ids[local_i, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            hazard_item = parse_verified_hazard(txt, c["code"])
            results.append({**c, "accepted": hazard_item is not None, "hazard": hazard_item, "raw_verify_text": txt})
        del inputs, gen_ids
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "train"], required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_dev.json"))
    parser.add_argument("--prompt", type=Path, default=Path("kwater_lab/system_prompt_11code.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/vlm-11code-fold0"))
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.4e-4)
    parser.add_argument("--max-pixels", type=int, default=448 * 32 * 32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-steps", type=int, default=5)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()

    train_multimodal_qlora(
        root=args.root.resolve(),
        manifest_path=args.manifest.resolve(),
        prompt_path=args.prompt.resolve(),
        out_dir=args.out_dir.resolve(),
        val_fold=args.val_fold,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_pixels=args.max_pixels,
        seed=args.seed,
        preflight_only=(args.mode == "preflight"),
        checkpoint_steps=args.checkpoint_steps,
        resume_from=args.resume_from.resolve() if args.resume_from is not None else None,
        stop_after_steps=args.stop_after_steps,
    )


if __name__ == "__main__":
    main()
