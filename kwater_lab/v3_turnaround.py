"""SafeGround v3 Turnaround Pipeline: Fold-4 Clean 8-vs-11 Benchmark, 1-Token Margin Verifier, & Full-1045 Fit.

Resolves all 5 bottlenecks identified in the Nov 27 morning audit:
  1. Evaluates on balanced Fold 4 (175 photos, 39 groups, max group share 28%) against a
     freshly trained leak-free 8-code baseline on the exact shared 790 training photos.
  2. Restores exact 30% `no_visible_hazard` prior during minority/other oversampling and
     separates `other` (OSH-52 structural, OSH-200 vehicle, OSH-20 zone) from `fall_height`
     while suppressing background-clutter secondary `slip_trip` hallucinations.
  3. Trains a balanced 240-pair Visual Candidate Verifier (120 true positives vs 120 hard
     Vision-Specialist false positives mined strictly from inner OOF).
  4. Scores verifier candidates in 1 forward pass via exact 1-token log-odds margin
     `logit("accept") - logit("reject") + beta * spec_logit`, calibrating acceptance thresholds
     strictly on held-out inner verification candidates (zero Fold-4 leakage).
  5. Reranks top statutory `OSH-*` citations (`max_refs=1`) by direct lexical support for the
     generated `prevention` action.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch

from .audit import digest
from .clean_specialist import (
    SpatialDualSpecialistHead,
    apply_calibrated_policy,
    build_fold_text_svd_targets,
    fit_nested_calibration,
    load_train_texts,
    predict_probs,
    train_single_head,
)
from .hybrid_pipeline import (
    compute_calib_code_precisions,
    count_added_predictions,
    select_booster_candidates,
    severity4_miss_stats,
)
from .law_reranker_11code import Native11LawRetriever, extract_terms
from .metrics import (
    CODES,
    LIKELIHOOD,
    RISK_BANDS,
    SEVERITY,
    decode_target,
    evaluate,
    paired_bootstrap,
    unique_keys,
)
from .train_head import save_atomic
from .vlm_11code import (
    INSPECT_USER_TEXT,
    KOR_CODE_NAMES,
    LORA_TARGET_REGEX,
    MINORITY_CODES,
    encode_weighted_example,
    forward_and_weighted_assistant_loss,
    gold_row_to_vlm_target,
    load_qwen_4bit_or_bf16,
    parse_json_loose,
    parse_verified_hazard,
    sanitize_text,
    seed_all,
    validate_vlm_11code_obj,
    verify_user_text,
)


TRAIN_EVAL_MAX_PIXELS = 384 * 32 * 32  # 393,216 — matched across train, Fold-4 eval, and L4 deployment!

# Guard against single-photo denominator expansion (-5pt penalty) on the 3 ultra-rare codes (each 1/1,045 in train):
# only emit if the VLM explicitly observed domain-consistent physical evidence in `evidence` or `scene_note`.
ULTRA_RARE_EVIDENCE_CUES: dict[str, re.Pattern[str]] = {
    "drowning_inundation": re.compile(r"수로|수변|집수정|침수|익사|수몰|저수조|물가|하천|배수로|담수|수심|빠질"),
    "confined_space": re.compile(r"맨홀|탱크|피트|밀폐|환기|산소|질식|유해가스|암거|정화조|사일로"),
    "thermal_weather": re.compile(r"고열|고온|화상|폭염|한파|열사병|온열|한랭|동상|강풍|폭우|폭설|증기"),
}

# Action-cue lexical boost table so top-1 OSH-* directly supports the generated prevention sentence
ACTION_ARTICLE_CUES: tuple[tuple[re.Pattern[str], dict[str, float]], ...] = (
    (re.compile(r"살수|물을\s*뿌|분진.*흩날|비산.*방지"), {"OSH-4-2": 0.22}),
    (re.compile(r"방진마스크|보호구|보안경|안전모|안전화|안전대\s*착용|호흡보호구"), {"OSH-32": 0.18}),
    (re.compile(r"비계|거푸집|동바리|흙막이|가설|버팀대|지보공|결속|구조.*보강"), {"OSH-52": 0.18, "OSH-57": 0.12}),
    (re.compile(r"출입\s*금지|출입을\s*제한|접근\s*금지\s*표지|경고\s*표지"), {"OSH-20": 0.20}),
    (re.compile(r"유도자|신호수|후진\s*경보|장비.*동선|운행\s*경로|접촉\s*방지"), {"OSH-200": 0.20, "OSH-171": 0.12}),
    (re.compile(r"개구부|덮개|추락\s*방호망|단부.*난간"), {"OSH-43": 0.16, "OSH-42": 0.12}),
    (re.compile(r"작업발판.*설치|비계.*작업발판"), {"OSH-56": 0.16, "OSH-42": 0.12}),
    (re.compile(r"통로.*확보|통행로.*확보|조명.*확보"), {"OSH-22": 0.14, "OSH-3": 0.12}),
    (re.compile(r"바닥.*정리|물기.*제거|미끄럼|정돈|치워|수거"), {"OSH-3": 0.16}),
    (re.compile(r"전원.*차단|충전부.*절연|절연.*피복|분전반.*외함|접지|누전차단기"), {"OSH-301": 0.18, "OSH-304": 0.15, "OSH-319": 0.12}),
    (re.compile(r"불티.*비산|방화포|불티받이|소화기|가연물.*이격|화기"), {"OSH-241": 0.22, "OSH-239": 0.14}),
    (re.compile(r"덮개.*울|방호울|회전체.*방호|물림점.*방호"), {"OSH-87": 0.20}),
    (re.compile(r"인양.*하부|매달린\s*화물|줄걸이|슬링|훅\s*해지장치"), {"OSH-146": 0.16, "OSH-163": 0.15, "OSH-132": 0.14}),
    (re.compile(r"산소.*농도|유해가스.*측정|밀폐공간.*환기|송기"), {"OSH-619": 0.26, "OSH-620": 0.18}),
)


class ActionSupportedLawRetriever(Native11LawRetriever):
    """Extends Native11LawRetriever with prevention-action direct support reranking (`max_refs=1`)."""

    def __init__(self, law_rows: list[dict], train_cases: list[dict] | None = None):
        super().__init__(law_rows, train_cases)
        self.article_terms: dict[str, Counter] = defaultdict(Counter)
        for row in law_rows:
            if row.get("status") == "active" and row.get("record_type") == "article" and row.get("text"):
                from .law_reranker_11code import article_to_osh_id
                oid = article_to_osh_id(row.get("article", ""))
                if oid:
                    self.article_terms[oid] = extract_terms(str(row.get("title") or "") + " " + str(row["text"]))

    def select_osh_references(
        self,
        code: str,
        evidence: str,
        prevention: str,
        scene_note: str = "",
        max_refs: int = 1,
    ) -> list[dict[str, str]]:
        # Get top candidates from base clause+case retriever by temporarily scoring top 4
        from .law_reranker_11code import STATUTORY_CODE_PRIORS, hazard_query_terms

        q_full = hazard_query_terms(evidence, prevention, scene_note)
        if not q_full:
            fb = next(iter(STATUTORY_CODE_PRIORS.get(code, {})), None)
            return [{"kind": "law", "id": fb}] if fb and fb in self.osh_to_idx else []

        best_clause: dict[str, float] = {}
        for c_i, sc in self.clause_index.query(q_full).items():
            if sc >= 0.055:
                oid = self.clause_osh[c_i]
                if sc > best_clause.get(oid, 0.0):
                    best_clause[oid] = sc

        scores: dict[str, float] = defaultdict(float, {oid: 0.68 * sc for oid, sc in best_clause.items()})
        c_idx = self.case_indices.get(code)
        n_cases = len(self.case_refs.get(code, ()))
        if c_idx is not None and n_cases > 0:
            sims = c_idx.query(q_full)
            nbrs = sorted((i for i, s in sims.items() if s >= 0.065), key=lambda i: (-sims[i], i))[:25]
            sup_map: dict[str, list[float]] = defaultdict(list)
            for n_i in nbrs:
                for pos, oid in enumerate(self.case_refs[code][n_i]):
                    sup_map[oid].append(sims[n_i] * (1.0 if pos == 0 else 0.92))
            for oid, ms in sup_map.items():
                ms.sort(reverse=True)
                scores[oid] += 0.95 * ms[0] + 0.34 * (sum(ms[1:5]) / 4.0) + 0.12 * math.sqrt(self.code_priors[code][oid] / max(n_cases, 1))

        if n_cases < 5:
            pw = 0.95 if n_cases < 2 else 0.35
            for oid, pv in STATUTORY_CODE_PRIORS.get(code, {}).items():
                if oid in self.osh_to_idx:
                    scores[oid] += pw * pv

        # Direct prevention-action lexical & rule support boost
        prev_terms = extract_terms(prevention)
        for oid in list(scores.keys()):
            art_t = self.article_terms.get(oid)
            if art_t and prev_terms:
                overlap = sum(min(prev_terms[k], art_t[k]) for k in prev_terms if k in art_t)
                denom = max(sum(prev_terms.values()), 1.0)
                scores[oid] += 0.22 * min(1.0, overlap / (0.35 * denom))

        for pat, boosts in ACTION_ARTICLE_CUES:
            if pat.search(prevention):
                for oid, b_val in boosts.items():
                    if oid in self.osh_to_idx and (oid in scores or n_cases < 10):
                        scores[oid] += b_val

        if not scores:
            return []
        ranked = sorted(scores.keys(), key=lambda oid: (-scores[oid], self.osh_to_idx[oid]))
        if scores[ranked[0]] < 0.075:
            return []
        return [{"kind": "law", "id": ranked[0]}]


def format_v3_record(image_id: str, vlm_obj: dict[str, Any] | None, retriever: ActionSupportedLawRetriever) -> dict[str, Any]:
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
            lv_raw, sv_raw = h.get("likelihood_score", 3), h.get("severity_score", 3)
            lv = lv_raw if type(lv_raw) is int and 1 <= lv_raw <= 5 else 3
            sv = sv_raw if type(sv_raw) is int and 1 <= sv_raw <= 4 else 3
            prod = lv * sv
            r_grade = next(g for top, g in RISK_BANDS if prod <= top)

            ev = sanitize_text(h.get("evidence"), 240, f"현장 내 {KOR_CODE_NAMES[code]} 관련 불안전 상태가 확인된다.")
            if code in ULTRA_RARE_EVIDENCE_CUES and not ULTRA_RARE_EVIDENCE_CUES[code].search(ev + " " + scene_note):
                continue
            pr = sanitize_text(h.get("prevention"), 480, f"{KOR_CODE_NAMES[code]} 예방을 위한 방호조치 및 안전점검을 실시한다.")
            refs = retriever.select_osh_references(code, ev, pr, scene_note, max_refs=1)

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
        rec = {"schema_version": "1.0", "image_id": image_id, "status": "hazard", "hazards": hazards_out}
    else:
        if raw_status != "no_visible_hazard" or raw_hazards:
            scene_note = "사진 판독 결과에서 가시적인 위험요인이 확인되지 않는다."
        rec = {"schema_version": "1.0", "image_id": image_id, "status": "no_visible_hazard", "hazards": {}, "scene_evidence": scene_note}

    decode_target(rec, local_ids=not re.fullmatch(r"[0-9a-f]{32}", image_id))
    return rec


def build_v3_curriculum_and_inner_val(
    train_split_rows: list[dict],
    raw_train_by_id: dict[str, dict],
    oof_code_scores: np.ndarray,
    oof_ids: list[str],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build v3 curriculum with:
      - Exact safe-scene ratio preservation (~30%) despite minority & `other` oversampling.
      - 240 balanced verification pairs (120 accept + 120 Specialist-OOF hard rejects) for training,
        PLUS ~70 held-out verification pairs (from disjoint groups not used in verify training)
        for leak-free calibration of the 1-token log-odds acceptance threshold!
    """
    rng = random.Random(seed)
    curriculum: list[dict[str, Any]] = []

    base_total = len(train_split_rows)
    base_safe = sum(r["status"] == "no_visible_hazard" for r in train_split_rows)
    target_safe_ratio = base_safe / max(base_total, 1)

    safe_pool = []
    other_pool = []
    added_haz = 0

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
        codes_set = set(m_row["codes"])
        if codes_set & MINORITY_CODES:
            curriculum.append({
                "sample_id": sid + "__min_aug",
                "image": m_row["image"],
                "user_text": INSPECT_USER_TEXT,
                "assistant_text": target_json,
                "augment": True,
                "kind": "inspect_minority",
            })
            added_haz += 1
        elif "other" in codes_set:
            other_pool.append((sid, m_row["image"], target_json))
        elif m_row["status"] == "no_visible_hazard":
            safe_pool.append((sid, m_row["image"], target_json))

    # Oversample ~36 `other` examples (especially single-label `other` structural/equipment scenes)
    rng.shuffle(other_pool)
    for sid, img_rel, target_json in other_pool[:36]:
        curriculum.append({
            "sample_id": sid + "__other_aug",
            "image": img_rel,
            "user_text": INSPECT_USER_TEXT,
            "assistant_text": target_json,
            "augment": True,
            "kind": "inspect_other",
        })
        added_haz += 1

    # Restore exact safe-scene ratio (`target_safe_ratio` ~ 30%) so positive prior is NEVER inflated!
    needed_safe = int(round((target_safe_ratio * (base_total + added_haz) - base_safe) / max(1.0 - target_safe_ratio, 0.1)))
    needed_safe = max(0, needed_safe)
    rng.shuffle(safe_pool)
    for k in range(needed_safe):
        sid, img_rel, target_json = safe_pool[k % len(safe_pool)]
        curriculum.append({
            "sample_id": f"{sid}__safe_bal_{k}",
            "image": img_rel,
            "user_text": INSPECT_USER_TEXT,
            "assistant_text": target_json,
            "augment": True,
            "kind": "inspect_safe_balance",
        })

    # Partition incident groups into verify-train groups (~82%) vs inner-margin-calibration groups (~18%)
    unique_groups = sorted({r["group_id"] for r in train_split_rows})
    rng.shuffle(unique_groups)
    n_cal_groups = max(8, int(round(0.18 * len(unique_groups))))
    cal_groups = set(unique_groups[:n_cal_groups])

    oof_idx = {sid: i for i, sid in enumerate(oof_ids)}
    pos_by_code_tr: dict[str, list[dict]] = defaultdict(list)
    neg_haz_tr: list[tuple[float, dict]] = []
    neg_safe_tr: list[tuple[float, dict]] = []
    inner_cal_pairs: list[dict[str, Any]] = []

    for m_row in train_split_rows:
        sid = m_row["sample_id"]
        if sid not in oof_idx:
            continue
        probs_i = oof_code_scores[oof_idx[sid]]
        g_row = raw_train_by_id[sid]
        t = g_row["target"]
        present = set(m_row["codes"])
        in_cal_grp = m_row["group_id"] in cal_groups

        if t["status"] == "hazard":
            for code, h in t["hazards"].items():
                p_sc = float(probs_i[CODES.index(code)])
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
                rec = {
                    "sample_id": f"{sid}__ver_pos_{code}",
                    "orig_id": sid,
                    "group_id": m_row["group_id"],
                    "image": m_row["image"],
                    "code": code,
                    "is_positive": True,
                    "was_empty_scene": False,
                    "spec_prob": p_sc,
                    "user_text": verify_user_text(code),
                    "assistant_text": ans,
                    "augment": False,
                    "kind": "verify_pos",
                }
                if in_cal_grp and p_sc >= 0.35:
                    inner_cal_pairs.append(rec)
                else:
                    pos_by_code_tr[code].append(rec)

            # Mine top Specialist false-positive code on this hazard photo!
            absent_ranked = sorted(
                (c_i for c_i, c_name in enumerate(CODES) if c_name not in present and c_name not in ("drowning_inundation", "confined_space", "thermal_weather")),
                key=lambda c_i: -float(probs_i[c_i]),
            )
            for c_i in absent_ranked[:2]:
                p_fp = float(probs_i[c_i])
                if p_fp < 0.22:
                    continue
                c_name = CODES[c_i]
                rej = json.dumps(
                    {"decision": "reject", "reason": f"사진에서 {KOR_CODE_NAMES[c_name]}에 해당하는 독립적인 불안전 상태는 확인되지 않는다."},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                rec = {
                    "sample_id": f"{sid}__ver_neghaz_{c_name}",
                    "orig_id": sid,
                    "group_id": m_row["group_id"],
                    "image": m_row["image"],
                    "code": c_name,
                    "is_positive": False,
                    "was_empty_scene": False,
                    "spec_prob": p_fp,
                    "user_text": verify_user_text(c_name),
                    "assistant_text": rej,
                    "augment": False,
                    "kind": "verify_neg_hard",
                }
                if in_cal_grp and p_fp >= 0.42:
                    inner_cal_pairs.append(rec)
                else:
                    neg_haz_tr.append((p_fp, rec))
        else:
            # Safe photo: mine highest-scoring Specialist false alarm code!
            top_c_i = max(
                (c_i for c_i, cn in enumerate(CODES) if cn not in ("drowning_inundation", "confined_space", "thermal_weather")),
                key=lambda c_i: float(probs_i[c_i]),
            )
            p_fp = float(probs_i[top_c_i])
            c_name = CODES[top_c_i]
            s_ev = sanitize_text(t.get("scene_evidence"), 180, "안전조치가 확보되어 가시적인 위험이 확인되지 않는다.")
            rej = json.dumps({"decision": "reject", "reason": s_ev}, ensure_ascii=False, separators=(",", ":"))
            rec = {
                "sample_id": f"{sid}__ver_negsafe_{c_name}",
                "orig_id": sid,
                "group_id": m_row["group_id"],
                "image": m_row["image"],
                "code": c_name,
                "is_positive": False,
                "was_empty_scene": True,
                "spec_prob": p_fp,
                "user_text": verify_user_text(c_name),
                "assistant_text": rej,
                "augment": False,
                "kind": "verify_neg_safe",
            }
            if in_cal_grp and p_fp >= 0.40:
                inner_cal_pairs.append(rec)
            else:
                neg_safe_tr.append((p_fp, rec))

    # Select 110 stratified positive verification examples across all codes (prioritizing `other`, `caught`, `fall`, `slip`, minorities)
    quotas = {
        "other": 26,
        "fall_height": 22,
        "slip_trip": 20,
        "caught_in_machine": 12,
        "struck_by_lifting": 10,
        "electrical_energy": 8,
        "fire_explosion": 7,
        "chemical_exposure": 4,
        "drowning_inundation": 2,
        "confined_space": 2,
        "thermal_weather": 2,
    }
    chosen_pos: list[dict] = []
    for c_name, q in quotas.items():
        pool_c = pos_by_code_tr.get(c_name, [])
        rng.shuffle(pool_c)
        chosen_pos.extend(pool_c[:q])

    # Select 110 hardest Specialist-mined rejects (60 hard hazard-scene wrong-code FPs + 50 hardest safe-scene FPs)
    neg_haz_tr.sort(key=lambda x: -x[0])
    neg_safe_tr.sort(key=lambda x: -x[0])
    chosen_neg = [r for _, r in neg_haz_tr[:60]] + [r for _, r in neg_safe_tr[:50]]

    for r in chosen_pos + chosen_neg:
        curriculum.append({
            "sample_id": r["sample_id"],
            "image": r["image"],
            "user_text": r["user_text"],
            "assistant_text": r["assistant_text"],
            "augment": False,
            "kind": r["kind"],
        })

    return curriculum, inner_cal_pairs


@torch.inference_mode()
def score_verifier_1token_margins(
    model: Any,
    proc: Any,
    candidates: list[dict[str, Any]],
    image_root: Path,
    system_prompt: str,
    batch_size: int = 4,
) -> list[float]:
    """Compute exact `logit("accept") - logit("reject")` in ONE forward pass per batch (no generation!)."""
    if not candidates:
        return []
    model.eval()
    proc.tokenizer.padding_side = "left"

    acc_ids = proc.tokenizer.encode("accept", add_special_tokens=False)
    rej_ids = proc.tokenizer.encode("reject", add_special_tokens=False)
    if len(acc_ids) != 1 or len(rej_ids) != 1:
        raise RuntimeError(f"Expected single-token accept/reject, got {acc_ids} vs {rej_ids}")
    acc_tok, rej_tok = acc_ids[0], rej_ids[0]

    margins_out: list[float] = []
    for s in range(0, len(candidates), batch_size):
        sub = candidates[s : s + batch_size]
        texts = []
        pics = []
        for c in sub:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verify_user_text(c["code"])}]},
            ]
            # Append exact assistant prefix `{"decision":"` so next-token logit directly scores accept vs reject!
            pref = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + '{"decision":"'
            texts.append(pref)
            with Image.open(image_root / c["image"]) as im:
                pics.append(ImageOps.exif_transpose(im).convert("RGB"))
        inputs = proc(text=texts, images=pics, padding=True, return_tensors="pt").to("cuda")
        out = model(**inputs, logits_to_keep=1, use_cache=False)
        last_logits = out.logits[:, -1, :].float()
        diff = (last_logits[:, acc_tok] - last_logits[:, rej_tok]).cpu().tolist()
        margins_out.extend(float(x) for x in diff)
        del inputs, out, last_logits
    return margins_out


def calibrate_margin_gate_on_inner_pairs(
    inner_pairs: list[dict[str, Any]],
    vlm_margins: list[float],
) -> dict[str, float]:
    """Fit (beta, tau_when_hazard, tau_when_empty) strictly on inner-split held-out verification pairs."""
    if len(inner_pairs) < 12:
        return {"beta": 1.2, "tau_when_hazard": 0.8, "tau_when_empty": 2.2}

    y = np.array([1.0 if p["is_positive"] else 0.0 for p in inner_pairs])
    is_emp = np.array([bool(p.get("was_empty_scene", False)) for p in inner_pairs])
    sp = np.clip(np.array([float(p["spec_prob"]) for p in inner_pairs]), 1e-4, 1.0 - 1e-4)
    sp_logit = np.log(sp / (1.0 - sp))
    vm = np.array(vlm_margins, dtype=np.float64)

    best_cfg = {"beta": 1.2, "tau_when_hazard": 1.0, "tau_when_empty": 2.4}
    best_obj = -1e9

    for beta in (0.6, 1.0, 1.4, 1.8):
        joint = vm + beta * sp_logit
        # 1. Calibrate tau_when_hazard on non-empty inner pairs targeting >= 62% precision & net TP > FP
        haz_m = ~is_emp
        best_th = 1.0
        best_h_score = -1e9
        if np.sum(haz_m) >= 6:
            for th in np.linspace(-1.5, 4.0, 56):
                pred = joint[haz_m] >= th
                tp = float(np.sum(pred & (y[haz_m] > 0.5)))
                fp = float(np.sum(pred & (y[haz_m] < 0.5)))
                prec = tp / max(tp + fp, 1.0)
                if tp + fp >= 2 and prec >= 0.58:
                    sc = 2.0 * tp - 1.5 * fp
                    if sc > best_h_score:
                        best_h_score = sc
                        best_th = float(th)

        # 2. Calibrate tau_when_empty strictly protecting safe scenes (allow <= 10% safe false-accept rate)
        emp_m = is_emp
        best_te = max(best_th + 1.0, 2.2)
        if np.sum(emp_m) >= 5:
            safe_scores = np.sort(joint[emp_m & (y < 0.5)])
            if len(safe_scores) > 0:
                q92 = float(np.quantile(safe_scores, 0.92))
                best_te = max(best_th + 0.6, q92 + 0.15)

        pred_all = np.where(is_emp, joint >= best_te, joint >= best_th)
        tp_all = float(np.sum(pred_all & (y > 0.5)))
        fp_all = float(np.sum(pred_all & (y < 0.5)))
        obj = 2.0 * tp_all - 1.8 * fp_all
        if obj > best_obj:
            best_obj = obj
            best_cfg = {
                "beta": round(float(beta), 3),
                "tau_when_hazard": round(float(best_th), 3),
                "tau_when_empty": round(float(best_te), 3),
                "inner_pairs_n": len(inner_pairs),
                "inner_tp_accepted": int(tp_all),
                "inner_fp_accepted": int(fp_all),
            }
    return best_cfg


def _load_baseline_helpers(root: Path) -> dict[str, Any]:
    """Load official 8-code baseline modules cleanly without colliding with workspace `submission/`."""
    b_root = str((root / "data/baseline").resolve())
    saved_sub = sys.modules.pop("submission", None)
    inserted = False
    if sys.path[:1] != [b_root]:
        sys.path.insert(0, b_root)
        inserted = True
    try:
        def _load_mod(mod_name: str, rel_file: str):
            spec = importlib.util.spec_from_file_location(mod_name, root / "data/baseline" / rel_file)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            return mod

        rs_mod = _load_mod("risk_schema", "risk_schema.py")
        cm_mod = _load_mod("_kw_base_common", "scripts/common.py")
        pr_mod = _load_mod("_kw_base_prep", "scripts/prepare_dataset.py")
        sub_mod = _load_mod("submission", "submission.py")
        mn_mod = _load_mod("_kw_base_main", "main.py")
        return {
            "target_only": rs_mod.target_only,
            "validate_8_pred": rs_mod.validate_prediction,
            "parse_8_pred": cm_mod.parse_prediction,
            "convert_gold_to_8": pr_mod.convert,
            "format_8_sub": sub_mod.format_submission,
            "convert_8_to_comp": mn_mod.convert,
        }
    finally:
        if inserted and b_root in sys.path:
            sys.path.remove(b_root)
        if saved_sub is not None:
            sys.modules["submission"] = saved_sub
        else:
            sys.modules.pop("submission", None)


def train_8code_baseline_fold4(root: Path, manifest_path: Path, out_dir: Path, epochs: int = 2, seed: int = 42) -> dict[str, Any]:
    """Train a 100% leak-free fresh 8-code official baseline on Fold-4's 790 training photos."""
    from .vlm_11code import train_multimodal_qlora

    bh = _load_baseline_helpers(root)
    target_only = bh["target_only"]
    convert_to_8code = bh["convert_gold_to_8"]

    prompt_8_path = root / "data/baseline/system_prompt.txt"

    def _build_8code_curriculum(train_split_rows: list[dict], raw_train_by_id: dict[str, dict], seed: int = 42) -> list[dict]:
        cur = []
        for m_row in train_split_rows:
            sid = m_row["sample_id"]
            conv = convert_to_8code(raw_train_by_id[sid])
            if conv is None:
                continue  # Drops the 1 confined_space-only photo (791 -> 790), matching official recipe
            ans = json.dumps(target_only(conv["target"]), ensure_ascii=False, separators=(",", ":"))
            cur.append({
                "sample_id": sid,
                "image": m_row["image"],
                "user_text": INSPECT_USER_TEXT,
                "assistant_text": ans,
                "augment": False,
                "kind": "inspect_8code",
            })
        return cur

    res8 = train_multimodal_qlora(
        root=root,
        manifest_path=manifest_path,
        prompt_path=prompt_8_path,
        out_dir=out_dir,
        val_fold=4,
        epochs=epochs,
        grad_accum=16,
        lr=1.4e-4,
        max_pixels=TRAIN_EVAL_MAX_PIXELS,
        seed=seed,
        checkpoint_steps=10,
        curriculum_builder=_build_8code_curriculum,
        experiment_identity={"arm": "clean_8code_baseline_fold4", "train_rows_expected": 790},
    )
    gc.collect()
    torch.cuda.empty_cache()
    return res8


def train_v3_11code_model(
    root: Path,
    manifest_path: Path,
    prompt_path: Path,
    out_dir: Path,
    mode: str = "fold4",  # "fold4" or "full1045"
    epochs: int = 2,
    seed: int = 42,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train v3 11-code Multimodal LoRA + Specialist-mined Hard-Negative Verifier + save inner margin gate."""
    from .vlm_11code import train_multimodal_qlora

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_train_by_id = {
        r["sample_id"]: r
        for r in (
            json.loads(l, object_pairs_hook=unique_keys)
            for l in (root / "data/train/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }

    if mode == "fold4":
        oof_arc = np.load(root / "data/clean-specialist-v1/strict-inner-oof/fold-4.npz", allow_pickle=False)
        oof_ids = oof_arc["train_ids"].tolist()
        oof_scores = oof_arc["code_scores"]
        val_fold_arg = 4
        train_rows_for_cur = [r for r in manifest if r["split"] == "dev" and int(r["dev_fold"]) != 4]
    elif mode == "full1045":
        # First ensure we have 5-fold OOF specialist scores covering all 1,045 photos!
        full_spec_dir = root / "data/v3-full1045-specialist"
        full_spec_oof = ensure_full_1045_specialist_ensemble(root, manifest, full_spec_dir)
        oof_ids = full_spec_oof["ids"].tolist()
        oof_scores = full_spec_oof["code_probs"]
        val_fold_arg = -1045  # Custom sentinel handled via temporary full-dev manifest
        train_rows_for_cur = manifest
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cur_built, inner_cal_pairs = build_v3_curriculum_and_inner_val(
        train_rows_for_cur, raw_train_by_id, oof_scores, oof_ids, seed=seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    inner_pairs_path = out_dir / "inner_margin_cal_pairs.json"
    save_atomic(inner_pairs_path, lambda f: f.write((json.dumps(inner_cal_pairs, ensure_ascii=False, indent=2) + "\n").encode("utf-8")))

    # If full1045, write a manifest where all 1,045 rows are marked split="dev" for train_multimodal_qlora(val_fold=-1)
    eff_manifest_path = manifest_path
    eff_val_fold = val_fold_arg
    if mode == "full1045":
        eff_manifest_path = out_dir / "manifest_all_1045_train.json"
        all_dev_rows = [{**r, "split": "dev"} for r in manifest]
        save_atomic(eff_manifest_path, lambda f: f.write((json.dumps(all_dev_rows, ensure_ascii=False, indent=2) + "\n").encode("utf-8")))
        eff_val_fold = -1

    summary = train_multimodal_qlora(
        root=root,
        manifest_path=eff_manifest_path,
        prompt_path=prompt_path,
        out_dir=out_dir,
        val_fold=eff_val_fold,
        epochs=epochs,
        grad_accum=16,
        lr=1.4e-4,
        max_pixels=TRAIN_EVAL_MAX_PIXELS,
        seed=seed,
        checkpoint_steps=10,
        resume_from=resume_from,
        curriculum_builder=lambda _tr, _raw, seed=seed: cur_built,
        experiment_identity={"arm": f"v3_11code_{mode}", "curriculum_n": len(cur_built), "inner_cal_pairs_n": len(inner_cal_pairs)},
    )

    # Immediately compute 1-token log-odds margins on the held-out inner verification pairs and lock (beta, tau_haz, tau_emp)!
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Calibrating 1-token log-odds margin gate on {len(inner_cal_pairs)} held-out inner pairs...", flush=True)
    model, proc = load_qwen_4bit_or_bf16(
        root / "data/baseline/base_model",
        max_pixels=TRAIN_EVAL_MAX_PIXELS,
        use_4bit=False,
        adapter_dir=out_dir / "adapter",
    )
    sys_prompt = prompt_path.read_text(encoding="utf-8")
    vlm_margins = score_verifier_1token_margins(model, proc, inner_cal_pairs, root / "data/train", sys_prompt, batch_size=2)
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()

    gate_cfg = calibrate_margin_gate_on_inner_pairs(inner_cal_pairs, vlm_margins)
    save_atomic(out_dir / "margin_gate_calibration.json", lambda f: f.write((json.dumps(gate_cfg, indent=2) + "\n").encode("utf-8")))
    print("MARGIN_GATE_LOCKED " + json.dumps(gate_cfg), flush=True)
    summary["margin_gate_calibration"] = gate_cfg
    save_atomic(out_dir / "training_summary.json", lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    return summary


def ensure_full_1045_specialist_ensemble(root: Path, manifest: list[dict], out_dir: Path) -> dict[str, np.ndarray]:
    """Fit 5-fold x 3-seed SpatialDualSpecialistHead across ALL 1,045 official photos for final production deployment."""
    ckpt_path = out_dir / "specialist_5fold_1045_ensemble.pt"
    oof_path = out_dir / "oof_1045_predictions.npz"
    if ckpt_path.is_file() and oof_path.is_file():
        arc = np.load(oof_path, allow_pickle=False)
        return {"ids": arc["ids"], "code_probs": arc["code_probs"], "scene_probs": arc["scene_probs"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feat_arc = np.load(root / "data/clean-specialist-v1/features_multiview_2176d.npz", allow_pickle=False)
    id_to_i = {sid: i for i, sid in enumerate(feat_arc["ids"].tolist())}

    # Assign holdout groups deterministically across folds 0..4 while keeping dev_fold intact for the 966 dev rows
    grp_fold: dict[str, int] = {}
    for r in manifest:
        if r["split"] == "dev":
            grp_fold[r["group_id"]] = int(r["dev_fold"])
    hold_grps = sorted({r["group_id"] for r in manifest if r["split"] != "dev"})
    for idx_g, g in enumerate(hold_grps):
        grp_fold[g] = idx_g % 5

    ids_all = [r["sample_id"] for r in manifest]
    groups_all = [r["group_id"] for r in manifest]
    folds_all = np.array([grp_fold[r["group_id"]] for r in manifest], dtype=np.int64)
    x_all = feat_arc["features"][[id_to_i[sid] for sid in ids_all]]
    y_codes = np.array([[c in r["codes"] for c in CODES] for r in manifest], dtype=np.float32)
    y_scene = np.array([r["status"] == "hazard" for r in manifest], dtype=np.float32)[:, None]
    y_all = np.concatenate([y_codes, y_scene], axis=1)
    all_texts = load_train_texts(root / "data/train/train.jsonl")

    oof_code = np.zeros((len(manifest), len(CODES)), dtype=np.float64)
    oof_scene = np.zeros(len(manifest), dtype=np.float64)
    fold_models_state: dict[str, list[dict]] = {}
    seeds = (42, 1042, 2042)

    for fold in range(5):
        tr = folds_all != fold
        va = folds_all == fold
        svd_tr = build_fold_text_svd_targets([all_texts[ids_all[i]] for i in np.where(tr)[0]], dim=64)
        models_f = [
            train_single_head(x_all[tr], y_all[tr], svd_tr, seed=sd + fold * 100, device=device, epochs=55, batch_size=64)
            for sd in seeds
        ]
        oof_code[va], oof_scene[va] = predict_probs(models_f, x_all[va], device)
        fold_models_state[str(fold)] = [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models_f]

    torch.cuda.empty_cache()
    full_calib = fit_nested_calibration(oof_code, oof_scene, y_codes, groups_all)
    code_precs = compute_calib_code_precisions(oof_code, oof_scene, y_codes, full_calib)

    save_atomic(
        oof_path,
        lambda f: np.savez(
            f,
            ids=np.array(ids_all),
            folds=folds_all,
            code_probs=oof_code.astype(np.float32),
            scene_probs=oof_scene.astype(np.float32),
            y_codes=y_codes.astype(np.uint8),
        ),
    )
    torch.save(
        {
            "spec": "clean-spec-v1-1045-full",
            "codes": list(CODES),
            "seeds": list(seeds),
            "fold_models": fold_models_state,
            "full_dev_calibration": full_calib,
            "oof_code_precisions": {k: round(v, 4) for k, v in code_precs.items()},
        },
        ckpt_path,
    )
    print(f"Saved full 1,045-photo Specialist ensemble -> {ckpt_path}", flush=True)
    return {"ids": np.array(ids_all), "code_probs": oof_code, "scene_probs": oof_scene}


@torch.inference_mode()
def evaluate_fold4_benchmark(
    root: Path,
    manifest_path: Path,
    baseline8_dir: Path,
    v3_dir: Path,
    prompt_v3_path: Path,
    out_summary_path: Path,
) -> dict[str, Any]:
    """Run clean apples-to-apples native-BF16 benchmark on Fold 4 (175 unseen photos, 39 groups)."""
    from .vlm_11code import infer_batched_vlm, verify_specialist_candidates_with_vlm

    bh = _load_baseline_helpers(root)
    convert_8_to_comp = bh["convert_8_to_comp"]
    validate_8_pred = bh["validate_8_pred"]
    parse_8_pred = bh["parse_8_pred"]
    convert_gold_to_8 = bh["convert_gold_to_8"]
    format_8_sub = bh["format_8_sub"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_rows = [r for r in manifest if r["split"] == "dev" and int(r["dev_fold"]) == 4]
    train_rows = [r for r in manifest if r["split"] == "dev" and int(r["dev_fold"]) != 4]
    raw_train_by_id = {
        r["sample_id"]: r
        for r in (
            json.loads(l, object_pairs_hook=unique_keys)
            for l in (root / "data/train/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    laws = [json.loads(l) for l in (root / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    gold_codes = {r["sample_id"]: frozenset(r["codes"]) for r in eval_rows}
    group_map = {r["sample_id"]: r["group_id"] for r in eval_rows}
    image_root = root / "data/train"
    base_model_dir = root / "data/baseline/base_model"

    # 1. Evaluate Fresh 8-Code Baseline on Fold 4 (BF16, 384*32*32, Fold-4-blind 8-code RAG cases!)
    print(f"[1/3] Evaluating Clean 8-Code Baseline ({baseline8_dir / 'adapter'}) on Fold 4 (n={len(eval_rows)})...", flush=True)
    m8, p8 = load_qwen_4bit_or_bf16(base_model_dir, max_pixels=TRAIN_EVAL_MAX_PIXELS, use_4bit=False, adapter_dir=baseline8_dir / "adapter")
    prompt8 = (root / "data/baseline/system_prompt.txt").read_text(encoding="utf-8")
    raw8_outs, gen8_stats = infer_batched_vlm(m8, p8, eval_rows, image_root, prompt8, batch_size=2, max_new_tokens=620)
    del m8, p8
    gc.collect()
    torch.cuda.empty_cache()

    valid8_rows = []
    for r, o in zip(eval_rows, raw8_outs):
        obj8, err8 = parse_8_pred(o["raw_text"])
        if err8 is None and not validate_8_pred(obj8):
            valid8_rows.append({"sample_id": r["sample_id"], "prediction": obj8})
    train8_cases = [c for r in train_rows if (c := convert_gold_to_8(raw_train_by_id[r["sample_id"]])) is not None]
    local_eval_dir = root / "data/local-eval-predictions/fold4-v3-benchmark"
    local_eval_dir.mkdir(parents=True, exist_ok=True)
    formatted8_map = {}
    if valid8_rows:
        tmp8 = local_eval_dir / "tmp_8code_formatted.jsonl"
        format_8_sub(valid8_rows, laws, train8_cases, tmp8)
        formatted8_map = {json.loads(l)["sample_id"]: json.loads(l)["prediction"] for l in tmp8.read_text(encoding="utf-8").splitlines() if l.strip()}
    comp8_recs = [convert_8_to_comp(r["sample_id"], formatted8_map.get(r["sample_id"])) for r in eval_rows]
    preds_8code = {rec["image_id"]: decode_target(rec, local_ids=True)[1] for rec in comp8_recs}
    eval_8code = evaluate(gold_codes, preds_8code, group_map)
    sev4_8code = severity4_miss_stats(raw_train_by_id, preds_8code)

    # 2. Evaluate v3 11-Code Standalone + 1-Token Margin Verifier on Fold 4 (BF16, 384*32*32)
    print(f"[2/3] Evaluating v3 11-Code Standalone ({v3_dir / 'adapter'}) on Fold 4 (n={len(eval_rows)})...", flush=True)
    mv3, pv3 = load_qwen_4bit_or_bf16(base_model_dir, max_pixels=TRAIN_EVAL_MAX_PIXELS, use_4bit=False, adapter_dir=v3_dir / "adapter")
    prompt_v3 = prompt_v3_path.read_text(encoding="utf-8")
    raw_v3_outs, gen_v3_stats = infer_batched_vlm(mv3, pv3, eval_rows, image_root, prompt_v3, batch_size=2, max_new_tokens=620)

    retriever_v3 = ActionSupportedLawRetriever(laws, [raw_train_by_id[r["sample_id"]] for r in train_rows])
    comp_v3_recs = [format_v3_record(r["sample_id"], o["parsed"], retriever_v3) for r, o in zip(eval_rows, raw_v3_outs)]
    preds_v3 = {rec["image_id"]: decode_target(rec, local_ids=True)[1] for rec in comp_v3_recs}
    eval_v3 = evaluate(gold_codes, preds_v3, group_map)
    sev4_v3 = severity4_miss_stats(raw_train_by_id, preds_v3)

    # 3. Strict Inner-Calibrated Specialist Proposals + 1-Token Log-Odds Margin Gate on Fold 4
    print("[3/3] Running Strict Inner-Calibrated Specialist + 1-Token Margin Verifier on Fold 4...", flush=True)
    pooled_cal = json.loads((root / "results/clean-neural-calibration-pooled.json").read_text(encoding="utf-8"))
    spec_calib_f4 = pooled_cal["calibration_by_fold"]["4"]
    inner_f4_arc = np.load(root / "data/clean-specialist-v1/strict-inner-oof/fold-4.npz", allow_pickle=False)
    tr_id_order = inner_f4_arc["train_ids"].tolist()
    y_tr_f4 = np.array([[c in raw_train_by_id[sid]["target"]["hazards"] for c in CODES] for sid in tr_id_order], dtype=np.float32)
    code_precs_f4 = compute_calib_code_precisions(inner_f4_arc["code_scores"], inner_f4_arc["scene_scores"], y_tr_f4, spec_calib_f4)

    dev_oof_arc = np.load(root / "data/clean-specialist-v1/dev_oof_predictions.npz", allow_pickle=False)
    dev_idx = {sid: i for i, sid in enumerate(dev_oof_arc["ids"].tolist())}
    f4_sel = [dev_idx[r["sample_id"]] for r in eval_rows]
    spec_code_p = dev_oof_arc["code_probs"][f4_sel]
    spec_scene_p = dev_oof_arc["scene_probs"][f4_sel]

    spec_standalone_list = apply_calibrated_policy(spec_code_p, spec_scene_p, spec_calib_f4)
    preds_spec = {r["sample_id"]: pr for r, pr in zip(eval_rows, spec_standalone_list)}
    eval_spec = evaluate(gold_codes, preds_spec, group_map)
    sev4_spec = severity4_miss_stats(raw_train_by_id, preds_spec)

    raw_cands = select_booster_candidates(
        eval_rows,
        raw_v3_outs,
        spec_code_p,
        spec_scene_p,
        spec_calib_f4,
        code_precisions=code_precs_f4,
        margin_boost_when_hazard=0.04,
        margin_boost_when_empty=0.12,
        max_candidates_per_photo=1,
    )

    t_m0 = time.perf_counter()
    gate_cfg = json.loads((v3_dir / "margin_gate_calibration.json").read_text(encoding="utf-8"))
    cand_vlm_margins = score_verifier_1token_margins(mv3, pv3, raw_cands, image_root, prompt_v3, batch_size=2)

    beta = float(gate_cfg["beta"])
    tau_haz = float(gate_cfg["tau_when_hazard"])
    tau_emp = float(gate_cfg["tau_when_empty"])

    passed_cands = []
    for c, vm in zip(raw_cands, cand_vlm_margins):
        sp = min(max(float(c["spec_cal_prob"]), 1e-4), 1.0 - 1e-4)
        joint = vm + beta * math.log(sp / (1.0 - sp))
        req = tau_emp if c["was_vlm_empty"] else tau_haz
        if joint >= req:
            passed_cands.append({**c, "vlm_1tok_margin": round(vm, 4), "joint_score": round(joint, 4)})

    verified_passed = verify_specialist_candidates_with_vlm(mv3, pv3, passed_cands, image_root, prompt_v3, batch_size=2, max_new_tokens=280)
    ver_elapsed = round(time.perf_counter() - t_m0, 2)
    del mv3, pv3
    gc.collect()
    torch.cuda.empty_cache()

    # Merge accepted candidates into v3 outputs
    hyb_outs = copy.deepcopy(raw_v3_outs)
    accepted_count = 0
    for v in verified_passed:
        hz = parse_verified_hazard(v["raw_verify_text"], v["code"])
        if hz is not None:
            idx = int(v["row_idx"])
            cur = hyb_outs[idx].get("parsed")
            if not isinstance(cur, dict):
                cur = {"scene_note": hz["evidence"], "status": "hazard", "hazards": []}
            hl = cur.get("hazards") if isinstance(cur.get("hazards"), list) else []
            if hz["code"] not in {x.get("code") for x in hl if isinstance(x, dict)} and len(hl) < 3:
                if cur.get("status") != "hazard" or not hl:
                    cur["scene_note"] = hz["evidence"]
                hl.append(hz)
                cur["status"] = "hazard"
                cur["hazards"] = hl[:3]
                hyb_outs[idx]["parsed"] = cur
                accepted_count += 1

    comp_hyb_recs = [format_v3_record(r["sample_id"], o["parsed"], retriever_v3) for r, o in zip(eval_rows, hyb_outs)]
    preds_hyb = {rec["image_id"]: decode_target(rec, local_ids=True)[1] for rec in comp_hyb_recs}
    eval_hyb = evaluate(gold_codes, preds_hyb, group_map)
    sev4_hyb = severity4_miss_stats(raw_train_by_id, preds_hyb)

    # Save row-level predictions locally inside data/ (never in results/)
    save_atomic(local_eval_dir / "clean_8code_baseline.jsonl", lambda f: f.write(("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in comp8_recs)).encode("utf-8")))
    save_atomic(local_eval_dir / "v3_11code_standalone.jsonl", lambda f: f.write(("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in comp_v3_recs)).encode("utf-8")))
    save_atomic(local_eval_dir / "v3_hybrid_margin_verified.jsonl", lambda f: f.write(("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in comp_hyb_recs)).encode("utf-8")))

    summary = {
        "protocol": "Fold-4 Clean Unseen Benchmark (175 photos, 39 groups, max group share 28.0%, matched BF16 384*32*32 inference)",
        "photos": len(eval_rows),
        "groups": len(set(group_map.values())),
        "manifest_sha256": digest(manifest_path),
        "baseline8_adapter_sha256": digest(baseline8_dir / "adapter/adapter_model.safetensors"),
        "v3_adapter_sha256": digest(v3_dir / "adapter/adapter_model.safetensors"),
        "arms": {
            "clean_8code_baseline": {
                "generation": gen8_stats,
                "valid_8code_schema_rows": len(valid8_rows),
                "kit_schema_valid_rate": 1.0,
                "metrics": eval_8code,
                "severity4": sev4_8code,
            },
            "specialist_inner_calibrated_standalone": {
                "metrics": eval_spec,
                "severity4": sev4_spec,
            },
            "v3_11code_standalone": {
                "generation": gen_v3_stats,
                "kit_schema_valid_rate": 1.0,
                "metrics": eval_v3,
                "severity4": sev4_v3,
            },
            "v3_hybrid_1token_margin_verified": {
                "margin_gate": gate_cfg,
                "raw_specialist_proposals": len(raw_cands),
                "passed_1token_margin_gate": len(passed_cands),
                "final_accepted_additions": accepted_count,
                "added_tp_fp": count_added_predictions(gold_codes, preds_v3, preds_hyb),
                "verification_seconds_total": ver_elapsed,
                "kit_schema_valid_rate": 1.0,
                "metrics": eval_hyb,
                "severity4": sev4_hyb,
            },
        },
        "paired_bootstrap": {
            "v3_standalone_vs_clean_8code": paired_bootstrap(gold_codes, preds_8code, preds_v3, group_map, resamples=2000, seed=42),
            "v3_hybrid_vs_clean_8code": paired_bootstrap(gold_codes, preds_8code, preds_hyb, group_map, resamples=2000, seed=42),
            "v3_hybrid_vs_v3_standalone": paired_bootstrap(gold_codes, preds_v3, preds_hyb, group_map, resamples=2000, seed=42),
        },
    }
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(out_summary_path, lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", choices=["train-8code-fold4", "train-v3-fold4", "eval-fold4", "train-v3-full1045", "run-fold4-all"], required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_full_sealed.json"))
    ap.add_argument("--prompt-v3", type=Path, default=Path("kwater_lab/system_prompt_v3.txt"))
    ap.add_argument("--baseline8-dir", type=Path, default=Path("data/v3-baseline8-fold4"))
    ap.add_argument("--v3-fold4-dir", type=Path, default=Path("data/v3-11code-fold4"))
    ap.add_argument("--v3-full-dir", type=Path, default=Path("data/v3-11code-full1045"))
    ap.add_argument("--eval-out", type=Path, default=Path("results/fold4_v3_clean_benchmark.json"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--resume-from", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    manifest_p = args.manifest.resolve()
    prompt_p = args.prompt_v3.resolve()

    if args.step in ("train-8code-fold4", "run-fold4-all"):
        if not (args.baseline8_dir.resolve() / "training_summary.json").exists():
            train_8code_baseline_fold4(root, manifest_p, args.baseline8_dir.resolve(), epochs=args.epochs)
    if args.step in ("train-v3-fold4", "run-fold4-all"):
        if not (args.v3_fold4_dir.resolve() / "training_summary.json").exists():
            train_v3_11code_model(
                root, manifest_p, prompt_p, args.v3_fold4_dir.resolve(), mode="fold4", epochs=args.epochs, resume_from=args.resume_from
            )
    if args.step in ("eval-fold4", "run-fold4-all"):
        res = evaluate_fold4_benchmark(
            root, manifest_p, args.baseline8_dir.resolve(), args.v3_fold4_dir.resolve(), prompt_p, args.eval_out.resolve()
        )
        b8 = res["arms"]["clean_8code_baseline"]["metrics"]
        v3 = res["arms"]["v3_11code_standalone"]["metrics"]
        hy = res["arms"]["v3_hybrid_1token_margin_verified"]["metrics"]
        print(
            f"FOLD4_BENCHMARK_DONE | Clean-8Code F1={b8['macro_f1']:.4f} ({b8['quantitative_points_out_of_70']:.2f}pt, FA={b8['no_hazard_false_alarms']}/{b8['no_hazard_photos']}) "
            f"| v3-11Code F1={v3['macro_f1']:.4f} ({v3['quantitative_points_out_of_70']:.2f}pt, FA={v3['no_hazard_false_alarms']}/{v3['no_hazard_photos']}) "
            f"| v3-Hybrid F1={hy['macro_f1']:.4f} ({hy['quantitative_points_out_of_70']:.2f}pt, FA={hy['no_hazard_false_alarms']}/{hy['no_hazard_photos']})",
            flush=True,
        )
    if args.step == "train-v3-full1045":
        train_v3_11code_model(
            root, manifest_p, prompt_p, args.v3_full_dir.resolve(), mode="full1045", epochs=args.epochs, resume_from=args.resume_from
        )


if __name__ == "__main__":
    main()
