"""Build a local blinded review and text/citation proxies after full fold-0 coverage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from .fold0_production_compare import fold_rows, image_id
from .law_reranker_11code import article_to_osh_id, read_selected_cases
from .metrics import decode_target, read_index
from .production_law_replay import read_json, read_rows, refs


RUBRIC = """# 사진별 A/B 정성 검토

같은 사진을 보고 A와 B를 각각 0–2로 기록합니다. 평가하지 않은 항목은 null로 둡니다.

- 근거의 가시성. 0은 사진과 모순되거나 보이지 않는 전제, 1은 장면을 반영하지만 물체·위치·상태 중 일부가 모호함, 2는 위험원과 불안전 상태를 사진에서 짚을 수 있음.
- 조치의 구체성. 0은 누락·위험과 무관하거나 해로움, 1은 관련되지만 대상 또는 실행 방법이 모호함, 2는 사진 속 위험원에 연결된 실행 가능한 조치를 제시함.
- 인용의 지지. 0은 잘못되었거나 불필요한 인용 또는 필요한 근거 누락, 1은 해당 주제이나 구체적인 주장을 일부만 지지함, 2는 적용되는 조문이 주장과 조치를 직접 지지함. 인용 없이 평가가 성립하면 해당 없음으로 둡니다.

무위험 판정에서는 장면 근거의 가시성을 검토합니다. 조치와 인용이 적용되지 않으면 해당 없음으로 둡니다. 인용 지지를 검토할 때는 동봉된 공식 `data/baseline/assets/laws.jsonl`의 active 조문을 확인합니다. 검토를 마친 후 별도 답지에서 A/B를 해제합니다.

자동 집계는 빈 필드, 반복 문장, 활성 조문 존재, 제공 정답의 위험코드·인용 일치만 측정합니다. 실제 사람 평가와 공식 정성 30점은 별도입니다.
"""

FALLBACK_PATTERNS = {
    "scene_evidence": r"(?:사진에서 확인되는 위험이 없습니다\.|사진 판독 결과에서 위험 유무를 확정할 수 없습니다\.|판독 결과에 현장 관찰 설명이 없습니다\.)",
    "evidence": r"(?:사진 근거|현장 내 .+ 관련 불안전 상태가 확인된다\.)",
    "prevention": r"(?:예방수칙|.+ 예방을 위한 방호조치 및 작업 전 안전점검을 실시한다\.)",
}


def load_predictions(path, local_ids):
    def decode(row):
        key, _ = decode_target(row, local_ids=local_ids)
        return key, row

    return read_index(path, decode, limit_size=True)


def proxy(rows, output, truth, active_laws):
    measures = Counter()
    texts = Counter()
    for row in rows:
        sid = row["sample_id"]
        record = output[sid]
        predicted = record["hazards"]
        gold = truth[sid]["target"]["hazards"]
        measures.update(photos=1, correct_code_hazards=len(predicted.keys() & gold.keys()),
                        extra_code_hazards=len(predicted.keys() - gold.keys()), missing_code_hazards=len(gold.keys() - predicted.keys()))
        text_fields = [("scene_evidence", record.get("scene_evidence"))] if not predicted else [
            (key, hazard.get(key)) for hazard in predicted.values() for key in ("evidence", "prevention")]
        for field, value in text_fields:
            measures[f"{field}_fields"] += 1
            if not isinstance(value, str) or not value.strip():
                measures[f"{field}_blank"] += 1
                continue
            if re.fullmatch(FALLBACK_PATTERNS[field], value.strip()):
                measures[f"{field}_fallback_phrase_matches"] += 1
            normalized = re.sub(r"[\W_]+", " ", value.casefold()).strip()
            if normalized:
                texts[field, normalized] += 1
        for code in predicted.keys() | gold.keys():
            cited = set(refs(predicted.get(code, {})))
            gold_refs = set(refs(gold.get(code, {})))
            measures.update(citations=len(cited), active_citations=len(cited & active_laws),
                            unannotated_citations=len(cited - gold_refs), matched_citations=len(cited & gold_refs),
                            missing_gold_citations=len(gold_refs - cited))
            if code in predicted and code in gold and gold_refs:
                measures.update(gold_code_hazards_with_annotated_citations=1, exact_citation_sets=cited == gold_refs)
    for (field, _), count in texts.items():
        measures[f"{field}_repeated_after_first"] += count - 1
    for field in ("evidence", "prevention", "scene_evidence"):
        measures.setdefault(f"{field}_blank", 0)
        measures.setdefault(f"{field}_repeated_after_first", 0)
        measures.setdefault(f"{field}_fallback_phrase_matches", 0)
    return dict(measures)


def prepare(root, manifest, panel, baseline_path, candidate_path, local_output):
    local_output = local_output.resolve()
    if not local_output.is_relative_to((root / "data").resolve()):
        raise ValueError("Blinded outputs and answer key must stay under data/")
    rows = fold_rows(manifest)
    registration = read_json(panel)
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != registration["canonical_manifest_sha256"]:
        raise ValueError("Panel registration differs from the selected fold manifest")
    panel_ids = registration["sample_ids"]
    by_id = {row["sample_id"]: row for row in rows}
    panel_rows = [by_id[sid] for sid in panel_ids]
    if len(panel_rows) != 36 or len({r["group_id"] for r in panel_rows}) != len(panel_rows) or any(r["exposed_prior"] for r in panel_rows):
        raise ValueError("Expected 36 independent previously unexposed panel photos")
    baseline_anon = load_predictions(baseline_path, local_ids=False)
    candidate = load_predictions(candidate_path, local_ids=True)
    expected_anon = {image_id(sid): sid for sid in by_id}
    if baseline_anon.keys() != expected_anon.keys() or candidate.keys() != by_id.keys():
        raise ValueError("Blinded comparison requires both sides to cover all fold-0 photos")
    baseline = {expected_anon[k]: v for k, v in baseline_anon.items()}
    truth = {row["sample_id"]: row for row in read_selected_cases(root / "data/train/train.jsonl", by_id)}
    laws = read_rows(root / "data/baseline/assets/laws.jsonl")
    active_laws = {law_id for r in laws if r.get("status") == "active" and r.get("record_type") == "article"
                   and (law_id := article_to_osh_id(r.get("article")))}

    local_output.mkdir(parents=True, exist_ok=False)
    review_rows, key_rows = [], []
    rng = random.Random(42)
    rng.shuffle(panel_rows)
    for index, row in enumerate(panel_rows, 1):
        sid = row["sample_id"]
        options = [("official_production_lora", baseline[sid]), ("candidate", candidate[sid])]
        rng.shuffle(options)
        case = f"R{index:02d}"
        review = {"case": case, "photo": str((root / 'data/train' / row['image']).resolve())}
        key = {"case": case, "sample_id": sid}
        for letter, (source, record) in zip(("A", "B"), options, strict=True):
            review[letter] = {k: v for k, v in record.items() if k not in ("schema_version", "image_id")}
            review[f"{letter}_review"] = {"visibility": None, "prevention_specificity": None, "citation_support": None, "notes": None}
            key[letter] = source
        review_rows.append(review)
        key_rows.append(key)
    (local_output / "rubric.md").write_text(RUBRIC, encoding="utf-8")
    for filename, data in (("review.jsonl", review_rows), ("answer-key.jsonl", key_rows)):
        (local_output / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    return {
        "human_review_completed": False,
        "official_qualitative_score": None,
        "panel_rule": registration["rule"],
        "automatic_proxies_not_an_official_score": {
            sample: {name: proxy(selected, output, truth, active_laws) for name, output in
                     (("official_production_lora", baseline), ("candidate", candidate))}
            for sample, selected in (("fold0", rows), ("blind_panel", panel_rows))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_dev.json"))
    parser.add_argument("--panel", type=Path, default=Path("data/vlm-11code-fold0/fold0-panel-preregister.json"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare(args.root, args.manifest, args.panel, args.baseline, args.candidate, args.local_output)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
