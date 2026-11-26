"""Unit tests for the SafeGround v5 11-code turnaround pipeline."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
import numpy as np

from kwater_lab.hybrid_pipeline import merge_verified_candidates, select_booster_candidates
from kwater_lab.law_reranker_11code import Native11LawRetriever, article_to_osh_id
from kwater_lab.metrics import CODES
from kwater_lab.vlm_11code import (
    DECISION_SPAN_RE,
    format_competition_record,
    gold_row_to_vlm_target,
    parse_json_loose,
    validate_vlm_11code_obj,
)


ROOT = Path(__file__).resolve().parents[1]


class TurnaroundPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema_path = ROOT / "data/local-kit/kit/prediction.schema.json"
        cls.validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
        laws_path = ROOT / "data/baseline/assets/laws.jsonl"
        cls.laws = [json.loads(l) for l in laws_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cls.retriever = Native11LawRetriever(cls.laws, [])

    def test_clean_split_invariants(self) -> None:
        manifest_path = ROOT / "data/clean-split-v1/split_manifest_full_sealed.json"
        excl_path = ROOT / "data/holdout-exclusions.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        excl = set(json.loads(excl_path.read_text(encoding="utf-8"))["exposed_ids"])

        self.assertEqual(len(manifest), 1045)
        self.assertEqual(len({r["sample_id"] for r in manifest}), 1045)

        group_assignments = defaultdict(set)
        for row in manifest:
            group_assignments[row["group_id"]].add((row["split"], row["dev_fold"]))
            if row["split"] == "holdout":
                self.assertNotIn(row["sample_id"], excl)
                self.assertEqual(row["dev_fold"], -1)
            else:
                self.assertIn(row["dev_fold"], {0, 1, 2, 3, 4})

        # Zero group crossing across splits or folds
        for gid, splits in group_assignments.items():
            self.assertEqual(len(splits), 1, f"Group {gid} crossed splits/folds: {splits}")

    def test_article_to_osh_id(self) -> None:
        self.assertEqual(article_to_osh_id("제4조의2"), "OSH-4-2")
        self.assertEqual(article_to_osh_id("제0043조"), "OSH-43")
        self.assertEqual(article_to_osh_id("619"), "OSH-619")
        self.assertEqual(article_to_osh_id("OSH-241"), "OSH-241")
        self.assertIsNone(article_to_osh_id(""))

    def test_rare_code_law_retrieval_priors(self) -> None:
        refs_cs = self.retriever.select_osh_references(
            "confined_space",
            "맨홀 내부 밀폐공간 출입구에 환기설비 없이 작업자가 진입하고 있다.",
            "산소 및 유해가스 농도를 측정하고 송기설비로 환기한 뒤 출입한다.",
        )
        self.assertTrue(refs_cs)
        self.assertEqual(refs_cs[0]["kind"], "law")
        self.assertRegex(refs_cs[0]["id"], r"^OSH-\d+(?:-\d+)?$")

        refs_dust = self.retriever.select_osh_references(
            "chemical_exposure",
            "절단 작업 중 시멘트 분진이 심하게 흩날려 작업자 호흡기를 덮고 있다.",
            "물을 뿌리는 등 분진이 흩날리는 것을 방지하는 조치를 실시한다.",
        )
        self.assertIn("OSH-4-2", [r["id"] for r in refs_dust])

    def test_format_competition_record_schema_firewall(self) -> None:
        hex_id = "0123456789abcdef0123456789abcdef"
        # 1. Malformed/empty VLM output must produce valid no_visible_hazard record
        rec_empty = format_competition_record(hex_id, None, self.retriever)
        self.assertEqual(list(self.validator.iter_errors(rec_empty)), [])
        self.assertEqual(rec_empty["status"], "no_visible_hazard")
        self.assertNotIn("references", rec_empty)

        # 2. Control chars, duplicate codes, >3 codes, out-of-range scores must be sanitized to 100% valid schema
        dirty_vlm = {
            "scene_note": "줄바꿈\n탭\t제어문자\x07 포함 설명",
            "status": "hazard",
            "hazards": [
                {"code": "fall_height", "likelihood_score": 4, "severity_score": 4, "evidence": "단부 난간 미설치\n확인.", "prevention": "안전난간 설치\r\n실시."},
                {"code": "fall_height", "likelihood_score": 2, "severity_score": 2, "evidence": "중복 코드.", "prevention": "무시됨."},
                {"code": "confined_space", "likelihood_score": 99, "severity_score": 4, "evidence": "밀폐 피트 무환기.", "prevention": "환기 및 농도 측정."},
                {"code": "other", "likelihood_score": 3, "severity_score": 3, "evidence": "비계 기둥 변형.", "prevention": "비계 보강 및 점검."},
                {"code": "slip_trip", "likelihood_score": 2, "severity_score": 1, "evidence": "4번째 초과 코드.", "prevention": "통로 정리."},
            ],
        }
        rec_haz = format_competition_record(hex_id, dirty_vlm, self.retriever)
        self.assertEqual(list(self.validator.iter_errors(rec_haz)), [])
        self.assertEqual(rec_haz["status"], "hazard")
        self.assertNotIn("scene_evidence", rec_haz)
        self.assertEqual(len(rec_haz["hazards"]), 3)
        self.assertEqual(set(rec_haz["hazards"].keys()), {"fall_height", "confined_space", "other"})

    def test_decision_span_regex_and_loose_parser(self) -> None:
        sample = {
            "sample_id": "T1",
            "target": {
                "status": "hazard",
                "hazards": {
                    "drowning_inundation": {
                        "evidence": "집수정 상부 개방 단부에 추락·익사 방호울이 없다.",
                        "prevention": "집수정 주변 안전난간 및 구명장비를 비치한다.",
                        "risk_level": {"likelihood": {"score": 4}, "severity": {"score": 4}},
                    }
                },
            },
            "notes": {"scene_note": "물이 찬 집수정 주변 작업 구간이다."},
        }
        vlm_t = gold_row_to_vlm_target(sample)
        self.assertEqual(validate_vlm_11code_obj(vlm_t), [])
        dumped = json.dumps(vlm_t, ensure_ascii=False, separators=(",", ":"))
        matched = [m.group(0) for m in DECISION_SPAN_RE.finditer(dumped)]
        self.assertEqual(len(matched), 4)  # status, code, likelihood_score, severity_score

        fenced = "```json\n" + dumped + "\n```"
        parsed, err = parse_json_loose(fenced)
        self.assertIsNone(err)
        self.assertEqual(parsed, vlm_t)

    def test_hybrid_candidate_selection_and_merge(self) -> None:
        eval_rows = [{"sample_id": "S1", "image": "images/000001.png"}]
        vlm_outs = [{"parsed": {"scene_note": "현장.", "status": "hazard", "hazards": [{"code": "fall_height", "likelihood_score": 4, "severity_score": 4, "evidence": "난간 없음.", "prevention": "난간 설치."}]}}]
        code_p = np.full((1, len(CODES)), 0.05)
        code_p[0, CODES.index("slip_trip")] = 0.72
        scene_p = np.array([0.85])
        calib = {"temperatures": [1.0] * len(CODES), "thresholds": [0.45] * len(CODES), "scene_threshold": 0.40}

        cands = select_booster_candidates(eval_rows, vlm_outs, code_p, scene_p, calib)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["code"], "slip_trip")

        verified = [{**cands[0], "accepted": True, "hazard": {"code": "slip_trip", "likelihood_score": 3, "severity_score": 2, "evidence": "통로 자재 방치.", "prevention": "자재 정리."}}]
        merged = merge_verified_candidates(vlm_outs, verified)
        self.assertEqual([h["code"] for h in merged[0]["parsed"]["hazards"]], ["fall_height", "slip_trip"])


if __name__ == "__main__":
    unittest.main()
