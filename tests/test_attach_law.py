import copy
import unittest
from unittest.mock import patch

from kwater_lab.attach_law import choose_examples, prediction_with_article


class AttachLawTests(unittest.TestCase):
    def setUp(self):
        self.exposure = {"exposed_ids": ["A", "B", "C"], "by_prefix": {"p": ["A", "B"], "q": ["C"]}}
        self.inventory = {k: {"pixels_sha256": k, "dhash64": bits} for k, bits in
                          (("A", "0000000000000000"), ("B", "0fffff00ffffffff"),
                           ("C", "ffffffffffffffff"), ("X", "fffffffffffffffc"))}
        self.articles = {"OSH-3": {"title": "전도의 방지"}}
        self.item = {"evidence": "바닥에 놓인 케이블이 보인다.", "prevention": "통로를 정리한다.",
                     "risk_level": {"likelihood": {"grade": "하", "score": 2},
                                    "severity": {"grade": "소", "score": 1},
                                    "risk": {"grade": "매우낮음", "score": 2}}}
        self.row = {"schema_version": "1.0", "image_id": "f"*32, "status": "hazard",
                    "hazards": {"slip_trip": self.item}}

    def test_no_opaque_identity_uses_no_examples(self):
        self.assertEqual(choose_examples(self.exposure, self.inventory, None, None)[0], [])

    def test_known_or_visually_near_samples_excluded(self):
        self.assertEqual(choose_examples(self.exposure, self.inventory, ["A"], None)[0], ["C"])
        self.assertEqual(choose_examples(self.exposure, self.inventory, None, [self.inventory["X"]])[0], ["A", "B"])
        self.assertEqual(choose_examples(self.exposure, self.inventory, ["A"], [self.inventory["X"]])[0], [])
        with self.assertRaises(ValueError):
            choose_examples(self.exposure, self.inventory, ["unknown"], None)

    def test_only_candidate_active_law_changes(self):
        nonlaw = {"kind": "catalog", "id": "P1234567890abcd"}
        self.item["references"] = [{"kind": "law", "id": "OSH-99"}, nonlaw]
        before = copy.deepcopy(self.row)
        with patch("kwater_lab.attach_law.official_ranking", return_value=["OSH-999", "OSH-3", "OSH-4"]) as rank:
            updated, choices = prediction_with_article(self.row, object(), self.articles)
        self.assertEqual(self.row, before)
        expected = copy.deepcopy(before)
        expected["hazards"]["slip_trip"]["references"] = [nonlaw, {"kind": "law", "id": "OSH-3"}]
        self.assertEqual(updated, expected)
        self.assertEqual(choices[0]["article_id"], "OSH-3")
        self.assertEqual(rank.call_args.args[1:], (self.item["evidence"], self.item["prevention"], "slip_trip"))

    def test_full_references_and_no_hazard(self):
        self.item["references"] = [{"kind": "catalog", "id": "P" + d*14} for d in "abc"]
        safe = {**self.row, "status": "no_visible_hazard", "hazards": {}, "scene_evidence": "통로가 비어 있다."}
        with patch("kwater_lab.attach_law.official_ranking", side_effect=AssertionError):
            result, choices = prediction_with_article(self.row, object(), self.articles)
            self.assertEqual(result, self.row)
            self.assertEqual(choices[0]["status"], "no_slot_after_nonlaw_references")
            self.assertEqual(prediction_with_article(safe, object(), self.articles), (safe, []))
