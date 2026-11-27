import copy
import unittest
from unittest.mock import patch

from kwater_lab.official_anchored_verifier import add_verified_records, preserve_official_negative, reusable_raw


class OfficialAnchoredVerifierTests(unittest.TestCase):
    def test_cache_reuses_only_the_same_photo_and_code(self):
        candidates = [{"sample_id": "a", "image": "x.png", "code": "fall_height"},
                      {"sample_id": "b", "image": "y.png", "code": "other"}]
        identity = {"model_key": "bf16", "prompt_sha256": "prompt", "max_new_tokens": 320}
        cache = {"fingerprint": {"batch_size": 1, "inputs": {**identity, "selected_images_sha256": {"x.png": "x", "y.png": "old"}},
                                 "candidates": candidates},
                 "results": [{**c, "raw_verify_text": "raw"} for c in candidates]}
        self.assertEqual(reusable_raw(cache, identity, candidates, {"x.png": "x", "y.png": "new"}),
                         {("a", "fall_height", "x.png"): "raw"})
        with self.assertRaisesRegex(ValueError, "different precision"):
            reusable_raw(cache, {**identity, "model_key": "nf4"}, candidates, {"x.png": "x", "y.png": "old"})
        cache["results"].pop()
        with self.assertRaisesRegex(ValueError, "completed"):
            reusable_raw(cache, identity, candidates, {"x.png": "x", "y.png": "old"})

    def test_verified_law_references_only_touch_the_new_hazard(self):
        old = {"evidence": "공식 근거", "prevention": "공식 조치", "references": [{"kind": "law", "id": "OSH-3"}],
               "risk_level": {"likelihood": {"grade": "상", "score": 4}, "severity": {"grade": "최대", "score": 4},
                              "risk": {"grade": "매우높음", "score": 16}}}
        baseline = {"a": {"schema_version": "1.0", "image_id": "0" * 32, "status": "hazard", "hazards": {"other": old}},
                    "b": {"schema_version": "1.0", "image_id": "1" * 32, "status": "no_visible_hazard", "hazards": {},
                          "scene_evidence": "공식 장면 설명"}}
        original = copy.deepcopy(baseline)
        hazard = {"code": "fall_height", "evidence": "새 근거", "prevention": "새 조치", "severity_score": 4, "likelihood_score": 2}
        with patch("kwater_lab.official_anchored_verifier.read_selected_cases", return_value=[]), \
             patch("kwater_lab.official_anchored_verifier.Native11LawRetriever") as ranker:
            ranker.return_value.select_osh_references.return_value = [{"kind": "law", "id": "OSH-42"}]
            merged = add_verified_records(baseline, [{"sample_id": sid, "code": "fall_height", "hazard": hazard, "accepted": True}
                                                      for sid in baseline])
        self.assertEqual(baseline, original)
        self.assertEqual(merged["a"]["hazards"]["other"], old)
        self.assertEqual(merged["a"]["hazards"]["fall_height"]["references"], [{"kind": "law", "id": "OSH-42"}])
        self.assertNotIn("scene_evidence", merged["b"])
        self.assertEqual(merged["b"]["status"], "hazard")
        accepted = {"a": "fall_height", "b": "fall_height"}
        replayed = preserve_official_negative(baseline, merged, accepted)
        self.assertEqual(replayed, {"a": merged["a"], "b": original["b"]})
        merged["a"]["hazards"]["other"]["evidence"] = "다른 기존 근거"
        with self.assertRaisesRegex(ValueError, "preserve official hazards"):
            preserve_official_negative(baseline, merged, accepted)
