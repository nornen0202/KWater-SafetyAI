import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kwater_lab.paired_verifier_ab_score import decision, read_verifications, scored_arm, write_checked_predictions


class PairedVerifierScoringTests(unittest.TestCase):
    def test_malformed_rejection_cannot_inflate_negative_decision_accuracy(self):
        self.assertEqual(decision('{"decision":"reject","reason":"사진에서 확인되지 않습니다."}', "fall_height"), "reject")
        for text in ('{"decision":"reject","reason":"..."}', '{"decision":"reject","reason":"\\ud800"}',
                     '{"decision":"reject","decision":"reject","reason":"사유"}', '아니오'):
            self.assertEqual(decision(text, "fall_height"), "malformed")
        result = scored_arm({"a": "fall_height", "b": "fall_height"}, {"a": "malformed", "b": "reject"},
                            {"a": set(), "b": set()}, {"a": set(), "b": set()}, {"a": "g", "b": "h"},
                            {"a": set(), "b": set()})["verifier"]
        self.assertEqual(result["decision_accuracy"], 0.5)
        self.assertEqual(result["ambiguous_or_invalid_model_responses"], 1)
        self.assertIsNone(result["accepted_visually_ambiguous_ground_truth"])

    def test_verification_cache_matches_the_frozen_image_and_strict_parser(self):
        proposals = [{"sample_id": "a", "code": "fall_height", "image": "x.png"}]
        identity = {"adapter_sha256": "fake"}
        cache = {"fingerprint": {"batch_size": 1, "candidates": proposals,
                                 "inputs": {**identity, "device_model_key": "model", "evaluation_runtime": {"precision": "BF16"}}},
                 "results": [{**proposals[0], "accepted": False, "hazard": None,
                              "raw_verify_text": '{"decision":"reject","reason":"없습니다."}'}],
                 "verification_seconds": 1.0}
        scratch = Path(".runtime/temp")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder, \
             patch("kwater_lab.paired_verifier_ab_score.verification_identity", return_value=identity):
            workspace = Path(folder)
            path = workspace / "A/raw-verifier-local.json"
            path.parent.mkdir()
            path.write_text(json.dumps(cache), encoding="utf-8")
            actual, _ = read_verifications({"candidates": proposals}, workspace, "A", None)
            self.assertEqual(actual["a"]["verdict"], "reject")
            cache["results"][0]["accepted"] = True
            path.write_text(json.dumps(cache), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict parser"):
                read_verifications({"candidates": proposals}, workspace, "A", None)
            cache["results"][0]["image"] = "y.png"
            path.write_text(json.dumps(cache), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same order"):
                read_verifications({"candidates": proposals}, workspace, "A", None)

    def test_writer_keeps_original_bytes_for_unmodified_rows(self):
        first = b'{"image_id":"one", "hazards":{}}\r\n'
        second = b'{"image_id":"two", "hazards":{}}\r\n'
        baseline = {"a": json.loads(first), "b": json.loads(second)}
        result = {"a": baseline["a"], "b": {**baseline["b"], "hazards": {"other": {"evidence": "근거"}}}}
        scratch = Path(".runtime/temp")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            root = Path(folder)
            source = root / "official"
            (source / "output").mkdir(parents=True)
            (source / "output/submission.jsonl").write_bytes(first + second)
            with patch("kwater_lab.paired_verifier_ab_score.OFFICIAL", source), \
                 patch("kwater_lab.paired_verifier_ab_score.check_official_output") as checker:
                receipt = write_checked_predictions(root, "A", baseline, result)
            checker.assert_called_once()
            rows = (root / "A/official-anchored/output/submission.jsonl").read_bytes().splitlines(keepends=True)
            self.assertEqual(rows[0], first)
            self.assertEqual(json.loads(rows[1]), result["b"])
            self.assertEqual(receipt["unchanged_rows_verbatim"], 1)
