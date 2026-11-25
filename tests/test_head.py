"""Synthetic vectors exercise the training and leakage checks, not photo accuracy."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kwater_lab import train_head as head
from kwater_lab.loss_pilot import asymmetric_gradient


class HeadTests(unittest.TestCase):
    def test_asymmetric_logit_gradient_matches_the_stated_objective(self):
        logits = np.array([-4.2, -0.7, 1.1, -3.7, 2.8])
        labels = np.array([0., 0., 0., 1., 1.])
        for power in (0, 4):
            def loss(values):
                p = head.sigmoid(values)
                q = np.maximum(p - 0.05, 0.0)
                return np.where(labels, -np.log(p), -(q ** power) * np.log1p(-q))
            h = 1e-5
            expected = (loss(logits + h) - loss(logits - h)) / (2*h)
            actual = asymmetric_gradient(head.sigmoid(logits), labels, power=power)
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.audit_path = root / "audit.json"
        self.feature_path = root / "features.npz"
        self.model_path = root / "head.npz"
        self.report_path = root / "report.json"
        self.audit = {"status": "ready", "labels_sha256": "a"*64, "assignments": {}, "inventory": {}}
        ids, rows = [], []
        for prefix, size in (("train", 8), ("dev", 4), ("holdout", 2)):
            for i in range(size):
                sid = f"{prefix}{i}"
                self.audit["assignments"][sid] = {"split": prefix, "group_id": f"{prefix}-{i//2}"}
                if prefix == "holdout":
                    # No labels or image paths are available to the trainer for these rows.
                    self.audit["inventory"][sid] = {}
                    continue
                self.audit["inventory"][sid] = {"codes": ["fall_height"] if i%2 == 0 else []}
                ids.append(sid)
                vector = np.zeros(512, np.float32)
                vector[1], vector[0] = 1, 1 if i%2 == 0 else -1
                rows.append(vector)
        self.save_audit()
        self.archive = {"ids": np.array(ids[::-1]), "features": np.array(rows[::-1], np.float32),
                        "revision": "synthetic-test", "framing": "full_frame",
                        "labels_sha256": "a"*64, "audit_sha256": head.file_hash(self.audit_path)}
        self.save_archive()

    def save_audit(self):
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")

    def save_archive(self):
        np.savez(self.feature_path, **self.archive)

    def test_end_to_end_and_saved_model(self):
        report = head.train(self.audit_path, self.feature_path, self.model_path, self.report_path, steps=40)
        self.assertEqual(report["training_groups"], 4)
        self.assertEqual(report["development_groups"], 2)
        self.assertEqual(report["holdout_images_read"], 0)
        self.assertEqual(report["linear_head"]["fixed_0.5"]["macro_f1"], 1)
        self.assertAlmostEqual(report["prior_baseline"]["selected"]["macro_f1"], 2/3)
        self.assertEqual(len(report["linear_head"]["grid"]), 7)
        self.assertEqual(head.file_hash(self.model_path), report["model_file_sha256"])
        self.assertEqual(json.loads(self.report_path.read_text(encoding="utf-8")), report)
        with np.load(self.model_path, allow_pickle=False) as saved:
            self.assertEqual(tuple(saved["code_order"]), head.CODES)
            data, _ = head.checked_data(self.audit_path, self.feature_path)
            x, labels, groups = data["dev"]
            probs = head.probabilities(saved, x)
            decisions = head.decisions(labels, probs, saved["threshold"].item())
            self.assertEqual(head.evaluate(labels, decisions, groups), report["linear_head"]["selected"])
            self.assertTrue(np.all(probs[:, 1:] == 0))
            np.testing.assert_array_equal(saved["center"], data["train"][0].mean(axis=0))

    def test_reject_group_leak_including_holdout(self):
        self.audit["assignments"]["holdout0"]["group_id"] = self.audit["assignments"]["train0"]["group_id"]
        self.save_audit()
        self.archive["audit_sha256"] = head.file_hash(self.audit_path)
        self.save_archive()
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            head.checked_data(self.audit_path, self.feature_path)

    def test_reject_feature_integrity_errors(self):
        for field, value, message in [
            ("ids", np.append(self.archive["ids"], "holdout0"), "exactly the train/dev"),
            ("ids", np.repeat("dev0", len(self.archive["ids"])), "exactly the train/dev"),
            ("ids", self.archive["ids"][1:], "exactly the train/dev"),
            ("features", self.archive["features"].astype(np.float64), "finite float32"),
            ("features", np.full_like(self.archive["features"], np.nan), "finite float32"),
            ("labels_sha256", "b"*64, "hashes differ"), ("audit_sha256", "b"*64, "hashes differ"),
        ]:
            with self.subTest(field=field, message=message):
                np.savez(self.feature_path, **{**self.archive, field: value})
                with self.assertRaisesRegex(ValueError, message):
                    head.checked_data(self.audit_path, self.feature_path)

    def test_threshold_top_three_and_empty_development_set_of_codes(self):
        probs = np.zeros((1, len(head.CODES)))
        probs[0, :4] = 0.8
        self.assertEqual(head.decisions(["a"], probs, 0.8)["a"], frozenset(head.CODES[:3]))
        result = head.calibration(np.zeros_like(probs), {"a": frozenset()}, {"a": "g"})
        self.assertEqual(result["selected_threshold"], 0.5)
        self.assertIsNone(result["selected"]["macro_f1"])
        self.assertEqual(result["selection"], "fixed_0.5_no_positive_dev_codes")


if __name__ == "__main__":
    unittest.main()
