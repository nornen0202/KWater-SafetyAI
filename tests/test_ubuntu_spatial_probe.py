"""Behavioral coverage for the label-blind spatial probe's evaluation boundary."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kwater_lab.spatial_probe import Row, _atomic_npz, _valid_cached, evaluate_cached, load_rows


class SpatialProbeTests(unittest.TestCase):
    def test_manifest_hash_and_sealed_rows_are_rejected(self):
        with TemporaryDirectory(dir=".runtime") as scratch:
            root = Path(scratch)
            manifest = root / "manifest.json"
            summary = root / "summary.json"
            manifest.write_text(json.dumps([{"sample_id": "a", "image": "a.png", "sha256": "a" * 64,
                                              "group_id": "g", "split": "holdout", "dev_fold": 1, "codes": []}]), encoding="utf-8")
            summary.write_text(json.dumps({"dev_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed"):
                load_rows(manifest, summary)
            summary.write_text(json.dumps({"dev_manifest_sha256": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_rows(manifest, summary)

    def test_equal_budget_heads_report_group_disjoint_common_code_metrics(self):
        with TemporaryDirectory(dir=".runtime") as scratch:
            cache = Path(scratch)
            rows = []
            for fold in (1, 2, 3):
                for index in range(6):
                    positive = index < 3
                    sample_id = f"f{fold}-{index}"
                    code = "slip_trip" if fold == 1 else "fall_height"
                    row = Row(sample_id, f"{sample_id}.png", f"{fold}{index}".ljust(64, "0"), f"group-{fold}-{index}", fold,
                              frozenset({code}) if positive else frozenset())
                    rows.append(row)
                    feature = np.zeros((16, 3), dtype=np.float32)
                    feature[:, 0] = 1.0 if positive else -1.0
                    _atomic_npz(cache / f"{row.sha256}-encoder.npz", features=feature,
                                source_sha256=np.asarray(row.sha256), encoder_sha256=np.asarray("encoder"))
            report = evaluate_cached(rows, cache, "encoder", top_k=1, seed=7, steps=30)
            self.assertEqual(report["parameter_budget"], "each head has D*C+C trainable parameters")
            for fold in ("1", "2", "3"):
                result = report["per_fold"][fold]
                self.assertEqual((result["fit_groups"], result["validation_groups"]), (12, 6))
                self.assertEqual(result["global_linear"]["ap_denominator_codes"], 0 if fold == "1" else 1)
                self.assertEqual(result["regional_mil_topk"]["recall_at_1_denominator"], 3)
            self.assertEqual(report["per_fold"]["1"]["global_linear"]["ranking_eligible_codes_from_fit_only"], ["fall_height"])

    def test_corrupt_content_cache_is_quarantined_for_rebuild(self):
        with TemporaryDirectory(dir=".runtime") as scratch:
            cache = Path(scratch)
            source = "a" * 64
            encoder = "encoder"
            broken = cache / f"{source}-{encoder}.npz"
            broken.write_bytes(b"not an npz")
            self.assertFalse(_valid_cached(source, cache, encoder))
            self.assertFalse(broken.exists())
            self.assertEqual(len(list(cache.glob("*.corrupt-*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
