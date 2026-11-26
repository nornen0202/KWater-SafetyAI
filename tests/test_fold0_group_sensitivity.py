"""Identity and whole-group exclusion regression for the Fold 0 sensitivity."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kwater_lab.fold0_group_sensitivity import compare, sha256
from kwater_lab.fold0_production_compare import image_id


class Fold0GroupSensitivityTests(unittest.TestCase):
    def test_group_mask_after_exact_coverage_for_each_arm(self):
        with TemporaryDirectory(dir=".runtime") as scratch:
            root = Path(scratch)
            source = root / "manifest.json"
            exclusions = root / "exclusions.json"
            labels = root / "labels.jsonl"
            baseline, candidate, hybrid = [root / f"{arm}.jsonl" for arm in ("baseline", "candidate", "hybrid")]
            rows = [
                {"sample_id": f"local-{i}", "split": "dev", "dev_fold": 0,
                 "group_id": "shared" if i < 181 else f"unique-{i}", "codes": []}
                for i in range(216)
            ]
            train = {"sample_id": "local-training", "split": "dev", "dev_fold": 1, "group_id": "train", "codes": []}
            source.write_text(json.dumps(rows + [train]), encoding="utf-8")
            exclusion = {
                "canonical_manifest_sha256": sha256(source), "review_status": "synthetic",
                "confirmed_cross_group_links": [
                    {"review_number": "A", "train_group_id": "train", "fold0_group_id": "unique-181"},
                    {"review_number": "B", "train_group_id": "train", "fold0_group_id": "shared"},
                ],
            }
            exclusions.write_text(json.dumps(exclusion), encoding="utf-8")

            def empty(sid):
                return {"schema_version": "1.0", "image_id": sid, "status": "no_visible_hazard",
                        "hazards": {}, "scene_evidence": "Synthetic safe scene."}

            def write(path, records):
                path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

            write(labels, [{"sample_id": r["sample_id"], "target": empty(r["sample_id"])} for r in rows])
            local = [empty(r["sample_id"]) for r in rows]
            anon = [empty(image_id(r["sample_id"])) for r in rows]
            write(baseline, anon)
            write(candidate, local)
            write(hybrid, anon)
            result = compare(source, exclusions, labels, baseline, candidate, hybrid)
            self.assertEqual((result["primary_photos"], result["primary_groups"]), (216, 36))
            self.assertEqual((result["excluded_photos"], result["excluded_groups"]), (182, 2))
            self.assertEqual((result["retained_photos"], result["retained_groups"]), (34, 34))
            largest = result["largest_primary_group_diagnostic"]
            self.assertEqual((largest["photos"], largest["groups"], largest["other_photos"]), (181, 1, 35))
            self.assertEqual(largest["metrics"]["hybrid"]["photos"], 181)
            self.assertEqual(len(result["paired_group_bootstrap"]), 3)
            self.assertTrue(all(x["groups"] == 34 for x in result["paired_group_bootstrap"].values()))
            self.assertNotIn("local-", json.dumps(result))
            for path, records in ((baseline, anon), (candidate, local), (hybrid, anon)):
                write(path, records[1:])
                with self.assertRaisesRegex(ValueError, "must cover all 216"):
                    compare(source, exclusions, labels, baseline, candidate, hybrid)
                write(path, records)
            exclusion["canonical_manifest_sha256"] = "wrong"
            exclusions.write_text(json.dumps(exclusion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 differ"):
                compare(source, exclusions, labels, baseline, candidate, hybrid)


if __name__ == "__main__":
    unittest.main()
