"""Controlled incident-split checks; independent of competition data."""

import json
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kwater_lab.audit import audit


class IncidentAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.labels = self.root / "train.jsonl"
        self.rows = []
        for i in range(8):
            key = f"S16C{i:03d}"
            # The first two encodings have identical displayed pixels. The next
            # two have different pixels with the same gradient for near review.
            if i < 2:
                image = Image.frombytes("L", (9, 8), random.Random(17).randbytes(72))
            elif i in (2, 3):
                image = Image.frombytes("L", (9, 8), bytes((240 - x * 14 - i)
                                                          for y in range(8) for x in range(9)))
            else:
                image = Image.frombytes("L", (9, 8), random.Random(i).randbytes(72))
            name = f"photo{i}.png"
            image.save(self.root / name, compress_level=(i % 2) * 9)
            self.rows.append({"sample_id": key, "image": name, "meta": {"incident": f"case{i}"},
                              "target": {"schema_version": "1.0", "image_id": key,
                                         "status": "no_visible_hazard", "hazards": {},
                                         "scene_evidence": "보이는 통로가 비어 있다."}})
        # Same incident with two distinct photographs.
        self.rows[5]["meta"]["incident"] = self.rows[4]["meta"]["incident"]
        self.flush()

    def flush(self):
        self.labels.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                         for row in self.rows) + "\n", encoding="utf-8")

    def reviewed(self, decision="separate"):
        path = self.root / "reviewed.csv"
        path.write_text(f"left,right,decision\nS16C002,S16C003,{decision}\n", encoding="utf-8")
        return path

    def test_missing_case_ids_are_not_guessed(self):
        result = audit(self.labels)
        self.assertEqual(result["status"], "needs_group_review")
        self.assertEqual(len(result["unmapped_sample_ids"]), 8)
        self.assertNotIn("assignments", result)
        with self.assertRaises(ValueError):
            audit(self.labels, case_field="sample_id")
        self.assertEqual(result["exact_pixel_duplicates"], [["S16C000", "S16C001"]])

    def test_explicit_case_pixel_and_review_grouping(self):
        pending = audit(self.labels, case_field="meta.incident")
        self.assertEqual(pending["pending_pair_reviews"], 1)
        self.assertNotIn("assignments", pending)
        report = audit(self.labels, case_field="meta.incident", review_pairs=self.reviewed("same"))
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["cross_split_group_violations"], 0)
        self.assertEqual(report["incident_groups"], 5)
        parts = report["assignments"]
        for x, y in [(0, 1), (2, 3), (4, 5)]:
            self.assertEqual(parts[f"S16C{x:03d}"], parts[f"S16C{y:03d}"])
        self.assertEqual({p["split"] for p in parts.values()}, {"train", "dev", "holdout"})
        self.rows.reverse()
        self.flush()
        rerun = audit(self.labels, case_field="meta.incident", review_pairs=self.reviewed("same"))
        self.assertEqual(parts, rerun["assignments"])

    def test_reviewed_group_map_and_negative_pair(self):
        mapping = self.root / "groups.csv"
        values = "".join(f"{r['sample_id']},{r['meta']['incident']}\n" for r in self.rows)
        mapping.write_text("sample_id,group_id\n" + values, encoding="utf-8")
        report = audit(self.labels, group_map=mapping, review_pairs=self.reviewed())
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["incident_groups"], 6)
        self.assertEqual(report["pending_pair_reviews"], 0)
        self.assertNotEqual(report["assignments"]["S16C002"]["group_id"],
                            report["assignments"]["S16C003"]["group_id"])

    def test_exposure_excludes_whole_linked_incident_from_final_holdout(self):
        manifest = self.root / "exposure.json"
        sha = hashlib.sha256(self.labels.read_bytes()).hexdigest()
        config = dict(case_field="meta.incident", review_pairs=self.reviewed("same"), exclude_holdout_ids=manifest)
        exposed = ["S16C000", "S16C002", "S16C004", "S16C006"]
        manifest.write_text(json.dumps({"labels_sha256": sha, "exposed_ids": exposed}), encoding="utf8")
        for seed in (0, 5, 23):
            report = audit(self.labels, seed=seed, **config)
            self.assertEqual(report["status"], "ready")
            holdout = {k for k, r in report["assignments"].items() if r["split"] == "holdout"}
            self.assertEqual(holdout, {"S16C007"})
        manifest.write_text(json.dumps({"labels_sha256": sha, "exposed_ids": exposed + ["S16C007"]}), encoding="utf8")
        self.assertEqual(audit(self.labels, **config)["status"], "no_unexposed_holdout_group")
        manifest.write_text(json.dumps({"labels_sha256": "0"*64, "exposed_ids": exposed}), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            audit(self.labels, **config)

    def test_invalid_images_and_conflicting_pair_fail_closed(self):
        pair = self.reviewed()
        with pair.open("a", encoding="utf-8") as stream:
            stream.write("S16C000,S16C001,separate\n")
        conflict = audit(self.labels, case_field="meta.incident", review_pairs=pair)
        self.assertEqual(conflict["status"], "invalid_data")
        self.assertNotIn("assignments", conflict)
        (self.root / self.rows[-1]["image"]).write_bytes(b"broken image")
        result = audit(self.labels, case_field="meta.incident")
        self.assertEqual(result["status"], "invalid_data")
        self.assertEqual(result["summary"]["photos"], 7)
        self.assertIn("line 8:", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
