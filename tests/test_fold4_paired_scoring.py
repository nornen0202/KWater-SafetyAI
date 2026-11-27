import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from kwater_lab import fold4_crosssplit_audit as audit
from kwater_lab.metrics import LIKELIHOOD, RISK_BANDS, SEVERITY


class Fold4PairedScoringTests(unittest.TestCase):
    def test_same_mask_and_pair_deltas_require_complete_inputs(self):
        with TemporaryDirectory(dir=".runtime") as scratch:
            root = Path(scratch)
            manifest, summary, output = (root / name for name in ("manifest", "audit", "output"))
            labels = root / "labels.jsonl"
            mask = root / "fold4-group-exclusions-local.json"
            arms = {key: root / key for key in ("legacy8", "v3", "hybrid")}
            code = "fall_height"
            rows = [
                {"sample_id": f"synthetic-{i}", "split": "dev", "dev_fold": 4,
                 "group_id": f"group-{i % 36}" if i < 107 else f"excluded-{i % 3}",
                 "codes": [code] if i in (0, 174) else []}
                for i in range(175)
            ]

            def target(i, hazard):
                row = {"schema_version": "1.0", "image_id": rows[i]["sample_id"],
                       "status": "hazard" if hazard else "no_visible_hazard", "hazards": {}}
                if hazard:
                    row["hazards"][code] = {
                        "evidence": "Visible hazard.", "prevention": "Install protection.",
                        "risk_level": {"likelihood": {"score": 1, "grade": LIKELIHOOD[0]},
                                       "severity": {"score": 4, "grade": SEVERITY[3]},
                                       "risk": {"score": 4, "grade": RISK_BANDS[1][1]}},
                    }
                else:
                    row["scene_evidence"] = "No visible hazard."
                return row

            manifest.write_text(json.dumps(rows), encoding="utf-8")
            mask.write_text(json.dumps({
                "canonical_manifest_sha256": audit.digest(manifest),
                "confirmed_cross_group_links": [
                    {"fold4_group_id": f"excluded-{i}"} for i in range(3)
                ],
            }), encoding="utf-8")
            summary.write_text(json.dumps({
                "confirmed_mask_sha256": audit.digest(mask),
                "sensitivity_photos": 107, "sensitivity_groups": 36,
            }), encoding="utf-8")
            gold = [{"sample_id": r["sample_id"], "target": target(i, bool(r["codes"]))}
                    for i, r in enumerate(rows)]
            labels.write_text("".join(json.dumps(row) + "\n" for row in gold), encoding="utf-8")
            for name, positive in {"legacy8": (), "v3": (0,), "hybrid": (0, 1, 174)}.items():
                arms[name].write_text("".join(json.dumps(target(i, i in positive)) + "\n"
                                              for i in (reversed(range(175)) if name == "v3" else range(175))), encoding="utf-8")

            with patch.object(audit, "DEV", manifest), patch.object(audit, "RESULT", summary), patch(
                "kwater_lab.fold4_crosssplit_audit.LABELS", labels
            ):
                audit.score_three(root, arms, output)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(result["prediction_rows_in_canonical_manifest_order"],
                                 {"legacy8": True, "v3": False, "hybrid": True})
                full, kept = (result["panels"][key] for key in ("primary_175", "sensitivity_107"))
                self.assertEqual((full["photos"], full["groups"], kept["photos"], kept["groups"]),
                                 (175, 39, 107, 36))
                exploratory = kept["leave_largest_group_out"]
                self.assertEqual((exploratory["excluded_photos"], exploratory["photos"], exploratory["groups"]),
                                 (3, 104, 35))
                self.assertEqual(len(exploratory["macro_f1_delta_b_minus_a"]), 3)
                self.assertEqual(exploratory["macro_f1_codes_lost_after_exclusion"]["v3"], [code])
                self.assertEqual(exploratory["macro_f1_active_codes"]["hybrid"], [code])
                for panel in (full, kept):
                    self.assertEqual(len(panel["comparisons"]), 3)
                    legacy = panel["arm_metrics"]["legacy8"]
                    hybrid = panel["arm_metrics"]["hybrid"]
                    self.assertEqual(legacy["severity4"]["missed_code_instances"], legacy["severity4"]["photos"])
                    self.assertEqual(hybrid["severity4"]["missed_code_instances"], 0)
                    self.assertEqual(hybrid["no_hazard_false_alarms"], 1)
                    self.assertEqual(hybrid["per_code"][code]["fp"], 1)
                    pair = panel["comparisons"]["v3_vs_hybrid"]
                    self.assertEqual(pair["groups"], panel["groups"])
                    self.assertAlmostEqual(pair["macro_f1_delta_b_minus_a"],
                                           hybrid["macro_f1"] - panel["arm_metrics"]["v3"]["macro_f1"])
                    if panel is full:
                        self.assertGreater(pair["macro_f1_delta_b_minus_a"], 0)
                    else:
                        self.assertLess(pair["macro_f1_delta_b_minus_a"], 0)
                    expected_changed = 2 if panel is full else 1
                    self.assertEqual(pair["changed_code_set_photo_rate"], expected_changed / panel["photos"])
                    self.assertEqual(pair["per_code_change"][code]["added_by_b"], expected_changed)
                epoch1_output = root / "epoch1.json"
                audit._score_panels(root, {"legacy8": arms["legacy8"], "v3_epoch1": arms["v3"]},
                                    epoch1_output, "v3_epoch1_standalone")
                epoch1 = json.loads(epoch1_output.read_text(encoding="utf-8"))
                self.assertEqual(epoch1["evaluation_run"], "v3_epoch1_standalone")
                for key, panel in epoch1["panels"].items():
                    self.assertEqual(len(panel["comparisons"]), 1)
                    self.assertEqual(panel["comparisons"]["legacy8_vs_v3_epoch1"],
                                     result["panels"][key]["comparisons"]["legacy8_vs_v3"])
                self.assertNotIn("synthetic-", json.dumps(result))
                broken = arms["hybrid"]
                lines = broken.read_text(encoding="utf-8").splitlines()
                output.unlink()
                invalid = (
                    (lines[:-1], "hybrid must contain every original"),
                    (lines[:-1] + [json.dumps({**target(174, True), "image_id": "invalid-id"})],
                     "hybrid must contain every original"),
                    (lines[:-1] + [lines[0]], "duplicate ID"),
                    (lines[:-1] + ['{"unfinished":'], "hybrid:175"),
                )
                for records, message in invalid:
                    with self.subTest(message=message):
                        broken.write_text("\n".join(records), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, message):
                            audit.score_three(root, arms, output)
                        self.assertFalse(output.exists())
                broken.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Missing hybrid"):
                    audit.score_three(root, arms, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
