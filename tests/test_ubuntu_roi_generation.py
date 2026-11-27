"""Frozen cohort ordering and completed-output integrity for generation scoring."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kwater_lab.ubuntu_roi_generation import (
    AnchorError,
    seal_outputs,
    select_cohort,
    verify_seal,
    write_checked_submission,
)
from kwater_lab.ubuntu_roi_views import Photo


class GenerationAnchorTests(unittest.TestCase):
    def test_official_checker_accepts_submission_only_directory_on_retry(self) -> None:
        with TemporaryDirectory() as folder:
            diagnostics = Path(folder) / "arm"
            inputs = Path(folder) / "inputs"
            diagnostics.mkdir()
            inputs.mkdir()
            image_id = "a" * 32
            (inputs / "manifest.jsonl").write_text(
                json.dumps({"image_id": image_id}) + "\n"
            )
            (diagnostics / "internal.jsonl").write_text("private intermediate\n")
            (diagnostics / "checker.log").write_text("previous failed attempt\n")
            text = (
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "image_id": image_id,
                        "status": "no_visible_hazard",
                        "hazards": {},
                        "scene_evidence": "No visible hazard.",
                    }
                )
                + "\n"
            )

            for _ in range(2):
                submission = write_checked_submission(text, diagnostics, inputs)

            self.assertEqual(submission.read_text(), text)
            self.assertEqual(
                {path.name for path in submission.parent.iterdir()},
                {"submission.jsonl"},
            )
            self.assertIn("1/1", (diagnostics / "checker.log").read_text())
            self.assertEqual(
                (diagnostics / "internal.jsonl").read_text(), "private intermediate\n"
            )

    def test_frozen_order_first_two_per_prefix_excludes_later_candidates(self) -> None:
        rows = [
            Photo(f"{prefix}-{number}", "unused.png", "0" * 64, str(prefix))
            for number in range(4)
            for prefix in range(6)
        ]

        selected = select_cohort(rows)

        self.assertEqual(
            [row.sample_id for row in selected],
            [f"{prefix}-{number}" for number in range(2) for prefix in range(6)],
        )

    def test_missing_condition_cannot_be_sealed(self) -> None:
        with TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "plan.json").write_text(
                json.dumps({"rows": [{"image_id": "photo"}], "arms": ["a", "b"]})
            )
            (run / "records").mkdir()
            (run / "records/photo-a.json").write_text("{}")

            with self.assertRaises(AnchorError):
                seal_outputs(run)

            self.assertFalse((run / "sealed.json").exists())

    def test_changed_prediction_is_rejected_after_successful_seal(self) -> None:
        with TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "plan.json").write_text(
                json.dumps({"rows": [{"image_id": "photo"}], "arms": ["a"]})
            )
            (run / "records").mkdir()
            record = run / "records/photo-a.json"
            record.write_text('{"prediction": "original"}')
            seal_outputs(run)
            verify_seal(run)
            record.write_text('{"prediction": "changed"}')

            with self.assertRaises(AnchorError):
                verify_seal(run)


if __name__ == "__main__":
    unittest.main()
