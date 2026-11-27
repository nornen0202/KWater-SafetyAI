"""Local CPU checks for the matched hard-negative curriculum boundary."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from kwater_lab.spatial_probe import Row
from kwater_lab.ubuntu_mined_verifier import (
    MANIFEST,
    MiningBoundaryError,
    main,
    mine_codes,
    prepare,
    replace_hard_negatives,
    validate_rows,
)
from kwater_lab.vlm_11code import verify_user_text


def rows():
    return [
        Row("a", "a.jpg", "sha-a", "group-a", 1, frozenset({"fall_height"})),
        Row("b", "b.jpg", "sha-b", "group-b", 1, frozenset({"slip_trip"})),
        Row("c", "c.jpg", "sha-c", "group-c", 2, frozenset({"electrical_energy"})),
        Row("d", "d.jpg", "sha-d", "group-d", 2, frozenset({"slip_trip"})),
    ]


class MinedVerifierTests(unittest.TestCase):
    def test_fold_group_image_and_source_hash_boundaries(self):
        validate_rows(rows())
        for update in ({"fold": 3}, {"fold": 0}, {"fold": 4}, {"group_id": "group-a"},
                       {"sha256": "sha-a"}, {"image": "a.jpg"}, {"sample_id": "a"}):
            with self.subTest(update=update):
                changed = rows()
                changed[2] = replace(changed[2], **update)
                with self.assertRaises(MiningBoundaryError):
                    validate_rows(changed)

    def test_miner_fits_only_opposite_fold_and_ties_use_sorted_fit_codes(self):
        features = np.stack([np.full((16, 2), index, dtype=np.float32) for index in range(4)])

        def score(fit_features, fit_labels, held_features, **settings):
            if settings["seed"] == 20261128:
                np.testing.assert_array_equal(fit_features[:, 0, 0], [2, 3])
                np.testing.assert_array_equal(held_features[:, 0, 0], [0, 1])
            else:
                np.testing.assert_array_equal(fit_features[:, 0, 0], [0, 1])
                np.testing.assert_array_equal(held_features[:, 0, 0], [2, 3])
            np.testing.assert_array_equal(fit_labels, [[1, 0], [0, 1]])
            self.assertEqual((settings["top_k"], settings["regional"], settings["device_name"]), (3, True, "cpu"))
            return np.zeros((2, 2), dtype=np.float32)

        with patch("kwater_lab.ubuntu_mined_verifier._fit_scores", side_effect=score):
            selected = mine_codes(rows(), features)
        self.assertEqual(selected, {
            "a": "electrical_energy", "b": "electrical_energy", "c": "fall_height", "d": "fall_height",
        })

    def test_real_cpu_head_produces_absent_fit_supported_codes(self):
        features = np.random.default_rng(9).normal(size=(4, 16, 3)).astype(np.float32)
        selected = mine_codes(rows(), features)
        for row in rows():
            supported = {code for other in rows() if other.fold != row.fold for code in other.codes}
            self.assertIn(selected[row.sample_id], supported - row.codes)
        self.assertEqual(selected, mine_codes(rows(), features))

    def test_replacements_keep_order_images_nonhard_examples_and_reject_labels(self):
        baseline = [
            {"sample_id": "a", "image": "a.jpg", "kind": "inspect", "augment": False,
             "user_text": "inspect", "assistant_text": "unchanged"},
            {"sample_id": "a__ver_neg_other", "image": "a.jpg", "kind": "verify_neg_hard", "augment": False,
             "user_text": verify_user_text("other"), "assistant_text": "original reject"},
            {"sample_id": "b__ver_neg_other", "image": "b.jpg", "kind": "verify_neg_safe", "augment": False,
             "user_text": "safe verification", "assistant_text": "unchanged safe reject"},
        ]
        original = deepcopy(baseline)
        result = replace_hard_negatives(baseline, rows(), {"a": "electrical_energy"})
        self.assertEqual(baseline, original)
        self.assertEqual((result[0], result[2]), (baseline[0], baseline[2]))
        self.assertEqual([item["image"] for item in result], ["a.jpg", "a.jpg", "b.jpg"])
        self.assertEqual(result[1]["sample_id"], "a__ver_neg_electrical_energy")
        self.assertEqual(result[1]["user_text"], verify_user_text("electrical_energy"))
        answer = json.loads(result[1]["assistant_text"])
        self.assertEqual(answer["decision"], "reject")
        self.assertIn("감전", answer["reason"])
        for code in ("fall_height", "other"):
            with self.subTest(code=code), self.assertRaises(MiningBoundaryError):
                replace_hard_negatives(baseline, rows(), {"a": code})

    def test_unapproved_manifest_fails_before_reading_labels_or_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / MANIFEST).parent.mkdir(parents=True)
            (root / MANIFEST).write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(MiningBoundaryError, "SHA-256"):
                prepare(root, root / "prepared")

    def test_training_refuses_unchanged_scheduled_examples_before_gpu_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"changed_slots_in_first_96": 0, "first_changed_checkpoint_step": 18}
            with (
                patch("kwater_lab.ubuntu_mined_verifier.Path.cwd", return_value=root),
                patch("sys.argv", ["mined", "train", str(root / "data/mined")]),
                patch("kwater_lab.ubuntu_mined_verifier.prepare", return_value=([], [], report)),
                patch("kwater_lab.ubuntu_mined_verifier.train_multimodal_qlora") as training,
            ):
                with self.assertRaisesRegex(MiningBoundaryError, "checkpoint is 18"):
                    main()
                training.assert_not_called()


if __name__ == "__main__":
    unittest.main()
