"""Constraint selection uses only inner-fold predictions."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from kwater_lab.metrics import CODES
from kwater_lab.prefix_pilot import calibrate_fpr20, get_features, policy_decisions


class PilotConstraintTests(unittest.TestCase):
    def test_siglip_cache_accepts_only_the_declared_encoder_and_width(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "features.npz"
            vectors = np.zeros((1,768), dtype=np.float32)
            vectors[0,0] = 1.0
            np.savez(path, ids=["photo"], features=vectors, revision="siglip")
            args = (path, ["photo"], {}, None, None)
            values, how = get_features(*args, {"revision": "siglip"}, dimension=768)
            self.assertEqual(how, "reused")
            np.testing.assert_array_equal(values["photo"], vectors[0])
            with self.assertRaises(ValueError):
                get_features(*args, {"revision": "other"}, dimension=768)
            with self.assertRaises(ValueError):
                get_features(*args, {"revision": "siglip"})

    def setUp(self):
        self.gold = {"p": frozenset({"fall_height"}), **{f"n{i}": frozenset() for i in range(5)}}
        self.groups = {k: k for k in self.gold}
        self.scores = np.zeros((len(self.gold), len(CODES)))

    def test_one_of_five_is_feasible_and_two_are_not(self):
        self.scores[:, 0] = [0.9, 0.7, 0.4, 0.1, 0.1, 0.1]
        result = calibrate_fpr20(self.scores, self.gold, self.groups)
        grid = {v["threshold"]: v for v in result["candidates"]}
        self.assertEqual(grid[0.7]["no_hazard_false_alarms"], 1)
        self.assertEqual(grid[0.4]["no_hazard_false_alarms"], 2)
        self.assertEqual(result["chosen"]["threshold"], 0.8)
        self.assertEqual(result["eligible_numeric_thresholds"], 4)

    def test_only_abstention_feasible(self):
        self.scores[:, 0] = 0.99
        result = calibrate_fpr20(self.scores, self.gold, self.groups)
        self.assertEqual(result["status"], "no_numeric_candidate_meets_limit_abstain")
        self.assertIsNone(result["chosen"]["threshold"])
        self.assertTrue(all(not s for s in policy_decisions(self.gold, self.scores, None).values()))

    def test_no_inner_negatives_is_unmeasurable(self):
        result = calibrate_fpr20(self.scores[:1], {"p": self.gold["p"]}, {"p": "g"})
        self.assertEqual(result["status"], "constraint_unmeasurable_no_inner_negatives_abstain")
        self.assertIsNone(result["chosen"]["no_hazard_false_alarm_rate"])
