"""Small CPU scenario checks for the nested anchored replay."""

import unittest

from kwater_lab.official_anchored_nested_replay import summarize


class NestedReplayTests(unittest.TestCase):
    def test_exact_cached_transfer_reduces_scenario_space(self):
        rows = [
            {"sample_id": "x", "codes": ["fall_height"], "group_id": "a"},
            {"sample_id": "y", "codes": [], "group_id": "b"},
        ]
        x = {"sample_id": "x", "image": "x.jpg", "code": "fall_height"}
        y = {"sample_id": "y", "image": "y.jpg", "code": "slip_trip"}
        known = {("x", "x.jpg", "fall_height"): (True, "prior_followup")}
        metrics = summarize(rows, {"x": frozenset(), "y": frozenset()}, [], [x, y], known,
                            {"x": {"fall_height"}, "y": set()})
        old = metrics["exhaustive_existing_anchor_cache_only"]
        new = metrics["exhaustive_all_exact_matched_caches"]
        self.assertEqual((old["scenarios"], new["scenarios"]), (4, 2))
        self.assertEqual(new["ranges"]["no_hazard_false_alarms"], [0, 1])
        self.assertEqual(new["ranges"]["severity4_complete_misses"], [0, 0])
        self.assertEqual(metrics["nested"]["confirmed_accepted_tp"], 1)
        self.assertEqual(metrics["nested"]["unknown_unannotated"], 1)


if __name__ == "__main__":
    unittest.main()
