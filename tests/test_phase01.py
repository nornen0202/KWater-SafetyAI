import importlib.util
import unittest

import numpy as np

from kwater_lab.phase01_data import balanced_folds, reserve, source_block


class SplitTests(unittest.TestCase):
    def test_whole_source_and_exposure_isolation(self):
        groups = [str(g) for g in range(12) for _ in range(3)]
        y = np.zeros((len(groups), 11))
        y[::4, 0] = 1
        y[1, 8] = 1
        folds = balanced_folds(groups, y, 5, 42)
        self.assertEqual(len(set(folds)), 5)
        np.testing.assert_array_equal(folds, balanced_folds(groups, y, 5, 42))
        for g in groups:
            self.assertEqual(len(set(folds[np.array(groups) == g])), 1)
        held = reserve(groups, set(groups) - {"0", "3"}, 9, 42)
        self.assertTrue(held.isdisjoint({"0", "3"}))
        self.assertEqual(sum(g in held for g in groups), 9)
        self.assertEqual(source_block("EXAMPLE933"), "stage3")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires training runtime")
class HeadTests(unittest.TestCase):
    def test_group_resampling_matches_reference_scorer(self):
        from kwater_lab.metrics import CODES, paired_bootstrap
        from kwater_lab.phase01_experiment import bootstrap_arms
        from kwater_lab.phase01_head import code_sets

        rng = np.random.default_rng(61126)
        groups = np.repeat(np.array(["a", "b", "c", "d", "e"]), 5)
        y, a, b = [rng.random((25, len(CODES))) < p for p in (0.15, 0.22, 0.18)]
        ids = [str(i) for i in range(len(y))]
        expected = paired_bootstrap(
            code_sets(ids, y),
            code_sets(ids, a),
            code_sets(ids, b),
            dict(zip(ids, groups)),
            resamples=130,
            seed=42,
        )
        got = bootstrap_arms(y, {"single_fixed": a, "multi_class": b}, groups, 130, 42)[
            "multi_class"
        ]
        np.testing.assert_allclose(
            got["percentile_95_ci"], expected["percentile_95_ci"], atol=1e-14
        )
        self.assertAlmostEqual(
            got["macro_f1_delta_b_minus_a"], expected["macro_f1_delta_b_minus_a"]
        )

    def test_calibration_zero_positive_and_top_three(self):
        from kwater_lab.phase01_head import calibrate, decide, fixed

        y = np.zeros((30, 11))
        y[:12, 0] = 1
        y[29, 1] = 1
        groups = np.repeat(np.arange(10), 3)
        z = np.zeros((30, 12))
        z[:, 2] = -100
        z[:12, 0] = 3
        pol = calibrate(z, y, groups)
        self.assertEqual(pol["common"]["thresholds"][0], pol["class"]["thresholds"][1])
        self.assertEqual(pol["common"]["thresholds"][0], pol["class"]["thresholds"][2])
        for policy in pol.values():
            output = decide(z, policy)
            self.assertLessEqual(int(output.sum(1).max()), 3)
            self.assertFalse(output[:, 2].any())
        empty = decide(np.zeros((1, 12)), dict(fixed(), scene_gate=0.9))
        self.assertFalse(empty.any())

    def test_evaluation_ignores_training_only_jpeg(self):
        import torch
        from kwater_lab.phase01_head import SpatialHead, logits

        x = torch.randn(3, 6, 17, 128)
        for multi in (False, True):
            net = SpatialHead(multi).eval()
            np.testing.assert_allclose(
                logits(net, x[:, :5]), logits(net, x), rtol=1e-5, atol=1e-6
            )
            if not multi:
                np.testing.assert_allclose(
                    logits(net, x[:, :1]), logits(net, x), rtol=1e-5, atol=1e-6
                )
