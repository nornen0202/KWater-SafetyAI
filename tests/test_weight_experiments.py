import unittest

import numpy as np

from kwater_lab.adapter_scale_pilot import scale_default_adapter, sample
from kwater_lab.adapter_spectrum import singular_values
from kwater_lab.weight_pilot import edit_columns, neighbor_scores


class WeightExperimentTests(unittest.TestCase):
    def test_clip_edit_and_sham_have_equal_magnitude_only_on_targets(self):
        w = np.array([[1., 2., 3.], [2., 3., 5.], [-1., 0., 2.]])
        edited, sham, lengths = edit_columns(w, (0, 1))
        np.testing.assert_array_equal(edited[:, 2], w[:, 2])
        np.testing.assert_array_equal(sham[:, 2], w[:, 2])
        for target, other in ((0, 1), (1, 0)):
            np.testing.assert_allclose(np.linalg.norm(sham[:, target] - w[:, target]), lengths[target])
            self.assertAlmostEqual(np.dot(sham[:,target]-w[:,target], w[:,other]), 0.)
        negative = np.array([[1., -1.], [0., 0.]])
        np.testing.assert_array_equal(edit_columns(negative, (0, 1))[0], negative)

    def test_kernel_votes_accept_multilabel_and_have_only_training_support(self):
        x = np.eye(2)
        y = np.array([[1., 1., 0.], [1., 0., 0.]])
        np.testing.assert_allclose(neighbor_scores(x, y, x, k=1), y)
        self.assertLess(neighbor_scores(x, y, x)[1, 1], 0.0001)

    def test_rank_spectrum_matches_dense_update(self):
        rng = np.random.default_rng(21)
        a, b = rng.normal(size=(3, 9)), rng.normal(size=(8, 3))
        expected = np.linalg.svd(2 * b @ a, compute_uv=False)[:3]
        np.testing.assert_allclose(singular_values(a, b, 2), expected, rtol=1e-12)

    def test_lora_amplitude_restored_even_if_generation_fails(self):
        class Layer:
            lora_A = {"default": object()}
            scaling = {"default": 2., "separate": 5.}
        class Model:
            def modules(self):
                return [Layer()]
        with self.assertRaises(KeyError):
            with scale_default_adapter(Model(), 0.75) as n:
                self.assertEqual(n, 1)
                self.assertEqual(Layer.scaling, {"default": 1.5, "separate": 5.})
                raise KeyError
        self.assertEqual(Layer.scaling["default"], 2.)

    def test_scale_cohort_ignores_input_order(self):
        ids = ["a", "b", "c", "d", "e"]
        self.assertEqual(sample({"by_prefix": {"x": ids}}),
                         sample({"by_prefix": {"x": list(reversed(ids))}}))
