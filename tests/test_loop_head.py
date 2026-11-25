import unittest

import numpy as np

from kwater_lab.loop_head import forward, initialize, loss_and_gradient


class LoopedHeadTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8)
        self.patches = self.rng.normal(size=(4, 5, 4))
        self.labels = self.rng.integers(0, 2, size=(4, 3)).astype(float)

    def parameters(self, count):
        initial = initialize(4, self.labels.sum(axis=0), 4, 17, count)
        return {k: v.astype(float) for k, v in initial.items()}

    def test_shared_gradient_accumulates_both_passes(self):
        shared = self.parameters(1)
        separate = self.parameters(2)
        a, ga = loss_and_gradient(shared, self.patches, self.labels, (0, 0))
        b, gb = loss_and_gradient(separate, self.patches, self.labels, (0, 1))
        self.assertAlmostEqual(a, b, places=12)
        for name, g in ga.items():
            expected = gb[name] + gb['1/' + name[2:]] if name.startswith('0/') else gb[name]
            np.testing.assert_allclose(g, expected, rtol=1e-10, atol=1e-12)

    def test_each_parameter_has_the_correct_directional_derivative(self):
        params = self.parameters(1)
        order = (0, 0)
        _, grads = loss_and_gradient(params, self.patches, self.labels, order)
        for name, value in params.items():
            direction = self.rng.normal(size=value.shape)
            direction /= np.linalg.norm(direction)
            losses = []
            for sign in (-1, 1):
                varied = {**params, name: value + sign * 1e-6 * direction}
                logits = forward(varied, self.patches, order)[0]
                losses.append(np.mean(np.logaddexp(0, logits) - self.labels * logits))
            observed = (losses[1] - losses[0]) / 2e-6
            self.assertAlmostEqual(float((grads[name] * direction).sum()), observed, places=7, msg=name)
