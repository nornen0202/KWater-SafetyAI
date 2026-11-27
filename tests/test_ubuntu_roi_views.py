"""ROI selection and pixel-matched segmentation controls."""

import unittest

import numpy as np
import torch
from PIL import Image

from kwater_lab.ubuntu_roi_views import (
    Prediction,
    choose_detection,
    contours,
    overlay_pair,
)


class RoiViewsTests(unittest.TestCase):
    def test_largest_person_beats_more_confident_small_person_and_object(self) -> None:
        prediction = Prediction(
            boxes=torch.tensor([[0, 0, 80, 80], [0, 0, 20, 20], [0, 0, 100, 100]]),
            labels=torch.tensor([1, 1, 3]),
            scores=torch.tensor([0.51, 0.99, 1.0]),
            masks=torch.empty(0),
        )
        selected = choose_detection(prediction)
        self.assertEqual((selected.index, selected.reason), (0, "largest_person"))

    def test_low_confidence_person_uses_strongest_object_then_center(self) -> None:
        for scores, expected in [([0.49, 0.51, 0.9], 2), ([0.49, 0.1, 0.2], None)]:
            with self.subTest(scores=scores):
                prediction = Prediction(
                    boxes=torch.zeros(3, 4),
                    labels=torch.tensor([1, 3, 4]),
                    scores=torch.tensor(scores),
                    masks=torch.empty(0),
                )
                self.assertEqual(choose_detection(prediction).index, expected)

    def test_contour_and_sham_change_equal_pixels_preserving_all_other_rgb(
        self,
    ) -> None:
        binary = np.zeros((512, 512), dtype=np.uint8)
        binary[10:70, 30:100] = 255
        base = Image.fromarray(np.full((512, 512, 3), [255, 0, 255], dtype=np.uint8))

        real, sham = contours(Image.fromarray(binary))
        masked, placebo, color = overlay_pair(base, (real, sham))

        self.assertEqual(int(real.sum()), 520)
        self.assertEqual(int(sham.sum()), 520)
        self.assertEqual(int((real & sham).sum()), 0)
        for rendered, expected in [(masked, real), (placebo, sham)]:
            array = np.asarray(rendered)
            np.testing.assert_array_equal(
                np.any(array != np.asarray(base), axis=2), expected
            )
            np.testing.assert_array_equal(
                array[expected], np.broadcast_to(color, (520, 3))
            )


if __name__ == "__main__":
    unittest.main()
