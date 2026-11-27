"""Coverage for the kit-only positional control's held-out comparison."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from kwater_lab.spatial_probe import _atomic_npz
from kwater_lab.ubuntu_position_control import FitOptions, KitPhoto, evaluate, split_photos


class UbuntuPositionControlTests(unittest.TestCase):
    def test_uninformative_features_report_a_completed_negative_comparison(self) -> None:
        """A measured null result retains its held-out denominators and both scores."""
        with TemporaryDirectory(dir=".runtime") as scratch:
            cache = Path(scratch)
            photos = [KitPhoto(f"photo-{index}", f"photo-{index}.jpg", f"{index:064x}") for index in range(10)]
            for photo in photos:
                _atomic_npz(cache / f"{photo.sha256}-encoder.npz", features=np.zeros((16, 4), dtype=np.float32),
                            source_sha256=np.asarray(photo.sha256), encoder_sha256=np.asarray("encoder"))
            report = evaluate(photos, cache, "encoder", FitOptions(7, 2, 0.03, 0.02, "cpu"))
            self.assertEqual(report["comparison_gate"], "NO_ADVANTAGE")
            self.assertEqual(report["true_quadrant_heldout_accuracy"], 0.25)
            self.assertEqual(report["train_only_shuffled_quadrant_heldout_accuracy"], 0.25)
            self.assertEqual((report["heldout_photo_count"], report["heldout_correlated_tile_count"]), (4, 64))

    def test_true_quadrants_outperform_train_only_shuffled_labels_on_heldout_photos(self) -> None:
        """A regression that loses native spatial labels fails on held-out tiles."""
        with TemporaryDirectory(dir=".runtime") as scratch:
            cache = Path(scratch)
            photos = [KitPhoto(f"photo-{index}", f"photo-{index}.jpg", f"{index:064x}") for index in range(10)]
            train, heldout = split_photos(photos, 7)
            self.assertEqual((len(train), len(heldout)), (6, 4))
            for photo in photos:
                feature = np.zeros((16, 4), dtype=np.float32)
                labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3])
                feature[np.arange(16), labels] = 1.0
                _atomic_npz(cache / f"{photo.sha256}-encoder.npz", features=feature,
                            source_sha256=np.asarray(photo.sha256), encoder_sha256=np.asarray("encoder"))
            report = evaluate(photos, cache, "encoder", FitOptions(7, 60, 0.03, 0.02, "cpu"))
            self.assertGreater(report["true_quadrant_heldout_accuracy"], report["train_only_shuffled_quadrant_heldout_accuracy"])
            self.assertEqual(report["learned_head_persisted"], False)


if __name__ == "__main__":
    unittest.main()
