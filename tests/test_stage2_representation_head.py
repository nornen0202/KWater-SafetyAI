from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from kwater_lab import stage2_representation_head as pilot


class RepresentationHeadTests(unittest.TestCase):
    def test_one_prefill_extracts_only_unpadded_visual_positions(self):
        hidden = torch.arange(24.).reshape(1, 6, 4)
        body = Mock(return_value=SimpleNamespace(last_hidden_state=hidden))
        inputs = {"input_ids": torch.tensor([[5, 99, 99, 7, 8, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
                  "image_grid_thw": torch.tensor([[1, 2, 4]]), "pixel_values": torch.ones(8, 2)}
        artifact = pilot.visual_prefill(body, inputs, 99, 2, feature_width=4)
        self.assertEqual(artifact["visual_tokens"], 2)
        self.assertTrue(torch.equal(artifact["features"], hidden[0, 1:3].bfloat16()))
        self.assertEqual(body.call_count, 1)
        self.assertIs(body.call_args.kwargs["use_cache"], False)
        poisoned = dict(inputs, attention_mask=torch.tensor([[1, 1, 0, 1, 1, 0]]))
        with self.assertRaisesRegex(ValueError, "unpadded visual-token mask"):
            pilot.visual_prefill(body, poisoned, 99, 2, feature_width=4)
        with self.assertRaisesRegex(ValueError, "inspection-only"):
            pilot.visual_prefill(body, dict(inputs, labels=inputs["input_ids"]), 99, 2, feature_width=4)
        self.assertEqual(body.call_count, 1)

    def test_padding_cannot_influence_head_and_training_loss_is_finite(self):
        torch.manual_seed(5)
        model = pilot.VisualTokenHead(hidden=4, width=8)
        tokens, mask = pilot.padded([torch.ones(2, 4), torch.ones(4, 4)])
        warped = tokens.clone()
        warped[~mask] = -10000
        first = model(tokens, mask)
        second = model(warped, mask)
        self.assertEqual(first.shape, (2, 12))
        self.assertTrue(torch.allclose(first, second, rtol=0, atol=1e-6))
        target = torch.zeros_like(first)
        target[1, 0] = target[1, -1] = 1
        loss = pilot.asymmetric_focal_bce(first, target, torch.ones(12), smooth=0.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))

    def test_calibration_is_joint_limited_and_keeps_absent_code_fixed(self):
        probabilities = np.full((15, len(pilot.CODES)+1), .05)
        normal = np.arange(15) < 8
        truth = np.zeros((15, len(pilot.COMMON)), dtype=bool)
        probabilities[-7:, -1] = .80
        for row, code in enumerate(pilot.COMMON[:-1], start=8):
            truth[row, row-8] = True
            probabilities[row, pilot.CODES.index(code)] = .76
        probabilities[:2, pilot.COMMON_INDEX[0]] = .97
        probabilities[:2, -1] = .40
        result = pilot.choose_cuts(probabilities, truth, normal)
        self.assertEqual(result["normal_false_alarm_maximum"], 1)
        self.assertLessEqual(result["normal_false_alarms"], 1)
        self.assertGreater(result["calibration_fixed8_macro_f1"], 0)
        self.assertEqual(result["thresholds"][-1], .50)
        self.assertEqual(result["unsupported_common_codes_fixed_at_0_50"], [pilot.COMMON[-1]])
        self.assertEqual(pilot.decisions(probabilities, result["thresholds"], result["scene_threshold"]).shape, (15, 8))

    def test_atomic_feature_roundtrip_and_head_byte_cap(self):
        with TemporaryDirectory() as temp:
            work = Path(temp)
            row = {"sample_id": "fixture", "sha256": "photo"}
            payload = {"sample_id": "fixture", "photo_sha256": "photo", "context_sha256": "nf4",
                       "features": torch.ones(2, pilot.HIDDEN, dtype=torch.bfloat16),
                       "image_grid_thw": [1, 2, 4], "visual_tokens": 2}
            path = pilot.feature_path(work, row)
            pilot.save_atomic(path, lambda stream: torch.save(payload, stream))
            actual = pilot.read_feature(work, row, "nf4")
            self.assertTrue(torch.equal(actual["features"], payload["features"]))
            with self.assertRaisesRegex(ValueError, "another|differs"):
                pilot.read_feature(work, row, "bf16")
            model = pilot.VisualTokenHead()
            head = work / "head.pt"
            pilot.save_atomic(head, lambda stream: torch.save({"context_sha256": "nf4", "state": model.state_dict()},
                                                              stream))
            self.assertLess(head.stat().st_size, 5_000_000)
            self.assertFalse(pilot.load_head(work, "nf4").training)

    def test_one_buffer_verification_and_prefetch_preserves_order(self):
        from PIL import Image

        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "photo.png"
            Image.new("RGB", (2, 2)).save(source)
            sha = pilot.sha256(source.read_bytes()).hexdigest()
            rows = [{"image": source.name, "sha256": sha, "sample_id": name} for name in ("one", "two")]
            with patch.object(pilot, "IMAGE_ROOT", root), patch.object(
                pilot, "decode_verified_photo", wraps=pilot.decode_verified_photo
            ) as decode:
                self.assertEqual([(row["sample_id"], image.size) for row, image in pilot.prefetched_photos(rows)],
                                 [("one", (2, 2)), ("two", (2, 2))])
                self.assertEqual(decode.call_count, 2)
                with self.assertRaisesRegex(ValueError, "image digest"):
                    pilot.decode_verified_photo(dict(rows[0], sha256="wrong"))


if __name__ == "__main__":
    unittest.main()
