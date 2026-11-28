import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from kwater_lab import fold2_8b_baseline as baseline


def publish_fixture(path):
    path.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "training_state.pt"):
        (path / name).write_bytes(b"example")


class Fold2MatchedBaselineTests(unittest.TestCase):
    def test_latest_checkpoint_ignores_staging_and_rejects_post_epoch(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            self.assertEqual(baseline.latest_checkpoint(out), (0, None))
            publish_fixture(out / "checkpoint-2")
            publish_fixture(out / "checkpoint-10")
            (out / ".checkpoint-interrupted").mkdir()
            self.assertEqual(baseline.latest_checkpoint(out), (10, out / "checkpoint-10"))
            publish_fixture(out / "epoch-1")
            self.assertEqual(baseline.latest_checkpoint(out), (67, out / "epoch-1"))
            publish_fixture(out / "checkpoint-68")
            with self.assertRaisesRegex(ValueError, "outside the first-epoch schedule"):
                baseline.latest_checkpoint(out)

    def test_step10_resume_uses_original_schedule_and_stops_at_67(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            resume = out / "checkpoint-10"
            publish_fixture(resume)
            model_module = ModuleType("kwater_lab.vlm_11code")
            model_module.train_multimodal_qlora = Mock()
            curriculum = [{"kind": "inspect"}]
            manifest = [{"split": "dev"}]
            recipe = {"arm": "arm", "training": {"prompt_sha256": baseline.BASE_CONFIG_SHA,
                                                "strict_inner_oof_sha256": "fixture",
                                                "curriculum_training_sha256": baseline.sha_json(curriculum)},
                      "train_manifest_sha256": baseline.sha_json(manifest)}
            with patch.dict(sys.modules, {model_module.__name__: model_module}), patch.dict(baseline.os.environ), \
                    patch.object(baseline, "digest", return_value=baseline.BASE_CONFIG_SHA), \
                    patch.object(baseline, "read", side_effect=[recipe, curriculum, manifest]), \
                    patch.object(baseline, "checkpoint_state") as state_check:
                baseline.train(out, out)
            kwargs = model_module.train_multimodal_qlora.call_args.kwargs
            keys = ("val_fold", "epochs", "grad_accum", "checkpoint_steps", "resume_from", "stop_after_steps")
            self.assertEqual(tuple(kwargs[k] for k in keys), (2, 2, 16, 10, resume, 67))
            self.assertEqual([call.args[:2] for call in state_check.call_args_list],
                             [(resume, 10), (out / "epoch-1", 67)])

    def test_checkpoint_state_rejects_cursor_and_scheduler_mismatch(self):
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint-10"
            publish_fixture(checkpoint)
            optimizer = {"lr": .00014, "grad_accum": 16, "max_pixels": 393216, "seed": 42}
            recipe = {"arm": "arm", "training": {"optimizer": optimizer, "strict_inner_oof_sha256": "oof",
                                                "curriculum_training_sha256": "curriculum", "prompt_sha256": "prompt"}}
            configuration = dict(optimizer, val_fold=2, epochs=2, curriculum_sha256="curriculum",
                                 prompt_sha256="prompt", base_config_sha256=baseline.BASE_CONFIG_SHA,
                                 experiment=baseline.experiment_identity(recipe))
            state = {"configuration": configuration, "global_step": 10, "next_epoch": 1, "next_offset": 160,
                     "records": [{"epoch": 1, "step": step, "total_steps": 134} for step in range(1, 11)],
                     "scheduler": {"last_epoch": 10}}
            torch = ModuleType("torch")
            torch.load = Mock(return_value=state)
            with patch.dict(sys.modules, {"torch": torch}):
                self.assertIs(baseline.checkpoint_state(checkpoint, 10, recipe), state)
                state["next_offset"] = 176
                with self.assertRaisesRegex(ValueError, "cursor"):
                    baseline.checkpoint_state(checkpoint, 10, recipe)
                state["next_offset"] = 160
                state["scheduler"]["last_epoch"] = 134
                with self.assertRaisesRegex(ValueError, "scheduler"):
                    baseline.checkpoint_state(checkpoint, 10, recipe)

    def test_transfer_tar_has_exact_declared_members(self):
        with TemporaryDirectory() as temp:
            out = Path(temp) / "cohort.tar"
            members = {Path("data/member.json"): b'{"example": true}'}
            with patch.object(baseline, "add_images") as images:
                receipt = baseline.make_tar(out, members, [])
            self.assertEqual(receipt["sha256"], baseline.digest(out))
            self.assertFalse(out.with_suffix(".tar.part").exists())
            with baseline.tarfile.open(out) as archive:
                self.assertEqual(archive.getnames(), ["data/member.json"])
                self.assertEqual(json.load(archive.extractfile("data/member.json")), {"example": True})
            self.assertEqual(images.call_args.args[1], [])


if __name__ == "__main__":
    unittest.main()
