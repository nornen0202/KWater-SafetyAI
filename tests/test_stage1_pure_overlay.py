import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from kwater_lab import fold2_8b_baseline as base
from kwater_lab import stage1_pure_overlay as overlay
from kwater_lab.fold4_same_verifier import digest, read


class PureOverlayTests(unittest.TestCase):
    @unittest.skipUnless(all(path.is_file() for path in (base.RUN / "train.tar", overlay.PURE_LOCAL, overlay.FREEZE_RECEIPT)),
                         "Verified private Fold 2 training source unavailable")
    def test_overlay_matches_archived_contract_and_has_no_photo_or_mask_members(self):
        with TemporaryDirectory(dir=Path("data"), prefix="pure-overlay-fixture-") as temp:
            out = Path(temp)
            report = out / "report.json"
            overlay.prepare(base.RUN, out, report)
            with tarfile.open(out / "pure-v3-overlay.tar") as archive:
                members = [base.CURRICULUM.as_posix(), base.RECIPE.as_posix(), overlay.OVERLAY_INFO.as_posix()]
                self.assertEqual(archive.getnames(), members)
                pure = json.load(archive.extractfile(members[0]))
                recipe = json.load(archive.extractfile(members[1]))
            original, _, mixed = overlay.archived_train_inputs(base.RUN)
            original = original["recipe"]
            self.assertEqual(recipe["eval_manifest_sha256"], original["eval_manifest_sha256"])
            self.assertEqual(recipe["train_manifest_sha256"], original["train_manifest_sha256"])
            self.assertEqual(recipe["training"]["optimizer"], original["training"]["optimizer"])
            self.assertEqual(base.experiment_identity(recipe)["arm"], overlay.ARM)
            self.assertEqual(len(pure), 1066)
            self.assertEqual(sum(old != new for old, new in zip(mixed, pure, strict=True)), 156)
            self.assertEqual(digest(out / "pure-v3-overlay.tar"), read(report)["overlay"]["sha256"])
            with self.assertRaisesRegex(ValueError, "changed an image"):
                overlay.make_recipe(original, mixed, [dict(pure[0], image="fixture"), *pure[1:]])

    def test_pure_resume_accepts_only_pure_configuration(self):
        with TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint-10"
            checkpoint.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors", "training_state.pt"):
                (checkpoint / name).write_bytes(b"fixture")
            optimizer = {"lr": .00014, "grad_accum": 16, "max_pixels": 393216, "seed": 42}
            recipe = {"arm": overlay.ARM, "training": {"optimizer": optimizer, "strict_inner_oof_sha256": "oof",
                                                       "curriculum_training_sha256": "pure", "prompt_sha256": "prompt"}}
            config = dict(optimizer, val_fold=2, epochs=2, curriculum_sha256="pure", prompt_sha256="prompt",
                          base_config_sha256=base.BASE_CONFIG_SHA, experiment=base.experiment_identity(recipe))
            records = [{"epoch": 1, "step": number, "total_steps": 134} for number in range(1, 11)]
            state = {"configuration": config, "global_step": 10, "next_epoch": 1, "next_offset": 160,
                     "records": records, "scheduler": {"last_epoch": 10}}
            fake = ModuleType("torch")
            fake.load = Mock(return_value=state)
            with patch.dict(sys.modules, {"torch": fake}):
                self.assertEqual(base.latest_checkpoint(Path(temp)), (10, checkpoint))
                self.assertIs(base.checkpoint_state(checkpoint, 10, recipe), state)
                config["experiment"] = dict(config["experiment"], arm="fold2_8b_matched_epoch1")
                with self.assertRaisesRegex(ValueError, "identity"):
                    base.checkpoint_state(checkpoint, 10, recipe)

    @unittest.skipUnless(all(path.is_file() for path in (base.RUN / "train.tar", overlay.PURE_LOCAL, overlay.PREP_REPORT)),
                         "Verified Pure-Inspect overlay provenance unavailable")
    def test_package_bridge_accepts_only_audited_overlay_identity(self):
        from kwater_lab import prepare_stage1_epoch1_package as package

        receipt, train, mixed = overlay.archived_train_inputs(base.RUN)
        pure = read(overlay.PURE_LOCAL)
        recipe = overlay.make_recipe(receipt["recipe"], mixed, pure)
        with TemporaryDirectory(dir=Path("data"), prefix="pure-package-fixture-") as temp:
            root = Path(temp)
            fit = root / "fit"
            checkpoint = fit / "epoch-1"
            checkpoint.mkdir(parents=True)
            for name in ("adapter_config.json", "adapter_model.safetensors", "training_state.pt"):
                (checkpoint / name).write_bytes(b"fixture")
            config = {key: value for key, value in recipe["training"]["optimizer"].items() if key != "epochs_scheduler"}
            config.update(epochs=2, val_fold=2, base_config_sha256=base.BASE_CONFIG_SHA,
                          prompt_sha256=recipe["training"]["prompt_sha256"],
                          curriculum_sha256=recipe["training"]["curriculum_training_sha256"],
                          experiment=base.experiment_identity(recipe))
            history = [{"epoch": 1}]
            base.write_json(fit / "history.json", history)
            state = {"configuration": config, "next_epoch": 2, "next_offset": 0, "global_step": 67,
                     "scheduler": {"last_epoch": 67}, "history": history, "records": [
                         {"epoch": 1, "step": step, "total_steps": 134} for step in range(1, 68)
                     ]}
            fake = ModuleType("torch")
            fake.load = Mock(return_value=state)
            with patch.dict(sys.modules, {"torch": fake}), patch.object(
                package, "load_scope", return_value=(train, [], [], base.TRAIN_MANIFEST, mixed, {})
            ):
                summary = package.prepare(fit, root / "candidate", "fold2", "pure-inspect", "v3")
                self.assertEqual(summary["configuration"], config)
                self.assertFalse((root / "candidate/adapter/training_state.pt").exists())
                config["experiment"] = dict(config["experiment"], strict_inner_oof_sha256="unexpected")
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    package.prepare(fit, root / "bad", "fold2", "pure-inspect", "v3")


if __name__ == "__main__":
    unittest.main()
