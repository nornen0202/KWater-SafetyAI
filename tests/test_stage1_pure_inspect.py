import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kwater_lab import stage1_pure_inspect as stage1


class PureInspectTests(unittest.TestCase):
    def setUp(self):
        self.train = [{"sample_id": "a", "image": "a.png"}, {"sample_id": "b", "image": "b.png"}]
        hazard = {"code": "fall_height", "likelihood_score": 3, "severity_score": 4,
                  "evidence": "Visible edge.", "prevention": "Install edge protection."}
        second = {**hazard, "code": "slip_trip"}
        negative = {"scene_note": "Clear.", "status": "no_visible_hazard", "hazards": []}
        positive = {"scene_note": "Two hazards.", "status": "hazard", "hazards": [hazard, second]}
        self.a = self.inspect("a", negative)
        self.b = self.inspect("b", positive)
        self.mixed = [self.a, self.verify("b", "verify_pos"), self.b,
                      {**self.b, "sample_id": "b__min_aug", "kind": "inspect_minority", "augment": True},
                      self.verify("a", "verify_neg_safe"), self.verify("b", "verify_neg_hard")]

    @staticmethod
    def inspect(sid, target):
        return {"sample_id": sid, "image": f"{sid}.png", "assistant_text": json.dumps(target),
                "user_text": stage1.INSPECT_USER_TEXT, "kind": "inspect", "augment": False}

    @staticmethod
    def verify(sid, kind):
        return {"sample_id": f"{sid}__ver_{kind}", "image": f"{sid}.png", "assistant_text": "{}",
                "user_text": "Verify candidate.", "kind": kind, "augment": False}

    def test_same_picture_replacements_keep_original_slots_and_targets(self):
        pure, targets = stage1.replace_verify_slots(self.mixed, self.train)
        self.assertEqual([self.mixed[i] for i in (0, 2, 3)], [pure[i] for i in (0, 2, 3)])
        self.assertEqual([(row["image"], row["augment"]) for row in self.mixed],
                         [(row["image"], row["augment"]) for row in pure])
        self.assertEqual([pure[i]["assistant_text"] for i in (1, 4, 5)],
                         [self.b["assistant_text"], self.a["assistant_text"], self.b["assistant_text"]])
        self.assertTrue(all(row["user_text"] == stage1.INSPECT_USER_TEXT for row in pure))
        self.assertEqual(sum(row["kind"] == stage1.REPLACEMENT_KIND for row in pure), 3)
        self.assertEqual(stage1.replace_verify_slots(self.mixed, self.train), (pure, targets))
        stats = stage1.summarize_curricula(self.mixed, pure, targets)
        self.assertEqual(stats["photo_repetitions_across_all_slots"], {2: 1, 4: 1})
        self.assertEqual(stats["photo_repetitions_in_replacement_slots"], {1: 1, 2: 1})
        self.assertEqual(stats["replacement_targets"]["codes_per_target"], {0: 1, 2: 2})

    def test_rejects_nontraining_photo_and_target_drift(self):
        for changed in ({**self.mixed[1], "image": "outer.png"},
                        {**self.mixed[1], "augment": True},
                        {**self.mixed[3], "assistant_text": self.a["assistant_text"]}):
            with self.subTest(kind=changed["kind"]):
                rows = [changed if row["sample_id"] == changed["sample_id"] else row for row in self.mixed]
                with self.assertRaises(ValueError):
                    stage1.replace_verify_slots(rows, self.train)

    def test_v4_only_changes_multi_hazard_and_slip_trip_rules(self):
        old, new = (path.read_text(encoding="utf-8").splitlines() for path in stage1.PROMPTS.values())
        changed = [(before, after) for before, after in zip(old, new, strict=True) if before != after]
        self.assertEqual(len(changed), 2)
        self.assertTrue(changed[0][0].startswith("1. "))
        self.assertIn("제2·제3 위험", changed[0][1])
        self.assertIn("배경 자재만으로 부수 slip_trip을 추가하지 않는다", changed[0][1])
        self.assertTrue(all(line.startswith("- slip_trip: ") for line in changed[1]))

    def test_trainer_uses_selected_prompt_curriculum_and_historical_epoch_schedule(self):
        pure, _ = stage1.replace_verify_slots(self.mixed, self.train)
        raw = {row["sample_id"]: row for row in self.train}
        manifest = Path("fixture-manifest.json")
        with TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "config.json").touch()
            (out / "epoch-1").mkdir()
            (out / "epoch-1/training_state.pt").touch()
            with patch.object(stage1, "prepare", return_value=(manifest, raw, {"pure-inspect": pure})), \
                    patch("kwater_lab.vlm_11code.train_multimodal_qlora", return_value={"optimizer_steps": 1}) as train, \
                    patch.object(stage1, "BASE_CONFIG_SHA", stage1.digest(out / "config.json")), \
                    patch.object(stage1, "RUNTIME", out / "runtime"), \
                    patch.object(stage1, "write_scoped_train_inputs") as scoped, \
                    patch.dict(os.environ):
                stage1.train_one("fold2", out, out, "pure-inspect", "v4", None)
            kwargs = train.call_args.kwargs
            self.assertEqual(kwargs["prompt_path"], stage1.PROMPTS["v4"])
            self.assertEqual(kwargs["manifest_path"], manifest)
            self.assertEqual((kwargs["val_fold"], kwargs["epochs"], kwargs["stop_after_steps"]), (2, 2, 1))
            self.assertEqual(kwargs["curriculum_builder"](), pure)
            scoped.assert_called_once_with(kwargs["root"], raw, pure)
            self.assertFalse(kwargs["root"].exists())

    def test_scoped_training_tree_contains_only_selected_labels_and_images(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            for name in ("a.png", "b.png", "outer.png"):
                (images / name).write_text(name, encoding="utf-8")
            raw = {row["sample_id"]: row for row in self.train}
            with patch.object(stage1, "IMAGES", images):
                stage1.write_scoped_train_inputs(root / "scoped", raw, self.mixed)
            scoped = root / "scoped/data/train"
            self.assertEqual({p.name for p in scoped.iterdir()}, {"train.jsonl", "a.png", "b.png"})
            self.assertEqual([json.loads(line) for line in (scoped / "train.jsonl").read_text().splitlines()],
                             list(raw.values()))
            self.assertEqual((scoped / "a.png").read_bytes(), (images / "a.png").read_bytes())

    def test_train_rejects_unpinned_model_before_preparing_inputs(self):
        with TemporaryDirectory() as temp, patch.object(stage1, "prepare") as prepare:
            base = Path(temp)
            (base / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned Qwen3-VL-8B"):
                stage1.train_one("fold2", base, base, "pure-inspect", "v4", None)
            prepare.assert_not_called()

    def test_inference_only_prompt_reuses_adapter_with_fixed_batch_boundaries(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            weights = out / "adapter_model.safetensors"
            weights.touch()
            image = out / "image.png"
            image.touch()
            laws = out / "laws.jsonl"
            laws.write_text("{}\n", encoding="utf-8")
            outer = [{"sample_id": f"example-{i}", "image": image.name, "sha256": stage1.digest(image)}
                     for i in range(197)]
            state = {"configuration": {"val_fold": 2, "base_config_sha256": stage1.BASE_CONFIG_SHA,
                                       "max_pixels": stage1.PIXELS, "curriculum_sha256": "synthetic",
                                       "prompt_sha256": stage1.digest(stage1.PROMPTS["v3"])},
                     "global_step": stage1.STEPS, "next_epoch": 2, "next_offset": 0}
            max_pixels = 655360
            controls = {"adapter_sha256": stage1.digest(weights), "max_pixels": max_pixels}

            def fake_infer(_model, _processor, rows, _images, system, **settings):
                self.assertEqual(system, stage1.PROMPTS["v4"].read_text(encoding="utf-8"))
                settings["cache_path"].write_bytes(b"fixture")
                return [{"parsed": None}] * len(rows), {"final_effective_batch_size": settings["batch_size"]}

            with patch.object(stage1, "require_8b_base"), patch.object(stage1, "under_data", return_value=out), \
                    patch.object(stage1, "IMAGES", out), patch.object(stage1.fold2, "LAWS", laws), \
                    patch.object(stage1.fold2, "split", return_value=([], outer)), \
                    patch.object(stage1, "read_selected_cases", return_value=[]), \
                    patch.object(stage1, "load_qwen_processor") as processor, patch("torch.load", return_value=state), \
                    patch("kwater_lab.stage1_fold2_quality.generation_controls", return_value=controls) as bind, \
                    patch("kwater_lab.v3_turnaround.ActionSupportedLawRetriever"), \
                    patch("kwater_lab.v3_turnaround.format_v3_record", return_value={}), \
                    patch("kwater_lab.vlm_11code.infer_batched_vlm", side_effect=fake_infer) as infer:
                stage1.infer_fold2(out, out, out, "v4", max_pixels)
                self.assertEqual([(len(c.args[2]), c.kwargs["batch_size"]) for c in infer.call_args_list],
                                 [(44, 4), (153, 2)])
                processor.assert_called_once_with(out, max_pixels)
                self.assertEqual(bind.call_args.kwargs["train_configuration"], state["configuration"])
                self.assertEqual(bind.call_args.kwargs["prompt_path"], stage1.PROMPTS["v4"])
                receipt = stage1.read(next(out.glob("train-*-px655360-*/receipt.json")))
                self.assertEqual((receipt["train_prompt"], receipt["infer_prompt"]), ("v3", "v4"))
                self.assertEqual(receipt["train_prompt_sha256"], state["configuration"]["prompt_sha256"])
                self.assertEqual(receipt["controls"], controls)
                self.assertTrue(all(call.kwargs["cache_key"] == receipt["cache_key"] for call in infer.call_args_list))

                infer.side_effect = lambda *_args, **_kwargs: ([{"parsed": None}] * 44,
                                                              {"final_effective_batch_size": 2})
                with self.assertRaisesRegex(RuntimeError, "matched batch boundaries"):
                    stage1.infer_fold2(out, out, out, "v4", max_pixels)


if __name__ == "__main__":
    unittest.main()
