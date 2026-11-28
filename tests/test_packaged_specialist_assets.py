import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from safetensors.numpy import save_file

from kwater_lab import package_submission as package


RESEARCH_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TOWER_REVISIONS = {
    "https://huggingface.co/google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    "https://huggingface.co/facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
}


class PackagedSpecialistAssetsTests(unittest.TestCase):
    def setUp(self):
        runtime = Path(".runtime/temp").resolve()
        runtime.mkdir(parents=True, exist_ok=True)
        scratch = TemporaryDirectory(dir=runtime)
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.base = self.root / "base"
        self.base.mkdir()
        for name in ("chat_template.json", "config.json", "generation_config.json", "merges.txt",
                     "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                     "video_preprocessor_config.json", "vocab.json"):
            (self.base / name).write_text("{}", encoding="utf-8")
        self.base_sha = package.digest(self.base / "config.json")
        weight_map = {}
        for number in range(1, 5):
            filename = f"model-{number:05d}-of-00004.safetensors"
            key = f"fixture.{number}"
            save_file({key: np.ones((1,), dtype=np.float32)}, self.base / filename)
            weight_map[key] = filename
        self.write_json(self.base / "model.safetensors.index.json", {"weight_map": weight_map})
        self.write_json(self.base / "source.json", {
            "repo": "Qwen/Qwen3-VL-8B-Instruct", "revision": RESEARCH_REVISION,
            "files": [{"rfilename": p.name, "size": p.stat().st_size} for p in self.base.iterdir()],
        })
        self.prompt = self.root / "system_prompt_v3.txt"
        self.prompt.write_text("fixture prompt\n", encoding="utf-8")
        baseline = self.root / "data/baseline"
        (baseline / "wheelhouse").mkdir(parents=True)
        (baseline / "assets").mkdir()
        (baseline / "requirements.lock").touch()
        (baseline / "assets/laws.jsonl").touch()
        for scope in ("fold2", "full1045"):
            path = Path(f"results/stage1-pure-inspect-{scope}.json")
            self.write_json(self.root / path, json.loads(path.read_bytes()))

    @staticmethod
    def write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def build(self, name, *, scope, curriculum, experimental):
        if scope == "legacy":
            val_fold, count, examples = 4, 791, 1092
        else:
            val_fold, count, examples = (2, 769, 1066) if scope == "fold2" else (-1, 1045, 1345)
        roster = [{"sample_id": f"fixture-{index}", "split": "dev", "dev_fold": 0} for index in range(count)]
        adapter = self.root / name / "adapter"
        adapter.mkdir(parents=True)
        manifest = (adapter.parent / "manifest_all_1045_train.json" if scope == "full1045"
                    else self.root / "data/clean-split-v1/split_manifest_dev.json")
        self.write_json(manifest, roster)
        labels = self.root / "data/train/train.jsonl"
        labels.parent.mkdir(parents=True, exist_ok=True)
        labels.write_text("".join(json.dumps({"sample_id": row["sample_id"],
                                               "target": {"status": "no_visible_hazard"}}) + "\n" for row in roster),
                          encoding="utf-8")
        self.write_json(adapter / "adapter_config.json", {"peft_type": "LORA"})
        weights = adapter / "adapter_model.safetensors"
        save_file({key: np.ones((1,), dtype=np.float32) for key in (
            "language_model.layers.fixture", "visual.merger.fixture", "visual.deepstack_merger_list.fixture",
        )}, weights)
        audit = (json.loads((self.root / f"results/stage1-pure-inspect-{scope}.json").read_bytes())
                 if scope != "legacy" else {})
        recipe_hash = audit.get(f"{curriculum}_curriculum_sha256", "f" * 64)
        steps = (examples + 15) // 16
        self.write_json(adapter.parent / "training_summary.json", {
            "version": "vlm-11code-multimodal-qlora-v2-epoch-boundary", "val_fold": val_fold,
            "train_photos": count, "curriculum_examples": examples, "grad_accum": 16,
            "configured_epochs": 2, "completed_epochs": 1, "optimizer_steps": steps,
            "scheduled_total_steps": 2 * steps, "history": [{"epoch": 1}],
            "resume_cursor": {"next_epoch": 2, "next_offset": 0, "global_step": steps},
            "adapter_safetensors_sha256": package.digest(weights),
            "configuration": {"epochs": 2, "base_config_sha256": self.base_sha,
                              "prompt_sha256": package.digest(self.prompt), "curriculum_sha256": recipe_hash,
                              "experiment": {"base_revision": RESEARCH_REVISION, "arm": f"stage1_{scope}_mixed_v3"}},
        })
        archive = self.root / f"{name}.zip"
        with patch.object(package, "RESEARCH_8B_BASE_CONFIG_SHA256", self.base_sha):
            package.build_submission_package(self.root, adapter, self.root / "unused-specialist.pt",
                                             self.root / f"stage-{name}", archive, enable_hybrid=False,
                                             prompt_path=self.prompt, research_8b_base_dir=self.base,
                                             allow_experimental_8b_adapter=experimental)
        with ZipFile(archive) as zipped:
            self.assertFalse(any(item.startswith("specialist/") for item in zipped.namelist()))
            return json.loads(zipped.read("assets.json"))["assets"]

    def test_verified_mixed_and_pure_curricula_declare_training_only_towers(self):
        for scope, curriculum, step in (("fold2", "mixed", "검증 예제 선택"),
                                        ("full1045", "pure_inspect", "Pure-Inspect 슬롯 구성")):
            with self.subTest(scope=scope, curriculum=curriculum):
                rows = self.build(scope, scope=scope, curriculum=curriculum, experimental=True)
                towers = {row["source"]: row for row in rows if row["source"] in TOWER_REVISIONS}
                self.assertEqual({url: row["revision"] for url, row in towers.items()}, TOWER_REVISIONS)
                for row in towers.values():
                    self.assertEqual((row["kind"], row["license"]), ("model", "Apache-2.0"))
                    self.assertIn("학습 전용", row["use"])
                    self.assertIn(step, row["use"])

    def test_unknown_experimental_lineage_and_measured_solo_default_remain_four_assets(self):
        self.assertEqual(len(self.build("unknown", scope="fold2", curriculum="unknown", experimental=True)), 4)
        self.assertEqual(len(self.build("historical", scope="legacy", curriculum="mixed", experimental=False)), 4)


if __name__ == "__main__":
    unittest.main()
