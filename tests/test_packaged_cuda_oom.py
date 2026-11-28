"""CPU checks of the generated package entrypoint's CUDA OOM recovery."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kwater_lab.fold4_native_archive_replay import ReplayModel, ReplayProcessor, load_packaged_entrypoint
from kwater_lab.package_submission import render_entrypoint


class TypedOutOfMemory(RuntimeError):
    pass


class PackagedCudaOomTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(".runtime/temp")
        scratch.mkdir(parents=True, exist_ok=True)
        folder = TemporaryDirectory(dir=scratch)
        self.addCleanup(folder.cleanup)
        root = Path(folder.name).resolve()
        entrypoint = root / "main.py"
        entrypoint.write_text(render_entrypoint(), encoding="utf-8")
        self.runtime = load_packaged_entrypoint(entrypoint)
        self.runtime.torch.OutOfMemoryError = TypedOutOfMemory
        self.clear_cache = Mock()
        self.runtime.torch.cuda = SimpleNamespace(empty_cache=self.clear_cache)
        (root / "pipeline_config.json").write_text(json.dumps({
            "enable_hybrid_booster": False, "max_pixels_infer": 393216, "batch_size": 2,
        }), encoding="utf-8")
        (root / "system_prompt_11code.txt").write_text("fixture", encoding="utf-8")
        (root / "assets").mkdir()
        for name in ("laws", "cases_11code"):
            (root / "assets" / f"{name}.jsonl").touch()
        self.inputs, self.outputs = root / "input", root / "output"
        (self.inputs / "images").mkdir(parents=True)
        self.ids = ["1" * 32, "2" * 32]
        for iid in self.ids:
            (self.inputs / "images" / f"{iid}.png").touch()
        (self.inputs / "manifest.jsonl").write_text(
            "".join(json.dumps({"image_id": iid}) + "\n" for iid in self.ids), encoding="utf-8"
        )
        self.proc = ReplayProcessor(['{"hazards":[],"scene_note":"fixture"}'] * 2, 2, "fixture")

    def run_main(self, errors):
        model = ReplayModel()
        success = model.generate
        attempts = iter(errors)

        def generate(**kwargs):
            error = next(attempts, None)
            if error is not None:
                raise error
            return success(**kwargs)

        self.runtime.load_vlm = lambda *args, **kwargs: (SimpleNamespace(generate=generate), self.proc)
        with patch.object(sys, "argv", ["main.py", "--input-dir", str(self.inputs), "--output-dir", str(self.outputs)]):
            with patch.object(self.runtime.Image, "open", lambda path: self.runtime.Image.new("RGB", (1, 1))):
                with patch.dict(self.runtime.os.environ, {"KW_BATCH": "2"}):
                    self.runtime.main()

    def test_plain_cuda_oom_retries_exact_rows_at_half_batch(self):
        self.run_main([RuntimeError("CUDA error: out of memory\nCUDA errors may be asynchronously reported")])
        self.assertEqual(self.proc.observed_batches, [2, 1, 1])
        self.clear_cache.assert_called_once()
        result = (self.outputs / "submission.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(row)["image_id"] for row in result], self.ids)

    def test_singleton_oom_propagates_without_repeating(self):
        error = RuntimeError("CUDA error: out of memory")
        with self.assertRaises(RuntimeError) as raised:
            self.run_main([TypedOutOfMemory("CUDA OOM"), error])
        self.assertIs(raised.exception, error)
        self.assertEqual(self.proc.observed_batches, [2, 1])
        self.clear_cache.assert_called_once()

    def test_general_runtime_error_is_not_caught_as_oom(self):
        for message in ("CUDA error: illegal memory access", "CPU out of memory",
                        "error parsing CUDA out of memory report", "CUDA error: out of memory disabled"):
            with self.subTest(message=message):
                self.assertFalse(self.runtime.is_cuda_oom(RuntimeError(message)))
        error = RuntimeError("CUDA error: illegal memory access")
        with self.assertRaises(RuntimeError) as raised:
            self.run_main([error])
        self.assertIs(raised.exception, error)
        self.assertEqual(self.proc.observed_batches, [2])
        self.clear_cache.assert_not_called()
