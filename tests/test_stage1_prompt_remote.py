import io
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kwater_lab import stage1_fold2_quality as quality
from kwater_lab import stage1_prompt_remote as remote


class PromptRemoteTests(unittest.TestCase):
    def test_eval_targets_and_unsafe_member_paths_are_rejected(self):
        photo = {"sample_id": "sample", "image": "images/photo.jpg", "sha256": "a" * 64}
        self.assertEqual(remote.image_members([photo]), {"data/train/images/photo.jpg": "a" * 64})
        for row in ({**photo, "codes": []}, {**photo, "image": "../escape.jpg"}, {**photo, "image": "G:\\x.jpg"}):
            with self.assertRaises(ValueError):
                remote.image_members([row])
        with self.assertRaises(ValueError):
            remote.image_members([photo, {**photo, "sample_id": "second"}])

    def test_archive_rejects_duplicate_extra_link_and_changed_members(self):
        good = {"source.py": b"pass\n"}
        digest = remote.sha256(good["source.py"]).hexdigest()
        with TemporaryDirectory() as temp:
            source = Path(temp) / "input.tar"
            with tarfile.open(source, "w") as archive:
                remote.add_tar_bytes(archive, "source.py", good["source.py"])
            remote.verify_tar(source, {"source.py": digest})
            for unexpected in ("source.py", "train/targets.jsonl", "../escape.py"):
                with tarfile.open(source, "w") as archive:
                    remote.add_tar_bytes(archive, "source.py", good["source.py"])
                    remote.add_tar_bytes(archive, unexpected, b"data")
                with self.assertRaises(ValueError):
                    remote.verify_tar(source, {"source.py": digest})
            with tarfile.open(source, "w") as archive:
                link = tarfile.TarInfo("source.py")
                link.type = tarfile.SYMTYPE
                link.linkname = "other.py"
                archive.addfile(link, io.BytesIO())
            with self.assertRaises(ValueError):
                remote.verify_tar(source, {"source.py": digest})
            with tarfile.open(source, "w") as archive:
                remote.add_tar_bytes(archive, "source.py", b"changed")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                remote.verify_tar(source, {"source.py": digest})

    def test_both_arms_share_model_processor_runtime_and_fixed_batches(self):
        segments = ((0, 2, 2), (2, 3, 2))
        rows = [{"sample_id": str(i)} for i in range(3)]
        prompts = {name: Mock() for name in ("v3", "v4")}
        for tag, prompt in prompts.items():
            prompt.read_text.return_value = tag
        plan = {"base": {}, "arms": {tag: {"controls": {"prompt_sha256": tag},
                                           "prefix_sha256": remote.sha256(tag.encode()).hexdigest()}
                                     for tag in prompts}}
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1, is_bf16_supported=lambda: True),
                                bfloat16="bf16")
        model = Mock()
        model.eval.return_value = model
        model.get_base_model.return_value.dtype = "bf16"
        model.parameters.return_value = []
        processor = Mock()
        processor.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"]

        def generate(actual_model, actual_processor, subset, image_root, system, *, batch_size, max_new_tokens,
                     cache_path, cache_key):
            self.assertIs(actual_model, model)
            self.assertIs(actual_processor, processor)
            self.assertEqual(batch_size, 2)
            self.assertEqual(max_new_tokens, 620)
            self.assertEqual(cache_key, remote.receipt_cache_key({"prompt_sha256": system, "runtime_sha256": "hash"}))
            cache_path.write_text("raw", encoding="utf-8")
            return subset, {"final_effective_batch_size": batch_size}

        engine = SimpleNamespace(INSPECT_USER_TEXT="image only", infer_batched_vlm=Mock(side_effect=generate),
                                 load_qwen_model=Mock(return_value=model), load_qwen_processor=Mock(return_value=processor))
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with (patch.dict(sys.modules, {"torch": torch, "kwater_lab.vlm_11code": engine}),
                  patch.object(remote, "read", side_effect=lambda path: plan if path == remote.PLAN else rows),
                  patch.multiple(remote, EVAL_SEGMENTS=segments, PROMPTS=prompts),
                  patch.object(remote, "verify_inputs"), patch.object(remote, "loaded_adapter_count", return_value=520),
                  patch.object(quality, "runtime_fingerprint", return_value="hash")):
                remote.infer(Path("base"), out)
            engine.load_qwen_model.assert_called_once_with(Path("base"), use_4bit=False, adapter_dir=remote.REMOTE_ADAPTER)
            engine.load_qwen_processor.assert_called_once_with(Path("base"), 393216)
            self.assertEqual(engine.infer_batched_vlm.call_count, 4)
            receipt = remote.read(out / remote.RETURN_RECEIPT)
            self.assertEqual(set(receipt["arms"]), {"v3", "v4"})
            self.assertEqual(receipt["loaded_adapter_tensors"], 520)

    def test_local_formatter_preserves_raw_and_common_receipt_contract(self):
        one = [{"sample_id": "photo", "image": "photo.jpg", "sha256": "a" * 64}]
        segment = ((0, 1, 1),)
        controls = {key: "b" * 64 for key in quality.CONTROL_KEYS if key.endswith("_sha256")}
        controls.update(precision="bf16", base_revision=remote.BASE_REVISION, min_pixels=remote.MIN_PIXELS,
                        max_pixels=remote.PIXELS, do_sample=False, use_cache=True, max_new_tokens=remote.MAX_NEW_TOKENS)
        runtime = controls["runtime_sha256"]
        profile = {"controls": {key: value for key, value in controls.items() if key != "runtime_sha256"},
                   "prefix_sha256": "c" * 64}
        identity = {"samples": [["photo", "photo.jpg"]], "image_sha256": {"photo.jpg": "a" * 64},
                    "processor_size": {"shortest_edge": remote.MIN_PIXELS, "longest_edge": remote.PIXELS},
                    "model_key": remote.receipt_cache_key(controls), "prefix_sha256": profile["prefix_sha256"],
                    "max_new_tokens": remote.MAX_NEW_TOKENS, "batch_size": 1}
        value = {"sample_id": "photo", "parsed": {}, "parse_error": None, "vlm_schema_errors": [], "token_cap_hit": False}
        stats = {"photos": 1, "initial_batch_size": 1, "final_effective_batch_size": 1, "raw_json_parse_success": 1,
                 "strict_vlm_schema_success": 1, "token_cap_hits": 0}
        formatter = SimpleNamespace(format_v3_record=lambda sid, obj, ranker: {
            "image_id": sid, "schema_version": "1.0", "status": "no_visible_hazard", "hazards": {},
            "scene_evidence": "synthetic row",
        })
        with TemporaryDirectory() as tmp:
            returned, scored = (Path(tmp) / name for name in ("returned", "scored"))
            (returned / "v3").mkdir(parents=True)
            path = returned / "v3/raw-0-1.json"
            remote.write_json(path, {"identity": identity, "outputs": {"photo": value}, "effective_batch_size": 1})
            state = {"runtime_sha256": runtime, "arms": {"v3": {"controls": controls, "segments": [
                {"start": 0, "end": 1, "raw_sha256": remote.digest(path), "stats": stats}
            ]}}}
            remote.write_json(returned / remote.RETURN_RECEIPT, state)
            with (patch.dict(sys.modules, {"kwater_lab.v3_turnaround": formatter}),
                  patch.object(remote, "EVAL_SEGMENTS", segment), patch.object(quality, "EVAL_SEGMENTS", segment)):
                pred, witness = remote.format_arm("v3", {"arms": {"v3": profile}}, returned, scored, state, one, None)
                decoded, proof = quality.validate_arm(pred, witness, one)
                self.assertEqual(list(decoded), ["photo"])
                self.assertTrue(proof["loaded_adapter_verified"])
                self.assertEqual((scored / "v3" / path.name).read_bytes(), path.read_bytes())
                path.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "bytes or boundaries"):
                    remote.format_arm("v3", {"arms": {"v3": profile}}, returned, scored, state, one, None)
