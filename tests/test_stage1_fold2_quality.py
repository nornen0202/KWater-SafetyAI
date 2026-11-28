from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kwater_lab import stage1_fold2_quality as quality
from kwater_lab.fold2_8b_baseline import CONTROL, EVAL_SEGMENTS, write_json
from kwater_lab.fold2_v3_epoch1 import LABELS, LAWS, MANIFEST, MASK, split
from kwater_lab.fold4_same_verifier import digest, read
from kwater_lab.law_reranker_11code import read_selected_cases


class Stage1Fold2QualityTests(unittest.TestCase):
    def test_base_metadata_fingerprints_ignore_fp8_quantization_serialization(self):
        with TemporaryDirectory() as temp:
            work = Path(temp)
            adapter = work / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
            (adapter / "adapter_config.json").write_bytes(b"fixture config")
            prompt = work / "prompt.txt"
            prompt.write_text("fixture", encoding="utf-8")
            fingerprints = []
            for precision, revision in quality.REVISION.items():
                base = work / precision
                base.mkdir()
                config = {"model_type": "qwen3_vl", "text_config": {"n": 8}}
                if precision == "fp8":
                    config["quantization_config"] = {"quant_method": "fp8"}
                write_json(base / "config.json", config)
                files = ("tokenizer.json", "tokenizer_config.json", "chat_template.json", "vocab.json",
                         "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json")
                for name in files:
                    (base / name).write_text("{  \"fixture\" : true }" if precision == "fp8" else '{"fixture":true}',
                                             encoding="utf-8")
                write_json(base / "tokenizer.json", {"model": {"merges": ["fixture", "second"]}})
                if precision == "bf16":
                    (base / "merges.txt").write_text("#version: 0.2\nfixture\nsecond", encoding="utf-8")
                with patch.object(quality, "runtime_fingerprint", return_value="3"*64):
                    fingerprints.append(quality.generation_controls(
                        base=base, base_revision=revision, precision=precision, adapter=adapter,
                        train_configuration={"val_fold": 2}, prompt_path=prompt, max_pixels=393216,
                        formatter_path=prompt, engine_path=prompt, train_cases=[],
                    ))
            a, b = ({"controls": item, "rendered_prefix_sha256": "b"*64} for item in fingerprints)
            self.assertEqual(quality.audit_changed_factor(a, b, "precision"), ["base_revision", "precision"])
            b["controls"]["processor_assets_sha256"] = "c"*64
            with self.assertRaisesRegex(ValueError, "processor_assets_sha256"):
                quality.audit_changed_factor(a, b, "precision")

    @unittest.skipUnless(all(path.is_file() for path in (MANIFEST, MASK, LABELS, CONTROL / "epoch1-bf16-predictions.jsonl",
                                                        CONTROL / "epoch1-bf16-raw-cache.json")),
                         "Frozen private Fold 2 source unavailable")
    def test_real_4b_self_check_and_malformed_contrasts(self):
        train, outer = split()
        generation = read(CONTROL / "epoch1-bf16-raw-cache.json")
        original = CONTROL / "epoch1-bf16-predictions.jsonl"
        # Fixture-only controls exercise the scoring contract. Generation is the original 4B arm in both.
        common = {key: "a"*64 for key in quality.CONTROL_KEYS if key.endswith("_sha256")}
        cases = {row["sample_id"]: row for row in read_selected_cases(LABELS, {row["sample_id"] for row in train})}
        common.update(law_corpus_sha256=digest(LAWS),
                      train_case_bank_sha256=quality.sha_json([cases[row["sample_id"]] for row in train]), min_pixels=131072,
                      max_pixels=393216, max_new_tokens=620, do_sample=False, use_cache=True)
        with TemporaryDirectory(dir=Path("data"), prefix="fold2-quality-contract-") as temp:
            root = Path(temp)

            def fixture(precision, pixels=393216):
                arm = root / precision
                arm.mkdir(exist_ok=True)
                settings = dict(common, precision=precision, base_revision=quality.REVISION[precision],
                                max_pixels=pixels)
                cache_key = quality.receipt_cache_key(settings)
                segments = []
                for start, end, batch in EVAL_SEGMENTS:
                    rows = outer[start:end]
                    outputs = {row["sample_id"]: generation["outputs"][row["sample_id"]] for row in rows}
                    identity = dict(generation["identity"], samples=[[row["sample_id"], row["image"]] for row in rows],
                                    image_sha256={row["image"]: row["sha256"] for row in rows}, model_key=cache_key,
                                    processor_size={"shortest_edge": 131072, "longest_edge": pixels}, batch_size=batch)
                    path = arm / f"raw-{start}-{end}.json"
                    write_json(path, {"identity": identity, "outputs": outputs, "effective_batch_size": batch})
                    stats = {"photos": end-start, "initial_batch_size": batch, "final_effective_batch_size": batch,
                             "raw_json_parse_success": sum(row["parse_error"] is None and row["parsed"] is not None
                                                           for row in outputs.values()),
                             "strict_vlm_schema_success": sum(not row["vlm_schema_errors"] for row in outputs.values()),
                             "token_cap_hits": sum(row["token_cap_hit"] for row in outputs.values())}
                    segments.append({"start": start, "end": end, "raw_sha256": digest(path), "stats": stats})
                receipt = arm / "receipt.json"
                quality.write_generation_receipt(receipt, original, outer, segments, settings)
                return receipt

            bf16, fp8 = fixture("bf16"), fixture("fp8")
            result = quality.compare(original, original, bf16, fp8, "precision")
            self.assertEqual([(p["photos"], p["metrics"]["control"] == p["metrics"]["candidate"])
                              for p in result["panels"].values()], [(197, True), (55, True)])
            self.assertTrue(all(gate["point_threshold_met"] for gate in result["precision_point_gates"].values()))
            self.assertEqual(result["precision_loaded_adapter_evidence"], "unverified")
            self.assertEqual(result["runtime_parity_status"], "unmeasured_by_this_scorer")
            self.assertFalse(any(row["sample_id"] in str(result) for row in outer))
            with self.assertRaisesRegex(ValueError, "max_pixels"):
                quality.compare(original, original, bf16, fixture("fp8", 655360), "precision")

            partial = root / "other.jsonl"
            lines = original.read_bytes().splitlines(keepends=True)
            receipt = read(bf16)
            for altered in (lines[:-1], [lines[1], lines[0], *lines[2:]]):
                partial.write_bytes(b"".join(altered))
                receipt["prediction_sha256"] = digest(partial)
                write_json(bf16, receipt)
                with self.assertRaisesRegex(ValueError, "197 photos in canonical order"):
                    quality.validate_arm(partial, bf16, outer)


if __name__ == "__main__":
    unittest.main()
