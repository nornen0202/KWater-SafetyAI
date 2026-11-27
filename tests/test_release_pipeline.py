"""Regression tests for bundle code and local provenance gates; no model generation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from kwater_lab.fold0_official_resume import infer_cached
from kwater_lab.fold0_production_compare import generation_conditions
from kwater_lab.hybrid_pipeline import count_added_predictions
from kwater_lab.law_reranker_11code import export_compact_case_bank, Native11LawRetriever
from kwater_lab.metrics import CODES
from kwater_lab.package_submission import (
    PRODUCTION_VERIFY_MAX_NEW_TOKENS,
    _describe_training_and_calibration_scope,
    build_submission_package,
    render_entrypoint,
    validate_margin_gate_config,
)
from kwater_lab.verifier_cache import verify_with_cache
from kwater_lab.vlm_11code import infer_batched_vlm


ROOT = Path(__file__).resolve().parents[1]


class ReleasePipelineTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".runtime" / "temp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name)

    def test_emitted_code_turns_all_eleven_codes_into_kit_valid_rows(self):
        main = self.path / "main.py"
        main.write_text(render_entrypoint(), encoding="utf-8")
        runtime = runpy.run_path(str(main))
        self.assertEqual(runtime["CODES"], CODES)
        laws = [json.loads(line) for line in (ROOT / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if line]
        reranker = runtime["Native11LawRetriever"](laws, [])
        input_dir, output_dir = self.path / "input", self.path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        rows = []
        for index, code in enumerate(CODES):
            hazard = {"code": code, "evidence": "사진 속 작업구간 단부에 보호 난간이 없다.", "prevention": "단부에 고정 난간을 설치한다.",
                      "likelihood_score": 4, "severity_score": 4}
            self.assertIsNone(runtime["verified_hazard"]({"decision": "accept", "hazard": hazard}, "not_the_candidate"))
            verified = runtime["verified_hazard"]({"decision": "accept", "hazard": hazard}, code)
            self.assertIsNotNone(verified)
            parsed = {"status": "hazard", "scene_note": "작업 구간", "hazards": [verified]}
            rows.append(runtime["format_competition_record"](f"{index:032x}", parsed, reranker))
        rows.append(runtime["format_competition_record"](f"{len(CODES):032x}", None, reranker))
        self.assertEqual(rows[-1]["status"], "no_visible_hazard")
        self.assertEqual({code for row in rows for code in row["hazards"]}, set(CODES))
        (input_dir / "manifest.jsonl").write_text("".join(json.dumps({"image_id": r["image_id"]}) + "\n" for r in rows), encoding="utf-8")
        (output_dir / "submission.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        checker = ROOT / "data/local-kit/kit/check_output.py"
        checked = subprocess.run([sys.executable, "-X", "utf8", str(checker), str(input_dir), str(output_dir)],
                                 capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertIn("유효 결과 12/12", checked.stdout)

    def test_compact_case_bank_preserves_native_ranker(self):
        sample = [{"target": {"status": "hazard", "hazards": {"slip_trip": {
            "evidence": "통행로 바닥에 연장선이 놓여 있다.",
            "prevention": "연장선을 통행로 밖으로 정리한다.",
            "references": [{"kind": "law", "id": "OSH-3"}],
        }}}, "notes": {"scene_note": "작업 통행로"}}]
        output = self.path / "cases_11code.jsonl"
        export_compact_case_bank(sample, output)
        compact = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        laws = [json.loads(line) for line in (ROOT / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if line]
        full, bundled = Native11LawRetriever(laws, sample), Native11LawRetriever(laws, compact)
        self.assertTrue(compact)
        for c in compact:
            question = (c["code"], c["evidence"], c["prevention"], c["scene_note"])
            self.assertEqual(full.select_osh_references(*question), bundled.select_osh_references(*question))

    def test_release_rejects_an_incomplete_adapter_without_mutating_output(self):
        destination, archive = self.path / "stage", self.path / "future.zip"
        with self.assertRaisesRegex(FileNotFoundError, "Completed native 11-code"):
            build_submission_package(ROOT, self.path / "adapter", self.path / "specialist.pt", destination, archive)
        self.assertFalse(destination.exists())
        self.assertFalse(archive.exists())
        destination.mkdir()
        existing = destination / "keep.txt"
        existing.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            build_submission_package(ROOT, self.path / "adapter", self.path / "specialist.pt", destination, archive)
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep")

    def test_official_resume_retains_committed_batch(self):
        rows = [{"sample_id": "a"}, {"sample_id": "b"}]
        calls = []
        def stub_entry(fail):
            entry = SimpleNamespace(load_model=lambda: (None, None), parse_prediction=lambda value: value)
            def run(batch, _):
                entry.load_model()
                row = batch[0]
                calls.append(row["sample_id"])
                if fail and row["sample_id"] == "b":
                    raise RuntimeError("injected second-batch failure")
                predicted = entry.parse_prediction(f"raw-{row['sample_id']}")
                return [{**row, "prediction": predicted}]
            entry.infer_batched = run
            return entry

        with self.assertRaisesRegex(RuntimeError, "injected"):
            list(infer_cached(stub_entry(True), rows, self.path, self.path, 1))
        self.assertTrue((self.path / "000000.json").is_file())
        self.assertFalse((self.path / "000001.json").exists())
        timings = {}
        resumed = list(infer_cached(stub_entry(False), rows, self.path, self.path, 1, timings=timings))
        self.assertEqual([r["raw_text"] for r in resumed], ["raw-a", "raw-b"])
        self.assertEqual(calls, ["a", "b", "b"])
        self.assertEqual(timings["generated_photos"], 1)
        self.assertLessEqual(timings["model_load_seconds"], timings["generation_and_load_seconds"])

    def test_complete_vlm_cache_skips_model_loading_and_rejects_reordering(self):
        images = {"a.png": b"cached-image-a", "b.png": b"cached-image-b"}
        for name, data in images.items():
            (self.path / name).write_bytes(data)
        rows = [{"sample_id": "a", "image": "a.png"}, {"sample_id": "b", "image": "b.png"}]
        sizes = {"shortest_edge": 1, "longest_edge": 2}
        proc = SimpleNamespace(image_processor=SimpleNamespace(size=sizes), tokenizer=SimpleNamespace(),
                               apply_chat_template=lambda *args, **kwargs: "prefix")
        cache_path = self.path / "raw-cache.json"
        cache_path.write_text(json.dumps({
            "identity": {"version": 2, "image_root": str(self.path.resolve()),
                         "samples": [[r["sample_id"], r["image"]] for r in rows],
                         "image_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in images.items()},
                         "model_key": "test", "prefix_sha256": hashlib.sha256(b"prefix").hexdigest(),
                         "processor_size": sizes, "max_new_tokens": 8, "batch_size": 2},
            "outputs": {r["sample_id"]: {"sample_id": r["sample_id"], "parse_error": None, "parsed": {},
                                          "vlm_schema_errors": [], "token_cap_hit": False} for r in rows},
            "seconds": 0.3, "effective_batch_size": 2,
        }), encoding="utf-8")

        def no_weight_load():
            self.fail("Complete cache must not load model weights")

        def replay(items):
            return infer_batched_vlm(None, proc, items, self.path, "prompt", batch_size=2,
                                     max_new_tokens=8, cache_path=cache_path, cache_key="test",
                                     model_factory=no_weight_load)

        self.assertEqual(replay(rows)[1]["cache_reused_rows"], 2)
        with self.assertRaisesRegex(ValueError, "Inference settings differ"):
            replay(rows[::-1])

    def test_generation_conditions_bind_to_prediction_bytes(self):
        original, candidate = self.path / "official.jsonl", self.path / "candidate.jsonl"
        for path in (original, candidate):
            path.write_bytes(b'{}\n')
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        official_report = self.path / "official.json"
        official_report.write_text(json.dumps({"output_sha256": digest,
                                               "generation_settings": {"max_new_tokens": 900}}), encoding="utf-8")
        candidate_report = self.path / "candidate.json"
        candidate_report.write_text(json.dumps({"formatted_output_sha256": digest, "qlora_inference_weight_mode": "BF16 base",
                                                 "vlm_max_pixels": 524288, "inspect_max_new_tokens": 620, "vlm_batch_size": 1,
                                                 "finetuned_vlm_11code_standalone": {"generation": {"final_effective_batch_size": 1}}}),
                                    encoding="utf-8")
        generation = generation_conditions(official_report, candidate_report, original, candidate)
        self.assertEqual(generation["official"]["max_new_tokens"], 900)
        self.assertEqual(generation["candidate"]["max_new_tokens"], 620)
        candidate.write_bytes(b'{}\r\n')
        with self.assertRaisesRegex(ValueError, "different prediction file"):
            generation_conditions(official_report, candidate_report, original, candidate)

    def test_verifier_cache_resumes_without_repeating_a_candidate(self):
        proposals = [{"sample_id": sid, "code": "fall_height"} for sid in ("a", "b")]
        cache = self.path / "verification.json"
        calls = []

        def verify(batch):
            item = batch[0]
            calls.append(item["sample_id"])
            if calls == ["a", "b"]:
                raise RuntimeError("interrupted")
            return [{**item, "accepted": False, "raw_verify_text": "rejected"}]

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            verify_with_cache(proposals, verify, cache, {"model": "original"}, 1)
        results, stats = verify_with_cache(proposals, verify, cache, {"model": "original"}, 1)
        self.assertEqual([row["sample_id"] for row in results], ["a", "b"])
        self.assertEqual(calls, ["a", "b", "b"])
        self.assertEqual(stats["cache_reused_candidates"], 1)
        for items, identity in ((proposals[::-1], {"model": "original"}), (proposals, {"model": "changed"})):
            with self.assertRaisesRegex(ValueError, "settings or candidate order differ"):
                verify_with_cache(items, verify, cache, identity, 1)

    def test_added_hazard_metrics_keep_before_after_and_normal_alarm_separate(self):
        gold = {"a": {"fall_height"}, "b": set(), "c": {"other"}}
        before = {"a": set(), "b": set(), "c": {"other"}}
        after = {"a": {"fall_height"}, "b": {"slip_trip"}, "c": {"other", "fall_height"}}
        self.assertEqual(count_added_predictions(gold, before, after), {
            "new_true_positive_codes": 1, "new_false_positive_codes": 2, "new_false_alarm_photos": 1,
        })
        after["c"].remove("other")
        with self.assertRaisesRegex(ValueError, "removed a primary hazard prediction"):
            count_added_predictions(gold, before, after)

    def test_hybrid_entrypoint_enforces_margin_gate_single_candidate_and_verify_batch_two(self):
        for bad_gate in (
            None,
            {},
            {"beta": 0.0, "tau_when_hazard": 0.5, "tau_when_empty": 1.5},
            {"beta": True, "tau_when_hazard": 0.5, "tau_when_empty": 1.5},
            {"beta": float("nan"), "tau_when_hazard": 0.5, "tau_when_empty": 1.5},
        ):
            with self.assertRaises(ValueError):
                validate_margin_gate_config(bad_gate)
        valid_gate = validate_margin_gate_config({"beta": 0.6, "tau_when_hazard": -0.9, "tau_when_empty": 1.489})
        self.assertEqual(valid_gate["beta"], 0.6)

        src = render_entrypoint()
        self.assertEqual(PRODUCTION_VERIFY_MAX_NEW_TOKENS, 280)
        self.assertIn("VERIFY_MAX_NEW_TOKENS = 280", src)
        self.assertIn("max_candidates_per_photo=max_cands_per_photo", src)
        self.assertIn('cfg.get("max_candidates_per_photo", 1)', src)
        self.assertIn('cfg.get("verify_batch_size", 2)', src)
        self.assertIn('cfg.get("verify_max_new_tokens", VERIFY_MAX_NEW_TOKENS)', src)
        self.assertIn("max_new_tokens=verify_max_new_tokens", src)
        self.assertIn('"verify_max_new_tokens": verify_max_new_tokens if use_hybrid else 0', src)
        self.assertIn("score_candidates_1token_margin(\n                model, proc, candidates, prompt, verify_batch_sz\n            )", src)

        bundle_dir = self.path / "bundle_check"
        bundle_dir.mkdir()
        main_py = bundle_dir / "main.py"
        main_py.write_text(src, encoding="utf-8")
        (bundle_dir / "pipeline_config.json").write_text(
            json.dumps({"enable_hybrid_booster": True, "margin_gate": None}), encoding="utf-8"
        )
        runtime = runpy.run_path(str(main_py))
        self.assertEqual(runtime["VERIFY_MAX_NEW_TOKENS"], 280)
        old_argv = sys.argv[:]
        try:
            sys.argv = ["main.py", "--input-dir", str(self.path / "in"), "--output-dir", str(self.path / "out")]
            with self.assertRaisesRegex(ValueError, "calibrated margin_gate"):
                runtime["main"]()
        finally:
            sys.argv = old_argv

        # Verify scope & overlap audit against real Fold-4 artifacts (0 exact pair overlap, 3 photo / 2 group overlap, 65/65 inspect)
        f4_dir = ROOT / "data/v3-11code-fold4"
        if (f4_dir / "training_summary.json").is_file() and (f4_dir / "margin_gate_calibration.json").is_file():
            f4_sum = json.loads((f4_dir / "training_summary.json").read_text(encoding="utf-8"))
            dev_m = json.loads((ROOT / "data/clean-split-v1/split_manifest_dev.json").read_text(encoding="utf-8"))
            tr_ids = {r["sample_id"] for r in dev_m if r["split"] == "dev" and int(r["dev_fold"]) != 4}
            f4_gate = json.loads((f4_dir / "margin_gate_calibration.json").read_text(encoding="utf-8"))
            scope = _describe_training_and_calibration_scope(
                root=ROOT,
                adapter_dir=f4_dir / "adapter",
                summary=f4_sum,
                train_ids=tr_ids,
                resolved_prompt=ROOT / "kwater_lab/system_prompt_v3.txt",
                enable_hybrid=True,
                specialist_ckpt=ROOT / "data/clean-specialist-v1/specialist_5fold_ensemble.pt",
                oof_path=ROOT / "data/clean-specialist-v1/dev_oof_predictions.npz",
                gate_file=f4_dir / "margin_gate_calibration.json",
                margin_gate=f4_gate,
            )
            self.assertIn("not a separate inner calibration split", scope["specialist_calibration_method"])
            mg_audit = scope["margin_gate_exposure_audit"]
            self.assertTrue(mg_audit["exact_pair_disjoint_from_verify_supervision"])
            self.assertEqual(mg_audit["verify_supervision_exact_pair_overlap_count"], 0)
            self.assertEqual(mg_audit["verify_supervision_photo_overlap_count"], 3)
            self.assertEqual(mg_audit["verify_supervision_group_overlap_count"], 2)
            self.assertEqual(mg_audit["base_inspect_photo_overlap_count"], 65)
            for k in ("adapter_safetensors_sha256", "prompt_sha256", "margin_gate_calibration_sha256", "inner_margin_cal_pairs_sha256"):
                self.assertRegex(str(scope["provenance_sha256"][k]), r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
