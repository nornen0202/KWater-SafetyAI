"""Regression tests for bundle code and local provenance gates; no model generation."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from kwater_lab.fold0_official_resume import infer_cached
from kwater_lab.law_reranker_11code import export_compact_case_bank, Native11LawRetriever
from kwater_lab.metrics import CODES
from kwater_lab.package_submission import build_submission_package, render_entrypoint


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
            hazard = {"code": code, "evidence": "사진 속 작업구간 단부에 보호 난간이 없다.\n", "prevention": "단부에 고정 난간을 설치한다.",
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
        resumed = list(infer_cached(stub_entry(False), rows, self.path, self.path, 1))
        self.assertEqual([r["raw_text"] for r in resumed], ["raw-a", "raw-b"])
        self.assertEqual(calls, ["a", "b", "b"])


if __name__ == "__main__":
    unittest.main()
