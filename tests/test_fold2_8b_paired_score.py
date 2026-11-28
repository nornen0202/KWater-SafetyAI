import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kwater_lab import fold2_8b_paired_score as paired
from kwater_lab.fold2_8b_baseline import BASE_REVISION, CONTROL, EVAL_SEGMENTS, RUN, write_json
from kwater_lab.fold2_v3_epoch1 import PROMPT
from kwater_lab.fold4_same_verifier import digest, read


class Fold2MatchedPairedScoreTests(unittest.TestCase):
    def test_arm_specific_union_differs_from_shared_diagnostic(self):
        ids = ["x", "y", "z"]
        gold = {"x": {"fall_height"}, "y": set(), "z": {"fall_height", "electrical_energy"}}
        first = {sid: gold[sid].copy() for sid in ids}
        second = {sid: gold[sid].copy() for sid in ids}
        second["y"].add("thermal_weather")
        panel = paired.score_panel(ids, dict(zip(ids, ids)), gold, first, second, gold)
        self.assertAlmostEqual(panel["paired_group_bootstrap"]["active_per_arm"]["macro_f1_delta_b_minus_a"], -1/3)
        self.assertEqual(panel["paired_group_bootstrap"]["supplemental_shared_active"]["delta_8b_minus_4b"], 0)
        self.assertEqual(panel["paired_group_bootstrap"]["fixed_8"]["delta_8b_minus_4b"], 0)
        self.assertEqual(panel["paired_outcomes"]["correct_normal"]["losses"], 1)
        self.assertEqual(panel["metrics"]["8b"]["multi_hazard"]["micro_recall"], 1)

    @unittest.skipUnless((RUN / "transfer-receipt.json").is_file() and (CONTROL / "epoch1-bf16-raw-cache.json").is_file(),
                         "Frozen Fold 2 private control files unavailable")
    def test_real_control_reformatted_with_frozen_batch_segments(self):
        with TemporaryDirectory(dir=RUN, prefix="paired-control-replay-") as temp:
            work = Path(temp)
            fit, evaluated = work / "fit", work / "eval"
            evaluated.mkdir()
            adapter = fit / "epoch-1/adapter_model.safetensors"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"pairing fixture")
            raw = read(CONTROL / "epoch1-bf16-raw-cache.json")
            records = [json.loads(line) for line in (CONTROL / "steps.jsonl").read_bytes().splitlines()]
            (fit / "steps.jsonl").write_bytes((CONTROL / "steps.jsonl").read_bytes())
            recipe = read(RUN / "transfer-receipt.json")["recipe"]
            cache_key = f"fold2-8b:{BASE_REVISION}:{digest(adapter)}:{digest(PROMPT)}:bf16:px393216"
            parts = []
            for start, end, batch in EVAL_SEGMENTS:
                samples = raw["identity"]["samples"][start:end]
                outputs = {sid: raw["outputs"][sid] for sid, _ in samples}
                identity = dict(raw["identity"], model_key=cache_key, samples=samples, batch_size=batch,
                                image_sha256={name: raw["identity"]["image_sha256"][name] for _, name in samples})
                path = evaluated / f"raw-{start}-{end}.json"
                write_json(path, dict(raw, identity=identity, outputs=outputs, effective_batch_size=batch))
                stats = {"photos": end-start,
                         "raw_json_parse_success": sum(r["parse_error"] is None and r["parsed"] is not None
                                                       for r in outputs.values()),
                         "strict_vlm_schema_success": sum(not row["vlm_schema_errors"] for row in outputs.values()),
                         "initial_batch_size": batch, "final_effective_batch_size": batch}
                parts.append({"start": start, "end": end, "raw_sha256": digest(path), "stats": stats})
            write_json(evaluated / "inference-receipt.json", {
                "adapter_sha256": digest(adapter), "cache_key": cache_key,
                "eval_order_sha256": recipe["eval_order_sha256"], "segments": parts,
            })
            result = work / "private-self-comparison.json"
            with patch.object(paired, "checkpoint_state", return_value={"records": records}):
                paired.score(RUN, fit, evaluated, result)
            panels = read(result)["panels"]
            for panel in panels.values():
                self.assertEqual(panel["metrics"]["4b"], panel["metrics"]["8b"])
                self.assertEqual(panel["paired_group_bootstrap"]["active_per_arm"]["macro_f1_delta_b_minus_a"], 0)
                self.assertEqual(panel["paired_group_bootstrap"]["active_per_arm"]["defined_resamples"], 2000)
            self.assertEqual([panel["photos"] for panel in panels.values()], [197, 55, 19])


if __name__ == "__main__":
    unittest.main()
