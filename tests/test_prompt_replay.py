"""Exercise swapped image binding and partial official submissions with synthetic rows."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from kwater_lab.adapter_scale_pilot import write_json
from kwater_lab.prompt_pilot import ARMS, digest, paired_changes, parse_generation
from kwater_lab.prompt_replay import replay
from kwater_lab.visual_probe import GenerationRequestError


class PromptReplayTests(unittest.TestCase):
    def test_http_rejection_carries_typed_usage_but_is_not_a_generation(self):
        detail = {"type": "exceed_context_size_error", "n_prompt_tokens": 4200}
        error = GenerationRequestError(400, {"error": detail}, 0.5)
        self.assertEqual(error.details["n_prompt_tokens"], 4200)
        self.assertEqual(parse_generation("direct_8", "x", {"finish_reason": "context_overflow"}), (None, None))
        self.assertEqual(GenerationRequestError(503, {"error": "unavailable"}, 0.1).details, {})

    def test_pairwise_changes_exclude_invalid_without_discarding_complete_neighbors(self):
        gold = {"x": frozenset({"a"}), "y": frozenset()}
        data = {arm: {"x": (gold["x"], 0), "y": (gold["y"], 0)} for arm in ARMS}
        data["direct_8"]["x"] = (frozenset(), 0)
        data["direct_8"]["y"] = (None, 0)
        result = paired_changes(data, gold)
        self.assertEqual(result["direct_8/direct_11"],
                         {"pairwise_valid": 1, "code_set_changed": 1, "exact_set_gains": 1, "exact_set_losses": 0})
        self.assertEqual(result["direct_11/facts_11"]["pairwise_valid"], 2)

    def test_incomplete_generations_keep_full_manifest_and_correct_swapped_id(self):
        project = Path(__file__).resolve().parents[1]
        kit = project / "data/local-kit/kit"
        checker_python = project / ".runtime/baseline-venv/Scripts/python.exe"
        if not checker_python.exists() or not (kit / "check_output.py").exists():
            self.skipTest("Local organizer checker and checker interpreter required")
        with TemporaryDirectory(dir=project / ".runtime") as tmp:
            root = Path(tmp)
            local = root / "data/prompt-pilot"
            checker_dir = root / "data/local-kit/kit"
            checker_dir.mkdir(parents=True)
            for name in ("check_output.py", "prediction.schema.json"):
                shutil.copyfile(kit / name, checker_dir / name)
            anchors = [{"sid": s, "sha256": digest(s)} for s in ("first", "second")]
            design = {"arms": ARMS, "anchors": anchors}
            signature = digest(design)
            receipts = local / signature[:12]
            write_json(local / "design.json", design)
            content = json.dumps({"scene_observation": "작업 통로가 비어 있다.", "hazards": []})
            counts = {}
            for arm in ARMS:
                counts[arm] = {"valid": 0 if arm == "direct_8" else 1}
                for i in range(len(anchors)):
                    case = anchors[(i + int(arm == "swapped_11")) % len(anchors)]
                    finish = "stop" if i == 0 and arm != "direct_8" else "length"
                    write_json(receipts / f"{i}-{arm}.json", {
                        "signature": signature, "image_sha256": case["sha256"],
                        "generation": {"finish_reason": finish, "raw_output": content},
                    })
            write_json(root / "results/prompt_pilot.json", {"signature": signature, "conditions": counts})
            replay(root, checker_python)
            result = json.loads((root / "results/prompt_format_replay.json").read_text(encoding="utf-8"))
            for arm, stats in result["conditions"].items():
                self.assertEqual(stats["expected"], 2)
                self.assertEqual(stats["kit_valid"], counts[arm]["valid"])
            self.assertEqual(result["conditions"]["direct_8"]["checker_exit"], 20)
            path = receipts / "format-replay/swapped_11/out/submission.jsonl"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["image_id"], digest("second")[:32])
