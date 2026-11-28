import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from kwater_lab.fold2_v3_epoch1 import LABELS, MANIFEST, MASK, digest, read, save, score, split, train_one


class Fold2Epoch1Tests(unittest.TestCase):
    @unittest.skipUnless(MANIFEST.is_file() and MASK.is_file(), "private Fold 2 manifest and mask are unavailable")
    def test_frozen_split_and_mask(self):
        train, outer = split()
        self.assertEqual((len(train), len(outer)), (769, 197))
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in outer})
        mask = set(read(MASK)["prospective_mask_retained_ids"])
        selected = [r for r in outer if r["sample_id"] in mask]
        self.assertEqual((len(selected), len({r["group_id"] for r in selected})), (55, 34))

    def test_epoch_boundary_preserves_full_schedule_and_resume_parameter(self):
        base = Path.cwd().resolve()
        with TemporaryDirectory(dir=base, prefix="fold2-epoch1-train-test-") as temp:
            out = Path(temp).resolve()
            self.assertTrue(out.is_relative_to(base))
            checkpoint = out / "checkpoint-2"
            model_module = ModuleType("kwater_lab.vlm_11code")
            model_module.train_multimodal_qlora = Mock()
            epoch = out / "epoch-1"
            epoch.mkdir()
            (epoch / "training_state.pt").touch()
            witness = {"steps_per_epoch": 67, "strict_inner_oof_sha256": "fixture"}
            with patch.dict(sys.modules, {model_module.__name__: model_module}), \
                    patch("kwater_lab.fold2_v3_epoch1.freeze", return_value=([], witness)):
                train_one(out, checkpoint)
            arguments = model_module.train_multimodal_qlora.call_args.kwargs
            self.assertEqual((arguments["val_fold"], arguments["epochs"], arguments["grad_accum"],
                              arguments["stop_after_steps"], arguments["resume_from"]), (2, 2, 16, 67, checkpoint))

    @unittest.skipUnless(all(path.is_file() for path in (MANIFEST, MASK, LABELS)),
                         "private Fold 2 manifest, mask, or labels are unavailable")
    def test_score_requires_all_primary_rows_and_keeps_aggregate_only(self):
        _, outer = split()
        base = Path("data").resolve()
        with TemporaryDirectory(dir=base, prefix="fold2-epoch1-score-test-") as temp:
            out = Path(temp).resolve()
            self.assertTrue(out.is_relative_to(base))
            predictions = out / "epoch1-bf16-predictions.jsonl"
            receipt = out / "epoch1-bf16-inference-receipt.json"
            result = out / "result.json"
            rows = [json.dumps({"schema_version": "1.0", "image_id": row["sample_id"],
                                "status": "no_visible_hazard", "scene_evidence": "시험 응답", "hazards": {}})
                    for row in outer]
            save(out / "frozen-recipe.json", {"test_fixture": True})
            predictions.write_text("\n".join(rows) + "\n", encoding="utf-8")
            save(receipt, {"prediction_sha256": digest(predictions)})
            with contextlib.redirect_stdout(io.StringIO()):
                score(out, result)
                output = read(result)
                primary, retained = (output["panels"][name] for name in ("primary_197", "sensitivity_55"))
                self.assertEqual([(p["photos"], p["no_hazard_photos"], p["severity4"]["complete_misses"])
                                  for p in (primary, retained)], [(197, 67, 59), (55, 12, 15)])
                serialized = result.read_text(encoding="utf-8")
                self.assertFalse(any(row["sample_id"] in serialized for row in outer))
                predictions.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
                save(receipt, {"prediction_sha256": digest(predictions)})
                with self.assertRaisesRegex(ValueError, "completeness"):
                    score(out, result)
