from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kwater_lab import stage2_rare_base_probe as probe


def select(*, first):
    return [3.0 if option == first else 0.0 for option, *_ in probe.PICK]


class RareBaseProbeTests(unittest.TestCase):
    def test_combinations_and_independent_confirmation(self):
        water, enclosed, weather = probe.CODES
        self.assertEqual(probe.picked(select(first="H")), ("H", probe.CODES))
        self.assertEqual(probe.picked([0.] * 9), ("I", ()))
        self.assertEqual(probe.picked(select(first="A")), ("A", ()))
        strong = [probe.MARGIN, 0, -1]
        no = [2, 0, 0]
        unknown = [4, 0, 2]
        record = {"select_logits": select(first="H"), "gates": {
            water: [strong, strong], enclosed: [no, strong], weather: [strong, unknown],
        }}
        self.assertEqual(probe.outcome(record), ("H", set(probe.CODES), {water}))
        del record["gates"][weather]
        with self.assertRaisesRegex(ValueError, "exactly the stage-one"):
            probe.outcome(record)
        with self.assertRaisesRegex(ValueError, "choice logits"):
            probe.margin([float("nan"), 0, 1])

    def test_model_options_are_single_tokens_only_in_actual_prefix_context(self):
        class Tokenizer:
            def encode(self, string, *, add_special_tokens):
                return [99, 25, ord(string[-1])] if string[-2] == " " else [99, 25]

        processor = SimpleNamespace(tokenizer=Tokenizer(), apply_chat_template=Mock(return_value="<|image_pad|>"))
        queries = probe.render_queries(processor)
        self.assertEqual(len(queries), 7)
        self.assertEqual(queries["select"].tokens, tuple(map(ord, "ABCDEFGHI")))
        self.assertEqual(queries["confined_space:protection"].tokens, tuple(map(ord, "ABC")))
        self.assertFalse(processor.apply_chat_template.call_args_list[0].kwargs["tokenize"])
        with patch.object(processor.tokenizer, "encode", return_value=[19]):
            with self.assertRaisesRegex(ValueError, "single-token"):
                probe.render_queries(processor)

    def test_rare_negative_union_and_per_code_negative_have_separate_denominators(self):
        water, enclosed, weather = probe.CODES
        truth = [({water}, False), ({enclosed}, False), ({weather}, False), (set(), True), (set(), False)]
        decisions = [
            ("E", {water, enclosed}, {water, enclosed}),
            ("G", {enclosed, weather}, set()),
            ("I", set(), set()),
            ("E", {water, enclosed}, {water, enclosed}),
            ("A", set(), set()),
        ]
        report = probe.metrics(truth, decisions)
        self.assertEqual(report["nonrare_photos"], 2)
        self.assertEqual(report["normal_photos"], 1)
        self.assertEqual(report["rare_positive_photos_fully_recovered_in_candidates"], 2)
        self.assertEqual(report["rare_positive_photos_fully_recovered_in_final"], 1)
        self.assertEqual(report["nonrare_photos_with_any_final_rare_fp"], 1)
        self.assertEqual(report["by_code"][enclosed]["official_negatives"], 4)
        self.assertEqual(report["by_code"][enclosed]["final_fp"], 2)
        self.assertEqual(report["by_code"][enclosed]["final_fp_among_nonrare"], 1)

    def test_resumes_atomic_image_rows_with_every_forward_disabled(self):
        layer = SimpleNamespace(disable_adapters=False, merged=False)
        model = SimpleNamespace(eval=lambda: None)

        @contextmanager
        def disable():
            layer.disable_adapters = True
            try:
                yield
            finally:
                layer.disable_adapters = False

        def measure(actual_model, processor, queries, image_root, image, image_sha):
            self.assertTrue(layer.disable_adapters)
            self.assertIs(actual_model, model)
            self.assertIn((image, image_sha), inputs)
            return {"select_logits": select(first="A"), "gates": {},
                    "seconds": {"input": 0.0, "select": 0.1, "gates": 0.0}}

        model.disable_adapter = disable
        cuda = SimpleNamespace(device_count=lambda: 1, get_device_name=lambda: "RTX 3060",
                               is_bf16_supported=lambda: True, synchronize=lambda: None,
                               reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 10,
                               max_memory_reserved=lambda: 15)
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="fake"), __version__="fake",
                                inference_mode=nullcontext)
        msvcrt = SimpleNamespace(locking=lambda *args: None, LK_NBLCK=0)
        inputs = [("a.jpg", "sha-a"), ("b.jpg", "sha-b")]
        with TemporaryDirectory() as temp:
            out = Path(temp)
            with (patch.dict(sys.modules, torch=torch, msvcrt=msvcrt),
                  patch.multiple(probe, PHOTOS=2, PILOT_PHOTOS=1),
                  patch.object(probe, "version", return_value="test"),
                  patch.object(probe, "work_root", return_value=out),
                  patch.object(probe, "frozen_recipe", return_value=(inputs, {"recipe": "test"}, {})),
                  patch.object(probe, "resident_layers", return_value=(layer,)),
                  patch.object(probe, "measure_image", side_effect=measure) as measured):
                arguments = dict(base=Path("base"), adapter=Path("adapter"), manifest=Path("manifest"),
                                 image_root=out, out=out)
                probe.run_with_model(model, None, **arguments, pilot_only=True)
                self.assertEqual(measured.call_count, 1)
                self.assertEqual(len(list((out / "rows").glob("*.json"))), 1)
                probe.run_with_model(model, None, **arguments)
                self.assertEqual(measured.call_count, 2)
                self.assertEqual(len(list((out / "rows").glob("*.json"))), 2)
                self.assertFalse(layer.disable_adapters)
                cached = probe.row_file(out, 0)
                row = probe.read(cached)
                row["context_sha256"] = "stale"
                probe.write_json(cached, row)
                with self.assertRaisesRegex(ValueError, "frozen image, context or timing"):
                    probe.run_with_model(model, None, **arguments)


if __name__ == "__main__":
    unittest.main()
