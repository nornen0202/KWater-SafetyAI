import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from PIL import Image

from kwater_lab.metrics import CODES, decode_target
from kwater_lab.phase2_lora import observed_thinking_tokens, parse_codes
from kwater_lab.phase2_paths import file_stem
from kwater_lab.phase2_release import changes
from kwater_lab.phase2_exclusive_review import shuffled
from kwater_lab.phase2_proposer_strata import classify
from kwater_lab.phase2_review import apply_review, assign, interpret
from kwater_lab.phase2_sam3 import adapt_session, detect, restore_rope_views
from kwater_lab.submission2_output import make_row


class Phase2Tests(unittest.TestCase):
    def test_cached_writer_reuses_only_matching_photo_code_and_nohaz_status(self):
        from kwater_lab.phase2_cached_writing import compose

        a = dict(
            valid=True,
            final={"hazards": {"other": {}}},
            raw={
                "scene_note": "visible scene",
                "hazards": [
                    {
                        "code": "collision_contact",
                        "evidence": "photo observation",
                        "prevention": "matched action",
                    }
                ],
            },
        )

        def clip(v, n):
            return v[:n]

        fixed, actual, counts = compose(
            "f" * 32, ["other", "fall_height"], a, {"collision_contact": "other"}, clip
        )
        self.assertEqual(actual["hazards"]["other"]["evidence"], "photo observation")
        self.assertEqual(
            actual["hazards"]["fall_height"], fixed["hazards"]["fall_height"]
        )
        self.assertIn("fallback", counts["fall_height"])
        fixed, actual, counts = compose("f" * 32, [], a, {}, clip)
        self.assertEqual(actual, fixed)
        self.assertIn("fallback", counts["no_visible_hazard"])

    def test_low_commit_bf16_reader_returns_only_requested_bytes(self):
        import json
        import struct
        from kwater_lab.phase2_tensor_reader import BF16TensorFile

        with TemporaryDirectory() as temp:
            path = Path(temp) / "small.safetensors"
            header = json.dumps(
                {
                    "__metadata__": {"format": "pt"},
                    "first": {"shape": [2], "dtype": "BF16", "data_offsets": [0, 4]},
                    "last": {"shape": [1], "dtype": "BF16", "data_offsets": [4, 6]},
                }
            ).encode()
            prefix = struct.pack("<Q", len(header)) + header
            weights = struct.pack("<3H", 0x3F80, 0x4000, 0xC000)
            path.write_bytes(prefix + weights)
            with BF16TensorFile(path, framework="pt") as file:
                self.assertEqual(file.metadata(), {"format": "pt"})
                self.assertEqual(file.keys(), ["first", "last"])
                self.assertEqual(file.get_slice("first").get_shape(), [2])
                self.assertEqual(file.get_slice("last").get_dtype(), "BF16")
                self.assertEqual(file.read_bytes_for(file.tensors["last"]), weights[4:])
                self.assertEqual(
                    file.read_bytes_for(file.tensors["first"]), weights[:4]
                )
            path.write_bytes(prefix + weights[:-1])
            with self.assertRaisesRegex(ValueError, "payload size"):
                BF16TensorFile(path, framework="pt")

    def test_first_fit_waits_for_only_the_frozen_originals(self):
        from unittest.mock import patch
        from kwater_lab.phase2_io import write
        from tools import phase2_gpu_sequence as queue
        import sys
        from types import SimpleNamespace

        missing_process = SimpleNamespace(
            Process=Mock(side_effect=ProcessLookupError),
            NoSuchProcess=ProcessLookupError,
        )

        with (
            TemporaryDirectory() as temp,
            patch.object(queue, "RUN", Path(temp)),
            patch.dict(sys.modules, psutil=missing_process),
        ):
            write(
                queue.RUN / "plan.json",
                {"panel": ["EXAMPLE949"], "review": ["EXAMPLE949"]},
            )
            with self.assertRaisesRegex(RuntimeError, "before all frozen"):
                queue.wait_official(123)
            write(queue.RUN / "A" / (file_stem("EXAMPLE949") + ".json"), {"raw": {}})
            queue.wait_official(123)
            self.assertFalse(queue.source_core_complete())

    def test_verifier_requires_observation_and_correct_prevention_fields(self):
        import json
        from kwater_lab.phase2_verifier import parse, question

        for decision, prevention in (
            ("accept", "덮개를 고정한다."),
            ("reject", ""),
            ("unsure", ""),
        ):
            row = dict(
                decision=decision,
                evidence="개구부에 덮개가 보인다.",
                prevention=prevention,
            )
            self.assertEqual(parse(json.dumps(row)), row)
        for evidence, prevention in (("", ""), ("관찰 내용", "임의 조치")):
            with self.assertRaises(ValueError):
                parse(
                    json.dumps(
                        dict(
                            decision="reject", evidence=evidence, prevention=prevention
                        )
                    )
                )
        self.assertIn("fall_height", question("fall_height"))
        with self.assertRaises(ValueError):
            question("train_id")

    def test_agreement_uses_rank_one_and_respects_existing_three_code_limit(self):
        from kwater_lab.phase2_agreement import additions

        self.assertEqual(additions([], "other", [])["all_top1"], ["other"])
        self.assertEqual(additions([], "other", [])["confirmed"], [])
        self.assertEqual(additions([], "other", ["other"])["confirmed"], ["other"])
        for original in (["other"], ["fall_height", "fire_explosion", "slip_trip"]):
            self.assertFalse(any(additions(original, "other", ["other"]).values()))

    def test_photo_and_group_overlap_are_distinct_and_resampled_support_is_unique(self):
        from kwater_lab.phase2_lineage import overlap, support

        rows = {
            "a": {"group": "g", "codes": ["other"]},
            "b": {"group": "g", "codes": []},
            "c": {"group": "h", "codes": ["other"]},
        }
        self.assertEqual(
            overlap(["a", "a"], ["a", "b", "c"], rows),
            dict(
                same_photos=["a"],
                photos_from_training_groups=["a", "b"],
                same_groups=["g"],
            ),
        )
        self.assertEqual(
            support(["a", "a", "c"], rows)["positive_by_type"]["other"],
            dict(photos=2, groups=2),
        )

    def test_native_sam_retains_gpu_session_and_published_rope_views(self):
        class Tracking:
            def init_state(
                self,
                resource_path,
                offload_video_to_cpu=False,
                async_loading_frames=False,
            ):
                return resource_path, offload_video_to_cpu, async_loading_frames

        from types import SimpleNamespace

        predictor = SimpleNamespace(model=Tracking())
        self.assertTrue(adapt_session(predictor))
        self.assertEqual(
            predictor.model.init_state(
                resource_path="photo.jpg", offload_state_to_cpu=False
            ),
            ("photo.jpg", False, False),
        )
        with self.assertRaises(ValueError):
            predictor.model.init_state(
                resource_path="photo.jpg", offload_state_to_cpu=True
            )
        name = "detector.backbone.trunk.blocks.0.attn.freqs_cis"
        source = SimpleNamespace(is_complex=lambda: True, real=object(), imag=object())
        state = {name: source}
        self.assertEqual(restore_rope_views(state), [name + "_real", name + "_imag"])
        self.assertIs(state[name + "_real"], source.real)
        self.assertIs(state[name + "_imag"], source.imag)
        self.assertEqual(restore_rope_views(state), [])

    def test_exclusive_control_preserves_frequency_with_unavoidable_matches(self):
        ids = list("abcdef")
        codes = dict(zip(ids, ["fall_height"] * 4 + ["other", "slip_trip"]))
        result = shuffled(ids, codes, "dino")
        self.assertEqual(Counter(codes.values()), Counter(result.values()))
        self.assertEqual(sum(codes[s] == result[s] for s in ids), 2)
        self.assertEqual(shuffled(ids, codes, "dino"), result)
        a, b = (
            dict(called=True, candidate="other"),
            dict(called=False, candidate="other"),
        )
        self.assertEqual(classify(a, b), "dino_only")
        self.assertEqual(classify(b, a), "combined_only")

    def test_paired_review_keeps_original_on_rejection_and_accepts_only_novel_code(
        self,
    ):
        import json

        original = make_row("a" * 32, ["other"])
        for flag in ("reject", "unsure"):
            status, final = apply_review(
                original, {"text": json.dumps({"decision": flag})}, "slip_trip"
            )
            self.assertEqual(status, flag)
            self.assertEqual(final, original)
        yes = {
            "decision": "accept",
            "code": "slip_trip",
            "evidence": "통로 위에 전선이 가로놓여 있다.",
            "prevention": "전선을 통로 밖으로 옮긴다.",
        }
        status, final = apply_review(original, {"text": json.dumps(yes)}, "slip_trip")
        self.assertEqual(status, "accept")
        self.assertEqual(set(final["hazards"]), {"other", "slip_trip"})
        self.assertEqual(set(original["hazards"]), {"other"})

    def test_static_sam_sessions_reset_concepts_and_close_each_photo(self):
        calls = []

        def handle(req):
            calls.append(req)
            if req["type"] == "start_session":
                return {"session_id": req["resource_path"]}
            if req["type"] == "add_prompt":
                return {"frame_index": 0, "outputs": {"out_binary_masks": []}}

        model = Mock()
        model.handle_request.side_effect = handle
        with TemporaryDirectory() as d:
            for p in ("one.png", "two.png"):
                self.assertEqual(
                    detect(
                        model, "sam31-frame", Path(p), Image.new("RGB", (9, 6)), Path(d)
                    ),
                    [],
                )
        sessions = [r for r in calls if r["type"] == "start_session"]
        self.assertEqual([r["resource_path"] for r in sessions], ["one.png", "two.png"])
        for photo in ("one.png", "two.png"):
            commands = [r for r in calls if r.get("session_id") == photo]
            self.assertEqual(
                [r["type"] for r in commands],
                ["add_prompt", "reset_session"] * 6 + ["close_session"],
            )
            self.assertEqual(
                {r["frame_index"] for r in commands if r["type"] == "add_prompt"}, {0}
            )

    def test_frozen_formatter_counts_actual_severe_code_and_other(self):
        before = make_row("1" * 32, ["slip_trip"])
        after = make_row("1" * 32, ["other"])
        delta = changes(
            before, after, {"other": {"risk_level": {"severity": {"score": 4}}}}
        )
        self.assertEqual(delta["gained_tp"], ["other"])
        self.assertEqual(delta["removed_fp"], ["slip_trip"])
        self.assertEqual(delta["severity4_completely_missed"], [True, False])
        self.assertEqual(
            changes(before, after, {})["no_hazard_false_alarm"], [True, True]
        )

    def test_reasoning_counts_template_open_and_truncated_blocks(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [2, 3] if text == "<think>" else [3, 4]

        result = observed_thinking_tokens(Tokenizer(), [2, 3], [9, 9, 3, 4, 7])
        self.assertEqual(result["delimited_reasoning_tokens"], 2)
        self.assertFalse(result["reasoning_open_at_stop"])
        truncated = observed_thinking_tokens(Tokenizer(), [2, 3, 3, 4], [2, 3, 9])
        self.assertEqual(truncated["delimited_reasoning_tokens"], 1)
        self.assertTrue(truncated["reasoning_open_at_stop"])

    def test_real_official_identifiers_have_distinct_portable_filenames(self):
        stems = [file_stem(k) for k in ("EXAMPLE1029", "stage9%3AN106", "EXAMPLE491")]
        self.assertEqual(stems, ["stage9%3AN106", "stage9%253AN106", "EXAMPLE491"])

    def test_eleven_codes_and_true_negative(self):
        for c in CODES:
            self.assertEqual(parse_codes('{"codes":["' + c + '"]}'), [c])
            self.assertEqual(decode_target(make_row("1" * 32, [c]))[1], {c})
        self.assertEqual(parse_codes('{"codes":[]}'), [])
        for text in (
            '{"codes":["collision_contact"]}',
            '{"codes":["other","other"]}',
            '{"codes":[],"codes":["other"]}',
            '<think>1</think>{"codes":[]}',
        ):
            with self.assertRaises(ValueError):
                parse_codes(text)

    def test_rejection_preserves_original(self):
        for text in (
            '{"decision":"unsure"}',
            "not json",
            '{"decision":"reject"}',
            '{"decision":"accept","code":"other","evidence":"","prevention":"조치"}',
        ):
            self.assertIsNone(interpret(text, {"fall_height"}, "other")[1])
        yes = '{"decision":"accept","code":"other","evidence":"현장 근거","prevention":"조치"}'
        self.assertEqual(interpret(yes, {"fall_height"}, "other")[0], "accept")
        self.assertIsNone(interpret(yes, {"other"}, None)[1])
        self.assertIsNone(interpret(yes, set(), "fall_height")[1])

    def test_wrong_photo_hint_frequency(self):
        ids = list("abcd")
        candidate = dict(zip(ids, ["other", "fall_height", "other", "slip_trip"]))
        base = {s: set() for s in ids}
        rows = {s: {"group": i // 2} for i, s in enumerate(ids)}
        donor = assign(ids, candidate, base, rows)
        self.assertEqual(set(donor.values()), set(ids))
        self.assertEqual(
            Counter(candidate.values()), Counter(candidate[d] for d in donor.values())
        )
        self.assertTrue(
            all(rows[s]["group"] != rows[d]["group"] for s, d in donor.items())
        )
