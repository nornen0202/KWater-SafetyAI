import unittest

from kwater_lab.sub78_reject_audit import REJECT, accepted_equivalence


class BoundaryTokenizer:
    """Minimal deterministic fixture to test accounting without weights or downloads."""

    def encode(self, text, **_):
        return [text[:20], *text[20:]]

    def decode(self, ids, **_):
        return "".join(ids)


class RejectAuditTests(unittest.TestCase):
    def test_pattern_signature_distinguishes_a_field_value_from_other_text(self):
        self.assertTrue(REJECT.match('  { "decision":"reject"'))
        for text in ('{"decision":"rejecting"', '{"hazard":{"evidence":"reject"',
                     '{"decision":"reject', 'prefix {"decision":"reject"'):
            with self.subTest(text=text):
                self.assertIsNone(REJECT.match(text))

    def test_other_row_can_leave_dense_iterations_unchanged(self):
        reject = '{"decision":"reject","reason":"short"}'
        accepted = '{"decision":"accept","hazard":"much longer here and it keeps going"}'

        def parse(text, unused):
            return {"fixture": True} if 'accept' in text else None

        replay = accepted_equivalence([[("x", reject), ("x", accepted)]], BoundaryTokenizer(), parse, 2)
        self.assertLess(replay["reencoded_token_estimate"]["early_stop_nonpad"],
                        replay["reencoded_token_estimate"]["baseline_nonpad"])
        self.assertEqual(replay["dense_batch_estimate"]["shorter_batches"], 0)
        self.assertEqual(replay["rows"]["strict_object_changes_anchored"], 0)
