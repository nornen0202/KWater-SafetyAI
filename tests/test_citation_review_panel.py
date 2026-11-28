import unittest

from kwater_lab.citation_site_generation import match_controls
from kwater_lab.citation_blind_judges import law_excerpt


def row(sid, group, code, corrected=False):
    return dict(sid=sid, group=group, code=code, current="OSH-319", candidate="OSH-87" if corrected else "OSH-319")


class CitationSamplingTests(unittest.TestCase):
    def test_controls_are_same_code_from_different_previously_unused_sites(self):
        rows = [row("a", 1, "machine", True), row("b", 1, "machine"),
                row("c", 2, "other"), row("d", 3, "machine"), row("e", 4, "fire", True)]
        matched, summary = match_controls(rows)
        self.assertEqual([r["sid"] for r in matched], ["a", "d", "e"])
        self.assertEqual(matched[-1]["kind"], "unmatched_change")
        self.assertEqual(summary["actual_changed_sites"], 2)
        self.assertEqual(summary["unmatched_changed_rows"], 1)

    def test_same_site_cannot_supply_two_matched_pairs(self):
        rows = [row("a", 1, "machine", True), row("b", 2, "machine", True),
                row("c", 3, "machine"), row("d", 3, "machine")]
        matched, summary = match_controls(rows)
        self.assertEqual(len({r["group"] for r in matched}), len(matched))
        self.assertEqual(summary["matched_pairs"], 1)

    def test_sham_excerpt_is_deterministic_and_marked_if_shortened(self):
        text = "제1조\n" + ("바닥에 있는 장애물을 제거한다.\n" * 90)
        self.assertEqual(law_excerpt("짧은 조문","바닥","정리"), "짧은 조문")
        self.assertEqual(law_excerpt(text,"바닥","정리"), law_excerpt(text,"바닥","정리"))
        self.assertTrue(law_excerpt(text,"바닥","정리").endswith("[발췌]"))

    def test_multiple_codes_at_one_control_site_do_not_block_two_pairs(self):
        rows = [row("f", 1, "fire", True), row("m", 2, "machine", True),
                row("x", 3, "fire"), row("x", 3, "machine"), row("y", 4, "fire")]
        cards, stats = match_controls(rows)
        self.assertEqual(stats["matched_pairs"], 2)
        self.assertEqual(len({card["group"] for card in cards}), 4)

    def test_controls_without_a_change_are_labelled_separately(self):
        rows = [row(str(i), i, "slip_trip") for i in range(5)]
        cards, stats = match_controls(rows)
        self.assertEqual(stats["matched_pairs"], 0)
        self.assertEqual(stats["unchanged_only"], 2)
        self.assertEqual({card["kind"] for card in cards}, {"unchanged_only"})
