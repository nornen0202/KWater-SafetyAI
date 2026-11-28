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
