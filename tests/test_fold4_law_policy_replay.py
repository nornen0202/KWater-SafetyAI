"""Check the Fold 4 law replay denominators and paired first-citation transitions."""

import unittest

from kwater_lab.fold4_law_policy_replay import paired, summarize


class Fold4LawPolicyReplayTests(unittest.TestCase):
    def test_only_shared_gold_hazards_enter_common_panel(self):
        shared, only_eight, only_v3, outside = ((name, "fall_height") for name in ("a", "b", "c", "outside"))
        gold = {shared: {"OSH-1", "OSH-2"}, only_eight: {"OSH-2"}, only_v3: {"OSH-3"}, outside: {"OSH-4"}}
        eight = {shared: {"published": "OSH-1", "native": "OSH-2", "action": None},
                 only_eight: {"published": "OSH-2", "native": "OSH-2", "action": "OSH-2"},
                 outside: {"published": "OSH-4", "native": "OSH-4", "action": "OSH-4"}}
        v3 = {shared: {"published": "OSH-2", "native": "OSH-9", "action": "OSH-2"},
              only_v3: {"published": None, "native": "OSH-3", "action": None}}
        report = summarize({"a", "b", "c"}, gold, {"eight": eight, "v3": v3})
        self.assertEqual(report["common_annotated_law_hazards"], 1)
        self.assertEqual(report["common_hits"]["eight"], {"published": 1, "native": 1, "action": 0})
        self.assertEqual(report["common_hits"]["v3"], {"published": 1, "native": 0, "action": 1})
        self.assertEqual(report["v3_published_to_action_changed_first_all_emitted"], 0)
        self.assertEqual(report["v3_native_to_action_full_eligible"],
                         {"gained": 1, "lost": 1, "both_hit": 0, "neither_hit": 0, "changed_first": 2})

    def test_changed_article_can_keep_hit_status(self):
        report = paired({"row"}, {"row": {"OSH-1", "OSH-2"}}, {"row": "OSH-1"}, {"row": "OSH-2"})
        self.assertEqual(report, {"gained": 0, "lost": 0, "both_hit": 1, "neither_hit": 0, "changed_first": 1})


if __name__ == "__main__":
    unittest.main()
