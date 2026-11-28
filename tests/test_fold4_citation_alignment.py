import unittest

from kwater_lab.fold4_citation_alignment import scope_veto


class ScopeVetoTests(unittest.TestCase):
    floor = {("OSH-3", "article:3:p:1")}
    fire = {("OSH-241", "article:241:p:2:i:3")}

    def test_roof_context_and_negation(self):
        query = ("바닥에 파편이 있다.", "파편을 제거한다.", "")
        def check(query, active):
            return scope_veto("slip_trip", query, "OSH-3", "OSH-45", active)
        self.assertEqual(check(query, self.floor), "OSH-3")
        self.assertEqual(check((*query[:2], "지붕 작업"), self.floor), "OSH-45")
        self.assertEqual(check((query[0], "파편을 제거하지 않는다.", ""), self.floor), "OSH-45")
        self.assertEqual(check(query, set()), "OSH-45")

    def test_mixed_electrical_and_hot_work(self):
        query = ("불티가 튄다.", "가연물을 제거한다.", "")
        def check(query, code):
            return scope_veto(code, query, "OSH-241", "OSH-319", self.fire)
        self.assertEqual(check(query, "fire_explosion"), "OSH-241")
        self.assertEqual(check((*query[:2], "노출된 단자에서 전기 작업"), "fire_explosion"), "OSH-319")
        self.assertEqual(check(query, "electrical_energy"), "OSH-319")
