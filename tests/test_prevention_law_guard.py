import unittest

from kwater_lab.law_reranker_11code import Native11LawRetriever
from kwater_lab.prevention_law_guard import (
    RULES,
    action_support,
    build_active_records,
    check_unsafe_action_order,
    rerank_top1_by_action_support,
)


class PreventionLawGuardTests(unittest.TestCase):
    def setUp(self):
        self.active = {(rule.law, rule.record) for rule in RULES}

    def test_only_existing_top_three_can_move(self):
        text = "분진이 심한 곳 바닥에 물을 뿌린다."
        self.assertEqual(rerank_top1_by_action_support(text, ["OSH-52", "OSH-4-2", "OSH-32"], self.active).top1, "OSH-4-2")
        self.assertEqual(rerank_top1_by_action_support(text, ["OSH-52", "OSH-20", "OSH-32", "OSH-4-2"], self.active).top1, "OSH-52")
        self.assertIsNone(rerank_top1_by_action_support(text, [], self.active).top1)

    def test_specific_applicability_and_denial(self):
        self.assertNotIn("OSH-32", action_support("일반 작업장에서 방진마스크를 착용한다.", self.active))
        self.assertIn("OSH-32", action_support("선창의 심한 분진 하역 중 방진마스크를 착용한다.", self.active))
        self.assertNotIn("OSH-32", action_support("낙하 위험이 있지만 안전모는 착용하지 않는다.", self.active))
        self.assertNotIn("OSH-32", action_support("높이 1.2m 추락 구역에서 안전대를 착용한다.", self.active))
        self.assertIn("OSH-32", action_support("높이 2미터에서 추락할 수 있으므로 안전대를 착용한다.", self.active))
        self.assertNotIn("OSH-32", action_support("높이 2미터 미만에서 안전대를 착용한다.", self.active))
        self.assertNotIn("OSH-20", action_support("해체 작업장에 출입금지 표지를 붙인다.", self.active))
        self.assertIn("OSH-20", action_support("해체 작업장의 출입을 금지한다.", self.active))

    def test_structural_vehicle_and_electrical_boundaries(self):
        self.assertNotIn("OSH-52", action_support("비계 구조를 보강한다.", self.active))
        self.assertNotIn("OSH-52", action_support("비계의 구조 검토를 실시한다.", self.active))
        self.assertIn("OSH-52", action_support("붕괴 우려가 있는 비계의 구조 검토를 실시한다.", self.active))
        self.assertNotIn("OSH-200", action_support("지게차 운전구간에 유도자를 배치한다.", self.active))
        self.assertIn("OSH-200", action_support("굴착기 선회 구역에 유도자를 배치한다.", self.active))
        self.assertNotIn("OSH-301", action_support("이동전선에 절연덮개를 설치한다.", self.active))
        self.assertIn("OSH-301", action_support("충전부에 절연덮개를 설치한다.", self.active))
        self.assertNotIn("OSH-304", action_support("노출 충전부 수리 전 전원을 차단한다.", self.active))
        self.assertIn("OSH-319", action_support("노출 충전부 수리 전 전원을 차단한다.", self.active))
        self.assertIn("OSH-304", action_support("습윤 장소의 저압 전기기구에 누전차단기를 설치한다.", self.active))

    def test_platform_and_hot_work_use_article_conditions(self):
        self.assertIn("OSH-43", action_support("작업발판 끝에 고정식 난간을 설치한다.", self.active))
        self.assertNotIn("OSH-43", action_support("기계 옆에 고정식 난간을 설치한다.", self.active))
        action = "용접점 주변의 불연 덮개를 둔다."
        self.assertNotIn("OSH-241", action_support(action, self.active))
        self.assertIn("OSH-241", action_support(action, self.active, evidence="용접점 옆에 목재가 있다."))

    def test_exact_active_record_required(self):
        laws = [{"id": "273603:article:4-2", "article": "제4조의2", "status": "repealed", "text": "비산 방지"}]
        self.assertNotIn("OSH-4-2", action_support("분진에 물을 뿌린다.", build_active_records(laws)))
        laws[0]["status"] = "active"
        self.assertIn("OSH-4-2", action_support("분진에 물을 뿌린다.", build_active_records(laws)))

    def test_candidate_api_leaves_production_reference_gate_intact(self):
        laws = [
            {"id": f"law:article:{number}", "article": f"제{number}조", "status": "active",
             "record_type": "article", "title": title, "text": text}
            for number, title, text in (
                (43, "작업발판 안전난간", "작업발판 끝에 안전난간을 설치한다"),
                (52, "작업발판 안전성", "작업발판 구조검토 및 안전성 평가"),
            )
        ]
        ranker = Native11LawRetriever(laws)
        args = ("fall_height", "작업발판 구조검토", "작업발판 끝에 안전난간을 설치한다")
        published = ranker.select_osh_references(*args, max_refs=3)
        candidates = ranker.select_osh_references(*args, max_refs=3, ranked_candidates=True)
        self.assertEqual(len(published), 1)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[:1], published)

    def test_order_findings_distinguish_instruction_from_prohibition(self):
        cases = (
            ("전원을 차단하지 않고 노출 충전부를 만진다.", "unsafe_order"),
            ("충전부를 접촉한 후 전원을 차단한다.", "unsafe_order"),
            ("접속부와 전선을 절연한 뒤 전원을 차단한다.", "unsafe_order"),
            ("전원을 차단하지 않은 채 충전부에 접촉하지 않는다.", "explicit_prevention"),
            ("전원을 차단한 뒤 충전부를 수리한다.", "explicit_prevention"),
            ("노출 충전부를 절연물로 감싼다.", "order_unclear"),
            ("불티가 튀는 방향에 작업자를 배치한다.", "unsafe_order"),
            ("불티가 튀는 방향으로 작업자가 서도록 한다.", "unsafe_order"),
            ("불티가 튀는 방향에 작업자를 배치하지 않는다.", "explicit_prevention"),
            ("불티가 튀는 반대 방향에 작업자를 배치한다.", "explicit_prevention"),
            ("작업장에 소화기를 비치한다.", "not_applicable"),
        )
        for prevention, expected in cases:
            with self.subTest(prevention=prevention):
                self.assertEqual(check_unsafe_action_order(prevention).status, expected)


if __name__ == "__main__":
    unittest.main()
