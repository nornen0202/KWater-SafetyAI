import unittest
from types import SimpleNamespace

import numpy as np

from kwater_lab.compositional_pilot import binary_metrics, support_order
from kwater_lab.depth_distillation import predict_auxiliary
from kwater_lab.evidence_factorial import marginal_utility, paired_targets
from kwater_lab.generated_law_comparison import raw_scale_cases, rerank
from kwater_lab.observation_pilot import ridge_projection


class ObservationPilotTest(unittest.TestCase):
    def test_depth_student_predicts_when_query_has_no_depth_entry(self):
        train = ['a', 'b', 'c', 'd']
        groups = {'a':'p0', 'b':'p0', 'c':'p1', 'd':'p1', 'q':'p2'}
        rng = np.random.default_rng(4)
        data = SimpleNamespace(clip={s: rng.normal(size=8) for s in groups}, linked={s:set() for s in groups})
        teacher = {s: rng.normal(size=(3, 32)) for s in train}
        predicted = predict_auxiliary(train, ['q'], data, teacher, groups, 'test')
        self.assertEqual(predicted.shape, (1, 4, 32))
        self.assertTrue(np.isfinite(predicted).all())

    def test_ridge_leave_one_out_matches_refitting_without_each_photo(self):
        rng = np.random.default_rng(15)
        x, y = rng.normal(size=(8, 6)), rng.normal(size=(8, 3))
        _, loo = ridge_projection(x, y, x)
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            observed, _ = ridge_projection(x[keep], y[keep], x[i:i+1])
            np.testing.assert_allclose(loo[i], observed[0], rtol=1e-10, atol=1e-10)

    def test_support_nested_and_distinct_from_validation_groups(self):
        groups = {str(i): str(i // 2) for i in range(12)}
        train = [sid for sid in groups if groups[sid] != '4']
        a = support_order(train, groups, 'target', '4', 11)
        self.assertEqual(a, support_order(reversed(train), groups, 'target', '4', 11))
        self.assertEqual(len({groups[s] for s in a}), 5)
        self.assertTrue(set(a).issubset(train))

    def test_average_precision_handles_ties_and_absent_positives(self):
        valid = [True, False, False]
        m = binary_metrics(valid, [0, 0, 0], [False, False, False], [False, True, True])
        self.assertAlmostEqual(m['average_precision'], 1/3)
        self.assertIsNone(binary_metrics([False] * 3, [1, 2, 3], valid, valid)['average_precision'])

    def test_text_shuffle_moves_only_training_sentences_of_same_code(self):
        rows = {'a': {'code1': '풀린 철사'}, 'b': {'code1': '회전축 노출'},
                'c': {'code2': '물이 고임'}, 'held': {'code1': None}}
        paired, sham = paired_targets(['a', 'b', 'c'], rows)
        np.testing.assert_array_equal(sham, paired[[1, 0, 2]])

    def test_no_supervised_utility_for_unseen_code(self):
        p = np.array([[.1,.2],[.9,.4]])
        y = np.array([[0.,0.],[1.,0.]])
        u = marginal_utility(p, y, y, p)
        self.assertTrue(np.isfinite(u[:,0]).all())
        self.assertTrue(np.isneginf(u[:,1]).all())

    def test_action_reranking_stays_within_retrieved_candidate_set(self):
        result = rerank(['A'], ['B'], ['A', 'B'], ['C','B'])
        self.assertEqual(result, ['B', 'A'])

    def test_completed_raw_law_arm_preserves_failed_rows_and_first_mapped_wording(self):
        official = SimpleNamespace(CODE_MAP={'lift': 'struck_by_lifting'}, validate_prediction=lambda p: [],
                                   clip=lambda text, limit: text.strip()[:limit])
        codes = [{'code': 'lift', 'evidence': ' observed ', 'prevention': ' act '},
                 {'code': 'lift', 'evidence': 'duplicate', 'prevention': 'duplicate'}]
        rows = [{'sample_id': 'a', 'prediction': {'hazards': codes}}, {'sample_id': 'b', 'prediction': {}}]
        lookup = {'a': 'first', 'b': 'second'}
        self.assertEqual(raw_scale_cases(rows, lookup, official),
                         [('first', {'struck_by_lifting': {'evidence': 'observed', 'prevention': 'act'}}), ('second', {})])
        with self.assertRaises(ValueError):
            raw_scale_cases(rows[::-1], lookup, official)
