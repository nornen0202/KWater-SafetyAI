from collections import Counter
import unittest

import numpy as np

from kwater_lab.metrics import CODES
from kwater_lab.sub78_clean_cal_specialist import fold_splits, proposals


class CleanCalV2SpecialistTests(unittest.TestCase):
    def test_connected_aliases_share_one_validation_fold(self):
        groups = ["alias-a", "b", "alias-a", "c", "d", "b", "e", "f"]
        folds = fold_splits(groups)
        seen = Counter()
        for train, test in folds:
            self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})
            seen.update(test.tolist())
        self.assertEqual(seen, Counter({i: 1 for i in range(len(groups))}))

    def test_fixed_empty_draft_proposals_respect_scene_and_top_one(self):
        rows = [{"sample_id": str(i), "image": f"{i}.png"} for i in range(3)]
        code_probs = np.full((3, len(CODES)), 0.49)
        code_probs[0, :2] = [0.80, 0.75]
        code_probs[1, 0] = 0.80
        code_probs[2, 0] = 0.61
        got = proposals(rows, code_probs, np.array([0.60, 0.57, 0.62]))
        self.assertEqual([(row["sample_id"], row["code"]) for row in got], [("0", CODES[0])])
