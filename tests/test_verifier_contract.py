"""CPU-only regressions for the learned verifier contract and bundled acceptance."""

import copy
import json
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from kwater_lab.hybrid_pipeline import merge_verified_candidates, select_booster_candidates
from kwater_lab.metrics import CODES
from kwater_lab.package_submission import render_entrypoint
from kwater_lab.vlm_11code import forward_and_weighted_assistant_loss, parse_verified_hazard, verify_user_text


CODE = "fall_height"
HAZARD = {"code": CODE, "likelihood_score": 4, "severity_score": 4,
          "evidence": "발판 단부에 보호 난간이 없다.", "prevention": "단부에 보호 난간을 설치한다."}


class VerifierContractTests(unittest.TestCase):
    def test_complete_response_matrix_matches_bundled_entrypoint(self):
        runtime = {"__name__": "verifier_contract_test", "__file__": __file__}
        exec(render_entrypoint(), runtime)
        self.assertEqual(runtime["verify_user_text"](CODE), verify_user_text(CODE))
        affirmative = {"decision": "accept", "hazard": HAZARD}
        text = json.dumps(affirmative, ensure_ascii=False)
        rejected = {
            "explicit reject": '{"decision":"reject","reason":"위험이 확인되지 않는다."}',
            "different hazard code": json.dumps({"decision": "accept", "hazard": {**HAZARD, "code": "other"}}),
            "inspection object": json.dumps({"status": "hazard", "hazards": [HAZARD]}),
            "contradictory status": json.dumps({**affirmative, "status": "no_visible_hazard"}),
            "contradictory reason": json.dumps({**affirmative, "reason": "위험이 확인되지 않는다."}),
            "negated risk": json.dumps({"decision": "accept", "hazard": {**HAZARD, "evidence": "사진에서 추락 위험이 확인되지 않는다."}}),
            "no visible risk evidence": json.dumps({"decision": "accept", "hazard": {**HAZARD, "evidence": "사진에서 추락 위험은 보이지 않는다."}}),
            "bool grade": json.dumps({"decision": "accept", "hazard": {**HAZARD, "likelihood_score": True}}),
            "placeholder": json.dumps({"decision": "accept", "hazard": {**HAZARD, "evidence": "..."}}),
            "invalid unicode": json.dumps({"decision": "accept", "hazard": {**HAZARD, "evidence": "문자\ud800다."}}),
            "duplicate key": '{"decision":"reject",' + text[1:],
            "truncated": text[:-1],
            "negative preface": "후보 위험은 보이지 않는다. " + text,
            "negative trailing": text + " 다만 위험이 확인되지 않는다.",
            "unwrapped alternative": "[" + text + "]",
            "markdown fenced": "```json\n" + text + "\n```",
        }
        for parser in (parse_verified_hazard, runtime["parse_verified_hazard"]):
            self.assertEqual(parser(" \n" + text + "\n ", CODE), HAZARD)
            for evidence in ("개구부에 안전난간이 설치되어 있지 않다.", "근로자가 안전대를 착용하지 않았다."):
                hazard = {**HAZARD, "evidence": evidence}
                self.assertEqual(parser(json.dumps({"decision": "accept", "hazard": hazard}), CODE), hazard)
            for case, raw in rejected.items():
                with self.subTest(case=case, parser=parser.__module__):
                    self.assertIsNone(parser(raw, CODE))

    def test_merge_turns_a_safe_scene_note_into_supported_hazard(self):
        base = [{"parsed": {"status": "no_visible_hazard", "scene_note": "난간이 확보되어 있다.", "hazards": []}}]
        before = copy.deepcopy(base)
        proposal = {"row_idx": 0, "code": CODE, "accepted": True, "hazard": HAZARD}
        mismatch = {**proposal, "code": "other"}
        self.assertEqual(merge_verified_candidates(base, [mismatch]), before)
        add = merge_verified_candidates(base, [proposal])
        self.assertEqual(base, before)
        self.assertEqual(add[0]["parsed"], {"status": "hazard", "scene_note": HAZARD["evidence"], "hazards": [HAZARD]})

    def test_fixed_candidate_gate_uses_only_scores_and_inspection_output(self):
        rows = [{"sample_id": str(i), "image": str(i)} for i in range(4)]
        empty = {"parsed": {"status": "no_visible_hazard", "hazards": []}}
        one = {"parsed": {"status": "hazard", "hazards": [{**HAZARD, "code": "other"}]}}
        drafts = [one, empty, empty, empty]
        probs = np.zeros((len(rows), len(CODES)))
        probs[:, CODES.index(CODE)] = [0.55, 0.63, 0.60, 0.63]
        scene = np.array([0.40, 0.59, 0.59, 0.57])
        fixed = {"temperatures": [1.0] * len(CODES), "thresholds": [0.5] * len(CODES), "scene_threshold": 0.5}
        proposals = select_booster_candidates(rows, drafts, probs, scene, fixed)
        self.assertEqual([item["row_idx"] for item in proposals], [0, 1])
        self.assertEqual(select_booster_candidates(rows, drafts, probs, scene, fixed, {CODE: 0.34}), [])

    def test_tail_weighted_assistant_loss_equals_full_sequence_reference(self):
        from types import SimpleNamespace

        logits = torch.arange(35, dtype=torch.float64).reshape(1, 5, 7).sin()
        labels = torch.tensor([[-100, -100, 3, 4, 2]])
        weights = torch.tensor([[0.0, 0.0, 4.5, 1.0, 4.5]])
        batch = {"input_ids": torch.tensor([[4, 2, 3, 4, 2]]), "labels": labels.clone()}

        def model(input_ids, logits_to_keep, use_cache):
            self.assertEqual(logits_to_keep, 4)
            self.assertFalse(use_cache)
            return SimpleNamespace(logits=logits[:, -logits_to_keep:, :])

        observed, supervised = forward_and_weighted_assistant_loss(model, batch, weights, 2)
        per_token = F.cross_entropy(logits[:, :-1].float().reshape(-1, 7), labels[:, 1:].reshape(-1),
                                    ignore_index=-100, reduction="none")
        goal = (per_token * weights[:, 1:].reshape(-1)).sum() / weights.sum()
        self.assertEqual(supervised, 3)
        torch.testing.assert_close(observed, goal)


if __name__ == "__main__":
    unittest.main()
