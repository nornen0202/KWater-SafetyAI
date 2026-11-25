import json
from pathlib import Path
import tempfile
import unittest

from kwater_lab.law_retrieval import article_texts, digest, run, tokenize
from kwater_lab.law_examples import disjoint_training, training_examples


class LawRetrievalTests(unittest.TestCase):
    def test_excludes_deleted_articles_and_keeps_missing_gold_as_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            laws, labels, exposure = [root / name for name in ("laws", "labels", "exposure")]
            corpus = [
                {"id": "1:article:4-2", "record_type": "article", "status": "active",
                 "title": "분진", "text": "분진 발생 방지"},
                {"id": "1:article:5", "record_type": "article", "status": "deleted",
                 "text": "폐지된 조문"},
            ]
            laws.write_text("\n".join(map(json.dumps, corpus)), encoding="utf-8")
            self.assertEqual(set(article_texts(laws)), {"OSH-4-2"})
            body = {"evidence": "분진", "prevention": "분진 방지",
                    "references": [{"kind": "law", "id": "OSH-4-2"}]}
            absent = dict(body, references=[{"kind": "law", "id": "OSH-5"}])
            data = [{"sample_id": k, "target": {"hazards": {"chemical_exposure": v}}}
                    for k, v in [("a", body), ("b", absent), ("untouched", body)]]
            labels.write_text("\n".join(map(json.dumps, data)), encoding="utf-8")
            exposure.write_text(json.dumps({"exposed_ids": ["a", "b"], "labels_sha256": digest(labels)}))
            result = run(laws, labels, exposure)
            self.assertEqual(result["counts"]["without_in_corpus_reference"], 1)
            self.assertEqual(result["methods"]["bigram_action"]["hit_at_1"], .5)

    def test_explicit_gold_identifiers_cannot_drive_token_matching(self):
        self.assertEqual(tokenize("OSH-43 제4조의2 분진", "word"), ["분진"])

    def test_examples_cannot_include_cross_fold_near_duplicates(self):
        inventory = {sid: {"dhash64": value} for sid, value in
                     (("valid", "0"), ("near", "f"), ("far", "1f"))}
        self.assertEqual(disjoint_training(["near", "far"], ["valid"], inventory), ["far"])

    def test_case_library_preserves_training_reference_order_and_scrubs_queries(self):
        h = {"evidence": "OSH-7 분진", "prevention": "제13조 환기",
             "references": [{"kind": "law", "id": ref} for ref in ("OSH-7", "OSH-2", "OSH-7", "OSH-99")]}
        articles = {"OSH-7": {"id": "7"}, "OSH-2": {"id": "2"}}
        rows, prior = training_examples(["train"], {"train": {"c": h}, "other": {"d": h}}, articles)
        self.assertEqual([r["id"] for r in rows[0]["hazards"][0]["related_laws"]], ["7", "2"])
        self.assertNotIn("OSH-", rows[0]["hazards"][0]["evidence"])
        self.assertEqual(prior, {"c": {"OSH-7": 1, "OSH-2": 1}})
