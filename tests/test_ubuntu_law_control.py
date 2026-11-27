"""Behavioral checks for corpus isolation, honest ranking and learned pairing."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from kwater_lab.ubuntu_law_control import (
    Article,
    ArticleId,
    CorpusError,
    features,
    fit_head,
    load_corpus,
    permuted_targets,
    score_metrics,
    split_articles,
    strip_pair,
)


class UbuntuLawControlTests(unittest.TestCase):
    def test_heading_references_law_names_and_exact_title_are_removed(self) -> None:
        query, body = strip_pair(
            "제15조 산업규칙 안전 난간",
            "제15조(안전 난간) 산업규칙 제20조의2 안전난간을 설치한다. "
            "「작업법」 <개정 2026. 1. 1.> 작업자를 보호한다.",
            "산업규칙",
        )
        self.assertEqual(query, "안전난간")
        self.assertEqual(body, "을설치한다작업자를보호한다")

    def test_ambiguous_pairs_are_excluded_before_order_invariant_split(self) -> None:
        records = [
            {
                "id": f"law:article:{index}",
                "record_type": "article",
                "status": "active",
                "title": f"작업{chr(0xAC00 + index)}",
                "text": f"제{index}조(작업{chr(0xAC00 + index)}) "
                + chr(0xB000 + index) * 35,
                "law_name": "규칙",
            }
            for index in range(40)
        ]
        records[1]["title"] = records[0]["title"]
        records[3]["text"] = records[2]["text"]
        records[4]["text"] = records[5]["text"] = (
            str(records[4]["title"]) + str(records[5]["title"]) + "공통본문" * 12
        )
        records.append({**records[39], "id": "empty", "title": ""})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "laws.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8"
            )
            corpus = load_corpus(path)
            self.assertEqual(
                (corpus.active, corpus.short_or_empty, corpus.ambiguous), (41, 1, 6)
            )
            self.assertEqual(
                {row.article_id for row in corpus.articles},
                {f"law:article:{i}" for i in range(6, 40)},
            )
            train, heldout = split_articles(corpus.articles)
            self.assertEqual(
                (train, heldout), split_articles(tuple(reversed(corpus.articles)))
            )
            self.assertTrue(train and heldout)
            self.assertFalse({r.query for r in train} & {r.query for r in heldout})
            self.assertFalse({r.body for r in train} & {r.body for r in heldout})
            records.append(records[10])
            path.write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8"
            )
            with self.assertRaises(CorpusError):
                load_corpus(path)

    def test_heldout_only_terms_cannot_enter_vocabulary_or_idf(self) -> None:
        train = (
            Article(ArticleId("a"), "가나다", "가나다라"),
            Article(ArticleId("b"), "마바사", "마바사아"),
        )
        heldout = (Article(ArticleId("c"), "자차카", "타파하"),)
        queries, bodies, vocabulary = features(train, heldout)
        self.assertEqual(int(np.count_nonzero(queries[-1])), 0)
        self.assertEqual(int(np.count_nonzero(bodies[-1])), 0)
        _, _, other_vocabulary = features(train, heldout + heldout)
        np.testing.assert_array_equal(vocabulary.idf_, other_vocabulary.idf_)

    def test_ties_are_pessimistic_and_zero_queries_are_counted(self) -> None:
        scores = np.asarray(
            [[0.8, 0.2, 0.1], [0.5, 0.5, 0.1], [0, 0, 0]], dtype=np.float32
        )
        metrics = score_metrics(scores, np.asarray([0, 1, 2]))
        self.assertEqual(
            (metrics.queries, metrics.candidates, metrics.top1_hits), (3, 3, 1)
        )
        self.assertAlmostEqual(metrics.mrr, (1 + 0.5 + 1 / 3) / 3)
        self.assertEqual(metrics.tied_positive_queries, 2)

    def test_head_learns_cross_feature_pairing_above_permuted_control(self) -> None:
        torch.set_num_threads(2)
        latent = np.random.default_rng(17).normal(size=(48, 8)).astype(np.float32)
        zeros = np.zeros_like(latent)
        queries = F.normalize(
            torch.tensor(np.concatenate((latent, zeros), axis=1)), dim=1
        )
        bodies = F.normalize(
            torch.tensor(np.concatenate((zeros, latent), axis=1)), dim=1
        )
        labels = np.arange(32)
        permuted = permuted_targets(32)
        np.testing.assert_array_equal(np.sort(permuted), labels)
        self.assertFalse(np.any(permuted == labels))
        results = []
        for targets in (labels, permuted):
            left, right, first, last = fit_head((queries[:32], bodies[:32]), targets)
            self.assertLess(last, first)
            scores = (
                F.normalize(queries[32:] @ left, dim=1)
                @ F.normalize(bodies @ right, dim=1).T
            )
            results.append(score_metrics(scores.numpy(), np.arange(32, 48)))
        self.assertGreater(results[0].recall_at_1, 0.8)
        self.assertGreater(results[0].recall_at_1, results[1].recall_at_1)


if __name__ == "__main__":
    unittest.main()
