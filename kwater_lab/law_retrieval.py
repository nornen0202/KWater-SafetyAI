"""Offline, article-level retrieval experiment on the previously exposed pilot photos."""

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


ARTICLE_ID = re.compile(r"[^:]+:article:([0-9]+(?:-[0-9]+)?)")
MENTION = re.compile(r"OSH\s*-\s*\d+(?:\s*-\s*\d+)?|제\s*\d+\s*조(?:\s*의\s*\d+)?", re.I)
WORD = re.compile(r"[가-힣]+|[a-z]+|[0-9]+")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def article_texts(path):
    articles = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") != "article" or row.get("status") != "active":
                continue
            match = ARTICLE_ID.fullmatch(row["id"])
            if not match:
                raise ValueError(f"Unrecognized active article ID: {row['id']}")
            key = "OSH-" + match[1]
            if key in articles or not row.get("text"):
                raise ValueError(f"Duplicate or empty active article: {key}")
            articles[key] = (row.get("title") or "") + " " + row["text"]
    if not articles:
        raise ValueError("No active articles in the supplied official corpus")
    return articles


def tokenize(text, method):
    words = WORD.findall(MENTION.sub(" ", text.lower()))
    if method == "word":
        return words
    if method == "bigram":
        return [token for word in words for token in
                ([word] if len(word) == 1 else [word[i:i + 2] for i in range(len(word) - 1)])]
    raise ValueError(method)


class ArticleIndex:
    def __init__(self, articles, method):
        self.method = method
        self.postings = collections.defaultdict(list)
        self.lengths = {}
        for article, text in articles.items():
            tokens = tokenize(text, method)
            self.lengths[article] = len(tokens)
            for token, count in collections.Counter(tokens).items():
                self.postings[token].append((article, count))
        self.average_length = sum(self.lengths.values()) / len(self.lengths)
        self.idf = {t: math.log1p((len(articles) - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def rank(self, query):
        scores = collections.Counter()
        for token in sorted(set(tokenize(query, self.method))):
            for article, frequency in self.postings.get(token, ()):
                normalization = 1.2 * (0.25 + 0.75 * self.lengths[article] / self.average_length)
                scores[article] += self.idf[token] * frequency * 2.2 / (frequency + normalization)
        return sorted(scores, key=lambda article: (-scores[article], article))


def summarize(rows):
    n = len(rows)
    return {
        "queries": n,
        "top1_hits": sum(row["first_rank"] == 1 for row in rows),
        "top3_hits": sum(0 < row["first_rank"] <= 3 for row in rows),
        "hit_at_1": sum(row["first_rank"] == 1 for row in rows) / n if n else None,
        "hit_at_3": sum(0 < row["first_rank"] <= 3 for row in rows) / n if n else None,
        "mean_reciprocal_rank": sum(1 / row["first_rank"] for row in rows if row["first_rank"]) / n if n else None,
    }


def official_ranker_type(rag):
    spec = importlib.util.spec_from_file_location("kwater_participant_rag", rag)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LawRetriever


def load_official_ranker(rag, laws):
    with laws.open(encoding="utf-8") as stream:
        return official_ranker_type(rag)([json.loads(line) for line in stream])


def official_ranking(ranker, evidence, action, code=""):
    hazard = {"code": code, "evidence": MENTION.sub(" ", evidence), "prevention": MENTION.sub(" ", action)}
    ids = []
    for reference in ranker.select(hazard):
        record = ranker.source(reference)
        match = ARTICLE_ID.fullmatch(":".join(record["id"].split(":")[:3]))
        if not match:
            raise ValueError(f"Unexpected official legal record ID: {record['id']}")
        article = "OSH-" + match[1]
        if article not in ids:
            ids.append(article)
    return ids


def run(laws, labels, exposure, rag=None):
    subset = json.loads(exposure.read_text(encoding="utf-8"))
    if digest(labels) != subset["labels_sha256"]:
        raise ValueError("Pilot manifest and label SHA-256 differ")
    ids = set(subset["exposed_ids"])
    if len(ids) != len(subset["exposed_ids"]):
        raise ValueError("Duplicate ID in pilot manifest")
    articles = article_texts(laws)
    indices = {method: ArticleIndex(articles, method) for method in ("word", "bigram")}
    results = {name: [] for name in ("word_evidence_action", "bigram_evidence_action", "bigram_action")}
    if rag is not None:
        official_ranker = load_official_ranker(rag, laws)
        results["official_rag_no_examples"] = []
    seen = set()
    counts = collections.Counter()
    missing = collections.Counter()
    with labels.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in ids:
                continue
            if sid in seen:
                raise ValueError(f"Duplicate sample: {sid}")
            seen.add(sid)
            for code, hazard in row["target"]["hazards"].items():
                counts["hazard_rows"] += 1
                refs = {r["id"] for r in hazard.get("references", []) if r["kind"] == "law"}
                if not refs:
                    counts["without_law_reference"] += 1
                    continue
                missing.update(refs - articles.keys())
                if not refs.intersection(articles):
                    counts["without_in_corpus_reference"] += 1
                action = hazard["prevention"]
                full = hazard["evidence"] + " " + action
                for name, method, query in (("word_evidence_action", "word", full),
                                            ("bigram_evidence_action", "bigram", full),
                                            ("bigram_action", "bigram", action)):
                    ranking = indices[method].rank(query)
                    first = next((i for i, candidate in enumerate(ranking, 1) if candidate in refs), 0)
                    results[name].append({"sample_id": sid, "code": code, "first_rank": first})
                if rag is not None:
                    ranking = official_ranking(official_ranker, hazard["evidence"], action)
                    first = next((i for i, article in enumerate(ranking, 1) if article in refs), 0)
                    results["official_rag_no_examples"].append({"sample_id": sid, "code": code, "first_rank": first})
    if seen != ids:
        raise ValueError("Label file does not cover all pilot IDs")
    original, candidate = results["word_evidence_action"], results["bigram_evidence_action"]
    changes = collections.Counter()
    for old, new in zip(original, candidate, strict=True):
        a, b = old["first_rank"] == 1, new["first_rank"] == 1
        changes["gains" if b else "losses"] += int(a != b)
    return {
        "scope": "Pilot only; oracle Korean annotations. An exact ID hit means agreement with an annotated source.",
        "photos": len(seen), "active_articles": len(articles), "counts": dict(counts),
        "reference_ids_outside_active_corpus": missing,
        "query_policy": "Remove explicit article numbers; filter to previously exposed sample IDs. No label/code prior.",
        "bm25_tie_break": "Descending score then lexical article ID. Articles with no matching term have no rank.",
        "official_rag": {"sha256": digest(rag), "examples_used": 0,
                         "input": "Evidence and action only, no scene note. Up to five returned legal units are collapsed to unique parent articles."} if rag else None,
        "labels_sha256": subset["labels_sha256"], "exposure_sha256": digest(exposure), "laws_sha256": digest(laws),
        "methods": {name: summarize(rows) for name, rows in results.items()},
        "paired_top1_bigram_vs_word": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("laws", "labels", "exposure", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline-rag", type=Path, help="Official competition scripts/rag.py")
    args = parser.parse_args()
    report = run(args.laws, args.labels, args.exposure, args.baseline_rag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["methods"], ensure_ascii=False))


if __name__ == "__main__":
    main()
