"""Test what labelled law examples add to retrieval on six exploratory ID prefixes."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .law_retrieval import ARTICLE_ID, MENTION, digest, official_ranker_type, official_ranking, summarize


TOP1_OUTCOME = {(False, False): "both_missed", (True, True): "both_hit",
                (False, True): "retrieval_only", (True, False): "frequency_only"}


def disjoint_training(training, validation, inventory):
    valid_hashes = [int(inventory[sid]["dhash64"], 16) for sid in validation]
    return [sid for sid in training if all(
        (int(inventory[sid]["dhash64"], 16) ^ other).bit_count() > 4 for other in valid_hashes)]


def select_articles(laws):
    articles = {}
    for row in laws:
        if row["record_type"] == "article" and row["status"] == "active":
            match = ARTICLE_ID.fullmatch(row["id"])
            if match:
                articles["OSH-" + match[1]] = row
    return articles


def law_ids(hazard):
    return list(dict.fromkeys(r["id"] for r in hazard.get("references", ()) if r["kind"] == "law"))


def training_examples(ids, labels, articles):
    examples = []
    frequencies = defaultdict(Counter)
    for sid in ids:
        hazards = []
        for code, entry in labels[sid].items():
            refs = [ref for ref in law_ids(entry) if ref in articles]
            frequencies[code].update(refs)
            hazards.append({"code": code, "evidence": MENTION.sub(" ", entry["evidence"]),
                            "prevention": MENTION.sub(" ", entry["prevention"]),
                            "related_laws": [articles[ref] for ref in refs]})
        examples.append({"hazards": hazards})
    return examples, frequencies


def experiment(laws_path, rag_path, labels_path, exposure_path, audit_path):
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    labels_sha = digest(labels_path)
    if not labels_sha == exposure["labels_sha256"] == audit["labels_sha256"]:
        raise ValueError("Label SHA mismatch")
    groups = exposure["by_prefix"]
    ids = [sid for group in groups.values() for sid in group]
    if len(ids) != len(set(ids)) or set(ids) != set(exposure["exposed_ids"]):
        raise ValueError("Exposure must contain disjoint prefixes and cover the pilot exactly")
    ids = set(ids)
    with labels_path.open(encoding="utf-8") as f:
        labels = {r["sample_id"]: r["target"]["hazards"] for r in map(json.loads, f) if r["sample_id"] in ids}
    with laws_path.open(encoding="utf-8") as f:
        laws = [json.loads(line) for line in f]
    articles = select_articles(laws)
    ranker_type = official_ranker_type(rag_path)
    by_method = defaultdict(list)
    paired_top1 = Counter()
    folds = {}
    for prefix, validation in groups.items():
        training = sorted(ids.difference(validation))
        kept = disjoint_training(training, validation, audit["inventory"])
        examples, frequencies = training_examples(kept, labels, articles)
        retriever = ranker_type(laws, train=examples)
        fold_scores = defaultdict(list)
        for sid in validation:
            for code, hazard in labels[sid].items():
                gold = set(law_ids(hazard))
                if not gold:
                    continue
                frequency = frequencies[code]
                prior = sorted(frequency, key=lambda ref: (-frequency[ref], ref))[:3]
                ranked = official_ranking(retriever, hazard["evidence"], hazard["prevention"], code)
                prior_hit = bool(prior and prior[0] in gold)
                ranked_hit = bool(ranked and ranked[0] in gold)
                paired_top1[TOP1_OUTCOME[prior_hit, ranked_hit]] += 1
                for method, suggestions in (("code_frequency", prior), ("official_train_examples", ranked)):
                    first = next((i for i, ref in enumerate(suggestions, 1) if ref in gold), 0)
                    fold_scores[method].append({"first_rank": first})
        for method, rows in fold_scores.items():
            by_method[method].extend(rows)
        folds[prefix] = {"train_photos": len(kept), "query_photos": len(validation),
                         "removed_for_image_similarity": len(training) - len(kept),
                         "scores": {method: summarize(rows) for method, rows in fold_scores.items()}}
        del retriever
    return {"scope": "Exploratory leave-one-ID-prefix-out. No verified incident identifiers. Oracle hazard code, evidence and prevention; no scene notes. No bundled training examples or adapter.",
            "query_count_unit": "hazard annotation with at least one gold law ID",
            "retriever_output": "The official selector returns five legal units, collapsed to unique articles. The prior returns up to three articles.",
            "reference_order_for_training": "Original order of distinct active article IDs",
            "labels_sha256": labels_sha, "exposure_sha256": digest(exposure_path), "laws_sha256": digest(laws_path),
            "rag_sha256": digest(rag_path), "audit_sha256": digest(audit_path), "folds": folds,
            "paired_top1": dict(paired_top1),
            "methods": {key: summarize(rows) for key, rows in by_method.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("laws", "rag", "labels", "exposure", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = experiment(args.laws, args.rag, args.labels, args.exposure, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["methods"]))


if __name__ == "__main__":
    main()
