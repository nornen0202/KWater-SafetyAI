"""Corpus-only title retrieval diagnostic; never a photo/action accuracy estimate.

Run: .venv/bin/python -m kwater_lab.ubuntu_law_control --device cpu
Only active article rows are read. No article text, IDs, vectors or weights are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, NewType, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import functional as F

ArticleId = NewType("ArticleId", str)
ROOT: Final = Path(__file__).resolve().parents[1]
ARTICLE: Final = re.compile(r"제\s*\d+\s*(?:조|항|호|장|절)(?:\s*의\s*\d+)?")
HEADING: Final = re.compile(r"^\s*제\s*\d+\s*조(?:\s*의\s*\d+)?\s*\([^)]*\)")
LAW_NAME: Final = re.compile(r"[「『][^」』]+(?:법|령|규칙)[」』]")
SEED: Final = 20261127
FEATURES: Final = 4096
RANK: Final = 32
EPOCHS: Final = 80
BATCH: Final = 64
TEMPERATURE: Final = 0.1
RESIDUAL_SCALE: Final = 0.5


@dataclass(frozen=True, slots=True)
class CorpusError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Article:
    article_id: ArticleId
    query: str
    body: str


@dataclass(frozen=True, slots=True)
class Corpus:
    articles: tuple[Article, ...]
    active: int
    short_or_empty: int
    ambiguous: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Metrics:
    queries: int
    candidates: int
    top1_hits: int
    recall_at_1: float
    mrr: float
    tied_positive_queries: int


class Report(TypedDict):
    schema: str
    input_sha256: str
    split_sha256: str
    device: str
    seed: int
    active_articles: int
    excluded_short_or_empty: int
    excluded_ambiguous_title_or_body: int
    train_articles: int
    heldout_articles: int
    candidate_articles: int
    candidate_pool: str
    split_policy: str
    feature_policy: str
    parameters: dict[str, int | float]
    zero_feature_heldout_queries: int
    zero_feature_candidate_bodies: int
    permutation_fixed_points: int
    methods: dict[str, dict[str, int | float]]
    elapsed_seconds: float
    tie_policy: str
    selection_policy: str
    limitations: list[str]


def normalize(text: str) -> str:
    """Ignore whitespace/punctuation/digits for both dedupe and model features."""
    return re.sub(r"[^가-힣a-z]", "", unicodedata.normalize("NFKC", text).lower())


def strip_pair(title: str, body: str, law_name: str) -> tuple[str, str]:
    """Remove headings, citations, law names and every exact normalized title."""
    query = normalize(ARTICLE.sub("", LAW_NAME.sub("", title.replace(law_name, ""))))
    cleaned = HEADING.sub("", body)
    cleaned = re.sub(r"<[^>]*>", "", cleaned)
    cleaned = normalize(
        ARTICLE.sub("", LAW_NAME.sub("", cleaned.replace(law_name, "")))
    )
    return query, cleaned.replace(query, "") if query else cleaned


def load_corpus(path: Path) -> Corpus:
    """Reject malformed/duplicate IDs; remove all ambiguous normalized pairs."""
    rows: list[Article] = []
    original_bodies: dict[ArticleId, str] = {}
    seen: set[ArticleId] = set()
    active = short = 0
    if path.stat().st_size > 64 * 1024 * 1024:
        raise CorpusError("Corpus exceeds the fixed 64 MiB input budget")
    raw_bytes = path.read_bytes()
    for line in raw_bytes.decode("utf-8").splitlines():
        raw = json.loads(line)
        if raw.get("record_type") != "article" or raw.get("status") != "active":
            continue
        active += 1
        fields = [raw.get(key) for key in ("id", "title", "text", "law_name")]
        if not all(isinstance(value, str) for value in fields):
            raise CorpusError("Active article lacks required string fields")
        identifier, title, body, law_name = fields
        article_id = ArticleId(identifier)
        if not article_id or article_id in seen:
            raise CorpusError("Empty or duplicate active article ID")
        seen.add(article_id)
        query, stripped = strip_pair(title, body, law_name)
        if len(query) < 3 or len(stripped) < 30 or query == stripped:
            short += 1
            continue
        rows.append(Article(article_id, query, stripped))
        original_bodies[article_id] = normalize(HEADING.sub("", body))
    titles = Counter(row.query for row in rows)
    bodies = Counter(row.body for row in rows)
    originals = Counter(original_bodies.values())
    unique = tuple(
        row
        for row in rows
        if titles[row.query]
        == bodies[row.body]
        == originals[original_bodies[row.article_id]]
        == 1
    )
    if len(unique) < 20 or len(unique) > 2048:
        raise CorpusError("Retained corpus must contain 20..2048 unambiguous articles")
    return Corpus(
        unique,
        active,
        short,
        len(rows) - len(unique),
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def split_articles(
    articles: tuple[Article, ...],
) -> tuple[tuple[Article, ...], tuple[Article, ...]]:
    """SHA-256 split of normalized title/body singleton groups (20% reserved)."""
    train, heldout = [], []
    for row in sorted(articles, key=lambda item: (item.query, item.body)):
        digest = hashlib.sha256(f"{SEED}\0{row.query}\0{row.body}".encode()).digest()
        (heldout if int.from_bytes(digest[:8]) % 5 == 0 else train).append(row)
    if len(train) < 2 or not heldout:
        raise CorpusError(
            "Split must have at least two training and one held-out article"
        )
    return tuple(train), tuple(heldout)


def features(
    train: tuple[Article, ...], heldout: tuple[Article, ...]
) -> tuple[NDArray[np.float32], NDArray[np.float32], TfidfVectorizer]:
    """Fit vocabulary/IDF only on training titles and stripped training bodies."""
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), max_features=FEATURES
    )
    vectorizer.fit([text for row in train for text in (row.query, row.body)])
    all_rows = train + heldout
    queries = np.asarray(
        csr_matrix(vectorizer.transform([row.query for row in all_rows])).toarray(),
        dtype=np.float32,
    )
    bodies = np.asarray(
        csr_matrix(vectorizer.transform([row.body for row in all_rows])).toarray(),
        dtype=np.float32,
    )
    return queries, bodies, vectorizer


def permuted_targets(size: int) -> NDArray[np.int64]:
    """A seeded single cycle permutes every training pair with no fixed points."""
    order = np.random.default_rng(SEED).permutation(size)
    targets = np.empty(size, dtype=np.int64)
    targets[order] = np.roll(order, 1)
    return targets


def score_metrics(scores: NDArray[np.float32], labels: NDArray[np.int64]) -> Metrics:
    """Pessimistic tie ranks prevent candidate ordering from acting as an ID prior."""
    correct = scores[np.arange(len(labels)), labels, None]
    ranks = np.sum(scores >= correct, axis=1)
    ties = np.sum(scores == correct, axis=1) > 1
    hits = int(np.sum(ranks == 1))
    return Metrics(
        len(labels),
        scores.shape[1],
        hits,
        hits / len(labels),
        float(np.mean(1.0 / ranks)),
        int(ties.sum()),
    )


def fit_head(
    tensors: tuple[torch.Tensor, torch.Tensor], targets: NDArray[np.int64]
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Optimize only train-pair CE; all three comparisons share frozen features."""
    queries, bodies = tensors
    torch.manual_seed(SEED)
    left = torch.nn.Parameter(
        torch.randn(queries.shape[1], RANK, device=queries.device)
        / np.sqrt(queries.shape[1])
    )
    right = torch.nn.Parameter(
        torch.randn(queries.shape[1], RANK, device=queries.device)
        / np.sqrt(queries.shape[1])
    )
    optimizer = torch.optim.AdamW([left, right], lr=0.003, weight_decay=0.01)
    labels = torch.tensor(targets, dtype=torch.long, device=queries.device)
    order_rng = np.random.default_rng(SEED)
    losses: list[float] = []
    for _ in range(EPOCHS):
        total = 0.0
        for indices in np.array_split(
            order_rng.permutation(len(queries)), (len(queries) + BATCH - 1) // BATCH
        ):
            batch = torch.tensor(indices, device=queries.device)
            baseline = queries[batch] @ bodies.T
            residual = (
                F.normalize(queries[batch] @ left, dim=1)
                @ F.normalize(bodies @ right, dim=1).T
            )
            loss = F.cross_entropy(
                (baseline + RESIDUAL_SCALE * residual) / TEMPERATURE, labels[batch]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(queries))
    return left.detach(), right.detach(), losses[0], losses[-1]


def run(laws: Path, device: Literal["cpu", "cuda"]) -> Report:
    """Run once with fixed parameters; return aggregate-only descriptive evidence."""
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    corpus = load_corpus(laws)
    train, heldout = split_articles(corpus.articles)
    query_array, body_array, vectorizer = features(train, heldout)
    queries = torch.tensor(query_array, device=device)
    bodies = torch.tensor(body_array, device=device)
    count = len(train)
    labels = np.arange(count, len(corpus.articles))
    with torch.no_grad():
        baseline = queries[count:] @ bodies.T
    methods = {"frozen_tfidf": asdict(score_metrics(baseline.cpu().numpy(), labels))}
    for name, targets in (
        ("true_pairs", np.arange(count)),
        ("permuted_train_pairs", permuted_targets(count)),
    ):
        fit_start = time.perf_counter()
        left, right, first, last = fit_head((queries[:count], bodies[:count]), targets)
        with torch.no_grad():
            residual = (
                F.normalize(queries[count:] @ left, dim=1)
                @ F.normalize(bodies @ right, dim=1).T
            )
            scores = baseline + RESIDUAL_SCALE * residual
        methods[name] = {
            **asdict(score_metrics(scores.cpu().numpy(), labels)),
            "first_epoch_train_ce": first,
            "last_epoch_train_ce": last,
            "fit_seconds": time.perf_counter() - fit_start,
        }
    split_payload = [
        [partition, row.article_id, row.query, row.body]
        for partition, rows in (("train", train), ("heldout", heldout))
        for row in rows
    ]
    return {
        "schema": "corpus_only_law_control_v1",
        "input_sha256": corpus.sha256,
        "split_sha256": hashlib.sha256(
            json.dumps(split_payload, ensure_ascii=False).encode()
        ).hexdigest(),
        "device": device,
        "seed": SEED,
        "active_articles": corpus.active,
        "excluded_short_or_empty": corpus.short_or_empty,
        "excluded_ambiguous_title_or_body": corpus.ambiguous,
        "train_articles": count,
        "heldout_articles": len(heldout),
        "candidate_articles": len(corpus.articles),
        "candidate_pool": "All retained train + heldout stripped bodies; "
        "original article IDs are internal labels only.",
        "split_policy": "All repeated normalized titles/bodies excluded; "
        "SHA-256 of seed/title/body singleton groups modulo 5 reserves bucket 0.",
        "feature_policy": "Char 2..4 grams; vocabulary and IDF fit on train titles/bodies only; "
        "same frozen vectors for every method.",
        "parameters": {
            "max_features": FEATURES,
            "actual_features": len(vectorizer.vocabulary_),
            "rank": RANK,
            "epochs": EPOCHS,
            "max_batch": BATCH,
            "temperature": TEMPERATURE,
            "residual_scale": RESIDUAL_SCALE,
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "cpu_threads": 2,
        },
        "zero_feature_heldout_queries": int(
            np.sum(np.linalg.norm(query_array[count:], axis=1) == 0)
        ),
        "zero_feature_candidate_bodies": int(
            np.sum(np.linalg.norm(body_array, axis=1) == 0)
        ),
        "permutation_fixed_points": int(
            np.sum(permuted_targets(count) == np.arange(count))
        ),
        "methods": methods,
        "elapsed_seconds": time.perf_counter() - started,
        "tie_policy": "Worst rank among exact score ties; all-zero queries receive rank equal to candidate count.",
        "selection_policy": "Fixed parameters/epochs; no heldout tuning, stopping, "
        "threshold selection or head persistence.",
        "limitations": [
            (
                "Descriptive corpus diagnostic only; "
                "no independent photo, action-evidence or legal applicability accuracy."
            ),
            (
                "Exact titles, headings, law names, article references and digits stripped, "
                "but title fragments/inflections remain lexical shortcuts."
            ),
            (
                "Identical normalized titles/bodies excluded; "
                "near-duplicates and shared legal templates can still cross the split."
            ),
            (
                "Heldout bodies are retrieval candidates only; "
                "their query-body pairs never enter fitting or vocabulary/IDF."
            ),
            "One corpus, one deterministic split and seed; no uncertainty or generalization claim.",
            "No photos, train annotations, generated responses, external model, article ID features or saved weights.",
        ],
    }


def save_report(report: Report, destination: Path) -> None:
    """Atomically save aggregates only beneath the repository's ignored data tree."""
    output = destination.resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        raise CorpusError(
            "--out must be inside the repository's ignored data directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--laws", type=Path, default=ROOT / "data/baseline/assets/laws.jsonl"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/ubuntu-law-control/report.json"
    )
    args = parser.parse_args()
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = run(args.laws, args.device)
    save_report(report, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "methods": report["methods"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
