"""Cross training-only Korean evidence with a capped expected-Macro-F1 decision rule."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import zlib

import numpy as np

from .encoder_pilot import scores
from .metrics import CODES, evaluate, paired_bootstrap
from .observation_pilot import label_matrix, load_cohort, ridge_projection, write_aggregate
from .prefix_pilot import filtered_training
from .train_head import calibration, decisions
from .weight_pilot import neighbor_scores


SOURCES = ("image", "paired", "within_code_shuffle")
POLICIES = ("common", "marginal", "marginal_same_count")
BINS = np.asarray([0.1, 0.25, 0.5, 0.75])
REMOVED = re.compile(r"[A-Za-z]+(?:_[A-Za-z]+)+|제\s*\d+\s*조(?:의\s*\d+)?|"
                     r"추락|낙상|전도|걸림|끼임|협착|충돌|낙하|비래|감전|익수|침수|화재|폭발|밀폐공간|열사병|한랭질환|위험성?|재해")


@lru_cache(maxsize=1024)
def text_vector(observation):
    clean = re.sub(r"[^가-힣a-z0-9]+", " ", REMOVED.sub(" ", observation.lower())).strip()
    vector = np.zeros(512)
    for n in (2, 3):
        for i in range(len(clean) - n + 1):
            bucket = zlib.crc32(clean[i:i+n].encode())
            vector[bucket % len(vector)] += 1 if (bucket >> 31) == 0 else -1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def normalize_rows(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def paired_targets(ids, observations):
    by_code = defaultdict(list)
    for i, sid in enumerate(ids):
        for code, text in observations[sid].items():
            by_code[code].append((i, sid, text_vector(text)))
    paired = np.zeros((len(ids), 512))
    shuffled = np.zeros_like(paired)
    for code, entries in by_code.items():
        entries.sort(key=lambda v: hashlib.sha256((code + '/' + v[1]).encode()).digest())
        for j, (row, _, value) in enumerate(entries):
            paired[row] += value
            shuffled[row] += entries[(j+1) % len(entries)][2]
    return normalize_rows(paired), normalize_rows(shuffled)


def split_scores(train, valid, cohort):
    base = scores(train, valid, cohort.siglip, cohort.gold)["blend"]
    result = {"image": base}
    x, q = (np.stack([cohort.siglip[s] for s in keys]) for keys in (train, valid))
    y = label_matrix(train, cohort.gold)
    paired, sham = paired_targets(train, cohort.observations)
    # Fit both targets through the same ridge system. A query supplies only image features.
    estimated, _ = ridge_projection(x, np.column_stack((paired, sham)), q)
    for source, target, prediction in zip(SOURCES[1:], (paired, sham), np.hsplit(estimated, 2), strict=True):
        aux = neighbor_scores(target, y, normalize_rows(prediction))
        result[source] = 0.75 * base + 0.25 * aux
    return result


def marginal_utility(inner, truth, emitted, outer):
    """Frozen plug-in approximation of the marginal class F1 gain for one prediction."""
    active = truth.any(axis=0)
    assignment = np.searchsorted(BINS, inner, side="right")
    test_bins = np.searchsorted(BINS, outer, side="right")
    calibrated = np.zeros_like(outer)
    for b, midpoint in enumerate((0.05, 0.175, 0.375, 0.625, 0.875)):
        mask = assignment == b
        pooled = mask[:, active]
        prior = (truth[:, active][pooled].sum() + 2 * midpoint) / (pooled.sum() + 2)
        rates = ((truth * mask).sum(axis=0) + 8 * prior) / (mask.sum(axis=0) + 8)
        calibrated = np.where(test_bins == b, rates, calibrated)
    denominator = (truth + emitted).sum(axis=0)
    twice_tp = 2 * (truth * emitted).sum(axis=0)
    pooled_f1 = twice_tp.sum() / max(denominator.sum(), 1)
    baseline_f1 = (twice_tp + 10 * pooled_f1) / (denominator + 10)
    utility = (2 * calibrated - baseline_f1) / (denominator + 11)
    utility[:, ~active] = -np.inf
    return utility


def run(root):
    cohort = load_cohort(root)
    groups = {s: g for g, ids in cohort.buckets.items() for s in ids}
    decisions_by_arm = {src: {policy: {} for policy in POLICIES} for src in SOURCES}
    output_top = defaultdict(int)
    chosen = {}
    for outer, valid in cohort.buckets.items():
        train = filtered_training(cohort.gold, valid, cohort.linked)
        test = split_scores(train, valid, cohort)
        dev_ids, inner_scores = [], defaultdict(list)
        for other, subset in cohort.buckets.items():
            if other == outer:
                continue
            dev = sorted(set(subset).intersection(train))
            inner_train = filtered_training(train, dev, cohort.linked)
            batch = split_scores(inner_train, dev, cohort)
            dev_ids.extend(dev)
            for src in SOURCES:
                inner_scores[src].append(batch[src])
        dev_gold = {s: cohort.gold[s] for s in dev_ids}
        dev_groups = {s: groups[s] for s in dev_ids}
        truth = label_matrix(dev_ids, dev_gold)
        chosen[outer] = {}
        for src in SOURCES:
            inner = np.vstack(inner_scores[src])
            threshold = calibration(inner, dev_gold, dev_groups)["selected_threshold"]
            chosen[outer][src] = threshold
            predicted = decisions(valid, test[src], threshold)
            emitted = label_matrix(dev_ids, decisions(dev_ids, inner, threshold))
            utility = marginal_utility(inner, truth, emitted, test[src])
            arms = decisions_by_arm[src]
            arms["common"].update(predicted)
            for i, sid in enumerate(valid):
                top = np.argsort(-utility[i], kind="stable")[:3]
                arms["marginal"][sid] = frozenset(CODES[c] for c in top if utility[i, c] > 0)
                slots = len(predicted[sid])
                arms["marginal_same_count"][sid] = frozenset(CODES[c] for c in top[:slots] if np.isfinite(utility[i, c]))
                assert len(arms["marginal_same_count"][sid]) == slots
                rank = np.argsort(-test[src][i], kind="stable")[:3]
                output_top[src] += len(cohort.gold[sid] & {CODES[c] for c in rank})
    results = {}
    for src, arms in decisions_by_arm.items():
        results[src] = {}
        for policy, predicted in arms.items():
            m = evaluate(cohort.gold, predicted, groups)
            m["hazard_photos_with_no_correct_code"] = sum(bool(g) and g.isdisjoint(predicted[s]) for s, g in cohort.gold.items())
            m["total_emitted"] = sum(map(len, predicted.values()))
            results[src][policy] = m
    archived = json.loads((root / "results/encoder_pilot_384.json").read_text(encoding="utf-8"))["pooled"]["siglip2"]["blend"]["inner_calibrated"]
    base = results["image"]["common"]
    if any(base[k] != archived[k] for k in ("macro_f1", "no_hazard_false_alarms", "per_code")):
        raise ValueError("Registered image-only control did not reproduce")
    pairs = (("image", "common", "paired", "common"), ("within_code_shuffle", "common", "paired", "common"),
             ("paired", "common", "paired", "marginal"))
    uncertainty = {f"{a}_{p}_to_{b}_{q}": paired_bootstrap(cohort.gold, decisions_by_arm[a][p], decisions_by_arm[b][q], groups)
                   for a, p, b, q in pairs}
    return {"scope": "170 exposed photos; nested six-prefix proxy CV, no independent event or public score",
            "hashes": cohort.hashes, "photos": len(cohort.gold), "instances": sum(map(len, cohort.gold.values())),
            "validation_inputs": "image only", "photo_specific_text": "training-fold-only Korean evidence or scene_evidence; within-code cyclic shuffle control",
            "common_thresholds": chosen, "raw_top3_hits": dict(output_top), "results": results,
            "paired_resampling_of_unverified_prefixes": uncertainty,
            "source_weight": 0.25, "calibration_bin_edges": BINS.tolist(), "calibration_pseudocount": 8, "f1_pseudocount": 10}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run(root)
    write_aggregate(root / "results/evidence_factorial.json", result)
    for src, policies in result["results"].items():
        print(src, {p: (round(m['macro_f1'], 4), m['no_hazard_false_alarms']) for p, m in policies.items()})
