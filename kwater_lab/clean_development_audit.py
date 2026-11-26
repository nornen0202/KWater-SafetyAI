"""Development-only, group-aware evaluation using fixed frozen image kernels."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from .metrics import CODES


MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
FEATURES = Path("data/clean-specialist-v1/features_multiview_2176d.npz")
GEMINI = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
LABELS = Path("data/train/train.jsonl")
PRIOR = Path("data/phase01-runs/phase01-20261126-0513-2155e8fa/evaluation-manifest.json")
PAIR_FILES = [PRIOR.parent / name for name in ("pairs.json", "semantic-pairs.json")]
COMMON = np.array([c not in {"drowning_inundation", "confined_space", "thermal_weather"} for c in CODES])
ID_FIELD = re.compile(rb'"sample_id"\s*:\s*"([^"]+)"')
PROTOCOL = {
    "ridge_regularization": 0.2,
    "minimum_calibration_positive_groups": 3,
    "baseline_threshold": 0.5,
    "candidate_false_alarm_delta_limit": 0.03,
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20261127,
    "source_shift_families": ["ADD", "SBE"],
}


def fit_predict(kernel, labels, train, test):
    train_kernel = kernel[np.ix_(train, train)].copy()
    train_kernel.flat[::len(train) + 1] += PROTOCOL["ridge_regularization"]
    y = labels[train].astype(float)
    mean = y.mean(axis=0)
    weights = solve(train_kernel, y - mean, assume_a="pos")
    return kernel[np.ix_(test, train)] @ weights + mean


def nested_prediction(kernel, labels, folds, groups, eligible, target):
    baseline = np.zeros_like(labels[:, :-1])
    candidate = np.zeros_like(labels[:, :-1])
    policies = {}
    for outer in sorted(set(folds[target])):
        test = np.flatnonzero((folds == outer) & target)
        train = np.flatnonzero((folds != outer) & eligible)
        if set(groups[test]) & set(groups[train]):
            raise ValueError("outer group overlap")
        train_folds = folds[train]
        inner_scores = np.zeros((len(train), labels.shape[1]))
        for inner in sorted(set(train_folds)):
            inner_train = train[train_folds != inner]
            inner_test = train[train_folds == inner]
            if set(groups[inner_train]) & set(groups[inner_test]):
                raise ValueError("inner group overlap")
            inner_scores[train_folds == inner] = fit_predict(kernel, labels, inner_train, inner_test)
        thresholds, gate = calibrate(inner_scores, labels[train], groups[train])
        scores = fit_predict(kernel, labels, train, test)
        baseline[test] = top3(scores, np.full(len(CODES), PROTOCOL["baseline_threshold"]))
        candidate[test] = top3(scores, thresholds) & (scores[:, -1:] >= gate)
        policies[str(int(outer))] = {
            "outer_train": len(train), "outer_test": len(test),
            "thresholds": dict(zip(CODES, [float(t) if np.isfinite(t) else None for t in thresholds])),
            "scene_threshold": float(gate) if np.isfinite(gate) else None,
        }
    return baseline, candidate, policies


def top3(scores, thresholds):
    margins = scores[:, :len(CODES)] - thresholds
    order = np.argsort(-margins, axis=1, kind="stable")[:, :3]
    chosen = np.zeros_like(margins, dtype=bool)
    np.put_along_axis(chosen, order, True, axis=1)
    return chosen & (margins >= 0)


def calibrate(scores, labels, groups):
    thresholds = np.full(len(CODES), np.inf)
    for c in range(len(CODES)):
        positive = labels[:, c]
        if len(set(groups[positive])) < PROTOCOL["minimum_calibration_positive_groups"]:
            continue
        grid = np.linspace(0.05, 0.90, 35)
        predicted = scores[:, c, None] >= grid
        f1 = 2 * (predicted & positive[:, None]).sum(axis=0) / (positive.sum() + predicted.sum(axis=0))
        thresholds[c] = grid[np.flatnonzero(f1 == f1.max())[-1]]
    raw = top3(scores, np.full(len(CODES), PROTOCOL["baseline_threshold"]))
    pred = top3(scores, thresholds)
    clean = ~labels[:, -1]
    fpr_cap = raw[clean].any(axis=1).mean() + PROTOCOL["candidate_false_alarm_delta_limit"]
    gates = np.r_[np.linspace(0.0, 1.0, 41), np.inf]
    feasible = []
    for gate in gates:
        gated = pred & (scores[:, -1:] >= gate)
        if gated[clean].any(axis=1).mean() <= fpr_cap:
            result = point(labels[:, :-1], gated)
            feasible.append((result["macro_f1"], -result["hazard_empty"], -gate))
    return thresholds, -max(feasible)[-1]


def point(y, pred):
    support = y.sum(axis=-2)
    count = pred.sum(axis=-2)
    active = (support + count) > 0
    hits = (y & pred).sum(axis=-2)
    f1 = np.divide(2 * hits, support + count, out=np.zeros_like(hits, dtype=float), where=active)
    yh, ph = y.any(axis=-1), pred.any(axis=-1)
    return {
        "macro_f1": float(f1[active].mean()) if active.any() else 0.0,
        "fixed_8_common_macro_f1": float(f1[COMMON].mean()),
        "active_codes": int(active.sum()),
        "prediction_only_active_codes": [CODES[i] for i in np.flatnonzero((support == 0) & (count > 0))],
        "no_visible_false_alarms": int((~yh & ph).sum()),
        "no_visible": int((~yh).sum()),
        "hazard_empty": int((yh & ~ph).sum()),
        "hazard": int(yh.sum()),
        "per_code": {
            code: {"support": int(support[i]), "predicted": int(count[i]), "true_positive": int(hits[i]), "f1": float(f1[i])}
            for i, code in enumerate(CODES)
        },
    }


def bootstrap(y, a, b, groups):
    grouped = [groups == key for key in sorted(set(groups))]
    clean = np.array([(~y[mask].any(axis=1)).sum() for mask in grouped])
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    n = len(grouped)
    draw = rng.multinomial(n, np.full(n, 1 / n), size=PROTOCOL["bootstrap_resamples"])
    support = draw @ np.array([y[mask].sum(axis=0) for mask in grouped])
    clean_n = draw @ clean

    def resample(pred):
        count = draw @ np.array([pred[mask].sum(axis=0) for mask in grouped])
        tp = draw @ np.array([(y[mask] & pred[mask]).sum(axis=0) for mask in grouped])
        active = (support + count) > 0
        f1 = np.divide(2 * tp, support + count, out=np.zeros_like(tp, dtype=float), where=active)
        fp = draw @ np.array([(~y[mask].any(axis=1) & pred[mask].any(axis=1)).sum() for mask in grouped])
        active_n = active.sum(axis=1)
        return np.divide(f1.sum(axis=1), active_n, out=np.zeros(len(draw)), where=active_n > 0), fp

    score_a, fp_a = resample(a)
    score_b, fp_b = resample(b)
    interval = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    valid = clean_n > 0
    return {
        "macro_f1_delta_95_percent": interval(score_b - score_a),
        "no_visible_groups": int((clean > 0).sum()),
        "no_visible_false_alarm_rate_a_95_percent": interval(fp_a[valid] / clean_n[valid]) if valid.any() else None,
        "no_visible_false_alarm_rate_b_95_percent": interval(fp_b[valid] / clean_n[valid]) if valid.any() else None,
        "no_visible_false_alarm_rate_delta_95_percent": interval((fp_b - fp_a)[valid] / clean_n[valid]) if valid.any() else None,
    }


def comparison(labels, a, b, groups, mask, severe):
    y, before, after, g = labels[mask, :-1], a[mask], b[mask], groups[mask]
    severe_gold = severe[mask]
    severe_rows = severe_gold.any(axis=1)
    return {
        "photos": len(y), "groups": len(set(g)), "a": point(y, before), "b": point(y, after),
        "paired_group_bootstrap": bootstrap(y, before, after, g),
        "severity_4": {
            "photos": int(severe_rows.sum()), "groups": len(set(g[severe_rows])),
            "complete_miss_a": int((severe_rows & ~(severe_gold & before).any(axis=1)).sum()),
            "complete_miss_b": int((severe_rows & ~(severe_gold & after).any(axis=1)).sum()),
        },
    }


def severity4(path, ids):
    expected = {sid.encode("utf-8"): sid for sid in ids}
    severe = {}
    with path.open("rb") as stream:
        for line in stream:
            match = ID_FIELD.search(line)
            if not match or match[1] not in expected:
                continue
            sid = expected[match[1]]
            if sid in severe:
                raise ValueError("duplicate development label row")
            target = json.loads(line)["target"]
            severe[sid] = [
                c in target["hazards"] and target["hazards"][c]["risk_level"]["severity"]["score"] == 4
                for c in CODES
            ]
    return np.array([severe[sid] for sid in ids], dtype=bool)


def write_local_fold0(root, path, ids, folds, gemini, candidate):
    if not path.resolve().is_relative_to((root / "data").resolve()):
        raise ValueError("prediction rows must be written beneath local-only data/")
    entries = []
    for i in np.flatnonzero(folds == 0):
        entries.append({
            "sample_id": ids[i],
            "gemini_raw05": [CODES[c] for c in np.flatnonzero(gemini[i])],
            "cpu_nested": [CODES[c] for c in np.flatnonzero(candidate[i])],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_pair_crossings(root, rows):
    dev = {row["sample_id"]: row for row in rows}
    audit = {}
    for path in PAIR_FILES:
        counts = Counter()
        for pair in json.loads((root / path).read_text(encoding="utf-8")):
            counts["recorded_pairs"] += 1
            left, right = dev.get(pair["left"]), dev.get(pair["right"])
            if left is None or right is None:
                continue
            counts["both_development"] += 1
            if left["dev_fold"] == right["dev_fold"]:
                continue
            counts["cross_fold"] += 1
            if pair["decision"] == "unresolved":
                counts["cross_fold_unresolved"] += 1
                if {left["family"], right["family"]} & {"ADD", "SBE"}:
                    counts["cross_fold_unresolved_with_add_or_sbe"] += 1
            inliers = int(pair.get("inliers") or 0)
            ratio = float(pair.get("inlier_fraction", pair.get("ratio", 0)) or 0)
            if pair["decision"] in {"quarantine_same_visual_source", "quarantine"} or (inliers >= 16 and ratio >= 0.18):
                counts["cross_fold_known_links"] += 1
        counts.setdefault("cross_fold_known_links", 0)
        audit[path.name] = dict(counts)
    return audit


def run(root, fold0_output=None):
    rows = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("development manifest contains an evaluation row")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample ID")
    folds = np.array([row["dev_fold"] for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    families = np.array([row["family"] for row in rows])
    labels = np.array([[c in row["codes"] for c in CODES] + [row["status"] == "hazard"] for row in rows])
    if not np.array_equal(labels[:, :-1].any(axis=1), labels[:, -1]):
        raise ValueError("inconsistent scene vs code labels")
    severe = severity4(root / LABELS, ids)
    if (severe & ~labels[:, :-1]).any():
        raise ValueError("severity label code differs from development manifest")
    with np.load(root / FEATURES, allow_pickle=False) as archive:
        index = {sid: i for i, sid in enumerate(archive["ids"].tolist())}
        features = archive["features"][[index[sid] for sid in ids]].astype(np.float32)
    kernel = np.zeros((len(rows), len(rows)), dtype=np.float64)
    for start, end in ((0, 1152), (1152, 2176)):
        for view in (features[:, 0, start:end], features[:, 1:, start:end].mean(axis=1)):
            view /= np.linalg.norm(view, axis=1, keepdims=True).clip(min=1e-8)
            similarity = view @ view.T
            kernel += (similarity + similarity**2) / 8
    all_dev = np.ones(len(rows), dtype=bool)
    raw, calibrated, policies = nested_prediction(kernel, labels, folds, groups, all_dev, all_dev)
    with np.load(root / GEMINI, allow_pickle=False) as archive:
        if (archive["ids"].tolist() != ids or not np.array_equal(archive["folds"], folds)
                or not np.array_equal(archive["y_codes"], labels[:, :-1])):
            raise ValueError("stored specialist probabilities do not match development IDs/folds/labels")
        gemini = top3(archive["code_probs"], np.full(len(CODES), 0.5))
    if fold0_output is not None:
        write_local_fold0(root, fold0_output, ids, folds, gemini, calibrated)
    shift = {}
    for family in PROTOCOL["source_shift_families"]:
        mask = families == family
        eligible = ~np.isin(groups, groups[mask])
        _, shifted, shift_policies = nested_prediction(kernel, labels, folds, groups, eligible, mask)
        shift[family] = {
            **comparison(labels, calibrated, shifted, groups, mask, severe),
            "groups_removed_from_training": len(set(groups[mask])),
            "photos_removed_from_training": int((~eligible).sum()),
            "policies": shift_policies,
        }
    prior = json.loads((root / PRIOR).read_text(encoding="utf-8"))
    development_ids = set(ids)
    prior_partition = Counter("phase01_development" if row["dev_fold"] >= 0 else "phase01_holdout"
                              for row in prior if row["sample_id"] not in development_ids)
    return {
        "protocol": PROTOCOL,
        "input_sha256": {str(path): sha256(root / path) for path in (MANIFEST, FEATURES, GEMINI, LABELS, PRIOR, *PAIR_FILES)},
        "clean_v1_panel_partition_in_phase01": dict(prior_partition),
        "prior_pair_audit_development_only": audit_pair_crossings(root, rows),
        "gemini_fixed_vs_cpu_nested": comparison(labels, gemini, calibrated, groups, all_dev, severe),
        "cpu_fixed_vs_cpu_nested": comparison(labels, raw, calibrated, groups, all_dev, severe),
        "by_fold": {str(int(fold)): comparison(labels, gemini, calibrated, groups, folds == fold, severe)
                    for fold in sorted(set(folds))},
        "source_shift": shift, "calibrations": policies,
        "development_families": dict(Counter(families.tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/clean-development-nested-audit.json"))
    parser.add_argument("--local-fold0-output", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.local_fold0_output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
