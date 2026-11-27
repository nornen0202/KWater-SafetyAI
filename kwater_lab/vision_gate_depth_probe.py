"""Matched CPU depth test on the 170 previously exposed photos, using frozen modern RGB features."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

from .fold4_code_diagnosis import DEV


DEPTH = Path("data/prefix-pilot/relative-depth-features.npz")
RGB = Path("data/clean-specialist-v1/features_multiview_2176d.npz")
OUT = Path("results/vision-gate-depth-probe-20261128.json")


def rgb_kernel(views):
    result = np.zeros((len(views), len(views)))
    for view in range(views.shape[1]):
        for block in (views[:, view, :1152], views[:, view, 1152:]):
            x = block.astype(np.float32)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
            result += x @ x.T
    return result / (2 * views.shape[1])


def auxiliary_kernel(values, train):
    z = (values - values[train].mean(0)) / np.maximum(values[train].std(0), 1e-6)
    distances = cdist(z, z, metric="sqeuclidean")
    fit_dist = distances[np.ix_(train, train)]
    return np.exp(-distances / max(np.median(fit_dist[fit_dist > 0]), 1e-6))


def predict(kernel, labels, train, valid):
    clf = SVC(kernel="precomputed", C=1, class_weight="balanced")
    clf.fit(kernel[np.ix_(train, train)], labels[train])
    return clf.predict(kernel[np.ix_(valid, train)])


def counts(y, pred, normal):
    tp, fp, fn = (int(mask.sum()) for mask in (pred & y, pred & ~y, ~pred & y))
    return dict(tp=tp, fp=fp, fn=fn, f1=2 * tp / max(2 * tp + fp + fn, 1),
                normal_alerts=int(sum(pred & normal)))


def main():
    rows = {r["sample_id"]: r for r in json.loads(DEV.read_bytes())}
    with np.load(DEPTH) as cache:
        ids, vectors = cache["ids"].tolist(), cache["vectors"]
        signature = str(cache["signature"])
    with np.load(RGB) as cache:
        idx = {sid: i for i, sid in enumerate(cache["ids"].tolist())}
        views = cache["features"][[idx[sid] for sid in ids]]
        spec, rgb_digest = str(cache["spec"]), str(cache["digest_key"])
    if len(set(ids)) != 170 or views.shape != (170, 7, 2176) or vectors.shape != (170, 3, 32):
        raise ValueError("Unexpected paired feature dimensions or repeated inputs")
    folds = np.array([rows[s]["dev_fold"] for s in ids])
    groups = np.array([rows[s]["group_id"] for s in ids])
    y = np.array(["struck_by_lifting" in rows[s]["codes"] for s in ids])
    normal = np.array([not rows[s]["codes"] for s in ids])
    rgb = rgb_kernel(views)
    modes = ["rgb", "brightness", "edge", "depth"] + [f"shuffled_{i}" for i in range(20)]
    pooled = {m: np.zeros(len(y), dtype=bool) for m in modes}
    by_fold = {}
    for fold in sorted(set(folds)):
        train, valid = np.where(folds != fold)[0], np.where(folds == fold)[0]
        if set(groups[train]) & set(groups[valid]) or len(set(y[train])) != 2:
            raise ValueError("Invalid group split or missing training class")
        by_fold[str(fold)] = dict(n=len(valid), positive=int(sum(y[valid])), groups=len(set(groups[valid])))
        for m in modes:
            kernel = rgb
            if m != "rgb":
                channel = {"brightness": 0, "edge": 1}.get(m, 2)
                values = vectors[:, channel]
                if m.startswith("shuffled_"):
                    rng = np.random.default_rng(1000 * fold + int(m.split("_")[-1]))
                    # Only other-group training donors. No held-out donor serves a training photo.
                    donors = [rng.choice(train[groups[train] != g]) for g in groups]
                    values = values[donors]
                kernel = rgb + .25 * auxiliary_kernel(values, train)
            pooled[m][valid] = predict(kernel, y, train, valid)
    stats = {m: counts(y, pred, normal) for m, pred in pooled.items()}
    report = dict(scope="Exposed 170, three existing grouped folds. Single lifting code; fixed kernel SVM diagnostic.",
                  inputs=dict(rgb_spec=spec, rgb_cache_digest=rgb_digest, depth_signature=signature,
                              depth_sha256=sha256(DEPTH.read_bytes()).hexdigest(), manifest_sha256=sha256(DEV.read_bytes()).hexdigest()),
                  groups=len(set(groups)), normal=int(sum(normal)), fold_counts=by_fold, results=stats,
                  depth_beats_wrong=sum(stats["depth"]["f1"] > stats[m]["f1"] for m in modes[4:]))
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
