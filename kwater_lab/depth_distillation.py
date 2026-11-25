"""Distill monocular depth descriptors into RGB features with grouped cross-fitting."""

import json
from pathlib import Path

import numpy as np

from .depth_pilot import unrelated
from .metrics import evaluate, paired_bootstrap
from .observation_pilot import label_matrix, load_cohort, ridge_projection, write_aggregate
from .prefix_pilot import filtered_training
from .train_head import decisions, file_hash, fit, probabilities


CHANNELS = ("brightness", "edge", "depth", "wrong_depth")


def predict_auxiliary(train, query, data, descriptors, groups, salt):
    """A query uses only its RGB embedding. All teacher descriptors come from train IDs."""
    donors = unrelated(train, train, groups, data.linked, salt)
    teacher = np.stack([np.concatenate((*descriptors[s], descriptors[donors[s]][2])) for s in train])
    x = np.stack([data.clip[s] for s in train])
    q = np.stack([data.clip[s] for s in query])
    prediction, _ = ridge_projection(x, teacher, q)
    return prediction.reshape(len(query), len(CHANNELS), 32)


def run(root):
    data = load_cohort(root)
    ref_path = root / "results/depth_pilot.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    cache = root / "data/prefix-pilot/relative-depth-features.npz"
    if file_hash(cache) != ref["feature_cache_sha256"] or ref["official_rgb_feature_sha256"] != data.hashes["clip_cache"]:
        raise ValueError("Completed depth baseline does not match the two feature caches")
    with np.load(cache, allow_pickle=False) as archive:
        keys, maps = archive["ids"].tolist(), archive["vectors"]
    if keys != sorted(data.gold) or maps.shape != (len(keys), 3, 32) or not np.isfinite(maps).all():
        raise ValueError("Expected all 170 exposed photos with three finite 32-dimensional descriptors")
    descriptors = dict(zip(keys, maps, strict=True))
    groups = {s: g for g, subset in data.buckets.items() for s in subset}
    predictions = {m: {} for m in ("rgb", *CHANNELS)}
    squared_error, mean_error = np.zeros(3), np.zeros(3)
    for outside, valid in data.buckets.items():
        train = filtered_training(keys, valid, data.linked)
        aux_rows = {}
        for inner, subset in data.buckets.items():
            if inner == outside:
                continue
            dev = sorted(set(subset).intersection(train))
            keep = filtered_training(train, dev, data.linked)
            predicted = predict_auxiliary(keep, dev, data, descriptors, groups, f"{outside}/{inner}")
            aux_rows.update(zip(dev, predicted, strict=True))
        if set(aux_rows) != set(train):
            raise ValueError("Every training feature must be predicted outside its proxy group")
        v_aux = predict_auxiliary(train, valid, data, descriptors, groups, outside + "/outer")
        t_aux = np.stack([aux_rows[s] for s in train])
        x, q = (np.stack([data.clip[s] for s in ids]) for ids in (train, valid))
        y = label_matrix(train, data.gold)
        for name in predictions:
            if name == "rgb":
                x_aug, q_aug = x, q
            else:
                channel = CHANNELS.index(name)
                x_aug, q_aug = np.column_stack((x, t_aux[:, channel])), np.column_stack((q, v_aux[:, channel]))
            p = probabilities(fit(x_aug, y, .01, 250), q_aug)
            predictions[name].update(decisions(valid, p, .5))
        # Access held-out geometry only now to evaluate auxiliary regression accuracy.
        actual = np.stack([descriptors[s] for s in valid])
        average = np.stack([descriptors[s] for s in train]).mean(axis=0)
        squared_error += np.square(actual - v_aux[:, :3]).sum(axis=(0, 2))
        mean_error += np.square(actual - average).sum(axis=(0, 2))
    metrics = {name: evaluate(data.gold, rows, groups) for name, rows in predictions.items()}
    if any(metrics["rgb"][key] != ref["metrics"]["rgb"][key] for key in ("macro_f1", "no_hazard_false_alarms", "per_code")):
        raise ValueError("RGB control did not reproduce the relative-depth experiment")
    pairs = {name: paired_bootstrap(data.gold, predictions[name], predictions["depth"], groups)
             for name in ("rgb", "wrong_depth")}
    return {"scope": "170 exposed photos and six unverified prefix proxies; code classification only",
            "source_hashes": data.hashes, "depth_cache_sha256": ref["feature_cache_sha256"],
            "depth_model_revision": ref["original_model_revision"], "rgb_only_at_inference": True,
            "training": "Inner leave-one-prefix-out RGB-to-descriptor ridge 0.25. Final classifier L2 0.01, 250 updates, threshold 0.5, max three codes.",
            "auxiliary_r2_vs_outer_training_mean": dict(zip(CHANNELS[:3], (1 - squared_error / mean_error).tolist(), strict=True)),
            "metrics": metrics, "paired_unverified_prefix_resampling_vs_distilled_depth": pairs}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = run(root)
    write_aggregate(root / "results/depth_distillation.json", report)
    print({m: (round(v['macro_f1'], 4), v['no_hazard_false_alarms']) for m, v in report['metrics'].items()})
