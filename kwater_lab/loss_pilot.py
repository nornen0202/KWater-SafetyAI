"""Compare three multi-label objectives on the registered exploratory K-water cohort."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from .metrics import CODES, evaluate, paired_bootstrap
from .prefix_pilot import calibrate_fpr20, filtered_training, policy_decisions
from .train_head import calibration, file_hash, fit, probabilities, save_atomic
from .weight_pilot import pilot_data


def asymmetric_gradient(probability, targets, *, power, margin=0.05):
    """Derivative with respect to logits, including the derivative of the focal weight."""
    shifted = np.maximum(probability - margin, 0.0)
    slope = shifted ** power / (1.0 - shifted)
    if power:
        slope -= power * shifted ** (power - 1) * np.log1p(-shifted)
    negative = np.where(shifted > 0, slope * probability * (1.0 - probability), 0.0)
    return np.where(targets == 1, probability - 1.0, negative)


LOSSES = {"bce": None, "negative_clip": partial(asymmetric_gradient, power=0),
          "asymmetric": partial(asymmetric_gradient, power=4)}
POLICIES = ("fixed", "inner_calibrated", "inner_fpr20")


def fit_scores(train, valid, features, gold):
    train_x = np.stack([features[s] for s in train])
    test_x = np.stack([features[s] for s in valid])
    y = np.array([[code in gold[s] for code in CODES] for s in train], dtype=float)
    scores = {}
    for name, gradient in LOSSES.items():
        head = fit(train_x, y, 0.01, 500, logit_gradient=gradient, step_scale=0.25)
        scores[name] = probabilities(head, test_x)
        if not np.isfinite(scores[name]).all():
            raise FloatingPointError(f"Non-finite outputs from {name}")
    return scores


def run(root):
    audit = root / "results/train_audit_full.json"
    exposure = root / "data/prefix-pilot/exposure.json"
    cache = root / "data/prefix-pilot/features.npz"
    reference = json.loads((root / "results/prefix_pilot_fpr20.json").read_text(encoding="utf-8"))
    buckets, features, gold, linked = pilot_data(audit, exposure, cache, reference)
    groups = {s: prefix for prefix, ids in buckets.items() for s in ids}
    outputs = {p: {name: {} for name in LOSSES} for p in POLICIES}
    folds = {}
    for prefix, valid in buckets.items():
        train = filtered_training(gold, valid, linked)
        outer = fit_scores(train, valid, features, gold)
        keys, inner = [], {name: [] for name in LOSSES}
        for other, ids in buckets.items():
            if other == prefix:
                continue
            calibration_ids = sorted(set(ids).intersection(train))
            if not calibration_ids:
                continue
            allowed = filtered_training(train, calibration_ids, linked)
            batch = fit_scores(allowed, calibration_ids, features, gold)
            keys.extend(calibration_ids)
            for name in LOSSES:
                inner[name].append(batch[name])
        inner_gold = {s: gold[s] for s in keys}
        inner_groups = {s: groups[s] for s in keys}
        folds[prefix] = {}
        for name in LOSSES:
            data = np.vstack(inner[name])
            thresholds = {"fixed": 0.5, "inner_calibrated": calibration(data, inner_gold, inner_groups)["selected_threshold"],
                          "inner_fpr20": calibrate_fpr20(data, inner_gold, inner_groups)["chosen"]["threshold"]}
            folds[prefix][name] = thresholds
            for policy, threshold in thresholds.items():
                outputs[policy][name].update(policy_decisions(valid, outer[name], threshold))
    scores, paired = {}, {}
    for policy, by_name in outputs.items():
        scores[policy] = {}
        for name, pred in by_name.items():
            metrics = evaluate(gold, pred, groups)
            metrics["hazard_photos_with_no_correct_code"] = sum(
                bool(labels) and labels.isdisjoint(pred[s]) for s, labels in gold.items())
            scores[policy][name] = metrics
        paired[policy] = paired_bootstrap(gold, by_name["bce"], by_name["asymmetric"], groups)
    return {"protocol": "docs/improvement-experiments.md", "protocol_section": "E2", "scope": "Previously exposed 170 photos; not independent incidents",
            "exposure_sha256": file_hash(exposure), "feature_sha256": file_hash(cache), "photos": len(gold),
            "steps": 500, "step_scale": 0.25, "l2": 0.01, "negative_margin": 0.05, "negative_power": 4,
            "thresholds_by_outer_group": folds, "pooled": scores, "paired_asymmetric_minus_bce": paired}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/loss_pilot.json"))
    args = parser.parse_args()
    result = run(args.root)
    save_atomic(args.output, lambda f: f.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode()))
    for name, metrics in result["pooled"]["inner_calibrated"].items():
        print(name, metrics["macro_f1"], metrics["no_hazard_false_alarms"], metrics["hazard_photos_with_no_correct_code"])
