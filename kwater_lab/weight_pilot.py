"""Run the fixed head-edit and visual-retrieval experiments on the exposed pilot only."""

import argparse
import json
from pathlib import Path

import numpy as np

from .clip_probe import REVISION
from .metrics import CODES, evaluate, paired_bootstrap
from .prefix_pilot import calibrate_fpr20, filtered_training, get_features, policy_decisions, prefix_cohort
from .train_head import calibration, decisions, file_hash, fit, probabilities, save_atomic


METHODS = ("original", "shared_axis_edit", "random_control", "neighbors", "blend")
PAIR = ("slip_trip", "fall_height")


def edit_columns(weight, indices, fraction=0.5, seed=42):
    """Attenuate positive alignment, or apply the same-sized orthogonal sham edit."""
    targeted, control = weight.copy(), weight.copy()
    rng = np.random.default_rng(seed)
    first, second = indices
    changes = []
    for target, other in ((first, second), (second, first)):
        w, axis = weight[:, target], weight[:, other]
        norm = np.linalg.norm(axis)
        unit = axis / norm if norm else np.zeros_like(axis)
        edit = fraction * max(0.0, np.dot(w, unit)) * unit
        targeted[:, target] -= edit
        random_direction = rng.normal(size=len(axis))
        random_direction -= np.dot(random_direction, unit) * unit
        control[:, target] += np.linalg.norm(edit) * random_direction / np.linalg.norm(random_direction)
        changes.append(float(np.linalg.norm(edit)))
    return targeted, control, changes


def neighbor_scores(x, y, probe, k=8, sharpness=20):
    similarities = probe @ x.T
    neighbors = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(similarities, neighbors, axis=1)
    weights = np.exp(sharpness * (top - top.max(axis=1, keepdims=True)))
    return np.einsum("nk,nkc->nc", weights / weights.sum(axis=1, keepdims=True), y[neighbors])


def predictions(train, valid, features, gold):
    x = np.stack([features[s] for s in train])
    probe = np.stack([features[s] for s in valid])
    y = np.array([[c in gold[s] for c in CODES] for s in train], dtype=float)
    head = fit(x, y, 0.01, 250)
    idx = [CODES.index(c) for c in PAIR]
    a, b = head["weight"][:, idx].T
    length = np.linalg.norm(a) * np.linalg.norm(b)
    weights, random, changes = edit_columns(head["weight"], idx)
    base = probabilities(head, probe)
    cache = neighbor_scores(x, y, probe)
    return {"original": base, "shared_axis_edit": probabilities(dict(head, weight=weights), probe),
            "random_control": probabilities(dict(head, weight=random), probe),
            "neighbors": cache, "blend": (base + cache) / 2}, {
                "axis_cosine": float(np.dot(a,b) / length) if length else None, "edit_l2_by_target": changes}


def pilot_data(audit_path, exposure_path, cache_path, reference):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    buckets = prefix_cohort(audit["inventory"])
    keys = sorted(k for seq in buckets.values() for k in seq)
    if (buckets != exposure["by_prefix"] or keys != sorted(exposure["exposed_ids"])
            or audit["labels_sha256"] != exposure["labels_sha256"]):
        raise ValueError("Cohort differs from the registered exposure")
    if not cache_path.is_file() or file_hash(cache_path) != reference["feature_cache"]["sha256"]:
        raise ValueError("This experiment requires the archived, verified pilot feature cache")
    provenance = {"scope": "exploratory_prefix_subset", "framing": "full_frame", "revision": REVISION,
                  "labels_sha256": audit["labels_sha256"], "audit_sha256": file_hash(audit_path)}
    inventory = {s: audit["inventory"][s] for s in keys}
    features, _ = get_features(cache_path, keys, inventory, None, None, provenance)
    gold = {s: frozenset(row["codes"]) for s, row in inventory.items()}
    linked = {s: set() for s in keys}
    for i, s in enumerate(keys):
        x = inventory[s]
        for t in keys[i+1:]:
            z = inventory[t]
            if (x["pixels_sha256"] == z["pixels_sha256"] or
                    (int(x["dhash64"], 16) ^ int(z["dhash64"], 16)).bit_count() <= 4):
                linked[s].add(t)
                linked[t].add(s)
    return buckets, features, gold, linked


def run(audit, exposure, cache, reference_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    buckets, features, gold, linked = pilot_data(audit, exposure, cache, reference)
    groups = {s: p for p, ids in buckets.items() for s in ids}
    policies = ("fixed", "base_threshold", "inner_calibrated")
    collected = {p: {m: {} for m in METHODS} for p in policies}
    constrained = {m: {} for m in ("original", "blend")}
    folds = {}
    for prefix, valid in buckets.items():
        train = filtered_training(gold, valid, linked)
        outer, geometry = predictions(train, valid, features, gold)
        keys, inner = [], {m: [] for m in METHODS}
        for other_prefix, subset in buckets.items():
            if prefix == other_prefix:
                continue
            test = sorted(set(train).intersection(subset))
            if not test:
                continue
            allowed = filtered_training(train, test, linked)
            inner_batch, _ = predictions(allowed, test, features, gold)
            keys.extend(test)
            for m in METHODS:
                inner[m].append(inner_batch[m])
        inner_gold = {s: gold[s] for s in keys}
        inner_groups = {s: groups[s] for s in keys}
        threshold = {m: calibration(np.vstack(inner[m]), inner_gold, inner_groups)
                     ["selected_threshold"] for m in METHODS}
        fold_scores = {}
        for m in METHODS:
            for policy, t in zip(policies, (0.5, threshold["original"], threshold[m])):
                pred = decisions(valid, outer[m], t)
                collected[policy][m].update(pred)
                if policy == "inner_calibrated":
                    fold_scores[m] = evaluate({s: gold[s] for s in valid}, pred)["macro_f1"]
        folds[prefix] = {"train": len(train), "validation": len(valid), "geometry": geometry,
                         "inner_threshold": threshold, "macro_f1": fold_scores}
        limits = {}
        for m, output in constrained.items():
            chosen = calibrate_fpr20(np.vstack(inner[m]), inner_gold, inner_groups)["chosen"]
            decisions_out = policy_decisions(valid, outer[m], chosen["threshold"])
            output.update(decisions_out)
            measured = evaluate({s: gold[s] for s in valid}, decisions_out)
            limits[m] = {"inner_selected": chosen,
                         "outer_false_alarms": measured["no_hazard_false_alarms"],
                         "outer_no_hazard_photos": measured["no_hazard_photos"],
                         "outer_macro_f1": measured["macro_f1"]}
        folds[prefix]["followup_inner_fpr20"] = limits
    pooled = {p: {m: evaluate(gold, pred, groups) for m, pred in models.items()}
              for p, models in collected.items()}
    for p, key in (("inner_calibrated", "inner_calibrated"), ("fixed", "fixed_0.5")):
        if pooled[p]["original"] != reference["pooled_out_of_prefix"]["linear_head"][key]:
            raise ValueError("Unedited control did not reproduce the archived benchmark")
    constrained_metrics = {m: evaluate(gold, pred, groups) for m, pred in constrained.items()}
    if constrained_metrics["original"] != reference["pooled_out_of_prefix"]["linear_head"]["inner_fpr20"]:
        raise ValueError("Constrained control differs from the archived benchmark")
    chosen = collected["inner_calibrated"]
    double = [s for s,v in gold.items() if set(PAIR) <= v]
    paired = {m: paired_bootstrap(gold, chosen["original"], chosen[m], groups)
              for m in METHODS if m != "original"}
    no_correct_code = {p: {m: sum(bool(labels) and labels.isdisjoint(pred[s]) for s, labels in gold.items())
                            for m, pred in models.items()}
                       for p, models in {**collected, "followup_fpr20": constrained}.items()}
    return {"scope": "Exploratory, repeated use of 170 exposed images; six unverified ID prefixes",
            "protocol": "docs/weight-pilot-protocol.md", "photos": len(gold), "reference_reproduced": True,
            "seed": 42, "edit_fraction": 0.5, "edit_target_order": PAIR, "top_k": 8, "kernel_sharpness": 20,
            "feature_sha256": file_hash(cache), "exposure_sha256": file_hash(exposure),
            "audit_sha256": file_hash(audit), "folds": folds, "pooled": pooled,
            "hazard_photos_with_no_correct_code": no_correct_code,
            "followup_fpr20": {"metrics": constrained_metrics, "paired_prefix_resampling":
                               paired_bootstrap(gold, constrained["original"], constrained["blend"], groups)},
            "paired_prefix_resampling": paired, "both_target_codes_in_gold": len(double),
            "both_targets_recalled": {m: sum(set(PAIR) <= pred[s] for s in double) for m,pred in chosen.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, path in {"audit":"results/train_audit_full.json", "exposure":"data/prefix-pilot/exposure.json",
                      "cache":"data/prefix-pilot/features.npz", "reference":"results/prefix_pilot_fpr20.json",
                      "output":"results/weight_pilot.json"}.items():
        parser.add_argument("--" + key, type=Path, default=Path(path))
    args = parser.parse_args()
    report = run(args.audit, args.exposure, args.cache, args.reference)
    save_atomic(args.output, lambda f: f.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode()))
    for m, s in report["pooled"]["inner_calibrated"].items():
        print(m, round(s["macro_f1"],5), s["no_hazard_false_alarms"], s["hazard_completely_missed"])


if __name__ == "__main__":
    main()
