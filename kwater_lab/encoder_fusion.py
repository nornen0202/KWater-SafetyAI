"""Evaluate a fixed 1:1 CLIP-224 and SigLIP2-384 ensemble with nested threshold selection."""

import json
from pathlib import Path

import numpy as np

from .adapter_scale_pilot import write_json
from .encoder_pilot import MODELS, POLICIES, cross_validate, scores
from .metrics import evaluate, paired_bootstrap
from .prefix_pilot import filtered_training, policy_decisions
from .train_head import file_hash
from .weight_pilot import pilot_data


def average_scores(train, valid, features, gold):
    clip, siglip = (scores(train, valid, encoder, gold) for encoder in features)
    return {method: (clip[method] + siglip[method]) / 2 for method in MODELS}


def run(root):
    audit = root / "results/train_audit_full.json"
    exposure = root / "data/prefix-pilot/exposure.json"
    comparison = root / "results/encoder_pilot_384.json"
    ref = json.loads(comparison.read_text(encoding="utf-8"))
    control = json.loads((root / "results/prefix_pilot_fpr20.json").read_text(encoding="utf-8"))
    buckets, clip, gold, linked = pilot_data(audit, exposure, root / "data/prefix-pilot/features.npz", control)
    path = root / "data/prefix-pilot/siglip2-base16-384-features.npz"
    if file_hash(path) != ref["cache_sha256"] or file_hash(exposure) != ref["provenance"]["exposure_sha256"]:
        raise ValueError("The high-resolution comparison and feature cache differ")
    with np.load(path, allow_pickle=False) as data:
        sig = dict(zip(data["ids"].tolist(), data["features"], strict=True))
    groups = {sid: prefix for prefix, valid in buckets.items() for sid in valid}
    combined, thresholds = cross_validate(buckets, (clip, sig), gold, groups, linked, score_fn=average_scores)

    baseline = {name: {m: {p: {} for p in POLICIES} for m in MODELS} for name in ("clip", "siglip2")}
    for prefix, valid in buckets.items():
        train = filtered_training(gold, valid, linked)
        for name, features in (("clip", clip), ("siglip2", sig)):
            for method, matrix in scores(train, valid, features, gold).items():
                for policy, threshold in ref["thresholds"][name][prefix][method].items():
                    baseline[name][method][policy].update(policy_decisions(valid, matrix, threshold))
    pooled, paired = {}, {}
    for method in MODELS:
        pooled[method], paired[method] = {}, {}
        for policy in POLICIES:
            pred = combined[method][policy]
            metrics = evaluate(gold, pred, groups)
            metrics["hazard_photos_with_no_correct_code"] = sum(bool(g) and g.isdisjoint(pred[s]) for s, g in gold.items())
            pooled[method][policy], paired[method][policy] = metrics, {}
            for name, original in baseline.items():
                base = original[method][policy]
                if evaluate(gold, base, groups)["macro_f1"] != ref["pooled"][name][method][policy]["macro_f1"]:
                    raise ValueError("Published standalone F1 was not reproduced")
                paired[method][policy][name] = paired_bootstrap(gold, base, pred, groups)
    return {"scope": "Post-hoc exploratory fixed-ratio fusion on the exposed 170 photos; six unverified prefixes",
            "source_sha256": file_hash(comparison), "ratio_clip_to_siglip2": [1, 1], "photos": len(gold),
            "thresholds": thresholds, "pooled": pooled, "paired_fusion_minus_single": paired}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = run(root)
    write_json(root / "results/encoder_fusion.json", output)
    for method in MODELS:
        m = output["pooled"][method]["inner_calibrated"]
        print(method, m["macro_f1"], m["no_hazard_false_alarms"], m["per_code"]["fall_height"]["tp"])
