"""Fit the deployment heads on the exposed development cohort and replay their shared threshold."""

import json
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from .metrics import CODES, evaluate
from .train_head import decisions, file_hash, fit, probabilities, save_atomic


def read(path):
    return json.loads(path.read_bytes())


def main():
    root = Path(__file__).resolve().parents[1]
    local = root/"data/prefix-pilot"
    ref = read(root/"results/encoder_pilot_384.json")
    control = read(root/"results/prefix_pilot_fpr20.json")
    fused = read(root/"results/encoder_fusion.json")
    exposure = read(local/"exposure.json")
    audit_path = root/"results/train_audit_full.json"
    audit = read(audit_path)
    if (file_hash(local/"exposure.json") != ref["provenance"]["exposure_sha256"]
            or file_hash(root/"results/encoder_pilot_384.json") != fused["source_sha256"]
            or exposure["labels_sha256"] != audit["labels_sha256"]):
        raise ValueError("Unexpected source cohort")
    buckets = exposure["by_prefix"]
    ids = sorted(k for members in buckets.values() for k in members)
    if len(ids) != 170 or len(set(ids)) != 170 or ids != sorted(exposure["exposed_ids"]):
        raise ValueError("Expected the fixed 170-photo cohort")
    groups = {k: prefix for prefix, members in buckets.items() for k in members}
    gold = {k: frozenset(audit["inventory"][k]["codes"]) for k in ids}
    y = np.asarray([[code in gold[k] for code in CODES] for k in ids], dtype=np.float64)
    paths = {"clip": (local/"features.npz", control["feature_cache"]["sha256"],512),
             "sig": (local/"siglip2-base16-384-features.npz", ref["cache_sha256"],768)}
    by_encoder, packed = {}, {}
    for name,(path,sha,width) in paths.items():
        if file_hash(path) != sha:
            raise ValueError(f"Unexpected {name} feature fingerprint")
        with np.load(path, allow_pickle=False) as cache:
            keys = cache["ids"].tolist()
            if set(keys) != set(ids) or len(keys) != len(ids) or cache["features"].shape != (len(keys),width):
                raise ValueError(f"Unexpected {name} dimensions or sample IDs")
            if name == "clip" and cache["audit_sha256"].item() != file_hash(audit_path):
                raise ValueError("Audit and image features differ")
            positions = {k:i for i,k in enumerate(keys)}
            x = cache["features"][[positions[k] for k in ids]]
        if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4):
            raise ValueError("Invalid image embeddings")
        by_encoder[name] = x
        packed.update({name+"_"+k:v for k,v in fit(x,y,0.01,250).items()})
    thresholds = [fused["thresholds"][prefix]["head"]["inner_calibrated"] for prefix in buckets]
    threshold = float(median(thresholds))
    path = root/".runtime/submission2-stage/head.npz"
    save_atomic(path, lambda f: np.savez_compressed(f, **packed, codes=np.array(CODES), threshold=threshold))
    with np.load(path, allow_pickle=False) as artifact:
        if artifact["codes"].tolist() != list(CODES) or artifact["threshold"].shape != () or any(
            not np.array_equal(artifact[k],v) for k,v in packed.items()):
            raise ValueError("Deployment model round-trip failure")
    print("HEAD READY", path, "threshold",threshold,"shapes",{k:v.shape for k,v in packed.items()},flush=True)

    fixed, nested = {}, {}
    for prefix, valid in buckets.items():
        val = np.array([groups[k]==prefix for k in ids])
        # The archived audit found no cross-photo exact or near duplicate pairs.
        if audit["exact_pixel_duplicates"] or audit["cross_group_near_pairs"]:
            raise ValueError("Repeat the original cross-fold duplicate filtering before scoring")
        probs = sum(probabilities(fit(x[~val],y[~val],0.01,250), x[val]) for x in by_encoder.values())/2
        keys = [k for k,flag in zip(ids,val,strict=True) if flag]
        fixed.update(decisions(keys,probs,threshold))
        nested.update(decisions(keys,probs,fused["thresholds"][prefix]["head"]["inner_calibrated"]))
    metrics = evaluate(gold,fixed,groups)
    previous = evaluate(gold,nested,groups)
    if abs(previous["macro_f1"]-fused["pooled"]["head"]["inner_calibrated"]["macro_f1"])>1e-12:
        raise ValueError("Archived nested ensemble score was not reproduced")
    summary = {"scope":"170 exposed photos, held-out ID prefixes; fixed threshold informed by the six archived inner calibrations. Not an independent estimate of the final refit or an official score.",
               "photos":len(ids),"encoder_weight_ratio":[1,1],"bce_l2":0.01,"steps":250,
               "head_sha256":file_hash(path),"feature_sha256":{k:v[1] for k,v in paths.items()},
               "codes":list(CODES),"schema":{k:list(v.shape) for k,v in packed.items()},
               "policy":"Median of six nested inner-calibrated head thresholds. Stable descending organizer code order, >= threshold, at most three codes.",
               "inner_thresholds":thresholds,"deployment_threshold":threshold,"oof_fixed_deployment_threshold":metrics,
               "oof_prediction_cardinality":dict(sorted(Counter(map(len,fixed.values())).items())),
               "prior_nested_macro_f1":previous["macro_f1"],"delta_vs_prior_nested_macro_f1":metrics["macro_f1"]-previous["macro_f1"],
               "oof_changed_code_sets_vs_prior_nested":sum(fixed[k]!=nested[k] for k in ids)}
    save_atomic(root/"results/submission2_head.json",lambda f:f.write((json.dumps(summary,indent=2)+"\n").encode()))
    print("OOF",metrics["macro_f1"],"previous",previous["macro_f1"],flush=True)


if __name__ == "__main__":
    main()
