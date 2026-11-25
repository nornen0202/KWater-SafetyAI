"""Exploratory, nested leave-one-ID-prefix-out benchmark when case IDs are unavailable.

S-number prefixes are *unverified proxy groups*. The output is a local diagnostic.
Reserve all exposed images and their subsequently identified incidents for development.
"""

import argparse
import hashlib
import io
import json
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from .clip_probe import FILES, REVISION, ImageEncoder
from .metrics import CODES, evaluate
from .train_head import THRESHOLDS, calibration, decisions, file_hash, fit, probabilities, save_atomic


def prefix_cohort(inventory):
    buckets = defaultdict(list)
    for sid in inventory:
        match = re.fullmatch(r"(S[0-9]+)C[0-9]+", sid)
        if match:
            buckets[match[1]].append(sid)
    # Select entire prefixes using IDs and counts alone, before reading labels.
    eligible = [p for p, ids in buckets.items() if 12 <= len(ids) <= 40]
    ordered = sorted(eligible, key=lambda p: hashlib.sha256(("kwater-prefix-pilot-v1\0" + p).encode()).digest())
    if len(ordered) < 6:
        raise ValueError("Fewer than six eligible prefixes")
    return {p: sorted(buckets[p]) for p in ordered[:6]}


def write_exposure(audit_path, output):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    buckets = prefix_cohort(audit["inventory"])
    record = {"scope": "predeclared exploratory use; exclude these photos and all related incidents from final holdout",
              "labels_sha256": audit["labels_sha256"], "by_prefix": buckets,
              "exposed_ids": sorted(k for subset in buckets.values() for k in subset)}
    # The original manifest was written on Windows. Preserve its byte fingerprint on every host.
    payload = (json.dumps(record, indent=2) + "\n").replace("\n", "\r\n").encode()
    if output.exists() and output.read_bytes() != payload:
        raise ValueError("An existing exposure manifest differs from the reconstructed cohort")
    save_atomic(output, lambda stream: stream.write(payload))


def filtered_training(pool, validation, linked):
    forbidden = set(validation)
    for key in validation:
        forbidden.update(linked[key])
    remaining = sorted(set(pool) - forbidden)
    if not remaining:
        raise ValueError("empty training set after possible duplicate filtering")
    return remaining


def policy_decisions(keys, scores, threshold):
    if threshold is None:
        return dict.fromkeys(keys, frozenset())
    return decisions(keys, scores, threshold)


def calibrate_fpr20(scores, gold, groups):
    # No outer labels enter this function. None is a real "always abstain" policy.
    records = []
    for t in (*THRESHOLDS, None):
        result = evaluate(gold, policy_decisions(gold, scores, t), groups)
        records.append({"threshold": t, "macro_f1": result["macro_f1"],
                        "no_hazard_false_alarms": result["no_hazard_false_alarms"],
                        "no_hazard_photos": result["no_hazard_photos"],
                        "no_hazard_false_alarm_rate": result["no_hazard_false_alarm_rate"]})
    feasible = [r for r in records if r["no_hazard_photos"] and
                5 * r["no_hazard_false_alarms"] <= r["no_hazard_photos"]]
    chosen = records[-1]
    if not feasible:
        status = "constraint_unmeasurable_no_inner_negatives_abstain"
    elif not any(gold.values()):
        status = "f1_unmeasurable_no_inner_positives_abstain"
    else:
        def preference(r):
            t = r["threshold"]
            return (r["macro_f1"], -r["no_hazard_false_alarm_rate"], t is not None,
                    -abs(round(t - 0.5, 2)) if t is not None else -1, t or 0)
        chosen = max(feasible, key=preference)
        status = ("selected_feasible_numeric_threshold" if chosen["threshold"] is not None else
                  "no_numeric_candidate_meets_limit_abstain" if len(feasible) == 1 else "selected_abstain")
    return {"status": status, "chosen": chosen, "eligible_numeric_thresholds": sum(
        r["threshold"] is not None for r in feasible), "candidates": records}


def get_features(cache, keys, inventory, root, model_root, provenance, *, encoder_type=None, dimension=512):
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            if (archive["ids"].tolist() != keys or
                    any(archive[k].item() != v for k,v in provenance.items())):
                raise ValueError("Pilot feature cache IDs or provenance mismatch")
            features = archive["features"]
        if (features.shape != (len(keys), dimension) or features.dtype != np.float32 or
                not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features,axis=1), 1, atol=1e-4)):
            raise ValueError("Invalid cached CLIP features")
        return dict(zip(keys, features)), "reused"
    encoder = (encoder_type or ImageEncoder)(model_root, verify_vision=True)
    vectors = {}
    with (ZipFile(root) if root.is_file() else nullcontext()) as archive:
        sequence = sorted(keys, key=lambda k: archive.getinfo(inventory[k]["image"]).header_offset) if archive else keys
        for k in sequence:
            item = inventory[k]
            if archive:
                raw = archive.read(item["image"])
            else:
                path = (root / item["image"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Path escapes image root")
                raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(f"Image checksum mismatch: {k}")
            with Image.open(io.BytesIO(raw)) as img:
                vectors[k] = encoder.encode(img, "full_frame")
    if cache:
        save_atomic(cache, lambda stream: np.savez_compressed(stream, ids=np.array(keys),
                    features=np.stack([vectors[k] for k in keys]), **provenance))
    return vectors, "encoded"


def run(audit_path, labels_path, image_root, model_root, output, cache=None):
    audit_hash = file_hash(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    if audit["errors"] or audit["status"] not in ("audit_only", "needs_group_review"):
        raise ValueError("Expected a structurally valid audit with unresolved case groups")
    if file_hash(labels_path) != audit["labels_sha256"]:
        raise ValueError("Label checksum differs from the audit")
    inventory = audit["inventory"]
    buckets = prefix_cohort(inventory)
    keys = sorted(k for bucket in buckets.values() for k in bucket)
    group = {k: p for p, bucket in buckets.items() for k in bucket}
    # Retain local provenance to exclude the exposed incidents from future final holdout.
    report = {
        "scope": "exploratory_leave_one_unverified_sample_id_prefix_out",
        "case_independence_verified": False, "is_official_score": False,
        "limitation": "Identical incidents can cross distinct prefixes, especially with new viewpoints or crops.",
        "holdout_rule": "Exclude exposed_ids and all subsequently identified related incidents from any final holdout.",
        "cohort_rule": "S<number>C<number> IDs; prefixes with 12-40 photos sorted by SHA256('kwater-prefix-pilot-v1'+NUL+prefix); first six",
        "exposed_ids": keys, "prefix_photo_counts": {p: len(v) for p, v in buckets.items()},
        "audit_sha256": audit_hash, "labels_sha256": audit["labels_sha256"],
        "audit_status": audit["status"], "encoder_revision": REVISION,
        "encoder_sha256": {k: FILES[k] for k in ("onnx/vision_model_quantized.onnx","preprocessor_config.json")},
        "framing": "full_frame", "l2": 0.01, "steps": 250,
        "calibration": "nested leave-one-training-prefix-out on each outer fit, pooled internal predictions",
        "folds": [],
    }
    start = time.perf_counter()
    provenance = {"scope": "exploratory_prefix_subset", "revision": REVISION, "framing": "full_frame",
                  "labels_sha256": audit["labels_sha256"], "audit_sha256": audit_hash}
    vectors, source = get_features(cache, keys, inventory, image_root.resolve(), model_root, provenance)
    report["feature_cache"] = {"access": source, "sha256": file_hash(cache) if cache else None}
    gold = {k: frozenset(inventory[k]["codes"]) for k in keys}
    linked = {k: set() for k in keys}
    report["feature_seconds"] = time.perf_counter() - start
    # Conservative cross-fold exclusion. Exclude a train image even if a near
    # dHash is a false match. Distant hashes do not establish independence.
    for n, a in enumerate(keys):
        for b in keys[n+1:]:
            if (inventory[a]["pixels_sha256"] == inventory[b]["pixels_sha256"] or
                    (int(inventory[a]["dhash64"], 16) ^ int(inventory[b]["dhash64"], 16)).bit_count() <= 4):
                linked[a].add(b)
                linked[b].add(a)

    def train_predict(training, validation):
        training = filtered_training(training, validation, linked)
        x = np.stack([vectors[k] for k in training])
        y = np.array([[c in gold[k] for c in CODES] for k in training], dtype=float)
        head = fit(x, y, 0.01, 250)
        return {"prior": np.broadcast_to(head["prior"], (len(validation), len(CODES))),
                "linear_head": probabilities(head, np.stack([vectors[k] for k in validation]))}

    report["fpr20_hypothesis"] = ("Post-hoc development operating hypothesis: constrain inner-CV no-hazard FPR "
                                  "to <=20%, then apply unchanged on the outer prefix. No external FPR guarantee.")
    predictions = {model: {condition: {} for condition in ("fixed_0.5", "inner_calibrated", "inner_fpr20")}
                   for model in ("prior", "linear_head")}
    for prefix, validation in buckets.items():
        outer_train = filtered_training(keys, validation, linked)
        inner_keys = []
        inner_scores = {name: [] for name in predictions}
        for inner_prefix in buckets:
            if inner_prefix == prefix:
                continue
            inner_val = sorted(k for k in outer_train if group[k] == inner_prefix)
            if not inner_val:
                continue
            batch = train_predict(outer_train, inner_val)
            inner_keys.extend(inner_val)
            for name in batch:
                inner_scores[name].append(batch[name])
        inner_gold = {k: gold[k] for k in inner_keys}
        inner_groups = {k: group[k] for k in inner_keys}
        outer = train_predict(outer_train, validation)
        fold = {"prefix": prefix, "evaluation_photos": len(validation), "train_photos": len(outer_train),
                "near_or_exact_train_exclusions": len(keys) - len(validation) - len(outer_train)}
        for name, scores in outer.items():
            inner_matrix = np.vstack(inner_scores[name])
            threshold = calibration(inner_matrix, inner_gold, inner_groups)["selected_threshold"]
            limited = calibrate_fpr20(inner_matrix, inner_gold, inner_groups)
            fixed = decisions(validation, scores, 0.5)
            calibrated = decisions(validation, scores, threshold)
            predictions[name]["fixed_0.5"].update(fixed)
            predictions[name]["inner_calibrated"].update(calibrated)
            limited_out = policy_decisions(validation, scores, limited["chosen"]["threshold"])
            predictions[name]["inner_fpr20"].update(limited_out)
            metrics = evaluate({k: gold[k] for k in validation}, calibrated)
            check = evaluate({k: gold[k] for k in validation}, limited_out)
            rate = check["no_hazard_false_alarm_rate"]
            limited["outer_fpr_limit_met"] = rate <= 0.2 if rate is not None else None
            limited["outer_macro_f1"] = check["macro_f1"]
            limited["outer_false_alarms"] = check["no_hazard_false_alarms"]
            limited["outer_no_hazard_photos"] = check["no_hazard_photos"]
            fold[name] = {"threshold": threshold, "macro_f1": metrics["macro_f1"],
                          "no_hazard_false_alarm_rate": metrics["no_hazard_false_alarm_rate"],
                          "fpr20": limited}
        report["folds"].append(fold)
    report["pooled_out_of_prefix"] = {
        model: {condition: evaluate(gold, pred, group) for condition, pred in by_condition.items()}
        for model, by_condition in predictions.items()}
    report["paired_out_of_prefix_predictions"] = [
        {"sample_id": k, "proxy_prefix": group[k], "gold": sorted(gold[k]),
         **{f"{model}_{condition}": sorted(pred[k])
            for model, by_condition in predictions.items() for condition, pred in by_condition.items()}}
        for k in keys]
    report["total_seconds"] = time.perf_counter() - start
    save_atomic(output, lambda stream: stream.write((json.dumps(report, indent=2, ensure_ascii=False,
                                                                allow_nan=False) + "\n").encode("utf-8")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "labels", "image-root", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="optional aggregate report without photo IDs or row labels")
    parser.add_argument("--feature-cache", type=Path, help="read/write local pilot-only frozen image features")
    parser.add_argument("--exposure", type=Path, help="also reconstruct the original local exposure manifest")
    args = parser.parse_args()
    if args.summary and args.output.resolve() == args.summary.resolve():
        parser.error("summary and diagnostic paths must differ")
    if args.exposure:
        if args.exposure.resolve() in {p.resolve() for p in (args.audit, args.labels, args.output, args.summary,
                                                             args.feature_cache) if p is not None}:
            parser.error("Use a distinct file for the exposure manifest")
        write_exposure(args.audit, args.exposure)
    report = run(args.audit, args.labels, args.image_root, args.model, args.output, args.feature_cache)
    if args.summary:
        save_summary(report, args.output, args.summary)
    print("Diagnostic only. Pooled out-of-prefix results:")
    for model, by_condition in report["pooled_out_of_prefix"].items():
        for condition, result in by_condition.items():
            print(model, condition, "macro F1", result["macro_f1"], "no-hazard FPR", result["no_hazard_false_alarm_rate"])


def save_summary(report, detailed_path, output):
    summary = {k: v for k, v in report.items() if k not in ("exposed_ids", "paired_out_of_prefix_predictions")}
    summary["diagnostic_sha256"] = file_hash(detailed_path)
    save_atomic(output, lambda stream: stream.write((json.dumps(summary, indent=2, ensure_ascii=False,
                                                                allow_nan=False) + "\n").encode("utf-8")))


if __name__ == "__main__":
    main()
