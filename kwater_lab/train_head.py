"""Fit eleven sigmoid classifiers on audited frozen image features using NumPy.

python -m kwater_lab.train_head --audit audit.json --features features.npz \
    --model head.npz --report head-dev.json

Both the featureless training-prevalence baseline and the fitted head are
evaluated at 0.5 and at a development-selected common threshold. The saved head
is a code-ranking component. It does not generate Korean evidence or citations.
"""

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .metrics import CODES, evaluate, unique_keys


THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_data(audit_path, feature_path):
    raw = audit_path.read_bytes()
    audit = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
    if audit["status"] != "ready" or set(audit["assignments"]) != set(audit["inventory"]):
        raise ValueError("A complete ready audit is required")
    groups = {}
    expected = {"train": set(), "dev": set()}
    for sid, row in audit["assignments"].items():
        group, split = row["group_id"], row["split"]
        if not isinstance(group, str) or not group.strip() or split not in ("train", "dev", "holdout"):
            raise ValueError(f"Invalid assignment for {sid}")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Incident group crosses splits: {group}")
        if split != "holdout":
            expected[split].add(sid)
    if not all(expected.values()):
        raise ValueError("Both training and development photos are required")
    audit_hash = hashlib.sha256(raw).hexdigest()
    label_hash = audit["labels_sha256"]
    if not isinstance(label_hash, str) or not re.fullmatch("[0-9a-f]{64}", label_hash):
        raise ValueError("Invalid audit label SHA-256")
    feature_bytes = feature_path.read_bytes()
    with np.load(io.BytesIO(feature_bytes), allow_pickle=False) as archive:
        if archive["audit_sha256"].item() != audit_hash or archive["labels_sha256"].item() != label_hash:
            raise ValueError("Feature archive and audit hashes differ")
        ids = archive["ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("Expected a vector of string sample IDs")
        ids = ids.tolist()
        if len(ids) != len(set(ids)) or set(ids) != expected["train"] | expected["dev"]:
            raise ValueError("Features must cover exactly the train/dev IDs once; check for holdout or missing IDs")
        matrix = archive["features"]
        if matrix.dtype != np.float32 or matrix.shape != (len(ids), 512) or not np.isfinite(matrix).all():
            raise ValueError("Expected finite float32 features of shape (N, 512)")
        metadata = {"revision": archive["revision"].item(), "framing": archive["framing"].item(),
                    "labels_sha256": label_hash, "audit_sha256": audit_hash,
                    "feature_file_sha256": hashlib.sha256(feature_bytes).hexdigest()}
    if not all(isinstance(value, str) and value.strip() for value in metadata.values()):
        raise ValueError("Missing feature provenance")
    position = {key: i for i, key in enumerate(ids)}
    data = {}
    for split, subset in expected.items():
        keys = sorted(subset)
        labels = {}
        for key in keys:
            codes = audit["inventory"][key]["codes"]
            if (not isinstance(codes, list) or len(codes) > 3 or any(c not in CODES for c in codes)
                    or len(codes) != len(set(codes))):
                raise ValueError(f"Invalid hazard codes in audit: {key}")
            labels[key] = frozenset(codes)
        data[split] = (matrix[[position[k] for k in keys]], labels,
                       {k: audit["assignments"][k]["group_id"] for k in keys})
    return data, metadata


def sigmoid(logits):
    return np.exp(-np.logaddexp(0.0, -logits))


def fit(x, targets, l2, steps):
    """Full-batch accelerated gradient on mean binary cross entropy plus L2."""
    if steps < 1 or not np.isfinite(l2) or l2 <= 0:
        raise ValueError("steps and l2 must be positive")
    x = x.astype(np.float64)
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - center) / scale
    x = np.column_stack([x, np.ones(len(x))])
    prior = targets.mean(axis=0)
    active = (prior > 0) & (prior < 1)
    beta = np.zeros((x.shape[1], len(CODES)))
    if active.any():
        y = targets[:, active]
        current = beta[:, active].copy()
        current[-1] = np.log(prior[active] / (1 - prior[active]))
        momentum = current.copy()
        # Largest row sum bounds the spectral norm of the covariance matrix.
        # For logistic loss the curvature is at most 1/4 of that bound.
        curvature_bound = np.abs((x.T @ x) / len(x)).sum(axis=1).max() / 4 + l2
        for step in range(steps):
            gradient = x.T @ (sigmoid(x @ momentum) - y) / len(x)
            gradient[:-1] += l2 * momentum[:-1]
            updated = momentum - gradient / curvature_bound
            momentum = updated + step / (step + 3) * (updated - current)
            current = updated
        beta[:, active] = current
    return {"center": center, "scale": scale, "weight": beta[:-1], "bias": beta[-1], "prior": prior}


def probabilities(model, features):
    """Predict from fit() output or an np.load()'ed head.npz of the same encoder and framing."""
    scores = sigmoid(((features - model["center"]) / model["scale"]) @ model["weight"] + model["bias"])
    prior = model["prior"]
    constant = (prior == 0) | (prior == 1)
    scores[:, constant] = prior[constant]
    return scores


def decisions(keys, scores, threshold):
    # Stable sort breaks equal probabilities in the organizer's published code order.
    top = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    return {key: frozenset(CODES[j] for j in top[i] if scores[i, j] >= threshold)
            for i, key in enumerate(keys)}


def calibration(scores, gold, groups):
    grid = {t: evaluate(gold, decisions(gold, scores, t), groups) for t in THRESHOLDS}
    chosen = 0.5
    status = "fixed_0.5_no_positive_dev_codes"
    if any(gold.values()):
        def preference(t):
            score = grid[t]
            return (score["macro_f1"], -(score["no_hazard_false_alarm_rate"] or 0.0),
                    -abs(round(t - 0.5, 2)), t)
        chosen = max(THRESHOLDS, key=preference)
        status = "selected_on_dev"
    return {
        "selection": status, "selected_threshold": chosen,
        "grid": [{"threshold": t, **{k: result[k] for k in
                  ("macro_f1", "no_hazard_false_alarm_rate", "hazard_predicted_empty_rate")}}
                 for t, result in grid.items()],
        "fixed_0.5": grid[0.5], "selected": grid[chosen],
    }


def save_atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(audit, features, model_path, report_path, l2=0.01, steps=250):
    if len({p.resolve() for p in (audit, features, model_path, report_path)}) != 4:
        raise ValueError("Input and output paths must be distinct")
    data, metadata = checked_data(audit, features)
    train_x, train_gold, train_groups = data["train"]
    dev_x, dev_gold, dev_groups = data["dev"]
    target = np.array([[code in codes for code in CODES] for codes in train_gold.values()], dtype=float)
    model = fit(train_x, target, l2, steps)
    prior = np.broadcast_to(model["prior"], (len(dev_gold), len(CODES)))
    baseline = calibration(prior, dev_gold, dev_groups)
    candidate = calibration(probabilities(model, dev_x), dev_gold, dev_groups)
    train_support = {c: {"photos": sum(c in values for values in train_gold.values()),
                         "groups": len({train_groups[k] for k in train_gold if c in train_gold[k]})}
                     for c in CODES}
    report = {
        "evaluation_split": "dev", "holdout_images_read": 0,
        "training_photos": len(train_gold), "training_groups": len(set(train_groups.values())),
        "development_photos": len(dev_gold), "development_groups": len(set(dev_groups.values())),
        "train_support_per_code": train_support, "feature_source": metadata,
        "feature_file_sha256": metadata["feature_file_sha256"], "numpy_version": np.__version__,
        "model": {"objective": "per-code mean BCE + L2/2 on weights; train-only standardization",
                  "optimizer": "deterministic full-batch accelerated gradient", "l2": l2, "steps": steps,
                  "constant_training_labels": "return empirical probability 0 or 1"},
        "selection_rule": "maximize dev Macro F1, minimize no-hazard FPR, nearest 0.5, then higher threshold",
        "selection_scope": "Selected thresholds and scores use the same dev photos. Holdout performance is unknown.",
        "prior_baseline": baseline, "linear_head": candidate,
    }
    save_atomic(model_path, lambda stream: np.savez_compressed(
        stream, **model, **metadata, code_order=np.array(CODES), threshold=candidate["selected_threshold"],
        l2=l2, steps=steps))
    report["model_file_sha256"] = file_hash(model_path)
    save_atomic(report_path, lambda stream: stream.write(
        (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("audit", "features", "model", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    try:
        report = train(args.audit, args.features, args.model, args.report, args.l2, args.steps)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(f"dev photos={report['development_photos']}, chosen threshold={report['linear_head']['selected_threshold']}, "
          f"macro F1={report['linear_head']['selected']['macro_f1']}")


if __name__ == "__main__":
    main()
