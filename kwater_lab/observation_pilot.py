"""Training-only observations and checked visual caches for the exposed comparison cohort."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .metrics import CODES
from .train_head import file_hash, save_atomic
from .weight_pilot import pilot_data


@dataclass
class Cohort:
    buckets: dict
    clip: dict
    siglip: dict
    gold: dict
    linked: dict
    observations: dict
    hashes: dict


def load_cohort(root):
    audit = root / "results/train_audit_full.json"
    exposure = root / "data/prefix-pilot/exposure.json"
    prefix = root / "data/prefix-pilot"
    labels = root / "data/train/train.jsonl"
    cache = prefix / "siglip2-base16-384-features.npz"
    reference = json.loads((root / "results/encoder_pilot_384.json").read_text(encoding="utf-8"))
    hashes = {"labels": file_hash(labels), "audit": file_hash(audit), "exposure": file_hash(exposure),
              "siglip_cache": file_hash(cache), "clip_cache": file_hash(prefix / "features.npz")}
    if (hashes["siglip_cache"] != reference["cache_sha256"] or hashes["audit"] != reference["provenance"]["audit_sha256"]
            or hashes["exposure"] != reference["provenance"]["exposure_sha256"]
            or hashes["labels"] != json.loads(exposure.read_text(encoding="utf-8"))["labels_sha256"]):
        raise ValueError("The pilot inputs differ from their registered hashes")
    buckets, clip, gold, linked = pilot_data(audit, exposure, prefix / "features.npz",
        json.loads((root / "results/prefix_pilot_fpr20.json").read_text(encoding="utf-8")))
    with np.load(cache, allow_pickle=False) as archive:
        if archive["ids"].tolist() != sorted(gold):
            raise ValueError("Unexpected SigLIP sample IDs")
        sig = dict(zip(archive["ids"].tolist(), archive["features"], strict=True))
    observations = {}
    with labels.open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            sid = row["sample_id"]
            if sid not in gold:
                continue
            target = row["target"]
            if set(target["hazards"]) != gold[sid] or sid in observations:
                raise ValueError("Pilot supervision does not match the audit")
            observations[sid] = {c: h["evidence"] for c, h in target["hazards"].items()} or {"no_visible_hazard": target["scene_evidence"]}
    if set(gold) != set(observations):
        raise ValueError("Missing exposure observations")
    return Cohort(buckets, clip, sig, gold, linked, observations, hashes)


def label_matrix(keys, gold):
    return np.asarray([[c in gold[s] for c in CODES] for s in keys], dtype=float)


def ridge_projection(x, y, query, ridge=0.25):
    """Return out-of-sample predictions and exact leave-one-out training predictions."""
    x = np.asarray(x, dtype=np.float64)
    gram = x @ x.T + 1.0
    inverse = np.linalg.solve(gram + ridge * np.eye(len(x)), np.eye(len(x)))
    weights = inverse @ y
    return (query @ x.T + 1.0) @ weights, y - weights / np.diag(inverse)[:, None]


def write_aggregate(path, result):
    save_atomic(Path(path), lambda f: f.write((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()))
