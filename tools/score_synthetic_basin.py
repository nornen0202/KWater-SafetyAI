"""Measure the accepted fusion model on two artificial basin views, without training."""

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "submission/fusion"
sys.path.insert(0, str(MODEL))
from main import RiskEncoder


def measure():
    archive = ROOT / "data/submissions/kwater-fusion-20261125.zip"
    expected = json.loads((ROOT / "results/public_submission_002.json").read_bytes())["sha256"]
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != expected:
            raise ValueError("Competition archive differs from the accepted receipt")
    with ZipFile(archive) as submitted:
        for name in ("main.py", "head.npz", "clip.onnx", "sig.onnx", "clip.json", "sig.json"):
            with submitted.open(name) as packed, (MODEL / name).open("rb") as local:
                if hashlib.file_digest(packed, "sha256").digest() != hashlib.file_digest(local, "sha256").digest():
                    raise ValueError(f"Fusion artifact differs: {name}")
    with np.load(MODEL / "head.npz", allow_pickle=False) as heads:
        encoders = [RiskEncoder("clip", "image_embeds", heads), RiskEncoder("sig", "pooler_output", heads)]
        codes = heads["codes"].tolist()
        threshold = float(heads["threshold"].item())
    result = {}
    for state in ("open", "railed"):
        path = ROOT / f"reports/figures/synthetic-basin-{state}.png"
        with Image.open(path) as photo:
            rgb = photo.convert("RGB")
        scores = (encoders[0].scores(rgb) + encoders[1].scores(rgb)) / 2
        result[state] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "scores": dict(zip(codes, scores.tolist(), strict=True)),
                         "emitted": [codes[i] for i in np.argsort(-scores, kind="stable")[:3]
                                     if scores[i] >= threshold]}
    return {"scope": "One synthetic scene, two PNG views. Exploratory sensitivity probe.",
            "archive_sha256": expected, "threshold": threshold, "views": result}


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
