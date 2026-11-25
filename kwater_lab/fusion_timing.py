"""Time the full CLIP-224 + SigLIP2-384 risk classifier on the registered CPU timing photos."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort
from PIL import Image

from .adapter_scale_pilot import write_json
from .clip_probe import ImageEncoder
from .encoder_pilot import SiglipEncoder
from .metrics import CODES
from .prefix_pilot import filtered_training, policy_decisions
from .train_head import file_hash, fit, probabilities
from .weight_pilot import pilot_data


def predict(vectors, heads, sid, threshold):
    first, second = (probabilities(head, vector[None, :]) for vector, head in zip(vectors, heads, strict=True))
    scores = (first + second) / 2
    return scores, policy_decisions([sid], scores, threshold)[sid]


def benchmark(root, clip_folder):
    result = root / "results/encoder_fusion.json"
    source = root / "results/encoder_pilot_384.json"
    fusion = json.loads(result.read_text(encoding="utf-8"))
    ref = json.loads(source.read_text(encoding="utf-8"))
    audit = root / "results/train_audit_full.json"
    exposure = root / "data/prefix-pilot/exposure.json"
    sig_cache = root / "data/prefix-pilot/siglip2-base16-384-features.npz"
    if (file_hash(source) != fusion["source_sha256"] or file_hash(sig_cache) != ref["cache_sha256"]
            or file_hash(exposure) != ref["provenance"]["exposure_sha256"]
            or file_hash(audit) != ref["provenance"]["audit_sha256"]):
        raise ValueError("Encoder benchmark inputs differ from the registered comparison")
    control = json.loads((root / "results/prefix_pilot_fpr20.json").read_text(encoding="utf-8"))
    folds, clip, gold, linked = pilot_data(audit, exposure, root / "data/prefix-pilot/features.npz", control)
    with np.load(sig_cache, allow_pickle=False) as cache:
        sig = dict(zip(cache["ids"].tolist(), cache["features"], strict=True))
    ordered = sorted(gold, key=lambda key: hashlib.sha256(("kwater-encoder-timing/" + key).encode()).digest())[:13]
    inventory = json.loads(audit.read_text(encoding="utf-8"))["inventory"]
    heads, thresholds = {}, {}
    for prefix, valid in folds.items():
        train = filtered_training(gold, valid, linked)
        labels = np.array([[code in gold[sid] for code in CODES] for sid in train], dtype=float)
        pair = [fit(np.stack([features[sid] for sid in train]), labels, .01, 250) for features in (clip, sig)]
        for sid in set(valid).intersection(ordered):
            heads[sid] = pair
            thresholds[sid] = fusion["thresholds"][prefix]["head"]["inner_calibrated"]
    with ZipFile(root / "data/packages/kwater-train.zip") as archive:
        images = {sid: archive.read(inventory[sid]["image"]) for sid in ordered}
    if any(hashlib.sha256(raw).hexdigest() != inventory[sid]["sha256"] for sid, raw in images.items()):
        raise ValueError("The timing images differ from the audit")
    sig_folder = root / "data/encoders/siglip2-base-p16-384"
    start = perf_counter()
    encoders = ImageEncoder(clip_folder, verify_vision=True), SiglipEncoder(sig_folder, resolution=384)
    load_seconds = perf_counter() - start
    times = []
    for index, sid in enumerate([ordered[0]] + ordered[1:] * 3):
        start = perf_counter()
        with Image.open(io.BytesIO(images[sid])) as image:
            image.load()
            vectors = [encoder.encode(image, "full_frame") for encoder in encoders]
        output, codes = predict(vectors, heads[sid], sid, thresholds[sid])
        elapsed = perf_counter() - start
        reference, expected = predict((clip[sid], sig[sid]), heads[sid], sid, thresholds[sid])
        if not np.allclose(output, reference, rtol=0, atol=1e-5) or codes != expected:
            raise ValueError("Live encoder scores or thresholded decisions differ from feature replay")
        if index:
            times.append(elapsed)
    return {"scope": "Local CPU from in-memory compressed photo bytes to 0-3 hazard codes, sequential encoders. No disk I/O, model startup, VLM or law retrieval in per-call times.",
            "source_sha256": file_hash(result), "cpu": platform.processor(), "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider", "onnx_intra_op_threads": 4, "batch_size": 1,
            "weight_bytes": sum((folder / "vision_model_quantized.onnx").stat().st_size for folder in (clip_folder, sig_folder)),
            "weight_verification_and_session_load_seconds": load_seconds, "warmup_photos": 1,
            "unique_timed_photos": len(ordered)-1, "calls": len(times), "all_decisions_match_cache": True,
            "seconds": times, "median_seconds": float(np.median(times)), "p90_seconds": float(np.percentile(times, 90))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clip", type=Path, default=Path("G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32"))
    args = parser.parse_args()
    report = benchmark(args.root, args.clip)
    write_json(args.root / "results/fusion_timing.json", report)
    print(f"{report['calls']} calls, median {report['median_seconds']:.3f}s, p90 {report['p90_seconds']:.3f}s")
