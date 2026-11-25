"""Compare CLIP B/32 and SigLIP 2 B/16 on the same full-frame exploratory folds."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from statistics import median
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import onnxruntime as ort
from PIL import Image

from .clip_probe import ImageEncoder, session
from .metrics import CODES, evaluate, paired_bootstrap
from .prefix_pilot import calibrate_fpr20, filtered_training, get_features, policy_decisions
from .train_head import calibration, file_hash, fit, probabilities, save_atomic
from .weight_pilot import neighbor_scores, pilot_data


EXPORT = "onnx-community/siglip2-base-patch16-224-ONNX"
REVISION = "ba1f3b0843f24bc5417d38e19c37b287d719b2f4"
FILES = {"onnx/vision_model_quantized.onnx": "5f2b401c1a4fc095702a5d45348e17ad46c4f87064085365b43c6e8eaa5c0070",
         "preprocessor_config.json": "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc"}
VARIANTS = {
    224: (EXPORT, REVISION, FILES),
    384: ("onnx-community/siglip2-base-patch16-384-ONNX", "99f5d40afb5331b81eecee4bb02c360fbb9cdf8a",
          {"onnx/vision_model_quantized.onnx": "9f354d80bbb46b319f46f9e9d0e24ae523e3eb5921aac3d34979d74dbd0db6cd",
           "preprocessor_config.json": "fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90"}),
}
MODELS = ("head", "blend")
POLICIES = ("fixed", "inner_calibrated", "inner_fpr20")


def prepare(folder, resolution=224):
    export, revision, assets = VARIANTS[resolution]
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in assets.items():
        dest = folder / Path(name).name
        if dest.is_file() and file_hash(dest) == expected:
            continue
        url = f"https://huggingface.co/{export}/resolve/{revision}/{name}?download=true"
        def download(stream):
            digest = hashlib.sha256()
            with urlopen(Request(url, headers={"User-Agent": "K-water-research"}), timeout=120) as src:
                while block := src.read(1024**2):
                    stream.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Unexpected model bytes for {name}")
        save_atomic(dest, download)


class SiglipEncoder(ImageEncoder):
    output_name = "pooler_output"

    def __init__(self, folder, *, verify_vision=True, resolution=224):
        _, _, assets = VARIANTS[resolution]
        config = (folder / "preprocessor_config.json").read_bytes()
        raw = (folder / "vision_model_quantized.onnx").read_bytes()
        if verify_vision and (hashlib.sha256(config).hexdigest() != assets["preprocessor_config.json"] or
                              hashlib.sha256(raw).hexdigest() != assets["onnx/vision_model_quantized.onnx"]):
            raise ValueError("SigLIP 2 weights or preprocessing differ from the pinned export")
        cfg = json.loads(config)
        if cfg["size"] != {"height": resolution, "width": resolution}:
            raise ValueError("Unexpected fixed-resolution SigLIP 2 processor")
        self.edge = cfg["size"]["height"]
        self.mean = np.asarray(cfg["image_mean"], dtype=np.float32)
        self.std = np.asarray(cfg["image_std"], dtype=np.float32)
        self.model = session(raw)


def scores(train, valid, features, gold):
    x = np.stack([features[s] for s in train])
    probe = np.stack([features[s] for s in valid])
    y = np.array([[code in gold[s] for code in CODES] for s in train], dtype=float)
    head = probabilities(fit(x, y, .01, 250), probe)
    return {"head": head, "blend": (head + neighbor_scores(x, y, probe)) / 2}


def cross_validate(buckets, features, gold, groups, linked, score_fn=scores):
    collected = {m: {p: {} for p in POLICIES} for m in MODELS}
    folds = {}
    for prefix, valid in buckets.items():
        train = filtered_training(gold, valid, linked)
        outer = score_fn(train, valid, features, gold)
        keys, inner = [], {m: [] for m in MODELS}
        for other_prefix, ids in buckets.items():
            if other_prefix == prefix:
                continue
            dev = sorted(set(ids).intersection(train))
            if not dev:
                continue
            result = score_fn(filtered_training(train, dev, linked), dev, features, gold)
            keys.extend(dev)
            for m in MODELS:
                inner[m].append(result[m])
        inner_gold = {s: gold[s] for s in keys}
        inner_groups = {s: groups[s] for s in keys}
        folds[prefix] = {}
        for m in MODELS:
            matrix = np.vstack(inner[m])
            thresholds = {"fixed": 0.5, "inner_calibrated": calibration(matrix, inner_gold, inner_groups)["selected_threshold"],
                          "inner_fpr20": calibrate_fpr20(matrix, inner_gold, inner_groups)["chosen"]["threshold"]}
            folds[prefix][m] = thresholds
            for policy, t in thresholds.items():
                collected[m][policy].update(policy_decisions(valid, outer[m], t))
    return collected, folds


def timings(ids, inventory, package, clip_folder, sig_folder, original, resolution=224):
    encoders = {"clip": ImageEncoder(clip_folder, verify_vision=True), "siglip2": SiglipEncoder(sig_folder, resolution=resolution)}
    ordered = sorted(ids, key=lambda s: hashlib.sha256(("kwater-encoder-timing/" + s).encode()).digest())[:13]
    latency = {name: [] for name in encoders}
    with ZipFile(package) as archive:
        for i, sid in enumerate(ordered):
            item = inventory[sid]
            raw = archive.read(item["image"])
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Timing sample differs from the audited image")
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                for name in list(encoders)[::1 if i % 2 == 0 else -1]:
                    start = time.perf_counter()
                    vector = encoders[name].encode(image, "full_frame")
                    elapsed = time.perf_counter() - start
                    if not np.allclose(vector, original[name][sid], atol=1e-5):
                        raise ValueError(f"Timing vectors differ from the acknowledged {name} cache")
                    if i:
                        latency[name].append(elapsed)
    return {"photos_after_one_warmup": len(ordered)-1, "batch_size": 1, "intra_op_threads": 4,
            "provider": "CPUExecutionProvider", "scope": "EXIF/RGB, letterbox, normalization and inference on decoded inputs; no image I/O or model startup",
            "alternating_first_encoder": True, "median_seconds": {k: median(v) for k,v in latency.items()},
            "seconds_by_encoder": latency}


def run(root, clip_folder, sig_folder, resolution=224):
    from functools import partial
    export, revision, assets = VARIANTS[resolution]
    audit = root / "results/train_audit_full.json"
    exposure = root / "data/prefix-pilot/exposure.json"
    ref = json.loads((root / "results/prefix_pilot_fpr20.json").read_text(encoding="utf-8"))
    buckets, clip, gold, linked = pilot_data(audit, exposure, root / "data/prefix-pilot/features.npz", ref)
    groups = {s: p for p, ids in buckets.items() for s in ids}
    keys = sorted(gold)
    inventory = json.loads(audit.read_text(encoding="utf-8"))["inventory"]
    package = root / "data/packages/kwater-train.zip"
    suffix = "" if resolution == 224 else f"-{resolution}"
    cache = root / f"data/prefix-pilot/siglip2-base16{suffix}-features.npz"
    provenance = {"revision": revision, "frame": "full_frame_same_bicubic", "audit_sha256": file_hash(audit),
                  "exposure_sha256": file_hash(exposure), "weights_sha256": assets["onnx/vision_model_quantized.onnx"]}
    sig, mode = get_features(cache, keys, inventory, package, sig_folder, provenance,
                             encoder_type=partial(SiglipEncoder, resolution=resolution), dimension=768)
    print(f"SigLIP 2 full-frame vectors: {len(sig)} ({mode})", flush=True)
    encoders = {"clip": clip, "siglip2": sig}
    pooled, predictions, folds = {}, {}, {}
    for name, features in encoders.items():
        pred, folds[name] = cross_validate(buckets, features, gold, groups, linked)
        predictions[name] = pred
        pooled[name] = {}
        for m in MODELS:
            pooled[name][m] = {}
            for policy, values in pred[m].items():
                metrics = evaluate(gold, values, groups)
                if name == "clip" and m == "head":
                    key = "fixed_0.5" if policy == "fixed" else policy
                    if metrics != ref["pooled_out_of_prefix"]["linear_head"][key]:
                        raise ValueError(f"CLIP control failed to reproduce {key}")
                metrics["hazard_photos_with_no_correct_code"] = sum(bool(g) and g.isdisjoint(values[s]) for s,g in gold.items())
                pooled[name][m][policy] = metrics
    paired = {m: {p: paired_bootstrap(gold, predictions["clip"][m][p], predictions["siglip2"][m][p], groups)
                  for p in POLICIES} for m in MODELS}
    return {"protocol": "docs/vlm-model-selection.md", "scope": "170 repeatedly exposed images, six unverified ID prefixes, no independent public-score estimate",
            "photos": len(gold), "clip_control_reproduced": True, "siglip2_export": export, "origin": f"google/siglip2-base-patch16-{resolution}",
            "revision": revision, "asset_digests": assets, "provenance": provenance, "cache_sha256": file_hash(cache),
            "onnxruntime": ort.__version__, "thresholds": folds, "pooled": pooled, "paired_siglip2_minus_clip": paired,
            "timing": timings(keys, inventory, package, clip_folder, sig_folder, encoders, resolution)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clip", type=Path, default=Path("G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32"))
    parser.add_argument("--resolution", type=int, choices=VARIANTS, default=224)
    parser.add_argument("--siglip", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true", help="Download and verify SigLIP 2 before evaluating")
    args = parser.parse_args()
    args.siglip = args.siglip or args.root / f"data/encoders/siglip2-base-p16-{args.resolution}"
    suffix = "" if args.resolution == 224 else f"_{args.resolution}"
    args.output = args.output or args.root / f"results/encoder_pilot{suffix}.json"
    if args.prepare:
        prepare(args.siglip, args.resolution)
    report = run(args.root.resolve(), args.clip, args.siglip, args.resolution)
    save_atomic(args.output, lambda f: f.write((json.dumps(report, indent=2, allow_nan=False)+"\n").encode()))
    for name, models in report["pooled"].items():
        s = models["head"]["inner_calibrated"]
        print(name, s["macro_f1"], s["no_hazard_false_alarms"], s["hazard_photos_with_no_correct_code"])
