"""Compare one pass, two tied passes and two untied passes on full-frame safety photos."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .clip_probe import unit_rows
from .encoder_pilot import VARIANTS
from .loop_head import fit_head, forward
from .metrics import CODES, evaluate
from .prefix_pilot import filtered_training
from .train_head import decisions, file_hash, fit, probabilities, save_atomic, sigmoid
from .weight_pilot import pilot_data


WIDTH, STEPS, RATE, THRESHOLD = 24, 32, 0.003, 0.30
PROJECTION_SEED, SEEDS = 38401, (42, 137)
PATCH_WEIGHT = 0.25
ORDERS = {"one": (0,), "tied_two": (0, 0), "untied_two": (0, 1)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def patch_grid(root, keys, inventory, reference, pooled):
    directory = root / "data/encoders/siglip2-base-p16-384"
    _, revision, assets = VARIANTS[384]
    weights = directory / "vision_model_quantized.onnx"
    config_path = directory / "preprocessor_config.json"
    if (revision != reference["revision"] or file_hash(weights) != assets["onnx/vision_model_quantized.onnx"]
            or file_hash(config_path) != assets["preprocessor_config.json"]):
        raise ValueError("Encoder or preprocessing changed since the archived benchmark")
    config = read_json(config_path)
    mean, std = (np.asarray(config[k], dtype=np.float32) for k in ("image_mean", "image_std"))
    fill = tuple(int(round(float(v) * 255)) for v in mean)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(weights), sess_options=opts, providers=["CPUExecutionProvider"])
    generator = np.random.default_rng(PROJECTION_SEED)
    projection = generator.standard_normal((768, WIDTH), dtype=np.float32) / np.float32(np.sqrt(768))
    grids = []
    source = (root / "data/train").resolve()
    for sid in keys:
        row = inventory[sid]
        path = (source / row["image"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Image outside the training directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Image differs from the audit")
        with Image.open(io.BytesIO(raw)) as image:
            frame = ImageOps.pad(ImageOps.exif_transpose(image).convert("RGB"), (384, 384),
                                 method=Image.Resampling.BICUBIC, color=fill)
            pixels = (np.asarray(frame, dtype=np.float32) / 255.0 - mean) / std
        hidden, full = sess.run(["last_hidden_state", "pooler_output"],
                                {"pixel_values": np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])})
        if hidden.shape != (1, 576, 768) or not np.allclose(unit_rows(full)[0], pooled[sid], atol=1e-5):
            raise ValueError("Whole-frame encoder did not reproduce the registered pooled vector")
        # Each token covers a 4x4 group of adjacent patch tokens from the same encoder pass.
        grid = hidden.reshape(6, 4, 6, 4, 768).mean(axis=(1, 3)).reshape(36, 768) @ projection
        grid -= grid.mean(axis=1, keepdims=True)
        grid /= np.sqrt(np.mean(grid * grid, axis=1, keepdims=True) + 1e-6)
        grids.append(grid)
    matrix = np.stack(grids)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite patch features")
    return matrix


def summarize(keys, scores, gold):
    pred = decisions(keys, scores, THRESHOLD)
    metric = evaluate(gold, pred)
    raw = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    top = {s: {CODES[j] for j in indices} for s, indices in zip(keys, raw, strict=True)}
    metric["top3_hits"] = sum(len(gold[s] & top[s]) for s in keys)
    metric["top3_hits_by_code"] = {c: sum(c in gold[s] and c in top[s] for s in keys) for c in CODES}
    metric["hazard_photos_with_no_correct_code"] = sum(bool(gold[s]) and gold[s].isdisjoint(pred[s]) for s in keys)
    return metric


def prevalence_control(keys, targets, folds, gold):
    scores = np.empty_like(targets)
    for train, valid in folds:
        scores[valid] = targets[train].mean(axis=0)
    return summarize(keys, scores, gold)


def run(root):
    audit, exposure = root / "results/train_audit_full.json", root / "data/prefix-pilot/exposure.json"
    reference = read_json(root / "results/encoder_pilot_384.json")
    cache = root / "data/prefix-pilot/siglip2-base16-384-features.npz"
    digests = {"audit_sha256": file_hash(audit), "exposure_sha256": file_hash(exposure)}
    if any(v != reference["provenance"][k] for k, v in digests.items()) or file_hash(cache) != reference["cache_sha256"]:
        raise ValueError("Pilot cohort or reference features changed")
    buckets, _, gold, linked = pilot_data(audit, exposure, root / "data/prefix-pilot/features.npz",
                                         read_json(root / "results/prefix_pilot_fpr20.json"))
    keys = sorted(gold)
    with np.load(cache, allow_pickle=False) as archive:
        if archive["ids"].tolist() != keys:
            raise ValueError("Expected the registered full-frame cache in pilot ID order")
        full = archive["features"]
    print(f"Extracting spatial features from {len(keys)} exposed photos on one CPU thread", flush=True)
    started = time.perf_counter()
    grid = patch_grid(root, keys, read_json(audit)["inventory"], reference, dict(zip(keys, full, strict=True)))
    extraction_seconds = time.perf_counter() - started
    targets = np.array([[c in gold[s] for c in CODES] for s in keys], dtype=np.float32)
    locations = {key: i for i, key in enumerate(keys)}
    folds = [(np.array([locations[k] for k in filtered_training(keys, valid, linked)]),
              np.array([locations[k] for k in valid])) for valid in buckets.values()]
    if sorted(int(i) for _, valid in folds for i in valid) != list(range(len(keys))):
        raise ValueError("Outer test folds must partition the pilot exactly")
    baseline = np.empty_like(targets, dtype=np.float64)
    for train, valid in folds:
        baseline[valid] = probabilities(fit(full[train], targets[train], .01, 250), full[valid])
    archived = reference["pooled"]["siglip2"]["head"]["fixed"]["macro_f1"]
    if abs(evaluate(gold, decisions(keys, baseline, 0.5))["macro_f1"] - archived) > 1e-12:
        raise ValueError("The published fixed-threshold linear reference did not reproduce")
    report = {"scope": "Exploratory 170 previously exposed photos, six unverified ID prefixes; code classification only.",
              "photos": len(keys), "gold_instances": int(targets.sum()), "projection_seed": PROJECTION_SEED,
              "grid_shape": list(grid.shape), "steps": STEPS, "learning_rate": RATE, "fixed_threshold": THRESHOLD,
              "encoder": reference["origin"], "revision": reference["revision"], "weights_sha256": reference["provenance"]["weights_sha256"],
              **digests, "pooled_cache_sha256": reference["cache_sha256"], "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
              "numpy": np.__version__, "onnxruntime": ort.__version__, "encoder_runtime": "CPU, one thread",
              "extraction_seconds_including_model_load_and_checks": extraction_seconds,
              "linear_full_pooled": summarize(keys, baseline, gold),
              "posthoc_train_prevalence": prevalence_control(keys, targets, folds, gold),
              "posthoc_mix_fraction_from_patch_head": PATCH_WEIGHT,
              "seeds": {}, "trainable_parameters": {}}
    for seed in SEEDS:
        curves = {name: np.empty_like(targets) for name in (*ORDERS, "tied_two_read_after_one")}
        timings = {name: [] for name in ORDERS}
        for train, valid in folds:
            x, y = grid[train], targets[train]
            for name, order in ORDERS.items():
                params = fit_head(x, y, order, seed, STEPS, RATE)
                report["trainable_parameters"][name] = sum(w.size for w in params.values())
                curves[name][valid] = sigmoid(forward(params, grid[valid], order)[0]) * y.any(axis=0)
                if name == "tied_two":
                    curves["tied_two_read_after_one"][valid] = sigmoid(forward(params, grid[valid], (0,))[0]) * y.any(axis=0)
                forward(params, grid[valid[:1]], order)
                for i in valid[:8]:
                    start = time.perf_counter()
                    forward(params, grid[i:i + 1], order)
                    timings[name].append((time.perf_counter() - start) * 1000)
        metrics = {name: summarize(keys, curve, gold) for name, curve in curves.items()}
        anchored = {name: summarize(keys, (1 - PATCH_WEIGHT) * baseline + PATCH_WEIGHT * curves[name], gold)
                    for name in ORDERS}
        report["seeds"][str(seed)] = {"metrics": metrics, "posthoc_mix_with_full_frame": anchored, "warm_head_only_cpu_ms": {
            name: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "samples": len(v)} for name, v in timings.items()}}
        print(f"seed {seed}: " + str({k: (round(v['macro_f1'], 4), v['top3_hits']) for k, v in metrics.items()}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = run(args.root)
    save_atomic(args.root / "results/loop_pilot.json", lambda f: f.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode()))
