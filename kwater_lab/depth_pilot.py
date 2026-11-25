"""Test full-frame RGB and monocular relative depth only on the exposed 170 photos."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .aihub_ablation import AUDIT, EXPOSURE, FEATURES, REFERENCE, read
from .metrics import CODES, evaluate
from .prefix_pilot import filtered_training
from .train_head import decisions, file_hash, fit, probabilities, save_atomic
from .weight_pilot import pilot_data


EXPORT = "4472b7362082ad9968fee890ca0f1e5aca36b93d"
ORIGINAL = "03876f8651c73a60fe4c2c48294e09fcb6838fcf"
ASSETS = {"onnx/model.onnx": "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
          "preprocessor_config.json": "03576db3c13dd0471fdf5f5e1428befcb95de063fe699879150b293dc9e0a2c6"}
MODEL = Path("G:/CodexData/kwater-rd-20261125/depth-anything-v2-small")
CACHE = Path("data/prefix-pilot/relative-depth-features.npz")
TARGETS = ("fall_height", "struck_by_lifting")
WIDTH = 32
MODES = ("rgb", "brightness", "edge", "depth", "wrong_depth")


def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    origin = f"https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/{EXPORT}/"
    for key, expected in ASSETS.items():
        out = folder / Path(key).name
        if out.is_file() and file_hash(out) == expected:
            continue
        def receive(stream):
            digest = sha256()
            with urlopen(origin + key, timeout=120) as remote:
                while block := remote.read(2**20):
                    stream.write(block)
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError("Depth-model checksum mismatch")
        save_atomic(out, receive)


def descriptor(field):
    resized = Image.fromarray(np.asarray(field, dtype=np.float32)).resize((64, 64), Image.Resampling.BILINEAR)
    values = np.asarray(resized)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite image map")
    lo, hi = np.percentile(values, [2, 98])
    norm = np.clip((values - lo) / max(hi - lo, 1e-8), 0., 1.)
    gradient = np.hypot(*np.gradient(norm))
    medians = np.median(norm.reshape(4, 16, 4, 16), axis=(1, 3))
    edges = np.mean(gradient.reshape(4, 16, 4, 16), axis=(1, 3))
    return np.concatenate([medians.ravel(), edges.ravel()]).astype(np.float32)


class DepthEncoder:
    def __init__(self, folder):
        if any(file_hash(folder / Path(k).name) != value for k, value in ASSETS.items()):
            raise ValueError("Unexpected ONNX model or processor checksum")
        config = read(folder / "preprocessor_config.json")
        self.edge = config["size"]["height"]
        self.mean = np.array(config["image_mean"], dtype=np.float32)
        self.std = np.array(config["image_std"], dtype=np.float32)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        self.net = ort.InferenceSession(str(folder / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])

    def encode(self, source):
        picture = ImageOps.exif_transpose(source).convert("RGB")
        size = (self.edge, self.edge)
        resized = ImageOps.contain(picture, size, method=Image.Resampling.BICUBIC)
        w, h = resized.size
        left, top = (self.edge - w) // 2, (self.edge - h) // 2
        canvas = Image.new("RGB", size, tuple(int(round(float(c) * 255)) for c in self.mean))
        canvas.paste(resized, (left, top))
        pixels = (np.asarray(canvas, dtype=np.float32) / 255. - self.mean) / self.std
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        depth, = self.net.run(["predicted_depth"], {"pixel_values": tensor})
        if depth.shape != (1, self.edge, self.edge):
            raise ValueError(f"Unexpected depth map size {depth.shape}")
        gray = np.asarray(resized.convert("L"), dtype=np.float32)
        return np.stack([descriptor(gray), descriptor(np.hypot(*np.gradient(gray))),
                         descriptor(depth[0, top:top+h, left:left+w])])


def collect(keys, inventory, image_root, model):
    signature = sha256(json.dumps(["518-full-p2p98-grid4-v1", ASSETS, [(k, inventory[k]["sha256"]) for k in keys]],
                                  sort_keys=True).encode()).hexdigest()
    vectors, durations = {}, {}
    if CACHE.is_file():
        with np.load(CACHE, allow_pickle=False) as c:
            if str(c["signature"]) != signature:
                raise ValueError("Cached depth input or method differs")
            seen = c["ids"].tolist()
            existing, times = c["vectors"], c["seconds"]
            if (not set(seen) <= set(keys) or len(set(seen)) != len(seen)
                    or existing.shape != (len(seen), 3, WIDTH) or not np.isfinite(existing).all()
                    or times.shape != (len(seen),) or not np.isfinite(times).all()):
                raise ValueError("Invalid depth feature cache")
            vectors.update(zip(seen, existing))
            durations.update(zip(seen, times.tolist()))
    remaining = [k for k in keys if k not in vectors]
    if remaining:
        encoder = DepthEncoder(model)
        root = image_root.resolve()
        for index, key in enumerate(remaining, 1):
            start = perf_counter()
            p = (root / inventory[key]["image"]).resolve()
            if not p.is_relative_to(root):
                raise ValueError("Image path escaped source root")
            raw = p.read_bytes()
            if sha256(raw).hexdigest() != inventory[key]["sha256"]:
                raise ValueError("Photo SHA-256 differs from the registered audit")
            with Image.open(BytesIO(raw)) as img:
                vectors[key] = encoder.encode(img)
            durations[key] = perf_counter() - start
            if index % 10 == 0 or index == len(remaining):
                saved = sorted(vectors)
                save_atomic(CACHE, lambda f: np.savez_compressed(
                    f, signature=signature, ids=np.array(saved), vectors=np.stack([vectors[k] for k in saved]),
                    seconds=np.asarray([durations[k] for k in saved])))
                print(f"Depth descriptors {len(saved)}/{len(keys)}", flush=True)
    return vectors, list(durations.values())


def unrelated(recipients, donors, groups, linked, salt):
    """Find deterministic, distinct training donors in other nonduplicate proxy groups."""
    assigned = {}
    candidates = {}
    for key in recipients:
        allowed = [s for s in donors if groups[s] != groups[key] and s not in linked[key]]
        candidates[key] = sorted(allowed, key=lambda s: sha256(f"{salt}|{key}|{s}".encode()).digest())
    def match(key, seen):
        for s in candidates[key]:
            if s in seen:
                continue
            seen.add(s)
            if s not in assigned or match(assigned[s], seen):
                assigned[s] = key
                return True
        return False
    for key in sorted(recipients):
        if not match(key, set()):
            raise ValueError("Could not assign unrelated training depth maps")
    return {key: s for s, key in assigned.items()}


def condition_metrics(gold, pred, base, groups):
    out = evaluate(gold, pred, groups)
    clean = [k for k, y in gold.items() if not y]
    stats = {}
    for code in TARGETS:
        pos = [k for k, y in gold.items() if code in y]
        neg = [k for k, y in gold.items() if code not in y]
        stats[code] = {"positives": len(pos), "negatives": len(neg), "clean_photos": len(clean),
                       "code_fp_on_clean": sum(code in pred[k] for k in clean),
                       "tp_gained": sum(code in pred[k] and code not in base[k] for k in pos),
                       "tp_lost": sum(code not in pred[k] and code in base[k] for k in pos),
                       "fp_added": sum(code in pred[k] and code not in base[k] for k in neg),
                       "fp_removed": sum(code not in pred[k] and code in base[k] for k in neg)}
    out["focus"] = stats
    out["macro_f1_fixed11"] = sum(v["f1"] or 0 for v in out["per_code"].values()) / len(CODES)
    return out


def predict_fold(method, train, val, rgb, maps, labels, wrong):
    def matrix(ids):
        image = np.stack([rgb[k] for k in ids])
        if method == "rgb":
            return image
        index = {"brightness": 0, "edge": 1, "depth": 2, "wrong_depth": 2}[method]
        extra = np.stack([maps[wrong[k] if method == "wrong_depth" else k][index] for k in ids])
        return np.column_stack([image, extra])
    scores = probabilities(fit(matrix(train), labels, .01, 250), matrix(val))
    return decisions(val, scores, .5)


def run(model, image_root):
    buckets, feat, gold, linked = pilot_data(AUDIT, EXPOSURE, FEATURES, read(REFERENCE))
    keys = sorted(gold)
    inventory = read(AUDIT)["inventory"]
    maps, times = collect(keys, inventory, image_root, model)
    groups = {s: g for g, keys in buckets.items() for s in keys}
    predictions = {m: {} for m in MODES}
    fold_rows = []
    for group, val in buckets.items():
        train = filtered_training(gold, val, linked)
        wrong = {**unrelated(train, train, groups, linked, group + "|train"),
                 **unrelated(val, train, groups, linked, group + "|test")}
        y = np.array([[c in gold[k] for c in CODES] for k in train], dtype=float)
        actual = {k: gold[k] for k in val}
        fold_score = {}
        for method in MODES:
            pred = predict_fold(method, train, val, feat, maps, y, wrong)
            predictions[method].update(pred)
            fold_score[method] = evaluate(actual, pred)["macro_f1"]
        fold_rows.append({"photos": len(val), "training": len(train), "macro_f1": fold_score})
    output = {m: condition_metrics(gold, predictions[m], predictions["rgb"], groups) for m in MODES}
    previous = read(Path("results/external_flood.json"))["official_pilot"]["official_only"]
    if any(output["rgb"][k] != v for k, v in previous.items()):
        raise ValueError("RGB baseline differs from the previous matched 170-photo run")
    return {"scope": "Previously exposed photos, six unverified ID-prefix groups, local CPU, no independent contest score",
            "original_model_revision": ORIGINAL, "onnx_export_revision": EXPORT, "sha256": ASSETS,
            "feature_cache_sha256": file_hash(CACHE), "official_rgb_feature_sha256": file_hash(FEATURES),
            "photos": len(keys), "feature_dimensions": {"rgb": 512, "augmented": 544},
            "encoding_seconds_including_photo_io": {"sum": sum(times), "p50": float(np.median(times)),
                                                    "p95": float(np.percentile(times, 95))},
            "training": {"l2": .01, "steps": 250, "threshold": .5, "max_codes": 3},
            "matched_control": "Every substituted depth map came from a nonduplicate training photo in another prefix; unique donors within each train/test assignment.",
            "folds": fold_rows, "metrics": output}


def control_sensitivity(model, image_root):
    buckets, rgb, gold, linked = pilot_data(AUDIT, EXPOSURE, FEATURES, read(REFERENCE))
    original = read(Path("results/depth_pilot.json"))
    if file_hash(CACHE) != original["feature_cache_sha256"]:
        raise ValueError("Repeat controls require the same completed depth cache")
    maps, _ = collect(sorted(gold), read(AUDIT)["inventory"], image_root, model)
    groups = {k: g for g, ids in buckets.items() for k in ids}
    folds = []
    for group, val in buckets.items():
        train = filtered_training(gold, val, linked)
        y = np.array([[c in gold[k] for c in CODES] for k in train], dtype=float)
        folds.append((group, train, val, y))
    rows = []
    for repeat in range(21):
        pred = {}
        for group, train, val, y in folds:
            salt = group if repeat == 0 else f"{group}|sensitivity-{repeat:02d}"
            wrong = {**unrelated(train, train, groups, linked, salt + "|train"),
                     **unrelated(val, train, groups, linked, salt + "|test")}
            pred.update(predict_fold("wrong_depth", train, val, rgb, maps, y, wrong))
        metrics = evaluate(gold, pred)
        if repeat == 0:
            if metrics["macro_f1"] != original["metrics"]["wrong_depth"]["macro_f1"]:
                raise ValueError("Primary wrong-depth result did not reproduce")
            continue
        rows.append({"permutation": repeat, "macro_f1": metrics["macro_f1"],
                     "no_hazard_false_alarms": metrics["no_hazard_false_alarms"],
                     **{t + "_f1": metrics["per_code"][t]["f1"] for t in TARGETS}})
    actual = original["metrics"]["depth"]
    thresholds = {"macro_f1": actual["macro_f1"], **{t + "_f1": actual["per_code"][t]["f1"] for t in TARGETS}}
    summary = {}
    for metric, threshold in thresholds.items():
        values = [r[metric] for r in rows]
        summary[metric] = {"matched_depth": threshold, "min": min(values), "median": float(np.median(values)),
                           "max": max(values), "wrong_depth_at_least_matched": sum(v >= threshold for v in values)}
    return {"scope": "Post-primary sensitivity to 20 fixed wrong-image assignments on the same exposed 170 photos. "
                     "Descriptive counts; each repeat uses the same six unverified prefix groups.",
            "repetitions": len(rows), "official_depth_feature_sha256": original["feature_cache_sha256"],
            "summary": summary, "repeats": rows}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", type=Path, default=MODEL)
    cli.add_argument("--image-root", type=Path, default=Path("data/train"))
    cli.add_argument("--download", action="store_true")
    cli.add_argument("--repeat-controls", action="store_true", help="Run 20 further mismatched-depth assignments")
    args = cli.parse_args()
    if args.download:
        download(args.model)
    if args.repeat_controls:
        result, name = control_sensitivity(args.model, args.image_root), "depth_control_sensitivity.json"
    else:
        result, name = run(args.model, args.image_root), "depth_pilot.json"
    save_atomic(Path("results") / name, lambda f: f.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()))
    print(f"Saved results/{name}", flush=True)
