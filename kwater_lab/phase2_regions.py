"""Local, fixed-panel grounding, box-prompted segmentation and relative depth."""

import argparse
from collections import Counter
import time

import numpy as np
from PIL import Image, ImageOps

from .phase2_paths import file_stem
from .phase2_panel import (
    PROJECT,
    RUN,
    file_hash,
    order,
    read,
    registry,
    write,
)


ROOT = RUN / "regions"
ROLES = (
    "person",
    "suspended load",
    "crane hook",
    "guardrail",
    "floor opening",
    "elevated edge",
)
TEXT = ". ".join(ROLES) + "."
THRESHOLD = 0.35


def design():
    plan, rows = read(RUN / "plan.json"), registry()
    cache = PROJECT / "data/prefix-pilot/relative-depth-features.npz"
    with np.load(cache, allow_pickle=False) as f:
        pool = set(f["ids"].tolist()) & rows.keys()
        cache_info = {
            "signature": str(f["signature"]),
            "count": len(f["ids"]),
            "shape": list(f["vectors"].shape),
            "sha256": file_hash(cache),
        }
    if (
        cache_info["sha256"]
        != read(PROJECT / "results/depth_pilot.json")["feature_cache_sha256"]
    ):
        raise ValueError("Previously measured depth cache differs")
    samples = list(dict.fromkeys(plan["review"] + plan["panel"]))
    donors = {}
    for s in samples:
        eligible = [
            t
            for t in pool
            if rows[s]["dev_fold"] != rows[t]["dev_fold"]
            and t not in donors.values()
            and t not in samples
        ]
        donors[s] = min(eligible, key=lambda t: order("region-donor/" + s, t))
    spec = {
        "recipients": samples,
        "review": plan["review"],
        "donors": donors,
        "roles": list(ROLES),
        "threshold": THRESHOLD,
        "nms_iou": 0.5,
        "group_rule": "distinct official training donors outside each recipient outer fold",
        "old_depth_cache": cache_info,
        "allocation": "first 16 fixed before detection; target alternates by blinded sample order",
        "target": {
            s: ("fall_height", "struck_by_lifting")[i % 2]
            for i, s in enumerate(samples)
        },
    }
    p = ROOT / "design.json"
    if p.exists() and read(p) != spec:
        raise ValueError("Region assignments changed")
    write(p, spec)
    return spec, rows


def image_for(row):
    with Image.open(PROJECT / "data/train" / row["image"]) as src:
        return ImageOps.exif_transpose(src).convert("RGB")


def session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def pixels(pic, cfg):
    sz = cfg["size"]
    x = np.asarray(
        pic.resize((sz["width"], sz["height"]), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) * np.float32(cfg["rescale_factor"])
    x = (x - np.float32(cfg["image_mean"])) / np.float32(cfg["image_std"])
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def iou(a, b):
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = np.maximum(rb - lt, 0).prod()
    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter
    return inter / max(union, 1e-10)


class Detector:
    def __init__(self, base=False):
        from tokenizers import Tokenizer

        p = ROOT / "models/detector"
        self.net = session(
            ROOT / "models/detector_base/onnx/model.onnx"
            if base
            else p / "onnx/model_int8.onnx"
        )
        self.cfg = read(p / "preprocessor_config.json")
        tokenizer = Tokenizer.from_file(str(p / "tokenizer.json"))
        if base:
            tokenizer.enable_padding(
                length=256, pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]"
            )
        e = tokenizer.encode(TEXT)
        self.words = [
            [
                i
                for i, (a, b) in enumerate(e.offsets)
                if b > a and a >= TEXT.index(r) and b <= TEXT.index(r) + len(r)
            ]
            for r in ROLES
        ]
        if not all(self.words):
            raise ValueError("Tokenizer omitted a noun")
        self.feed = {
            k: np.asarray([z], np.int64)
            for k, z in (
                ("input_ids", e.ids),
                ("token_type_ids", e.type_ids),
                ("attention_mask", e.attention_mask),
            )
        }
        self.feed["pixel_mask"] = np.ones((1, 800, 800), np.int64)

    def infer(self, pic, s):
        logits, xywh = self.net.run(
            ["logits", "pred_boxes"],
            dict(self.feed, pixel_values=pixels(pic, self.cfg)),
        )
        probs = 1 / (1 + np.exp(-np.clip(logits[0], -80, 80)))
        scores = np.stack([probs[:, w].max(axis=1) for w in self.words], axis=1)
        xy, wh = np.split(xywh[0], 2, axis=-1)
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=-1).clip(0, 1)
        detected = []
        for k, role in enumerate(ROLES):
            kept = []
            for j in np.argsort(-scores[:, k], kind="stable"):
                if scores[j, k] < THRESHOLD:
                    break
                box = boxes[j]
                if (
                    scores[j].argmax() != k
                    or np.any(box[2:] <= box[:2])
                    or any(iou(box, boxes[l]) > 0.5 for l in kept)
                ):
                    continue
                kept.append(j)
                detected.append(
                    {"role": role, "score": float(scores[j, k]), "xyxy": box.tolist()}
                )
        return {
            "detections": detected,
            "missing": [r for r in ROLES if not any(d["role"] == r for d in detected)],
            "best_scores": dict(zip(ROLES, scores.max(axis=0).tolist())),
        }


def selection(rec, target):
    det = rec.get("detections", [])
    category = ROLES[1:3] if target == "struck_by_lifting" else ROLES[3:]
    best = lambda arr: max(arr, key=lambda d: d["score"], default=None)
    person = best([d for d in det if d["role"] == "person"])
    partner = best([d for d in det if d["role"] in category])
    relevant = [d for d in [person, partner] if d]
    if not relevant:
        return {
            "crop": [0.0, 0.0, 1.0, 1.0],
            "pair_visible": False,
            "box": None,
            "partner": None,
            "person": None,
        }
    boxes = np.asarray([d["xyxy"] for d in relevant])
    lo, hi = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    pad = np.maximum((hi - lo) * 0.16, 0.025)
    return {
        "crop": np.r_[lo - pad, hi + pad].clip(0, 1).tolist(),
        "pair_visible": len(relevant) == 2,
        "box": (person or partner)["xyxy"],
        "person": person,
        "partner": partner,
    }


class Segmenter:
    def __init__(self, model_path=None):
        p = model_path if model_path is not None else ROOT / "models/segmenter"
        self.enc = session(p / "onnx/vision_encoder_int8.onnx")
        self.dec = session(p / "onnx/prompt_encoder_mask_decoder_int8.onnx")
        self.cfg = read(p / "preprocessor_config.json")

    def infer(self, pic, s):
        a = read(ROOT / "detect" / (file_stem(s) + ".json"))
        box = a.get("selection", {}).get("box")
        return self.segment(pic,box,ROOT / "masks" / (file_stem(s) + ".png"))

    def segment(self, pic, box, dst):
        if box is None:
            return {"status": "no_box", "area": 0}
        embeds = dict(
            zip(
                [x.name for x in self.enc.get_outputs()],
                self.enc.run(None, {"pixel_values": pixels(pic, self.cfg)}),
            )
        )
        inputs = dict(
            embeds,
            input_boxes=np.float32([[box]]) * 1024,
            input_points=np.zeros((1, 1, 1, 2), np.float32),
            input_labels=np.full((1, 1, 1), -1, np.int64),
        )
        quality, masks, obj = self.dec.run(
            ["iou_scores", "pred_masks", "object_score_logits"], inputs
        )
        j = int(quality[0, 0].argmax())
        mask = Image.fromarray(np.float32(masks[0, 0, j])).resize(
            pic.size, Image.Resampling.BILINEAR
        )
        out = np.asarray(mask) > 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(dst)
        return {
            "box": box,
            "status": "generated",
            "quality": float(quality[0, 0, j]),
            "object_logit": float(obj.ravel()[0]),
            "area": int(out.sum()),
            "sha256": file_hash(dst),
        }


class Depth:
    def __init__(self):
        from .depth_pilot import DepthEncoder, MODEL

        self.enc = DepthEncoder(MODEL)

    def infer(self, pic, s):
        enc = self.enc
        size = (enc.edge, enc.edge)
        resized = ImageOps.contain(pic, size, method=Image.Resampling.BICUBIC)
        left, top = (enc.edge - resized.width) // 2, (enc.edge - resized.height) // 2
        canvas = Image.new(
            "RGB", size, tuple(int(round(float(c) * 255)) for c in enc.mean)
        )
        canvas.paste(resized, (left, top))
        arr = (np.asarray(canvas, dtype=np.float32) / 255 - enc.mean) / enc.std
        (field,) = enc.net.run(
            ["predicted_depth"],
            {"pixel_values": np.ascontiguousarray(arr.transpose(2, 0, 1)[None])},
        )
        field = field[0, top : top + resized.height, left : left + resized.width]
        if not np.isfinite(field).all():
            raise ValueError("nonfinite depth")
        lo, hi = np.percentile(field, [2, 98])
        norm = np.uint8(np.clip((field - lo) / max(hi - lo, 1e-8), 0, 1) * 255)
        dest = ROOT / "depth-maps" / (file_stem(s) + ".png")
        dest.parent.mkdir(exist_ok=True)
        Image.fromarray(norm).save(dest)
        return {
            "normalization": "within-picture p2-p98 relative only",
            "p2": float(lo),
            "p98": float(hi),
            "sha256": file_hash(dest),
        }


def run(mode, limit, recipients_only=False):
    spec, rows = design()
    ids = spec["recipients"][:limit]
    source = {t: s for s, t in spec["donors"].items()}
    if not recipients_only:
        ids = [t for s in ids for t in (s, spec["donors"][s])]
    t0 = time.perf_counter()
    engine = (
        Detector(base=True)
        if mode == "detect-base"
        else {"detect": Detector, "segment": Segmenter, "depth": Depth}[mode]()
    )
    print(mode, "LOADED", round(time.perf_counter() - t0, 2), flush=True)
    for s in ids:
        path = ROOT / mode / (file_stem(s) + ".json")
        if path.exists():
            previous = read(path)
            if not previous.get("error"):
                continue
            write(
                ROOT
                / "failed-attempts"
                / (
                    mode
                    + "-"
                    + file_stem(s)
                    + "-"
                    + str(path.stat().st_mtime_ns)
                    + ".json"
                ),
                previous,
            )
        t0 = time.perf_counter()
        pic = image_for(rows[s])
        try:
            result = engine.infer(pic, s)
            if mode in ("detect", "detect-base"):
                result["selection"] = selection(
                    result, spec["target"][source.get(s, s)]
                )
        except Exception as err:
            result = {"error": repr(err), "status": "failed"}
        result.update(
            sample_id=s,
            image_sha256=rows[s]["sha256"],
            size=pic.size,
            seconds=time.perf_counter() - t0,
        )
        write(path, result)
        print(
            mode,
            s,
            result.get("status", "ok"),
            dict(Counter(d["role"] for d in result.get("detections", []))),
            round(result["seconds"], 2),
            result.get("error", ""),
            flush=True,
        )
        if result.get("error"):
            raise RuntimeError(result["error"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("detect", "detect-base", "segment", "depth"))
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--recipients-only", action="store_true")
    args = ap.parse_args()
    run(args.mode, args.limit, args.recipients_only)
