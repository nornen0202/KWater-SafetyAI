"""Offline CLIP image features and a zero-shot ranking probe for K-water research."""

import argparse
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import onnxruntime as ort
import PIL
import tokenizers
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from .metrics import CODES


REPO = "Xenova/clip-vit-base-patch32"
REVISION = "6b7c523703ae26e39be4b03774966f8571c7f523"
FILES = {
    "onnx/vision_model_quantized.onnx": "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
    "onnx/text_model_quantized.onnx": "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
    "tokenizer.json": "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
    "preprocessor_config.json": "6f638fb9401a6d6296feff533ee7efe657b787c49f954f82f5906b36ef2a1b1f",
}
# Fixed descriptions of organizer categories, independent of an individual photograph.
DESCRIPTIONS = {
    "fall_height": "a fall from height hazard",
    "slip_trip": "a slip or trip hazard on a walking surface",
    "caught_in_machine": "a risk of entanglement or crushing in machinery",
    "struck_by_lifting": "a risk of being struck by a falling, flying or suspended load",
    "electrical_energy": "an electric shock hazard",
    "drowning_inundation": "a drowning or flooding hazard",
    "fire_explosion": "a fire, explosion or pressure rupture hazard",
    "confined_space": "a confined space entry hazard",
    "chemical_exposure": "a chemical leak, inhalation or contact hazard",
    "thermal_weather": "a hazard from heat, cold or severe weather",
    "other": "another visible occupational safety hazard",
    "no_visible_hazard": "no visible occupational safety hazard",
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assets(folder, download=False):
    if download:
        folder.mkdir(parents=True, exist_ok=True)
    for remote, expected in FILES.items():
        path = folder / Path(remote).name
        if path.is_file() and file_hash(path) == expected:
            continue
        if not download:
            raise ValueError(f"Missing or incorrect model file: {path}")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        part = path.with_suffix(path.suffix + ".part")
        with urlopen(Request(url, headers={"User-Agent": "kwater-research/1"}), timeout=120) as src, part.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
        if file_hash(part) != expected:
            raise ValueError(f"Hash mismatch: {remote}")
        part.replace(path)


def session(source):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    model = source if isinstance(source, bytes) else str(source)
    return ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])


def unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Invalid embedding")
    return matrix / norms


class ImageEncoder:
    def __init__(self, folder, *, verify_vision=False):
        config = (folder / "preprocessor_config.json").read_bytes()
        source = folder / "vision_model_quantized.onnx"
        if verify_vision:
            source = source.read_bytes()
            if (hashlib.sha256(source).hexdigest() != FILES["onnx/vision_model_quantized.onnx"] or
                    hashlib.sha256(config).hexdigest() != FILES["preprocessor_config.json"]):
                raise ValueError("Incorrect image encoder or preprocessor checksum")
        cfg = json.loads(config)
        self.edge = cfg["size"]["shortest_edge"]
        self.mean = np.asarray(cfg["image_mean"], dtype=np.float32)
        self.std = np.asarray(cfg["image_std"], dtype=np.float32)
        self.model = session(source)

    def encode(self, image, framing):
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = (self.edge, self.edge)
        if framing == "center_crop":
            # Reference CLIP transform: resize shorter edge first, then center crop.
            w, h = image.size
            scale_size = (self.edge, self.edge * h // w) if w <= h else (self.edge * w // h, self.edge)
            image = image.resize(scale_size, Image.Resampling.BICUBIC)
            x = (image.width - self.edge) // 2
            y = (image.height - self.edge) // 2
            image = image.crop((x, y, x + self.edge, y + self.edge))
        elif framing == "full_frame":
            fill = tuple(int(round(float(c) * 255)) for c in self.mean)
            image = ImageOps.pad(image, size, method=Image.Resampling.BICUBIC, color=fill)
        else:
            raise ValueError(f"Unknown framing: {framing}")
        pixels = (np.asarray(image, dtype=np.float32) / 255.0 - self.mean) / self.std
        pixels = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        vector, = self.model.run(["image_embeds"], {"pixel_values": pixels})
        return unit_rows(vector)[0]


def rank_image(args):
    assets(args.model)
    encoder = ImageEncoder(args.model)
    text_session = session(args.model / "text_model_quantized.onnx")
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    codes = [*CODES, "no_visible_hazard"]
    prompts = [f"A photo of an industrial worksite with {DESCRIPTIONS[code]}." for code in codes]
    # Run each text at its own length so padding cannot influence text vector selection.
    vectors = []
    for prompt in prompts:
        ids = np.array([tokenizer.encode(prompt).ids], dtype=np.int64)
        vectors.append(text_session.run(["text_embeds"], {"input_ids": ids})[0][0])
    texts = unit_rows(np.stack(vectors))
    raw = args.image.read_bytes()
    report = {
        "type": "public_single_image_zero_shot_ranking", "model": REPO, "revision": REVISION,
        "model_file_sha256": FILES, "image_sha256": hashlib.sha256(raw).hexdigest(),
        "prompt_by_code": dict(zip(codes, prompts)), "versions": {
            "python": platform.python_version(), "onnxruntime": ort.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__, "tokenizers": tokenizers.__version__,
        }, "runtime": "CPUExecutionProvider, 4 threads", "framing": {},
    }
    with Image.open(io.BytesIO(raw)) as image:
        report["image_wh"] = list(image.size)
        for framing in ("center_crop", "full_frame"):
            start = time.perf_counter()
            scores = texts @ encoder.encode(image, framing)
            ranking = sorted(zip(codes, map(float, scores)), key=lambda entry: -entry[1])
            report["framing"][framing] = {
                "warm_model_image_seconds": time.perf_counter() - start,
                "ranking": [{"code": k, "cosine": v} for k, v in ranking],
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract(args):
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    if report["status"] != "ready" or set(report["assignments"]) != set(report["inventory"]):
        raise ValueError("A complete, reviewed, ready audit is required.")
    if file_hash(args.labels) != report["labels_sha256"]:
        raise ValueError("The label file has changed since the audit.")
    seen_groups = {}
    selected = []
    for sid, row in sorted(report["assignments"].items()):
        group, split = row["group_id"], row["split"]
        if not group or split not in {"train", "dev", "holdout"} or seen_groups.setdefault(group, split) != split:
            raise ValueError("Invalid group or cross-split group")
        if split != "holdout":
            selected.append(sid)
    if not selected:
        raise ValueError("Empty training/development set")
    assets(args.model)
    encoder = ImageEncoder(args.model)
    vectors = []
    root = args.image_root.resolve()
    for sid in selected:
        item = report["inventory"][sid]
        path = (root / item["image"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image outside the dataset: {sid}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Image has changed since the audit: {sid}")
        with Image.open(io.BytesIO(data)) as image:
            vectors.append(encoder.encode(image, args.framing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".part")
    with temp.open("wb") as out:
        np.savez_compressed(out, ids=np.array(selected), features=np.stack(vectors),
                            labels_sha256=report["labels_sha256"], revision=REVISION, framing=args.framing,
                            audit_sha256=file_hash(args.audit))
    os.replace(temp, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fetch = sub.add_parser("fetch", help="download pinned model assets")
    fetch.add_argument("--model", type=Path, required=True)
    probe = sub.add_parser("rank", help="rank hazard descriptions on a single local photo")
    feature = sub.add_parser("extract", help="extract reviewed train/dev features; skips holdout images")
    for cmd in (probe, feature):
        cmd.add_argument("--model", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    probe.add_argument("--image", type=Path, required=True)
    feature.add_argument("--audit", type=Path, required=True)
    feature.add_argument("--labels", type=Path, required=True)
    feature.add_argument("--image-root", type=Path, required=True)
    feature.add_argument("--framing", choices=["center_crop", "full_frame"], default="full_frame")
    args = parser.parse_args()
    if args.action == "fetch":
        assets(args.model, download=True)
    elif args.action == "rank":
        rank_image(args)
    else:
        extract(args)


if __name__ == "__main__":
    main()
