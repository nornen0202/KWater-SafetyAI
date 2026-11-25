"""Offline competition entry point for the two-encoder hazard classifier."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from safety_output import make_row


ROOT = Path(__file__).resolve().parent


class RiskEncoder:
    def __init__(self, name, output, heads):
        config = json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))
        size = config["size"]
        self.edge = size.get("shortest_edge") or size["height"]
        self.mean = np.asarray(config["image_mean"], dtype=np.float32)
        self.std = np.asarray(config["image_std"], dtype=np.float32)
        self.fill = tuple(int(round(float(value) * 255)) for value in self.mean)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(ROOT / f"{name}.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.output = output
        self.head = {key: heads[f"{name}_{key}"] for key in ("center", "scale", "weight", "bias", "prior")}

    def scores(self, image):
        resized = ImageOps.pad(image, (self.edge, self.edge), method=Image.Resampling.BICUBIC, color=self.fill)
        pixels = (np.asarray(resized, dtype=np.float32) / 255.0 - self.mean) / self.std
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        vector, = self.session.run([self.output], {"pixel_values": tensor})
        vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
        h = self.head
        logits = ((vector - h["center"]) / h["scale"]) @ h["weight"] + h["bias"]
        prob = np.exp(-np.logaddexp(0.0, -logits))[0]
        prior = h["prior"]
        constant = (prior == 0) | (prior == 1)
        prob[constant] = prior[constant]
        return prob


def run(input_dir, output_dir):
    with np.load(ROOT / "head.npz", allow_pickle=False) as heads:
        encoders = [RiskEncoder("clip", "image_embeds", heads), RiskEncoder("sig", "pooler_output", heads)]
        codes, threshold = heads["codes"].tolist(), float(heads["threshold"].item())
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / "submission.tmp"
    with (input_dir / "manifest.jsonl").open(encoding="utf-8") as manifest, temp.open("w", encoding="utf-8", newline="\n") as results:
        for line in manifest:
            if not line.strip():
                continue
            image_id = json.loads(line)["image_id"]
            if not re.fullmatch(r"[0-9a-f]{32}", image_id):
                raise ValueError("Invalid input ID")
            with Image.open(input_dir / "images" / f"{image_id}.jpg") as photo:
                image = ImageOps.exif_transpose(photo).convert("RGB")
            scores = (encoders[0].scores(image) + encoders[1].scores(image)) / 2
            top = np.argsort(-scores, kind="stable")[:3]
            row = make_row(image_id, [codes[i] for i in top if scores[i] >= threshold])
            results.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(output_dir / "submission.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
