"""Audit source PNG dimensions and recorded token usage without loading an image model."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import struct

from .adapter_scale_pilot import write_json
from .prompt_pilot import ARMS, digest
from .train_head import file_hash


def measured(values):
    if not values:
        return {"observations": 0, "minimum": None, "median": None, "maximum": None}
    return {"observations": len(values), "minimum": min(values), "median": median(values), "maximum": max(values)}


def png_size(path):
    with path.open("rb") as image:
        header = image.read(24)
    if (not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR"
            or len(header) < 24):
        raise ValueError("Expected an original PNG with an IHDR dimensions chunk")
    return struct.unpack(">II", header[16:24])


def audit(root):
    pilot = root / "data/prompt-pilot"
    design = json.loads((pilot / "design.json").read_text(encoding="utf-8"))
    runs = json.loads((root / "results/prompt_pilot.json").read_text(encoding="utf-8"))
    signature = digest(design)
    if runs["signature"] != signature:
        raise ValueError("Token audit and measured protocol differ")
    folder = pilot / signature[:12]
    dimensions = Counter()
    for case in design["anchors"]:
        path = root / "data/train" / case["image"]
        if file_hash(path) != case["sha256"]:
            raise ValueError("Source photo and measured protocol differ")
        width, height = png_size(path)
        dimensions[f"{width}x{height}"] += 1
    usage = {}
    image_counts = []
    for arm in ARMS:
        prompt, completion = [], []
        for i in range(len(design["anchors"])):
            record = json.loads((folder / f"{i}-{arm}.json").read_text(encoding="utf-8"))
            case = design["anchors"][(i + int(arm == "swapped_11")) % len(design["anchors"])]
            if record["signature"] != signature or record["image_sha256"] != case["sha256"]:
                raise ValueError("Generation receipt and measured input differ")
            counts = record["generation"].get("usage") or {}
            for field, values in (("prompt_tokens", prompt), ("completion_tokens", completion)):
                if type(counts.get(field)) is int:
                    values.append(counts[field])
            detail = counts.get("prompt_tokens_details") or {}
            if type(detail.get("image_tokens")) is int:
                image_counts.append(detail["image_tokens"])
        usage[arm] = {"full_prompt_tokens": measured(prompt), "completion_tokens": measured(completion)}
    log = folder / "model.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    result = {"signature": signature, "input_format": "Original full PNG; no external crop or resize",
              "source_resolution_counts": dict(sorted(dimensions.items())), "conditions": usage,
              "generation_limit": design["max_tokens"], "context_size": design["context_size"],
              "server_flags": runs["runtime"]["server_flags"],
              "startup_seconds": runs["runtime"]["startup_s"],
              "image_tokens_in_api_usage": measured(image_counts),
              "runtime_warned_about_1024_tokens_for_grounding":
                  "at minimum 1024 image tokens" in log_text,
              "server_log_sha256": file_hash(log)}
    write_json(root / "results/prompt_vision_budget.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(audit(parser.parse_args().root.resolve()), ensure_ascii=False, indent=2))
