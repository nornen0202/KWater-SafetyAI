"""Standalone, offline Phase 1 image-to-official-JSONL entry."""

import argparse
import json
import os
from pathlib import Path
import re
import time

from PIL import Image
import torch

from .phase01_head import SpatialHead, decide, logits
from .phase01_vision import Vision, SIGNATURE
from .metrics import CODES
from .submission2_output import make_row


def run(model_dir, input_dir, output_dir, device, loaded=None):
    profile = os.environ.get("KWATER_PROFILE") == "1"
    def mark():
        if profile and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()
    start = mark()
    cfg = json.loads((model_dir / "policy.json").read_bytes())
    if cfg["codes"] != list(CODES) or cfg["vision_signature"] != SIGNATURE:
        raise ValueError("Model format does not match this entry")
    if loaded is None:
        vision = Vision(model_dir / "models", device)
        backbones_loaded = mark()
        head = SpatialHead(cfg["multiscale"]).to(device)
        head.load_state_dict(
            torch.load(model_dir / cfg["head"], map_location=device, weights_only=True)
        )
        head.eval()
        if profile:
            print("PROFILE_LOAD", json.dumps({"backbones_s": backbones_loaded-start,
                  "head_s": mark()-backbones_loaded}), flush=True)
    else:
        vision, head = loaded
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / "submission.jsonl"
    tmp = output_dir / "submission.tmp"
    seen = set()
    try:
        with (
            (input_dir / "manifest.jsonl").open(encoding="utf8") as ins,
            tmp.open("w", encoding="utf8", newline="\n") as out,
        ):
            for line in ins:
                photo_start = mark()
                sid = json.loads(line)["image_id"]
                if not re.fullmatch("[0-9a-f]{32}", sid) or sid in seen:
                    raise ValueError("Invalid or repeated manifest ID")
                seen.add(sid)
                with Image.open(input_dir / "images" / (sid + ".jpg")) as image:
                    opened = mark()
                    if profile:
                        timing = {}
                        features = vision.encode(image, cfg["multiscale"], profile=timing)
                    else:
                        features = vision.encode(image, cfg["multiscale"])
                encoded = mark()
                z = logits(head, torch.from_numpy(features[None]).to(device))
                active = decide(z, cfg["policy"])[0]
                row = make_row(sid, [c for c, flag in zip(CODES, active) if flag])
                prepared = mark()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                if profile:
                    end = mark()
                    timing.update(image_id=sid, open_s=opened-photo_start,
                                  backbone_s=encoded-opened-timing["preprocess_s"],
                                  head_policy_format_s=prepared-encoded, write_s=end-prepared,
                                  total_s=end-photo_start)
                    print("PROFILE_PHOTO", json.dumps(timing), flush=True)
            closing = mark()
        tmp.replace(path)
        if profile:
            print("PROFILE_CLOSE", mark()-closing, flush=True)
    finally:
        tmp.unlink(missing_ok=True)


def main(model_dir=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir", type=Path, default=model_dir, required=model_dir is None
    )
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    run(a.model_dir, a.input_dir, a.output_dir, a.device)


if __name__ == "__main__":
    main()
