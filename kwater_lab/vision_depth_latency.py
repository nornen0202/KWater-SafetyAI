"""Measure incremental relative-depth time on the fixed 200 official-kit rehearsal inputs."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

from .depth_pilot import ASSETS, DepthEncoder


def measure(input_dir, receipt, model, provider):
    manifest = (input_dir / "manifest.jsonl").read_bytes()
    if sha256(manifest).hexdigest() != "8c86d0b60b7b879573ccdf4c81558ed4a3840588bb6d2fbbe656dfdb911335b6":
        raise ValueError("Unexpected 200-photo timing manifest")
    rows = json.loads(receipt.read_bytes())
    ids = [json.loads(line)["image_id"] for line in manifest.splitlines()]
    if len(set(ids)) != len(ids) or len(ids) != 200 or ids != [r["id"] for r in rows]:
        raise ValueError("Timing IDs differ")
    if provider == "CUDAExecutionProvider":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.cuda.get_device_name(0)
        # Load the CUDA/cuDNN shared libraries used by the signed-in VM environment.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    else:
        device = "CPU"
    start = perf_counter()
    encoder = DepthEncoder(model, provider)
    load_seconds = perf_counter() - start
    per_arm = {"decode": [], "decode_and_depth": []}
    encode_times = []
    for i, row in enumerate(rows):
        name = row["filename"]
        if Path(name).name != name:
            raise ValueError("Invalid timing filename")
        raw = (input_dir / name).read_bytes()
        if sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Image digest differs")
        arms = ("decode", "decode_and_depth")
        for arm in (arms if i % 2 == 0 else arms[::-1]):
            start = perf_counter()
            with Image.open(BytesIO(raw)) as source:
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                if arm == "decode_and_depth":
                    t = perf_counter()
                    vec = encoder.encode(rgb, depth_only=True)
                    encode_times.append(perf_counter() - t)
                    if vec.shape != (32,) or not np.isfinite(vec).all():
                        raise ValueError("Invalid depth descriptor")
            per_arm[arm].append(perf_counter() - start)
        if (i+1) % 50 == 0:
            print(f"timing {provider} {i+1}/200", flush=True)
    elapsed = {k: sum(v) for k, v in per_arm.items()}
    delta = elapsed["decode_and_depth"] - elapsed["decode"]
    return dict(photos=200, input_manifest_sha256=sha256(manifest).hexdigest(),
                input_receipt_sha256=sha256(receipt.read_bytes()).hexdigest(), weights=ASSETS,
                device=device, provider=encoder.net.get_providers(), onnxruntime=ort.__version__,
                elapsed_seconds=elapsed, relative_depth_seconds=sum(encode_times), model_load_seconds=load_seconds,
                warm_incremental_seconds=delta, cold_incremental_seconds=delta+load_seconds,
                p50_depth_seconds=float(np.median(encode_times)), p95_depth_seconds=float(np.percentile(encode_times, 95)),
                scope="Paired alternating decode vs decode+depth, same encoded bytes in memory. Structural module benchmark; excludes specialist and VLM.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "receipt", "model", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--provider", choices=["CPUExecutionProvider", "CUDAExecutionProvider"], required=True)
    a = parser.parse_args()
    a.out.write_text(json.dumps(measure(a.input, a.receipt, a.model, a.provider), indent=2) + "\n", encoding="utf-8")
