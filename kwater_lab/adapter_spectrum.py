"""Measure the spectrum of each official LoRA update without creating dense base-model matrices."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from .train_head import file_hash, save_atomic


def singular_values(a, b, scale):
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("Invalid LoRA A/B dimensions")
    ra = np.linalg.qr(a.T.astype(np.float64), mode="r")
    rb = np.linalg.qr(b.astype(np.float64), mode="r")
    return abs(scale) * np.linalg.svd(rb @ ra.T, compute_uv=False)


def read_tensor(stream, offset, entry):
    if entry["dtype"] != "F32":
        raise ValueError("The distributed adapter audit expects float32 tensors")
    start, end = entry["data_offsets"]
    if end-start != 4*int(np.prod(entry["shape"])) or start < 0:
        raise ValueError("Invalid tensor byte range")
    stream.seek(offset + start)
    return np.frombuffer(stream.read(end-start), dtype="<f4").reshape(entry["shape"])


def statistics(spectra):
    energy = np.square(np.stack(spectra))
    total = float(energy.sum())
    retained = {str(k): float(energy[:, :k].sum()/total) if total else None for k in (1,4,8)}
    return {"updates": len(energy), "squared_frobenius_sum": total,
            "retained_energy_at_rank": retained}


def measure(adapter):
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    rank, alpha = config["r"], config["lora_alpha"]
    if (config["use_dora"] or config["use_rslora"] or config["rank_pattern"] or config["alpha_pattern"]):
        raise ValueError("Scaling rule differs from the uniform original LoRA configuration")
    path = adapter / "adapter_model.safetensors"
    by_module, by_layer = defaultdict(list), defaultdict(list)
    all_values, parameter_count = [], 0
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 2 <= size <= min(path.stat().st_size-8, 100_000_000):
            raise ValueError("Invalid safetensors header")
        entries = json.loads(stream.read(size))
        consumed = {"__metadata__"} & entries.keys()
        for key in entries:
            match = re.fullmatch(r".*language_model\.layers\.(\d+)\.(?:mlp|self_attn)\.(\w+)\.lora_A\.weight", key)
            if not match:
                continue
            other = key.replace(".lora_A.", ".lora_B.")
            a, b = (read_tensor(stream, 8+size, entries[k]) for k in (key, other))
            if a.shape[0] != rank:
                raise ValueError("Stored rank differs from adapter configuration")
            singular = singular_values(a, b, alpha/rank)
            by_layer[int(match[1])].append(singular)
            by_module[match[2]].append(singular)
            all_values.append(singular)
            parameter_count += a.size + b.size
            consumed.update((key, other))
    if set(entries) != consumed or not all_values:
        raise ValueError("Unaccounted adapter tensors")
    return {"scope": "Read-only adapter geometry; no accuracy or inference-speed inference from energy.",
            "adapter_sha256": file_hash(path), "alpha": alpha, "rank": rank,
            "parameter_count": parameter_count, "layers": len(by_layer), "overall": statistics(all_values),
            "by_module": {k: statistics(v) for k,v in sorted(by_module.items())},
            "by_layer": {str(k): statistics(v) for k,v in sorted(by_layer.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("data/baseline/adapter"))
    parser.add_argument("--output", type=Path, default=Path("results/adapter_spectrum.json"))
    args = parser.parse_args()
    report = measure(args.adapter)
    save_atomic(args.output, lambda f: f.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode()))
    print(report["layers"], report["parameter_count"], report["overall"])
