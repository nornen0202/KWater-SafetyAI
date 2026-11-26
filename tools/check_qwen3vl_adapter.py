"""Check all language and visual LoRA tensor pairs without loading the base model."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open


def inspect(adapter_dir, base_dir):
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    if adapter_config["peft_type"] != "LORA" or adapter_config["task_type"] != "CAUSAL_LM":
        raise ValueError("Expected a causal language model LoRA adapter")
    rank = adapter_config["r"]
    scale = adapter_config["lora_alpha"] / rank
    text = base_config["text_config"]
    vision = base_config["vision_config"]
    hidden = text["hidden_size"]
    intermediate = text["intermediate_size"]
    head_dim = text.get("head_dim", hidden // text["num_attention_heads"])
    query = text["num_attention_heads"] * head_dim
    kv = text["num_key_value_heads"] * head_dim
    text_dims = {
        "self_attn.q_proj": (hidden, query),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (query, hidden),
        "mlp.gate_proj": (hidden, intermediate),
        "mlp.up_proj": (hidden, intermediate),
        "mlp.down_proj": (intermediate, hidden),
    }
    expected = {
        f"language_model.layers.{layer}.{name}": dims
        for layer in range(text["num_hidden_layers"])
        for name, dims in text_dims.items()
    }
    merger_hidden = vision["hidden_size"] * vision["spatial_merge_size"] ** 2
    merger_names = ["merger", *(f"deepstack_merger_list.{i}" for i in range(len(vision["deepstack_visual_indexes"])))]
    for name in merger_names:
        expected[f"visual.{name}.linear_fc1"] = (merger_hidden, merger_hidden)
        expected[f"visual.{name}.linear_fc2"] = (merger_hidden, vision["out_hidden_size"])

    weights = adapter_dir / "adapter_model.safetensors"
    counts = Counter()
    min_norm = math.inf
    max_norm = 0.0
    with safe_open(weights, framework="numpy") as tensors:
        found = {}
        for key in tensors.keys():
            match = re.fullmatch(r"base_model\.model\.model\.(.+)\.lora_([AB])\.weight", key)
            if not match:
                raise ValueError(f"Unexpected tensor name: {key}")
            found.setdefault(match[1], {})[match[2]] = key
        if found.keys() != expected.keys():
            raise ValueError(f"Tensor pair keys mismatch: expected {len(expected)}, found {len(found)}")
        for name, (input_width, output_width) in expected.items():
            if found[name].keys() != {"A", "B"}:
                raise ValueError(f"Missing a LoRA tensor in {name}")
            a = tensors.get_tensor(found[name]["A"]).astype(np.float64)
            b = tensors.get_tensor(found[name]["B"]).astype(np.float64)
            if a.shape != (rank, input_width) or b.shape != (output_width, rank):
                raise ValueError(f"Shape mismatch in {name}: {a.shape}, {b.shape}")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Nonfinite LoRA tensor in {name}")
            # Tr(A A^T B^T B) gives the exact squared Frobenius norm of B A.
            squared_norm = float(np.sum((a @ a.T) * (b.T @ b)))
            if squared_norm <= 0:
                raise ValueError(f"Zero LoRA delta in {name}")
            norm = math.sqrt(squared_norm) * scale
            min_norm = min(min_norm, norm)
            max_norm = max(max_norm, norm)
            counts["vision" if name.startswith("visual.") else "language"] += 1
    with weights.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "adapter_sha256": sha256,
        "rank": rank,
        "tensor_pairs_nonzero_delta": dict(counts),
        "min_delta_frobenius_norm": min_norm,
        "max_delta_frobenius_norm": max_norm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("data/baseline/base_model"))
    args = parser.parse_args()
    print(json.dumps(inspect(args.adapter_dir, args.base_dir), indent=2))


if __name__ == "__main__":
    main()
