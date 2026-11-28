"""Fetch and inspect the pinned Qwen FP8 snapshot on CPU without loading a model."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil, isfinite, prod
from pathlib import Path
from struct import iter_unpack, unpack
from subprocess import CalledProcessError
from time import sleep
from urllib.request import urlopen

from tools.resume_large_qwen_assets import receive

from .fold2_8b_baseline import write_json
from .metrics import unique_keys
from .stage1_fp8_l4_smoke import FP8_RECEIPT, verify_base


BASE = Path("data/stage1-fp8-base")
RESULT = Path("results/stage1-fp8-offline-files-20261129.json")
WIDTH_BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_keys)


def validate_source(source, trusted):
    if any(source[key] != trusted[key] for key in ("repo", "revision", "license")):
        raise ValueError("Source does not match the pinned public Qwen checkpoint")
    entries = {row["rfilename"]: row for row in source["files"]}
    if len(entries) != len(source["files"]) or entries.keys() != trusted["files"].keys():
        raise ValueError("Source file roster differs from the pinned model")
    for name, row in entries.items():
        expected = trusted["files"][name]
        if row["size"] != expected["bytes"]:
            raise ValueError(f"Source file size differs: {name}")
        if name.endswith(".safetensors") and (row["lfs"]["sha256"] != expected["sha256"]
                                                or row["lfs"]["size"] != row["size"]):
            raise ValueError(f"Source LFS object differs: {name}")
    return entries


def fetch(base, trusted):
    base.mkdir(parents=True, exist_ok=True)
    manifest = base / "source.json"
    if manifest.is_file():
        source = read_json(manifest)
    else:
        url = f"https://huggingface.co/api/models/{trusted['repo']}/revision/{trusted['revision']}?blobs=true"
        with urlopen(url, timeout=30) as reply:
            info = json.load(reply, object_pairs_hook=unique_keys)
        source = {"repo": info["id"], "revision": info["sha"], "license": info["cardData"]["license"],
                  "model_card": f"https://huggingface.co/{info['id']}",
                  "files": [row for row in info["siblings"] if row["rfilename"] in trusted["files"]]}
        validate_source(source, trusted)
        write_json(manifest, source)
    pending = [row for name, row in validate_source(source, trusted).items() if not (base / name).is_file()]

    def resume(row):
        for attempt in range(3):
            try:
                receive(base, source, row)
                return
            except CalledProcessError as error:
                if attempt == 2 or error.returncode not in {18, 28, 52, 55, 56, 92}:
                    raise
                print(f"Retrying interrupted HTTPS transfer for {row['rfilename']}", flush=True)
                sleep(5)

    for row in pending:
        resume(row)


def shard_header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Missing safetensors header length: {path.name}")
        length, = unpack("<Q", prefix)
        if not 1 <= length <= 100_000_000 or length + 8 >= path.stat().st_size:
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        raw = stream.read(length)
    if len(raw) != length or not raw.startswith(b"{"):
        raise ValueError(f"Invalid safetensors JSON header: {path.name}")
    tensors = json.loads(raw, object_pairs_hook=unique_keys)
    metadata = tensors.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(x, str) for x in metadata.values()):
        raise ValueError(f"Invalid safetensors metadata: {path.name}")

    layout = []
    dtype_count = Counter()
    for name, row in tensors.items():
        shape = row["shape"]
        offset, end = row["data_offsets"]
        dtype = row["dtype"]
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or type(offset) is not int or type(end) is not int or offset < 0
                or dtype not in WIDTH_BYTES or end - offset != WIDTH_BYTES[dtype] * prod(shape)):
            raise ValueError(f"Unexpected tensor shape, dtype or byte span: {name}")
        layout.append((offset, end))
        dtype_count[dtype] += 1
    cursor = 0
    for start, end in sorted(layout):
        if cursor != start:
            raise ValueError(f"Hole or overlap in safetensors payload: {path.name}")
        cursor = end
    if cursor != path.stat().st_size - (length + 8):
        raise ValueError(f"Safetensors payload does not cover the entire file: {path.name}")
    return tensors, {"header_bytes_excluding_length_prefix": length, "header_sha256": sha256(raw).hexdigest(),
                     "tensor_payload_bytes": cursor, "tensors": len(tensors), "tensor_count_by_dtype": dict(dtype_count)}


def projection_shapes(config):
    text = config["text_config"]
    hidden, middle = text["hidden_size"], text["intermediate_size"]
    query = text["num_attention_heads"] * text["head_dim"]
    key_value = text["num_key_value_heads"] * text["head_dim"]
    shapes = {"self_attn.q_proj": [query, hidden], "self_attn.k_proj": [key_value, hidden],
              "self_attn.v_proj": [key_value, hidden], "self_attn.o_proj": [hidden, query],
              "mlp.gate_proj": [middle, hidden], "mlp.up_proj": [middle, hidden], "mlp.down_proj": [hidden, middle]}
    return {f"model.language_model.layers.{layer}.{projection}.weight": shape
            for layer in range(text["num_hidden_layers"]) for projection, shape in shapes.items()}


def check_scales(base, config, descriptions, headers):
    block = config["quantization_config"]["weight_block_size"]
    if block != [128, 128]:
        raise ValueError("Expected 128 by 128 block scales")
    weights = projection_shapes(config)
    scales = {name + "_scale_inv" for name in weights}
    if ({name for name, (_, row) in descriptions.items() if row["dtype"] == "F8_E4M3"} != weights.keys()
            or {name for name in descriptions if name.endswith(".weight_scale_inv")} != scales):
        raise ValueError("FP8 weight and scale pairs differ from the expected language projections")

    total = 0
    smallest, largest = float("inf"), 0
    for shard, header in headers.items():
        with (base / shard).open("rb") as stream:
            prefix_bytes = header["header_bytes_excluding_length_prefix"] + 8
            for name in sorted(scales):
                scale_shard, scale = descriptions[name]
                if scale_shard != shard:
                    continue
                weight_name = name.removesuffix("_scale_inv")
                weight_shard, weight = descriptions[weight_name]
                expected = weights[weight_name]
                if (weight_shard != shard or weight["shape"] != expected or scale["dtype"] != "F32"
                        or scale["shape"] != [ceil(axis / size) for axis, size in zip(expected, block, strict=True)]):
                    raise ValueError(f"Weight and block scale do not match: {weight_name}")
                start, end = scale["data_offsets"]
                stream.seek(prefix_bytes + start)
                values = [value for (value,) in iter_unpack("<f", stream.read(end - start))]
                if len(values) != prod(scale["shape"]) or any(not isfinite(v) or v <= 0 for v in values):
                    raise ValueError(f"Invalid stored inverse scales: {name}")
                smallest, largest = min(smallest, min(values)), max(largest, max(values))
                total += len(values)
    excluded = [row for name, (_, row) in descriptions.items()
                if name.startswith("model.visual.") or name == "lm_head.weight"]
    if not excluded or any(row["dtype"] != "BF16" for row in excluded):
        raise ValueError("Expected BF16 visual tensors and output head")
    return {"paired_projection_weights": len(weights), "scale_tensors": len(scales), "scale_values": total,
            "scale_min": smallest, "scale_max": largest, "block_size": block,
            "each_weight_and_scale_same_shard": True, "visual_and_lm_head_bf16_tensors": len(excluded)}


def audit(base, trusted):
    validate_source(read_json(base / "source.json"), trusted)
    index = read_json(base / "model.safetensors.index.json")
    config = read_json(base / "config.json")
    files = {name: {"bytes": (base / name).stat().st_size, "sha256": row["sha256"]}
             for name, row in trusted["files"].items()}
    if any(row["bytes"] != trusted["files"][name]["bytes"] for name, row in files.items()):
        raise ValueError("Local model file sizes differ from the pinned source")
    verify_base(base, "fp8", {})
    headers, descriptions = {}, {}
    for name in sorted(files):
        if name.endswith(".safetensors"):
            tensors, headers[name] = shard_header(base / name)
            if descriptions.keys() & tensors.keys():
                raise ValueError("A tensor is present in two shards")
            descriptions.update({key: (name, row) for key, row in tensors.items()})
    if {key: shard for key, (shard, _) in descriptions.items()} != index["weight_map"]:
        raise ValueError("Safetensors header membership differs from the checkpoint weight map")
    total_payload = sum(header["tensor_payload_bytes"] for header in headers.values())
    if index.get("metadata", {}).get("total_size", total_payload) != total_payload:
        raise ValueError("Index total_size differs from the local tensor payload bytes")

    return {"version": "stage1-fp8-offline-files-v1", "repo": trusted["repo"], "revision": trusted["revision"],
            "license": trusted["license"], "files": files, "shards": headers, "indexed_tensors": len(descriptions),
            "shard_bytes": sum(files[name]["bytes"] for name in headers),
            "inference_file_bytes": sum(row["bytes"] for row in files.values()),
            "tensor_payload_bytes": total_payload, "scale_audit": check_scales(base, config, descriptions, headers),
            "l4_smoke_verify_base_executed": True, "model_loaded": False, "l4_kernel_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--fetch", action="store_true", help="Resume official public HTTPS downloads before auditing")
    args = parser.parse_args()
    trusted = read_json(FP8_RECEIPT)
    if not args.base.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Keep public model weights under the local data/ tree")
    if (args.result.resolve().is_relative_to(args.base.resolve()) or args.result.resolve() == FP8_RECEIPT.resolve()):
        parser.error("Keep the CPU audit receipt separate from the model folder and pinned source receipt")
    if args.fetch:
        fetch(args.base, trusted)
    write_json(args.result, audit(args.base, trusted))


if __name__ == "__main__":
    main()
