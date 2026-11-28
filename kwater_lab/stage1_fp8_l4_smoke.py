"""Research-only Qwen3-VL-8B precision smoke in the official offline L4 runtime."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time


FP8_RECEIPT = Path("results/stage1-fp8-source-audit-20261128.json")
BF16_RECEIPT = Path("results/fold4-8b-epoch1-research-package-20261128.json")
PROMPT = Path(__file__).with_name("system_prompt_v3.txt")
MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
PINNED_VERSIONS = {"torch": "2.11.0+cu128", "transformers": "4.57.1", "peft": "0.17.1",
                   "accelerate": "1.10.1", "triton": "3.6.0", "tokenizers": "0.22.1"}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_base(base: Path, mode: str, bf16: dict) -> None:
    if mode == "fp8":
        fp8 = json.loads(FP8_RECEIPT.read_bytes())
        expected = {name: row["sha256"] for name, row in fp8["files"].items()}
    else:
        expected = {Path(name).name: sha for name, sha in bf16["base_member_sha256"].items()}
    for name, sha in expected.items():
        if digest(base / name) != sha:
            raise ValueError(f"Base file differs from the pinned {mode} asset: {name}")
    allowed_metadata = {".cache", ".gitattributes", "README.md", "source.json"}
    if {path.name for path in base.iterdir()} - set(expected) - allowed_metadata:
        raise ValueError("Unexpected base model files")


def training_photos(root: Path, limit: int) -> list[Path]:
    manifest = json.loads(MANIFEST.read_bytes())
    eligible = [row for row in manifest if row["split"] == "dev" and row["dev_fold"] != 4]
    if len(eligible) != 791:
        raise ValueError("The measured Fold 4 adapter requires the original 791-photo training scope")
    images = []
    groups = set()
    for row in eligible:
        if row["group_id"] in groups:
            continue
        path = (root / row["image"]).resolve()
        if not path.is_relative_to(root) or digest(path) != row["sha256"]:
            raise ValueError("A selected training photo differs from the manifest")
        images.append(path)
        groups.add(row["group_id"])
        if len(images) == limit:
            return images
    raise ValueError("Insufficient independent training groups for the requested smoke count")


def restore_fp8_lora_storage(wrapped) -> int:
    """Place LoRA parameters in FP32 before loading their untouched stored values."""
    import torch

    fp8_targets = 0
    for layer in wrapped.modules():
        a, b = getattr(layer, "lora_A", None), getattr(layer, "lora_B", None)
        if a is None or b is None or "default" not in a or "default" not in b:
            continue
        if layer.get_base_layer().weight.dtype == torch.float8_e4m3fn:
            fp8_targets += 1
            a["default"].to(dtype=torch.float32)
            b["default"].to(dtype=torch.float32)
    return fp8_targets


def place_original_lora(model, adapter: Path, fp8: bool):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    import torch

    config = LoraConfig.from_pretrained(str(adapter), local_files_only=True)
    wrapped = get_peft_model(model, config).eval()
    originals = load_file(adapter / "adapter_model.safetensors", device="cpu")
    fp8_targets = restore_fp8_lora_storage(wrapped)
    if fp8_targets != (252 if fp8 else 0):
        raise ValueError("Unexpected number of LoRA targets on FP8 language layers")
    result = set_peft_model_state_dict(wrapped, originals)
    if result.unexpected_keys or any(".lora_" in key for key in result.missing_keys):
        raise ValueError("PEFT did not load every original LoRA tensor")
    loaded = get_peft_model_state_dict(wrapped)
    if (set(loaded) != set(originals) or len(loaded) != 520 or any(
            tensor.dtype != torch.float32 or originals[name].dtype != torch.float32
            or not torch.equal(tensor.detach().cpu(), originals[name])
            for name, tensor in loaded.items())):
        raise ValueError("The loaded adapter differs from the original FP32 A/B tensors")
    return wrapped, fp8_targets


def run(base: Path, adapter: Path, image_root: Path, mode: str, photos: int, token_limit: int) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this smoke inside the official Python 3.12 Linux kit image")
    installed = {name: version(name) for name in PINNED_VERSIONS}
    if installed != PINNED_VERSIONS:
        raise RuntimeError(f"The official offline kit wheel versions are required: {installed}")

    from PIL import Image, ImageOps
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9)
            or "L4" != torch.cuda.get_device_name().removeprefix("NVIDIA ")):
        raise RuntimeError("The precision smoke requires an NVIDIA L4")

    verify_start = time.perf_counter()
    bf16 = json.loads(BF16_RECEIPT.read_bytes())
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    if (adapter_sha != bf16["adapter_sha256"] or digest(adapter / "adapter_config.json")
            != "7ad0ec40f33ef84c4230542bcf9b99518e6c2466dba2b9ded179989282801f31"
            or digest(PROMPT) != bf16["prompt_sha256"]):
        raise ValueError("The smoke requires the exact measured Fold 4 adapter and v3 prompt")
    verify_base(base, mode, bf16)
    image_paths = training_photos(image_root, photos)
    verify_seconds = time.perf_counter() - verify_start

    load_start = time.perf_counter()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True, min_pixels=131072, max_pixels=393216)
    processor.image_processor.size = {"shortest_edge": 131072, "longest_edge": 393216}
    for attr, value in (("min_pixels", 131072), ("max_pixels", 393216)):
        if hasattr(processor.image_processor, attr):
            setattr(processor.image_processor, attr, value)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base), local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
    )
    model, fp8_targets = place_original_lora(model, adapter, fp8=mode == "fp8")
    model.generation_config.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, attr, None)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    messages = [
        {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "이 현장사진을 평가하고 지정된 JSON 하나만 출력한다."}]},
    ]
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    counts, prefill_seconds, generation_seconds = [], [], []
    token_hash = hashlib.sha256()
    with torch.inference_mode():
        for path in image_paths:
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            batch = processor(text=[prefix], images=[rgb], padding=True, return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            prefill = model(**batch, use_cache=False, logits_to_keep=1)
            if not torch.isfinite(prefill.logits).all().item():
                raise RuntimeError("Non-finite first-token logits")
            prefill_seconds.append(round(time.perf_counter() - start, 4))
            del prefill
            start = time.perf_counter()
            generated = model.generate(**batch, do_sample=False, max_new_tokens=token_limit, use_cache=True)
            tail = generated[0, batch["input_ids"].shape[-1]:].tolist()
            generation_seconds.append(round(time.perf_counter() - start, 4))
            if not tail:
                raise RuntimeError("The smoke did not generate any tokens")
            counts.append(len(tail))
            token_hash.update((json.dumps(tail, separators=(",", ":")) + "\n").encode())
            del batch, generated
    return {
        "precision": mode,
        "library_versions": installed,
        "photos": len(counts),
        "adapter_sha256": adapter_sha,
        "adapter_fp32_tensors_verified": 520,
        "fp8_lora_targets": fp8_targets,
        "pixel_cap": 393216,
        "generation_token_limit": token_limit,
        "base_file_verification_seconds": round(verify_seconds, 3),
        "base_adapter_processor_load_seconds": round(load_seconds, 3),
        "finite_prefill_rows": len(prefill_seconds),
        "prefill_seconds_sum": round(sum(prefill_seconds), 3),
        "first_prefill_seconds_including_cold_compile": prefill_seconds[0],
        "generation_seconds_sum": round(sum(generation_seconds), 3),
        "generated_tokens_total": sum(counts),
        "generated_token_sequence_sha256": token_hash.hexdigest(),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        "official_200_output_gate_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--images", type=Path, default=Path("data/train"))
    parser.add_argument("--mode", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--photos", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--max-new-tokens", type=int, choices=range(1, 621), default=64, metavar="1..620")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.adapter, args.images.resolve(), args.mode, args.photos, args.max_new_tokens)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
