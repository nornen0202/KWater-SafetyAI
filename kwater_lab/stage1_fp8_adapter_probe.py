"""Exercise PEFT 0.17.1 adapter dtype placement on CPU, without a base checkpoint or CUDA kernel."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .stage1_fp8_l4_smoke import restore_fp8_lora_storage


REPRESENTATIVE_LAYERS = {
    "language_projection": "language_model.layers.0.self_attn.q_proj",
    "vision_merger": "visual.merger.linear_fc1",
    "deepstack_merger": "visual.deepstack_merger_list.0.linear_fc1",
}


def probe(adapter: Path) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors import safe_open
    from torch import nn

    if version("peft") != "0.17.1":
        raise ValueError("Run the placement probe with the kit submission's PEFT 0.17.1 wheel")
    torch.set_num_threads(2)

    class ZeroProjection(nn.Linear):
        # Storage has the same dtype and shape as the base projection. A zero forward isolates PEFT's LoRA branch.
        def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
            super().__init__(in_features, out_features, bias=False, device="meta")
            self.weight = nn.Parameter(torch.zeros((1, 1), dtype=dtype).expand(out_features, in_features), requires_grad=False)

        def forward(self, inputs):
            return inputs.new_zeros((*inputs.shape[:-1], self.out_features))

    class Model(nn.Module):
        def __init__(self, modules):
            super().__init__()
            self.projections = nn.ModuleDict(modules)

        def forward(self, inputs):
            return {name: layer(inputs[name]) for name, layer in self.projections.items()}

    config = json.loads((adapter / "adapter_config.json").read_bytes())
    weight_path = adapter / "adapter_model.safetensors"
    weights = {}
    inputs = {}
    shapes = {}
    with safe_open(weight_path, framework="pt", device="cpu") as archive:
        for label, layer in REPRESENTATIVE_LAYERS.items():
            a, b = (archive.get_tensor(f"base_model.model.model.{layer}.lora_{matrix}.weight") for matrix in ("A", "B"))
            if a.dtype != torch.float32 or b.dtype != torch.float32 or a.shape[0] != b.shape[1] or a.shape[0] != config["r"]:
                raise ValueError(f"Expected original FP32 A/B pair and checkpoint rank: {label}")
            weights[f"base_model.model.projections.{label}.lora_A.weight"] = a
            weights[f"base_model.model.projections.{label}.lora_B.weight"] = b
            shapes[label] = (a.shape[1], b.shape[0])
            inputs[label] = torch.linspace(-1, 1, steps=a.shape[1], dtype=torch.bfloat16).reshape(1, -1)

    def build(language_dtype, restore_original_dtype=False):
        wrapped = get_peft_model(
            Model({label: ZeroProjection(*shape, language_dtype if label == "language_projection" else torch.bfloat16)
                   for label, shape in shapes.items()}),
            LoraConfig(r=config["r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                       target_modules=list(shapes), inference_mode=True),
        ).eval()
        if restore_original_dtype and restore_fp8_lora_storage(wrapped) != 1:
            raise ValueError("Expected exactly one target on FP8 base storage")
        result = set_peft_model_state_dict(wrapped, weights)
        if result.unexpected_keys or any(".lora_" in key for key in result.missing_keys):
            raise ValueError("PEFT probe did not load every original adapter tensor")
        return wrapped

    reference = build(torch.bfloat16)
    stock = build(torch.float8_e4m3fn)
    restored = build(torch.float8_e4m3fn, restore_original_dtype=True)
    tensors = {
        mode: get_peft_model_state_dict(model)
        for mode, model in (("reference", reference), ("stock_fp8", stock), ("restored_fp8", restored))
    }
    if any(set(saved) != set(weights) for saved in tensors.values()):
        raise ValueError("Unexpected PEFT adapter state dict keys")
    with torch.inference_mode():
        reference_result = reference(inputs)
        restored_result = restored(inputs)
        try:
            stock_result = stock(inputs)
        except RuntimeError as error:
            stock_forward = {"cpu_error": str(error).splitlines()[0]}
        else:
            stock_forward = {label: torch.equal(stock_result[label], reference_result[label]) for label in shapes}

    findings = {}
    for label in shapes:
        keys = [key for key in weights if f".{label}." in key]
        findings[label] = {
            "a_b_storage_dtypes": {mode: [str(saved[key].dtype) for key in keys] for mode, saved in tensors.items()},
            "a_b_match_original": {mode: [torch.equal(saved[key].float(), weights[key]) for key in keys] for mode, saved in tensors.items()},
            "restored_output_matches_bf16_reference": torch.equal(restored_result[label], reference_result[label]),
            "bf16_reference_nonzero_output_values": int(reference_result[label].count_nonzero()),
        }
    with weight_path.open("rb") as source:
        adapter_sha = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "version": "stage1-fp8-adapter-cpu-probe-v1",
        "peft": version("peft"),
        "torch": version("torch"),
        "adapter_sha256": adapter_sha,
        "method": "Stock PEFT LoRA wrapping and state-dict loading with original FP32 A/B tensors for three projections. Zero base forward isolates the LoRA branch. The restoration branch sets FP32 destination parameters before copying original checkpoint tensors.",
        "projection_results": findings,
        "stock_fp8_wrapper_cpu_forward": stock_forward,
        "official_kit_l4_kernel_or_model_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(args.adapter), indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
