"""Run the local visual-token, rare-code, and guarded prompt sequence with one 8B load."""

import os


def adapter_state(layers):
    return tuple((tuple(layer.active_adapters), layer.disable_adapters, layer.merged,
                  (layer.get_base_layer().weight.dtype, layer.get_base_layer().weight.device),
                  (layer.lora_A["default"].weight.dtype, layer.lora_A["default"].weight.device),
                  (layer.lora_B["default"].weight.dtype, layer.lora_B["default"].weight.device))
                 for layer in layers)


def run_followups(model, processor, controls):
    from . import stage2_rare_base_probe as rare
    from . import stage2_representation_head as head
    from .fold4_same_verifier import read

    layers = rare.resident_layers(model, head.BASE)
    starting = adapter_state(layers)
    if any(active != ("default",) or disabled or merged for active, disabled, merged, *_ in starting):
        raise RuntimeError("Expected an active unmerged adapter before the rare-code probe")
    rare.run_with_model(model, processor, base=head.BASE, adapter=head.ADAPTER, manifest=rare.MANIFEST,
                        image_root=head.IMAGE_ROOT, out=rare.WORK, pilot_only=False)
    receipt = read(max((rare.WORK / "receipts").glob("*.json")))
    if adapter_state(layers) != starting or receipt["selected_photos"] != rare.PHOTOS or receipt["complete"] is not True:
        raise RuntimeError("The rare-code probe or original adapter restoration did not complete")
    print(f"STAGE2_ADAPTER_RESTORED {len(layers)} layers", flush=True)
    from . import stage1_local_resolution as stage1

    run_v4 = getattr(stage1, "run_v4_with_model", None)
    if run_v4 is None:
        print("STAGE1_V4_RUNNER_UNAVAILABLE", flush=True)
        return
    run_v4(model, processor, base=head.BASE, adapter=head.ADAPTER, feature_controls=controls)


def main():
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from . import stage2_representation_head as head

    torch.set_num_threads(1)
    print(f"STAGE2_LOCAL_COMBINED_PID {os.getpid()}", flush=True)
    head.extract(head.WORK, head.BASE, "all", None, after_extract=run_followups)


if __name__ == "__main__":
    main()
