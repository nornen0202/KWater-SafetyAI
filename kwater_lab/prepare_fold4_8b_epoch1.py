"""Prepare the real Seoul Fold 4 Qwen3-VL-8B first-epoch adapter for research packaging."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(experiment: Path, candidate: Path) -> None:
    fit = experiment / "fit"
    summary = json.loads((fit / "training_summary.json").read_bytes())
    generation = json.loads((experiment / "eval1/generation.json").read_bytes())
    steps = [json.loads(line) for line in (fit / "steps.jsonl").read_bytes().splitlines() if line]
    if (summary["epochs"], summary["val_fold"], summary["train_photos"], summary["optimizer_steps"],
        generation["photos"], summary["curriculum_examples"], summary["grad_accum"]) != (2, 4, 791, 138, 175, 1092, 16):
        raise ValueError("Expected the actual two-epoch Seoul Fold 4 run")
    if [(step["epoch"], step["step"], step["total_steps"]) for step in steps] != [
        (int(number > 69) + 1, number, 138) for number in range(1, 139)
    ]:
        raise ValueError("Missing ordered first-epoch evidence in the actual training log")
    if [item["epoch"] for item in summary["history"]] != [1, 2]:
        raise ValueError("Missing the actual first-epoch summary")
    if summary["configuration"]["base_config_sha256"] != generation["base_config_sha256"]:
        raise ValueError("Training and evaluation base identities differ")
    source = fit / "epoch-1"
    with (source / "adapter_model.safetensors").open("rb") as handle:
        weights_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    if weights_sha != generation["adapter_sha256"]:
        raise ValueError("Adapter is not the exact checkpoint measured in the 175-photo evaluation")
    adapter = candidate / "adapter"
    adapter.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        os.link(source / name, adapter / name)

    metadata = {key: summary[key] for key in ("configuration", "val_fold", "train_photos", "curriculum_examples",
                                               "grad_accum", "max_pixels")}
    metadata.update(
        version="vlm-11code-multimodal-qlora-v2-epoch-boundary",
        checkpoint_kind="epoch_boundary_snapshot_of_multi_epoch_schedule",
        configured_epochs=2,
        completed_epochs=1,
        optimizer_steps=69,
        scheduled_total_steps=138,
        resume_cursor={"next_epoch": 2, "next_offset": 0, "global_step": 69},
        resume_cursor_evidence="Derived from the observed end of epoch 1 at step 69 in the ordered remote training log.",
        history=summary["history"][:1],
        adapter_safetensors_sha256=weights_sha,
    )
    (candidate / "training_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exact research 8B Fold 4 first epoch, 69/138 updates, adapter {weights_sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.experiment, args.candidate)
