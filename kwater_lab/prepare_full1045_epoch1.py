"""Prepare a standalone package input from the real full1045 epoch-one checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(source: Path, candidate: Path) -> None:
    import torch

    checkpoint = source / "epoch-1"
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True, mmap=True)
    configuration = state["configuration"]
    curriculum = configuration["experiment"]["curriculum_n"]
    accumulation = configuration["grad_accum"]
    steps = (curriculum + accumulation - 1) // accumulation
    cursor = {key: state[key] for key in ("next_epoch", "next_offset", "global_step")}
    manifest = source / "manifest_all_1045_train.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    train_ids = {row["sample_id"] for row in rows if row["split"] == "dev"}
    history = json.loads((source / "history.json").read_text(encoding="utf-8"))[:1]
    if (configuration["epochs"], configuration["val_fold"], steps, len(rows), len(train_ids)) != (2, -1, 85, 1045, 1045):
        raise ValueError("Expected the full1045, two-epoch, 170-step run")
    if cursor != {"next_epoch": 2, "next_offset": 0, "global_step": steps}:
        raise ValueError("Expected the preserved first epoch boundary")
    if len(history) != 1 or history[0]["epoch"] != 1:
        raise ValueError("Missing the original epoch-one history")
    with (checkpoint / "adapter_model.safetensors").open("rb") as handle:
        adapter_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    adapter = candidate / "adapter"
    adapter.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        os.link(checkpoint / name, adapter / name)
    os.link(manifest, candidate / manifest.name)
    summary = {
        "version": "vlm-11code-multimodal-qlora-v2-epoch-boundary",
        "checkpoint_kind": "epoch_boundary_snapshot_of_multi_epoch_schedule",
        "configuration": configuration,
        "val_fold": configuration["val_fold"],
        "train_photos": len(train_ids),
        "curriculum_examples": curriculum,
        "configured_epochs": configuration["epochs"],
        "completed_epochs": 1,
        "grad_accum": accumulation,
        "max_pixels": configuration["max_pixels"],
        "optimizer_steps": steps,
        "scheduled_total_steps": steps * configuration["epochs"],
        "resume_cursor": cursor,
        "history": history,
        "adapter_safetensors_sha256": adapter_sha,
    }
    (candidate / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Full1045 epoch-one checkpoint: {steps}/{steps * configuration['epochs']}, SHA-256 {adapter_sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.candidate)
