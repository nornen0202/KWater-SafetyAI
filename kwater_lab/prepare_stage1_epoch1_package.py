"""Prepare a standalone package input from a paused Stage 1 first-epoch checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from .package_submission import link_or_copy
from .stage1_pure_inspect import (
    ACCUM, BASE_CONFIG_SHA, BASE_REVISION, LR, PIXELS, PROMPTS, SCHEDULE_EPOCHS, SEED,
    curriculum_digest, digest, load_scope, replace_verify_slots,
)


def prepare(source: Path, candidate: Path, scope: str, variant: str, prompt_version: str) -> dict:
    import torch

    if candidate.exists():
        raise FileExistsError(candidate)
    checkpoint = source / "epoch-1"
    state_path = checkpoint / "training_state.pt"
    history_path = source / "history.json"
    weights = checkpoint / "adapter_model.safetensors"
    config = checkpoint / "adapter_config.json"
    for path in (state_path, weights, config, history_path):
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(f"Incomplete Stage 1 first-epoch checkpoint: {path}")

    train, _, _, manifest, mixed, _ = load_scope(scope)
    curriculum = mixed if variant == "mixed" else replace_verify_slots(mixed, train)[0]
    steps = math.ceil(len(curriculum) / ACCUM)
    val_fold = 2 if scope == "fold2" else -1
    expected = {
        "val_fold": val_fold, "epochs": SCHEDULE_EPOCHS, "grad_accum": ACCUM,
        "lr": LR, "max_pixels": PIXELS, "seed": SEED,
        "base_config_sha256": BASE_CONFIG_SHA,
        "prompt_sha256": digest(PROMPTS[prompt_version]),
        "curriculum_sha256": curriculum_digest(curriculum),
        "experiment": {
            "arm": f"stage1_{scope}_{variant}_{prompt_version}",
            "curriculum_n": len(curriculum), "base_revision": BASE_REVISION,
        },
    }
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    if state["configuration"] != expected:
        raise ValueError("Checkpoint configuration differs from the selected Stage 1 curriculum, prompt or base")
    cursor = {key: state[key] for key in ("next_epoch", "next_offset", "global_step")}
    history = state["history"]
    scheduled_steps = SCHEDULE_EPOCHS * steps
    if (cursor != {"next_epoch": 2, "next_offset": 0, "global_step": steps}
            or state["scheduler"]["last_epoch"] != steps
            or [record["epoch"] for record in history] != [1]
            or json.loads(history_path.read_bytes())[:1] != history
            or [(record["epoch"], record["step"], record["total_steps"]) for record in state["records"]]
            != [(1, step, scheduled_steps) for step in range(1, steps + 1)]):
        raise ValueError("Checkpoint is missing the ordered Stage 1 first-epoch boundary")
    del state

    summary = {
        "version": "vlm-11code-multimodal-qlora-v2-epoch-boundary",
        "checkpoint_kind": "epoch_boundary_snapshot_of_multi_epoch_schedule",
        "configuration": expected,
        "val_fold": val_fold,
        "train_photos": len(train),
        "curriculum_examples": len(curriculum),
        "configured_epochs": SCHEDULE_EPOCHS,
        "completed_epochs": 1,
        "grad_accum": ACCUM,
        "max_pixels": PIXELS,
        "optimizer_steps": steps,
        "scheduled_total_steps": scheduled_steps,
        "resume_cursor": cursor,
        "history": history,
        "adapter_safetensors_sha256": digest(weights),
    }
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=candidate.parent, prefix=".stage1-package-") as scratch:
        staged = Path(scratch) / "candidate"
        for path in (config, weights):
            link_or_copy(path, staged / "adapter" / path.name)
        if val_fold == -1:
            link_or_copy(manifest, staged / "manifest_all_1045_train.json")
        (staged / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        staged.rename(candidate)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scope", choices=("fold2", "full1045"), required=True)
    parser.add_argument("--curriculum", choices=("mixed", "pure-inspect"), required=True)
    parser.add_argument("--prompt-version", choices=tuple(PROMPTS), required=True)
    args = parser.parse_args()
    summary = prepare(args.source, args.candidate, args.scope, args.curriculum, args.prompt_version)
    print(f"Prepared Stage 1 epoch one at {summary['optimizer_steps']}/{summary['scheduled_total_steps']} steps. "
          f"Adapter SHA-256 {summary['adapter_safetensors_sha256']}")


if __name__ == "__main__":
    main()
