"""Fresh QLoRA decision supervision with atomic optimizer/RNG recovery.

Run: .venv/bin/python -m kwater_lab.ubuntu_focused_verifier prep|train|resume
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, TypedDict

import torch
from transformers import PreTrainedModel

from .spatial_probe import digest
from .ubuntu_focused_curriculum import (
    CONTROL_HORIZON,
    CONTROL_WARMUP,
    GRAD_ACCUM,
    LR,
    MAX_PIXELS,
    SEED,
    STEPS,
    FocusedError,
    prepare,
)
from .vlm_11code import LORA_TARGET_REGEX, load_qwen_4bit_or_bf16, seed_all

DEFAULT_OUTPUT: Final = Path("data/ubuntu3050-runs/qlora-inner12-focused-verifier")


class StepRecord(TypedDict, total=False):
    step: int
    examples: int
    supervised_tokens: int
    accept: int
    reject: int
    loss: float
    applied_lr: list[float]
    grad_norm: float
    sample_ids: list[str]


def decision_loss(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    """Supervise exactly the next decision token; no prompt, JSON, rationale or EOS loss."""
    target = torch.tensor([target_id], dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits[:, -1, :].float(), target)


def train(root: Path, output: Path, resume: bool = False) -> None:
    """Run 96 unique conditional decisions once with the control's fresh QLoRA setup."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    train_dir = output / "training"
    if train_dir.exists() and not resume:
        raise FocusedError("Fresh training requires a new training directory")
    checkpoints = sorted(
        train_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if resume and not checkpoints:
        raise FocusedError(
            "Resume requires a completed checkpoint; use a fresh output if initialization was interrupted"
        )
    curriculum, encoder, token_ids = prepare(root, output)
    if len(curriculum) != STEPS * GRAD_ACCUM:
        raise FocusedError("Curriculum must fill exactly 12 complete optimizer steps")
    saved = None
    if resume:
        saved = torch.load(
            checkpoints[-1] / "training_state.pt", map_location="cpu", weights_only=True
        )
        if saved["preparation_sha256"] != digest(output / "preparation.json"):
            raise FocusedError(
                "Resume preparation, source, or curriculum identity changed"
            )
    seed_all(SEED)
    model, _ = load_qwen_4bit_or_bf16(
        root / "data/baseline/base_model", MAX_PIXELS, use_4bit=True
    )
    if not isinstance(model, PreTrainedModel):
        raise FocusedError(
            "Expected a fresh base model without an existing PEFT adapter"
        )
    if resume:
        model = PeftModel.from_pretrained(
            model, str(checkpoints[-1]), is_trainable=True
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=LORA_TARGET_REGEX,
            ),
        )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    parameters = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    if not parameters or not all(
        ".lora_" in name and re.fullmatch(LORA_TARGET_REGEX, name.split(".lora_")[0])
        for name, _ in parameters
    ):
        raise FocusedError("Unexpected trainable parameters")
    if not resume:
        initial = hashlib.sha256()
        for name, tensor in sorted(parameters, key=lambda item: item[0]):
            initial.update(
                repr((name, tuple(tensor.shape), str(tensor.dtype))).encode()
            )
            initial.update(
                tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        reference = (
            root
            / "data/ubuntu3050-runs/qlora-inner12-mined-verifier/initialization-local.json"
        )
        expected = json.loads(reference.read_text())["initial_trainable_lora_sha256"]
        if initial.hexdigest() != expected:
            raise FocusedError(
                "Fresh LoRA initialization differs from the mined arm's recorded initialization"
            )
        train_dir.mkdir()
        (output / "initialization-local.json").write_text(
            json.dumps(
                {
                    "initial_trainable_lora_sha256": initial.hexdigest(),
                    "matches_mined_initialization": True,
                    "reference_receipt_sha256": digest(reference),
                    "control_initialization": "recipe match only; original control has no t=0 hash",
                }
            )
            + "\n"
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in parameters if "visual." not in n],
                "lr": LR,
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in parameters if "visual." in n],
                "lr": LR * 0.65,
                "weight_decay": 0.01,
            },
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, CONTROL_WARMUP, CONTROL_HORIZON
    )
    records: list[StepRecord] = []
    start_step = 0
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        records, start_step = saved["records"], saved["step"]
        restore_step_log(train_dir, records)

    def save_checkpoint(step: int) -> None:
        with TemporaryDirectory(dir=train_dir, prefix=".checkpoint-") as temp:
            staged = Path(temp) / "checkpoint"
            model.save_pretrained(str(staged))
            encoder.processor.save_pretrained(str(staged))
            torch.save(
                {
                    "step": step,
                    "records": records,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "preparation_sha256": digest(output / "preparation.json"),
                },
                staged / "training_state.pt",
            )
            staged.rename(train_dir / f"checkpoint-{step}")

    model.train()
    if not resume:
        save_checkpoint(0)
    for step in range(start_step + 1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        group = curriculum[(step - 1) * GRAD_ACCUM : step * GRAD_ACCUM]
        losses = []
        for example in group:
            batch = encoder.encode(example).to("cuda")
            result = model(**batch, logits_to_keep=1, use_cache=False)
            loss = decision_loss(result.logits, token_ids[0 if example.accepted else 1])
            if not torch.isfinite(loss):
                raise FocusedError(f"Nonfinite loss on {example.sample_id}")
            (loss / GRAD_ACCUM).backward()
            losses.append(float(loss.detach().item()))
            del result, batch, loss
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for _, p in parameters], 1.0, error_if_nonfinite=True
        )
        applied_lr = [float(group["lr"]) for group in optimizer.param_groups]
        optimizer.step()
        scheduler.step()
        record: StepRecord = {
            "step": step,
            "examples": len(group),
            "supervised_tokens": len(group),
            "accept": sum(x.accepted for x in group),
            "reject": sum(not x.accepted for x in group),
            "loss": sum(losses) / GRAD_ACCUM,
            "applied_lr": applied_lr,
            "grad_norm": float(grad_norm.item()),
            "sample_ids": [x.sample_id for x in group],
        }
        records.append(record)
        with (train_dir / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            json.dumps(
                {key: value for key, value in record.items() if key != "sample_ids"}
            ),
            flush=True,
        )
        if step in (6, 12):
            save_checkpoint(step)


def restore_step_log(train_dir: Path, records: Sequence[StepRecord]) -> None:
    """Atomically discard uncheckpointed step records when replaying interrupted updates."""
    with TemporaryDirectory(dir=train_dir, prefix=".steps-") as temp:
        staged = Path(temp) / "steps.jsonl"
        staged.write_text("".join(json.dumps(record) + "\n" for record in records))
        staged.replace(train_dir / "steps.jsonl")


def main() -> None:
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in {"prep", "train", "resume"}:
        raise SystemExit(
            "Usage: python -m kwater_lab.ubuntu_focused_verifier prep|train|resume [local-output-directory]"
        )
    root = Path.cwd()
    output = (Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_OUTPUT).resolve()
    if sys.argv[1] in {"train", "resume"}:
        train(root, output, resume=sys.argv[1] == "resume")
    else:
        curriculum, _, tokens = prepare(root, output)
        print(
            json.dumps(
                {
                    "prepared_examples": len(curriculum),
                    "decision_token_ids": tokens,
                    "receipt": str(output / "preparation.json"),
                }
            )
        )


if __name__ == "__main__":
    main()
