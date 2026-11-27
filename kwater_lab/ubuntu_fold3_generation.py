"""Official eight-code baseline on canonical Fold3; prepare, predict, export.

Run: .venv/bin/python -m kwater_lab.ubuntu_fold3_generation prepare
All sample outputs stay under data/. No evaluation gold is read.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageOps

from .ubuntu_fold3_generation_protocol import (
    BASE,
    MAX_PIXELS,
    MAX_TOKENS,
    ROOT,
    RUN,
    BaselineError,
    export,
    load_entry,
    load_record,
    prepare,
    seal,
    sha256,
    stable_hash,
    verify_plan,
    write_once,
)


def predict(run: Path, max_calls: int = 0) -> None:
    """Generate only missing exact keys under the shared exclusive GPU lock."""
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    plan = verify_plan(run)
    pending = [row for row in plan["rows"] if load_record(run, row) is None]
    if not pending:
        seal(run, plan)
        print("Complete exact cache sealed; no model loaded", flush=True)
        return
    import torch

    from .vlm_11code import load_qwen_model, load_qwen_processor

    entry = load_entry()
    lock_path = ROOT / ".runtime/ubuntu-weekend-gpu.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (ROOT / ".runtime/WEEKEND_STOP").exists():
            print("Stop marker present; no model loaded", flush=True)
            return
        if torch.cuda.get_device_name(0).find("3050") < 0:
            raise BaselineError("Preregistered local RTX3050 required")
        entry.seed_all(42)
        processor = load_qwen_processor(BASE / "base_model", MAX_PIXELS)
        processor.tokenizer.padding_side = "left"
        started = time.perf_counter()
        model = load_qwen_model(
            BASE / "base_model", use_4bit=True, adapter_dir=BASE / "adapter"
        )
        model.eval()
        generation_config = model.generation_config
        if generation_config is None:
            raise BaselineError("Model decoding configuration is missing")
        print(f"Model load seconds: {time.perf_counter() - started:.3f}", flush=True)
        for row in pending[: max_calls or None]:
            if (ROOT / ".runtime/WEEKEND_STOP").exists():
                print("Stop marker present; completed records retained", flush=True)
                return
            if load_record(run, row) is not None:
                continue
            path = ROOT / "data/train" / row["image"]
            if sha256(path) != row["sha256"]:
                raise BaselineError("Photo changed before generation")
            with Image.open(path) as image:
                oriented = ImageOps.exif_transpose(image, in_place=False)
                if oriented is None:
                    raise BaselineError("EXIF transpose did not return an image")
                pixels = oriented.convert("RGB")
            batch = processor(
                text=[plan["protocol"]["prefix"]],
                images=[pixels],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(
                    **batch, do_sample=False, max_new_tokens=MAX_TOKENS, use_cache=True
                )
            new_ids = ids[0, batch["input_ids"].shape[-1] :]
            text = processor.decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            parsed, error = entry.parse_prediction(text)
            errors = [] if error else entry.schema_errors(parsed)
            prediction = {} if error or errors else parsed
            fallback = bool(entry.validate_prediction(prediction))
            torch.cuda.synchronize()
            eos = generation_config.eos_token_id
            ended = (
                int(new_ids[-1]) in (eos if isinstance(eos, list) else [eos])
                if len(new_ids)
                else False
            )
            payload = {
                "text": text,
                "prediction": prediction,
                "parsed_prediction": parsed,
                "parse_error": error,
                "schema_errors": errors,
                "normal_fallback": fallback,
                "new_tokens": len(new_ids),
                "truncated_at_cap": len(new_ids) >= MAX_TOKENS and not ended,
                "seconds": time.perf_counter() - tick,
                "image_grid_thw": batch["image_grid_thw"].tolist(),
                "prompt_tokens": int(batch["input_ids"].shape[-1]),
                "encoded_input_sha256": stable_hash(
                    {
                        "ids": batch["input_ids"].tolist(),
                        "grid": batch["image_grid_thw"].tolist(),
                        "pixels": hashlib.sha256(
                            batch["pixel_values"]
                            .detach()
                            .cpu()
                            .float()
                            .numpy()
                            .tobytes()
                        ).hexdigest(),
                    }
                ),
                "device": torch.cuda.get_device_name(0),
                "cuda": str(__import__("torch.version", fromlist=["cuda"]).cuda),
            }
            write_once(
                run / "records" / f"{row['image_id']}.json",
                {
                    "sample_id": row["sample_id"],
                    "image_id": row["image_id"],
                    "input_identity": row["input_identity"],
                    "payload": payload,
                    "payload_sha256": stable_hash(payload),
                },
            )
            print(
                json.dumps(
                    {
                        "completed": len(list((run / "records").glob("*.json"))),
                        "seconds": payload["seconds"],
                        "new_tokens": len(new_ids),
                    }
                ),
                flush=True,
            )
            del batch, ids, new_ids
    if all(load_record(run, row) is not None for row in plan["rows"]):
        seal(run, plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "predict", "export"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    if args.max_calls < 0:
        parser.error("--max-calls must be nonnegative")
    if not args.run.resolve().is_relative_to((ROOT / "data").resolve()):
        parser.error("--run must stay inside the private project data/ directory")
    match args.action:
        case "prepare":
            prepare(args.run)
        case "predict":
            predict(args.run, args.max_calls)
        case "export":
            export(args.run)


if __name__ == "__main__":
    main()
