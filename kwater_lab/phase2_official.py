"""Instrument the submitted Qwen and record diagnostic decisions per photo."""

import argparse
from contextlib import ExitStack
import gc
from importlib import import_module
import os
import time
from unittest.mock import patch

from .adapter_scale_pilot import load_entry, weight_reader
from .phase2_paths import file_stem
from .phase2_panel import PROJECT, RUN, order, read, registry, write


def generate(model, proc, image, system, user, limit, **kwargs):
    import torch

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user}],
        },
    ]
    prefix = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    b = proc(text=[prefix], images=[image], padding=True, return_tensors="pt").to(
        model.device
    )
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        tokens = model.generate(
            **b, do_sample=False, max_new_tokens=limit, use_cache=True
        )[0, b["input_ids"].shape[-1] :]
    text = proc.decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    torch.cuda.synchronize()
    return {
        "text": text,
        "seconds": time.perf_counter() - start,
        "new_tokens": int(len(tokens)),
        "prompt_tokens": int(b["input_ids"].shape[-1]),
        "vision_grid": b["image_grid_thw"][0].tolist(),
        "limit_reached": len(tokens) == limit,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def offload_vision_blocks(model):
    import torch
    from accelerate import cpu_offload

    visual = next(
        m
        for name, m in model.named_modules()
        if name.endswith(".visual") and hasattr(m, "blocks")
    )
    for block in list(visual.blocks)[-8:]:
        cpu_offload(
            block,
            execution_device=torch.device("cuda:0"),
            preload_module_classes=[type(block).__name__],
        )
    model._p38_vision_offload = True
    torch.cuda.empty_cache()


def save_missing(model, proc, ids, rows):
    import torch

    base = PROJECT / "data/baseline"
    entry = load_entry(base)
    laws, cases = (
        entry.load_rows(base / f"assets/{f}.jsonl") for f in ("laws", "train")
    )

    for i, s in enumerate(ids):
        dest = RUN / "A" / (file_stem(s) + ".json")
        if dest.exists():
            continue
        anon = order("kwater/scale", s)[:32]
        t0 = time.perf_counter()
        capture = {}
        original = entry.parse_prediction

        def capture_parse(txt):
            parsed, err = original(txt)
            capture.update(
                text=txt,
                parse_error=err,
                schema_errors=[] if err else entry.schema_errors(parsed),
            )
            return parsed, err

        failures = RUN / "A-attempts" / (file_stem(s) + ".json")
        attempts = read(failures) if failures.exists() else []
        success = False
        for attempt in range(3):
            gc.collect()
            torch.cuda.empty_cache()
            if attempt == 2 and not getattr(model, "_p38_vision_offload", False):
                offload_vision_blocks(model)
            capture.clear()
            torch.cuda.reset_peak_memory_stats()
            try:
                with (
                    patch.object(entry, "load_model", return_value=(model, proc)),
                    patch.object(entry, "parse_prediction", capture_parse),
                ):
                    pred = entry.infer_batched(
                        [{"sample_id": anon, "image": rows[s]["image"]}],
                        PROJECT / "data/train",
                    )[0]
                success = True
            except RuntimeError as err:
                if "out of memory" not in str(err).lower():
                    raise
                attempts.append(
                    dict(
                        at=time.time(),
                        pid=os.getpid(),
                        error=str(err),
                        attempt=attempt + 1,
                        seconds_since_row_start=time.perf_counter() - t0,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        vision_offload=getattr(model, "_p38_vision_offload", False),
                    )
                )
                write(failures, attempts)
            if success:
                break
        if not success:
            print("A_PENDING_OOM", s, flush=True)
            continue
        torch.cuda.synchronize()
        valid = not entry.validate_prediction(pred["prediction"])
        formatted = None
        if valid:
            p = RUN / "A-format.jsonl"
            entry.format_submission([pred], laws, cases, p)
            formatted = entry.load_rows(p)[0]["prediction"]
        write(
            dest,
            dict(
                sample_id=s,
                final=entry.convert(anon, formatted),
                raw=pred["prediction"],
                valid=valid,
                prior_failed_attempts=len(attempts),
                vision_offload=getattr(model, "_p38_vision_offload", False),
                seconds=time.perf_counter() - t0,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                **capture,
            ),
        )
        print(
            "A_SAVED",
            i + 1,
            "/",
            len(ids),
            round(time.perf_counter() - t0, 2),
            flush=True,
        )


def run(sam_wrapper=None):
    os.environ["KW_BATCH"] = "1"
    import torch
    from PIL import Image, ImageOps

    plan, rows = read(RUN / "plan.json"), registry()
    base = PROJECT / "data/baseline"
    entry = load_entry(base)
    if sam_wrapper:
        # Complete imports before the handoff. CUDA weights have not been allocated.
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        print(
            "CPU_IMPORT_READY",
            Qwen3VLForConditionalGeneration.__name__,
            AutoProcessor.__name__,
            PeftModel.__name__,
            flush=True,
        )
        import psutil

        try:
            p = psutil.Process(sam_wrapper)
            if not any("run_phase2_sam.ps1" in a.lower() for a in p.cmdline()):
                raise RuntimeError("The reserved SAM wrapper does not match")
            p.wait()
        except psutil.NoSuchProcess:
            pass
    entry.seed_all(42)
    start = time.perf_counter()
    print("LOAD_START", flush=True)
    low_commit = (RUN / "official-loader-byte-check.json").is_file()
    with ExitStack() as stack:
        if low_commit:
            import transformers

            # The official greedy decoder never calls the optional assisted-decoding ROC code.
            with patch.object(
                transformers.utils, "is_sklearn_available", return_value=False
            ):
                import_module("transformers.generation.candidate_generator")
            from transformers import modeling_utils
            from .phase2_tensor_reader import BF16TensorFile

            stack.enter_context(
                patch.object(modeling_utils, "safe_open", BF16TensorFile)
            )
            stack.enter_context(
                patch.object(
                    modeling_utils, "caching_allocator_warmup", return_value=None
                )
            )
            print(
                "LOAD_STREAMING_BF16 no full-shard mapping or allocator preallocation",
                flush=True,
            )
        else:
            stack.enter_context(weight_reader("pread-cpu"))
        model, proc = entry.load_model(
            str(base / "base_model"), "sdpa", str(base / "adapter"), 384 * 32 * 32
        )
    if low_commit:
        offload_vision_blocks(model)
    model.eval()
    proc.tokenizer.padding_side = "left"
    print("LOADED", time.perf_counter() - start, flush=True)
    runtime_path = RUN / "official-runtime.json"
    if runtime_path.is_file():
        runtime_path = RUN / f"official-runtime-{os.getpid()}.json"
    write(
        runtime_path,
        {
            "load_seconds": time.perf_counter() - start,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "precision": "bf16",
            "batch": 1,
            "attention": "sdpa",
            "adapter_scale": 1.0,
            "weight_reader": "streaming-bf16" if low_commit else "pread-cpu",
            "cuda_compute_vision_blocks_stored_on_cpu": 8 if low_commit else 0,
            "prompt": (base / "system_prompt.txt").read_text(encoding="utf8"),
        },
    )

    def image_for(s):
        with Image.open(PROJECT / "data/train" / rows[s]["image"]) as im:
            return ImageOps.exif_transpose(im).convert("RGB")

    save_missing(model, proc, [plan["prototype"][0]] + plan["panel"], rows)
    from .phase2_review import review

    review(model, proc, image_for)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-sam-wrapper", type=int)
    run(parser.parse_args().wait_sam_wrapper)
