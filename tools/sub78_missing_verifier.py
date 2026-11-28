"""Run five blinded incomplete verifier requests with a hash-pinned handoff."""

import argparse
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
import math
import platform
from pathlib import Path
import time


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify_files(root, files):
    for name, expected in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Frozen file mismatch: {name}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base", type=Path, required=True)
    cli.add_argument("--input-sha256", required=True)
    cli.add_argument("--out", type=Path, default=Path("response.json"))
    cli.add_argument("--preflight-only", action="store_true")
    args = cli.parse_args()
    root = Path(__file__).resolve().parent
    manifest = root / "input.json"
    if digest(manifest) != args.input_sha256:
        raise ValueError("Request manifest differs from the approved SHA-256")
    data = json.loads(manifest.read_bytes())
    if args.out.exists():
        raise FileExistsError(args.out)
    verify_files(root, data["packet_files"])
    candidates = data["candidates"]
    if len(candidates) != 5 or len({row["sample_id"] for row in candidates}) != 5:
        raise ValueError("Expected the five distinct frozen inputs")
    verify_files(args.base, data["base_sha256"])
    if args.preflight_only:
        print("VERIFIED_PACKET " + digest(manifest))
        return
    import torch
    from kwater_lab.fold4_same_verifier import passes
    from kwater_lab.v3_turnaround import score_verifier_1token_margins
    from kwater_lab.vlm_11code import (
        load_qwen_4bit_or_bf16,
        seed_all,
        verify_specialist_candidates_with_vlm,
        verify_user_text,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized verifier GPU")
    for row in candidates:
        if sha256(verify_user_text(row["code"]).encode()).hexdigest() != row["verify_user_prompt_sha256"]:
            raise ValueError("Candidate user prompt mismatch")
    settings = data["recipe"]
    if settings != {
        "precision": "BF16",
        "min_pixels": 131072,
        "max_pixels": 393216,
        "batch_size": 2,
        "max_new_tokens": 280,
        "do_sample": False,
        "seed": 42,
    }:
        raise ValueError("Unexpected verification settings")
    seed_all(settings["seed"])
    t0 = time.perf_counter()
    model, proc = load_qwen_4bit_or_bf16(args.base, settings["max_pixels"], False, root / "adapter")
    if model.dtype != torch.bfloat16:
        raise ValueError("Verifier did not load in BF16")
    prompt = (root / "system_prompt.txt").read_text("utf-8")
    batch = settings["batch_size"]
    margins = score_verifier_1token_margins(model, proc, candidates, root, prompt, batch_size=batch)
    if len(margins) != len(candidates) or not all(map(math.isfinite, margins)):
        raise ValueError("Bad verifier margin results")
    passed = [
        row for row, margin in zip(candidates, margins, strict=True) if passes(row, margin, data["gate"])
    ]
    answers = verify_specialist_candidates_with_vlm(
        model,
        proc,
        passed,
        root,
        prompt,
        batch_size=batch,
        max_new_tokens=settings["max_new_tokens"],
    )
    if [row["sample_id"] for row in answers] != [row["sample_id"] for row in passed]:
        raise ValueError("Verifier response ordering changed")
    answer = {row["sample_id"]: row["raw_verify_text"] for row in answers}
    result = {
        "input_sha256": digest(manifest),
        "recipe": settings,
        "gate": data["gate"],
        "base_sha256_verified": data["base_sha256"],
        "adapter_sha256": data["packet_files"]["adapter/adapter_model.safetensors"],
        "system_prompt_sha256": data["packet_files"]["system_prompt.txt"],
        "runtime_versions": {pkg: version(pkg) for pkg in ("torch", "transformers", "peft")},
        "gpu": torch.cuda.get_device_name(0),
        "environment": {
            "os": platform.system(),
            "machine": platform.machine(),
            "torch_build": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "dtype": str(model.dtype),
        },
        "seconds_including_model_load": time.perf_counter() - t0,
        "rows": [
            {
                "key": row["sample_id"],
                "code": row["code"],
                "margin": m,
                "raw_verify_text": answer.get(row["sample_id"]),
            }
            for row, m in zip(candidates, margins, strict=True)
        ],
    }
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
