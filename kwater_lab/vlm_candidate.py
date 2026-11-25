"""Compare the two 4B generation backbones on the same published photo and Korean prompts."""

import argparse
import json
from pathlib import Path

from .visual_probe import (CONDITIONS, DEFAULT_CACHE, FILES, MODEL_REV, RUNTIME, digest, encoded_image,
                           fetch_file, local_server, run_trial)


REPO = "unsloth/Qwen3.5-4B-GGUF"
REV = "e87f176479d0855a907a41277aca2f8ee7a09523"
WEIGHTS = {
    "Qwen3.5-4B-Q4_K_M.gguf": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    "mmproj-F16.gguf": "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864",
}


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, sha in WEIGHTS.items():
        fetch_file(f"https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true", folder / name, sha)
        print(f"Verified {name}", flush=True)


def summarize(trials, expected):
    summary = {}
    for name, record in trials.items():
        pred = set(record["predicted_codes"] or [])
        summary[name] = {"valid": record["valid"], "expected": len(expected), "proposed": len(pred),
                         "tp": len(pred & expected), "fp": len(pred - expected), "fn": len(expected - pred),
                         "seconds": record["elapsed_s"], "finish_reason": record["finish_reason"]}
    return summary


def run(root, cache, folder):
    local = root / "data/vlm-candidate"
    local.mkdir(parents=True, exist_ok=True)
    reference = root / "results/visual_probe.json"
    ref = json.loads(reference.read_text(encoding="utf-8"))
    image, label = cache / "public-example.jpg", cache / "public-example-label.json"
    if (ref["revision"] != MODEL_REV or ref["runtime"] != RUNTIME or
            digest(image) != ref["sha256"][image.name] or digest(image) != FILES[image.name] or
            digest(label) != ref["label_sha256"] or
            any(ref["conditions"][k]["prompt"] != p for k, p in CONDITIONS.items())):
        raise ValueError("Public example or comparison prompts differ from the recorded reference")
    expected = set(json.loads(label.read_text(encoding="utf-8"))["target"]["hazards"])
    photo = encoded_image(image)
    flags = ("--reasoning", "off", "--chat-template-kwargs", '{"enable_thinking":false}')
    with local_server(folder, local / "model.log", weights=WEIGHTS, runtime_root=cache,
                      extra_flags=flags, startup_timeout=900) as (port, key, runtime):
        if runtime["server_sha256"] != ref["server_sha256"]:
            raise ValueError("Runtime binary differs from the comparison baseline")
        trials = {name: run_trial(port, key, photo, prompt) for name, prompt in CONDITIONS.items()}
    from .adapter_scale_pilot import write_json
    write_json(local / "generations.json", trials)
    report = {"scope": "One previously exposed organizer illustration; two zero-shot Q4_K_M backbones, no LoRA",
              "photos": 1, "matched_runtime": RUNTIME, "seed": 42, "temperature": 0, "max_tokens": 384,
              "source": "Qwen/Qwen3.5-4B", "conversion": REPO, "revision": REV, "weights_sha256": WEIGHTS,
              "reference_sha256": digest(reference), "local_raw_sha256": digest(local / "generations.json"),
              "runtime": runtime, "conditions": {"Qwen3-VL-4B": summarize(ref["conditions"], expected),
                                               "Qwen3.5-4B": summarize(trials, expected)}}
    write_json(root / "results/vlm_candidate_public.json", report)
    print(json.dumps(report["conditions"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prepare", action="store_true", help="Download and verify the GGUF pair before running")
    parser.add_argument("--prepare-only", action="store_true", help="Download and verify without starting a GPU model")
    args = parser.parse_args()
    folder = args.root / "data/encoders/qwen3.5-4b-q4"
    if args.prepare or args.prepare_only:
        prepare(folder)
    if not args.prepare_only:
        run(args.root, args.reference_cache, folder)
