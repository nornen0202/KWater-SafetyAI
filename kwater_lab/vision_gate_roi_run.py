"""Validate and run the frozen three-arm ROI test. Predictions stay in the private output directory."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image


ARMS = ("fixed", "roi", "roi_full")
ROOT = Path("data/vision-roi-48/reviewed-v1")
PROMPT = Path("kwater_lab/system_prompt_11code.txt")
ADAPTER = Path("data/vlm-11code-fold0-v2/adapter")
WITNESS = Path("results/vision-roi-review-20261127.json")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def checked_inputs(root, prompt, adapter):
    w = json.loads(WITNESS.read_bytes())
    index = root / "input-sha256-local.json"
    geometry = root / "full-geometry-local.json"
    for path, field in ((index, "input_index_sha256"), (geometry, "full_geometry_sha256"),
                        (prompt, "prompt_sha256"), (adapter / "adapter_model.safetensors", "adapter_sha256")):
        if digest(path) != w[field]:
            raise ValueError(f"Frozen input changed: {path}")
    expected = json.loads(index.read_bytes())
    expected.update({f"roi_full/{r['id']}.png": r["roi_full_sha256"] for r in json.loads(geometry.read_bytes())})
    if len(expected) != 144:
        raise ValueError("Invalid input hash index")
    rows = {arm: json.loads((root / f"{arm}.json").read_bytes()) for arm in ARMS}
    for i in range(48):
        sid = f"R{i + 1:02}"
        contexts = []
        for arm in ARMS:
            name = f"{arm}/{sid}.png"
            if len(rows[arm]) != 48 or rows[arm][i] != {"sample_id": sid, "image": name}:
                raise ValueError("Frozen input ID or order differs")
            path = root / name
            if digest(path) != expected[name]:
                raise ValueError(f"Changed panel image: {name}")
            with Image.open(path) as image:
                if image.size != (1024, 384):
                    raise ValueError("Changed pixel budget")
                contexts.append(image.convert("RGB").crop((0, 0, 512, 384)).tobytes())
        if contexts[0] != contexts[1] or contexts[0] != contexts[2]:
            raise ValueError("Full-scene pane differs across arms")
    return rows, w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/local-reports/vision-gate-v1"))
    args = parser.parse_args()
    rows, w = checked_inputs(args.root, args.prompt, args.adapter)
    print("verified 144 images, 48 identical left panes, adapter and prompt", flush=True)
    if args.base is None:
        return
    from .vlm_11code import infer_batched_vlm, load_qwen_model, load_qwen_processor
    args.output.mkdir(parents=True, exist_ok=True)
    base_files = sorted(args.base.glob("*.safetensors"))
    if not base_files:
        raise ValueError("Expected local Qwen3-VL-4B model weights")
    def stream_hash(path):
        with path.open("rb") as f:
            from hashlib import file_digest
            return file_digest(f, "sha256").hexdigest()
    identity = {p.name: stream_hash(p) for p in base_files}
    model = load_qwen_model(args.base, use_4bit=False, adapter_dir=args.adapter)
    proc = load_qwen_processor(args.base, 393216)
    for arm in ARMS:
        key = json.dumps([identity, w["adapter_sha256"], w["prompt_sha256"], "bf16", arm], sort_keys=True)
        _, stats = infer_batched_vlm(model, proc, rows[arm], args.root, args.prompt.read_text(encoding="utf-8"),
                                     batch_size=1, max_new_tokens=620, cache_path=args.output / f"{arm}.json",
                                     cache_key=key)
        (args.output / f"{arm}-stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
