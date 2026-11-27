"""Matched 32-photo QLoRA training and fresh-process adapter reload probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def prepare(root, panel):
    manifest = json.loads((root / "data/clean-split-v1/split_manifest_full_sealed.json").read_text("utf-8"))
    links = json.loads((root / "data/fold4-crosssplit-visual-review/fold4-group-exclusions-local.json").read_text("utf-8"))
    used = {r["train_group_id"] for r in links["confirmed_cross_group_links"]}
    eligible = [r for r in manifest if r["split"] == "dev" and r["dev_fold"] != 4 and not r["exposed_prior"]]
    ranked = sorted(eligible, key=lambda r: hashlib.sha256(("scale-pilot-42/" + r["sample_id"]).encode()).digest())
    train, smoke = [], []
    for target, status, count in [(train, "hazard", 22), (train, "no_visible_hazard", 10),
                                  (smoke, "hazard", 1), (smoke, "no_visible_hazard", 1)]:
        selected = 0
        for row in ranked:
            if row["status"] == status and row["group_id"] not in used:
                used.add(row["group_id"])
                target.append(row)
                selected += 1
                if selected == count:
                    break
        if selected != count:
            raise ValueError(f"Insufficient distinct groups for {status}")
    panel.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("smoke", smoke)]:
        (panel / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")


def inspect_only(rows, annotations, seed):
    from .vlm_11code import build_training_curriculum
    return [r for r in build_training_curriculum(rows, annotations, seed) if r["kind"] == "inspect"]


def run(args):
    from .vlm_11code import infer_batched_vlm, load_qwen_4bit_or_bf16, train_multimodal_qlora
    root, panel, out = args.root.resolve(), args.panel.resolve(), args.out.resolve()
    base = args.base.resolve(strict=True)
    prompt = root / "kwater_lab/system_prompt_v3.txt"
    if args.action == "train":
        os.environ["KWATER_LOCAL_BASE_MODEL"] = str(base)
        train_multimodal_qlora(root, panel / "train.json", prompt, out, val_fold=4,
                               epochs=1, grad_accum=2, max_pixels=393216, seed=42,
                               checkpoint_steps=16, curriculum_builder=inspect_only,
                               experiment_identity={"experiment": "matched_scale_32"})
    else:
        rows = json.loads((panel / "smoke.json").read_text("utf-8"))
        model, proc = load_qwen_4bit_or_bf16(base, 393216, True, out / "adapter")
        predictions, _ = infer_batched_vlm(model, proc, rows, root / "data/train",
                                           prompt.read_text("utf-8"), batch_size=1, max_new_tokens=620)
        result = {
            "photos": len(rows),
            "parseable": sum(p["parsed"] is not None for p in predictions),
            "schema_valid": sum(not p["vlm_schema_errors"] for p in predictions),
            "hit_token_limit": sum(p["token_cap_hit"] for p in predictions),
            "raw_sha256": [hashlib.sha256(p["raw_text"].encode()).hexdigest() for p in predictions],
        }
        (out / "reload.json").write_text(json.dumps(result, indent=2), "utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "reload"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.panel)
    elif args.base is None or args.out is None:
        parser.error("train and reload require --base and --out")
    else:
        run(args)
