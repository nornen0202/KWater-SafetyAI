"""Run the matched Fold 4 VLM training or evaluate a selected epoch with the v3 formatter."""
import argparse
import json
import os
from pathlib import Path

from .audit import digest
from .train_head import save_atomic


def evaluate_epoch(root, base, adapter, output, manifest, prompt):
    from .v3_turnaround import ActionSupportedLawRetriever, format_v3_record
    from .vlm_11code import infer_batched_vlm, load_qwen_4bit_or_bf16, seed_all

    seed_all(42)
    rows = json.loads(manifest.read_text("utf-8"))
    held = [r for r in rows if r["split"] == "dev" and r["dev_fold"] == 4]
    train = [r for r in rows if r["split"] == "dev" and r["dev_fold"] != 4]
    if (len(train), len(held), len({r["group_id"] for r in held})) != (791, 175, 39):
        raise ValueError("Unexpected Fold 4 membership")
    key = f"bf16:{digest(base / 'config.json')}:{digest(adapter / 'adapter_model.safetensors')}:{digest(prompt)}:px393216"
    model, proc = load_qwen_4bit_or_bf16(base, 393216, False, adapter)
    generated, stats = infer_batched_vlm(model, proc, held, root / "data/train", prompt.read_text("utf-8"),
                                         batch_size=4, max_new_tokens=620, cache_path=output / "cache.json", cache_key=key)
    raw = {r["sample_id"]: r for r in (json.loads(s) for s in (root / "data/train/train.jsonl").read_text("utf-8").splitlines())}
    laws = [json.loads(s) for s in (root / "data/baseline/assets/laws.jsonl").read_text("utf-8").splitlines()]
    ranker = ActionSupportedLawRetriever(laws, [raw[r["sample_id"]] for r in train])
    if [r["sample_id"] for r in held] != [g["sample_id"] for g in generated]:
        raise ValueError("Generation order differs from the Fold 4 manifest")
    formatted = [format_v3_record(r["sample_id"], g["parsed"], ranker) for r, g in zip(held, generated, strict=True)]
    lines = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in formatted)
    save_atomic(output / "predictions.jsonl", lambda f: f.write(lines.encode()))
    stats.update(adapter_sha256=digest(adapter / "adapter_model.safetensors"), base_config_sha256=digest(base / "config.json"))
    save_atomic(output / "generation.json", lambda f: f.write(json.dumps(stats, indent=2).encode()))


def score(scale_run, output):
    from .fold4_crosssplit_audit import LOCAL, V3_EVAL, _score_panels

    four = Path("data/v3-11code-fold4")
    eight = scale_run / "fit"
    fits = [json.loads((d / "training_summary.json").read_bytes()) for d in (four, eight)]
    controls = [{k: v for k, v in f["configuration"].items() if k != "base_config_sha256"} for f in fits]
    if controls[0] != controls[1] or any((f["train_photos"], f["optimizer_steps"]) != (791, 138) for f in fits):
        raise ValueError("Models differ in their declared matched training controls")
    logs = [[json.loads(s) for s in (d / "steps.jsonl").read_bytes().splitlines()] for d in (four, eight)]
    keys = ("epoch", "step", "examples", "decision_tokens", "supervised_tokens", "applied_lr")
    if any(len(log) != 138 for log in logs):
        raise ValueError("Both arms require 138 recorded updates")
    matching = all(all(a[k] == b[k] for k in keys) for a, b in zip(*logs, strict=True))
    arms = {
        "4b_epoch1": V3_EVAL / "v3_epoch1_11code_standalone.jsonl",
        "8b_epoch1": scale_run / "eval1/predictions.jsonl",
        "4b_epoch2": V3_EVAL / "v3_11code_standalone.jsonl",
        "8b_epoch2": scale_run / "eval2/predictions.jsonl",
    }
    if output.resolve() in {p.resolve() for p in arms.values()}:
        raise ValueError("Aggregate destination must differ from predictions")
    result = _score_panels(LOCAL, arms, None, "paired_qwen_4b_8b_fold4")
    result.update(training_controls=controls[0], step_controls_match=matching,
                  steps_sha256=[digest(d / "steps.jsonl") for d in (four, eight)],
                  base_config_sha256=[f["configuration"]["base_config_sha256"] for f in fits])
    save_atomic(output, lambda f: f.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate", "score"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path)
    parser.add_argument("--scale-run", type=Path, help="Directory with fit, eval1 and eval2 from the 8B run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if args.action == "score":
        if not args.scale_run or root != Path.cwd():
            parser.error("score requires --scale-run and the project root as working directory")
        score(args.scale_run.resolve(strict=True), output)
        raise SystemExit
    if not args.base:
        parser.error("train and evaluate require --base")
    base = args.base.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = root / "data/clean-split-v1/split_manifest_full_sealed.json"
    prompt = root / "kwater_lab/system_prompt_v3.txt"
    if args.action == "train":
        from .v3_turnaround import train_v3_11code_model

        os.environ["KWATER_LOCAL_BASE_MODEL"] = str(base)
        train_v3_11code_model(root, manifest, prompt, output, mode="fold4", epochs=2,
                              seed=42, resume_from=args.resume.resolve() if args.resume else None)
    elif args.adapter:
        evaluate_epoch(root, base, args.adapter.resolve(strict=True), output, manifest, prompt)
    else:
        parser.error("evaluate requires --adapter")
