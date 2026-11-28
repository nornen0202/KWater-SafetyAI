"""Freeze, train and evaluate a matched first-epoch v3 Fold 2 arm."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from .fold4_same_verifier import digest, read, save

MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
MASK = Path("data/sub78-fold2-prospective-readiness-20261128/private-prospective-mask-and-recipe.json")
OOF = Path("data/clean-specialist-v1/strict-inner-oof/fold-2.npz")
PROMPT = Path("kwater_lab/system_prompt_v3.txt")
LABELS = Path("data/train/train.jsonl")
LAWS = Path("data/baseline/assets/laws.jsonl")
BASE = Path("data/baseline/base_model")
DEFAULT_RUN = Path("data/v3-11code-fold2-4b-epoch1")


def split():
    rows = read(MANIFEST)
    train = [row for row in rows if row["split"] == "dev" and row["dev_fold"] != 2]
    outer = [row for row in rows if row["split"] == "dev" and row["dev_fold"] == 2]
    train_ids, outer_ids = ({row["sample_id"] for row in side} for side in (train, outer))
    train_groups, outer_groups = ({row["group_id"] for row in side} for side in (train, outer))
    receipt = read(MASK)
    if ((len(train_ids), len(outer_ids), len(train_groups), len(outer_groups)) != (769, 197, 154, 38)
            or train_ids & outer_ids or train_groups & outer_groups or receipt["manifest_sha256"] != digest(MANIFEST)
            or train_ids != set(receipt["matched_outer_training_ids_both_models"])
            or outer_ids != set(receipt["fold2_primary_ids"])
            or len(set(receipt["prospective_mask_retained_ids"])) != 55
            or not set(receipt["prospective_mask_retained_ids"]) <= outer_ids):
        raise ValueError("Fold 2 split does not match the fixed prospective receipt")
    return train, outer


def freeze(out):
    import numpy as np
    from .law_reranker_11code import read_selected_cases
    from .v3_turnaround import build_v3_curriculum_and_inner_val

    train, outer = split()
    ids = {row["sample_id"] for row in train}
    truth = {row["sample_id"]: row for row in read_selected_cases(LABELS, ids)}
    with np.load(OOF, allow_pickle=False) as inner:
        iid = inner["train_ids"].tolist()
        oof = inner["code_scores"]
    if len(iid) != len(ids) or set(iid) != ids or ids != truth.keys() or not np.isfinite(oof).all():
        raise ValueError("Strict inner archive must match only the Fold 2 outer training rows")
    curriculum, pairs = build_v3_curriculum_and_inner_val(train, truth, oof, iid, seed=42)
    images = {row["sample_id"]: row["image"] for row in train}
    for row in curriculum + pairs:
        # The canonical v3 builder appends deterministic suffixes to augmented and verification examples.
        origin = row.get("orig_id", row["sample_id"].split("__", 1)[0])
        if images.get(origin) != row["image"]:
            raise ValueError("Curriculum or calibration references a photo outside Fold 2 training")
    witness = {"manifest_sha256": digest(MANIFEST), "mask_sha256": digest(MASK), "strict_inner_oof_sha256": digest(OOF),
               "prompt_sha256": digest(PROMPT), "law_corpus_sha256": digest(LAWS),
               "label_source_sha256": digest(LABELS),
               "curriculum_training_sha256": sha256(json.dumps(curriculum, sort_keys=True).encode()).hexdigest(),
               "inner_calibration_pairs": len(pairs), "curriculum_examples": len(curriculum),
               "kinds": dict(Counter(row["kind"] for row in curriculum)), "steps_per_epoch": math.ceil(len(curriculum)/16),
               "optimizer": {"epochs_scheduler": 2, "lr": .00014, "grad_accum": 16, "seed": 42, "max_pixels": 393216},
               "training_photos": len(train), "evaluation_photos": len(outer),
               "training_oof_id_exact": True, "outer_id_or_group_overlap_count": 0,
               "source": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("v3_turnaround.py"),
                                                     Path(__file__).with_name("vlm_11code.py"))}}
    out.mkdir(parents=True, exist_ok=True)
    for path, payload in ((out / "curriculum-local.json", curriculum), (out / "inner-cal-pairs-local.json", pairs),
                          (out / "frozen-recipe.json", witness)):
        if path.exists() and read(path) != payload:
            raise ValueError(f"Frozen Fold 2 recipe changed: {path}")
        if not path.exists():
            save(path, payload)
    print("FOLD2_RECIPE " + json.dumps(witness), flush=True)
    return curriculum, witness


def train_one(out, resume):
    from .vlm_11code import train_multimodal_qlora

    curriculum, witness = freeze(out)
    train_multimodal_qlora(root=Path.cwd(), manifest_path=MANIFEST, prompt_path=PROMPT, out_dir=out,
                          val_fold=2, epochs=2, grad_accum=16, lr=.00014, max_pixels=393216, seed=42,
                          checkpoint_steps=10, resume_from=resume, stop_after_steps=witness["steps_per_epoch"],
                          curriculum_builder=lambda *_args, **_kw: curriculum,
                          experiment_identity={"arm": "v3_fold2_comparable_epoch1",
                                               "strict_inner_oof_sha256": witness["strict_inner_oof_sha256"]})
    if not (out / "epoch-1/training_state.pt").is_file():
        raise ValueError("Epoch 1 checkpoint did not materialize")


def infer(out):
    from .law_reranker_11code import read_selected_cases
    from .v3_turnaround import ActionSupportedLawRetriever, format_v3_record
    from .vlm_11code import infer_batched_vlm, load_qwen_4bit_or_bf16, load_qwen_processor

    train, outer = split()
    train_ids = {row["sample_id"] for row in train}
    truth = {row["sample_id"]: row for row in read_selected_cases(LABELS, train_ids)}
    if truth.keys() != train_ids:
        raise ValueError("Missing training corpus for fresh law bank")
    laws = [json.loads(line) for line in LAWS.read_text(encoding="utf-8").splitlines() if line]
    ranker = ActionSupportedLawRetriever(laws, [truth[row["sample_id"]] for row in train])
    adapter = out / "epoch-1"
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    if digest(PROMPT) != read(out / "frozen-recipe.json")["prompt_sha256"]:
        raise ValueError("Prompt changed since training")
    model_key = f"fold2:{adapter_sha}:{digest(PROMPT)}:bf16:px393216"
    processor = load_qwen_processor(BASE, 393216)
    outputs, stats = infer_batched_vlm(None, processor, outer, Path("data/train"), PROMPT.read_text(encoding="utf-8"),
                                     batch_size=4, max_new_tokens=620, cache_path=out / "epoch1-bf16-raw-cache.json",
                                     cache_key=model_key, model_factory=lambda **_: load_qwen_4bit_or_bf16(
                                         BASE, max_pixels=393216, use_4bit=False, adapter_dir=adapter)[0])
    formatted = [format_v3_record(row["sample_id"], output["parsed"], ranker) for row, output in zip(outer, outputs, strict=True)]
    pending = out / "epoch1-bf16-predictions.tmp"
    pending.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in formatted), encoding="utf-8")
    path = out / "epoch1-bf16-predictions.jsonl"
    pending.replace(path)
    save(out / "epoch1-bf16-inference-receipt.json", {"rows": len(formatted), "adapter_sha256": adapter_sha,
                                                     "prediction_sha256": digest(path), "stats": stats})


def score(out, result):
    from .fold0_group_sensitivity import arm_metrics
    from .law_reranker_11code import read_selected_cases
    from .metrics import read_index, decode_target

    _, outer = split()
    ids = {row["sample_id"] for row in outer}
    groups = {row["sample_id"]: row["group_id"] for row in outer}
    gold = {row["sample_id"]: set(row["codes"]) for row in outer}
    severe = {row["sample_id"]: {code for code, h in row["target"]["hazards"].items()
                               if h["risk_level"]["severity"]["score"] == 4}
              for row in read_selected_cases(LABELS, ids)}
    path = out / "epoch1-bf16-predictions.jsonl"
    prediction = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    receipt = read(out / "epoch1-bf16-inference-receipt.json")
    if prediction.keys() != ids or severe.keys() != ids or digest(path) != receipt["prediction_sha256"]:
        raise ValueError("Final Fold 2 outputs or labels failed completeness and identity checks")
    keep = set(read(MASK)["prospective_mask_retained_ids"])
    result_data = {label: arm_metrics({sid: gold[sid] for sid in selected}, {sid: prediction[sid] for sid in selected},
                            {sid: groups[sid] for sid in selected}, severe)
                   for label, selected in (("primary_197", ids), ("sensitivity_55", keep))}
    save(result, {"scope": "Fold 2 outer held-out 4B fresh epoch1; 55-photo sensitivity is not a pristine holdout.",
                  "recipe": read(out / "frozen-recipe.json"), "inference": receipt, "panels": result_data})
    print("FOLD2_SCORE " + str(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "train", "infer", "score"))
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--result", type=Path, default=Path("results/fold2-v3-4b-epoch1-20261128.json"))
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Training and prediction files must stay in data/")
    out = args.out.resolve()
    if args.stage == "train":
        if args.resume is None and (out / "epoch-1/training_state.pt").exists():
            print("EPOCH1_REUSED", flush=True)
            return
        train_one(out, args.resume.resolve() if args.resume else None)
    elif args.stage == "freeze":
        freeze(out)
    elif args.stage == "infer":
        infer(out)
    else:
        score(out, args.result)


if __name__ == "__main__":
    main()
