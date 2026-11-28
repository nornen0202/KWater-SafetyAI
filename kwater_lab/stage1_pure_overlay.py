"""Materialize a Pure-Inspect v3 overlay for the verified Seoul Fold 2 archive."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import tarfile

from .fold2_8b_baseline import (
    CURRICULUM, RECIPE, RUN, SEED, STEPS, TRAIN_MANIFEST, add_tar_bytes, order_digest, sha_json, write_json,
)
from .fold2_v3_epoch1 import LABELS, LAWS, MASK, PROMPT, split
from .fold4_same_verifier import digest, read

OUT = Path("data/fold2-8b-pure-v3")
PURE_LOCAL = Path("data/stage1-pure-inspect/fold2/freeze/pure-inspect-curriculum-local.json")
FREEZE_RECEIPT = Path("results/stage1-pure-inspect-fold2.json")
PREFLIGHT = Path("results/fold2-8b-matched-preflight-20261128.json")
MIXED_SCORE = Path("results/fold2-matched-4b-8b-epoch1-20261128.json")
PREP_REPORT = Path("results/stage1-pure-overlay-preflight-20261128.json")
SCORE_RESULT = Path("results/stage1-pure-vs-mixed-fold2-20261128.json")
OVERLAY_INFO = Path("data/fold2-pure-overlay.json")
ARM = "stage1_fold2_pure-inspect_v3"
WORKER = Path("kwater_lab/fold2_8b_baseline.py")


def archived_train_inputs(mixed):
    receipt = read(mixed / "transfer-receipt.json")
    path = mixed / "train.tar"
    witness = read(PREFLIGHT)
    if (receipt["train_tar"] != witness["prepared_payloads"]["train_tar"]
            or path.stat().st_size != receipt["train_tar"]["bytes"]):
        raise ValueError("Expected the locally verified mixed baseline archive and transport receipt")
    members = {p.as_posix() for p in (RECIPE, CURRICULUM, TRAIN_MANIFEST, WORKER, PROMPT)}
    packed = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if member.name in members:
                packed[member.name] = archive.extractfile(member).read()
            if packed.keys() == members:
                break
    if packed.keys() != members:
        raise ValueError("Archive is missing frozen settings, runtime or training members")
    recipe = json.loads(packed[RECIPE.as_posix()])
    mixed_rows = json.loads(packed[CURRICULUM.as_posix()])
    train = json.loads(packed[TRAIN_MANIFEST.as_posix()])
    if (recipe != receipt["recipe"] or recipe["arm"] != "fold2_8b_matched_epoch1"
            or sha_json(mixed_rows) != recipe["training"]["curriculum_training_sha256"]
            or sha_json(train) != recipe["train_manifest_sha256"]
            or sha256(packed[WORKER.as_posix()]).hexdigest() != witness["worker_payload_sha256"]
            or sha256(packed[PROMPT.as_posix()]).hexdigest() != recipe["training"]["prompt_sha256"]
            or len(train) != 769 or any(row["split"] != "dev" or row["dev_fold"] == 2 for row in train)):
        raise ValueError("Archived curriculum, training cohort, prompt or runtime differs from the verified mixed source")
    return receipt, train, mixed_rows


def make_recipe(original, mixed, pure):
    if len(mixed) != 1066 or len(pure) != len(mixed):
        raise ValueError("The mixed and pure first epochs must preserve all 1,066 example slots")
    mixed_slots = [(row["image"], row["augment"]) for row in mixed]
    if mixed_slots != [(row["image"], row["augment"]) for row in pure]:
        raise ValueError("Pure curriculum changed an image or augmentation slot")
    schedule = [[examples, lr] for examples, _, _, lr in original["epoch1_step_controls_4b"]]
    if (len(schedule) != STEPS or [examples for examples, _ in schedule] != [16]*66 + [10]
            or original["training"]["optimizer"] != {"epochs_scheduler": 2, "lr": .00014, "grad_accum": 16,
                                                       "seed": 42, "max_pixels": 393216}):
        raise ValueError("Original mixed epoch schedule differs from 67 of 134 updates")
    positions = list(range(len(mixed)))
    random.Random(SEED + 1000).shuffle(positions)
    if order_digest([mixed[i] for i in positions]) != original["training_order_sha256"]:
        raise ValueError("Archived first-epoch shuffled order differs from the frozen mixed order")
    recipe = {key: value for key, value in original.items() if key != "epoch1_step_controls_4b"}
    recipe.update(arm=ARM, training_order_sha256=order_digest([pure[i] for i in positions]),
                  training={**original["training"], "curriculum_training_sha256": sha_json(pure),
                            "kinds": dict(Counter(row["kind"] for row in pure))})
    recipe["pure_ab"] = {"mixed_curriculum_sha256": sha_json(mixed), "mixed_recipe_sha256": sha_json(original),
                         "shared_shuffled_image_augmentation_sha256": sha_json([mixed_slots[i] for i in positions]),
                         "epoch1_steps_examples_and_applied_lr": schedule}
    return recipe


def prepare(mixed, out, result):
    from .stage1_pure_inspect import replace_verify_slots

    receipt, train, mixed_rows = archived_train_inputs(mixed)
    pure = read(PURE_LOCAL)
    generated, _ = replace_verify_slots(mixed_rows, train)
    if pure != generated or sha_json(pure) != read(FREEZE_RECEIPT)["pure_inspect_curriculum_sha256"]:
        raise ValueError("Frozen Pure-Inspect v3 differs from the verified replacement recipe")
    recipe = make_recipe(receipt["recipe"], mixed_rows, pure)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pure-v3-overlay.tar"
    pending = path.with_suffix(".tar.part")
    info = {"source_archive": receipt["train_tar"], "separate_workspace_required": True,
            "reused_train_cli_sha256": read(PREFLIGHT)["worker_payload_sha256"],
            "eval_archive": receipt["eval_tar"], "pure_recipe_sha256": sha_json(recipe),
            "pure_curriculum_sha256": recipe["training"]["curriculum_training_sha256"]}
    members = {CURRICULUM: pure, RECIPE: recipe, OVERLAY_INFO: info}
    with tarfile.open(pending, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, value in members.items():
            add_tar_bytes(archive, name.as_posix(), json.dumps(value).encode())
    pending.replace(path)
    report = {**info, "overlay": {"bytes": path.stat().st_size, "sha256": digest(path)},
              "overlay_members": [path.as_posix() for path in members],
              "training_photos": len(train), "training_slots": len(pure), "verify_slots_replaced": 156,
              "optimizer_updates_first_epoch": STEPS, "optimizer_updates_schedule": 134,
              "shuffled_image_augmentation_sha256": recipe["pure_ab"]["shared_shuffled_image_augmentation_sha256"],
              "trained_models": 0, "generated_predictions": 0}
    write_json(result, report)
    print(f"PURE_OVERLAY_VERIFIED {path.stat().st_size} bytes", flush=True)


def score(mixed, prepared, pure_fit, pure_eval, result, preflight=PREP_REPORT):
    from .fold2_8b_paired_score import (
        format_received_predictions, original_inference_controls, read_received_raw, received_training,
    )
    from .law_reranker_11code import read_selected_cases
    from .metrics import decode_target, read_index
    from .stage1_fold2_quality import READINESS, compare_panel

    original = read(mixed / "transfer-receipt.json")["recipe"]
    ready = read(preflight)
    overlay = prepared / "pure-v3-overlay.tar"
    if digest(overlay) != ready["overlay"]["sha256"]:
        raise ValueError("Local Pure-Inspect overlay differs from the frozen receipt")
    with tarfile.open(overlay) as source:
        recipe = json.load(source.extractfile(RECIPE.as_posix()))
    if (recipe["arm"] != ARM or sha_json(recipe) != ready["pure_recipe_sha256"]
            or recipe["pure_ab"]["mixed_recipe_sha256"] != sha_json(original)
            or any(digest(path) != recipe["training"][key] for path, key in (
                (PROMPT, "prompt_sha256"), (LAWS, "law_corpus_sha256"), (LABELS, "label_source_sha256")))):
        raise ValueError("Overlay recipe, local formatter sources or baseline lineage changed")
    measured = read(MIXED_SCORE)
    existing = mixed / "eval/epoch1-bf16-predictions.jsonl"
    if (measured["recipe"] != original or measured["8b_predictions_sha256"] != digest(existing)
            or measured["generated"] != read(mixed / "eval/inference-receipt.json")
            or measured["ordered_training_step_controls_match"] is not True):
        raise ValueError("Requires the unchanged measured 8B mixed comparison")
    train, outer = split()
    ids = [row["sample_id"] for row in outer]
    mask_sha = digest(MASK)
    if mask_sha != read(READINESS)["sensitivity_mask"]["recipe_sha256"]:
        raise ValueError("The fixed sensitivity mask differs from the original declaration")
    basis, _ = original_inference_controls()
    records, adapter_sha = received_training(pure_fit, recipe)
    for actual, (examples, lr) in zip(records, recipe["pure_ab"]["epoch1_steps_examples_and_applied_lr"], strict=True):
        if actual["examples"] != examples or not math.isclose(actual["applied_lr"], lr, rel_tol=1e-14):
            raise ValueError("Pure epoch changed the fixed example count or learning rate schedule")
    if adapter_sha == measured["generated"]["adapter_sha256"]:
        raise ValueError("Pure and mixed adapters have identical file fingerprints")
    receipt, raw = read_received_raw(pure_eval, recipe, adapter_sha, basis, ids)
    current = pure_eval / "epoch1-bf16-predictions.jsonl"
    candidate = format_received_predictions(train, ids, raw, current)
    control = read_index(existing, lambda row: decode_target(row, local_ids=True), limit_size=True)
    if list(control) != ids or list(candidate) != ids:
        raise ValueError("Pure and mixed outputs must cover the exact ordered 197-photo cohort")
    hazards = {row["sample_id"]: row["target"]["hazards"] for row in read_selected_cases(LABELS, set(ids))}
    gold = {row["sample_id"]: set(row["codes"]) for row in outer}
    if hazards.keys() != gold.keys() or any(set(hazards[sid]) != gold[sid] for sid in ids):
        raise ValueError("Official labels disagree with the frozen Fold 2 roster")
    severe = {sid: {code for code, info in target.items() if info["risk_level"]["severity"]["score"] == 4}
              for sid, target in hazards.items()}
    groups = {row["sample_id"]: row["group_id"] for row in outer}
    keep = set(read(MASK)["prospective_mask_retained_ids"])
    panels = {"primary_197": ids, "sensitivity_55": [sid for sid in ids if sid in keep]}
    write_json(result, {
        "scope": "Returned Fold 2 8B Pure-Inspect v3 vs measured 8B mixed v3. The 55-photo sensitivity retains exposure.",
        "arm_names": {"control": "measured 8B mixed v3", "candidate": "8B Pure-Inspect v3"},
        "mixed_predictions_sha256": digest(existing), "pure_predictions_sha256": digest(current),
        "mixed_score_sha256": digest(MIXED_SCORE), "pure_steps_sha256": digest(pure_fit / "steps.jsonl"),
        "pure_inference": receipt, "overlay_sha256": ready["overlay"]["sha256"], "mask_sha256": mask_sha,
        "same_schedule_examples_and_applied_lr": True,
        "supervised_tokens": {"mixed": sum(row[1] for row in original["epoch1_step_controls_4b"]),
                              "pure": sum(row["supervised_tokens"] for row in records)},
        "panels": {name: compare_panel(member, groups, gold, control, candidate, severe)
                   for name, member in panels.items()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--mixed", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--overlay-preflight", type=Path, default=PREP_REPORT)
    parser.add_argument("--pure-fit", type=Path)
    parser.add_argument("--pure-eval", type=Path)
    args = parser.parse_args()
    if any(not path.resolve().is_relative_to(Path("data").resolve()) for path in (
            args.mixed, args.out, args.pure_fit or args.out, args.pure_eval or args.out)):
        parser.error("All private inputs and returned outputs must stay under data/")
    if args.stage == "prepare":
        prepare(args.mixed, args.out, args.result or PREP_REPORT)
    elif args.pure_fit and args.pure_eval:
        score(args.mixed, args.out, args.pure_fit, args.pure_eval, args.result or SCORE_RESULT, args.overlay_preflight)
    else:
        parser.error("Pair scoring requires --pure-fit and --pure-eval")
