"""Prepare matched first-epoch inspect curricula and run explicit prompt comparisons."""

import argparse
from collections import Counter
from contextlib import ExitStack
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
from statistics import median
from tempfile import TemporaryDirectory

from . import fold2_v3_epoch1 as fold2
from .fold2_8b_baseline import BASE_CONFIG_SHA, BASE_REVISION, EVAL_SEGMENTS, STEPS
from .fold4_same_verifier import digest, read, save
from .law_reranker_11code import read_selected_cases
from .vlm_11code import (
    DECISION_SPAN_RE, INSPECT_MAX_NEW_TOKENS, INSPECT_USER_TEXT, encode_weighted_example,
    load_qwen_processor, validate_vlm_11code_obj,
)


FULL = Path("data/v3-11code-full1045")
FULL_OOF = Path("data/v3-full1045-specialist/oof_1045_predictions.npz")
WORK = Path("data/stage1-pure-inspect")
RUNTIME = Path(".runtime")
BASE = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
IMAGES = Path("data/train")
PROMPTS = {"v3": fold2.PROMPT, "v4": Path("kwater_lab/system_prompt_v4.txt")}
INSPECT_KINDS = {"inspect", "inspect_minority", "inspect_other", "inspect_safe_balance"}
VERIFY_KINDS = {"verify_pos", "verify_neg_hard", "verify_neg_safe"}
REPLACEMENT_KIND = "inspect_verify_replacement"
SEED, ACCUM, PIXELS, LR = 42, 16, 393216, 1.4e-4
INFERENCE_PIXELS = (PIXELS, 655360, 786432)
SCHEDULE_EPOCHS = 2


def curriculum_digest(rows):
    # Identical serialization to the trainer's checkpoint configuration.
    return sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def under_data(path):
    path = path.resolve()
    if not path.is_relative_to(Path("data").resolve()):
        raise ValueError("Stage 1 row-level outputs must stay in data/")
    return path


def require_8b_base(base):
    config = base / "config.json"
    if not config.is_file() or digest(config) != BASE_CONFIG_SHA:
        raise ValueError("Stage 1 requires the pinned Qwen3-VL-8B-Instruct base configuration")


def origin_id(row):
    kind, sid = row["kind"], row["sample_id"]
    if kind == "inspect":
        return sid
    if kind in VERIFY_KINDS:
        return sid.rsplit("__ver_", 1)[0]
    if kind == REPLACEMENT_KIND:
        return sid.rsplit("__inspect_verify_", 1)[0]
    return sid.split("__", 1)[0]


def replace_verify_slots(mixed, train_rows):
    """Use the same training photo's unaugmented whole-scene target at each verify slot."""
    train_images = {row["sample_id"]: row["image"] for row in train_rows}
    base_items = {row["sample_id"]: row for row in mixed if row["kind"] == "inspect"}
    if (len(train_images) != len(train_rows) or len(set(train_images.values())) != len(train_rows)
            or base_items.keys() != train_images.keys()
            or len({row["sample_id"] for row in mixed}) != len(mixed)):
        raise ValueError("Mixed curriculum does not have one base inspect per training photo")
    targets = {}
    for sid, row in base_items.items():
        target = json.loads(row["assistant_text"])
        if (row["augment"] is not False or row["user_text"] != INSPECT_USER_TEXT
                or set(target) != {"scene_note", "status", "hazards"}
                or not isinstance(target["scene_note"], str) or not target["scene_note"].strip()
                or validate_vlm_11code_obj(target)):
            raise ValueError("Base target does not match the whole-scene inspect contract")
        targets[sid] = target

    pure = []
    for index, row in enumerate(mixed):
        sid = origin_id(row)
        base = base_items.get(sid)
        if base is None or row["image"] != train_images[sid]:
            raise ValueError("Curriculum row has no matching training photo")
        if row["kind"] in INSPECT_KINDS:
            if row["user_text"] != INSPECT_USER_TEXT or row["assistant_text"] != base["assistant_text"]:
                raise ValueError("Inspect augmentation changed the supervised target")
            pure.append(row)
        elif row["kind"] in VERIFY_KINDS and row["augment"] is False:
            pure.append({**base, "sample_id": f"{sid}__inspect_verify_{index}", "kind": REPLACEMENT_KIND})
        else:
            raise ValueError("Unsupported mixed curriculum slot")

    if len({row["sample_id"] for row in pure}) != len(pure):
        raise ValueError("Pure-Inspect row ID collision")
    return pure, targets


def load_scope(scope):
    if scope == "fold2":
        train, outer = fold2.split()
        manifest = fold2.MANIFEST
        frozen = read(fold2.DEFAULT_RUN / "frozen-recipe.json")
        mixed = read(fold2.DEFAULT_RUN / "curriculum-local.json")
        expected_sha = frozen["curriculum_training_sha256"]
        expected_prompt_sha = frozen["prompt_sha256"]
        if digest(manifest) != frozen["manifest_sha256"] or digest(fold2.LABELS) != frozen["label_source_sha256"]:
            raise ValueError("Fold 2 manifest or official labels differ from the historical run")
    elif scope == "full1045":
        # Reconstruct the existing deployment run from its own manifest and OOF lineage.
        # Its prior full-fit membership does not define a held-out evaluation set.
        manifest = FULL / "manifest_all_1045_train.json"
        train, outer = read(manifest), []
        frozen = read(FULL / "training_summary.json")
        expected_sha = frozen["configuration"]["curriculum_sha256"]
        expected_prompt_sha = frozen["configuration"]["prompt_sha256"]
    else:
        raise ValueError("Unknown Stage 1 scope")

    count, slots = (769, 1066) if scope == "fold2" else (1045, 1345)
    ids = {row["sample_id"] for row in train}
    if len(train) != count or len(ids) != count or any(row["split"] != "dev" for row in train):
        raise ValueError("Training roster does not match the fixed Stage 1 scope")
    overlap = {key: len({row[key] for row in train} & {row[key] for row in outer})
               for key in ("sample_id", "group_id", "image", "sha256")}
    if any(overlap.values()):
        raise ValueError("Fold 2 outer scope overlaps its training roster")

    cases = list(read_selected_cases(fold2.LABELS, ids))
    raw = {row["sample_id"]: row for row in cases}
    if len(cases) != count or raw.keys() != ids:
        raise ValueError("Selected official targets do not cover the training roster exactly")
    if scope == "full1045":
        import numpy as np
        from .v3_turnaround import build_v3_curriculum_and_inner_val

        with np.load(FULL_OOF, allow_pickle=False) as archive:
            oof_ids = archive["ids"].tolist()
            oof_scores = archive["code_probs"]
            if len(oof_ids) != count or set(oof_ids) != ids or not np.isfinite(oof_scores).all():
                raise ValueError("Historical full-fit OOF archive has unexpected coverage")
            mixed, inner_pairs = build_v3_curriculum_and_inner_val(train, raw, oof_scores, oof_ids, seed=SEED)
        if inner_pairs != read(FULL / "inner_margin_cal_pairs.json"):
            raise ValueError("Reconstructed full-fit inner pairs differ from the prior run")
    if (len(mixed) != slots or curriculum_digest(mixed) != expected_sha
            or digest(PROMPTS["v3"]) != expected_prompt_sha):
        raise ValueError("Mixed curriculum or v3 prompt differs from the historical run")
    return train, outer, raw, manifest, mixed, overlap


def audit_inputs(train, raw, targets):
    root = IMAGES.resolve()
    for row in train:
        sid = row["sample_id"]
        relative = Path(row["image"])
        image = (root / relative).resolve()
        official = raw[sid]["target"]
        target = targets[sid]
        shown = {item["code"]: (item["likelihood_score"], item["severity_score"]) for item in target["hazards"]}
        expected = {code: tuple(int(hazard["risk_level"][key]["score"]) for key in ("likelihood", "severity"))
                    for code, hazard in official["hazards"].items()}
        if (relative.is_absolute() or ".." in relative.parts or not image.is_relative_to(root)
                or not image.is_file() or digest(image) != row["sha256"]
                or shown != expected or set(shown) != set(row["codes"])
                or target["status"] != row["status"] or target["status"] != official["status"]):
            raise ValueError("Training image digest or official inspect supervision differs")


def summarize_curricula(mixed, pure, targets):
    positions = list(range(len(mixed)))
    random.Random(SEED + 1000).shuffle(positions)
    mixed_visual = [(mixed[i]["image"], mixed[i]["augment"]) for i in positions]
    pure_visual = [(pure[i]["image"], pure[i]["augment"]) for i in positions]
    if mixed_visual != pure_visual:
        raise ValueError("Stage 1 curricula differ in photo order or augmentation draws")
    kept = [i for i in positions if mixed[i]["kind"] in INSPECT_KINDS]
    if any(mixed[i] != pure[i] for i in kept):
        raise ValueError("An existing inspect slot changed")
    replacements = [row for row in pure if row["kind"] == REPLACEMENT_KIND]
    all_repeats = Counter(map(origin_id, pure))
    replacement_repeats = Counter(map(origin_id, replacements))

    def target_mix(rows):
        values = [targets[origin_id(row)] for row in rows]
        return {
            "statuses": dict(Counter(row["status"] for row in values)),
            "codes_per_target": dict(sorted(Counter(len(row["hazards"]) for row in values).items())),
            "code_exposures": dict(Counter(h["code"] for row in values for h in row["hazards"])),
        }

    return {
        "mixed_kinds": dict(Counter(row["kind"] for row in mixed)),
        "pure_inspect_kinds": dict(Counter(row["kind"] for row in pure)),
        "existing_inspect_slots_unchanged": len(kept),
        "verify_slots_replaced": len(replacements),
        "replacement_rule": "Same photo's original whole-scene inspect example, with no augmentation.",
        "epoch1_photo_and_augmentation_order_sha256": curriculum_digest(mixed_visual),
        "photo_repetitions_across_all_slots": dict(sorted(Counter(all_repeats.values()).items())),
        "photo_repetitions_in_replacement_slots": dict(sorted(Counter(replacement_repeats.values()).items())),
        "mixed_inspect_targets": target_mix([mixed[i] for i in kept]),
        "replacement_targets": target_mix(replacements),
        "pure_inspect_targets": target_mix(pure),
        "first_epoch_updates": math.ceil(len(mixed) / ACCUM),
        "last_update_examples": len(mixed) % ACCUM or ACCUM,
    }


def token_audit(base, mixed, pure):
    """Count exact assistant masks in the real template/tokenizer and spot-check image-path parity."""
    import transformers

    processor = load_qwen_processor(base, PIXELS)
    tokenizer = processor.tokenizer
    unique = {(row["user_text"], row["assistant_text"]) for row in mixed + pure}
    unique = sorted(unique)
    targets = tokenizer([answer for _, answer in unique], add_special_tokens=False, return_offsets_mapping=True)
    decisions = []
    for (_, answer), offsets in zip(unique, targets["offset_mapping"], strict=True):
        spans = [match.span() for match in DECISION_SPAN_RE.finditer(answer)]
        decisions.append(sum(end > start and any(start < hi and end > lo for lo, hi in spans)
                             for start, end in offsets))

    base_files = ("config.json", "preprocessor_config.json", "chat_template.json", "tokenizer.json", "tokenizer_config.json")
    reports = {"transformers": transformers.__version__, "processor_file_sha256": {f: digest(base / f) for f in base_files},
               "model_generation_cap": INSPECT_MAX_NEW_TOKENS, "prompts": {}}
    for name, path in PROMPTS.items():
        system = path.read_text(encoding="utf-8")
        by_user = {}
        rendered = []
        for user, answer in unique:
            if user not in by_user:
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
                prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                by_user[user] = messages, prefix
            messages, _ = by_user[user]
            rendered.append(processor.apply_chat_template(
                messages + [{"role": "assistant", "content": answer}], tokenize=False, add_generation_prompt=False,
            ))

        prefixes = tokenizer([prefix for _, prefix in by_user.values()], add_special_tokens=False)["input_ids"]
        prefix_ids = dict(zip(by_user, prefixes, strict=True))
        rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        counts = {}
        suffixes = set()
        for key, full, answer, decision_count in zip(unique, rendered_ids, targets["input_ids"], decisions, strict=True):
            prefix = prefix_ids[key[0]]
            if full[:len(prefix) + len(answer)] != prefix + answer:
                raise ValueError("Assistant tokenization differs at the template boundary")
            suffixes.add(tuple(full[len(prefix) + len(answer):]))
            counts[key] = len(full) - len(prefix), decision_count, len(answer), len(full)
        if len(suffixes) != 1 or next(iter(suffixes))[:1] != (tokenizer.eos_token_id,):
            raise ValueError("Unexpected template suffix after the assistant target")

        representatives = {row["kind"]: row for row in reversed(mixed + pure)}
        vision_counts = []
        for row in representatives.values():
            native = encode_weighted_example(processor, row, IMAGES, system, random.Random(SEED))
            n_sup, n_high, _, unexpanded = counts[row["user_text"], row["assistant_text"]]
            if int((native["labels"] != -100).sum()) != n_sup or int((native["token_weights"] > 1.5).sum()) != n_high:
                raise ValueError("Assistant token audit differs from the native image training encoder")
            vision_counts.append(native["input_ids"].shape[-1] - unexpanded + 1)

        def summarize(rows):
            sizes = [counts[row["user_text"], row["assistant_text"]] for row in rows]
            supervised = sum(size[0] for size in sizes)
            high = sum(size[1] for size in sizes)
            inspect_sizes = sorted(counts[row["user_text"], row["assistant_text"]][2] + 1
                                   for row in rows if row["user_text"] == INSPECT_USER_TEXT)
            return {"assistant_tokens": supervised, "decision_tokens": high,
                    "weight_mass_at_4_5": supervised + 3.5 * high,
                    "mean_decision_loss_weight_share": round(sum(4.5 * size[1] / (size[0] + 3.5 * size[1])
                                                                 for size in sizes) / len(sizes), 6),
                    "inspect_targets_including_eos": {"p50": median(inspect_sizes),
                        "p95": inspect_sizes[math.ceil(.95 * len(inspect_sizes)) - 1], "max": max(inspect_sizes),
                        "at_cap": inspect_sizes.count(INSPECT_MAX_NEW_TOKENS),
                        "above_cap": sum(n > INSPECT_MAX_NEW_TOKENS for n in inspect_sizes)}}

        reports["prompts"][name] = {
            "inspect_chat_prefix_sha256": sha256(by_user[INSPECT_USER_TEXT][1].encode()).hexdigest(),
            "inspect_prefix_tokens_with_single_image_placeholder": len(prefix_ids[INSPECT_USER_TEXT]),
            "assistant_suffix_tokens": len(next(iter(suffixes))),
            "native_encoder_representatives_checked": len(representatives),
            "native_vision_tokens_range_for_representatives": [min(vision_counts), max(vision_counts)],
            "mixed": summarize(mixed), "pure_inspect": summarize(pure),
        }
    return reports


def prepare(scope, out, *, result=None, base=None):
    out = under_data(out)
    if base is not None:
        require_8b_base(base)
    train, outer, raw, manifest, mixed, overlap = load_scope(scope)
    pure, targets = replace_verify_slots(mixed, train)
    audit_inputs(train, raw, targets)
    out.mkdir(parents=True, exist_ok=True)
    restricted = out / "pure-inspect-curriculum-local.json"
    if restricted.exists() and read(restricted) != pure:
        raise ValueError("Existing staged Pure-Inspect curriculum differs from the historical lineage")
    if not restricted.exists():
        save(restricted, pure)
    if result is not None:
        report = {
            "scope": scope, "deployment_only": scope == "full1045",
            "train_photos": len(train), "train_groups": len({row["group_id"] for row in train}),
            "evaluation_photos": len(outer),
            "evaluation_sensitivity_photos": 55 if scope == "fold2" else 0,
            "evaluation_segments": [list(segment) for segment in EVAL_SEGMENTS] if scope == "fold2" else [],
            "base_revision": BASE_REVISION,
            "train_outer_overlap": overlap,
            "training_inputs_verified_against_manifest": len(train),
            "source_manifest_sha256": digest(manifest),
            "official_label_source_sha256": digest(fold2.LABELS),
            "mixed_curriculum_sha256": curriculum_digest(mixed),
            "pure_inspect_curriculum_sha256": curriculum_digest(pure),
            "prompt_sha256": {version: digest(path) for version, path in PROMPTS.items()},
            "optimizer": {"configured_scheduler_epochs": SCHEDULE_EPOCHS, "run_epochs": 1, "seed": SEED,
                          "grad_accum": ACCUM, "max_pixels": PIXELS, "lr": LR},
            **summarize_curricula(mixed, pure, targets),
        }
        if base is not None:
            report["real_processor_token_audit"] = token_audit(base, mixed, pure)
        result.parent.mkdir(parents=True, exist_ok=True)
        save(result, report)
    return manifest, raw, {"mixed": mixed, "pure-inspect": pure}


def write_scoped_train_inputs(root, raw, curriculum):
    """Give the unchanged trainer its normal relative paths with only Fold 2 fit inputs."""
    images = {item["sample_id"]: item["image"] for item in curriculum if item["kind"] == "inspect"}
    if images.keys() != raw.keys():
        raise ValueError("Scoped fit labels and original inspect photos must cover the same rows")
    target = root / "data/train"
    target.mkdir(parents=True)
    with (target / "train.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in raw.values():
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in sorted(images.values()):
        source = IMAGES / name
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, destination)


def train_one(scope, out, base, variant, version, resume):
    from .vlm_11code import train_multimodal_qlora

    require_8b_base(base)
    manifest, raw, curricula = prepare(scope, out)
    curriculum = curricula[variant]
    steps = math.ceil(len(curriculum) / ACCUM)
    os.environ["KWATER_LOCAL_BASE_MODEL"] = str(base.resolve())
    with ExitStack() as resources:
        root = Path.cwd()
        if scope == "fold2":
            RUNTIME.mkdir(exist_ok=True)
            root = Path(resources.enter_context(TemporaryDirectory(dir=RUNTIME.resolve(), prefix="stage1-fold2-")))
            write_scoped_train_inputs(root, raw, curriculum)
        result = train_multimodal_qlora(
            root=root, manifest_path=manifest, prompt_path=PROMPTS[version], out_dir=out,
            val_fold=2 if scope == "fold2" else -1, epochs=SCHEDULE_EPOCHS, grad_accum=ACCUM, lr=LR,
            max_pixels=PIXELS, seed=SEED, checkpoint_steps=10, resume_from=resume, stop_after_steps=steps,
            curriculum_builder=lambda *_args, **_kwargs: curriculum,
            experiment_identity={"arm": f"stage1_{scope}_{variant}_{version}", "curriculum_n": len(curriculum),
                                 "base_revision": BASE_REVISION},
        )
    if result["optimizer_steps"] != steps or not (out / "epoch-1/training_state.pt").is_file():
        raise ValueError("Stage 1 training did not stop at the first-epoch boundary")


def infer_fold2(out, base, adapter, version, max_pixels=PIXELS):
    import torch
    from .stage1_fold2_quality import generation_controls, receipt_cache_key, write_generation_receipt
    from .v3_turnaround import ActionSupportedLawRetriever, format_v3_record
    from .vlm_11code import infer_batched_vlm, load_qwen_model

    if max_pixels not in INFERENCE_PIXELS:
        raise ValueError("Unsupported Stage 1 inference pixel cap")
    require_8b_base(base)
    train, outer = fold2.split()
    if len(outer) != EVAL_SEGMENTS[-1][1] or any(digest(IMAGES / row["image"]) != row["sha256"] for row in outer):
        raise ValueError("Fold 2 inference requires the canonical evaluation images")
    ids = {row["sample_id"] for row in train}
    truth = {row["sample_id"]: row for row in read_selected_cases(fold2.LABELS, ids)}
    if truth.keys() != ids:
        raise ValueError("Missing Fold 2 training cases for the law ranker")
    train_cases = [truth[row["sample_id"]] for row in train]
    checkpoint = torch.load(adapter / "training_state.pt", map_location="cpu", weights_only=True, mmap=True)
    configuration = checkpoint["configuration"]
    train_version = next((name for name, path in PROMPTS.items() if digest(path) == configuration["prompt_sha256"]), None)
    if (configuration["val_fold"] != 2 or configuration["base_config_sha256"] != BASE_CONFIG_SHA
            or configuration["max_pixels"] != PIXELS or train_version is None
            or (checkpoint["global_step"], checkpoint["next_epoch"], checkpoint["next_offset"]) != (STEPS, 2, 0)):
        raise ValueError("Expected a declared first-epoch Fold 2 adapter for this base")
    del checkpoint
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    controls = generation_controls(
        base=base, base_revision=BASE_REVISION, precision="bf16", adapter=adapter, train_configuration=configuration,
        prompt_path=PROMPTS[version], max_pixels=max_pixels, formatter_path=Path(__file__).with_name("v3_turnaround.py"),
        engine_path=Path(__file__).with_name("vlm_11code.py"), train_cases=train_cases,
    )
    key = receipt_cache_key(controls)
    dest = under_data(out) / f"train-{train_version}-infer-{version}-px{max_pixels}-{adapter_sha[:16]}"
    dest.mkdir(parents=True, exist_ok=True)
    laws = [json.loads(line) for line in fold2.LAWS.read_text(encoding="utf-8").splitlines() if line]
    ranker = ActionSupportedLawRetriever(laws, train_cases)
    processor = load_qwen_processor(base, max_pixels)
    system = PROMPTS[version].read_text(encoding="utf-8")
    model = None

    def load_model():
        nonlocal model
        if model is None:
            model = load_qwen_model(base, use_4bit=False, adapter_dir=adapter)
        return model

    outputs, segments = [], []
    for start, end, batch in EVAL_SEGMENTS:
        cache = dest / f"raw-{start}-{end}.json"
        generated, stats = infer_batched_vlm(
            None, processor, outer[start:end], IMAGES, system, batch_size=batch, max_new_tokens=INSPECT_MAX_NEW_TOKENS,
            cache_path=cache, cache_key=key, model_factory=load_model,
        )
        if len(generated) != end - start or stats["final_effective_batch_size"] != batch:
            raise RuntimeError("Fold 2 generation differs from the matched batch boundaries")
        outputs.extend(generated)
        segments.append({"start": start, "end": end, "raw_sha256": digest(cache), "stats": stats})
    rows = [format_v3_record(row["sample_id"], output["parsed"], ranker)
            for row, output in zip(outer, outputs, strict=True)]
    pending = dest / "predictions.tmp"
    pending.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    predictions = dest / "predictions.jsonl"
    pending.replace(predictions)
    write_generation_receipt(
        dest / "receipt.json", predictions, outer, segments, controls,
        extras={"train_arm": configuration.get("experiment", {}).get("arm"), "train_prompt": train_version,
                "train_prompt_sha256": configuration["prompt_sha256"], "infer_prompt": version},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "train", "infer"))
    parser.add_argument("--scope", choices=("fold2", "full1045"), default="fold2")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--base-model", type=Path, default=BASE)
    parser.add_argument("--curriculum", choices=("mixed", "pure-inspect"), default="mixed")
    parser.add_argument("--prompt", choices=tuple(PROMPTS), default="v3")
    parser.add_argument("--max-pixels", type=int, choices=INFERENCE_PIXELS, default=PIXELS)
    parser.add_argument("--audit-tokens", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.stage != "infer" and args.max_pixels != PIXELS:
        parser.error("Stage 1 training and training-token audits use a fixed 393216-pixel cap")
    directory = WORK / args.scope
    out = under_data(args.out or (directory / args.stage if args.stage != "train" else
                                  directory / f"{args.curriculum}-{args.prompt}"))
    if args.stage == "freeze":
        result = args.result or Path(f"results/stage1-pure-inspect-{args.scope}.json")
        prepare(args.scope, out, result=result, base=args.base_model if args.audit_tokens else None)
        print(f"Stage 1 input audit saved to {result}")
    elif args.stage == "train":
        train_one(args.scope, out, args.base_model, args.curriculum, args.prompt,
                  args.resume.resolve() if args.resume else None)
    elif args.scope != "fold2" or args.adapter is None:
        parser.error("Inference requires a Fold 2 epoch-one --adapter")
    else:
        infer_fold2(out, args.base_model, args.adapter.resolve(), args.prompt, args.max_pixels)


if __name__ == "__main__":
    main()
