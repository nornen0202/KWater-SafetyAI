"""CPU audit and isolated replay of the full1045 last accumulation group."""

import argparse
import ast
from collections import Counter, defaultdict, OrderedDict
from contextlib import ExitStack
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import struct
from typing import Any
from zipfile import ZipFile

import numpy as np

from .hybrid_public200_timing import panel
from .sub78_verifier_audit import (
    CODES, DEV, INNER, PROMPT, SPLIT, digest, load, read_selected_cases, write,
)

FULL = Path("data/v3-11code-full1045")
FOLD4 = Path("data/v3-11code-fold4")
FOLD2 = Path("data/v3-11code-fold2-4b-epoch1")
WORK = Path("data/local-reports/sub78-tail-replay")
LABELS = Path("data/train/train.jsonl")
TRAINER = Path("kwater_lab/vlm_11code.py")
TURNAROUND = Path("kwater_lab/v3_turnaround.py")
FULL_OOF = Path("data/v3-full1045-specialist/oof_1045_predictions.npz")
FULL_MANIFEST = FULL / "manifest_all_1045_train.json"


def curriculum_builder():
    context = {"Counter": Counter, "defaultdict": defaultdict, "random": random,
               "CODES": CODES, "Any": Any, "re": re, "json": json, "np": np}
    names = {"verify_user_text", "sanitize_text", "gold_row_to_vlm_target",
             "KOR_CODE_NAMES", "MINORITY_CODES", "INSPECT_USER_TEXT",
             "build_v3_curriculum_and_inner_val"}
    for path in (TRAINER, TURNAROUND):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), context)
    return context["build_v3_curriculum_and_inner_val"]


def replay_curriculum(directory, summary):
    if directory == FOLD2:
        cur = load(FOLD2 / "curriculum-local.json")
        pairs = load(FOLD2 / "inner-cal-pairs-local.json")
        expected = summary["curriculum_training_sha256"]
    else:
        rows = load(FULL_MANIFEST if directory == FULL else SPLIT)
        if directory == FOLD4:
            rows = [r for r in rows if r["split"] == "dev" and r["dev_fold"] != 4]
        ids = {r["sample_id"] for r in rows}
        labels = {r["sample_id"]: r for r in read_selected_cases(LABELS, ids)}
        with np.load(FULL_OOF if directory == FULL else INNER, allow_pickle=False) as archive:
            id_key, score_key = ("ids", "code_probs") if directory == FULL else ("train_ids", "code_scores")
            oof_ids = archive[id_key].tolist()
            if set(oof_ids) != ids or labels.keys() != ids or len(oof_ids) != len(ids):
                raise ValueError("OOF ID coverage differs")
            cur, pairs = curriculum_builder()(rows, labels, archive[score_key], oof_ids, seed=42)
        if pairs != load(directory / "inner_margin_cal_pairs.json"):
            raise ValueError("Inner pair replay differs")
        expected = summary["configuration"]["curriculum_sha256"]
    calculated = sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()
    if expected != calculated:
        raise ValueError(f"Curriculum SHA differs for {directory}")
    return cur, pairs, calculated


class CheckpointMetadata(pickle.Unpickler):
    """Read only metadata and storage references in a self-produced torch ZIP."""

    @staticmethod
    def persistent_load(storage):
        kind, dtype, key, location, size = storage
        if kind != "storage":
            raise ValueError("Unexpected checkpoint storage")
        return {"key": key, "dtype": dtype, "device": location, "elements": size}

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module == "torch" and name in ("FloatStorage", "ByteStorage", "LongStorage"):
            return name
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, shape, *_: (storage, offset, shape)
        raise ValueError(f"Unexpected checkpoint pickle symbol: {module}.{name}")


def checkpoint_metadata(path):
    if not path.exists():
        return {"exists": False}
    with ZipFile(path) as z:
        index = next(name for name in z.namelist() if name.endswith("/data.pkl"))
        state = CheckpointMetadata(io.BytesIO(z.read(index))).load()
        storage_prefix = index.removesuffix("data.pkl") + "data/"
        optimizer = state["optimizer"]
        steps = Counter()
        slots = list(optimizer["state"].values())
        for slot in (slots[0], slots[len(slots) // 2], slots[-1]):
            storage, offset, shape = slot["step"]
            if storage["dtype"] != "FloatStorage" or offset != 0 or shape != ():
                raise ValueError("Unexpected AdamW step tensor")
            steps[int(struct.unpack("<f", z.read(storage_prefix + storage["key"]))[0])] += 1
        rng = {}
        for key in ("cpu_rng", "cuda_rng"):
            storage, offset, size = state[key]
            rng[key] = sha256(z.read(storage_prefix + storage["key"])[offset:offset + size[0]]).hexdigest()
        rng["epoch_rng"] = sha256(repr(state["epoch_rng"]).encode()).hexdigest()
        return {
            "exists": True, "zip_bytes": path.stat().st_size, "global_step": state["global_step"],
            "next_offset": state["next_offset"], "next_epoch": state["next_epoch"],
            "optimizer_slots": len(optimizer["state"]), "optimizer_steps_in_three_sampled_slots": dict(steps),
            "optimizer_moments": all({"exp_avg", "exp_avg_sq", "step"} <= s.keys() for s in optimizer["state"].values()),
            "parameter_group": [{k: g[k] for k in ("lr", "betas", "eps", "weight_decay")} for g in optimizer["param_groups"]],
            "scheduler_last_epoch": state["scheduler"]["last_epoch"], "rng_sha256": rng,
            "configuration": state["configuration"],
        }


def row_type(item):
    value = json.loads(item["assistant_text"])
    return f"{item['kind']}:{value.get('status', value.get('decision'))}"


def read_steps(directory):
    return {item["step"]: item for item in map(json.loads, (directory / "steps.jsonl").read_text().splitlines())}


def prepare():
    summaries = {FULL: FULL / "training_summary.json", FOLD4: FOLD4 / "training_summary.json",
                 FOLD2: FOLD2 / "frozen-recipe.json"}
    aggregate = {}
    for directory, summary in summaries.items():
        cur, pairs, fingerprint = replay_curriculum(directory, load(summary))
        steps = read_steps(directory)
        per_epoch = math.ceil(len(cur) / 16)
        epochs = {}
        for epoch in (1, 2):
            ordered = cur.copy()
            random.Random(42 + 1000 * epoch).shuffle(ordered)
            step = epoch * per_epoch
            tail = ordered[16 * (per_epoch - 1):]
            logged = steps.get(step)
            if logged is not None and logged["examples"] != len(tail):
                raise ValueError("Logged tail group size differs")
            epochs[str(epoch)] = {
                "step": step, "ordered_sha256": sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest(),
                "examples": len(tail), "types": dict(Counter(map(row_type, tail))), "logged": logged,
            }
        inspect = {item["image"] for item in cur if item["kind"] == "inspect"}
        aggregate[directory.name] = {"examples": len(cur), "sha256": fingerprint, "steps_per_epoch": per_epoch,
                                     "epoch_tails": epochs, "inner_pairs": len(pairs),
                                     "inner_photos": len({p["image"] for p in pairs}),
                                     "inner_pairs_inspect_exposed": sum(p["image"] in inspect for p in pairs)}
    original = read_steps(FULL)
    checkpoints = {stage: checkpoint_metadata(FULL / stage / "training_state.pt")
                   for stage in ("checkpoint-80", "checkpoint-84", "epoch-1", "epoch-2")}
    full_config = load(summaries[FULL])["configuration"]
    if any(a["configuration"] != full_config for a in checkpoints.values() if a["exists"]):
        raise ValueError("Saved config differs from original training")
    fixed = {}
    for step in (85, 170):
        norm = original[step]["grad_norm"]
        fixed[str(step)] = {"logged_preclip_l2": norm, "original_effective_l2_approx": min(1.0, norm),
                            "counterfactual_preclip_if_identical_raw_gradient": norm / 16,
                            "counterfactual_postclip_if_identical_raw_gradient": min(1.0, norm / 16)}
    rows = panel(DEV)
    labels = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(LABELS, {r["sample_id"] for r in rows})}
    strata = ("normal", "severity4", "other_hazard")

    def stratum(row):
        hazards = labels[row["sample_id"]]
        if not hazards:
            return strata[0]
        return strata[1] if any(v["risk_level"]["severity"]["score"] == 4 for v in hazards.values()) else strata[2]

    ordered = sorted(rows, key=lambda r: sha256(f"sub78-tail-replay-v1/{r['sample_id']}".encode()).digest())
    picked, used = [], set()
    for name in strata:
        # Fixed priority, so the same incident cannot enter two strata.
        subset = [r for r in ordered if stratum(r) == name]
        n = 0
        for row in subset:
            group = row["group_id"]
            if group in used:
                continue
            used.add(group)
            keys = ("sample_id", "image", "sha256", "group_id")
            picked.append({**{key: row[key] for key in keys}, "stratum": name})
            n += 1
            if n == 10:
                break
        if n != 10:
            raise ValueError(f"Too few distinct groups in {name}")
    WORK.mkdir(parents=True, exist_ok=True)
    freeze = WORK / "cohort.json"
    write(freeze, picked)
    return {"scope": "CPU provenance and analytic clipping. No parameter update or causal prediction replay executed.",
            "inputs": {p.as_posix(): digest(p) for p in (
                TRAINER, TURNAROUND, FULL_MANIFEST, DEV, SPLIT, FULL_OOF, INNER, PROMPT,
                FOLD2 / "frozen-recipe.json", FULL / "steps.jsonl",
            )},
            "runs": aggregate, "full_checkpoints": checkpoints,
            "full_logs_80_85_and_170": [original[x] for x in (*range(80, 86), 170)],
            "analytic_gradient": fixed, "cohort": {"photos": len(picked), "groups": len(used),
                "labels_only": True, "training_exposed": True, "strata": dict(Counter(r["stratum"] for r in picked)),
                "frozen_sha256": digest(freeze)}}


def isolated_trainer():
    """Clone exactly the current function and inject one opt-in denominator."""
    from . import vlm_11code as original

    tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "train_multimodal_qlora")
    replacements = [node for node in ast.walk(fn) if isinstance(node, ast.BinOp)
                    and ast.unparse(node) == "loss / len(group)"]
    if len(replacements) != 1:
        raise ValueError("Expected one original backward normalization")
    replacements[0].right = ast.parse(
        "grad_accum if fixed_tail else len(group)", mode="eval"
    ).body
    scope = vars(original).copy()
    return fn, scope


def replay_train(mode):
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        raise RuntimeError("GPU hidden")
    audit = load(Path("results/sub78-tail-clipping-audit-20261128.json"))
    if any(digest(Path(path)) != expected for path, expected in audit["inputs"].items()):
        raise ValueError("Audit sources changed; do not resume a different recipe")
    cfg = load(FULL / "training_summary.json")["configuration"]
    cur, _, _ = replay_curriculum(FULL, {"configuration": cfg})
    prefix = mode == "prefix84"
    src = FULL / "checkpoint-80" if prefix else WORK / "prefix84/checkpoint-84"
    step = 84 if prefix else 85
    out = WORK / mode
    if out.exists():
        raise FileExistsError("Use the existing result or a fresh trial; preserving prior replay")
    expected_src_step = 80 if prefix else 84
    info = checkpoint_metadata(src / "training_state.pt")
    if info.get("global_step") != expected_src_step or info["configuration"] != cfg:
        raise ValueError("Resume checkpoint lacks the original optimizer, scheduler and RNG context")
    fn, scope = isolated_trainer()
    scope["fixed_tail"] = mode == "fixed85"
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(TRAINER), "exec"), scope)
    out.mkdir()
    resume = out / src.name
    shutil.copytree(src, resume, copy_function=os.link)
    summary = scope[fn.name](
        root=Path.cwd(), manifest_path=FULL_MANIFEST, prompt_path=PROMPT, out_dir=out,
        val_fold=cfg["val_fold"], epochs=cfg["epochs"], grad_accum=cfg["grad_accum"], lr=cfg["lr"],
        max_pixels=cfg["max_pixels"], seed=cfg["seed"], checkpoint_steps=10,
        resume_from=resume, stop_after_steps=step, curriculum_builder=lambda *_args, **_kwargs: cur,
        experiment_identity=cfg["experiment"],
    )
    ckpt = Path(summary["checkpoint"])
    if summary["optimizer_steps"] != step or checkpoint_metadata(ckpt / "training_state.pt")["global_step"] != step:
        raise ValueError("Replay stopped at an unexpected step")
    actual_adapter = digest(ckpt / "adapter_model.safetensors")
    original_steps = read_steps(FULL)
    logs = read_steps(out)
    original_adapter = None if prefix else digest(FULL / "epoch-1/adapter_model.safetensors")
    fields = ("step", "examples", "supervised_tokens", "decision_tokens", "loss", "applied_lr", "grad_norm")
    write(out / "completed.json", {
        "mode": mode,
        "step": step,
        "source_adapter_sha256": digest(src / "adapter_model.safetensors"),
        "source_training_state_sha256": digest(src / "training_state.pt"),
        "checkpoint_adapter_sha256": actual_adapter,
        "original_step85_adapter_sha256": original_adapter,
        "original_step85_adapter_exact": None if prefix else original_adapter == actual_adapter,
        "observed_step_log_matches_original": {
            str(i): {f: logs[i][f] == original_steps[i][f] for f in fields}
            for i in range(expected_src_step + 1, step + 1)
        },
        "final_step_log": logs[step],
    })


def inference():
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        raise RuntimeError("GPU hidden")
    from .v3_turnaround import _resolve_base_model_dir
    from .vlm_11code import infer_batched_vlm, load_qwen_4bit_or_bf16, seed_all
    import torch

    cfg = load(FULL / "training_summary.json")["configuration"]
    base = _resolve_base_model_dir(Path.cwd())
    cohort_path = WORK / "cohort.json"
    audit = load(Path("results/sub78-tail-clipping-audit-20261128.json"))
    if digest(cohort_path) != audit["cohort"]["frozen_sha256"]:
        raise ValueError("Fixed evaluation cohort changed")
    if digest(base / "config.json") != cfg["base_config_sha256"] or digest(PROMPT) != cfg["prompt_sha256"]:
        raise ValueError("Unexpected base config or prompt")
    rows = load(cohort_path)
    image_root = Path("data/train")
    if any(digest(image_root / r["image"]) != r["sha256"] for r in rows):
        raise ValueError("Fixed image bytes changed")
    adapters = {"original85": FULL / "epoch-1", "native85": WORK / "native85/epoch-1",
                "fixed85": WORK / "fixed85/epoch-1"}
    receipts = {arm: load(WORK / arm / "completed.json") for arm in ("native85", "fixed85")}
    if any(receipts["native85"][k] != receipts["fixed85"][k] for k in (
        "source_adapter_sha256", "source_training_state_sha256",
    )):
        raise ValueError("Arms did not start from the same step-84 state")
    pristine_by_adapter = {}
    for name, adapter in adapters.items():
        adapter_sha = digest(adapter / "adapter_model.safetensors")
        path = WORK / f"{name}-raw.json"
        if path.exists():
            raise FileExistsError(f"Preserving prior outputs: {path}")
        if name in receipts and receipts[name]["checkpoint_adapter_sha256"] != adapter_sha:
            raise ValueError("Replayed weights do not match completion receipt")
        if adapter_sha in pristine_by_adapter:
            origin, result = pristine_by_adapter[adapter_sha]
            timing = None
        else:
            origin = name
            seed_all(42)
            model, proc = load_qwen_4bit_or_bf16(base, 393216, False, adapter)
            result, timing = infer_batched_vlm(model, proc, rows, image_root, PROMPT.read_text(encoding="utf-8"),
                                               batch_size=2, max_new_tokens=620)
            del model, proc
            torch.cuda.empty_cache()
            pristine_by_adapter[adapter_sha] = (origin, result)
        write(path, {"adapter_sha256": adapter_sha, "cohort_sha256": digest(cohort_path),
                     "recipe": ["bf16", 393216, 2, 620],
                     "generation_origin_arm": origin, "generation_metadata": timing,
                     "newly_generated_rows": len(result) if origin == name else 0,
                     "reused_rows": len(result) if origin != name else 0,
                     "rows": [[r["sample_id"], output] for r, output in zip(rows, result, strict=True)]})


def tensor_header(stream):
    raw_size = stream.read(8)
    size, = struct.unpack("<Q", raw_size)
    if not 0 < size <= 8 * 2**20:
        raise ValueError("Invalid LoRA safetensor header length")
    header = json.loads(stream.read(size))
    header.pop("__metadata__", None)
    file_size = os.fstat(stream.fileno()).st_size
    offset = 0
    for name, spec in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
        start, end = spec["data_offsets"]
        if ".lora_" not in name or spec["dtype"] not in ("F32", "F16", "BF16"):
            raise ValueError(f"Unexpected LoRA tensor or dtype: {name}")
        width = 4 if spec["dtype"] == "F32" else 2
        if offset != start or end - start != math.prod(spec["shape"]) * width:
            raise ValueError(f"LoRA offsets or shape differ: {name}")
        offset = end
    if not header or offset + 8 + size != file_size:
        raise ValueError("LoRA payload length differs")
    return raw_size, size, header


def float64_chunk(raw, dtype):
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4").astype(np.float64)
    return np.frombuffer(raw, dtype={"F32": "<f4", "F16": "<f2"}[dtype]).astype(np.float64)


def adapter_movement(paths, expected_shas):
    """Compare all LoRA values using bounded FP64 chunks and content SHA."""
    with ExitStack() as context:
        streams = {arm: context.enter_context(p.open("rb")) for arm, p in paths.items()}
        meta = {arm: tensor_header(stream) for arm, stream in streams.items()}
        _, _, reference = meta["prefix84"]
        signature = {n: (v["shape"], v["dtype"]) for n, v in reference.items()}
        if any({n: (v["shape"], v["dtype"]) for n, v in h.items()} != signature for _, _, h in meta.values()):
            raise ValueError("LoRA tensor name, shape or dtype differs")
        import hashlib
        hashes = {}
        # Hash in each file's native order to compare serialized identity independently of tensor equality.
        for arm, stream in streams.items():
            stream.seek(0)
            hashes[arm] = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes != expected_shas:
            raise ValueError("LoRA tensor files differ from the frozen receipts")
        squares_native, squares_fixed, cross, exact_tensors = [], [], [], 0
        max_abs = 0.0
        for name, spec in sorted(reference.items(), key=lambda item: item[1]["data_offsets"][0]):
            for arm, stream in streams.items():
                _, size, header = meta[arm]
                stream.seek(8 + size + header[name]["data_offsets"][0])
            remaining = spec["data_offsets"][1] - spec["data_offsets"][0]
            native, fixed, dot, same = 0.0, 0.0, 0.0, True
            while remaining:
                take = min(remaining, 1024 * 1024)
                raw = {arm: stream.read(take) for arm, stream in streams.items()}
                if any(len(buf) != take for buf in raw.values()):
                    raise ValueError("Short LoRA stream")
                before, left, right, original = (float64_chunk(raw[arm], spec["dtype"])
                    for arm in ("prefix84", "native85", "fixed85", "original85"))
                if any(not np.isfinite(x).all() for x in (before, left, right, original)):
                    raise ValueError("Nonfinite LoRA tensor values")
                same &= raw["native85"] == raw["original85"]
                if len(left):
                    max_abs = max(max_abs, float(np.max(np.abs(left - original))))
                a, b = left - before, right - before
                native += float(np.dot(a, a))
                fixed += float(np.dot(b, b))
                dot += float(np.dot(a, b))
                remaining -= take
            exact_tensors += int(same)
            squares_native.append(native)
            squares_fixed.append(fixed)
            cross.append(dot)
    left, right = math.sqrt(math.fsum(squares_native)), math.sqrt(math.fsum(squares_fixed))
    return {
        "adapter_sha256": hashes, "checked_tensor_names_shapes_dtypes": len(reference),
        "elements": sum(math.prod(s["shape"]) for s in reference.values()),
        "prefix84_to_native85_delta_l2": left, "prefix84_to_fixed85_delta_l2": right,
        "native_over_fixed_delta_l2": left / right if right else None,
        "delta_cosine": math.fsum(cross) / (left * right) if left and right else None,
        "original_vs_native_exact_tensors": exact_tensors,
        "original_vs_native_tensor_values_bitwise_exact": exact_tensors == len(reference),
        "original_vs_native_max_abs": max_abs,
    }


def measure_adapter_movement():
    prefix = load(WORK / "prefix84/completed.json")
    native = load(WORK / "native85/completed.json")
    fixed = load(WORK / "fixed85/completed.json")
    sha = prefix["checkpoint_adapter_sha256"]
    if any(r["source_adapter_sha256"] != sha for r in (native, fixed)):
        raise ValueError("Arm checkpoint provenance differs")
    if (native["source_training_state_sha256"] != fixed["source_training_state_sha256"]
            or native["original_step85_adapter_sha256"] != fixed["original_step85_adapter_sha256"]):
        raise ValueError("Arm source optimizer or original reference differs")
    paths = {"prefix84": WORK / "prefix84/checkpoint-84", "native85": WORK / "native85/epoch-1",
             "fixed85": WORK / "fixed85/epoch-1", "original85": FULL / "epoch-1"}
    return adapter_movement({k: p / "adapter_model.safetensors" for k, p in paths.items()}, {
        "prefix84": sha, "native85": native["checkpoint_adapter_sha256"],
        "fixed85": fixed["checkpoint_adapter_sha256"], "original85": native["original_step85_adapter_sha256"],
    })


def score():
    from .sub78_verifier_audit import extract
    context = {"CODES": CODES}
    extract(Path("kwater_lab/hybrid_pipeline.py"), {"usable_draft_hazards"}, context)
    path = WORK / "cohort.json"
    rows = load(path)
    ids = [r["sample_id"] for r in rows]
    labels = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(LABELS, set(ids))}
    gold = {sid: {c for c, h in labels[sid].items() if h["risk_level"]["severity"]["score"] == 4} for sid in ids}
    by_arm, raw, generation_provenance = {}, {}, {}
    for arm in ("original85", "native85", "fixed85"):
        answer = load(WORK / f"{arm}-raw.json")
        if answer["cohort_sha256"] != digest(path) or answer["recipe"] != ["bf16", 393216, 2, 620]:
            raise ValueError("Prediction provenance differs")
        if [sid for sid, _ in answer["rows"]] != ids:
            raise ValueError("Prediction coverage or order differs")
        origin = answer["generation_origin_arm"]
        own = origin == arm
        if (answer["newly_generated_rows"], answer["reused_rows"]) != ((len(ids), 0) if own else (0, len(ids))):
            raise ValueError("Invalid generation provenance")
        if not own and (origin not in generation_provenance
                        or generation_provenance[origin]["adapter_sha256"] != answer["adapter_sha256"]
                        or raw[origin] != [pred for _, pred in answer["rows"]]):
            raise ValueError("Reused generation does not equal its source")
        generation_provenance[arm] = {key: answer[key] for key in (
            "adapter_sha256", "generation_origin_arm", "newly_generated_rows", "reused_rows",
        )}
        raw[arm] = [pred for _, pred in answer["rows"]]
        by_arm[arm] = {sid: set(context["usable_draft_hazards"](pred.get("parsed")))
                       for sid, pred in answer["rows"]}
    metrics = {}
    for arm, output in by_arm.items():
        metrics[arm] = {
            "normal_false_alarms": sum(bool(output[sid]) for sid in ids if not labels[sid]),
            "no_hazard_outputs": sum(not output[sid] for sid in ids),
            "severity4_complete_misses": sum(not (output[sid] & gold[sid]) for sid in ids if gold[sid]),
            "invalid_parse_rows": sum(not isinstance(out.get("parsed"), dict) for out in raw[arm]),
        }
    native, fixed = by_arm["native85"], by_arm["fixed85"]
    empty_to_hazard = [sid for sid in ids if not native[sid] and fixed[sid]]
    added = {sid: (fixed[sid] - native[sid]) & labels[sid].keys() for sid in ids}
    removed = {sid: (native[sid] - fixed[sid]) & labels[sid].keys() for sid in ids}
    return {
        "scope": "Small training-exposed paired diagnosis. Not an estimate of deployment accuracy.",
        "photos": len(ids), "normal_photos": sum(not labels[sid] for sid in ids),
        "severity4_photos": sum(bool(gold[sid]) for sid in ids),
        "original85_native85_code_parity": sum(by_arm["original85"][sid] == native[sid] for sid in ids),
        "original85_native85_generated_text_exact": sum(
            a["raw_text"] == b["raw_text"] for a, b in zip(raw["original85"], raw["native85"], strict=True)
        ),
        "adapter_movement": measure_adapter_movement(),
        "generation_provenance": {
            "unique_adapters_newly_generated": sum(p["newly_generated_rows"] > 0 for p in generation_provenance.values()),
            "newly_generated_rows": sum(p["newly_generated_rows"] for p in generation_provenance.values()),
            "reused_rows": sum(p["reused_rows"] for p in generation_provenance.values()),
            "arms": generation_provenance,
        },
        "metrics": metrics,
        "fixed_vs_native": {
            "severe_matches_gained": sum(not native[sid] & gold[sid] and bool(fixed[sid] & gold[sid]) for sid in ids),
            "severe_matches_lost": sum(bool(native[sid] & gold[sid]) and not fixed[sid] & gold[sid] for sid in ids),
            "hazard_to_empty": sum(bool(native[sid]) and not fixed[sid] for sid in ids),
            "empty_to_hazard": len(empty_to_hazard),
            "empty_to_hazard_with_any_gold_code": sum(bool(added[sid]) for sid in empty_to_hazard),
            "empty_to_hazard_gold_codes_added": sum(len(added[sid]) for sid in empty_to_hazard),
            "empty_to_hazard_new_codes": dict(Counter(code for sid in empty_to_hazard for code in fixed[sid])),
            "empty_to_hazard_correct_new_codes": dict(Counter(code for sid in empty_to_hazard for code in added[sid])),
            "changed_code_set_both_predicted_hazard": sum(
                bool(native[sid]) and bool(fixed[sid]) and native[sid] != fixed[sid] for sid in ids
            ),
            "all_changed_code_sets": sum(native[sid] != fixed[sid] for sid in ids),
            "gold_codes_added": sum(len(v) for v in added.values()),
            "gold_codes_lost": sum(len(v) for v in removed.values()),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prefix84", "native85", "fixed85", "infer", "score"))
    args = parser.parse_args()
    if args.stage == "prepare":
        write(Path("results/sub78-tail-clipping-audit-20261128.json"), prepare())
    elif args.stage == "infer":
        inference()
    elif args.stage == "score":
        write(Path("results/sub78-tail-replay-20261128.json"), score())
    else:
        replay_train(args.stage)
