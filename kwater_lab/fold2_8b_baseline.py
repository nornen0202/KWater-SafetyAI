"""Prepare isolated inputs, run the Fold 2 8B one-epoch arm and preserve its paired evidence."""

import argparse
import ast
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import tarfile
from zipfile import ZipFile

from .fold2_v3_epoch1 import (
    DEFAULT_RUN as CONTROL, LABELS, LAWS, MANIFEST, MASK, OOF, PROMPT, split,
)
from .fold4_same_verifier import digest, read
from .metrics import unique_keys

RUN = Path("data/fold2-8b-matched-epoch1")
BASE_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
BASE_CONFIG_SHA = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
CONTROL_COMMIT = "c61bd9ef16e8578d6f0b7c0a849962780a09b7d6"
STEPS = 67
TOTAL = 134
ACCUM = 16
PIXELS = 393216
SEED = 42
EVAL_SEGMENTS = ((0, 44, 4), (44, 197, 2))
TRAIN_MANIFEST = Path("data/fold2-train-manifest.json")
EVAL_MANIFEST = Path("data/fold2-eval-manifest.json")
CURRICULUM = Path("data/fold2-curriculum.json")
RECIPE = Path("data/fold2-recipe.json")
IMAGE_ROOT = Path("data/train")
IMAGE_ARCHIVE = Path("data/packages/kwater-train.zip")
LEADING_LABEL_ID = re.compile(r'^\s*\{\s*"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")')
LOCAL_BASE = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
PREFLIGHT_RESULT = Path("results/fold2-8b-matched-preflight-20261128.json")


def sha_json(payload):
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path, payload):
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    pending.replace(path)


def order_digest(rows):
    return sha_json([row["sample_id"] for row in rows])


def add_tar_bytes(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(payload))


def add_images(archive, rows):
    by_name = {row["image"]: row["sha256"] for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("Canonical payload includes duplicate image paths")
    with ZipFile(IMAGE_ARCHIVE) as source:
        # Stream the official ZIP in physical order to limit random reads on G:.
        images = sorted((source.getinfo(name) for name in by_name), key=lambda item: item.header_offset)
        for item in images:
            rel = Path(item.filename)
            payload = source.read(item)
            if rel.is_absolute() or ".." in rel.parts or sha256(payload).hexdigest() != by_name[item.filename]:
                raise ValueError("Official image differs from the canonical manifest")
            add_tar_bytes(archive, (IMAGE_ROOT / rel).as_posix(), payload)


def make_tar(path, members, rows):
    pending = path.with_suffix(".tar.part")
    with tarfile.open(pending, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in members.items():
            add_tar_bytes(archive, name.as_posix(), payload)
        add_images(archive, rows)
    pending.replace(path)
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    old = read(CONTROL / "frozen-recipe.json")
    curriculum = read(CONTROL / "curriculum-local.json")
    sources = {"manifest": MANIFEST, "mask": MASK, "strict_inner_oof": OOF,
               "prompt": PROMPT, "law_corpus": LAWS, "label_source": LABELS}
    if (old["steps_per_epoch"] != STEPS or sha_json(curriculum) != old["curriculum_training_sha256"]
            or any(digest(path) != old[f"{name}_sha256"] for name, path in sources.items())):
        raise ValueError("The Fold 2 inputs differ from the original 4B run")
    train_rows, eval_rows = split()
    images = {row["sample_id"]: row["image"] for row in train_rows}
    labels = []
    with LABELS.open(encoding="utf-8") as labels_file:
        for line in labels_file:
            if not line.strip():
                continue
            match = LEADING_LABEL_ID.match(line)
            if not match:
                raise ValueError("Official label line has no leading sample identifier")
            if json.loads(match[1]) in images:
                labels.append(json.loads(line, object_pairs_hook=unique_keys))
    if {row["sample_id"] for row in labels} != images.keys() or len(labels) != len(images):
        raise ValueError("Training label coverage differs from the frozen training split")
    if any(images.get(row.get("orig_id", row["sample_id"].split("__", 1)[0])) != row["image"]
           for row in curriculum):
        raise ValueError("A frozen curriculum example references a photo outside the training split")
    eval_input = [{key: row[key] for key in ("sample_id", "image", "sha256")} for row in eval_rows]
    ordered = curriculum.copy()
    random.Random(SEED + 1000).shuffle(ordered)
    steps = [json.loads(line) for line in (CONTROL / "steps.jsonl").read_bytes().splitlines()]
    if [(s["epoch"], s["step"], s["total_steps"], s["examples"]) for s in steps] != [
        (1, i, TOTAL, min(ACCUM, len(ordered) - (i-1)*ACCUM)) for i in range(1, STEPS+1)
    ]:
        raise ValueError("Historical 4B update boundaries disagree with the first-epoch recipe")
    recipe = {
        "arm": "fold2_8b_matched_epoch1", "base_revision": BASE_REVISION, "base_config_sha256": BASE_CONFIG_SHA,
        "control_source_commit": CONTROL_COMMIT, "training": old, "training_order_sha256": order_digest(ordered),
        "train_manifest_sha256": sha_json(train_rows), "eval_manifest_sha256": sha_json(eval_input),
        "eval_order_sha256": order_digest(eval_input), "eval_segments": [list(s) for s in EVAL_SEGMENTS],
        "epoch1_step_controls_4b": [
            [s[k] for k in ("examples", "supervised_tokens", "decision_tokens", "applied_lr")] for s in steps
        ],
        "control_4b_steps_sha256": digest(CONTROL / "steps.jsonl"),
    }
    labels_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels).encode()
    tracked_code = subprocess.check_output(["git", "archive", CONTROL_COMMIT, "kwater_lab"])
    with tarfile.open(fileobj=io.BytesIO(tracked_code)) as source:
        code = {Path(item.name): source.extractfile(item).read() for item in source
                if item.isfile() and Path(item.name).suffix == ".py"}
    if any(sha256(code[Path("kwater_lab") / name]).hexdigest() != value for name, value in old["source"].items()):
        raise ValueError("Pinned baseline implementation differs from the original 4B source")
    code[PROMPT] = PROMPT.read_bytes()
    source_path = Path(__file__)
    code[source_path.relative_to(Path.cwd())] = source_path.read_bytes()
    train = make_tar(out / "train.tar", code | {
        TRAIN_MANIFEST: json.dumps(train_rows).encode(), CURRICULUM: json.dumps(curriculum).encode(),
        LABELS: labels_bytes, RECIPE: json.dumps(recipe).encode(),
    }, train_rows)
    print("TRAIN_TAR " + json.dumps(train), flush=True)
    evaluate = make_tar(out / "eval.tar", {EVAL_MANIFEST: json.dumps(eval_input).encode()}, eval_rows)
    result = {"recipe": recipe, "train_tar": train, "eval_tar": evaluate}
    write_json(out / "transfer-receipt.json", result)
    print(json.dumps({"train_tar": train, "eval_tar": evaluate,
                      "training_order_sha256": recipe["training_order_sha256"],
                      "eval_order_sha256": recipe["eval_order_sha256"]}), flush=True)


def verify_prepared(out, result):
    train, outer = split()
    receipt = read(out / "transfer-receipt.json")
    recipe = receipt["recipe"]
    if recipe["training"] != read(CONTROL / "frozen-recipe.json"):
        raise ValueError("Prepared recipe differs from the frozen 4B control")
    counts = {}
    for label, rows, manifest in (("train", train, TRAIN_MANIFEST), ("eval", outer, EVAL_MANIFEST)):
        expected_images = {(IMAGE_ROOT / row["image"]).as_posix(): row for row in rows}
        path = out / f"{label}.tar"
        if path.stat().st_size != receipt[f"{label}_tar"]["bytes"]:
            raise ValueError("Prepared archive size differs from sender receipt")
        with tarfile.open(path) as archive:
            def member(name):
                return archive.extractfile(name.as_posix()).read()

            files = archive.getmembers()
            names = {item.name for item in files}
            images = {name for name in names if name.startswith("data/train/images/")}
            manifest_rows = json.loads(member(manifest))
            if (len(names) != len(files) or any(not item.isfile() for item in files)
                    or images != expected_images.keys() or manifest_rows != (
                        rows if label == "train" else [{key: row[key] for key in ("sample_id", "image", "sha256")}
                                                      for row in rows]
                    ) or sha_json(manifest_rows) != recipe[f"{label}_manifest_sha256"]):
                raise ValueError("Prepared archive contains incorrect photos, membership or order")
            if label == "eval":
                if names - images != {manifest.as_posix()}:
                    raise ValueError("Evaluation TAR contains material beyond frozen photos and manifest")
            else:
                expected_ids = {row["sample_id"] for row in rows}
                labs = [json.loads(line)["sample_id"] for line in member(LABELS).splitlines()]
                data_files = {name for name in names - images if name.startswith("data/")}
                if (data_files != {p.as_posix() for p in (manifest, CURRICULUM, LABELS, RECIPE)}
                        or len(labs) != len(expected_ids) or set(labs) != expected_ids
                        or json.loads(member(RECIPE)) != recipe
                        or sha_json(json.loads(member(CURRICULUM))) != recipe["training"]["curriculum_training_sha256"]):
                    raise ValueError("Prepared training labels or curriculum differ from the frozen training set")
                for filename, expected_hash in recipe["training"]["source"].items():
                    if sha256(member(Path("kwater_lab") / filename)).hexdigest() != expected_hash:
                        raise ValueError("Prepared training source differs from pinned implementation")
                worker = member(Path("kwater_lab/fold2_8b_baseline.py"))
                current = Path(__file__).read_bytes()
                funcs = {"latest_checkpoint", "require_checkpoint_files", "checkpoint_state",
                         "experiment_identity", "train", "infer"}

                def runtime_ast(source):
                    return {node.name: ast.dump(node) for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name in funcs}

                if runtime_ast(worker) != runtime_ast(current):
                    raise ValueError("Prepared worker functions differ from the local launch recipe")
            counts[label] = {"photos": len(images), "manifest_groups": len({row["group_id"] for row in rows})}
    source = read(LOCAL_BASE / "source.json")
    assets = [item for item in source["files"] if item.get("lfs")]
    if (source["revision"] != BASE_REVISION or digest(LOCAL_BASE / "config.json") != BASE_CONFIG_SHA
            or len(assets) != 4 or any((LOCAL_BASE / item["rfilename"]).stat().st_size != item["size"] for item in assets)):
        raise ValueError("Local 8B stock assets differ from the documented revision or sizes")
    batch = read(Path("results/fold2-v3-4b-epoch1-batch-witness-20261128.json"))
    if (batch["batch_transitions"] != [{"rows_already_complete": start, "effective_batch_size": size}
                                      for start, _, size in EVAL_SEGMENTS]
            or batch["raw_cache_sha256"] != digest(CONTROL / "epoch1-bf16-raw-cache.json")):
        raise ValueError("Original successful 4B evaluation batch boundaries differ")
    write_json(result, {
        "scope": "CPU preflight and isolated transfer membership audited on G:. Model training and BF16 evaluation are separate.",
        "prepared_payloads": {key: receipt[key] for key in ("train_tar", "eval_tar")},
        "memberships": counts, "eval_labels_in_payload_count": 0, "eval_split_images_in_train_payload_count": 0,
        "worker_payload_sha256": sha256(worker).hexdigest(), "worker_runtime_ast_matches_local": True,
        "control_source_commit": CONTROL_COMMIT, "frozen_control": recipe["training"],
        "training_order_sha256": recipe["training_order_sha256"], "eval_order_sha256": recipe["eval_order_sha256"],
        "control_4b_steps_sha256": recipe["control_4b_steps_sha256"], "eval_batch_witness": batch,
        "local_8b_assets": {"revision": source["revision"], "config_sha256": BASE_CONFIG_SHA,
                           "shard_bytes": sum(item["size"] for item in assets), "present_shards": len(assets),
                           "published_expected_sha256": {item["rfilename"]: item["lfs"]["sha256"] for item in assets},
                           "verification": "Current sizes and config fingerprint; verify published shard digests on run host."},
    })
    print(f"PREFLIGHT_VERIFIED {result}", flush=True)


def latest_checkpoint(out):
    found = []
    for candidate in out.iterdir():
        match = re.fullmatch(r"(?:checkpoint-(2|[1-6]0)|(epoch-1))", candidate.name)
        if match:
            require_checkpoint_files(candidate)
            found.append((STEPS if match[2] else int(match[1]), candidate))
        elif re.fullmatch(r"(checkpoint|epoch)-\d+", candidate.name):
            raise ValueError("Found an unexpected checkpoint outside the first-epoch schedule")
    return max(found, default=(0, None))


def require_checkpoint_files(path):
    for name in ("adapter_config.json", "adapter_model.safetensors", "training_state.pt"):
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f"Incomplete published checkpoint {path.name}")


def checkpoint_state(path, step, recipe):
    import torch

    require_checkpoint_files(path)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True, mmap=True)
    expected_config = {key: recipe["training"]["optimizer"][key] for key in ("lr", "grad_accum", "max_pixels", "seed")}
    expected_config.update(val_fold=2, epochs=2, curriculum_sha256=recipe["training"]["curriculum_training_sha256"],
                           prompt_sha256=recipe["training"]["prompt_sha256"], base_config_sha256=BASE_CONFIG_SHA,
                           experiment=experiment_identity(recipe))
    expected_cursor = (step, 2, 0) if step == STEPS else (step, 1, ACCUM*step)
    if (state["configuration"] != expected_config
            or (state["global_step"], state["next_epoch"], state["next_offset"]) != expected_cursor
            or [(r["epoch"], r["step"], r["total_steps"]) for r in state["records"]]
            != [(1, i, TOTAL) for i in range(1, step+1)]
            or state["scheduler"]["last_epoch"] != step):
        raise ValueError("Checkpoint identity, scheduler, cursor or ordered records disagree with the recipe")
    return state


def experiment_identity(recipe):
    return {"arm": recipe["arm"], "base_revision": BASE_REVISION,
            "strict_inner_oof_sha256": recipe["training"]["strict_inner_oof_sha256"]}


def train(base, out):
    from .vlm_11code import train_multimodal_qlora

    recipe, curriculum = read(RECIPE), read(CURRICULUM)
    if (digest(base / "config.json") != BASE_CONFIG_SHA
            or digest(PROMPT) != recipe["training"]["prompt_sha256"]
            or sha_json(curriculum) != recipe["training"]["curriculum_training_sha256"]
            or sha_json(read(TRAIN_MANIFEST)) != recipe["train_manifest_sha256"]):
        raise ValueError("The 8B base configuration or training material changed")
    out.mkdir(parents=True, exist_ok=True)
    step, resume = latest_checkpoint(out)
    if resume is not None:
        checkpoint_state(resume, step, recipe)
    if step == STEPS:
        print("EPOCH1_REUSED", flush=True)
        return
    if resume is None and (out / "steps.jsonl").exists():
        # A preemption before the step-2 sanity checkpoint cannot resume an optimizer state.
        abandoned = len(list(out.glob(".abandoned-steps-*.jsonl")))
        (out / "steps.jsonl").rename(out / f".abandoned-steps-{abandoned}.jsonl")
    os.environ["KWATER_LOCAL_BASE_MODEL"] = str(base.resolve())
    train_multimodal_qlora(root=Path.cwd(), manifest_path=TRAIN_MANIFEST, prompt_path=PROMPT, out_dir=out,
                          val_fold=2, epochs=2, grad_accum=ACCUM, lr=.00014, max_pixels=PIXELS, seed=SEED,
                          checkpoint_steps=10, resume_from=resume, stop_after_steps=STEPS,
                          curriculum_builder=lambda *_args, **_kw: curriculum,
                          experiment_identity=experiment_identity(recipe))
    checkpoint_state(out / "epoch-1", STEPS, recipe)


def infer(base, fit, out):
    from .vlm_11code import infer_batched_vlm, load_qwen_model, load_qwen_processor

    recipe = read(RECIPE)
    rows = read(EVAL_MANIFEST)
    adapter = fit / "epoch-1"
    checkpoint_state(adapter, STEPS, recipe)
    if (digest(base / "config.json") != BASE_CONFIG_SHA or digest(PROMPT) != recipe["training"]["prompt_sha256"]
            or sha_json(rows) != recipe["eval_manifest_sha256"]):
        raise ValueError("Evaluation base/config/prompt/photos differ from the frozen recipe")
    if any(digest(IMAGE_ROOT / row["image"]) != row["sha256"] for row in rows):
        raise ValueError("Evaluation image differs from the canonical manifest")
    out.mkdir(parents=True, exist_ok=True)
    processor = load_qwen_processor(base, PIXELS)
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    cache_key = f"fold2-8b:{BASE_REVISION}:{adapter_sha}:{digest(PROMPT)}:bf16:px{PIXELS}"
    model = None

    def load_model():
        nonlocal model
        if model is None:
            model = load_qwen_model(base, use_4bit=False, adapter_dir=adapter)
        return model

    segments = []
    for start, end, batch in EVAL_SEGMENTS:
        cache = out / f"raw-{start}-{end}.json"
        _, stats = infer_batched_vlm(None, processor, rows[start:end], IMAGE_ROOT,
                                    PROMPT.read_text(encoding="utf-8"), batch_size=batch,
                                    max_new_tokens=620, cache_path=cache, cache_key=cache_key, model_factory=load_model)
        if stats["final_effective_batch_size"] != batch:
            raise RuntimeError("CUDA backoff changed the original 4B batch boundaries")
        segments.append({"start": start, "end": end, "raw_sha256": digest(cache), "stats": stats})
    write_json(out / "inference-receipt.json", {"adapter_sha256": adapter_sha, "cache_key": cache_key,
                                                "eval_order_sha256": order_digest(rows), "segments": segments})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify-prepared", "train", "infer"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--result", type=Path, default=PREFLIGHT_RESULT)
    args = parser.parse_args()
    if args.stage == "prepare":
        if not args.out.resolve().is_relative_to(Path("data").resolve()):
            parser.error("Restricted Fold 2 payloads must stay in data/")
        prepare(args.out)
    elif args.stage == "verify-prepared":
        verify_prepared(args.out, args.result)
    elif args.stage == "train" and args.base:
        train(args.base.resolve(), args.out.resolve())
    elif args.stage == "infer" and args.base and args.fit:
        infer(args.base.resolve(), args.fit.resolve(), args.out.resolve())
    else:
        parser.error("Training needs --base; inference also needs --fit")


if __name__ == "__main__":
    main()
