"""Prepare an eval-only Fold 2 BF16 prompt contrast, then score returned raw caches locally."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from .fold2_8b_baseline import (
    BASE_CONFIG_SHA, BASE_REVISION, EVAL_MANIFEST, EVAL_SEGMENTS, IMAGE_ROOT, PIXELS, RUN, STEPS,
    add_tar_bytes, checkpoint_state, order_digest, sha_json, write_json,
)
from .fold4_same_verifier import digest, read
from .stage1_fold2_quality import (
    CONTROL_KEYS, MAX_NEW_TOKENS, MIN_PIXELS, receipt_cache_key, write_generation_receipt,
)


WORK = Path("data/stage1-prompt-remote-20261129")
BASE = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
ADAPTER = RUN / "fit/epoch-1"
EVAL_TAR = RUN / "eval.tar"
PREFLIGHT = Path("results/stage1-prompt-remote-preflight-20261129.json")
RESULT = Path("results/stage1-fold2-8b-mixed-inference-v3-v4-bf16-20261129.json")
PRIOR = Path("results/stage1-v4-prompt-contrast-preflight-20261129.json")
EVAL_PROOF = Path("results/fold2-8b-matched-preflight-20261128.json")
BASE_PROOF = Path("results/fold4-8b-epoch1-research-package-20261128.json")
MIXED_PROOF = Path("results/seoul-fold2-matched-8b-epoch1-20261129.json")
PACKAGE = Path("kwater_lab")
PLAN = Path("stage1-prompt-plan.json")
REMOTE_ADAPTER = Path("adapter")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
PROMPTS = {tag: PACKAGE / f"system_prompt_{tag}.txt" for tag in ("v3", "v4")}
FORMATTER, ENGINE = (PACKAGE / f"{name}.py" for name in ("v3_turnaround", "vlm_11code"))
SOURCE_FILES = tuple(PACKAGE / f"{name}.py" for name in (
    "audit", "fold0_group_sensitivity", "fold0_production_compare", "fold2_8b_baseline",
    "fold2_8b_paired_score", "fold2_v3_epoch1", "fold4_official_output_check", "fold4_same_verifier",
    "hybrid_public200_timing", "law_reranker_11code", "metrics", "offline_archive_audit",
    "stage1_fold2_quality", "stage1_prompt_remote", "sub78_public200_compare", "train_head",
    "v3_turnaround", "vlm_11code",
)) + tuple(PROMPTS.values())
RETURN_RECEIPT = Path("run-receipt.json")
RETURN_FILES = {RETURN_RECEIPT.as_posix()} | {
    f"{tag}/raw-{start}-{end}.json" for tag in PROMPTS for start, end, _ in EVAL_SEGMENTS
}


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def local_work(path):
    data, resolved = Path("data").resolve(), path.resolve()
    if resolved == data or not resolved.is_relative_to(data):
        raise ValueError("Keep restricted artifacts in a local data/ subdirectory")
    return path


def image_members(rows):
    members = {}
    ids = set()
    for row in rows:
        if set(row) != {"sample_id", "image", "sha256"} or row["sample_id"] in ids:
            raise ValueError("Evaluation input must contain only unique photo identity fields")
        name = PurePosixPath(row["image"])
        if name.is_absolute() or ".." in name.parts or "\\" in row["image"] or ":" in row["image"]:
            raise ValueError("Nonlocal evaluation photo path")
        member = (IMAGE_ROOT / name).as_posix()
        if member in members:
            raise ValueError("Duplicated evaluation photo path")
        members[member] = row["sha256"]
        ids.add(row["sample_id"])
    return members


def verify_tar(path, expected, manifest=None):
    """Check member allowlist, type, uniqueness and incremental file digests."""
    seen = set()
    with tarfile.open(path, mode="r:") as archive:
        for entry in archive:
            if not entry.isfile() or entry.name not in expected or entry.name in seen:
                raise ValueError("Archive contains a nonregular, unexpected or duplicate member")
            with archive.extractfile(entry) as stream:
                if entry.name == EVAL_MANIFEST.as_posix() and manifest is not None:
                    if json.load(stream) != manifest:
                        raise ValueError("Evaluation archive has the wrong ordered image-only manifest")
                elif sha256_stream(stream) != expected[entry.name]:
                    raise ValueError("Archive member does not match its frozen content fingerprint")
            seen.add(entry.name)
    if seen != expected.keys():
        raise ValueError("Archive is missing required members")


def sha256_stream(stream):
    checksum = sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(chunk)
    return checksum.hexdigest()


def base_members(base):
    upstream = read(base / "source.json")
    if upstream["repo"] != "Qwen/Qwen3-VL-8B-Instruct" or upstream["revision"] != BASE_REVISION:
        raise ValueError("Wrong pinned 8B model source")
    known = {Path(name).name: value for name, value in read(BASE_PROOF)["base_member_sha256"].items()}
    if known["config.json"] != BASE_CONFIG_SHA:
        raise ValueError("Pinned BF16 production-byte proof changed")
    members = {}
    for item in upstream["files"]:
        name = item["rfilename"]
        if name == "README.md":
            continue
        path = base / name
        expected = known[name]
        observed = item["lfs"]["sha256"] if "lfs" in item else digest(path)
        if observed != expected or path.stat().st_size != item["size"]:
            raise ValueError("BF16 source metadata or official shard proof differs")
        members[name] = {"bytes": item["size"], "sha256": expected}
    if members.keys() != known.keys():
        raise ValueError("Incomplete pinned BF16 source inventory")
    return members


def training_cases(train):
    from .fold2_v3_epoch1 import LABELS
    from .law_reranker_11code import read_selected_cases

    ids = {row["sample_id"] for row in train}
    cases = {row["sample_id"]: row for row in read_selected_cases(LABELS, ids)}
    if len(ids) != 769 or cases.keys() != ids:
        raise ValueError("Only the fixed 769 training-side cases are eligible for law formatting")
    return [cases[row["sample_id"]] for row in train]


def prepare(work, base, result):
    from .fold2_v3_epoch1 import LAWS, split

    work = local_work(work)
    work.mkdir(parents=True, exist_ok=True)
    train, outer = split()
    eval_rows = [{key: row[key] for key in ("sample_id", "image", "sha256")} for row in outer]
    exam = read(EVAL_PROOF)["prepared_payloads"]["eval_tar"]
    transfer = read(RUN / "transfer-receipt.json")
    if exam != transfer["eval_tar"] or identity(EVAL_TAR) != exam:
        raise ValueError("Expected the unchanged historical eval-only TAR")
    verify_tar(EVAL_TAR, {**image_members(eval_rows), EVAL_MANIFEST.as_posix(): None}, eval_rows)
    recipe = transfer["recipe"]
    if (recipe["eval_manifest_sha256"] != sha_json(eval_rows)
            or recipe["eval_order_sha256"] != order_digest(eval_rows)
            or recipe["eval_segments"] != [list(row) for row in EVAL_SEGMENTS]):
        raise ValueError("Frozen 197-photo or batch geometry changed")
    state = checkpoint_state(ADAPTER, STEPS, recipe)
    train_config_sha = sha_json(state["configuration"])
    del state
    adapter = {name: identity(ADAPTER / name) for name in ADAPTER_FILES}
    adapter_sha = adapter["adapter_model.safetensors"]["sha256"]
    if adapter_sha != read(MIXED_PROOF)["adapter_sha256"]:
        raise ValueError("Expected the recovered measured mixed Fold 2 first epoch")
    raw_profiles = read(PRIOR)["prompt_profiles"]
    base_inputs = base_members(base)
    sources = {path.as_posix(): path.read_bytes() for path in SOURCE_FILES}
    fingerprints = {name: sha256(value).hexdigest() for name, value in sources.items()}
    actual_common = {
        "adapter_sha256": adapter_sha, "adapter_config_sha256": adapter["adapter_config.json"]["sha256"],
        "train_configuration_sha256": train_config_sha, "law_corpus_sha256": digest(LAWS),
        "train_case_bank_sha256": sha_json(training_cases(train)), "formatter_sha256": fingerprints[FORMATTER.as_posix()],
        "engine_sha256": fingerprints[ENGINE.as_posix()], "precision": "bf16", "base_revision": BASE_REVISION,
        "min_pixels": MIN_PIXELS, "max_pixels": PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False, "use_cache": True,
    }
    arms = {tag: {"controls": raw_profiles[tag]["controls"],
                  "prefix_sha256": raw_profiles[tag]["rendered_prefix_sha256"]} for tag in PROMPTS}
    for tag, arm in arms.items():
        c = arm["controls"]
        if (c.keys() != CONTROL_KEYS - {"runtime_sha256"} or any(c[key] != val for key, val in actual_common.items())
                or c["prompt_sha256"] != fingerprints[PROMPTS[tag].as_posix()]):
            raise ValueError("Prepared control changed from the measured mixed v3/v4 prompt preflight")
    if {k for k in arms["v3"]["controls"] if arms["v3"]["controls"][k] != arms["v4"]["controls"][k]} != {"prompt_sha256"}:
        raise ValueError("Prompt must be the only changed static control")
    plan = {"eval_tar": exam, "eval_manifest_sha256": sha_json(eval_rows), "eval_order_sha256": order_digest(eval_rows),
            "segments": recipe["eval_segments"], "base": base_inputs, "adapter": adapter, "source": fingerprints,
            "arms": arms}
    write_json(work / PLAN, plan)
    source_tar = work / "source.tar"
    pending = source_tar.with_suffix(".tar.part")
    with tarfile.open(pending, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, value in sorted({**sources, PLAN.as_posix(): (work / PLAN).read_bytes()}.items()):
            add_tar_bytes(archive, name, value)
    verify_tar(pending, {**fingerprints, PLAN.as_posix(): digest(work / PLAN)})
    pending.replace(source_tar)
    write_json(result, {
        "scope": "Verified local transfer inputs for a single BF16 Fold 2 inference-only prompt contrast",
        "eval_tar": {"path": EVAL_TAR.as_posix(), **exam, "photo_members": len(eval_rows), "manifest_members": 1},
        "source_tar": {"path": source_tar.as_posix(), **identity(source_tar), "members": sorted([*fingerprints, PLAN.as_posix()])},
        "adapter_upload": {"local_directory": ADAPTER.as_posix(), "members": adapter},
        "return_members": sorted(RETURN_FILES), "plan_sha256": sha_json(plan),
        "plan_file_sha256": digest(work / PLAN), "eval_order_sha256": plan["eval_order_sha256"],
        "segments": plan["segments"], "prompt_profiles": arms, "pinned_base_files": base_inputs,
        "new_training_steps": 0, "new_generations": 0, "uploads": 0,
    })
    print(f"PROMPT_REMOTE_PREPARED {len(eval_rows)} {digest(source_tar)}", flush=True)


def verify_inputs(base, plan, rows):
    if (sha_json(rows) != plan["eval_manifest_sha256"] or order_digest(rows) != plan["eval_order_sha256"]
            or plan["segments"] != [list(s) for s in EVAL_SEGMENTS] or len(rows) != EVAL_SEGMENTS[-1][1]):
        raise ValueError("Wrong prompt comparison evaluation geometry")
    photos = image_members(rows)
    if {p.as_posix() for p in Path("data").rglob("*") if p.is_file()} != photos.keys() | {EVAL_MANIFEST.as_posix()}:
        raise ValueError("The remote data tree must contain exactly the evaluation manifest and photos")
    if (plan["source"].keys() != {path.as_posix() for path in SOURCE_FILES}
            or any(digest(Path(name)) != value for name, value in plan["source"].items())
            or any(digest(Path(name)) != value for name, value in photos.items())):
        raise ValueError("Transferred source or photo bytes differ from the verified input")
    if (plan["adapter"].keys() != set(ADAPTER_FILES)
            or {p.name for p in REMOTE_ADAPTER.iterdir()} != set(ADAPTER_FILES)
            or any(identity(REMOTE_ADAPTER / name) != value for name, value in plan["adapter"].items())):
        raise ValueError("Remote adapter directory differs from the exact two-file upload")
    allowed = set(plan["base"]) | {"source.json", "README.md", ".cache", ".gitattributes"}
    if {p.name for p in base.iterdir()} - allowed or any(identity(base / name) != value for name, value in plan["base"].items()):
        raise ValueError("Remote base differs from the exact pinned BF16 publisher files")


def loaded_adapter_count(model):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    imported = get_peft_model_state_dict(model)
    with safe_open(str(REMOTE_ADAPTER / "adapter_model.safetensors"), framework="pt", device="cpu") as original:
        if set(imported) != set(original.keys()) or len(imported) != 520 or any(
                value.dtype != torch.float32 or not torch.equal(value.detach().cpu(), original.get_tensor(name))
                for name, value in imported.items()):
            raise RuntimeError("The resident adapter differs from all 520 original mixed LoRA tensors")
    return len(imported)


def infer(base, out):
    import torch
    from .stage1_fold2_quality import runtime_fingerprint
    from .vlm_11code import INSPECT_USER_TEXT, infer_batched_vlm, load_qwen_model, load_qwen_processor

    plan, rows = read(PLAN), read(EVAL_MANIFEST)
    verify_inputs(base, plan, rows)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("An isolated, BF16-capable CUDA device is required")
    if out.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Put remote outputs outside the input-only data tree")
    out.mkdir(parents=True, exist_ok=True)
    runtime = runtime_fingerprint()
    processor = load_qwen_processor(base, PIXELS)
    model = load_qwen_model(base, use_4bit=False, adapter_dir=REMOTE_ADAPTER).eval()
    if model.get_base_model().dtype != torch.bfloat16 or any(p.device != torch.device("cuda:0") for p in model.parameters()):
        raise RuntimeError("Both arms require the same CUDA-resident BF16 model")
    receipt = {"plan_sha256": sha_json(plan), "runtime_sha256": runtime, "base_inventory_sha256": sha_json(plan["base"]),
               "eval_order_sha256": order_digest(rows), "loaded_adapter_tensors": loaded_adapter_count(model), "arms": {}}
    for tag, prompt in PROMPTS.items():
        system = prompt.read_text(encoding="utf-8")
        prefix = processor.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]},
        ], tokenize=False, add_generation_prompt=True)
        profile = plan["arms"][tag]
        if sha256(prefix.encode()).hexdigest() != profile["prefix_sha256"]:
            raise ValueError("Runtime-rendered prompt differs from the fixed v3/v4 preflight")
        controls = {**profile["controls"], "runtime_sha256": runtime}
        cache_key = receipt_cache_key(controls)
        dest = out / tag
        dest.mkdir(exist_ok=True)
        segments = []
        for start, end, batch in EVAL_SEGMENTS:
            cache = dest / f"raw-{start}-{end}.json"
            if cache.exists():
                state = read(cache)
                completed = len(state["outputs"])
                if state["effective_batch_size"] != batch or (completed != end-start and completed % batch):
                    raise ValueError("Only an exact-batch partial cache can resume")
            outputs, stats = infer_batched_vlm(model, processor, rows[start:end], IMAGE_ROOT, system, batch_size=batch,
                                              max_new_tokens=MAX_NEW_TOKENS, cache_path=cache, cache_key=cache_key)
            if len(outputs) != end-start or stats["final_effective_batch_size"] != batch:
                raise RuntimeError("CUDA fallback changed the fixed batch schedule")
            segments.append({"start": start, "end": end, "raw_sha256": digest(cache), "stats": stats})
        receipt["arms"][tag] = {"controls": controls, "segments": segments}
    if runtime_fingerprint() != runtime:
        raise ValueError("Runtime changed between inference arms")
    write_json(out / RETURN_RECEIPT, receipt)


def format_arm(tag, plan, returned, destination, receipt, outer, ranker):
    from .stage1_fold2_quality import validate_arm
    from .v3_turnaround import format_v3_record

    arm = receipt["arms"][tag]
    expected = {**plan["arms"][tag]["controls"], "runtime_sha256": receipt["runtime_sha256"]}
    if arm["controls"] != expected:
        raise ValueError("Returned arm changed a frozen control")
    dest = destination / tag
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for segment, (start, end, _) in zip(arm["segments"], EVAL_SEGMENTS, strict=True):
        name = f"raw-{start}-{end}.json"
        raw = returned / tag / name
        if (segment["start"], segment["end"], segment["raw_sha256"]) != (start, end, digest(raw)):
            raise ValueError("Returned segment bytes or boundaries differ")
        cache = read(raw)
        if cache["identity"]["prefix_sha256"] != plan["arms"][tag]["prefix_sha256"]:
            raise ValueError("Returned cache has the wrong rendered prompt")
        rows.extend(format_v3_record(row["sample_id"], cache["outputs"][row["sample_id"]]["parsed"], ranker)
                    for row in outer[start:end])
        shutil.copyfile(raw, dest / name)
    predictions = dest / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    witness = dest / "receipt.json"
    write_generation_receipt(witness, predictions, outer, arm["segments"], expected, loaded_adapter_verified=True,
                             extras={"remote_run_receipt_sha256": digest(returned / RETURN_RECEIPT), "infer_prompt": tag})
    validate_arm(predictions, witness, outer)
    return predictions, witness


def score(work, returned, result):
    from .fold2_v3_epoch1 import LAWS, split
    from .stage1_fold2_quality import compare
    from .v3_turnaround import ActionSupportedLawRetriever

    local_work(returned)
    local_work(work)
    if {p.relative_to(returned).as_posix() for p in returned.rglob("*") if p.is_file()} != RETURN_FILES:
        raise ValueError("Return only the completed run receipt and four original raw caches")
    plan, remote = read(work / PLAN), read(returned / RETURN_RECEIPT)
    train, outer = split()
    cases = training_cases(train)
    common = plan["arms"]["v3"]["controls"]
    if (remote["plan_sha256"] != sha_json(plan) or remote["base_inventory_sha256"] != sha_json(plan["base"])
            or remote["eval_order_sha256"] != order_digest(outer) or remote["loaded_adapter_tensors"] != 520
            or remote["arms"].keys() != PROMPTS.keys() or digest(LAWS) != common["law_corpus_sha256"]
            or sha_json(cases) != common["train_case_bank_sha256"] or digest(FORMATTER) != common["formatter_sha256"]):
        raise ValueError("Returned run or local law formatter differs from the frozen inputs")
    laws = [json.loads(line) for line in LAWS.read_text(encoding="utf-8").splitlines() if line]
    ranker = ActionSupportedLawRetriever(laws, cases)
    a, ar = format_arm("v3", plan, returned, work / "scored", remote, outer, ranker)
    b, br = format_arm("v4", plan, returned, work / "scored", remote, outer, ranker)
    audit = compare(a, b, ar, br, "prompt")
    write_json(result, {**audit, "remote_run_receipt_sha256": digest(returned / RETURN_RECEIPT),
                        "remote_plan_sha256": sha_json(plan), "remote_base_inventory_sha256": remote["base_inventory_sha256"],
                        "remote_lora_tensors_verified": remote["loaded_adapter_tensors"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "infer", "score"))
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--returned", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.work, args.base, args.result or PREFLIGHT)
    elif args.phase == "infer":
        infer(args.base, args.returned or Path("returned"))
    else:
        score(args.work, args.returned or args.work / "returned", args.result or RESULT)


if __name__ == "__main__":
    main()
