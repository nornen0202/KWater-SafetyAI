"""Preflight and run a matched official-only verifier A/B on frozen Fold 0 proposals."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random

from .audit import digest
from .fold0_group_sensitivity import EXCLUSIONS, MANIFEST, validate_mask
from .fold0_production_compare import fold_rows
from .law_reranker_11code import read_selected_cases
from .metrics import CODES, decode_gold, unique_keys
from .train_head import save_atomic
from .verifier_cache import verify_with_cache


ROOT = Path("data/fold0-verifier-ab")
PROBE = Path("data/local-eval-predictions/fold0/v2-official-anchored/frozen_proposals.json")
PROBE_WITNESS = Path("results/v2_official_anchored_verified.json")
SOURCE_PAIRS = Path("data/curation-side-20261127/real-pairs.json")
PROMPT = Path("kwater_lab/system_prompt_11code.txt")
BASE = Path("data/baseline/base_model")
LABELS = Path("data/train/train.jsonl")
PLAN = {"primary_min_accuracy_delta": 0.03, "require_positive_group_bootstrap_lower": True,
        "max_incorrect_accept_increase": 0, "max_correct_accept_decrease": 0,
        "max_anchored_false_alarm_rate": 0.20, "max_severity4_miss_increase": 0}
STAGES = ("prepare-contract", "preflight", "train", "verify", "score")
APPROVED_REAL = Path("data/curation-side-20261127/real-approved.jsonl")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def write(path, value):
    serialized = (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    save_atomic(path, lambda stream: stream.write(serialized))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def eligible_rows(manifest, excluded_groups):
    if any(r["split"] != "dev" for r in manifest):
        raise ValueError("Expected canonical development manifest only")
    eligible = [r for r in manifest if r["dev_fold"] in (1, 2, 3) and r["group_id"] not in excluded_groups]
    protected = [r for r in manifest if r["dev_fold"] in (0, 4) or r["group_id"] in excluded_groups]
    if (len({r["sample_id"] for r in manifest}) != len(manifest)
            or {r["group_id"] for r in eligible} & {r["group_id"] for r in protected}
            or {r["sha256"] for r in eligible} & {r["sha256"] for r in protected}):
        raise ValueError("Duplicate ID, group or image fingerprint crossing training/protected boundary")
    return eligible


def verified_source_pairs(records):
    indexed = {}
    for i, record in enumerate(records):
        if record["code"] not in CODES or record["control"] not in ("other_hazard", "no_visible"):
            raise ValueError("Unexpected candidate-pair code or control")
        a, b = record["members"]
        if a["family"] != b["family"] or a["group_id"] == b["group_id"]:
            raise ValueError("Candidate contrast must be same-source across distinct groups")
        for row, decision in ((a, "accept"), (b, "reject")):
            key = (row["sample_id"], record["code"], record["control"], decision)
            if key in indexed:
                raise ValueError("Candidate contrast reuses the same photo/code/direction")
            indexed[key] = i
    return indexed


def validate_slots(slots, canonical, candidate_pairs):
    photo_ids = {"A": set(), "B": set()}
    image_hashes = {"A": set(), "B": set()}
    by_type = Counter()
    treatment_pairs = Counter()
    for slot in slots:
        if slot.get("treatment_review") != "approved":
            raise ValueError("Every B slot requires a separately approved treatment review")
        code, control, direction = (slot[k] for k in ("code", "control", "decision"))
        if code not in CODES or control not in ("other_hazard", "no_visible") or direction not in ("accept", "reject"):
            raise ValueError("Invalid slot code, control or direction")
        pair = []
        for arm in ("A", "B"):
            sid = slot[arm]
            row = canonical.get(sid)
            if row is None or sid in photo_ids[arm] or row["sha256"] in image_hashes[arm]:
                raise ValueError("Auxiliary slot is protected or duplicated within one arm")
            photo_ids[arm].add(sid)
            image_hashes[arm].add(row["sha256"])
            positive = code in row["codes"]
            if ((direction == "accept") != positive
                    or (direction == "reject" and bool(row["codes"]) != (control == "other_hazard"))):
                raise ValueError("Auxiliary slot direction disagrees with the official annotation")
            pair.append(row)
        if pair[0]["family"] != pair[1]["family"]:
            raise ValueError("A and B auxiliary rows must match official source family")
        if direction == "accept" and slot["A"] != slot["B"]:
            raise ValueError("A and B must share the same positive anchor")
        if direction == "reject" and slot["A"] == slot["B"]:
            raise ValueError("The compared negative must change between A and B")
        index = candidate_pairs.get((slot["B"], code, control, direction))
        if index is None:
            raise ValueError("Treatment is missing from the originally mined contrast pair")
        treatment_pairs[index, direction] += 1
        by_type[code, control, direction] += 1
    if not slots or any(treatment_pairs[i, "accept"] != 1 or treatment_pairs[i, "reject"] != 1
                        for i in {i for i, _ in treatment_pairs}):
        raise ValueError("Use one confirmed acceptance and rejection per approved treatment pair")
    approved_pairs = len(treatment_pairs) // 2
    return {"approved_pairs": approved_pairs, "pilot_only": approved_pairs < 4,
            "slot_counts": [{"code": c, "control": s, "decision": d, "examples_per_arm": n}
                            for (c, s, d), n in sorted(by_type.items())],
            "photos_per_arm": {arm: len(photos) for arm, photos in photo_ids.items()}}


def build_items(slots, canonical, raw, arm):
    from .vlm_11code import sanitize_text, verify_user_text

    items = []
    for i, slot in enumerate(slots):
        sid, code = slot[arm], slot["code"]
        target = raw[sid]["target"]
        if slot["decision"] == "accept":
            item = target["hazards"][code]
            level = item["risk_level"]
            answer = {"decision": "accept", "hazard": {
                "code": code, "likelihood_score": level["likelihood"]["score"], "severity_score": level["severity"]["score"],
                "evidence": sanitize_text(item["evidence"], 220, "불안전 상태가 확인된다."),
                "prevention": sanitize_text(item["prevention"], 420, "위험원에 대한 안전조치를 실시한다."),
            }}
        else:
            answer = {"decision": "reject", "reason": "사진에서 제시된 후보 위험의 직접적인 단서가 확인되지 않는다."}
        items.append({"sample_id": f"{sid}__aux_{i}", "image": canonical[sid]["image"], "kind": "verify_aux",
                      "augment": False, "user_text": verify_user_text(code),
                      "assistant_text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))})
    return items


def control_matches(pool, reviewed, reserved):
    """Deterministic complete matching; same group is preferred without using model scores."""
    options = []
    for item, negative in reviewed:
        code, control = item["code"], item["control"]
        matches = [r for r in pool if r["sample_id"] not in reserved and code not in r["codes"]
                   and r["family"] == negative["family"] and bool(r["codes"]) == (control == "other_hazard")]
        matches.sort(key=lambda r: (r["group_id"] != negative["group_id"],
                                    fingerprint(["official-control-v1", code, control, r["sample_id"]])))
        options.append([r["sample_id"] for r in matches])
    owners = {}
    assigned = {}

    def place(index, visited):
        for sid in options[index]:
            if sid in visited:
                continue
            visited.add(sid)
            if sid not in owners or place(owners[sid], visited):
                owners[sid], assigned[index] = index, sid
                return True
        return False

    for index in sorted(range(len(options)), key=lambda i: (len(options[i]), i)):
        if not place(index, set()):
            raise ValueError("No unique official resampling match for every approved B negative")
    return [assigned[i] for i in range(len(options))]


def read_approved(path):
    records = [json.loads(line, object_pairs_hook=unique_keys) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if any(row["approval"] != "approved" or row["positive_review"] != "visible" or row["negative_review"] != "absent"
           for row in records):
        raise ValueError("Every reviewed contrast must confirm both directions")
    return records


def prepare_contract(approved_path, reviewed_sha, workspace, tolerance):
    if (not approved_path.resolve().is_relative_to(Path("data").resolve())
            or digest(approved_path) != reviewed_sha):
        raise ValueError("Review-approved source must be local and match its independently supplied SHA-256")
    approved = read_approved(approved_path)
    if not 0 <= tolerance <= 1:
        raise ValueError("Specify a frozen relative token tolerance")
    if not approved:
        raise ValueError("No approved real contrast; distinct A/B sample arms cannot be constructed")
    source_pairs = read(SOURCE_PAIRS)
    candidate_index = verified_source_pairs(source_pairs)
    validate_mask(MANIFEST, EXCLUSIONS)
    excluded = {row["train_group_id"] for row in read(EXCLUSIONS)["confirmed_cross_group_links"]}
    pool = eligible_rows(read(MANIFEST), excluded)
    canonical = {row["sample_id"]: row for row in pool}
    visited = set()
    approved_with_negatives = []
    for row in approved:
        code, control = row["code"], row["control"]
        pos, neg = row["positive_sample_id"], row["negative_sample_id"]
        positive = candidate_index.get((pos, code, control, "accept"))
        negative = candidate_index.get((neg, code, control, "reject"))
        if positive is None or positive != negative or positive in visited or pos not in canonical or neg not in canonical:
            raise ValueError("Rejected, unconfirmed, changed or protected contrast entered the approval file")
        visited.add(positive)
        approved_with_negatives.append((row, canonical[neg]))
    reserved = {r["sample_id"] for pair in source_pairs for r in pair["members"]}
    controls = control_matches(pool, approved_with_negatives, reserved)
    slots = []
    for (approved_row, _), selected in zip(approved_with_negatives, controls, strict=True):
        pair = {"code": approved_row["code"], "control": approved_row["control"], "treatment_review": "approved"}
        positive = approved_row["positive_sample_id"]
        slots.append({**pair, "decision": "accept", "A": positive, "B": positive})
        slots.append({**pair, "decision": "reject", "A": selected, "B": approved_row["negative_sample_id"]})
    validate_slots(slots, canonical, candidate_index)
    accum = 1 if len(approved) < 4 else 4
    steps_per_epoch = math.ceil(len(slots) / accum)
    epochs = max(2, math.ceil(12 / steps_per_epoch))
    contract = {
        "version": "paired-official-verifier-v1", "authorization": "approved",
        "review_evidence_path": approved_path.as_posix(), "review_evidence_sha256": reviewed_sha,
        "input_sha256": {"manifest_sha256": digest(MANIFEST), "cross_group_mask_sha256": digest(EXCLUSIONS),
                         "source_pairs_sha256": digest(SOURCE_PAIRS), "frozen_evaluation_sha256": digest(PROBE)},
        "training": {"mode": "fresh_base_verifier_only", "seed": 42, "epochs": epochs, "grad_accum": accum,
                     "lr": 1.4e-4, "max_pixels": 384 * 32 * 32,
                     "max_relative_token_difference": {"supervised_tokens": tolerance, "decision_tokens": tolerance}},
        "slots": slots,
    }
    target = workspace / "contract-local.json"
    if target.exists():
        raise FileExistsError("The frozen contract already exists; use a new workspace for a new experiment")
    write(target, contract)
    print(json.dumps({"contract_sha256": digest(target), "approved_pairs": len(approved), "examples_per_arm": len(slots),
                      "optimizer_calls_per_arm": epochs * steps_per_epoch}, ensure_ascii=True))


def load_plan(contract_path, approved_sha):
    if not contract_path.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Contract must stay in local data/")
    contract = read(contract_path)
    if (digest(contract_path) != approved_sha or contract["version"] != "paired-official-verifier-v1"
            or contract["authorization"] != "approved"):
        raise ValueError("An explicitly approved, hash-pinned experiment contract is required")
    review_file = Path(contract["review_evidence_path"])
    if (not review_file.resolve().is_relative_to(Path("data").resolve())
            or digest(review_file) != contract["review_evidence_sha256"]):
        raise ValueError("Local approval evidence must match its frozen digest")
    inputs = {key: digest(path) for key, path in {
        "manifest_sha256": MANIFEST, "cross_group_mask_sha256": EXCLUSIONS,
        "source_pairs_sha256": SOURCE_PAIRS, "frozen_evaluation_sha256": PROBE,
    }.items()}
    if any(contract["input_sha256"].get(key) != sha for key, sha in inputs.items()):
        raise ValueError("A canonical, mining or frozen evaluation artifact changed")
    if read(PROBE_WITNESS)["proposal_file_sha256"] != inputs["frozen_evaluation_sha256"]:
        raise ValueError("Frozen proposals differ from the completed official-anchor run")
    validate_mask(MANIFEST, EXCLUSIONS)
    links = read(EXCLUSIONS)["confirmed_cross_group_links"]
    blocked = {pair["train_group_id"] for pair in links}
    eligible = eligible_rows(read(MANIFEST), blocked)
    canonical = {r["sample_id"]: r for r in eligible}
    slots = contract["slots"]
    composition = validate_slots(slots, canonical, verified_source_pairs(read(SOURCE_PAIRS)))
    approved = read_approved(review_file)
    approved_arms = {
        (row[sid_field], row["code"], row["control"], direction)
        for row in approved
        for sid_field, direction in (("positive_sample_id", "accept"), ("negative_sample_id", "reject"))
    }
    if len(approved_arms) != 2 * len(approved) or approved_arms != {
        (slot["B"], slot["code"], slot["control"], slot["decision"]) for slot in slots
    }:
        raise ValueError("The experimental arm differs from the exact review-approved contrasts")
    used = {slot[arm] for slot in slots for arm in ("A", "B")}
    raw = {r["sample_id"]: r for r in read_selected_cases(LABELS, used)}
    if raw.keys() != used:
        raise ValueError("Missing official annotation for a training row")
    for sid, r in raw.items():
        _, codes = decode_gold(r)
        if (set(canonical[sid]["codes"]) != codes or canonical[sid]["image"] != r["image"]
                or digest(Path("data/train") / r["image"]) != canonical[sid]["sha256"]):
            raise ValueError("Canonical training row, annotation and photo bytes differ")
    settings = contract["training"]
    if settings["mode"] != "fresh_base_verifier_only" or "parent_adapter" in settings:
        raise ValueError("This short experiment starts both arms independently from the official base")
    for key in ("epochs", "grad_accum", "max_pixels"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Positive integer expected for {key}")
    if type(settings["seed"]) is not int or not 0 < settings["lr"] < 1:
        raise ValueError("Invalid training seed or learning rate")
    if settings["epochs"] * math.ceil(len(slots) / settings["grad_accum"]) < 12:
        raise ValueError("Predeclare at least 12 optimizer calls to avoid a near-warmup-only comparison")
    tolerances = settings["max_relative_token_difference"]
    if any(not 0 <= tolerances[key] <= 1 for key in ("supervised_tokens", "decision_tokens")):
        raise ValueError("Specify frozen per-token-budget tolerances")
    candidates = read(PROBE)["proposals"]
    eval_rows = {r["sample_id"]: r for r in fold_rows(MANIFEST)}
    if (len(candidates) != 102 or len({item["sample_id"] for item in candidates}) != len(candidates)
            or any(item["sample_id"] not in eval_rows or item["image"] != eval_rows[item["sample_id"]]["image"]
                   or item["code"] not in CODES for item in candidates)
            or used & eval_rows.keys()):
        raise ValueError("Expected the same 102 unique frozen Fold 0 verifier candidates and no training overlap")
    return {
        "identity": {"contract_sha256": approved_sha, "prompt_sha256": digest(PROMPT), "official_labels_sha256": digest(LABELS),
                     "base_index_sha256": digest(BASE / "model.safetensors.index.json"),
                     "base_shards_identity_sha256": fingerprint([
                         [p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(BASE.glob("*.safetensors"))
                     ]),
                     "trainer_sha256": digest(Path(__file__).with_name("vlm_11code.py")),
                     "runner_sha256": digest(Path(__file__)), **inputs},
        "settings": settings, "composition": composition,
        "train_rows": {arm: [canonical[sid] for sid in sorted(slot[arm] for slot in slots)] for arm in ("A", "B")},
        "common_law_train_ids": set(canonical),
        "eligible_count": len(eligible), "blocked_group_count": len(blocked), "candidates": candidates,
        "arms": {arm: build_items(slots, canonical, raw, arm) for arm in ("A", "B")},
    }


def token_steps(items, settings, processor, system_prompt, image_root=Path("data/train")):
    from .vlm_11code import encode_weighted_example

    counts = {}
    for item in items:
        batch = encode_weighted_example(processor, item, image_root, system_prompt, random.Random(settings["seed"]))
        labels, weights = batch["labels"], batch["token_weights"]
        counts[item["sample_id"]] = (int((labels != -100).sum()), int((weights > 1.5).sum()),
                                     int(batch["image_grid_thw"].prod(dim=1).sum()) // processor.image_processor.merge_size**2)
    grid = []
    step_size = settings["grad_accum"]
    for epoch in range(1, settings["epochs"] + 1):
        ordered = items.copy()
        random.Random(settings["seed"] + epoch * 1000).shuffle(ordered)
        for start in range(0, len(ordered), step_size):
            block = ordered[start:start + step_size]
            counts_by_slot = [counts[item["sample_id"]] for item in block]
            supervised, decision, visual = map(sum, zip(*counts_by_slot))
            grid.append({"epoch": epoch, "examples": len(block), "supervised_tokens": supervised,
                         "decision_tokens": decision, "visual_tokens": visual})
    return grid


def preflight(plan, workspace, output):
    from .vlm_11code import load_qwen_processor

    processor = load_qwen_processor(BASE, plan["settings"]["max_pixels"])
    prompt = PROMPT.read_text(encoding="utf-8")
    grids = {arm: token_steps(items, plan["settings"], processor, prompt) for arm, items in plan["arms"].items()}
    if [(x["epoch"], x["examples"]) for x in grids["A"]] != [(x["epoch"], x["examples"]) for x in grids["B"]]:
        raise ValueError("The A/B optimizer-step grids differ")
    keys = ("examples", "supervised_tokens", "decision_tokens", "visual_tokens")
    totals = {arm: {key: sum(x[key] for x in grid) for key in keys} for arm, grid in grids.items()}
    relative = {key: abs(totals["B"][key] - totals["A"][key]) / max(totals["A"][key], totals["B"][key], 1) for key in keys}
    limits = plan["settings"]["max_relative_token_difference"]
    passed = all(relative[key] <= limits[key] for key in ("supervised_tokens", "decision_tokens"))
    paths = {arm: workspace / f"train-{arm}-local.json" for arm in ("A", "B")}
    for arm, path in paths.items():
        write(path, plan["train_rows"][arm])
    private = {"identity": plan["identity"], "settings": plan["settings"], "grids": grids, "passed": passed,
               "train_manifest_sha256": {arm: digest(path) for arm, path in paths.items()},
               "curricula_sha256": {arm: fingerprint(items) for arm, items in plan["arms"].items()}}
    write(workspace / "preflight-local.json", private)
    report = {"stage": "CPU-only token preflight", "identity": plan["identity"], "settings": plan["settings"],
              "eligible_pool": plan["eligible_count"], "excluded_train_groups": plan["blocked_group_count"],
              "frozen_evaluation_candidates": len(plan["candidates"]), "composition": plan["composition"],
              "totals": totals, "optimizer_calls_per_arm": len(grids["A"]), "examples_per_optimizer_call": [x["examples"] for x in grids["A"]],
              "relative_token_difference": relative, "tokens_within_frozen_tolerance": passed,
              "pre_registered_decision_rule": PLAN}
    write(output, report)
    if not passed:
        raise ValueError("A/B token budgets exceeded the predeclared tolerance; do not launch either arm")


def validated_preflight(plan, workspace):
    report = read(workspace / "preflight-local.json")
    if (not report["passed"] or report["identity"] != plan["identity"] or report["settings"] != plan["settings"]
            or report["train_manifest_sha256"] != {
                arm: digest(workspace / f"train-{arm}-local.json") for arm in ("A", "B")}
            or report["curricula_sha256"] != {arm: fingerprint(items) for arm, items in plan["arms"].items()}):
        raise ValueError("Both arms need a passing, hash-matched CPU preflight before GPU use")
    return report


def train(plan, workspace, arm, stopping, resume):
    validated_preflight(plan, workspace)
    from .vlm_11code import train_multimodal_qlora

    out = workspace / arm
    if resume is not None and resume.resolve().parent != out.resolve():
        raise ValueError("An arm may resume only its own independent checkpoint")
    settings = plan["settings"]

    def curriculum(rows, raw, seed):
        if seed != settings["seed"] or rows != plan["train_rows"][arm]:
            raise ValueError("Trainer saw a different approved experiment population")
        return plan["arms"][arm]

    initialization = out / "initialization-local.json"
    peer = workspace / ("B" if arm == "A" else "A") / "initialization-local.json"

    def observe_initial(value):
        report = {"initial_trainable_lora_sha256": value, "identity": plan["identity"]}
        if peer.is_file() and read(peer) != report:
            raise ValueError("Fresh LoRA initialization differs between the two independently seeded arms")
        if initialization.is_file() and read(initialization) != report:
            raise ValueError("Fresh LoRA initialization differs from the recorded arm initialization")
        write(initialization, report)

    if resume is not None and (not initialization.is_file() or read(initialization)["identity"] != plan["identity"]):
        raise ValueError("Resume requires its original fresh LoRA initialization fingerprint")
    train_multimodal_qlora(
        root=Path(".").resolve(), manifest_path=workspace / f"train-{arm}-local.json", prompt_path=PROMPT,
        out_dir=out, val_fold=0, epochs=settings["epochs"], grad_accum=settings["grad_accum"], lr=settings["lr"],
        max_pixels=settings["max_pixels"], seed=settings["seed"], checkpoint_steps=2,
        resume_from=resume.resolve() if resume is not None else None, stop_after_steps=stopping,
        curriculum_builder=curriculum, experiment_identity={"arm": arm, **plan["identity"]},
        initialization_observer=observe_initial,
    )


def validate_completed_arm(plan, workspace, arm):
    flight = validated_preflight(plan, workspace)
    directory = workspace / arm
    adapter = directory / "adapter"
    summary = read(directory / "training_summary.json")
    steps = [json.loads(line, object_pairs_hook=unique_keys)
             for line in (directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    grid = flight["grids"][arm]
    expected_fields = ("epoch", "examples", "supervised_tokens", "decision_tokens")
    initialization = read(directory / "initialization-local.json")
    peer = workspace / ("B" if arm == "A" else "A") / "initialization-local.json"
    if (initialization["identity"] != plan["identity"] or (peer.is_file() and read(peer) != initialization)
            or summary["configuration"]["experiment"] != {"arm": arm, **plan["identity"]}
            or digest(adapter / "adapter_model.safetensors") != summary["adapter_safetensors_sha256"]
            or summary["optimizer_steps"] != len(grid) or len(steps) != len(grid)
            or any(step["step"] != k or any(step[key] != expected[key] for key in expected_fields)
                   for k, (step, expected) in enumerate(zip(steps, grid, strict=True), 1))):
        raise ValueError("Completed arm must match its independent checkpoint, optimizer grid and measured token budget")
    return adapter, [(step["applied_lr"], step["lr"]) for step in steps]


def verification_identity(plan, adapter):
    from .vlm_11code import CANDIDATE_MAX_PIXELS, VERIFY_MAX_NEW_TOKENS

    return {**plan["identity"], "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": digest(adapter / "adapter_config.json"),
            "eval_max_pixels": CANDIDATE_MAX_PIXELS, "eval_max_new_tokens": VERIFY_MAX_NEW_TOKENS}


def verify(plan, workspace, arm):
    import torch

    from .official_anchored_verifier import verifier_model_key
    from .vlm_11code import (
        CANDIDATE_MAX_PIXELS, VERIFY_MAX_NEW_TOKENS, load_qwen_model,
        load_qwen_processor, verify_specialist_candidates_with_vlm,
    )

    adapter, _ = validate_completed_arm(plan, workspace, arm)
    if not torch.cuda.is_available():
        raise RuntimeError("Actual verification requires the reserved local CUDA run")
    rows = {r["sample_id"]: r for r in fold_rows(MANIFEST)}
    for item in plan["candidates"]:
        if digest(Path("data/train") / item["image"]) != rows[item["sample_id"]]["sha256"]:
            raise ValueError("Frozen candidate image no longer matches manifest")
    identity = {**verification_identity(plan, adapter), "device_model_key": verifier_model_key(BASE, adapter),
                "evaluation_runtime": {"device": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                                       "cuda": torch.version.cuda, "precision": "BF16",
                                       "transformers": importlib.metadata.version("transformers"),
                                       "peft": importlib.metadata.version("peft")}}
    model = proc = None

    def batch_run(batch):
        nonlocal model, proc
        if model is None:
            proc = load_qwen_processor(BASE, CANDIDATE_MAX_PIXELS)
            model = load_qwen_model(BASE, use_4bit=False, adapter_dir=adapter)
        return verify_specialist_candidates_with_vlm(
            model, proc, batch, Path("data/train"), PROMPT.read_text(encoding="utf-8"),
            batch_size=1, max_new_tokens=VERIFY_MAX_NEW_TOKENS,
        )

    verify_with_cache(plan["candidates"], batch_run, workspace / arm / "raw-verifier-local.json", identity, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--approved-sha256")
    parser.add_argument("--approved-real", type=Path, default=APPROVED_REAL)
    parser.add_argument("--approved-review-sha256")
    parser.add_argument("--token-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if not args.workspace.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Training and raw output must stay under data/")
    if args.stage == "prepare-contract":
        if not args.approved_review_sha256:
            parser.error("Supply the independently reviewed input SHA")
        prepare_contract(args.approved_real, args.approved_review_sha256, args.workspace, args.token_relative_tolerance)
        return
    if not (args.contract and args.approved_sha256):
        parser.error("Supply a frozen contract path and expected SHA")
    if args.stage in ("train", "verify") and not args.arm:
        parser.error("Choose independent A or B")
    if args.stage in ("preflight", "score") and not args.output:
        parser.error("Choose an aggregate output path")
    plan = load_plan(args.contract, args.approved_sha256)
    if args.stage == "preflight":
        preflight(plan, args.workspace, args.output)
    elif args.stage == "train":
        train(plan, args.workspace, args.arm, args.stop_after_steps, args.resume_from)
    elif args.stage == "verify":
        verify(plan, args.workspace, args.arm)
    else:
        from .paired_verifier_ab_score import score

        score(plan, args.workspace, args.output)


if __name__ == "__main__":
    main()
