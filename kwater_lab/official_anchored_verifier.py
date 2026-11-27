"""Freeze Fold0 specialist proposals for the official LoRA and verify them locally."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from functools import cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .audit import digest
from .fold0_group_sensitivity import EXCLUSIONS, MANIFEST, arm_metrics, validate_mask
from .fold0_production_compare import fold_rows, image_id
from .hybrid_pipeline import count_added_predictions, select_booster_candidates
from .law_reranker_11code import Native11LawRetriever, read_selected_cases
from .metrics import CODES, decode_target, paired_bootstrap, read_index
from .train_head import save_atomic
from .verifier_cache import verify_with_cache
from .vlm_11code import (CANDIDATE_MAX_PIXELS, VERIFY_MAX_NEW_TOKENS, format_competition_record, load_qwen_model,
                         load_qwen_processor, parse_verified_hazard, verify_specialist_candidates_with_vlm)


OFFICIAL = Path("data/fold0-qlora-v2-118-official")
OFFICIAL_REPORT = Path("results/fold0_qlora_v2_118_official.json")
VERIFIED_REPORT = Path("results/v2_official_anchored_verified.json")
OOF = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
LOCAL = Path("data/local-eval-predictions/fold0/v2-official-anchored")
FT_CACHE = Path("data/local-eval-predictions/fold0/fold0-qlora-v2-118-bf16/specialist_verification_cache.json")
ADAPTER = Path("data/vlm-11code-fold0-v2/adapter")
PROMPT = Path("kwater_lab/system_prompt_11code.txt")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(path, lambda f: f.write((json.dumps(obj, indent=2, ensure_ascii=True) + "\n").encode("utf-8")))


def official_records(rows):
    output = OFFICIAL / "output/submission.jsonl"
    report = json.loads(OFFICIAL_REPORT.read_text(encoding="utf-8"))
    if report["official_checker_valid_rows"] != len(rows) or report["output_sha256"] != digest(output):
        raise ValueError("Official output does not match the complete checker receipt")
    index = read_index(output, lambda row: (decode_target(row)[0], row), limit_size=True)
    expected = {image_id(row["sample_id"]): row["sample_id"] for row in rows}
    raw = [row for path in sorted((OFFICIAL / "raw-batches").glob("[0-9]*.json"))
           for row in json.loads(path.read_text(encoding="utf-8"))]
    if list(index) != list(expected) or [r["sample_id"] for r in raw] != list(expected):
        raise ValueError("Official formatted/raw coverage or order differs from the frozen cohort")
    return {expected[key]: record for key, record in index.items()}


def freeze_proposals(rows, baseline):
    with np.load(OOF, allow_pickle=False) as archive:
        indices = {key: i for i, key in enumerate(archive["ids"].tolist())}
        selection = np.array([indices[r["sample_id"]] for r in rows])
        if len(indices) != len(archive["ids"]) or not np.all(archive["folds"][selection] == 0):
            raise ValueError("Expected original Fold0 out-of-fold specialist scores")
        code_scores = archive["code_probs"][selection]
        scene_scores = archive["scene_probs"][selection]
    inputs = [{"sample_id": r["sample_id"], "image": r["image"]} for r in rows]
    drafts = [{"parsed": {"status": baseline[r["sample_id"]]["status"], "hazards": [
        {"code": code} for code in baseline[r["sample_id"]]["hazards"]]}} for r in inputs]
    calibration = {"temperatures": [1.0] * len(CODES), "thresholds": [0.5] * len(CODES), "scene_threshold": 0.5}
    return select_booster_candidates(inputs, drafts, code_scores, scene_scores, calibration, max_candidates_per_photo=1)


def aggregate(rows, baseline, proposal, verified=None):
    excluded, _, _ = validate_mask(MANIFEST, EXCLUSIONS)
    gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    severity = {r["sample_id"]: {code for code, item in r["target"]["hazards"].items()
                               if item["risk_level"]["severity"]["score"] == 4}
                for r in read_selected_cases(Path("data/train/train.jsonl"), set(gold))}
    before = {sid: frozenset(r["hazards"]) for sid, r in baseline.items()}
    proposals = {item["sample_id"]: item["code"] for item in proposal}
    result = {}
    for name, cohort in (("full_fold0", rows), ("confirmed_group_exclusion", [r for r in rows if r["group_id"] not in excluded])):
        groups = {r["sample_id"]: r["group_id"] for r in cohort}
        truth = {sid: gold[sid] for sid in groups}
        original = {sid: before[sid] for sid in groups}
        additions = {sid: {proposals[sid]} if sid in proposals else set() for sid in groups}
        oracle = {sid: original[sid] | (additions[sid] & truth[sid]) for sid in groups}
        all_added = {sid: original[sid] | additions[sid] for sid in groups}
        report = {"photos": len(groups), "groups": len(set(groups.values())),
                  "proposals": sum(bool(codes) for codes in additions.values()),
                  "proposal_codes": dict(Counter(code for codes in additions.values() for code in codes)),
                  "proposals_on_safe_photos": sum(bool(additions[sid]) for sid in groups if not truth[sid]),
                  "proposed_annotated_codes": sum(len(additions[sid] & truth[sid]) for sid in groups),
                  "proposed_unannotated_codes": sum(len(additions[sid] - truth[sid]) for sid in groups),
                  "official": arm_metrics(truth, original, groups, severity),
                  "unguarded_all_proposed_codes_added": arm_metrics(truth, all_added, groups, severity),
                  "label_oracle_upper_bound": arm_metrics(truth, oracle, groups, severity)}
        if verified is not None:
            after = {sid: frozenset(verified[sid]["hazards"]) for sid in groups}
            report.update(verified=arm_metrics(truth, after, groups, severity),
                          verified_delta=count_added_predictions(truth, original, after),
                          verified_vs_official=paired_bootstrap(truth, original, after, groups, resamples=2000, seed=42))
        result[name] = report
    return result


def current_fingerprint():
    return {"manifest_sha256": digest(MANIFEST), "group_mask_sha256": digest(EXCLUSIONS),
            "official_predictions_sha256": digest(OFFICIAL / "output/submission.jsonl"), "oof_scores_sha256": digest(OOF),
            "selector_source_sha256": digest(Path(__file__).with_name("hybrid_pipeline.py"))}


def verifier_model_key(base, adapter):
    """Use the same model-key protocol as the completed BF16 Fold0 verifier."""
    import torch

    model_files = {}
    for path in sorted(base.iterdir()):
        if path.is_file():
            if path.suffix == ".safetensors":
                stats = path.stat()
                model_files[path.name] = [stats.st_size, stats.st_mtime_ns]
            else:
                model_files[path.name] = digest(path)
    identity = {"base": model_files, "device": torch.cuda.get_device_name(torch.device("cuda:0")),
                "packages": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "peft")},
                "torch_cuda_version": torch.version.cuda,
                "inference_source_sha256": digest(Path(__file__).with_name("vlm_11code.py")), "use_4bit": False,
                "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": digest(adapter / "adapter_config.json")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def reusable_raw(ft_cache, identity, proposals, selected_images):
    fingerprint = ft_cache["fingerprint"]
    if fingerprint["batch_size"] != 1 or any(fingerprint["inputs"].get(key) != value for key, value in identity.items()):
        raise ValueError("Existing FT verifier was generated with a different precision, prompt, model, processor, or token cap")
    frozen_rows = fingerprint["candidates"]
    raw_rows = ft_cache["results"]
    if len(raw_rows) != len(frozen_rows) or any(
        (r["sample_id"], r["code"], r["image"]) != (c["sample_id"], c["code"], c["image"])
        for r, c in zip(raw_rows, frozen_rows, strict=True)
    ):
        raise ValueError("Wait until the current FT verifier has completed its frozen list")
    indexed = {(r["sample_id"], r["code"], r["image"]): r["raw_verify_text"] for r in raw_rows}
    image_sha = fingerprint["inputs"]["selected_images_sha256"]
    matches = {}
    for item in proposals:
        key = (item["sample_id"], item["code"], item["image"])
        if key in indexed and image_sha.get(item["image"]) == selected_images[item["image"]]:
            matches[key] = indexed[key]
    return matches


def add_verified_records(baseline, results, *, law_train_ids=None):
    output = copy.deepcopy(baseline)
    accepted = [result for result in results if result["accepted"]]
    if accepted:
        train_ids = law_train_ids
        if train_ids is None:
            dev = json.loads(MANIFEST.read_text(encoding="utf-8"))
            train_ids = {r["sample_id"] for r in dev if r["split"] == "dev" and r["dev_fold"] != 0}
        laws = [json.loads(line) for line in Path("data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if line]
        ranker = Native11LawRetriever(laws, list(read_selected_cases(Path("data/train/train.jsonl"), train_ids)))
        for result in accepted:
            row = output[result["sample_id"]]
            code, hazard = result["code"], result["hazard"]
            if len(row["hazards"]) >= 3 or code in row["hazards"]:
                raise ValueError("Anchored verifier may append one new risk in an open slot only")
            addition = format_competition_record(row["image_id"], {"status": "hazard", "scene_note": hazard["evidence"],
                                                                  "hazards": [hazard]}, ranker)
            row["hazards"][code] = addition["hazards"][code]
            row["status"] = "hazard"
            row.pop("scene_evidence", None)
    changed = {r["sample_id"] for r in accepted}
    if len(changed) != len(accepted):
        raise ValueError("More than one verifier proposal was accepted for a photo")
    for sid, row in output.items():
        decode_target(row)
        if sid not in changed and row != baseline[sid]:
            raise ValueError("A row without an accepted addition changed")
        for code, old in baseline[sid]["hazards"].items():
            if row["hazards"].get(code) != old:
                raise ValueError("Official hazard contents were changed")
    return output


def run_verified(rows, baseline, proposals, frozen_path, ft_cache_path):
    import torch

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Use the reserved CUDA workflow for actual verification")
    base = Path("data/baseline/base_model")
    proc = load_qwen_processor(base, CANDIDATE_MAX_PIXELS)
    system_prompt = PROMPT.read_text(encoding="utf-8")
    shared = {"model_key": verifier_model_key(base, ADAPTER), "prompt_sha256": digest(PROMPT),
              "processor_size": dict(proc.image_processor.size), "max_new_tokens": VERIFY_MAX_NEW_TOKENS}
    ft_cache = json.loads(ft_cache_path.read_text(encoding="utf-8"))
    source_images = {r["image"]: r["sha256"] for r in rows}
    selected_images = {item["image"]: digest(Path("data/train") / item["image"]) for item in proposals}
    if any(source_images.get(name) != sha for name, sha in selected_images.items()):
        raise ValueError("Verification photo bytes differ from the canonical development manifest")
    matches = reusable_raw(ft_cache, shared, proposals, selected_images)
    identity = {**shared, "frozen_proposals_sha256": digest(frozen_path), "ft_verifier_cache_sha256": digest(ft_cache_path),
                "selected_images_sha256": selected_images, "anchored_source_sha256": digest(Path(__file__))}

    @cache
    def model():
        return load_qwen_model(base, use_4bit=False, adapter_dir=ADAPTER)

    def verify_batch(batch):
        item, = batch
        key = (item["sample_id"], item["code"], item["image"])
        if key in matches:
            raw = matches[key]
            hazard = parse_verified_hazard(raw, item["code"])
            return [{**item, "raw_verify_text": raw, "hazard": hazard, "accepted": hazard is not None, "source": "exact_ft_cache"}]
        begin = time.perf_counter()
        row, = verify_specialist_candidates_with_vlm(model(), proc, batch, Path("data/train"), system_prompt, batch_size=1)
        return [{**row, "source": "new_vlm", "new_generation_seconds_including_load": time.perf_counter() - begin}]

    verified, timings = verify_with_cache(proposals, verify_batch, LOCAL / "anchored_verifier_cache.json", identity, 1)
    model.cache_clear()
    torch.cuda.empty_cache()
    formatted = add_verified_records(baseline, verified)
    output = LOCAL / "output/submission.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(output, lambda f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in formatted.values()).encode("utf-8")))
    check_official_output(output, LOCAL / "checker.log")
    metrics = aggregate(rows, baseline, proposals, formatted)
    return {"phase": "actual BF16 verification of the frozen official-anchored proposals", "official_checker_valid_rows": 216,
            "output_sha256": digest(output), "proposal_file_sha256": digest(frozen_path),
            "accepted_proposals": sum(bool(row["accepted"]) for row in verified),
            "existing_official_hazard_records_preserved": sum(len(r["hazards"]) for r in baseline.values()),
            "official_rows_unchanged": sum(row == baseline[sid] for sid, row in formatted.items()),
            "official_negative_scenes_converted": sum(baseline[sid]["status"] == "no_visible_hazard" and row["status"] == "hazard"
                                                       for sid, row in formatted.items()),
            "exact_ft_cache_reuses": len(matches), "new_model_generations": len(verified) - len(matches),
            "new_generation_seconds_including_load": round(sum(r.get("new_generation_seconds_including_load", 0) for r in verified), 3),
            "current_invocation_seconds": round(time.perf_counter() - started, 3), **timings,
            "metrics": metrics}


def check_official_output(output, log):
    check = subprocess.run([sys.executable, "-X", "utf8", "data/local-kit/kit/check_output.py", "data/fold0-production-input",
                            str(output.parent)], capture_output=True, encoding="utf-8", check=True)
    log.write_text(check.stdout + check.stderr, encoding="utf-8")
    if "유효 결과 216/216" not in check.stdout:
        raise ValueError("Official output checker did not accept all 216 rows")


def preserve_official_negative(baseline, verified, accepted):
    if set(baseline) != set(verified) or not accepted.keys() <= baseline.keys():
        raise ValueError("Retrospective replay requires the same verified cohort")
    result = {}
    for sid, original in baseline.items():
        after = verified[sid]
        if sid not in accepted:
            if after != original:
                raise ValueError("An unaccepted official row changed")
        else:
            added = set(after["hazards"]) - set(original["hazards"])
            if added != {accepted[sid]} or any(after["hazards"].get(c) != h for c, h in original["hazards"].items()):
                raise ValueError("Verified row does not preserve official hazards and the recorded addition")
        result[sid] = after if original["status"] == "hazard" and 1 <= len(original["hazards"]) <= 2 else original
    return result


def replay_preserving_negative(rows, baseline, frozen_path):
    started = time.perf_counter()
    witness = json.loads(VERIFIED_REPORT.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    cache_path = LOCAL / "anchored_verifier_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    verified_path = LOCAL / "output/submission.jsonl"
    baseline_path = OFFICIAL / "output/submission.jsonl"
    proposal_digest = digest(frozen_path)
    proposals = frozen["proposals"]
    if (frozen["fingerprint"] != current_fingerprint() or witness["proposal_file_sha256"] != proposal_digest
        or cache["fingerprint"]["inputs"]["frozen_proposals_sha256"] != proposal_digest
        or cache["fingerprint"]["candidates"] != proposals or witness["official_checker_valid_rows"] != len(rows)
        or witness["output_sha256"] != digest(verified_path)):
        raise ValueError("Frozen and checked source artifacts do not match")
    if [(r["sample_id"], r["code"]) for r in cache["results"]] != [(r["sample_id"], r["code"]) for r in proposals]:
        raise ValueError("Original ordered verifier responses are incomplete")
    accepted = {r["sample_id"]: r["code"] for r in cache["results"] if r["accepted"]}
    if len(accepted) != witness["accepted_proposals"]:
        raise ValueError("Cached acceptances disagree with the completed run")
    ids = {r["image_id"]: sid for sid, r in baseline.items()}
    verified = {ids[i]: row for i, row in read_index(verified_path, lambda r: (decode_target(r)[0], r), limit_size=True).items()}
    result = preserve_official_negative(baseline, verified, accepted)
    eligible = [item for item in proposals if baseline[item["sample_id"]]["status"] == "hazard"
                and 1 <= len(baseline[item["sample_id"]]["hazards"]) <= 2]
    # Copy whichever original line applies. This preserves no-visible-hazard JSONL bytes as well as parsed text.
    def lines(path):
        return {json.loads(line)["image_id"]: line for line in path.read_bytes().splitlines(keepends=True)}

    original_lines, verified_lines = lines(baseline_path), lines(verified_path)
    output = LOCAL / "preserve-official-negative/output/submission.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    changed = {sid for sid in result if result[sid] != baseline[sid]}
    save_atomic(output, lambda stream: stream.write(b"".join(
        (verified_lines if sid in changed else original_lines)[row["image_id"]] for sid, row in baseline.items()
    )))
    check_official_output(output, output.parent.parent / "checker.log")
    metrics = aggregate(rows, baseline, eligible, result)
    return {"phase": "post-hoc CPU replay on the already observed Fold0; independent fold required before promotion",
            "rule": "Only append an already accepted proposal where official output had one or two risks; preserve all official negative rows verbatim",
            "official_checker_valid_rows": len(rows), "output_sha256": digest(output), "source_verified_output_sha256": digest(verified_path),
            "source_proposal_sha256": proposal_digest, "source_accepted_proposals": len(accepted), "retained_acceptances": len(changed),
            "removed_acceptances": len(accepted) - len(changed),
            "official_negative_rows_preserved_verbatim": sum(r["status"] == "no_visible_hazard" for r in baseline.values()),
            "cpu_replay_new_verifier_calls": 0, "fresh_deployment_verifier_calls_on_this_cohort": len(eligible),
            "cpu_elapsed_seconds": round(time.perf_counter() - started, 3), "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--preserve-official-negative", action="store_true")
    parser.add_argument("--ft-verifier-cache", type=Path, default=FT_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = fold_rows(MANIFEST)
    baseline = official_records(rows)
    path = LOCAL / "frozen_proposals.json"
    if args.preserve_official_negative:
        write_json(args.out, replay_preserving_negative(rows, baseline, path))
        return
    receipt = {"fingerprint": current_fingerprint(), "proposals": freeze_proposals(rows, baseline)}
    if args.verify:
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("The frozen proposal plan or its inputs have changed")
        write_json(args.out, run_verified(rows, baseline, receipt["proposals"], path, args.ft_verifier_cache))
        return
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Frozen proposal population has changed; keep the original witness")
    else:
        write_json(path, receipt)
    write_json(args.out, {"phase": "frozen proposals and label oracle; no verifier generated",
                          "proposal_file_sha256": digest(path), "cpu_seconds": round(time.perf_counter() - started, 3),
                          "metrics": aggregate(rows, baseline, receipt["proposals"])})


if __name__ == "__main__":
    main()
