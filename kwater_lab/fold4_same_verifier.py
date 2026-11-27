"""Compare frozen Fold 4 drafts using a single frozen epoch-2 verifier."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path

from .metrics import CODES, decode_target, read_index, unique_keys

MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
BENCH = Path("data/local-eval-predictions/fold4-v3-benchmark")
WORK = Path("data/local-eval-predictions/fold4-same-verifier")
TRAIN = Path("data/train")
RESULT = Path("results/fold4_v3_clean_benchmark.json")
EPOCHS = ("epoch2", "epoch1")
BATCH = 2
PIXELS = 393216
GEN_TOKENS = 280


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    # Stream adapter and data checks to avoid a large resident bytes object.
    with path.open("rb") as stream:
        hasher = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        return hasher.hexdigest()


def save(path, payload):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def candidate_key(item):
    return item["sample_id"] + "|" + item["code"]


def prioritized_union(first, second):
    """Keep original epoch-2 batches intact and append only novel epoch-1 items."""
    seen = {candidate_key(item) for item in first}
    return [first, [item for item in second if candidate_key(item) not in seen]]


def passes(item, margin, calibration):
    sp = min(max(float(item["spec_cal_prob"]), 1e-4), 1 - 1e-4)
    joint = margin + float(calibration["beta"]) * math.log(sp / (1 - sp))
    threshold = calibration["tau_when_empty" if item["was_vlm_empty"] else "tau_when_hazard"]
    return joint >= float(threshold)


def plan(work, v3_dir, prompt):
    rows = [r for r in read(MANIFEST) if r["split"] == "dev" and r["dev_fold"] == 4]
    if len(rows) != 175 or len({r["sample_id"] for r in rows}) != 175:
        raise ValueError("Expected 175 frozen Fold 4 rows")
    path = work / "proposals.json"
    bundle = read(path)
    proposals = {e: bundle[e] for e in EPOCHS}
    image_hashes = {}
    for candidates in proposals.values():
        if len({c["row_idx"] for c in candidates}) != len(candidates):
            raise ValueError("Expected at most one candidate per photo in each arm")
        for c in candidates:
            i = c["row_idx"]
            if type(i) is not int or not 0 <= i < len(rows):
                raise ValueError("Candidate row index outside frozen cohort")
            row = rows[i]
            if (c["sample_id"], c["image"]) != (row["sample_id"], row["image"]) or c["code"] not in CODES:
                raise ValueError("Candidate outside canonical photograph/code set")
            if type(c["was_vlm_empty"]) is not bool or not 0 <= float(c["spec_cal_prob"]) <= 1:
                raise ValueError("Invalid original proposal context")
            image_path = TRAIN / c["image"]
            if not image_path.resolve().is_relative_to(TRAIN.resolve()):
                raise ValueError("Candidate image outside local training directory")
            if c["image"] not in image_hashes:
                image_hashes[c["image"]] = digest(image_path)
            if image_hashes[c["image"]] != row["sha256"]:
                raise ValueError("Candidate image changed")
    benchmark = read(RESULT)
    original = benchmark["arms"]["v3_hybrid_1token_margin_verified"]
    original_gate = original["margin_gate"]
    gate_path = v3_dir / "margin_gate_calibration.json"
    gate = read(gate_path)
    adapter = v3_dir / "adapter/adapter_model.safetensors"
    adapter_sha = digest(adapter)
    prompt_sha = digest(prompt)
    raw_cache_key = read(BENCH / "v3_epoch2_infer_cache.json")["identity"]["model_key"]
    if (gate != original_gate or adapter_sha != benchmark["v3_adapter_sha256"]
            or raw_cache_key != f"bf16:{adapter_sha}:{prompt_sha}:px{PIXELS}"):
        raise ValueError("Verifier gate, adapter or prompt differs from epoch-2 benchmark")
    if len(proposals["epoch2"]) != original["raw_specialist_proposals"]:
        raise ValueError("Epoch-2 candidate count differs from the original benchmark")
    mark = {"proposal_sha256": digest(path), "manifest_sha256": digest(MANIFEST),
            "gate_sha256": digest(gate_path), "adapter_sha256": adapter_sha,
            "adapter_config_sha256": digest(adapter.with_name("adapter_config.json")),
            "prompt_sha256": prompt_sha, "recipe": ["BF16", PIXELS, BATCH, GEN_TOKENS]}
    return rows, proposals, gate, mark


def apply_item(output, hazard):
    if hazard is None:
        return False
    current = output.get("parsed")
    if not isinstance(current, dict):
        current = {"scene_note": hazard["evidence"], "status": "hazard", "hazards": []}
    hazards = current.get("hazards")
    hazards = hazards if isinstance(hazards, list) else []
    if hazard["code"] in {item.get("code") for item in hazards if isinstance(item, dict)} or len(hazards) >= 3:
        return False
    if not hazards or current.get("status") != "hazard":
        current["scene_note"] = hazard["evidence"]
    current.update(status="hazard", hazards=[*hazards, hazard])
    output["parsed"] = current
    return True


def render(work, rows, proposals, gate, decisions):
    from .law_reranker_11code import read_selected_cases
    from .v3_turnaround import ActionSupportedLawRetriever, format_v3_record
    from .vlm_11code import parse_verified_hazard

    train = [r["sample_id"] for r in read(MANIFEST) if r["split"] == "dev" and r["dev_fold"] != 4]
    by_id = {r["sample_id"]: r for r in read_selected_cases(TRAIN / "train.jsonl", set(train))}
    laws = [json.loads(line) for line in Path("data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if line]
    retriever = ActionSupportedLawRetriever(laws, [by_id[sid] for sid in train])
    ids = [r["sample_id"] for r in rows]
    report, output_by_epoch = {}, {}
    for epoch in EPOCHS:
        raw = read_index(BENCH / f"v3_{epoch}_raw_175.jsonl", lambda r: (r["sample_id"], r))
        standalone_name = "v3_11code_standalone" if epoch == "epoch2" else "v3_epoch1_11code_standalone"
        standalone = read_index(BENCH / f"{standalone_name}.jsonl", lambda r: (r["image_id"], r))
        if list(raw) != ids or list(standalone) != ids:
            raise ValueError("Frozen raw/formatted prediction IDs differ from cohort")
        accepted, passed = set(), 0
        merged = deepcopy(raw)
        for item in proposals[epoch]:
            key = candidate_key(item)
            if passes(item, decisions["margins"][key], gate):
                passed += 1
                hz = parse_verified_hazard(decisions["generations"][key], item["code"])
                if apply_item(merged[item["sample_id"]], hz):
                    accepted.add(item["sample_id"])
        output = []
        for sid in ids:
            record = format_v3_record(sid, merged[sid]["parsed"], retriever) if sid in accepted else standalone[sid]
            decode_target(record, local_ids=True)
            output.append(record)
        output_by_epoch[epoch] = output
        report[epoch] = {"proposed": len(proposals[epoch]), "passed": passed, "accepted": len(accepted)}
    prior = read_index(BENCH / "v3_hybrid_margin_verified.jsonl", lambda r: (r["image_id"], r))
    reproduced = sum(prior.get(r["image_id"]) == r for r in output_by_epoch["epoch2"])
    original = read(RESULT)["arms"]["v3_hybrid_1token_margin_verified"]
    if (len(prior) != len(rows) or reproduced != len(rows)
            or report["epoch2"]["passed"] != original["passed_1token_margin_gate"]
            or report["epoch2"]["accepted"] != original["final_accepted_additions"]):
        raise ValueError(f"Epoch-2 hybrid replay matched {reproduced}/{len(rows)} prior rows. No paired output published.")
    report["epoch2_exact_prior_hybrid_rows"] = reproduced
    for epoch, output in output_by_epoch.items():
        path = work / f"{epoch}_fixed_epoch2_verifier.jsonl"
        temp = path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
        os.replace(temp, path)
        report[epoch]["output_sha256"] = digest(path)
    save(work / "completed.json", report)
    return report


def verify(work, v3_dir, prompt, proposals, gate, mark):
    from importlib.metadata import version
    from .v3_turnaround import _resolve_base_model_dir, load_qwen_4bit_or_bf16, score_verifier_1token_margins
    from .vlm_11code import verify_specialist_candidates_with_vlm

    base = _resolve_base_model_dir(Path.cwd())
    module_files = (Path(__file__), Path(__file__).with_name("v3_turnaround.py"), Path(__file__).with_name("vlm_11code.py"))
    mark = {**mark, "source": {p.name: digest(p) for p in module_files},
            "runtime": {pkg: version(pkg) for pkg in ("torch", "transformers", "peft")},
            "base_files": {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(base.iterdir()) if p.is_file()}}
    cache = work / "fixed_verifier_cache.json"
    decisions = read(cache) if cache.is_file() else {"identity": mark, "margins": {}, "generations": {}}
    if decisions["identity"] != mark:
        raise ValueError("Cannot reuse cache with a changed verifier fingerprint")
    model, proc = None, None
    sys_prompt = prompt.read_text(encoding="utf-8")
    # Keep original two-photo batches for epoch 2. Add only novel items after those batches.
    def complete(groups, field, execute):
        nonlocal model, proc
        values = decisions[field]
        for group in groups:
            for i in range(0, len(group), BATCH):
                items = group[i:i+BATCH]
                keys = [candidate_key(item) for item in items]
                ready = [key in values for key in keys]
                if all(ready):
                    continue
                if any(ready):
                    raise ValueError(f"Incomplete atomic {field} batch")
                if model is None:
                    model, proc = load_qwen_4bit_or_bf16(base, max_pixels=PIXELS, use_4bit=False, adapter_dir=v3_dir / "adapter")
                measured = execute(items)
                values.update(zip(keys, measured, strict=True))
                save(cache, decisions)
                print(f"{field}: {len(values)}", flush=True)

    complete(prioritized_union(*(proposals[e] for e in EPOCHS)), "margins",
             lambda items: score_verifier_1token_margins(model, proc, items, TRAIN, sys_prompt, batch_size=BATCH))
    passed = [[c for c in proposals[e] if passes(c, decisions["margins"][candidate_key(c)], gate)] for e in EPOCHS]
    complete(prioritized_union(*passed), "generations",
             lambda items: [r["raw_verify_text"] for r in verify_specialist_candidates_with_vlm(
                 model, proc, items, TRAIN, sys_prompt, batch_size=BATCH, max_new_tokens=GEN_TOKENS)])
    save(cache, decisions)
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "verify"))
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--verifier-dir", type=Path, default=Path("data/v3-11code-fold4"))
    parser.add_argument("--prompt", type=Path, default=Path("kwater_lab/system_prompt_v3.txt"))
    args = parser.parse_args()
    if not args.work.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Work and outputs must stay in repository data/")
    rows, proposals, gate, mark = plan(args.work, args.verifier_dir, args.prompt)
    groups = prioritized_union(*(proposals[e] for e in EPOCHS))
    print(json.dumps({"proposal_counts": {e: len(proposals[e]) for e in EPOCHS},
                      "unique_inference_pairs": sum(map(len, groups)), "fingerprint": mark}), flush=True)
    if args.stage == "verify":
        decisions = verify(args.work, args.verifier_dir, args.prompt, proposals, gate, mark)
        print(json.dumps(render(args.work, rows, proposals, gate, decisions)), flush=True)


if __name__ == "__main__":
    main()
