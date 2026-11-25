"""Compare five full-image reading conditions on label-blind, previously exposed K-water anchors."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .adapter_scale_pilot import write_json
from .encoder_pilot import scores
from .law_examples import law_ids, select_articles, training_examples
from .metrics import CODES, evaluate, unique_keys
from .prefix_pilot import filtered_training, policy_decisions
from .prototype import RESPONSE_SCHEMA, assemble
from .train_head import file_hash
from .visual_probe import CODEBOOK, DEFAULT_CACHE, FILES, MODEL_REV, RUNTIME, GenerationRequestError, generate, local_server
from .weight_pilot import pilot_data


RARE = {"drowning_inundation", "confined_space", "thermal_weather"}
ARMS = ("direct_8", "direct_11", "facts_11", "verify_11", "swapped_11")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def instruction(arm, candidates):
    codes = [c for c in CODES if arm != "direct_8" or c not in RARE]
    common = ("현장 전체 사진 한 장에서 보이는 위험만 0~3개 판정하세요. 방호된 상태와 노출된 위험을 구별하세요. "
              "한국어로 scene_observation에 사진 상태, evidence에 관측 근거, prevention에 실행 가능한 조치를 쓰세요. "
              "likelihood는 1~5, severity는 1~4인 정수입니다. 코드 목록은 다음과 같습니다.\n" +
              "\n".join(f"{c} = {CODEBOOK[c]}" for c in codes))
    if arm.startswith("direct"):
        action = "hazards에 위험 유형을 먼저 판정하여 근거·조치를 쓰고 마지막에 scene_observation에 장면을 요약하세요."
    else:
        action = "대상·위치·방호 상태를 관찰해 먼저 scene_observation에 적으세요. 그 사실로 설명되는 위험과 조치를 판정하세요."
    if arm in ("verify_11", "swapped_11"):
        action += " 확인할 후보는 " + (", ".join(candidates) or "없음") + "입니다. 사진에서 독립적으로 지지되는 경우만 채택하고 그 밖의 관측 위험도 평가하세요."
    schema = deepcopy(RESPONSE_SCHEMA)
    if arm.startswith("direct"):
        schema["required"] = ["hazards", "scene_observation"]
        schema["properties"] = {key: schema["properties"][key] for key in schema["required"]}
    schema["properties"]["hazards"]["items"]["properties"]["code"]["enum"] = codes
    return common + "\n" + action, schema


def citation_counts(predictions, labels, first_article):
    counts = Counter(predicted=0, correct_code=0, cited=0, joint=0, false_risk_cited=0)
    for sid, predicted in predictions.items():
        for code in predicted:
            counts["predicted"] += 1
            correct = code in labels[sid]
            counts["correct_code"] += correct
            article = first_article.get(code)
            if article is not None:
                counts["cited"] += 1
                counts["false_risk_cited"] += not correct
                counts["joint"] += correct and article in law_ids(labels[sid][code])
    return counts


def parse_generation(arm, sid, result):
    if result["finish_reason"] != "stop":
        return None, None
    try:
        content = json.loads(result["raw_output"], object_pairs_hook=unique_keys)
        row = assemble(digest(sid)[:32], content)
        if arm == "direct_8" and set(row["hazards"]) & RARE:
            return None, None
        return content, row
    except (KeyError, ValueError, TypeError):
        return None, None


def paired_changes(measured, gold):
    counts = {}
    for left, right in (("direct_8", "direct_11"), ("direct_11", "facts_11"),
                        ("facts_11", "verify_11"), ("verify_11", "swapped_11")):
        pairs = [(measured[left][sid][0], measured[right][sid][0], target)
                 for sid, target in gold.items()
                 if measured[left][sid][0] is not None and measured[right][sid][0] is not None]
        counts[f"{left}/{right}"] = {
            "pairwise_valid": len(pairs),
            "code_set_changed": sum(first != second for first, second, _ in pairs),
            "exact_set_gains": sum(first != target and second == target for first, second, target in pairs),
            "exact_set_losses": sum(first == target and second != target for first, second, target in pairs),
        }
    return counts


def prepare(root, local, context_size=8192):
    ref = json.loads((root / "results/prefix_pilot_fpr20.json").read_text(encoding="utf-8"))
    enc = json.loads((root / "results/encoder_pilot.json").read_text(encoding="utf-8"))
    audit = root / "results/train_audit_full.json"
    exposure = root / "data/prefix-pilot/exposure.json"
    buckets, clip, gold, linked = pilot_data(audit, exposure, root/"data/prefix-pilot/features.npz", ref)
    cache = root / "data/prefix-pilot/siglip2-base16-features.npz"
    if file_hash(cache) != enc["cache_sha256"] or file_hash(exposure) != enc["provenance"]["exposure_sha256"]:
        raise ValueError("Encoder caches or cohort differ from the recorded comparison")
    with np.load(cache, allow_pickle=False) as a:
        sig = dict(zip(a["ids"].tolist(), a["features"], strict=True))
    inventory = json.loads(audit.read_text(encoding="utf-8"))["inventory"]
    labels_path = root / "data/train/train.jsonl"
    if file_hash(labels_path) != json.loads(exposure.read_text(encoding="utf-8"))["labels_sha256"]:
        raise ValueError("Citation evaluation labels differ from the exposed cohort")
    with labels_path.open(encoding="utf-8") as source:
        labels = {r["sample_id"]: r["target"]["hazards"] for r in map(json.loads, source) if r["sample_id"] in gold}
    laws_path = root / "data/baseline/assets/laws.jsonl"
    with laws_path.open(encoding="utf-8") as source:
        articles = select_articles(map(json.loads, source))
    joint = {}
    excluded = set(json.loads((root / "data/holdout-exclusions.json").read_text(encoding="utf-8"))["exposed_ids"])
    anchors, parts = [], {k: Counter() for k in ("both", "clip_only", "siglip2_only")}
    over_three = 0
    for prefix, valid in buckets.items():
        train = filtered_training(gold, valid, linked)
        _, frequencies = training_examples(train, labels, articles)
        first_article = {c: min(counts, key=lambda ref: (-counts[ref], ref))
                         for c, counts in frequencies.items() if counts}
        pred = {}
        for name, features in (("clip", clip), ("siglip2", sig)):
            for method, matrix in scores(train, valid, features, gold).items():
                for policy, threshold in enc["thresholds"][name][prefix][method].items():
                    decisions = policy_decisions(valid, matrix, threshold)
                    counts = joint.setdefault(f"{name}/{method}/{policy}", Counter())
                    counts.update(citation_counts(decisions, labels, first_article))
                    if method == "blend" and policy == "inner_calibrated":
                        pred[name] = decisions
        for sid in valid:
            first, second = pred["clip"][sid], pred["siglip2"][sid]
            over_three += len(first | second) > 3
            selections = {"both": first & second, "clip_only": first-second, "siglip2_only": second-first}
            for label, codes in selections.items():
                parts[label].update({"proposed": len(codes), "correct": len(codes & gold[sid])})
        # Select a disagreement in each outer group without consulting its gold labels.
        candidates = [sid for sid in valid if pred["clip"][sid] != pred["siglip2"][sid]] or valid
        sid = min(candidates, key=lambda k: digest(["kwater-prompt-pair-v1", k]))
        if sid not in excluded:
            raise ValueError("Prompt diagnostic must use an exposed, holdout-excluded photo")
        union = pred["clip"][sid] | pred["siglip2"][sid]
        anchors.append({"sid": sid, "prefix": prefix, "image": inventory[sid]["image"], "sha256": inventory[sid]["sha256"],
                        "candidates": [c for c in CODES if c in union]})
    if len(anchors) != 6:
        raise ValueError("Expected one anchor from each of the six outer groups")
    protocol = {"anchors": anchors, "arms": ARMS, "seed": 42, "model_revision": MODEL_REV, "runtime": RUNTIME,
                "gguf": {k:v for k,v in FILES.items() if k.endswith(".gguf")}, "max_tokens": 640, "context_size": context_size,
                "exposure_sha256": file_hash(exposure), "encoder_results_sha256": file_hash(root/"results/encoder_pilot.json"),
                "prompt_schema_sha256": digest([instruction(a, ["slip_trip"]) for a in ARMS])}
    write_json(local / "design.json", protocol)
    summary = {"scope": "Previously exposed, label-blind disagreement anchors; no independent efficacy estimate",
               "anchors": len(anchors), "conditions": list(ARMS), "planned_generations": len(anchors)*len(ARMS),
               "cohort_photos": len(gold), "holdout_excluded_candidates": len(excluded), "signature": digest(protocol),
               "shared_and_exclusive_candidates": {k:dict(v) for k,v in parts.items()}, "candidate_union_over_three": over_three}
    write_json(root / "results/prompt_pilot_design.json", summary)
    write_json(root / "results/encoder_law_pilot.json", {
        "scope": "Predicted code plus training-fold most frequent article; article IDs only, no assessment of action support",
        "photos": len(gold), "cited_gold_hazards": sum(bool(law_ids(h)) for row in labels.values() for h in row.values()),
        "conditions": {name: dict(counts) for name, counts in joint.items()}, "source_sha256": {
            "encoder_results": protocol["encoder_results_sha256"], "labels": file_hash(labels_path),
            "laws": file_hash(laws_path), "exposure": protocol["exposure_sha256"]}})
    return protocol


def run(root, local, cache, startup_timeout=1800, context_size=8192):
    import base64
    protocol = prepare(root, local, context_size)
    signature = digest(protocol)
    output = local / signature[:12]
    output.mkdir(parents=True, exist_ok=True)
    anchors = protocol["anchors"]
    inventory = json.loads((root / "results/train_audit_full.json").read_text(encoding="utf-8"))["inventory"]
    measured, blind, key = {arm:{} for arm in ARMS}, [], {}
    valid_ids = {a["sid"]: {} for a in anchors}
    observed_first = {arm: Counter() for arm in ARMS}
    jobs = [(i, ARMS[(i+j) % len(ARMS)]) for j in range(len(ARMS)) for i in range(len(anchors))]
    with local_server(cache, output / "model.log", extra_flags=("--load-mode", "none"),
                      startup_timeout=startup_timeout, context_size=context_size) as (port, secret, runtime):
        for i, arm in jobs:
            anchor = anchors[i]
            case = anchors[(i+1) % len(anchors)] if arm == "swapped_11" else anchor
            sid = case["sid"]
            receipt = output / f"{i}-{arm}.json"
            if receipt.is_file():
                record = json.loads(receipt.read_text(encoding="utf-8"))
                if record["signature"] != signature or record["image_sha256"] != case["sha256"]:
                    raise ValueError("Saved prompt trial provenance differs")
                result = record["generation"]
            else:
                raw = (root / "data/train" / case["image"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != case["sha256"]:
                    raise ValueError("Selected photo differs from the audit")
                # All audited competition photos are PNG. Only the local server receives them.
                if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("Expected an audited PNG")
                photo = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
                prompt, schema = instruction(arm, anchor["candidates"])
                try:
                    result = generate(port, secret, photo, prompt, schema, protocol["max_tokens"])
                except GenerationRequestError as error:
                    if error.status != 400 or error.details.get("type") != "exceed_context_size_error":
                        raise
                    result = {"finish_reason": "context_overflow", "raw_output": "", "elapsed_s": error.elapsed,
                              "prompt_tokens_required": error.details.get("n_prompt_tokens"), "context_limit": context_size}
                write_json(receipt, {"signature": signature, "image_sha256": case["sha256"], "generation": result})
            content, row = parse_generation(arm, sid, result)
            codes = None if row is None else frozenset(row["hazards"])
            measured[arm][sid] = (codes, result["elapsed_s"])
            if content is not None:
                review_id = digest(["blind", signature, sid, arm])[:16]
                observed_first[arm][next(iter(content))] += 1
                standard = {field: content[field] for field in ("scene_observation", "hazards")}
                blind.append({"review_id": review_id, "image": case["image"], "model_content": standard})
                key[review_id] = arm
                valid_ids[sid][arm] = review_id
            print(f"{i+1}/{len(anchors)} {arm} {'valid' if codes is not None else 'invalid'}", flush=True)
    write_json(output/"blind-review.json", sorted(blind, key=lambda r:r["review_id"]))
    write_json(output/"blind-key.json", key)
    pairs = []
    for anchor in anchors:
        available = valid_ids[anchor["sid"]]
        for left, right in (("direct_11", "facts_11"), ("facts_11", "verify_11"), ("verify_11", "swapped_11")):
            if left in available and right in available:
                pair_id = digest(["pair", signature, anchor["sid"], left, right])
                first, second = sorted((available[left], available[right]), key=lambda item: digest([pair_id, item]))
                pairs.append({"pair_id": pair_id[:16], "image": anchor["image"], "A": first, "B": second,
                              "photographic_support": None, "action_feasibility": None, "review_note": ""})
    write_json(output/"blind-pairs.json", {"choices":["A", "B", "tie", "both_inadequate"],
                                           "pairs":sorted(pairs, key=lambda pair:pair["pair_id"])})
    gold = {a["sid"]: frozenset(inventory[a["sid"]]["codes"]) for a in anchors}
    shared_valid = set.intersection(*[{s for s,(codes,_) in values.items() if codes is not None} for values in measured.values()])
    summary = {}
    for arm, values in measured.items():
        pred = {s:values[s][0] for s in shared_valid}
        data = {s:gold[s] for s in shared_valid}
        summary[arm] = {"valid": sum(c is not None for c,_ in values.values()), "count": len(values),
                        "seconds": sum(t for _,t in values.values()),
                        "matched_valid_metrics": evaluate(data,pred) if data else None,
                        "matched_valid_multi_hazard_exact": sum(len(g)>1 and g == pred[s] for s,g in data.items()),
                        "matched_valid_multi_hazard_photos": sum(len(g)>1 for g in data.values())}
    write_json(root/"results/prompt_pilot.json", {"scope":"Six disagreement-enriched exploratory anchors; five arms, single 4B Q4 model, identical 640-token limit",
               "signature":signature, "runtime":runtime, "paired_valid_photos":len(shared_valid), "conditions":summary,
               "paired_code_changes":paired_changes(measured, gold),
               "actual_first_field":{arm:dict(counts) for arm,counts in observed_first.items()},
               "blind_review_items":len(blind), "blind_pairs":len(pairs), "human_quality_ratings":None})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Start the local GPU server and produce paired generations")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--startup-timeout", type=int, default=1800, help="Seconds allowed for local model loading")
    p.add_argument("--context-size", type=int, default=8192, help="Input and generation context shared by all arms")
    args = p.parse_args()
    root = args.root.resolve()
    local = root / "data/prompt-pilot"
    if args.run:
        run(root, local, args.cache, args.startup_timeout, args.context_size)
    else:
        print("Prepared",len(prepare(root, local, args.context_size)["anchors"]),"paired photos")
