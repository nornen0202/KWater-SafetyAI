"""Fold 0 top-1 law replay from two existing, unchanged production generations."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from .fold0_production_compare import fold_rows, image_id
from .law_reranker_11code import Native11LawRetriever, article_to_osh_id, read_selected_cases
from .metrics import decode_target, percentile, read_index
from .production_law_replay import load_legacy, read_json, read_rows, refs


DEV = Path("data/clean-split-v1/split_manifest_dev.json")
BASE = Path("data/baseline")
MASK = Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")
OFFICIAL = Path("data/fold0-qlora-v2-118-official")
V2 = Path("data/local-eval-predictions/fold0/fold0-qlora-v2-118-bf16")
LAWS = BASE / "assets/laws.jsonl"
GOLD = Path("data/train/train.jsonl")
OUT = Path("results/fold0-fixed-output-law-replay.json")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_ci(by_group):
    members = [by_group[key] for key in sorted(by_group)]
    if len(members) < 2:
        return None
    rng = random.Random(42)
    values = []
    for _ in range(2000):
        group_sample = rng.choices(members, k=len(members))
        values.append(sum(delta for group in group_sample for delta in group) / sum(map(len, group_sample)))
    return [percentile(sorted(values), quantile) for quantile in (0.025, 0.975)]


def aggregate(items, photos, groups):
    selected = [item for item in items if item["sid"] in photos]
    supported = [item for item in selected if item["gold_refs"]]
    counts = Counter()
    deltas = defaultdict(list)
    for item in supported:
        gold = item["gold_refs"]
        original, legacy, native = (item[field] in gold for field in ("original", "legacy", "native"))
        delta = int(native) - int(legacy)
        counts["original_top1_hits"] += original
        counts["legacy_top1_hits"] += legacy
        counts["native_top1_hits"] += native
        counts["changed_top1_supported"] += item["legacy"] != item["native"]
        counts["native_only_hit" if delta > 0 else "legacy_only_hit" if delta < 0 else "tie_correct" if native else "tie_miss"] += 1
        deltas[groups[item["sid"]]].append(delta)
    fields = ("original_top1_hits", "legacy_top1_hits", "native_top1_hits", "changed_top1_supported",
              "native_only_hit", "legacy_only_hit", "tie_correct", "tie_miss")
    return {
        "photos": len(photos), "groups": len({groups[sid] for sid in photos}),
        "predicted_hazard_queries": len(selected),
        "changed_top1_all_queries": sum(item["legacy"] != item["native"] for item in selected),
        "matched_annotated_code_with_law": len(supported), "groups_with_matched_annotated_code_and_law": len(deltas),
        **{name: counts[name] for name in fields},
        "paired_group_bootstrap_native_minus_legacy_hit_rate_95_percent": paired_ci(deltas),
    }


def load_model_outputs(rows):
    ids = {r["sample_id"] for r in rows}
    anon_ids = {image_id(sid): sid for sid in ids}
    o_out = OFFICIAL / "output/submission.jsonl"
    v_out = V2 / "ft_vlm_11code.jsonl"
    o_report = read_json(Path("results/fold0_qlora_v2_118_official.json"))
    v_report = read_json(Path("results/fold0_qlora_v2_118_bf16_vlm.json"))
    if (o_report["official_checker_valid_rows"] != len(ids) or o_report["output_sha256"] != sha(o_out)
            or v_report["photos"] != len(ids) or v_report["formatted_output_sha256"] != sha(v_out)
            or v_report["manifest_sha256"] != sha(DEV)):
        raise ValueError("One original output differs from the completed production receipt")

    def formatted(path, local):
        return read_index(path, lambda r: (decode_target(r, local_ids=local)[0], r), limit_size=True)

    official = {anon_ids[sid]: r for sid, r in formatted(o_out, False).items()}
    v2 = formatted(v_out, True)
    official_batches = sorted((OFFICIAL / "raw-batches").glob("[0-9]*.json"))
    o_raw = [r for path in official_batches for r in read_json(path)]
    v_raw = read_rows(V2 / "ft_vlm_raw.jsonl")
    if (len(o_raw) != len(ids) or len(v_raw) != len(ids) or {r["sample_id"] for r in o_raw} != anon_ids.keys()
            or {r["sample_id"] for r in v_raw} != ids or official.keys() != ids or v2.keys() != ids):
        raise ValueError("Both the final output and raw generation must cover the same 216 photos")
    return {
        "official": (official, {anon_ids[row["sample_id"]]: row["prediction"] for row in o_raw}),
        "v2_bf16": (v2, {row["sample_id"]: row["parsed"] for row in v_raw}),
    }, {str(path): sha(path) for path in (o_out, v_out, V2 / "ft_vlm_raw.jsonl")} | {
        "official_raw_batch_hashes_sha256": hashlib.sha256("".join(sha(p) for p in official_batches).encode()).hexdigest(),
    }


def run():
    rows = fold_rows(DEV)
    groups = {r["sample_id"]: r["group_id"] for r in rows}
    ids = set(groups)
    train_ids = {r["sample_id"] for r in read_json(DEV) if r["split"] == "dev" and r["dev_fold"] != 0}
    training_rows = list(read_selected_cases(GOLD, train_ids))
    legacy_rows = list(read_selected_cases(BASE / "assets/train.jsonl", train_ids))
    if len(train_ids) != 750 or {r["sample_id"] for r in training_rows} != train_ids or {r["sample_id"] for r in legacy_rows} != train_ids:
        raise ValueError("Expected the exact same 750 Fold 0 training photos in both case banks")
    targets = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(GOLD, ids)}
    if targets.keys() != ids:
        raise ValueError("Missing official annotations")
    laws = read_rows(LAWS)
    legacy_type, code_map = load_legacy(BASE)
    native = Native11LawRetriever(laws, training_rows)
    legacy = legacy_type(laws, legacy_rows)
    inverse = {v: k for k, v in code_map.items()}
    outputs, input_hashes = load_model_outputs(rows)
    mask = read_json(MASK)
    if sha(DEV) != mask["canonical_manifest_sha256"]:
        raise ValueError("Confirmed group sensitivity mask no longer matches the benchmark")
    excluded = {key["fold0_group_id"] for key in mask["confirmed_cross_group_links"]}
    scoped = {"all_fold0": ids, "confirmed_cross_group_exclusion": {sid for sid, group in groups.items() if group not in excluded}}
    result = {}
    for name, (formatted, raw) in outputs.items():
        items = []
        for sid, final in formatted.items():
            if not final["hazards"]:
                continue
            raw_obj = raw[sid]
            raw_hazards = {}
            for hazard in raw_obj["hazards"]:
                code = code_map.get(hazard["code"]) if name == "official" else hazard["code"]
                raw_hazards.setdefault(code, hazard)
            for code, saved in final["hazards"].items():
                hazard = raw_hazards[code]
                evidence, action = hazard["evidence"], hazard["prevention"]
                scene = raw_obj.get("scene_note", "")
                a = legacy.select({"code": inverse.get(code, ""), "evidence": evidence, "prevention": action}, scene)[:1]
                b = native.select_osh_references(code, evidence, action, scene, max_refs=1)
                source = refs(saved)
                items.append({"sid": sid, "gold_refs": set(refs(targets[sid].get(code, {}))),
                              "original": source[0] if source else None,
                              "legacy": article_to_osh_id(a[0].get("article")) if a else None,
                              "native": b[0]["id"] if b else None})
        result[name] = {scope: aggregate(items, cohort, groups) for scope, cohort in scoped.items()}
    return {
        "scope": "Original predicted codes and raw evidence/action/scene text fixed within each 216-photo panel; compare existing first-citation rankers only.",
        "case_training_photos": len(train_ids), "case_ids_same": True, "max_refs": 1,
        "input_sha256": input_hashes | {path.as_posix(): sha(path) for path in
                                      (DEV, MASK, LAWS, GOLD, BASE / "assets/train.jsonl")},
        "panels": result,
    }


if __name__ == "__main__":
    import sys
    result = run()
    assert "torch" not in sys.modules
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
