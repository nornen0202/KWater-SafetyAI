"""Ablate query fields in fixed Fold 0 native lexical law retrieval."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from .fold0_law_top1_replay import BASE, DEV, GOLD, LAWS, MASK, load_model_outputs, paired_ci, sha
from .fold0_production_compare import fold_rows
from .law_reranker_11code import Native11LawRetriever, read_selected_cases
from .production_law_replay import load_legacy, read_json, read_rows, refs
from .train_head import save_atomic


OUT = Path("results/issue2-query-ablation-20261127.json")
VARIANTS = ("full", "evidence_only", "prevention_only")


def summarize(items, cohort, groups, gold):
    selected = [item for item in items if item[0] in cohort]
    supported = [(sid, choices, laws) for sid, code, choices, laws in selected if laws]
    comparators = {}
    for a, b in combinations(VARIANTS, 2):
        change = Counter()
        deltas = defaultdict(list)
        for sid, choices, laws in supported:
            x, y = choices[a] in laws, choices[b] in laws
            change["a_only_hit" if x and not y else "b_only_hit" if y and not x
                   else "tie_hit" if x else "tie_miss"] += 1
            deltas[groups[sid]].append(int(y) - int(x))
        comparators[f"{a}_vs_{b}"] = {
            **{key: change[key] for key in ("a_only_hit", "b_only_hit", "tie_hit", "tie_miss")},
            "changed_top1_all_queries": sum(choices[a] != choices[b] for _, _, choices, _ in selected),
            "paired_group_bootstrap_b_minus_a_95_ci": paired_ci(deltas),
        }
    wrong = sum(code not in gold[sid] for sid, code, _, _ in selected)
    predicted = {(sid, code) for sid, code, _, _ in selected}
    return {
        "photos": len(cohort),
        "groups": len({groups[sid] for sid in cohort}),
        "predicted_hazard_queries": len(selected),
        "excluded_predicted_code_not_in_gold": wrong,
        "excluded_matched_code_without_law": len(selected) - len(supported) - wrong,
        "gold_hazard_with_law_missing_predicted_code": sum(
            bool(refs(hazard)) and (sid, code) not in predicted
            for sid in cohort for code, hazard in gold[sid].items()
        ),
        "matched_annotated_code_with_law": len(supported),
        "groups_with_matched_annotated_code_and_law": len({groups[sid] for sid, _, _ in supported}),
        "top1_hits": {name: sum(choices[name] in laws for _, choices, laws in supported)
                      for name in VARIANTS},
        "pairs": comparators,
    }


def run():
    fold = fold_rows(DEV)
    groups = {row["sample_id"]: row["group_id"] for row in fold}
    train_ids = {r["sample_id"] for r in read_json(DEV) if r["split"] == "dev" and r["dev_fold"] != 0}
    train = list(read_selected_cases(GOLD, train_ids))
    if len(train) != 750 or {r["sample_id"] for r in train} != train_ids or train_ids & groups.keys():
        raise ValueError("Expected 750 distinct development training cases outside Fold 0")
    gold = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(GOLD, set(groups))}
    if gold.keys() != groups.keys():
        raise ValueError("Missing complete Fold 0 annotations")
    mask = read_json(MASK)
    if mask["canonical_manifest_sha256"] != sha(DEV):
        raise ValueError("The group exclusion mask does not match the development manifest")
    excluded = {link["fold0_group_id"] for link in mask["confirmed_cross_group_links"]}
    cohorts = {"all_fold0": set(groups),
               "confirmed_cross_group_exclusion": {sid for sid, group in groups.items() if group not in excluded}}
    if len(cohorts["confirmed_cross_group_exclusion"]) != 34:
        raise ValueError("Expected the fixed 34-photo sensitivity panel")
    models, hashes = load_model_outputs(fold)
    _, code_map = load_legacy(BASE)
    native = Native11LawRetriever(read_rows(LAWS), train)
    panels = {}
    for name, (formatted, raw) in models.items():
        items = []
        for sid, final in formatted.items():
            if not final["hazards"]:
                continue
            raw_obj = raw[sid]
            by_code = {}
            for hazard in raw_obj["hazards"]:
                code = code_map.get(hazard["code"]) if name == "official" else hazard["code"]
                by_code.setdefault(code, hazard)
            for code in final["hazards"]:
                evidence, prevention = (by_code[code][field] for field in ("evidence", "prevention"))
                scene = raw_obj.get("scene_note", "")
                fields = {"full": (evidence, prevention, scene),
                          "evidence_only": (evidence, "", ""), "prevention_only": ("", prevention, "")}
                choices = {}
                for label, query in fields.items():
                    ranked = native.select_osh_references(code, *query, max_refs=1)
                    choices[label] = ranked[0]["id"] if ranked else None
                items.append((sid, code, choices, set(refs(gold[sid].get(code, {})))))
        panels[name] = {key: summarize(items, cohort, groups, gold) for key, cohort in cohorts.items()}
    return {
        "scope": "Exploratory replay of already observed Fold 0. Only lexical query fields vary; code, generated text, index and case pool are fixed.",
        "training_pool_photos": len(train_ids), "holdout_used_for_retrieval": False,
        "variants": list(VARIANTS), "max_refs": 1, "bootstrap": {"seed": 42, "resamples": 2000},
        "input_sha256": hashes | {path.as_posix(): sha(path) for path in (DEV, GOLD, LAWS, MASK)},
        "panels": panels,
    }


if __name__ == "__main__":
    summary = run()
    payload = (json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    save_atomic(OUT, lambda stream: stream.write(payload))
