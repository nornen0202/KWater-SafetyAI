"""Compare native and legacy law retrieval on cached organizer LoRA generations."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import quote, unquote

from .law_reranker_11code import Native11LawRetriever, article_to_osh_id, read_selected_cases
from .metrics import unique_keys


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def read_rows(path):
    return [json.loads(line, object_pairs_hook=unique_keys) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_legacy(path):
    spec = importlib.util.spec_from_file_location("organizer_law_retriever", path / "scripts/rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    syntax = ast.parse((path / "main.py").read_text(encoding="utf-8"))
    code_map = next(ast.literal_eval(node.value) for node in syntax.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "CODE_MAP" for target in node.targets))
    return module.LawRetriever, code_map


def refs(hazard):
    return [ref["id"] for ref in hazard.get("references", ()) if ref["kind"] == "law"]


def score_references(counts, gold, predicted, match):
    gold = set(gold)
    predicted_set = set(predicted)
    common = gold & predicted_set
    counts.update(gold=len(gold), predicted=len(predicted_set), true_positive=len(common),
                  unannotated_citations=len(predicted_set - gold), missing_gold_citations=len(gold - predicted_set))
    if match:
        counts.update(matched_gold_hazards_with_citation=bool(gold),
                      matched_gold_hazards_exact=bool(gold) and gold == predicted_set,
                      matched_gold_hazards_hit_at_1=bool(predicted) and predicted[0] in gold,
                      matched_gold_hazards_missed_all=bool(gold) and not common)


def summarize(counts):
    return {**counts, "citation_micro_f1": round(2 * counts["true_positive"] / max(1, counts["gold"] + counts["predicted"]), 4)}


def replay(root: Path, cache: Path, manifest_path: Path) -> dict:
    base = root / "data/baseline"
    laws = read_rows(base / "assets/laws.jsonl")
    manifest = {row["sample_id"]: row for row in read_json(manifest_path)}
    legacy_cls, code_map = load_legacy(base)
    inverse_map = {value: key for key, value in code_map.items()}
    prototype_path = root / "data/adapter-scale-pilot/generated-scale-1.0.json"
    prototype_raw = {row["sample_id"]: row["prediction"] for row in read_json(prototype_path)}
    panels = {}
    skipped = {}
    for name in ("prototype18", "fresh64"):
        candidates = sorted((cache / "release-replay" / name).glob("*.json"))
        eligible = [path for path in candidates if manifest.get(unquote(path.stem), {}).get("split") == "dev"]
        panels[name] = [read_json(path) for path in eligible]
        skipped[name] = len(candidates) - len(eligible)
    if not all(panels.values()):
        raise ValueError("Expected both official production replay panels")
    evaluation_ids = [row["sample_id"] for panel in panels.values() for row in panel]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("One photo occurs more than once in cached panels")
    excluded_groups = {manifest[sid]["group_id"] for sid in evaluation_ids}
    training_ids = {sid for sid, row in manifest.items() if row["split"] == "dev" and row["group_id"] not in excluded_groups}
    labels = {row["sample_id"]: row for row in read_selected_cases(root / "data/train/train.jsonl", training_ids.union(evaluation_ids))}
    native = Native11LawRetriever(laws, [row for sid, row in labels.items() if sid in training_ids])
    legacy = legacy_cls(laws, list(read_selected_cases(base / "assets/train.jsonl", training_ids)))

    def rank_native(code, hazard, scene_note):
        return [ref["id"] for ref in native.select_osh_references(code, hazard["evidence"], hazard["prevention"], scene_note, max_refs=1)]

    def rank_legacy(code, hazard, scene_note):
        candidates = legacy.select({"code": inverse_map.get(code, ""), "evidence": hazard["evidence"],
                                    "prevention": hazard["prevention"]}, scene_note)
        return [canonical for ref in candidates[:1] if (canonical := article_to_osh_id(ref.get("article")))]

    result = {
        "case_scope": "Both replays use development rows excluding every evaluated incident group; production cache used the organizer's original bundled examples.",
        "max_refs_per_hazard": 1,
        "case_training_photos": len(training_ids),
        "excluded_incident_groups": len(excluded_groups),
        "cache_outside_development_skipped": skipped,
        "interpretation": "Cached official LoRA used official training images. Unannotated citations are disagreements with provided labels. No new generation or independent VLM accuracy estimate.",
        "panels": {},
    }
    for name, panel in panels.items():
        generated = {key: Counter() for key in ("original_production", "legacy_train_only", "native_train_only")}
        oracle = {key: Counter() for key in ("legacy_train_only", "native_train_only")}
        hazard_counts = Counter()
        for receipt in panel:
            sid = receipt["sample_id"]
            final = receipt["outputs"]["pristine"]
            anon = hashlib.sha256(f"kwater/scale/{sid}".encode()).hexdigest()[:32]
            if final["image_id"] != anon:
                raise ValueError("Cached prediction ID differs from the linked source photo")
            raw = prototype_raw[anon] if name == "prototype18" else read_json(cache / "A" / f"{quote(sid, safe='')}.json")["raw"]
            gold = labels[sid]["target"]["hazards"]
            predicted = final["hazards"]
            raw_hazards = {code_map.get(h["code"]): h for h in raw["hazards"]} if receipt["raw_valid"] else {}
            scene = raw["scene_note"] if receipt["raw_valid"] else ""
            hazard_counts.update(raw_valid_photos=receipt["raw_valid"], matched_codes=len(gold.keys() & predicted.keys()),
                                 missed_gold_codes=len(gold.keys() - predicted.keys()), extra_predicted_codes=len(predicted.keys() - gold.keys()))
            for code in sorted(gold.keys() | predicted.keys()):
                target_refs = refs(gold.get(code, {}))
                match = code in gold and code in predicted
                original = refs(predicted.get(code, {}))
                if len(original) > 1:
                    raise ValueError("Cached output exceeded the production top-1 citation limit")
                score_references(generated["original_production"], target_refs, original, match)
                for key, ranker in (("legacy_train_only", rank_legacy), ("native_train_only", rank_native)):
                    selected = ranker(code, raw_hazards[code], scene) if code in predicted else []
                    score_references(generated[key], target_refs, selected, match)
                    if code in gold:
                        gold_scene = (labels[sid].get("notes") or {}).get("scene_note", "")
                        score_references(oracle[key], target_refs, ranker(code, gold[code], gold_scene), True)
        result["panels"][name] = {
            "photos": len(panel),
            "fold0_overlap_photos": sum(manifest[r["sample_id"]]["dev_fold"] == 0 for r in panel),
            "hazard_counts": dict(hazard_counts),
            "generated_text_and_predicted_codes": {key: summarize(counts) for key, counts in generated.items()},
            "oracle_gold_text_and_codes": {key: summarize(counts) for key, counts in oracle.items()},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache", type=Path, default=Path("data/phase2-runs/p38-20261126-49bb82ce"))
    parser.add_argument("--manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_dev.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    values = replay(args.root, args.cache, args.manifest)
    args.out.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
