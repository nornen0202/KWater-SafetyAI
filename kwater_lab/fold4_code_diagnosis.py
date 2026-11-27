"""Aggregate Fold 4 code errors without reading image or generated text fields."""
import argparse
import ast
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from .metrics import CODES, read_index

DEV = Path("data/clean-split-v1/split_manifest_dev.json")
MASK = Path("data/fold4-crosssplit-visual-review/fold4-group-exclusions-local.json")
AUDIT = Path("results/fold4-crosssplit-audit.json")
GOLD = Path("data/train/train.jsonl")
BASELINE = Path("data/baseline/main.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def code_map():
    for node in ast.parse(BASELINE.read_bytes()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(key, ast.Name) and key.id == "CODE_MAP" for key in node.targets
        ):
            mapping = ast.literal_eval(node.value)
            if len(mapping) == 8 and len(set(mapping.values())) == 8 and set(mapping.values()) <= set(CODES):
                return mapping
    raise ValueError("Expected the official eight-code output mapping")


def fold4():
    mask = read(MASK)
    if mask["canonical_manifest_sha256"] != digest(DEV) or read(AUDIT)["confirmed_mask_sha256"] != digest(MASK):
        raise ValueError("Fold 4 manifest or finalized mask mismatch")
    rows = [row for row in read(DEV) if row["split"] == "dev" and row["dev_fold"] == 4]
    groups = {row["sample_id"]: row["group_id"] for row in rows}
    gold = {row["sample_id"]: set(row["codes"]) for row in rows}
    excluded = {row["fold4_group_id"] for row in mask["confirmed_cross_group_links"]}
    retained = {sid: group for sid, group in groups.items() if group not in excluded}
    if (len(rows), len(groups), len(set(groups.values())), len(retained), len(set(retained.values()))) != (175, 175, 39, 107, 36):
        raise ValueError("Unexpected Fold 4 cohort")
    from .law_reranker_11code import read_selected_cases

    severe = {}
    for row in read_selected_cases(GOLD, gold.keys()):
        sid = row["sample_id"]
        hazards = row["target"]["hazards"]
        if sid in severe or set(hazards) != gold[sid]:
            raise ValueError("Duplicate label or disagreement with Fold 4 manifest")
        severe[sid] = {code for code, hazard in hazards.items() if hazard["risk_level"]["severity"]["score"] == 4}
    if severe.keys() != gold.keys():
        raise ValueError("Incomplete Fold 4 severity labels")
    return {"primary_175": groups, "sensitivity_107": retained}, gold, severe


def project_prediction(row):
    codes = set(row["hazards"])
    status = row["status"]
    if (row["schema_version"] != "1.0" or not isinstance(row["hazards"], dict)
            or not codes <= set(CODES) or status not in {"hazard", "no_visible_hazard"}
            or (status == "hazard") != bool(codes) or len(codes) > 3):
        raise ValueError("Invalid prediction code set or schema status")
    return row["image_id"], codes


def summarize(groups, gold, severe, predicted, outside):
    rows = {code: dict.fromkeys(("support", "pred", "tp", "fp", "fn", "normal_fp",
                                "severity4_support", "severity4_fn", "severity4_completely_missed"), 0)
            for code in CODES}
    missed_photos = normal = normal_fp = severe_photos = outside_present = outside_missed = 0
    normal_sets = Counter()
    for sid in groups:
        actual, pred, high = gold[sid], predicted[sid], severe[sid]
        missed = bool(high) and not (high & pred)
        missed_photos += missed
        severe_photos += bool(high)
        normal += not actual
        normal_fp += not actual and bool(pred)
        outside_present += bool(actual & outside)
        outside_missed += bool((actual & outside) - pred)
        if not actual and pred:
            normal_sets[tuple(sorted(pred))] += 1
        for code, row in rows.items():
            row["support"] += code in actual
            row["pred"] += code in pred
            row["tp"] += code in (actual & pred)
            row["fp"] += code in (pred - actual)
            row["fn"] += code in (actual - pred)
            row["normal_fp"] += not actual and code in pred
            row["severity4_support"] += code in high
            row["severity4_fn"] += code in (high - pred)
            row["severity4_completely_missed"] += missed and code in high
    active = [r for r in rows.values() if r["support"] or r["pred"]]
    return {
        "photos": len(groups), "groups": len(set(groups.values())),
        "active_macro_f1": sum(2*r["tp"]/(r["support"]+r["pred"]) for r in active) / len(active),
        "normal_photos": normal, "normal_fp_photos": normal_fp,
        "normal_fp_code_sets": [{"codes": group, "photos": count} for group, count in normal_sets.most_common()],
        "severity4_photos": severe_photos, "severity4_completely_missed_photos": missed_photos,
        "severity4_codes": sum(r["severity4_support"] for r in rows.values()),
        "severity4_missed_code_instances": sum(r["severity4_fn"] for r in rows.values()),
        "outside_8code_present_photos": outside_present, "outside_8code_missed_photos": outside_missed,
        "outside_8code_present_instances": sum(rows[c]["support"] for c in outside),
        "outside_8code_missed_instances": sum(rows[c]["fn"] for c in outside),
        "by_code": rows,
    }


def compare(groups, gold, severe, before, after):
    """Count new and lost hits or false positives on identical photo/code pairs."""
    by_code = {}
    for code in CODES:
        recovered = lost = new_fp = cleared_fp = severe_recovered = severe_lost = 0
        for sid in groups:
            gained = code in after[sid] and code not in before[sid]
            removed = code in before[sid] and code not in after[sid]
            annotated = code in gold[sid]
            recovered += gained and annotated
            lost += removed and annotated
            new_fp += gained and not annotated
            cleared_fp += removed and not annotated
            severe_recovered += gained and code in severe[sid]
            severe_lost += removed and code in severe[sid]
        by_code[code] = dict(recovered_tp=recovered, lost_tp=lost, new_fp=new_fp, cleared_fp=cleared_fp,
                             severity4_recovered_tp=severe_recovered, severity4_lost_tp=severe_lost)
    return by_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "PREDICTIONS"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {name: Path(path) for name, path in args.arm}
    if len(arms) != len(args.arm) or args.output.resolve() in {
        path.resolve() for path in (DEV, MASK, AUDIT, GOLD, BASELINE, *arms.values())
    }:
        parser.error("Provide distinct arm names and an aggregate output outside the inputs")
    mapping = code_map()
    outside = set(CODES) - set(mapping.values())
    panels, gold, severe = fold4()
    result = {
        "scope": "Fold 4 code diagnosis. Fixed 175-photo cohort and confirmed cross-group exclusion at 107 photos. No thresholds selected.",
        "normal_fp_definition": "A code predicted for a photo with no annotated hazard. The code identifies the error pattern, without inferring its visual cause.",
        "severity4_complete_miss_definition": "No severity-4 ground-truth code appears in the prediction, regardless of predicted severity. One photo may contribute to multiple code counts.",
        "official_internal_to_output_code_map": mapping, "outside_8code_output_codes": sorted(outside),
        "cohort_sha256": {"manifest": digest(DEV), "mask": digest(MASK)},
        "arms": {},
    }
    predictions = {}
    for name, path in arms.items():
        before = digest(path)
        pred = read_index(path, project_prediction, limit_size=True)
        if pred.keys() != gold.keys() or digest(path) != before:
            raise ValueError(f"{name} requires immutable, exact once-per-photo 175-row coverage")
        predictions[name] = pred
        result["arms"][name] = {"prediction_sha256": before, "panels": {
            panel: summarize(groups, gold, severe, pred, outside) for panel, groups in panels.items()
        }}
    if len(predictions) > 1:
        result["paired"] = [
            {"before": first, "after": second, "panels": {
                panel: compare(groups, gold, severe, predictions[first], predictions[second])
                for panel, groups in panels.items()
            }}
            for first, second in combinations(predictions, 2)
        ]
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
