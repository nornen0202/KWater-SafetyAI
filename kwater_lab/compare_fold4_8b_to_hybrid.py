"""Replay the 791-photo Fold 4 4B hybrid and 8B standalone with paired groups."""
import argparse
import json
from pathlib import Path

from .fold4_crosssplit_audit import DEV, LABELS, LOCAL, V3_EVAL, _score_panels
from .metrics import read_index
from .sub78_public200_compare import codes, match_changes, severity4_codes


def replay():
    arms = {
        "4b_epoch2_hybrid": V3_EVAL / "v3_hybrid_margin_verified.jsonl",
        "8b_epoch1_standalone": Path("data/vlm-scale-seoul-8b-fold4-20261128/eval1/predictions.jsonl"),
    }
    result = _score_panels(LOCAL, arms, None, "fold4_hybrid_4b_epoch2_vs_standalone_8b_epoch1")
    result["scope"] = ("Paired development comparison of the same 791-photo training scope, comparing a "
                       "4B Fold 4 hybrid research arm with the 8B Fold 4 standalone. The actual seventh submission "
                       "used full1045 training; the research arm is a proxy. Architecture, epoch and verifier differ.")
    left, right = arms
    preds = {name: read_index(path, lambda row: (row["image_id"], codes(row)), limit_size=True)
             for name, path in arms.items()}
    rows = [r for r in json.loads(DEV.read_bytes()) if r["split"] == "dev" and r["dev_fold"] == 4]
    labels = read_index(LABELS, lambda row: (row["sample_id"], row["target"]["hazards"]))
    excluded = {r["fold4_group_id"] for r in
                json.loads((LOCAL / "fold4-group-exclusions-local.json").read_bytes())["confirmed_cross_group_links"]}
    for name, panel in result["panels"].items():
        ids = [r["sample_id"] for r in rows if name == "primary_175" or r["group_id"] not in excluded]
        assert len(ids) == panel["photos"]
        safe = [sid for sid in ids if not labels[sid]]
        severe = {sid: severity4_codes(labels[sid]) for sid in ids}
        severe = {sid: values for sid, values in severe.items() if values}
        safe_correct = [{sid for sid in safe if not preds[arm][sid]} for arm in arms]
        severe_correct = [{sid for sid, values in severe.items() if preds[arm][sid] & values} for arm in arms]
        panel["paired_outcomes"] = {
            "safe_correct_no_alarm": match_changes(*safe_correct, len(safe)),
            "severity4_at_least_one_correct_code": match_changes(*severe_correct, len(severe)),
        }
        old, new = (panel["arm_metrics"][arm]["per_code"] for arm in (left, right))
        panel["per_code_delta_yield"] = {
            code: {"f1_delta": new[code]["f1"] - old[code]["f1"],
                   **{f"{key}_delta": new[code][key] - old[code][key] for key in ("tp", "fp", "fn")}}
            for code in old if new[code]["f1"] is not None and old[code]["f1"] is not None
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_bytes((json.dumps(replay(), indent=2) + "\n").encode())
