"""Stratify frozen BF16 baseline errors by actual 393216-to-655360 image-token increase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fold2_8b_baseline import BASE_REVISION, PREFLIGHT_RESULT, order_digest, write_json
from .fold2_v3_epoch1 import DEFAULT_RUN, LABELS, MANIFEST, MASK as FOLD2_MASK, split
from .fold4_same_verifier import digest, read
from .metrics import counts, decode_gold, decode_target, ratio, read_index


SIDECAR = Path("data/stage1-image-tokens/token-counts-by-photo-sha256.jsonl")
TOKEN_RECEIPT = Path("results/stage1-image-token-audit-20261128.json")
FOLD4_MASK = Path("data/fold4-crosssplit-visual-review/fold4-group-exclusions-local.json")
FOLD4_PREDICTIONS = Path("data/vlm-scale-seoul-8b-fold4-20261128/eval1/predictions.jsonl")
FOLD4_RECEIPT = Path("results/fold4-8b-epoch1-vs-4b-epoch2-hybrid-20261128.json")
FOLD2_PREDICTIONS = DEFAULT_RUN / "epoch1-bf16-predictions.jsonl"
FOLD2_RECEIPT = Path("results/fold2-v3-4b-epoch1-20261128.json")
FOLD2_8B_RECEIPT = Path("results/fold2-matched-4b-8b-epoch1-20261128.json")
RESULT = Path("results/stage1-resolution-opportunity-20261128.json")
TARGET_CODES = ("fall_height", "struck_by_lifting", "caught_in_machine")


def require_digest(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Input differs from the frozen benchmark fingerprint: {path}")
    return actual


def token_counts(row):
    caps = row["image_tokens_by_max_pixels"]
    if set(caps) != {"393216", "655360", "786432"}:
        raise ValueError("Image-token sidecar has unexpected caps")
    small, large = caps["393216"], caps["655360"]
    if any(type(n) is not int or n < 1 for n in caps.values()) or large < small:
        raise ValueError("Invalid image-token sidecar counts")
    return row["sha256"], large > small


def official_labels(row):
    key, codes = decode_gold(row)
    severe = {code for code, hazard in row["target"]["hazards"].items()
              if hazard["risk_level"]["severity"]["score"] == 4}
    return key, (codes, severe)


def load_predictions(path, expected, canonical_rows):
    sha = require_digest(path, expected)
    try:
        predictions = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    except ValueError:
        raise ValueError("Prediction rows must be valid and unique") from None
    require_digest(path, sha)
    if list(predictions) != [row["sample_id"] for row in canonical_rows]:
        raise ValueError("Prediction rows must cover the exact primary cohort in canonical order")
    return predictions, sha


def measure(rows, gold, predicted, severity4):
    ids = [row["sample_id"] for row in rows]
    support, predictions, hit = counts(ids, gold, predicted)
    severe = [sid for sid in ids if severity4[sid]]
    normal = [sid for sid in ids if not gold[sid]]
    multi = [sid for sid in ids if len(gold[sid]) >= 2]
    target_codes = sum(len(gold[sid]) for sid in multi)
    correct_codes = sum(len(gold[sid] & predicted[sid]) for sid in multi)
    return {
        "photos": len(ids), "manifest_groups": len({row["group_id"] for row in rows}),
        "target_codes": {
            code: {"support": support[code], "tp": hit[code], "fn": support[code] - hit[code],
                   "fp": predictions[code] - hit[code]}
            for code in TARGET_CODES
        },
        "severity4": {"photos": len(severe), "complete_misses": sum(not (severity4[sid] & predicted[sid]) for sid in severe)},
        "normal": {"photos": len(normal), "false_alarm_photos": sum(bool(predicted[sid]) for sid in normal)},
        "multi_hazard": {"photos": len(multi), "target_codes": target_codes, "correct_codes": correct_codes,
                         "micro_recall": ratio(correct_codes, target_codes)},
    }


def measure_panel(rows, changed_by_hash, gold, predicted, severity4):
    increased, unchanged = [], []
    for row in rows:
        stratum = increased if changed_by_hash[row["sha256"]] else unchanged
        stratum.append(row)
    groups = ({row["group_id"] for row in stratum} for stratum in (increased, unchanged))
    return {
        "all": measure(rows, gold, predicted, severity4),
        "manifest_groups_in_both_token_strata": len(set.intersection(*groups)),
        "token_increase": measure(increased, gold, predicted, severity4),
        "unchanged": measure(unchanged, gold, predicted, severity4),
    }


def run(fold2_8b=None, fold2_8b_receipt=FOLD2_8B_RECEIPT):
    fold4 = read(FOLD4_RECEIPT)
    fold2 = read(FOLD2_RECEIPT)
    if any(fold4[a] != fold2["recipe"][b] for a, b in (
        ("manifest_sha256", "manifest_sha256"), ("official_labels_sha256", "label_source_sha256"),
    )):
        raise ValueError("Fold 2 and Fold 4 use different benchmark manifests or labels")
    token_receipt = read(TOKEN_RECEIPT)["local_per_photo_sidecar"]
    source_sha = {"development_manifest": require_digest(MANIFEST, fold4["manifest_sha256"]),
                  "official_labels": require_digest(LABELS, fold4["official_labels_sha256"]),
                  "fold4_mask": require_digest(FOLD4_MASK, fold4["mask_sha256"]),
                  "fold2_mask": require_digest(FOLD2_MASK, fold2["recipe"]["mask_sha256"]),
                  "image_tokens": require_digest(SIDECAR, token_receipt["sha256"])}
    by_hash = read_index(SIDECAR, token_counts)
    if len(by_hash) != token_receipt["photos"] or SIDECAR.stat().st_size != token_receipt["bytes"]:
        raise ValueError("Image-token sidecar does not match its verified file receipt")
    _, rows2 = split()
    rows4 = [row for row in read(MANIFEST) if row["split"] == "dev" and row["dev_fold"] == 4]
    excluded_groups = {item["fold4_group_id"] for item in read(FOLD4_MASK)["confirmed_cross_group_links"]}
    retained4 = [r for r in rows4 if r["group_id"] not in excluded_groups]
    kept2 = set(read(FOLD2_MASK)["prospective_mask_retained_ids"])
    retained2 = [r for r in rows2 if r["sample_id"] in kept2]
    if tuple(map(len, (rows4, retained4, rows2, retained2))) != (175, 107, 197, 55):
        raise ValueError("Primary or sensitivity membership differs from the frozen cohorts")
    labels = read_index(LABELS, official_labels)
    rows = rows2 + rows4
    gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    if len(gold) != len(rows) or any(labels[sid][0] != codes for sid, codes in gold.items()):
        raise ValueError("Official label codes do not match the unique canonical rows")
    severity4 = {sid: labels[sid][1] for sid in gold}

    arms = [
        ("fold4_8b_epoch1_bf16", FOLD4_PREDICTIONS, fold4["prediction_sha256"]["8b_epoch1_standalone"],
         {"primary_175": rows4, "sensitivity_107": retained4}),
        ("fold2_4b_epoch1_bf16", FOLD2_PREDICTIONS, fold2["inference"]["prediction_sha256"],
         {"primary_197": rows2, "sensitivity_55": retained2}),
    ]
    if fold2_8b is not None:
        matched = read(fold2_8b_receipt)
        generation = matched["generated"]
        cache_key = (f"fold2-8b:{BASE_REVISION}:{generation['adapter_sha256']}:"
                     f"{fold2['recipe']['prompt_sha256']}:bf16:px393216")
        if (matched["4b_predictions_sha256"] != fold2["inference"]["prediction_sha256"]
                or not matched["ordered_training_step_controls_match"]
                or matched["recipe"]["training"] != fold2["recipe"]
                or generation["cache_key"] != cache_key
                or not generation["eval_order_sha256"] == read(PREFLIGHT_RESULT)["eval_order_sha256"] == order_digest(rows2)):
            raise ValueError("Optional 8B arm requires the measured matched Fold 2 BF16 receipt")
        arms.append(("fold2_8b_epoch1_bf16", fold2_8b, matched["8b_predictions_sha256"], arms[1][3]))
        source_sha["fold2_8b_matched_receipt"] = digest(fold2_8b_receipt)

    results = {}
    for name, path, expected, panels in arms:
        first_panel = next(iter(panels.values()))
        predictions, sha = load_predictions(path, expected, first_panel)
        results[name] = {"predictions_sha256": sha, "panels": {
            group: measure_panel(members, by_hash, gold, predictions, severity4) for group, members in panels.items()
        }}
    return {
        "scope": "Distribution of frozen 393216-pixel BF16 baseline errors. No 655360-pixel predictions were used.",
        "cohort_caveats": {
            "fold4": "175 includes confirmed cross-split scene links. The nested 107 removes three linked groups; the limited review leaves possible undiscovered overlap.",
            "fold2": "197 includes known cross-site links. The nested 55 uses its frozen prospective group exclusions and retains 16 prior ROI review photos.",
            "strata": "Token strata describe image resize geometry. Manifest groups may contribute to both strata. These are separate baselines on different folds and do not estimate an effect of resolution or establish visible cue distance.",
        },
        "baseline_max_pixels": 393216, "comparison_max_pixels": 655360,
        "input_sha256": source_sha, "arms": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold2-8b-predictions", type=Path)
    parser.add_argument("--fold2-8b-receipt", type=Path, default=FOLD2_8B_RECEIPT)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    protected = (SIDECAR, TOKEN_RECEIPT, MANIFEST, LABELS, FOLD2_MASK, FOLD4_MASK, FOLD2_PREDICTIONS,
                 FOLD4_PREDICTIONS, FOLD2_RECEIPT, FOLD4_RECEIPT, PREFLIGHT_RESULT,
                 args.fold2_8b_predictions, args.fold2_8b_receipt)
    if args.output.resolve() in {p.resolve() for p in protected if p is not None}:
        parser.error("Aggregate output must differ from every input")
    if args.fold2_8b_predictions is not None and not args.fold2_8b_predictions.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Recovered predictions must stay under data/")
    report = run(args.fold2_8b_predictions, args.fold2_8b_receipt)
    write_json(args.output, report)
    print(json.dumps({"reported_arms": list(report["arms"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
