"""Verify group boundaries, rescore stored OOF predictions and run the official checker."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .metrics import (
    CODES,
    decode_target,
    evaluate,
    load_gold,
    load_predictions,
    unique_keys,
)
from .phase01_data import sha, write_json
from .submission2_output import make_row


def read(p):
    return json.loads(p.read_bytes(), object_pairs_hook=unique_keys)


def write_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def checked_output(checker, project, inputs, output, n):
    cmd = [
        str(checker),
        "-B",
        "-X",
        "utf8",
        str(project / "data/local-kit/kit/check_output.py"),
        str(inputs),
        str(output),
    ]
    # Run the distributed checker with its own dependency environment.
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, capture_output=True, encoding="utf8", env=env)
    if result.returncode or f"{n}/{n}" not in result.stdout:
        raise ValueError(
            f"Official checker: {result.returncode} {result.stdout} {result.stderr}"
        )
    return result.stdout.strip()


def replay(scores, policy):
    p = [
        1 / (1 + math.exp(-max(-700.0, min(700.0, z / t))))
        for z, t in zip(scores, policy["temperatures"], strict=True)
    ]
    if p[-1] < policy["scene_gate"]:
        return set()
    candidates = [j for j in range(len(CODES)) if p[j] >= policy["thresholds"][j]]
    candidates.sort(key=lambda j: -(p[j] - policy["offset"][j]))
    return {CODES[j] for j in candidates[:3]}


def verify(project, run, checker):
    done = read(run / "completed.json")
    phase0 = read(run / "data-summary.json")
    split = read(run / "evaluation-split.json")
    gold = load_gold(project / "data/train/train.jsonl")
    audit = read(project / "results/train_audit_full.json")
    manifest = read(run / "evaluation-manifest.json")
    by_id = {r["sample_id"]: r for r in manifest}
    if (
        len(by_id) != len(manifest)
        or set(by_id) != set(gold)
        or len(gold) != done["photos"]
    ):
        raise ValueError("Manifest does not cover the complete official train")
    if sha(project / "data/train/train.jsonl") != phase0["labels_sha256"]:
        raise ValueError("Official labels changed")
    reserved = {
        r["sample_id"] for r in read(run / "manifest.json") if r["dev_fold"] < 0
    }
    if reserved != {s for s, r in by_id.items() if r["dev_fold"] < 0}:
        raise ValueError("Changed sealed holdout membership")
    if any(
        (run / name).exists()
        for name in (
            "all_fold-oof.jsonl",
            "all_fold-metrics.json",
            "holdout-metrics.json",
        )
    ):
        raise ValueError("Unexpected sealed evaluation artifacts")
    dev_policy = read(run / "phase3-development/policy.json")
    full_policy = read(run / "inference/policy.json")
    if (
        dev_policy["head"] == full_policy["head"]
        or dev_policy["policy"] != full_policy["policy"]
    ):
        raise ValueError("Development and final-fit policies differ or share a head")
    for folder, cfg in (("phase3-development", dev_policy), ("inference", full_policy)):
        if sha(run / folder / cfg["head"]) != sha(run / cfg["head"]):
            raise ValueError("Incorrect packaged weights")
    for name, digest in done["files"].items():
        if sha(run / name) != digest:
            raise ValueError(f"Run artifact changed: {name}")
    group_index = {}
    for sid, r in by_id.items():
        if set(r["codes"]) != gold[sid] or (r["exposed"] and r["dev_fold"] < 0):
            raise ValueError("Gold or exposure mismatch")
        signature = (r["group"], r["dev_fold"])
        for typ, key in (
            ("source", r["source"]),
            ("group", r["group"]),
            ("pixels", audit["inventory"][sid]["pixels_sha256"]),
            ("bytes", r["sha256"]),
        ):
            if group_index.setdefault((typ, key), signature) != signature:
                raise ValueError(f"Fold crosses {typ}: {key}")
    for filename in ("pairs.json", "semantic-pairs.json"):
        for pair in read(run / filename):
            if (
                pair["decision"] != "unresolved"
                and by_id[pair["left"]]["group"] != by_id[pair["right"]]["group"]
            ):
                raise ValueError("Registered visual neighbours span groups")
    directory = run / "oof-checks"
    directory.mkdir(exist_ok=True)
    kit = {}
    historical = {}
    pilot = set(read(project / "data/prefix-pilot/exposure.json")["exposed_ids"])
    if not pilot <= gold.keys():
        raise ValueError("Historic cohort differs from current official train")
    for axis in ("dev_fold",):
        rows = [
            json.loads(line, object_pairs_hook=unique_keys)
            for line in (run / (axis + "-oof.jsonl")).read_bytes().splitlines()
        ]
        expected = {k: v for k, v in gold.items() if by_id[k][axis] >= 0}
        ids = [r["sample_id"] for r in rows]
        if set(ids) != set(expected) or len(ids) != len(expected):
            raise ValueError("Missing or repeated OOF photo")
        groups = {k: by_id[k]["group"] for k in ids}
        report = read(run / (axis + "-metrics.json"))
        metric = report["arms"]
        if {r["fold"] for r in rows} != set(range(5)):
            raise ValueError("Incomplete five-fold evaluation")
        for r in rows:
            sid = r["sample_id"]
            if (
                r["fold"] != by_id[sid][axis]
                or r["group"] != groups[sid]
                or set(r["gold"]) != expected[sid]
            ):
                raise ValueError("OOF metadata mismatch")
            if (
                set(r["predictions"]) != set(metric)
                or len(r["logits"]) != 2
                or not all(
                    len(v) == 12 and all(map(math.isfinite, v))
                    for v in r["logits"].values()
                )
            ):
                raise ValueError("Missing predictions or logits")
        local_ids = {
            s: hashlib.sha256(("phase01/" + s).encode()).hexdigest()[:32] for s in ids
        }
        if len(set(local_ids.values())) != len(ids):
            raise ValueError("Hashed ID collision")
        for name, reference in metric.items():
            preds = {}
            for r in rows:
                s = r["sample_id"]
                codes = r["predictions"][name]
                if (
                    len(codes) != len(set(codes))
                    or len(codes) > 3
                    or not set(codes) <= set(CODES)
                ):
                    raise ValueError(f"Invalid hazard list: {s}")
                _, preds[s] = decode_target(make_row(local_ids[s], codes))
                mode, arm = name.split("_", 1)
                policy = report["policies"][f"{axis}-{mode}-{r['fold']}"][arm]
                if replay(r["logits"][mode], policy) != preds[s]:
                    raise ValueError(
                        "Saved fold policy does not reproduce OOF prediction"
                    )
            independent = evaluate(expected, preds, groups)
            for code, values in independent["per_code"].items():
                if any(reference["per_code"][code][k] != v for k, v in values.items()):
                    raise ValueError("Per-class recount mismatch")
            if any(
                reference[k] != v for k, v in independent.items() if k != "per_code"
            ):
                raise ValueError("Pooled metric mismatch")
            if pilot:
                matched = [r for r in rows if r["sample_id"] in pilot]
                subsets = {s: gold[s] for s in pilot}
                out = {s: preds[s] for s in pilot}
                cohort = evaluate(subsets, out)
                lost_in = lost_out = rescued = 0
                for row in matched:
                    sid = row["sample_id"]
                    mode = name.split("_", 1)[0]
                    rank = sorted(
                        range(len(CODES)), key=lambda j: -row["logits"][mode][j]
                    )[:3]
                    top = {CODES[j] for j in rank}
                    lost_in += len((gold[sid] - preds[sid]) & top)
                    lost_out += len(gold[sid] - preds[sid] - top)
                    rescued += len((gold[sid] & preds[sid]) - top)
                historical[name] = dict(
                    photos=len(matched),
                    gold_instances=sum(map(len, subsets.values())),
                    macro_f1=cohort["macro_f1"],
                    fp_on_negatives=cohort["no_hazard_false_alarms"],
                    final_misses_inside_raw_top3=lost_in,
                    final_misses_outside_raw_top3=lost_out,
                    final_hits_from_outside_raw_top3=rescued,
                )
            if name in {
                "single_fixed",
                "multi_class_utility_scene",
                done["winner"],
            }:
                inputs, output = directory / "input", directory / name
                inputs.mkdir(exist_ok=True)
                output.mkdir(exist_ok=True)
                write_rows(
                    inputs / "manifest.jsonl",
                    (dict(image_id=local_ids[s]) for s in ids),
                )
                write_rows(
                    output / "submission.jsonl",
                    (
                        make_row(local_ids[s], [c for c in CODES if c in preds[s]])
                        for s in ids
                    ),
                )
                kit[name] = checked_output(checker, project, inputs, output, len(ids))
                hex_gold = {local_ids[s]: expected[s] for s in ids}
                reopened = load_predictions(output / "submission.jsonl", hex_gold)
                if evaluate(hex_gold, reopened)["macro_f1"] != independent["macro_f1"]:
                    raise ValueError("Round trip score mismatch")
    return dict(
        photos=len(gold),
        groups=split["groups"],
        folds=5,
        metrics_per_axis=len(metric),
        axes=["dev_fold"],
        isolated_holdout_scored=False,
        known_group_or_duplicate_crossings=0,
        exposed_holdout=0,
        bounded_rows=len(expected),
        official_kit=kit,
        historical_170_ids_new_training_protocol=historical,
        evaluation_manifest_sha256=sha(run / "evaluation-manifest.json"),
        oof_sha256=sha(run / "dev_fold-oof.jsonl"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--checker-python", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.project, a.run, a.checker_python)
    write_json(a.run / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
