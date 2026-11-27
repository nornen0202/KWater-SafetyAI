"""Score complete private ROI caches locally, keeping failed outputs in the coverage count."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .vision_gate_roi_run import ARMS, ROOT, WITNESS


def score(cache_dir, panel_dir):
    rows = json.loads((panel_dir / "key-local.json").read_bytes())
    truth = {r["id"]: r for r in rows}
    obs = {r["id"]: r["relation"] for r in csv.DictReader(
        (panel_dir / ROOT.name / "observations.csv").read_text(encoding="utf-8-sig").splitlines())}
    if len(truth) != 48 or len(rows) != 48 or truth.keys() != obs.keys():
        raise ValueError("Frozen panel and observations are not aligned")
    witness = json.loads(WITNESS.read_bytes())
    inputs = panel_dir / ROOT.name
    hashes = json.loads((inputs / "input-sha256-local.json").read_bytes())
    hashes.update({f"roi_full/{r['id']}.png": r["roi_full_sha256"] for r in
                   json.loads((inputs / "full-geometry-local.json").read_bytes())})
    previous_signature = None
    outputs, totals = {}, {}
    for arm in ARMS:
        cache = json.loads((cache_dir / f"{arm}.json").read_bytes())
        indexed = cache["outputs"]
        if indexed.keys() != truth.keys() or any(r["sample_id"] != sid for sid, r in indexed.items()):
            raise ValueError(f"{arm}: missing or duplicate outputs")
        if cache["identity"]["samples"] != [[sid, f"{arm}/{sid}.png"] for sid in sorted(truth)]:
            raise ValueError("Unexpected cache arm")
        identity = cache["identity"]
        expected = {p: h for p, h in hashes.items() if p.startswith(arm + "/")}
        base, adapter, prompt, precision, from_arm = json.loads(identity["model_key"])
        signature = (base, identity["prefix_sha256"], identity["processor_size"])
        if (not base or (adapter, prompt, precision, from_arm) !=
                (witness["adapter_sha256"], witness["prompt_sha256"], "bf16", arm)
                or identity["image_sha256"] != expected or identity["batch_size"] != 1
                or identity["max_new_tokens"] != 620 or identity["processor_size"]["longest_edge"] != 393216
                or previous_signature is not None and signature != previous_signature):
            raise ValueError("ROI cache weights, prompt, images or generation conditions differ")
        previous_signature = signature
        outputs[arm] = {}
        stats = Counter()
        for sid, rec in indexed.items():
            gold = truth[sid]["source"]["codes"]
            code = truth[sid]["family"]
            if rec["vlm_schema_errors"] or rec["parse_error"]:
                outputs[arm][sid] = None
                stats["invalid"] += 1
                stats["invalid_positive"] += code in gold
                stats["invalid_negative"] += code not in gold
                stats["invalid_normal"] += not gold
                continue
            codes = {h["code"] for h in rec["parsed"]["hazards"]}
            outputs[arm][sid] = codes
            stats["valid"] += 1
            stats["target_tp"] += code in gold and code in codes
            stats["target_fp"] += code not in gold and code in codes
            stats["target_fn"] += code in gold and code not in codes
            stats["normal_any_alert"] += not gold and bool(codes)
        def f1(tp, fp, fn):
            return 2 * tp / max(1, 2 * tp + fp + fn)
        tp, fp, fn = (stats[k] for k in ("target_tp", "target_fp", "target_fn"))
        totals[arm] = dict(stats, valid_target_f1=f1(tp, fp, fn), all_48_target_f1_bounds=[
            f1(tp, fp + stats["invalid_negative"], fn + stats["invalid_positive"]),
            f1(tp + stats["invalid_positive"], fp, fn)],
            all_normal_alert_bounds=[stats["normal_any_alert"], stats["normal_any_alert"] + stats["invalid_normal"]])
    comparisons = {}
    for arm in ARMS[1:]:
        strata = {}
        for sid, r in truth.items():
            left, right = outputs["fixed"][sid], outputs[arm][sid]
            code, gold = r["family"], r["source"]["codes"]
            for stratum in ("all", obs[sid], code):
                c = strata.setdefault(stratum, Counter())
                c["n"] += 1
                if left is None or right is None:
                    c["invalid_pair"] += 1
                    continue
                c["recovered_tp"] += code in gold and code not in left and code in right
                c["lost_tp"] += code in gold and code in left and code not in right
                c["new_fp"] += code not in gold and code not in left and code in right
                c["removed_fp"] += code not in gold and code in left and code not in right
                c["new_normal_alert"] += not gold and not left and bool(right)
                c["removed_normal_alert"] += not gold and bool(left) and not right
        comparisons[arm] = strata
    return dict(photos=48, arms=totals, compared_to_fixed=comparisons,
                normal_denominator=sum(not r["source"]["codes"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=ROOT.parent)
    args = parser.parse_args()
    print(json.dumps(score(args.caches, args.panel), ensure_ascii=False, indent=2))
