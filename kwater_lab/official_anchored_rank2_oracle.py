"""Read-only rank-two oracle for the same 102 originally proposed photos."""

from __future__ import annotations

import argparse
from ast import literal_eval
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import struct
from zipfile import ZipFile

from .fold0_production_compare import fold_rows, image_id
from .metrics import CODES, decode_target, evaluate, read_index


MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
EXCLUSIONS = Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")
FROZEN = Path("data/local-eval-predictions/fold0/v2-official-anchored/frozen_proposals.json")
VERIFIED_REPORT = Path("results/v2_official_anchored_verified.json")
OFFICIAL_OUTPUT = Path("data/fold0-qlora-v2-118-official/output/submission.jsonl")
SCORES = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
COMMON = tuple(code for code in CODES if code not in ("drowning_inundation", "confined_space", "thermal_weather"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    return json.loads(path.read_text(encoding="utf-8"))


def score_payload(archive, name, dtype, expected_shape):
    data = archive.read(name + ".npy")
    if data[:8] != b"\x93NUMPY\x01\x00":
        raise ValueError("Expected the frozen NPY v1 archive")
    end = 10 + int.from_bytes(data[8:10], "little")
    info = literal_eval(data[10:end].decode("ascii"))
    if info != {"descr": dtype, "fortran_order": False, "shape": expected_shape}:
        raise ValueError(f"Unexpected out-of-fold score shape or dtype: {name}")
    return data[end:]


def ranked_proposals(rows, baseline, frozen):
    with ZipFile(SCORES) as archive:
        count = 966
        key_bytes = score_payload(archive, "ids", "<U11", (count,))
        sample_ids = [key_bytes[i:i + 44].decode("utf-32-le").rstrip("\x00") for i in range(0, len(key_bytes), 44)]
        scores = list(struct.iter_unpack("<11f", score_payload(archive, "code_probs", "<f4", (count, len(CODES)))))
        scene = list(struct.iter_unpack("<f", score_payload(archive, "scene_probs", "<f4", (count,))))
        folds = list(struct.iter_unpack("<q", score_payload(archive, "folds", "<i8", (count,))))
    index = {sid: i for i, sid in enumerate(sample_ids)}
    if len(index) != count or not (len(scores) == len(scene) == len(folds) == count):
        raise ValueError("Invalid frozen specialist score coverage")
    result = {}
    reconstructed = []
    for row_index, row in enumerate(rows):
        sid = row["sample_id"]
        idx = index[sid]
        if folds[idx] != (0,):
            raise ValueError("Expected Fold0 out-of-fold probability for every photo")
        original = baseline[sid]["hazards"]
        if len(original) >= 3 or (not original and scene[idx][0] < 0.58):
            continue
        eligible = [(code, probability) for code, probability in zip(CODES, scores[idx], strict=True)
                    if code not in original and probability >= (0.54 if original else 0.62)]
        ordered = sorted(eligible, key=lambda item: -item[1])
        if any(left[1] == right[1] for left, right in zip(ordered[:2], ordered[1:3])):
            raise ValueError("A top-two score tie requires the original selector's tie-breaking")
        ranks = [code for code, _ in ordered[:2]]
        if ranks:
            result[sid] = tuple(ranks)
            reconstructed.append((sid, row["image"], row_index, ranks[0]))
    expected = [(r["sample_id"], r["image"], r["row_idx"], r["code"]) for r in frozen]
    if len(expected) != 102 or reconstructed != expected or any(len(set(v)) != len(v) for v in result.values()):
        raise ValueError("Rank one does not match the frozen photo/code order or rank two repeats a code")
    return result


def arm_metrics(gold, predicted, groups, severe_gold):
    result = evaluate(gold, predicted, groups)
    result["fixed_8_common_macro_f1"] = mean(result["per_code"][c]["f1"] or 0.0 for c in COMMON)
    severe = {sid: severe_gold[sid] for sid in gold if severe_gold[sid]}
    result["severity4"] = {"photos": len(severe), "groups": len({groups[sid] for sid in severe}),
                           "complete_misses": sum(not (codes & predicted[sid]) for sid, codes in severe.items())}
    return result


def cohort_diagnostic(rows, baseline, ranked, severe):
    groups = {row["sample_id"]: row["group_id"] for row in rows}
    gold = {row["sample_id"]: frozenset(row["codes"]) for row in rows}
    original = {sid: frozenset(baseline[sid]["hazards"]) for sid in groups}
    first_recovery, rank2_recovery = {}, {}
    leftover_inside, leftover_outside, extra = Counter(), Counter(), Counter()
    available_second, conditional_second = 0, 0
    for sid, expected in gold.items():
        codes = ranked.get(sid, ())
        missing = expected - original[sid]
        first = {codes[0]} & missing if codes else set()
        second = set()
        if len(codes) == 2:
            available_second += 1
            if not first:
                conditional_second += 1
                second = {codes[1]} & missing
        first_recovery[sid] = first
        rank2_recovery[sid] = first | second
        extra.update(second)
        (leftover_inside if codes else leftover_outside).update(missing - first)
    proposed_photos = sum(sid in ranked for sid in gold)
    baseline_fn = sum(len(gold[sid] - original[sid]) for sid in gold)
    first_count = sum(map(len, first_recovery.values()))
    top2_count = sum(map(len, rank2_recovery.values()))
    apply = lambda added: {sid: original[sid] | added[sid] for sid in gold}
    return {"photos": len(groups), "groups": len(set(groups.values())), "original_proposed_photos": proposed_photos,
            "official_code_fn": baseline_fn, "rank1_recovered_code_fn": first_count, "rank2_oracle_recovered_code_fn": top2_count,
            "rank2_additional_recovery_by_code": dict(extra), "rank1_residual_inside_proposal_photos": dict(leftover_inside),
            "rank1_residual_outside_proposal_photos": dict(leftover_outside),
            "rank1_residual_inside_total": sum(leftover_inside.values()), "rank1_residual_outside_total": sum(leftover_outside.values()),
            "rank2_existing_policy_additional_candidate_slots": available_second,
            "max_additional_physical_calls": available_second, "oracle_conditional_additional_calls": conditional_second,
            "mean_candidates_per_original_proposed_photo": 1 + available_second / proposed_photos,
            "official": arm_metrics(gold, original, groups, severe),
            "rank1_label_oracle": arm_metrics(gold, apply(first_recovery), groups, severe),
            "rank2_label_oracle": arm_metrics(gold, apply(rank2_recovery), groups, severe)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen_bytes = digest(FROZEN)
    frozen = record(FROZEN)
    exact_sources = {"manifest_sha256": MANIFEST, "group_mask_sha256": EXCLUSIONS,
                     "official_predictions_sha256": OFFICIAL_OUTPUT, "oof_scores_sha256": SCORES}
    if (record(VERIFIED_REPORT)["proposal_file_sha256"] != frozen_bytes
        or any(frozen["fingerprint"].get(key) != digest(path) for key, path in exact_sources.items())):
        raise ValueError("Input bytes differ from the original proposal witness")
    rows = fold_rows(MANIFEST)
    index = read_index(OFFICIAL_OUTPUT, lambda row: (decode_target(row)[0], row), limit_size=True)
    expected = {image_id(row["sample_id"]): row["sample_id"] for row in rows}
    if list(index) != list(expected):
        raise ValueError("Official rows do not cover the frozen cohort in order")
    baseline = {expected[key]: value for key, value in index.items()}
    ranked = ranked_proposals(rows, baseline, frozen["proposals"])
    mask = record(EXCLUSIONS)
    if mask["canonical_manifest_sha256"] != digest(MANIFEST):
        raise ValueError("Group mask does not match the original manifest")
    excluded = {link["fold0_group_id"] for link in mask["confirmed_cross_group_links"]}
    ids = set(expected.values())
    severe = {}
    with Path("data/train/train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            leader = re.match(r'^\s*\{\s*"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")', line)
            if leader and json.loads(leader[1]) in ids:
                row = json.loads(line)
                severe[row["sample_id"]] = {code for code, h in row["target"]["hazards"].items()
                                            if h["risk_level"]["severity"]["score"] == 4}
    if severe.keys() != ids:
        raise ValueError("Severity labels are missing for the frozen cohort")
    retained = [r for r in rows if r["group_id"] not in excluded]
    if len(retained) != 34 or len({r["group_id"] for r in retained}) != 34:
        raise ValueError("Expected the previously frozen 34-photo sensitivity")
    full = cohort_diagnostic(rows, baseline, ranked, severe)
    sensitivity = cohort_diagnostic(retained, baseline, ranked, severe)
    report = {"scope": "Retrospective label oracle; no new training, generation, scoring threshold or deployment change",
              "frozen_rank1_proposals_sha256": frozen_bytes, "reconstructed_rank1_exact": True,
              "rank1_reconstruction_fields": ["sample_id", "image", "row_idx", "code"],
              "original_photo_code_rank_has_duplicates": False, "actual_new_verifier_calls": 0,
              "full_fold0": full, "confirmed_group_exclusion": sensitivity}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
