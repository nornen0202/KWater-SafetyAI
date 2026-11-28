"""Summarize documented development exposure and freeze a prospective Fold 2 recipe."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

DATA = Path("data")
RESULTS = Path("results")
MANIFEST = DATA / "clean-split-v1/split_manifest_dev.json"
PREFIX = DATA / "prefix-pilot/exposure.json"
HOLDOUT = DATA / "holdout-exclusions.json"
LABELS = DATA / "train/train.jsonl"
MAPS = [
    (DATA / "fold0-crosssplit-visual-review/mapping-local.json", "pair_mapping"),
    (DATA / "fold0-crosssplit-visual-review-wave2/mapping-local.json", "pair_mapping"),
    (DATA / "fold4-crosssplit-visual-review/mapping-local.json", "pairs"),
]
LINKS = [
    (DATA / "fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json", "fold0_group_id"),
    (DATA / "fold4-crosssplit-visual-review/fold4-group-exclusions-local.json", "fold4_group_id"),
]
ROI = DATA / "vision-roi-48/key-local.json"
ROI_AGG = DATA / "vision-roi-48/aggregate.json"
CURATION_PAIRS = DATA / "curation-side-20261127/real-pairs.json"
PRIOR_OOF = DATA / "phase01-runs/phase01-20261126-0513-2155e8fa/dev_fold-oof.jsonl"
QUAL = RESULTS / "fold4-qualitative-panel-20261127.json"
SPECIALIST = RESULTS / "clean_specialist_v1_oof.json"
MAJORITY = re.compile(r'^\s*\{\s*"sample_id"\s*:\s*("(?:[^"\\]|\\.)*")')


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def annotations(ids):
    selected = {}
    with LABELS.open(encoding="utf-8") as stream:
        for line in stream:
            match = MAJORITY.match(line)
            if not match:
                raise ValueError("Expected sample_id as leading key")
            sid = json.loads(match[1])
            if sid in ids:
                if sid in selected:
                    raise ValueError("Duplicate selected official label")
                selected[sid] = json.loads(line)["target"]["hazards"]
    if selected.keys() != ids:
        raise ValueError("Missing selected official labels")
    return selected


def summarize():
    data = read(MANIFEST)
    manifest_sha = digest(MANIFEST)
    if any(row["split"] != "dev" for row in data):
        raise ValueError("Use the unsealed development manifest")
    by_id = {r["sample_id"]: r for r in data}
    group_members = {}
    for row in data:
        group_members.setdefault(row["group_id"], []).append(row)
    fold2 = {sid for sid, row in by_id.items() if row["dev_fold"] == 2}
    gold = annotations(fold2)
    if any(set(gold[sid]) != set(by_id[sid]["codes"]) for sid in fold2):
        raise ValueError("Official Fold 2 code mismatch")

    def groups(ids):
        return {by_id[sid]["group_id"] for sid in ids if sid in by_id}

    def measure(ids):
        severe = Counter(code for sid in ids for code, item in gold[sid].items()
                         if item["risk_level"]["severity"]["score"] == 4)
        normal = sum(not gold[sid] for sid in ids)
        return {"photos": len(ids), "groups": len(groups(ids)), "normal": normal,
                "hazard_photos": len(ids) - normal, "severity4_photos": sum(
                    any(item["risk_level"]["severity"]["score"] == 4 for item in gold[sid].values()) for sid in ids),
                "severity4_code_instances": severe.total(), "max_normal_false_alarms_at_18pct": 18*normal//100}

    def joined(ids):
        exact = fold2 & ids
        related = {sid for sid in fold2 if by_id[sid]["group_id"] in groups(ids)}
        return {"listed_ids": len(ids), "exact": measure(exact), "including_manifest_group": measure(related),
                "unjoinable_to_dev_ids": len(ids - by_id.keys())}, related

    primary = measure(fold2)
    if (primary["photos"], primary["groups"], primary["normal"]) != (197, 38, 67):
        raise ValueError("Unexpected frozen Fold 2 denominator")
    pilot = read(PREFIX)
    holdout = read(HOLDOUT)
    if pilot["labels_sha256"] != holdout["labels_sha256"] or digest(PREFIX) != holdout["pilot_exposure_sha256"]:
        raise ValueError("Pilot/holdout exposure lineage mismatch")
    audit = {}
    for name, ids in {
        "earlier_170_prefix_label_pilot": set(pilot["exposed_ids"]),
        "holdout_exclusion_ledger_union": set(holdout["exposed_ids"]),
        "published_examples": set(holdout["published_example_ids"]),
        "local_kit_candidate_or_other_ledger_entries": set(holdout["exposed_ids"]) - set(pilot["exposed_ids"]) - set(holdout["published_example_ids"]),
        "manifest_exposed_prior": {r["sample_id"] for r in data if r["exposed_prior"]},
    }.items():
        audit[name], _ = joined(ids)

    mapped = set()
    for path, key in MAPS:
        pairs = read(path)[key]
        ids = {row[side]["sample_id"] for row in pairs for side in ("left", "right")}
        mapped |= ids
        audit[path.parent.name + "_visually_selected_pairs"], _ = joined(ids)
    audit["any_crosssplit_audit_sheet_photo"], _ = joined(mapped)
    if read(ROI_AGG)["panel_sha256"] != digest(ROI):
        raise ValueError("Completed ROI panel mapping changed")
    roi_ids = set()
    for entry in read(ROI):
        source = entry["source"]
        sid = source["sample_id"]
        if by_id[sid]["sha256"] != source["sha256"]:
            raise ValueError("ROI source hash and development manifest disagree")
        roi_ids.add(sid)
    audit["completed_visual_roi_48_exact_photo"], roi_related = joined(roi_ids)
    curated_ids = set()
    for pair in read(CURATION_PAIRS):
        for source in pair["members"]:
            sid = source["sample_id"]
            if by_id[sid]["sha256"] != source["sha256"]:
                raise ValueError("Curation source hash and development manifest disagree")
            curated_ids.add(sid)
    audit["official_candidates_in_42_photo_curation_pilot"], curation_related = joined(curated_ids)

    confirmed = set()
    link_details = {}
    for path, outer_side in LINKS:
        info = read(path)
        if info["canonical_manifest_sha256"] != manifest_sha:
            raise ValueError("Known cross-group ledger uses another manifest")
        rows = info["confirmed_cross_group_links"]
        matched = {row[side] for row in rows for side in (outer_side, "train_group_id")} & groups(fold2)
        linked = {r["sample_id"] for g in matched for r in group_members[g] if r["dev_fold"] == 2}
        confirmed |= linked
        link_details[path.parent.name] = measure(linked)
    oof = set()
    with PRIOR_OOF.open(encoding="utf-8") as stream:
        for line in stream:
            sid = json.loads(line)["sample_id"]
            oof.add(sid)
    audit["early_specialist_oof_scored_rows"], _ = joined(oof)
    latest = read(SPECIALIST)
    qualitative = read(QUAL)
    specialist_fold2 = latest["per_fold_summary"]["2"]
    import numpy as np

    prediction_archive = DATA / "clean-specialist-v1/dev_oof_predictions.npz"
    with np.load(prediction_archive, allow_pickle=False) as scores:
        latest_oof = set(scores["ids"].tolist())
    if latest_oof != set(by_id) or specialist_fold2["photos"] != len(fold2):
        raise ValueError("Newer specialist local OOF archive does not cover the development cohort")
    audit["newer_specialist_oof_scored_rows"], _ = joined(latest_oof)
    if (qualitative["cohort"]["photos"], qualitative["panel"]["photos"]) != (107, 24):
        raise ValueError("Unexpected Fold 4 blind panel aggregate scope")
    pilot_risk = {sid for sid in fold2 if by_id[sid]["group_id"] in groups(set(holdout["exposed_ids"]))}
    dropped = pilot_risk | confirmed
    kept = fold2 - dropped
    outer_train = by_id.keys() - fold2
    linked_train_groups = {
        row[side] for path, outer_side in LINKS for row in read(path)["confirmed_cross_group_links"]
        for side, opposite in ((outer_side, "train_group_id"), ("train_group_id", outer_side))
        if row[opposite] in groups(fold2)
    }
    source_files = [MANIFEST, LABELS, PREFIX, HOLDOUT, ROI, ROI_AGG, CURATION_PAIRS, PRIOR_OOF, QUAL, SPECIALIST, prediction_archive,
                    *(p for p, _ in MAPS), *(p for p, _ in LINKS)]
    hashes = {path.as_posix(): digest(path) for path in source_files}
    private = {
        "manifest_sha256": manifest_sha, "source_sha256": hashes,
        "fold2_primary_ids": sorted(fold2), "prospective_mask_retained_ids": sorted(kept),
        "prospective_mask_excluded_ids": sorted(dropped),
        "excluded_due_to_exposure_group_ids": sorted(groups(pilot_risk)),
        "excluded_due_to_confirmed_cross_group_ids": sorted(groups(confirmed)),
        "matched_outer_training_ids_both_models": sorted(outer_train),
        "known_linked_outer_train_group_ids_for_review": sorted(linked_train_groups),
        "plan": "Train fresh 4B and 8B on identical Fold 2 outer-train IDs with locked prompts/curriculum; rebuild legal case bank and vision specialist from those IDs; derive calibration only from grouped inner splits disjoint from inspector and verifier fitting. Report primary 197 and the pre-frozen sensitivity mask; retain photo-level matching and compare runtime separately.",
    }
    output = DATA / "sub78-fold2-prospective-readiness-20261128"
    output.mkdir(exist_ok=True)
    mask_path = output / "private-prospective-mask-and-recipe.json"
    save(mask_path, private)
    return {
        "scope": "Known exposure ledgers joined by exact development ID, content hash or reviewed group only. No filename-prefix inference, sealed Fold 4 24-panel key, withheld labels or photo reads.",
        "primary_197": primary, "known_exposure_by_source": audit, "confirmed_cross_group_by_ledger": link_details,
        "sensitivity_mask": {"retained": measure(kept), "excluded_union": measure(dropped),
                             "holdout_ledger_group_exclusion": measure(pilot_risk),
                             "confirmed_duplicate_group_exclusion": measure(confirmed),
                             "overlap_of_two_exclusion_reasons": measure(pilot_risk & confirmed),
                             "retained_with_documented_roi_photo_observation": measure(kept & roi_ids),
                             "retained_in_roi_observed_manifest_groups": measure(kept & roi_related),
                             "retained_with_documented_curation_pilot_membership": measure(kept & curated_ids),
                             "optional_stricter_visual_exposure_sensitivity_roi_and_curation": measure(kept - roi_related - curation_related),
                             "recipe_sha256": digest(mask_path)},
        "prior_training_distinct_from_label_review": {"fold2_photos_in_prior_fold4_and_full1045_training": len(fold2),
            "fold2_photos_in_same_fixed_earlier_specialist_oof": len(fold2 & oof),
            "newer_specialist_oof_aggregate_has_same_unsealed_manifest_sha": latest["manifest_sha256"] == manifest_sha,
            "newer_specialist_oof_fold2_aggregate_photos": specialist_fold2.get("photos"),
            "newer_specialist_oof_fold2_exact_local_archive_photos": len(latest_oof & fold2),
            "known_rowwise_manual_gold_view_counts": None,
            "fold4_24_blind_panel_exact_fold2_overlap_by_scope": 0,
            "fold4_24_blind_panel_event_overlap_unknown_beyond_reviewed_groups": True},
        "candidate_design": {"fresh_both_same_outer_train_photos": len(outer_train),
                             "fresh_both_same_outer_train_groups": len(groups(outer_train)),
                             "confirmed_related_outer_training_groups_for_review": len(linked_train_groups),
                             "interpretation": "Newly trained VLMs may be outer-fold held out. Prior development exposure prevents describing Fold 2 as pristine. Newly acquired sites or an existing separately governed untouched holdout are needed for stronger confirmation."},
        "unknown_exposures": ["No complete rowwise human annotation-access or tuning ledger.",
                              "Cross-group audit is selective; unmatched similarities and unreviewed site links remain possible.",
                              "Potential sealed-panel related event beyond known Fold 4 scope cannot be inferred from group labels alone.",
                              "OOF scoring processes labels; this does not establish human viewing of individual labels."],
        "known_source_sha256": hashes,
    }


if __name__ == "__main__":
    save(RESULTS / "sub78-fold2-prospective-readiness-20261128.json", summarize())
