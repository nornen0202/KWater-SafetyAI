"""Freeze group-disjoint Fold 4 and Fold 2 Clean-Cal inputs, without training."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from .fold2_v3_epoch1 import MASK as FOLD2_MASK, OOF as FOLD2_OOF, split as fold2_split
from .sub78_fold2_readiness import LINKS
from .sub78_tail_replay import FOLD2, FOLD4, LABELS, replay_curriculum
from .sub78_verifier_audit import CODES, DEV, INNER, LOCAL, digest, load, read_selected_cases, write

SEED = "kwater-clean-cal-v1-20261128"
WORK = Path("data/local-reports/sub78-clean-cal-v2")
REPORT = Path("results/sub78-clean-cal-design-20261128.json")
OLD_GROUP_SEED = 42
THRESHOLDS = {"verify_pos": 0.35, "verify_neg_hard": 0.42, "verify_neg_safe": 0.40}


def keys(rows):
    return ({r["sample_id"] for r in rows}, {r["group_id"] for r in rows})


def partition(train, outer):
    ids, groups = keys(train)
    outer_ids, outer_groups = keys(outer)
    if len(ids) != len(train) or len(outer_ids) != len(outer):
        raise ValueError("Duplicate photo ID")
    if ids & outer_ids or groups & outer_groups:
        raise ValueError("Outer fold overlaps training")
    ordered = sorted(groups, key=lambda group: (sha256(f"{SEED}/{group}".encode()).digest(), group))
    count = (15 * len(ordered) + 50) // 100
    selected = set(ordered[:count])
    fit = [r for r in train if r["group_id"] not in selected]
    calibration = [r for r in train if r["group_id"] in selected]
    fit_ids, fit_groups = keys(fit)
    cal_ids, cal_groups = keys(calibration)
    if (fit_ids & cal_ids or fit_groups & cal_groups or fit_ids | cal_ids != ids
            or fit_groups | cal_groups != groups or not fit or not calibration):
        raise ValueError("Clean-Cal fit/holdout separation failed")
    return fit, calibration


def known_aliases(canonical):
    """Combine confirmed ledgers and exact photo SHA into group components."""
    aliases = {row["group_id"]: row["group_id"] for row in canonical}

    def connect(left, right):
        a, b = sorted((aliases[left], aliases[right]))
        if a != b:
            return {group: a if old == b else old for group, old in aliases.items()}
        return aliases

    edges = set()
    for path, outer_key in LINKS:
        ledger = load(path)
        if ledger["canonical_manifest_sha256"] != digest(DEV):
            raise ValueError("Confirmed link ledger belongs to a different manifest")
        for row in ledger["confirmed_cross_group_links"]:
            edge = tuple(sorted((row[outer_key], row["train_group_id"])))
            if edge[0] != edge[1]:
                edges.add(edge)
                aliases = connect(*edge)
    first_group = {}
    exact_edges = set()
    for row in canonical:
        first = first_group.setdefault(row["sha256"], row["group_id"])
        if first != row["group_id"]:
            edge = tuple(sorted((first, row["group_id"])))
            exact_edges.add(edge)
            aliases = connect(*edge)
    return aliases, edges, exact_edges


def promote_linked_groups(train, fit, calibration, outer, aliases):
    fit_ids, fit_groups = keys(fit)
    _, cal_groups = keys(calibration)
    _, outer_groups = keys(outer)
    def components(groups):
        return {aliases[g] for g in groups}
    cross = components(fit_groups) & components(cal_groups)
    if components(cal_groups) & components(outer_groups):
        raise ValueError("Known calibration/outer alias needs a separate scope decision")
    keep_cal = components(cal_groups)
    fit = [row for row in train if aliases[row["group_id"]] not in keep_cal]
    calibration = [row for row in train if aliases[row["group_id"]] in keep_cal]
    _, new_fit_groups = keys(fit)
    _, new_cal_groups = keys(calibration)
    if components(new_fit_groups) & components(new_cal_groups):
        raise ValueError("Known visual component crosses fit/calibration")
    return fit, calibration, {
        "v1_cross_components": len(cross),
        "groups_moved_from_fit_to_calibration": len(fit_groups - new_fit_groups),
        "photos_moved_from_fit_to_calibration": len(fit_ids - keys(fit)[0]),
        "v2_fit_calibration_component_intersection": 0,
        "v2_fit_outer_component_intersection": len(components(new_fit_groups) & components(outer_groups)),
        "v2_calibration_outer_component_intersection": len(components(new_cal_groups) & components(outer_groups)),
    }


def legacy_exposure(cur, pairs, train, oof_path):
    """Audit actual legacy allocations without using a fresh ranking."""
    inspect = {r["image"] for r in cur if r["kind"] == "inspect"}
    cal_groups = {r["group_id"] for r in pairs}
    inspect_exposed = [r for r in pairs if r["image"] in inspect]
    by_image = {row["image"]: row for row in train}
    verifier = [r for r in cur if r["kind"] in THRESHOLDS]
    verifier_overlap = [r for r in verifier if by_image[r["image"]]["group_id"] in cal_groups]
    with np.load(oof_path, allow_pickle=False) as oof:
        pos = {sid: i for i, sid in enumerate(oof["train_ids"].tolist())}
        values = oof["code_scores"]
    if set(pos) != {row["sample_id"] for row in train}:
        raise ValueError("Legacy specialist OOF scope differs")
    below = []
    for r in verifier_overlap:
        row = by_image[r["image"]]
        stem, candidate = r["sample_id"].split("__ver_", 1)
        code = candidate.split("_", 1)[1]
        if stem != row["sample_id"] or float(values[pos[stem], CODES.index(code)]) >= THRESHOLDS[r["kind"]]:
            raise ValueError("Unexpected threshold routing")
        below.append(r)
    clean_of_verifier_groups = {by_image[r["image"]]["group_id"] for r in verifier}
    no_shared_verify_group = [r for r in pairs if r["group_id"] not in clean_of_verifier_groups]
    return {
        "pairs": len(pairs), "photos": len({r["image"] for r in pairs}), "groups": len(cal_groups),
        "pair_kind_counts": dict(Counter(r["kind"] for r in pairs)),
        "calibration_inspect_exposed_pairs": len(inspect_exposed),
        "calibration_inspect_exposed_photos": len({r["image"] for r in inspect_exposed}),
        "verified_training_rows_from_legacy_calibration_groups_after_sampling": len(below),
        "verified_training_photos_from_legacy_calibration_groups": len({r["image"] for r in below}),
        "shared_verify_groups": len({by_image[r["image"]]["group_id"] for r in below}),
        "verified_training_below_calibration_threshold_by_kind": dict(Counter(r["kind"] for r in below)),
        "after_dropping_shared_verify_groups": {
            "pairs": len(no_shared_verify_group), "photos": len({r["image"] for r in no_shared_verify_group}),
            "groups": len({r["group_id"] for r in no_shared_verify_group}),
            "safe_negative_pairs": sum(r["kind"] == "verify_neg_safe" for r in no_shared_verify_group),
            "inspect_exposed_photos": len({r["image"] for r in no_shared_verify_group if r["image"] in inspect}),
        },
    }


def support(rows, labels):
    ids, groups = keys(rows)
    code_support = {code: {"positive_photos": 0, "severity4_photos": 0, "positive_groups": set()} for code in CODES}
    normal_groups, severe_groups = set(), set()
    normal = severe = 0
    for row in rows:
        target = labels[row["sample_id"]]
        hazards = target["hazards"]
        if set(hazards) != set(row["codes"]) or target["status"] != row["status"]:
            raise ValueError("Canonical labels differ")
        if not hazards:
            normal += 1
            normal_groups.add(row["group_id"])
        has_severe = False
        for c, hazard in hazards.items():
            entry = code_support[c]
            entry["positive_photos"] += 1
            entry["positive_groups"].add(row["group_id"])
            if hazard["risk_level"]["severity"]["score"] == 4:
                entry["severity4_photos"] += 1
                has_severe = True
        if has_severe:
            severe += 1
            severe_groups.add(row["group_id"])
    return {
        "photos": len(ids), "groups": len(groups), "normal_photos": normal, "normal_groups": len(normal_groups),
        "severity4_photos": severe, "severity4_groups": len(severe_groups), "other_hazard_photos": len(ids) - normal - severe,
        "max_normal_fp_for_20pct": normal // 5, "max_normal_fp_for_18pct": 18 * normal // 100,
        "per_code": {c: {**v, "positive_groups": len(v["positive_groups"])} for c, v in code_support.items()},
    }


def write_frozen(path, payload):
    if path.exists() and load(path) != payload:
        raise ValueError(f"Refusing to replace a changed freeze: {path}")
    if not path.exists():
        write(path, payload)


def run():
    canonical = load(DEV)
    aliases, curated_edges, exact_edges = known_aliases(canonical)
    train2, outer2 = fold2_split()
    fold4_mask_path = LOCAL / "fold4-group-exclusions-local.json"
    fold4_mask = load(fold4_mask_path)
    if fold4_mask["canonical_manifest_sha256"] != digest(DEV):
        raise ValueError("Fold 4 review mask scopes another canonical manifest")
    cases = {
        4: ([r for r in canonical if r["split"] == "dev" and r["dev_fold"] != 4],
            [r for r in canonical if r["split"] == "dev" and r["dev_fold"] == 4],
            FOLD4, FOLD4 / "training_summary.json", fold4_mask_path,
            {r["train_group_id"] for r in fold4_mask["confirmed_cross_group_links"]}),
        2: (train2, outer2, FOLD2, FOLD2 / "frozen-recipe.json", FOLD2_MASK,
            set(load(FOLD2_MASK)["known_linked_outer_train_group_ids_for_review"])),
    }
    result = {}
    WORK.mkdir(parents=True, exist_ok=True)
    for fold, (train, outer, original, legacy_recipe, mask, outer_linked_train_groups) in cases.items():
        # The allocation is completed before reading any label or score arrays.
        v1_fit, v1_calibration = partition(train, outer)
        fit, calibration, component_stats = promote_linked_groups(train, v1_fit, v1_calibration, outer, aliases)
        fit_ids, fit_groups = keys(fit)
        _, cal_groups = keys(calibration)
        train_ids, _ = keys(train)
        outer_ids, _ = keys(outer)
        if not outer_linked_train_groups <= fit_groups | cal_groups:
            raise ValueError("Known link witness points outside the source train pool")
        folder = WORK / f"fold-{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        input_files = {
            "fit": folder / "fit-manifest.json", "calibration": folder / "calibration-manifest.json",
            "outer": folder / "outer-manifest.json",
        }
        for name, rows in (("fit", fit), ("calibration", calibration), ("outer", outer)):
            write_frozen(input_files[name], rows)
        contract = {
            "fold": fold, "selection_seed": SEED, "inputs": {k: digest(p) for k, p in input_files.items()},
            "fit_only": ["VLM inspection", "verifier training", "expert fitting", "expert train OOF",
                         "negative mining", "law case corpus", "oversampling", "hyperparameter selection"],
            "fit_oof_requires_group_disjoint_inner_folds": True,
            "score_calibration_only_with_fit_fitted_vlm_expert_and_retriever": True,
            "legacy_train_scores_not_eligible": True,
            "outer_never_for_calibration": True,
            "common_comparator_and_candidate_fitting_manifest": "fit-manifest.json",
            "confirmed_link_ledger_sha256": {p.as_posix(): digest(p) for p, _ in LINKS},
            "known_visual_components_must_stay_together_in_specialist_oof": True,
            "status": "input freeze only; no component has been refit",
        }
        write_frozen(folder / "contract.json", contract)
        labels = {r["sample_id"]: r["target"] for r in read_selected_cases(LABELS, train_ids | outer_ids)}
        if labels.keys() != train_ids | outer_ids:
            raise ValueError("Missing selected official labels")
        cur, pairs, original_sha = replay_curriculum(original, load(legacy_recipe))
        result[str(fold)] = {
            "original_pool": support(train, labels), "clean_fit": support(fit, labels),
            "new_calibration": support(calibration, labels), "unchanged_outer": support(outer, labels),
            "candidate_comparator_fit_shared_photos": len(fit_ids),
            "known_visual_component_audit": component_stats,
            "v1_group_hash_only_fit": support(v1_fit, labels),
            "v1_group_hash_only_calibration": support(v1_calibration, labels),
            "documented_outer_link_train_groups": {
                "still_in_fit": len(fit_groups & outer_linked_train_groups),
                "reserved_for_calibration": len(cal_groups & outer_linked_train_groups),
            },
            "original_curriculum_sha256": original_sha, "legacy": legacy_exposure(cur, pairs, train, INNER if fold == 4 else FOLD2_OOF),
            "manifest_sha256": contract["inputs"], "contract_sha256": digest(folder / "contract.json"),
            "review_mask_sha256": digest(mask),
        }
    report = {
        "scope": "CPU input freeze only. Counts are official-label support, not mined candidates, calibration coverage or model scores.",
        "algorithm": "First nearest-integer 15% of outer-train groups by salted SHA-256. Promote the entire known visual component to calibration if one of its groups was selected. Preserve canonical row order.",
        "confirmed_review_link_count": len(curated_edges), "exact_cross_group_sha_link_count": len(exact_edges),
        "seed": SEED, "legacy_seed": OLD_GROUP_SEED, "legacy_calibration_thresholds": THRESHOLDS,
        "canonical_manifest_sha256": digest(DEV), "folds": result,
    }
    write(REPORT, report)
    return report


def verify_fit_witness(fold, witness):
    """Check declared component memberships against the frozen clean boundaries."""
    folder = WORK / f"fold-{fold}"
    contract = load(folder / "contract.json")
    fit, calibration, outer = (
        load(folder / f"{name}-manifest.json") for name in ("fit", "calibration", "outer")
    )
    for name, expected in contract["inputs"].items():
        if digest(folder / f"{name}-manifest.json") != expected:
            raise ValueError(f"Frozen {name} manifest changed")
    if witness["contract_sha256"] != digest(folder / "contract.json"):
        raise ValueError("Witness must identify this exact fit contract")
    fit_ids, _ = keys(fit)
    cal_ids, _ = keys(calibration)
    outer_ids, _ = keys(outer)
    if any(digest(Path(name)) != expected for name, expected in contract["confirmed_link_ledger_sha256"].items()):
        raise ValueError("Known visual links changed")
    aliases, _, _ = known_aliases(load(DEV))
    group_by_id = {r["sample_id"]: aliases[r["group_id"]] for r in fit}

    def check_ids(values, allowed, name, *, exact=False):
        actual = set(values)
        if len(actual) != len(values) or not actual <= allowed or exact and actual != allowed:
            raise ValueError(f"{name} differs from the allowed photo IDs")
        return actual

    if set(witness["arms"]) != {"comparator", "candidate"}:
        raise ValueError("Both paired arms must report membership")
    for name, arm in witness["arms"].items():
        for component in ("inspect", "specialist_fit"):
            check_ids(arm["fit_components"][component], fit_ids, f"{name}/{component}", exact=True)
        for component in ("verifier_train", "negative_mining", "law_cases", "oversampling", "hyperparameter_selection"):
            check_ids(arm["fit_components"][component], fit_ids, f"{name}/{component}")
        check_ids(arm["calibration_predict"], cal_ids, f"{name}/calibration_predict", exact=True)
        check_ids(arm["outer_predict"], outer_ids, f"{name}/outer_predict", exact=True)
        covered = set()
        for inner in arm["specialist_train_oof"]:
            train_ids = check_ids(inner["fit"], fit_ids, f"{name}/oof_fit")
            pred_ids = check_ids(inner["predict"], fit_ids, f"{name}/oof_predict")
            if (not train_ids or not pred_ids or covered & pred_ids
                    or {group_by_id[sid] for sid in train_ids} & {group_by_id[sid] for sid in pred_ids}):
                raise ValueError("Specialist OOF has missing or overlapping groups")
            covered.update(pred_ids)
        if covered != fit_ids:
            raise ValueError("Specialist OOF must predict every fit photo exactly once")
    return {"fold": fold, "arms_checked": 2, "fit_photos": len(fit_ids), "calibration_photos": len(cal_ids),
            "outer_photos": len(outer_ids), "scope": "Declared memberships only; requires component-generated receipts."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-witness", type=Path)
    parser.add_argument("--fold", type=int, choices=(2, 4))
    args = parser.parse_args()
    if args.check_witness is None:
        if args.fold is not None:
            parser.error("--fold applies to --check-witness only")
        run()
    else:
        if args.fold is None:
            parser.error("--check-witness requires --fold")
        print(verify_fit_witness(args.fold, load(args.check_witness)))
