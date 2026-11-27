"""Audit Fold 4 cross-split visual candidates on cached features and local photos."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .audit import digest
from .crosssplit_visual_review import build_sheet, group_key
from .train_head import save_atomic


DEV = Path("data/clean-split-v1/split_manifest_dev.json")
FOLD0_REVIEW = Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")
CACHE = Path("data/clean-specialist-v1/features_multiview_2176d.npz")
CACHE_WITNESS = Path("results/clean_specialist_v1_oof.json")
LOCAL = Path("data/fold4-crosssplit-visual-review")
RESULT = Path("results/fold4-crosssplit-audit.json")
LABELS = Path("data/train/train.jsonl")
V3_EVAL = Path("data/local-eval-predictions/fold4-v3-benchmark")
SELECTION = {"top_distinct_group_pairs": 16, "extra_cross_family_pairs": 6, "extra_adjacent_id_group_pairs": 6}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    data = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    save_atomic(path, lambda stream: stream.write(data))


def review_row(row):
    return {k: row[k] for k in ("sample_id", "image", "sha256", "group_id", "dev_fold", "family")}


def cross_cosine(rows, val_count):
    with np.load(CACHE, allow_pickle=False) as cache:
        keys = cache["ids"].tolist()
        index = {sid: i for i, sid in enumerate(keys)}
        if len(index) != len(keys):
            raise ValueError("Duplicate image IDs in cache")
        vectors = cache["features"][[index[row["sample_id"]] for row in rows], 0, :].astype("float32")
    if vectors.shape[1] != 2176 or not np.isfinite(vectors).all():
        raise ValueError("Unexpected frozen encoder vectors")
    result = np.zeros((val_count, len(rows) - val_count), dtype=np.float32)
    for matrix in (vectors[:, :1152], vectors[:, 1152:]):
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        result += 0.5 * (matrix[:val_count] @ matrix[val_count:].T)
    return result


def select_pairs(val, train, scores):
    from .clean_split import _numeric_order_key

    v_keys = [_numeric_order_key(r["sample_id"]) for r in val]
    t_keys = [_numeric_order_key(r["sample_id"]) for r in train]
    shape = scores.shape
    ranked = np.argsort(-scores.ravel(), kind="stable")
    selected = []
    seen = set()
    adjacent = [
        (i, j) for i, a in enumerate(v_keys) for j, b in enumerate(t_keys)
        if a and b and a[0] == b[0] and 0 < abs(a[1] - b[1]) <= 2
    ]

    def append(pairs, amount, route):
        before = len(selected)
        for i, j in pairs:
            left, right = val[i], train[j]
            group = group_key(left, right)
            if group in seen:
                continue
            seen.add(group)
            selected.append({"left": left, "right": right, "cosine": float(scores[i, j]), "route": route})
            if len(selected) - before >= amount:
                break
        return len(selected) - before

    def scored_indexes():
        for k in ranked:
            yield divmod(int(k), shape[1])

    counts = {}
    counts["top_distinct_group_pairs"] = append(scored_indexes(), SELECTION["top_distinct_group_pairs"], "top_cosine")
    counts["extra_cross_family_pairs"] = append(
        ((i, j) for i, j in scored_indexes() if val[i]["family"] != train[j]["family"]),
        SELECTION["extra_cross_family_pairs"], "cross_family",
    )
    adjacent.sort(key=lambda pair: (-float(scores[pair]), pair))
    counts["extra_adjacent_id_group_pairs"] = append(iter(adjacent), SELECTION["extra_adjacent_id_group_pairs"], "adjacent_id")
    return selected, counts, len(adjacent), len({group_key(val[i], train[j]) for i, j in adjacent})


def geometry(pairs):
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261127)
    extractors = {"SIFT": cv2.SIFT_create(nfeatures=1500), "ORB": cv2.ORB_create(nfeatures=1500)}
    matchers = {"SIFT": cv2.BFMatcher(cv2.NORM_L2), "ORB": cv2.BFMatcher(cv2.NORM_HAMMING)}

    @lru_cache(maxsize=36)
    def descriptors(image, expected_sha):
        path = Path("data/train") / image
        if digest(path) != expected_sha:
            raise ValueError("A review photo does not match its audited hash")
        gray = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("A review photo could not be decoded")
        scale = min(1.0, 1000 / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return {name: extractor.detectAndCompute(gray, None) for name, extractor in extractors.items()}, gray.shape

    def filtered(matcher, a, b, ratio):
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            return []
        return [p for pair in matcher.knnMatch(a, b, k=2) if len(pair) == 2
                for p, q in (pair,) if p.distance < ratio * q.distance]

    for pair in pairs:
        (a, a_shape), (b, b_shape) = (descriptors(pair[side]["image"], pair[side]["sha256"]) for side in ("left", "right"))
        evidence = {}
        for name, matcher in matchers.items():
            ak, ad = a[name]
            bk, bd = b[name]
            ratio = 0.75 if name == "SIFT" else 0.80
            reverse = {(m.trainIdx, m.queryIdx) for m in filtered(matcher, bd, ad, ratio)}
            mutual = [m for m in filtered(matcher, ad, bd, ratio) if (m.queryIdx, m.trainIdx) in reverse]
            result = {"keypoints": [len(ak), len(bk)], "mutual_matches": len(mutual),
                      "inliers": 0, "inlier_fraction": 0.0, "hull_area_fractions": [0.0, 0.0]}
            if len(mutual) >= 4:
                pa = np.float32([ak[m.queryIdx].pt for m in mutual])
                pb = np.float32([bk[m.trainIdx].pt for m in mutual])
                matrix, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999)
                if matrix is not None and mask is not None:
                    keep = mask.ravel().astype(bool)
                    count = int(keep.sum())
                    result.update(inliers=count, inlier_fraction=float(keep.mean()))
                    if count >= 3:
                        result["hull_area_fractions"] = [
                            float(cv2.contourArea(cv2.convexHull(points[keep])) / np.prod(shape))
                            for points, shape in ((pa, a_shape), (pb, b_shape))
                        ]
            evidence[name] = result
        pair["geometry"] = evidence
    return cv2.__version__


def prepare(local):
    manifest = load(DEV)
    if any(r["split"] != "dev" for r in manifest):
        raise ValueError("Development-only manifest required")
    val = [review_row(r) for r in manifest if r["dev_fold"] == 4]
    train = [review_row(r) for r in manifest if r["dev_fold"] in (0, 1, 2, 3)]
    v_groups, t_groups = ({r["group_id"] for r in group} for group in (val, train))
    if (len(val) != 175 or len(train) != 791 or len(v_groups) != 39 or v_groups & t_groups
            or load(CACHE_WITNESS)["feature_cache_sha256"] != digest(CACHE)
            or load(FOLD0_REVIEW)["canonical_manifest_sha256"] != digest(DEV)):
        raise ValueError("Existing benchmark or feature provenance changed")
    scores = cross_cosine(val + train, len(val))
    candidates, counts, adjacency, adjacent_groups = select_pairs(val, train, scores)
    cv = geometry(candidates)
    for i, pair in enumerate(candidates, 1):
        pair["number"] = f"{i:02d}"
        # Sheet orientation is deterministic but masks validation/training identity.
        if i % 2 == 0:
            pair["left"], pair["right"] = pair["right"], pair["left"]
    local.mkdir(parents=True, exist_ok=False)
    for i in range(0, len(candidates), 2):
        page = candidates[i:i + 2]
        filename = f"sheet-{page[0]['number']}-{page[-1]['number']}.jpg"
        build_sheet(Path("."), page, local / filename)
        for row in page:
            row["sheet"] = filename
    mapping_path = local / "mapping-local.json"
    write(mapping_path, {"pairs": candidates})
    flat = scores.ravel()
    values = {
        "scope": "All development cross-split cosine comparisons; geometry and image review on a representative fixed shortlist",
        "input_sha256": {p.as_posix(): digest(p) for p in (DEV, CACHE, FOLD0_REVIEW)},
        "validation_photos": len(val), "validation_groups": len(v_groups),
        "training_photos": len(train), "training_groups": len(t_groups),
        "all_cross_photo_pairs": int(flat.size), "all_cross_group_pairs": len(v_groups) * len(t_groups),
        "max_cosine": float(flat.max()), "cosine_quantiles_0_5_0_9_0_99_0_999": np.quantile(flat, [.5, .9, .99, .999]).tolist(),
        "diagnostic_cosine_counts": {str(t): int((flat >= t).sum()) for t in (.8, .85, .9, .95)},
        "adjacent_numeric_id_cross_pairs": adjacency, "adjacent_numeric_id_cross_group_pairs": adjacent_groups,
        "selected_routes": counts, "geometric_photo_pairs": len(candidates),
        "geometric_threshold_pairs": {method: sum(p["geometry"][method]["inliers"] >= 16
                                                   and p["geometry"][method]["inlier_fraction"] >= .18
                                                   for p in candidates) for method in ("SIFT", "ORB")},
        "opencv_version": cv, "mapping_sha256": digest(mapping_path),
        "visual_review_completed": False,
    }
    write(local / "scan-aggregate.json", values)
    print(json.dumps(values, ensure_ascii=True, indent=2))


def finalize(local):
    values = load(local / "scan-aggregate.json")
    mapping = local / "mapping-local.json"
    judgments = local / "judgments-local.json"
    pairs = {p["number"]: p for p in load(mapping)["pairs"]}
    review = load(judgments)
    if values["mapping_sha256"] != digest(mapping) or pairs.keys() != review.keys():
        raise ValueError("Each registered pair must be reviewed exactly once")
    choices = {"same_visible_scene", "possible_same_site_unproven", "different_visible_scene", "unresolved"}
    if any(item["decision"] not in choices or not item["basis"] for item in review.values()):
        raise ValueError("Unrecognized or ungrounded photo-review decision")
    identified = []
    excluded = set()
    for number, item in review.items():
        if item["decision"] != "same_visible_scene":
            continue
        pair = pairs[number]
        val, train = sorted((pair["left"], pair["right"]), key=lambda r: r["dev_fold"], reverse=True)
        identified.append({"review_number": number, "fold4_group_id": val["group_id"], "train_group_id": train["group_id"]})
        excluded.add(val["group_id"])
    mask = local / "fold4-group-exclusions-local.json"
    write(mask, {"canonical_manifest_sha256": digest(DEV), "review_sha256": digest(judgments),
                 "review_status": "one_AI_visual_review_confirmed_links_only",
                 "confirmed_cross_group_links": identified})
    validation = [r for r in load(DEV) if r["dev_fold"] == 4]
    retained = [r for r in validation if r["group_id"] not in excluded]
    values.update(visual_review_completed=True, visual_reviewer_count=1,
                  visual_review_outcome_counts=dict(Counter(x["decision"] for x in review.values())),
                  visual_review_sha256=digest(judgments), confirmed_mask_sha256=digest(mask),
                  confirmed_group_links=len(identified), evaluation_groups_excluded=len(excluded),
                  evaluation_photos_excluded=len(validation) - len(retained),
                  sensitivity_photos=len(retained), sensitivity_groups=len({r["group_id"] for r in retained}),
                  qualification="Inferred grouping and limited image review do not establish complete site independence.")
    write(RESULT, values)


def score(local, arms, output):
    from .fold0_group_sensitivity import arm_metrics
    from .law_reranker_11code import read_selected_cases
    from .metrics import decode_target, paired_bootstrap, read_index

    mask_path = local / "fold4-group-exclusions-local.json"
    mask = load(mask_path)
    audit = load(RESULT)
    if (audit["confirmed_mask_sha256"] != digest(mask_path)
            or mask["canonical_manifest_sha256"] != digest(DEV)):
        raise ValueError("Scoring requires the exact finalized Fold 4 mask and development manifest")
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    primary_gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    primary_groups = {r["sample_id"]: r["group_id"] for r in rows}
    if len(primary_gold) != 175 or len(set(primary_groups.values())) != 39:
        raise ValueError("Expected the original exact 175-photo/39-group cohort")
    predictions = {}
    for name, path in arms.items():
        pred = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
        if pred.keys() != primary_gold.keys():
            raise ValueError(f"{name} must contain every original Fold 4 canonical ID once")
        predictions[name] = pred
    excluded = {row["fold4_group_id"] for row in mask["confirmed_cross_group_links"]}
    retained = {k for k, group in primary_groups.items() if group not in excluded}
    groups = {k: primary_groups[k] for k in retained}
    if len(retained) != audit["sensitivity_photos"] or len(set(groups.values())) != audit["sensitivity_groups"]:
        raise ValueError("The retained photos and groups differ from the audit")
    actual = {sid: primary_gold[sid] for sid in groups}
    severe = {r["sample_id"]: {c for c, hazard in r["target"]["hazards"].items()
                             if hazard["risk_level"]["severity"]["score"] == 4}
              for r in read_selected_cases(Path("data/train/train.jsonl"), retained)}
    if severe.keys() != actual.keys():
        raise ValueError("Missing severity annotations for retained photos")
    predictions = {name: {k: pred[k] for k in groups} for name, pred in predictions.items()}
    comparisons = [("legacy8", "v3")] + ([("official", "v3"), ("official", "legacy8")] if "official" in arms else [])
    write(output, {
        "scope": "Secondary fixed-mask sensitivity. Full exact original coverage was verified in every arm.",
        "manifest_sha256": digest(DEV), "mask_sha256": digest(mask_path),
        "prediction_sha256": {name: digest(path) for name, path in arms.items()},
        "primary_photos": len(primary_gold), "primary_groups": len(set(primary_groups.values())),
        "retained_photos": len(groups), "retained_groups": len(set(groups.values())),
        "arm_metrics": {name: arm_metrics(actual, pred, groups, severe) for name, pred in predictions.items()},
        "paired_group_bootstrap": {
            f"{a}_vs_{b}": paired_bootstrap(actual, predictions[a], predictions[b], groups)
            for a, b in comparisons
        },
    })


def score_three(local, arms, output):
    if set(arms) != {"legacy8", "v3", "hybrid"}:
        raise ValueError("Scoring requires legacy8, v3, and hybrid predictions")
    _score_panels(local, arms, output, "three_arm")


def _score_panels(local, arms, output, evaluation_run, *, score_exploratory=False):
    from itertools import combinations

    from .fold0_group_sensitivity import arm_metrics
    from .law_reranker_11code import read_selected_cases
    from .metrics import CODES, decode_target, evaluate, paired_bootstrap, read_index

    mask_path = local / "fold4-group-exclusions-local.json"
    inputs = (DEV, RESULT, LABELS, mask_path, *arms.values())
    if output is not None and output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Aggregate output must differ from every input")
    mask = load(mask_path)
    audit = load(RESULT)
    manifest_sha = digest(DEV)
    mask_sha = digest(mask_path)
    if (audit["confirmed_mask_sha256"] != mask_sha
            or mask["canonical_manifest_sha256"] != manifest_sha):
        raise ValueError("Scoring requires the exact finalized Fold 4 mask and development manifest")
    rows = [r for r in load(DEV) if r["split"] == "dev" and r["dev_fold"] == 4]
    primary_gold = {r["sample_id"]: frozenset(r["codes"]) for r in rows}
    primary_groups = {r["sample_id"]: r["group_id"] for r in rows}
    if len(rows) != 175 or len(primary_gold) != 175 or len(set(primary_groups.values())) != 39:
        raise ValueError("Expected the original exact 175-photo/39-group cohort")
    predictions = {}
    canonical_order = {}
    prediction_hashes = {}
    for name, path in arms.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} complete Fold 4 prediction file: {path}")
        prediction_hashes[name] = digest(path)
        pred = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
        if digest(path) != prediction_hashes[name]:
            raise ValueError(f"{name} prediction file changed during preflight")
        if pred.keys() != primary_gold.keys():
            raise ValueError(f"{name} must contain every original Fold 4 canonical ID once")
        predictions[name] = pred
        canonical_order[name] = list(pred) == list(primary_gold)
    excluded = {row["fold4_group_id"] for row in mask["confirmed_cross_group_links"]}
    retained = {k: g for k, g in primary_groups.items() if g not in excluded}
    if (len(retained), len(set(retained.values()))) != (107, 36) or (
        audit["sensitivity_photos"], audit["sensitivity_groups"]
    ) != (107, 36):
        raise ValueError("Expected the frozen 107-photo/36-group sensitivity cohort")
    severe = {}
    for row in read_selected_cases(LABELS, set(primary_gold)):
        sid = row["sample_id"]
        target = row["target"]
        if sid in severe:
            raise ValueError("Duplicate official label for a primary photo")
        if set(target["hazards"]) != primary_gold[sid]:
            raise ValueError("Official labels differ from the canonical Fold 4 manifest")
        severe[sid] = {c for c, item in target["hazards"].items() if item["risk_level"]["severity"]["score"] == 4}
    if severe.keys() != primary_gold.keys():
        raise ValueError("Missing severity annotations for primary photos")

    sizes = Counter(retained.values())
    largest = min(sizes, key=lambda group: (-sizes[group], group))
    remaining_groups = {sid: group for sid, group in retained.items() if group != largest}
    cohorts = {"primary_175": primary_groups, "sensitivity_107": retained}
    if score_exploratory:
        cohorts["exploratory_58"] = remaining_groups
    panels = {}
    for panel, groups in cohorts.items():
        gold = {sid: primary_gold[sid] for sid in groups}
        kept = {arm: {sid: pred[sid] for sid in groups} for arm, pred in predictions.items()}
        metrics = {arm: arm_metrics(gold, pred, groups, severe) for arm, pred in kept.items()}
        for arm, pred in kept.items():
            metrics[arm]["severity4"]["code_instances"] = sum(len(severe[sid]) for sid in groups)
            metrics[arm]["severity4"]["missed_code_instances"] = sum(len(severe[sid] - pred[sid]) for sid in groups)
        comparisons = {}
        for a, b in combinations(kept, 2):
            first, second = kept[a], kept[b]
            added = Counter(code for sid in groups for code in second[sid] - first[sid])
            removed = Counter(code for sid in groups for code in first[sid] - second[sid])
            changed = sum(first[sid] != second[sid] for sid in groups)
            comparisons[f"{a}_vs_{b}"] = {
                **paired_bootstrap(gold, first, second, groups),
                "changed_code_set_photos": changed,
                "changed_code_set_photo_rate": changed / len(groups),
                "per_code_change": {
                    c: {"added_by_b": added[c], "removed_by_b": removed[c],
                        "photo_change_rate": (added[c] + removed[c]) / len(groups)}
                    for c in CODES
                },
            }
        panels[panel] = {
            "photos": len(groups), "groups": len(set(groups.values())),
            "arm_metrics": metrics, "comparisons": comparisons,
        }
    remaining = {sid: primary_gold[sid] for sid in remaining_groups}
    exploratory_metrics = {
        arm: evaluate(remaining, {sid: pred[sid] for sid in remaining})
        for arm, pred in predictions.items()
    }
    f1 = {arm: metrics["macro_f1"] for arm, metrics in exploratory_metrics.items()}
    active_codes = {
        arm: {code for code, counts in metrics["per_code"].items() if counts["f1"] is not None}
        for arm, metrics in exploratory_metrics.items()
    }
    panels["sensitivity_107"]["leave_largest_group_out"] = {
        "scope": "Exploratory point estimates after removing the largest retained group",
        "excluded_photos": sizes[largest],
        "photos": len(remaining), "groups": len(sizes) - 1,
        "macro_f1": f1,
        "macro_f1_active_codes": {arm: sorted(codes) for arm, codes in active_codes.items()},
        "macro_f1_codes_lost_after_exclusion": {
            arm: sorted({code for code, counts in metrics["per_code"].items()
                         if counts["f1"] is not None} - active_codes[arm])
            for arm, metrics in panels["sensitivity_107"]["arm_metrics"].items()
        },
        "macro_f1_delta_b_minus_a": {
            f"{a}_vs_{b}": f1[b] - f1[a] if f1[a] is not None and f1[b] is not None else None
            for a, b in combinations(f1, 2)
        },
    }
    result = {
        "scope": "Paired local code scoring. The 107-photo panel removes three confirmed cross-split scene groups.",
        "evaluation_run": evaluation_run,
        "manifest_sha256": manifest_sha, "mask_sha256": mask_sha, "official_labels_sha256": digest(LABELS),
        "prediction_sha256": prediction_hashes,
        "prediction_rows_in_canonical_manifest_order": canonical_order,
        "panels": panels,
    }
    if output is not None:
        write(output, result)
    return result


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("scan", "finalize", "score", "score-three", "score-epoch1"))
    parser.add_argument("--local", type=Path, default=LOCAL)
    parser.add_argument("--legacy8", type=Path, default=V3_EVAL / "clean_8code_baseline.jsonl")
    parser.add_argument("--v3", type=Path, default=V3_EVAL / "v3_11code_standalone.jsonl")
    parser.add_argument("--hybrid", type=Path, default=V3_EVAL / "v3_hybrid_margin_verified.jsonl")
    parser.add_argument("--epoch1", type=Path, help="Standalone v3 epoch-1 complete Fold 4 outputs")
    parser.add_argument("--official", type=Path)
    parser.add_argument("--score-output", type=Path, default=Path("results/fold4-fixed-mask-sensitivity.json"))
    parser.add_argument("--score-three-output", type=Path, default=Path("results/fold4-paired-code-evaluation.json"))
    parser.add_argument("--score-epoch1-output", type=Path, default=Path("results/fold4-v3-epoch1-paired-code-evaluation.json"))
    args = parser.parse_args()
    if not args.local.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Original IDs and review images must stay under data/")
    if args.epoch1 is not None and args.stage != "score-epoch1":
        parser.error("--epoch1 applies to score-epoch1 only")
    if args.stage == "score-epoch1":
        if args.official is not None or args.epoch1 is None:
            parser.error("score-epoch1 requires --epoch1 and does not accept --official")
        if args.legacy8.resolve() == args.epoch1.resolve():
            parser.error("The baseline and epoch-1 output must be distinct files")
        _score_panels(args.local, {"legacy8": args.legacy8, "v3_epoch1": args.epoch1},
                      args.score_epoch1_output, "v3_epoch1_standalone")
    elif args.stage == "score-three":
        if args.official is not None:
            parser.error("--official applies to score only")
        score_three(args.local, {"legacy8": args.legacy8, "v3": args.v3, "hybrid": args.hybrid},
                    args.score_three_output)
    elif args.stage == "score":
        arms = {"legacy8": args.legacy8, "v3": args.v3}
        if args.official is not None:
            arms["official"] = args.official
        score(args.local, arms, args.score_output)
    else:
        (prepare if args.stage == "scan" else finalize)(args.local)
    assert "torch" not in sys.modules
