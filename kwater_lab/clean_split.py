"""Reconstruct the historical grouped development and reserved-panel split.

Groups are inferred from S<n> prefixes, exact pixels, cached geometric pair
links, and perceptual hashes. The split checks the original exposure list and
known group boundaries. It does not account for later Phase-01 exposure and
must not be used to construct a fresh independent holdout.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np

from .audit import Components, digest
from .metrics import CODES, decode_gold, unique_keys
from .train_head import save_atomic


SEED = 20261127
PHASE01_RUN = Path("data/phase01-runs/phase01-20261126-0513-2155e8fa")
PHASE01_SIG = "phase01-r2-sig384-dino224-grid4-proj64-seed404-tiles60pct-training-jpeg"


def _tie_hash(text: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def source_family(sample_id: str) -> str:
    if re.fullmatch(r"S\d+C\d+", sample_id):
        return "S_C"
    if sample_id.startswith("ADD"):
        return "ADD"
    if sample_id.startswith("SBE"):
        return "SBE"
    if sample_id.startswith("CNEW"):
        return "CNEW"
    if sample_id.startswith("SBC"):
        return "SBC"
    if ":" in sample_id:
        return sample_id.split(":", 1)[0]
    return "OTHER"


def initial_prefix_group(sample_id: str) -> str:
    """Keep S<n>C<m> grouped by S<n>; fine-grained curated IDs start per-photo before visual merging."""
    m = re.fullmatch(r"(S\d+)C\d+", sample_id)
    if m:
        return m.group(1)
    return sample_id


def _numeric_order_key(sample_id: str) -> tuple[str, int] | None:
    m = re.fullmatch(r"(ADD|SBE|CNEW|SBC)(\d+)", sample_id)
    if m:
        return m.group(1), int(m.group(2))
    m = re.fullmatch(r"(stage\d+:[A-Z]+)(\d+)", sample_id)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _load_cached_global_cosine(rows: list[dict], root: Path) -> np.ndarray | None:
    feat_dir = root / PHASE01_RUN / "features"
    if not feat_dir.is_dir():
        return None
    vecs = []
    for r in rows:
        fname = hashlib.sha256((PHASE01_SIG + r["sha256"]).encode()).hexdigest() + ".npy"
        fpath = feat_dir / fname
        if not fpath.is_file():
            return None
        arr = np.load(fpath, allow_pickle=False)
        vecs.append(arr[0, 0].astype(np.float32))
    mat = np.stack(vecs, axis=0)
    left, right = np.split(mat, 2, axis=-1)
    left /= np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-8)
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)
    return 0.5 * (left @ left.T + right @ right.T)


def build_clean_split(root: Path, out_dir: Path, summary_path: Path, seed: int = SEED) -> dict:
    labels_path = root / "data/train/train.jsonl"
    audit_path = root / "results/train_audit_full.json"
    excl_path = root / "data/holdout-exclusions.json"

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    excl = json.loads(excl_path.read_text(encoding="utf-8"))
    labels_sha = digest(labels_path)
    if labels_sha != audit["labels_sha256"] or labels_sha != excl["labels_sha256"]:
        raise ValueError("train.jsonl SHA-256 mismatch against audit or exclusions")

    raw_rows = [
        json.loads(line, object_pairs_hook=unique_keys)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_rows.sort(key=lambda r: r["sample_id"])

    records = []
    for row in raw_rows:
        sid, codes = decode_gold(row)
        inv = audit["inventory"][sid]
        if set(inv["codes"]) != codes:
            raise ValueError(f"Code mismatch for {sid}")
        records.append({
            "sample_id": sid,
            "image": row["image"],
            "sha256": inv["sha256"],
            "pixels_sha256": inv["pixels_sha256"],
            "dhash64": inv["dhash64"],
            "width": inv["width"],
            "height": inv["height"],
            "status": row["target"]["status"],
            "codes": sorted(codes),
            "family": source_family(sid),
        })

    ids = [r["sample_id"] for r in records]
    idx_of = {sid: i for i, sid in enumerate(ids)}
    comps = Components(ids)

    # 1. Conservative S<n> prefix grouping + exact pixel hash grouping
    by_init = defaultdict(list)
    by_pix = defaultdict(list)
    for r in records:
        by_init[initial_prefix_group(r["sample_id"])].append(r["sample_id"])
        by_pix[r["pixels_sha256"]].append(r["sample_id"])
    for members in (*by_init.values(), *by_pix.values()):
        for other in members[1:]:
            comps.join(members[0], other)

    # 2. Incorporate quarantine decisions and geometric-threshold links from prior candidate files.
    geom_links = 0
    for pair_file in (root / PHASE01_RUN / "pairs.json", root / PHASE01_RUN / "semantic-pairs.json"):
        if pair_file.is_file():
            pdata = json.loads(pair_file.read_text(encoding="utf-8"))
            for item in pdata:
                inliers = int(item.get("inliers", 0) or 0)
                frac = float(item.get("inlier_fraction", item.get("ratio", 0.0)) or 0.0)
                is_geom = item.get("decision") in ("quarantine_same_visual_source", "quarantine") or (
                    inliers >= 16 and frac >= 0.18
                )
                if is_geom:
                    a, b = item["left"], item["right"]
                    if a in idx_of and b in idx_of and comps.find(a) != comps.find(b):
                        comps.join(a, b)
                        geom_links += 1

    # 3. Strict perceptual near-duplicate (dHash64 <= 6) + consecutive-ID same-shot merging
    #    (Do NOT use Phase-01's buggy 64-dim random Gaussian projection features, which chain unrelated sites!)
    dhashes = [int(r["dhash64"], 16) for r in records]
    visual_merges = 0
    n = len(records)
    num_keys = [_numeric_order_key(sid) for sid in ids]

    for i in range(n):
        for j in range(i + 1, n):
            if comps.find(ids[i]) == comps.find(ids[j]):
                continue
            ham = (dhashes[i] ^ dhashes[j]).bit_count()
            same_seq_near = (
                num_keys[i] is not None
                and num_keys[j] is not None
                and num_keys[i][0] == num_keys[j][0]
                and abs(num_keys[i][1] - num_keys[j][1]) <= 2
                and ham <= 12
            )
            if ham <= 6 or same_seq_near:
                comps.join(ids[i], ids[j])
                visual_merges += 1

    cluster_members = defaultdict(list)
    for sid in ids:
        cluster_members[comps.find(sid)].append(sid)
    canonical_group = {
        sid: f"GRP_{min(members)}"
        for members in cluster_members.values()
        for sid in members
    }
    groups_map = defaultdict(list)
    for i, r in enumerate(records):
        g = canonical_group[r["sample_id"]]
        r["group_id"] = g
        groups_map[g].append(i)

    exposed_ids = set(excl["exposed_ids"])
    exposed_groups = {canonical_group[sid] for sid in exposed_ids}

    y_all = np.array([[c in r["codes"] for c in CODES] for r in records], dtype=np.float64)
    total_pos = y_all.sum(axis=0)
    singleton_codes = {i for i, tot in enumerate(total_pos) if tot <= 1}

    # 4. Select the historical reserved panel against the original exclusions.
    target_holdout = 105
    target_code_holdout = 0.10 * total_pos
    max_code_holdout = np.where(total_pos >= 10, np.ceil(0.16 * total_pos), np.where(total_pos >= 3, 1.0, 0.0))

    eligible_groups = []
    for g, idxs in sorted(groups_map.items(), key=lambda kv: _tie_hash(kv[0], seed)):
        if g in exposed_groups:
            continue
        if len(idxs) > 18:
            continue  # Avoid single large prefix monopolizing holdout
        g_pos = y_all[idxs].sum(axis=0)
        if any(g_pos[c] > 0 for c in singleton_codes):
            continue  # Never starve 1-instance classes from training
        eligible_groups.append(g)

    holdout_groups: set[str] = set()
    hold_count = 0
    hold_pos = np.zeros(len(CODES), dtype=np.float64)
    hold_no_haz = 0
    target_no_haz = round(0.10 * sum(r["status"] == "no_visible_hazard" for r in records))
    fam_counts_total = Counter(r["family"] for r in records)
    hold_fam = Counter()

    remaining = set(eligible_groups)
    while remaining and hold_count < target_holdout:
        best_g = None
        best_score = float("inf")
        for g in sorted(remaining, key=lambda x: _tie_hash(x, seed + hold_count)):
            idxs = groups_map[g]
            if hold_count + len(idxs) > target_holdout + 6:
                continue
            g_pos = y_all[idxs].sum(axis=0)
            if np.any(hold_pos + g_pos > max_code_holdout):
                continue
            fams = Counter(records[i]["family"] for i in idxs)
            if any(hold_fam[f] + c > max(2, round(0.18 * fam_counts_total[f])) for f, c in fams.items()):
                continue
            new_pos = hold_pos + g_pos
            new_nh = hold_no_haz + sum(records[i]["status"] == "no_visible_hazard" for i in idxs)
            new_cnt = hold_count + len(idxs)
            # Normalized squared distance to ideal 10% stratified ratios
            active_mask = total_pos >= 10
            rel_err = ((new_pos[active_mask] - target_code_holdout[active_mask]) / np.maximum(target_code_holdout[active_mask], 1.0)) ** 2
            nh_err = ((new_nh - target_no_haz) / max(target_no_haz, 1.0)) ** 2
            size_pen = abs(new_cnt - target_holdout) * 0.015
            score = float(rel_err.mean() + 0.5 * nh_err + size_pen)
            if score < best_score:
                best_score = score
                best_g = g
        if best_g is None:
            break
        holdout_groups.add(best_g)
        remaining.remove(best_g)
        idxs = groups_map[best_g]
        hold_count += len(idxs)
        hold_pos += y_all[idxs].sum(axis=0)
        hold_no_haz += sum(records[i]["status"] == "no_visible_hazard" for i in idxs)
        for i in idxs:
            hold_fam[records[i]["family"]] += 1

    # 5. Stratified Multi-Label + Source-Family 5-Fold Assignment on Dev (~940 photos)
    dev_groups = sorted(set(groups_map) - holdout_groups)
    k_folds = 5
    fams_unique = sorted({r["family"] for r in records})
    fam_to_col = {f: i for i, f in enumerate(fams_unique)}

    group_vecs = {}
    group_sizes = {}
    for g in dev_groups:
        idxs = groups_map[g]
        group_sizes[g] = len(idxs)
        c_vec = y_all[idxs].sum(axis=0)
        nh = float(sum(records[i]["status"] == "no_visible_hazard" for i in idxs))
        f_vec = np.zeros(len(fams_unique), dtype=np.float64)
        for i in idxs:
            f_vec[fam_to_col[records[i]["family"]]] += 1.0
        group_vecs[g] = np.concatenate([c_vec, [nh], f_vec, [float(len(idxs))]])

    dev_tot = np.maximum(sum(group_vecs.values()), 1.0)
    fold_target = dev_tot / k_folds
    col_weights = 1.0 / (dev_tot ** 2)
    col_weights[:len(CODES)] *= 4.0  # Prioritize 11-code hazard balance across folds
    col_weights[-1] *= 6.0           # Keep fold photo sizes balanced

    # Order rarest-code / largest-impact groups first
    ordered_dev_groups = sorted(
        dev_groups,
        key=lambda g: (
            -float(np.sum((group_vecs[g][:len(CODES)] / dev_tot[:len(CODES)]) ** 2)),
            -group_sizes[g],
            _tie_hash(g, seed),
        ),
    )

    fold_loads = np.zeros((k_folds, len(dev_tot)), dtype=np.float64)
    group_to_fold: dict[str, int] = {}
    for g in ordered_dev_groups:
        v = group_vecs[g]
        best_f = min(
            range(k_folds),
            key=lambda f: (
                float(np.sum(col_weights * (2.0 * (fold_loads[f] - fold_target) * v + v ** 2))),
                fold_loads[f, -1],
                f,
            ),
        )
        group_to_fold[g] = int(best_f)
        fold_loads[best_f] += v

    # Integrity checks
    full_manifest = []
    dev_only_manifest = []
    for i, r in enumerate(records):
        g = r["group_id"]
        is_hold = g in holdout_groups
        if is_hold and r["sample_id"] in exposed_ids:
            raise AssertionError(f"Leakage: exposed ID {r['sample_id']} assigned to holdout")
        fold = -1 if is_hold else group_to_fold[g]
        entry = {
            "sample_id": r["sample_id"],
            "image": r["image"],
            "sha256": r["sha256"],
            "group_id": g,
            "family": r["family"],
            "split": "holdout" if is_hold else "dev",
            "dev_fold": fold,
            "exposed_prior": r["sample_id"] in exposed_ids,
            "status": r["status"],
            "codes": r["codes"],
        }
        full_manifest.append(entry)
        if not is_hold:
            dev_only_manifest.append(entry)

    # Verify zero group crossing across dev folds & holdout
    grp_splits = defaultdict(set)
    for row in full_manifest:
        grp_splits[row["group_id"]].add((row["split"], row["dev_fold"]))
    if any(len(v) != 1 for v in grp_splits.values()):
        raise AssertionError("Incident group crossed splits or folds!")

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "split_manifest_full_sealed.json"
    dev_path = out_dir / "split_manifest_dev.json"
    csv_path = out_dir / "reviewed_groups.csv"

    save_atomic(full_path, lambda f: f.write((json.dumps(full_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    save_atomic(dev_path, lambda f: f.write((json.dumps(dev_only_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    csv_lines = ["sample_id,group_id"] + [f"{r['sample_id']},{r['group_id']}" for r in full_manifest]
    save_atomic(csv_path, lambda f: f.write(("\n".join(csv_lines) + "\n").encode("utf-8")))

    def _subset_stats(rows_sub: list[dict]) -> dict:
        c_cnt = Counter(c for r in rows_sub for c in r["codes"])
        f_cnt = Counter(r["family"] for r in rows_sub)
        return {
            "photos": len(rows_sub),
            "groups": len({r["group_id"] for r in rows_sub}),
            "hazard_photos": sum(r["status"] == "hazard" for r in rows_sub),
            "no_hazard_photos": sum(r["status"] == "no_visible_hazard" for r in rows_sub),
            "families": dict(sorted(f_cnt.items())),
            "codes": {c: c_cnt[c] for c in CODES},
        }

    summary = {
        "version": "clean-split-v1",
        "seed": seed,
        "labels_sha256": labels_sha,
        "exclusions_sha256": digest(excl_path),
        "dev_manifest_sha256": digest(dev_path),
        "sealed_manifest_sha256": digest(full_path),
        "reviewed_groups_csv_sha256": digest(csv_path),
        "total_photos": len(full_manifest),
        "total_incident_groups": len(groups_map),
        "geometric_cross_links_applied": geom_links,
        "visual_similarity_merges_applied": visual_merges,
        "used_cached_dual_encoder_similarity": False,
        "holdout_exposed_overlap": 0,
        "cross_split_group_violations": 0,
        "dev_summary": _subset_stats(dev_only_manifest),
        "holdout_summary": _subset_stats([r for r in full_manifest if r["split"] == "holdout"]),
        "dev_folds": {
            str(f): _subset_stats([r for r in dev_only_manifest if r["dev_fold"] == f])
            for f in range(k_folds)
        },
    }
    save_atomic(summary_path, lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean-split-v1"))
    parser.add_argument("--summary", type=Path, default=Path("results/clean_split_v1_summary.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    summary = build_clean_split(args.root.resolve(), args.out_dir.resolve(), args.summary.resolve(), args.seed)
    print(
        f"Clean split ready: total={summary['total_photos']} groups={summary['total_incident_groups']} "
        f"dev={summary['dev_summary']['photos']} holdout={summary['holdout_summary']['photos']}"
    )
    print("Dev codes:", summary["dev_summary"]["codes"])
    print("Holdout codes:", summary["holdout_summary"]["codes"])
    for f, st in summary["dev_folds"].items():
        print(f"  Fold {f}: n={st['photos']} groups={st['groups']} no_haz={st['no_hazard_photos']} codes={st['codes']}")


if __name__ == "__main__":
    main()
