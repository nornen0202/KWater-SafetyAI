"""Local-only provenance audit and conservative source-block evaluation splits."""

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageOps

from .audit import Components, inspect_bytes
from .metrics import CODES, decode_gold, unique_keys


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf8",
    )
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tie(s, seed):
    return hashlib.sha256(f"{seed}/{s}".encode()).hexdigest()


def source_block(sample):
    m = re.fullmatch(r"(S\d+)C\d+", sample)
    if m:
        return m[1]
    m = re.fullmatch(r"(stage\d+):.+|(ADD|SBE|SBC|CNEW)\d+", sample)
    if m:
        return m[1] or m[2]
    raise ValueError(f"Unknown source convention: {sample}")


def source_family(sid):
    block = source_block(sid)
    return "S*C*" if re.fullmatch(r"S\d+", block) else block


def balanced_folds(group_ids, y, k, seed, families=None):
    groups = np.array(group_ids)
    sizes = Counter(group_ids)
    if len(sizes) < k:
        raise ValueError("Insufficient source groups for fold count")
    strata = [y, y.sum(1) == 0]
    if families is not None:
        family = np.asarray(families)
        strata.extend(family == key for key in sorted(set(family)))
    aug = np.column_stack([*strata, np.ones(len(y))])
    counts = {g: aug[groups == g].sum(0) for g in sizes}
    total = np.maximum(aug.sum(0), 1)
    target = total / k
    weights = 1 / total**2
    weights[-1] *= 8
    loads = np.zeros((k, len(total)))
    assignment = {}
    order = sorted(
        sizes,
        key=lambda g: (-np.sum((counts[g] / total) ** 2), -sizes[g], tie(g, seed)),
    )
    for pos, g in enumerate(order):
        allowed = [pos] if pos < k else range(k)
        f = min(
            allowed,
            key=lambda i: float(
                np.sum(weights * (2 * (loads[i] - target) * counts[g] + counts[g] ** 2))
            ),
        )
        assignment[g] = f
        loads[f] += counts[g]
    return np.array([assignment[g] for g in groups])


def reserve(groups, eligible, target, seed):
    sizes = Counter(groups)
    # Subset sum, label-blind. Prefer covering several source blocks at the same size.
    states = {(0, 0): ()}
    for g in sorted(eligible, key=lambda x: tie(x, seed)):
        for (n, k), old in list(states.items()):
            size = n + sizes[g]
            if size <= target + max(sizes.values()):
                states.setdefault((size, k + 1), old + (g,))
    choices = [(key, v) for key, v in states.items() if v and len(sizes) - len(v) >= 5]
    if not choices:
        raise ValueError("No eligible unexposed block for holdout")
    _, chosen = min(
        choices,
        key=lambda v: (abs(v[0][0] - target), abs(v[0][1] - 4), tie(str(v[1]), seed)),
    )
    return set(chosen)


def geom_pair(cv, a, b):
    pa, da = a
    pb, db = b
    if da is None or db is None or min(len(da), len(db)) < 12:
        return 0, 0.0, 0.0, False
    matches = cv.BFMatcher(cv.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < 0.70 * pair[1].distance
    ]
    if len(good) < 10:
        return 0, 0.0, 0.0, False
    left = np.float32([pa[m.queryIdx] for m in good])
    right = np.float32([pb[m.trainIdx] for m in good])
    _, mask = cv.findHomography(left, right, cv.RANSAC, 0.012)
    if mask is None:
        return 0, 0.0, 0.0, False
    keep = mask.ravel().astype(bool)
    areas = (
        [cv.contourArea(cv.convexHull(points[keep])) for points in (left, right)]
        if keep.any()
        else [0]
    )
    hits, ratio, area = int(keep.sum()), float(keep.mean()), float(min(areas))
    return hits, ratio, area, hits >= 22 and ratio >= 0.65 and area >= 0.08


def semantic_quarantine(project, run, rows, features, seed):
    """Audit cross-source neighbours without labels. Merge visually registered pairs conservatively."""
    import cv2 as cv
    from functools import lru_cache

    cv.setNumThreads(1)
    cv.setRNGSeed(seed)
    tok = features[:, 0, 0].astype(np.float32)
    left, right = np.split(tok, 2, axis=-1)
    sims = (left @ left.T + right @ right.T) / 2
    ids = [r["sample_id"] for r in rows]
    original = np.array([r["group"] for r in rows])
    sims[original[:, None] == original[None, :]] = -2
    pairs = set()
    for i in range(len(rows)):
        for j in np.argsort(-sims[i], kind="stable")[:3]:
            if sims[i, j] >= 0.72:
                pairs.add(tuple(sorted((i, int(j)))))
    detector = cv.ORB_create(650)

    @lru_cache(maxsize=1200)
    def descriptors(i):
        with Image.open(project / "data/train" / rows[i]["image"]) as im:
            img = ImageOps.exif_transpose(im).convert("L")
            img.thumbnail((600, 600))
            kp, des = detector.detectAndCompute(np.asarray(img), None)
            return (
                (np.float32([p.pt for p in kp]) / np.float32(img.size), des)
                if kp
                else ([], None)
            )

    comps = Components(set(original.tolist()))
    audit = []
    for i, j in sorted(pairs):
        n, ratio, area, match = geom_pair(cv, descriptors(i), descriptors(j))
        if match:
            comps.join(original[i], original[j])
        audit.append(
            dict(
                left=ids[i],
                right=ids[j],
                cosine=float(sims[i, j]),
                inliers=n,
                ratio=ratio,
                area=area,
                decision="quarantine" if match else "unresolved",
            )
        )
    groups = [comps.find(g) for g in original]
    sealed = np.array([r["dev_fold"] < 0 for r in rows])
    hold = {g for g, flag in zip(groups, sealed) if flag}
    write_json(run / "semantic-pairs.json", audit)
    if any(g in hold and not flag for g, flag in zip(groups, sealed)):
        raise ValueError(
            "A newly registered pair crosses the sealed holdout. Review semantic-pairs.json before fitting."
        )
    y = np.array([[c in r["codes"] for c in CODES] for r in rows], float)
    dev = [i for i, g in enumerate(groups) if g not in hold]
    family = np.asarray([source_family(sid) for sid in ids])
    folds = balanced_folds([groups[i] for i in dev], y[dev], 5, seed, family[dev])
    dev_folds = dict(zip(dev, folds.tolist()))
    for i, r in enumerate(rows):
        r.update(group=groups[i], dev_fold=dev_folds.get(i, -1))
        r.pop("all_fold")
    write_json(run / "evaluation-manifest.json", rows)
    summary = {
        "groups": len(set(groups)),
        "candidate_pairs": len(pairs),
        "quarantine_pairs": sum(r["decision"] == "quarantine" for r in audit),
        "holdout_groups": len(hold),
        "holdout_photos": len(rows) - len(dev),
        "exposure_overlap": sum(r["exposed"] and r["dev_fold"] < 0 for r in rows),
    }
    for key in ("dev_fold",):
        buckets = defaultdict(set)
        for r in rows:
            buckets[r["group"]].add(r[key])
        assert all(len(v) == 1 for v in buckets.values())
        data = []
        for f in sorted({r[key] for r in rows}):
            if f < 0:
                continue
            idx = [i for i, r in enumerate(rows) if r[key] == f]
            data.append(
                dict(
                    fold=f,
                    groups=len({groups[i] for i in idx}),
                    photos=len(idx),
                    negative=int((y[idx].sum(1) == 0).sum()),
                    support=y[idx].sum(0).astype(int).tolist(),
                )
            )
        summary[key] = data
    assert summary["exposure_overlap"] == 0
    write_json(run / "evaluation-split.json", summary)
    print(
        "FINAL_GROUPS",
        summary["groups"],
        summary["holdout_photos"],
        summary["quarantine_pairs"],
        flush=True,
    )
    return summary


def prepare(project, out, seed, reviewed=None):
    import cv2 as cv
    import io

    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    cv.setNumThreads(1)
    cv.setRNGSeed(seed)
    base = project / "data/train"
    label_path = base / "train.jsonl"
    audit = json.loads((project / "results/train_audit_full.json").read_bytes())
    exposure = json.loads((project / "data/holdout-exclusions.json").read_bytes())
    assert sha(label_path) == exposure["labels_sha256"] == audit["labels_sha256"]
    rows = [
        json.loads(line, object_pairs_hook=unique_keys)
        for line in label_path.read_bytes().splitlines()
    ]
    rows.sort(key=lambda r: r["sample_id"])
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(rows) or len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Duplicate ID or image path")
    verified = {}
    if reviewed:
        with reviewed.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if not r["group_id"].strip() or r["sample_id"] in verified:
                    raise ValueError("Invalid reviewed incident table")
                verified[r["sample_id"]] = r["group_id"]
        if set(verified) != set(ids):
            raise ValueError("Reviewed groups must cover all photos")
    blocks, pixels, orb = defaultdict(list), defaultdict(list), {}
    color, dhash, records = [], [], []
    detector = cv.ORB_create(650)
    for row in rows:
        sid, codes = decode_gold(row)
        raw = (base / row["image"]).read_bytes()
        obs = inspect_bytes(raw)
        old = audit["inventory"][sid]
        if (
            any(obs[k] != old[k] for k in ("sha256", "pixels_sha256", "dhash64"))
            or set(old["codes"]) != codes
        ):
            raise ValueError(f"Official audit mismatch: {sid}")
        block = verified.get(sid, source_block(sid))
        blocks[block].append(sid)
        pixels[obs["pixels_sha256"]].append(sid)
        dhash.append(int(obs["dhash64"], 16))
        with Image.open(io.BytesIO(raw)) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
            img.thumbnail((600, 600))
            ar = np.asarray(img.convert("L"))
            kp, des = detector.detectAndCompute(ar, None)
            orb[sid] = (
                (np.float32([p.pt for p in kp]) / np.float32(img.size), des)
                if kp
                else ([], None)
            )
            small = np.asarray(
                img.resize((20, 20)).convert("RGB"), dtype=np.float32
            ).ravel()
            small -= small.mean()
            color.append(small / max(np.linalg.norm(small), 1e-8))
        records.append(
            dict(
                sample_id=sid,
                image=row["image"],
                sha256=obs["sha256"],
                source=block,
                codes=sorted(codes),
                status=row["target"]["status"],
            )
        )
    comps = Components(ids)
    for seq in [*blocks.values(), *pixels.values()]:
        for other in seq[1:]:
            comps.join(seq[0], other)
    sims = np.stack(color) @ np.stack(color).T
    np.fill_diagonal(sims, -2)
    shortlist = set()
    for i in range(len(ids)):
        for j in np.argsort(-sims[i], kind="stable")[:5]:
            shortlist.add(tuple(sorted((i, int(j)))))
    for i, j in itertools.combinations(range(len(ids)), 2):
        if (dhash[i] ^ dhash[j]).bit_count() <= 12:
            shortlist.add((i, j))
    checks = []
    for i, j in sorted(shortlist):
        a, b = ids[i], ids[j]
        n, ratio, area, match = geom_pair(cv, orb[a], orb[b])
        if match:
            comps.join(a, b)
        checks.append(
            dict(
                left=a,
                right=b,
                hamming=(dhash[i] ^ dhash[j]).bit_count(),
                thumbnail_cosine=float(sims[i, j]),
                inliers=n,
                inlier_fraction=ratio,
                coverage=area,
                decision="quarantine_same_visual_source" if match else "unresolved",
            )
        )
    clusters = defaultdict(list)
    for sid in ids:
        clusters[comps.find(sid)].append(sid)
    groups = {s: min(members) for members in clusters.values() for s in members}
    exposed = set(exposure["exposed_ids"])
    eligible = set(groups.values()) - {groups[k] for k in exposed}
    hold = reserve(list(groups.values()), eligible, 125, seed)
    y = np.array([[c in r["codes"] for c in CODES] for r in records], float)
    dev = [i for i, s in enumerate(ids) if groups[s] not in hold]
    folds = balanced_folds([groups[ids[i]] for i in dev], y[dev], 5, seed)
    assignments = dict(zip(dev, folds.tolist()))
    all_folds = balanced_folds([groups[s] for s in ids], y, 5, seed)
    for i, r in enumerate(records):
        r.update(
            group=groups[r["sample_id"]],
            dev_fold=assignments.get(i, -1),
            all_fold=int(all_folds[i]),
            exposed=r["sample_id"] in exposed,
        )
    assert not any(r["exposed"] and r["dev_fold"] < 0 for r in records)
    for axis in ("all_fold", "dev_fold"):
        group_folds = defaultdict(set)
        for r in records:
            group_folds[r["group"]].add(r[axis])
        assert all(len(s) == 1 for s in group_folds.values())
    write_json(out / "pairs.json", checks)
    write_json(out / "manifest.json", records)
    summary = dict(
        seed=seed,
        verified_incident_ids=bool(verified),
        group_definition="reviewed incident + visual quarantine"
        if verified
        else "conservative whole source-ID blocks + exact pixels + strong cross-image geometry; actual incident boundaries unknown",
        labels_sha256=sha(label_path),
        manifest_sha256=sha(out / "manifest.json"),
        pairs_sha256=sha(out / "pairs.json"),
        reviewed_sha256=sha(reviewed) if reviewed else None,
        source_blocks=len(blocks),
        groups=len(clusters),
        photos=len(ids),
        hazard=int((y.sum(1) > 0).sum()),
        support=dict(zip(CODES, y.sum(0).astype(int).tolist())),
        heldout=len(ids) - len(dev),
        holdout_groups=len(hold),
        exposure_count=len(exposed),
        identical_pixel_sets=sum(len(v) > 1 for v in pixels.values()),
        geometry_candidates=len(checks),
        geometry_links=sum(c["decision"] != "unresolved" for c in checks),
        cross_source_geometry_links=sum(
            c["decision"] != "unresolved"
            and records[ids.index(c["left"])]["source"]
            != records[ids.index(c["right"])]["source"]
            for c in checks
        ),
        elapsed_seconds=time.monotonic() - start,
    )
    for axis in ("all_fold", "dev_fold"):
        entries = []
        for f in sorted({r[axis] for r in records}):
            sub = [i for i, r in enumerate(records) if r[axis] == f]
            entries.append(
                dict(
                    fold=f,
                    photos=len(sub),
                    groups=len({records[i]["group"] for i in sub}),
                    no_hazard=int((y[sub].sum(1) == 0).sum()),
                    support=y[sub].sum(0).astype(int).tolist(),
                )
            )
        summary[axis] = entries
    write_json(out / "data-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reviewed", type=Path)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    prepare(a.project, a.out, a.seed, a.reviewed)
