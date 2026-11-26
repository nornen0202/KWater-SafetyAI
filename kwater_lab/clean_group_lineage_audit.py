"""Reconstruct development group linkage without allocating any new split."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .audit import Components, digest
from .clean_split import _numeric_order_key, initial_prefix_group
from .train_head import save_atomic


DEV = Path("data/clean-split-v1/split_manifest_dev.json")
INVENTORY = Path("results/train_audit_full.json")
PHASE01 = Path("data/phase01-runs/phase01-20261126-0513-2155e8fa")
PAIR_FILES = [PHASE01 / filename for filename in ("pairs.json", "semantic-pairs.json")]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def grouped(keys, selector):
    result = defaultdict(list)
    for key in keys:
        result[selector(key)].append(key)
    return result


def run(root):
    manifest = load(root / DEV)
    if any(r["split"] != "dev" for r in manifest):
        raise ValueError("Development manifest required")
    rows = {row["sample_id"]: row for row in manifest}
    inventory = load(root / INVENTORY)["inventory"]
    ids = sorted(rows)
    components = Components(ids)
    current_size = {sid: 1 for sid in ids}
    edges = []

    def join(left, right, mechanism, evidence=None):
        a, b = components.find(left), components.find(right)
        if a == b:
            return
        sizes = sorted((current_size[a], current_size[b]))
        components.join(left, right)
        current_size[components.find(left)] = sum(sizes)
        edges.append((left, right, mechanism, sizes, evidence))

    prefix = grouped(ids, initial_prefix_group)
    exact = grouped(ids, lambda sid: inventory[sid]["pixels_sha256"])
    for key, unions in (("initial_numeric_prefix", prefix), ("exact_pixel_sha256", exact)):
        for members in unions.values():
            for other in members[1:]:
                join(members[0], other, key)
    for path in PAIR_FILES:
        for pair in load(root / path):
            left, right = pair["left"], pair["right"]
            if left not in rows or right not in rows:
                continue
            inliers = int(pair.get("inliers") or 0)
            ratio = float(pair.get("inlier_fraction", pair.get("ratio", 0)) or 0)
            evidence = {"inliers": inliers, "inlier_fraction": ratio, "prior_decision": pair.get("decision")}
            if pair.get("decision") in ("quarantine_same_visual_source", "quarantine"):
                join(left, right, f"{path.name}:recorded_quarantine", evidence)
            elif inliers >= 16 and ratio >= 0.18:
                join(left, right, f"{path.name}:geometric_threshold", evidence)
    dhash = {sid: int(inventory[sid]["dhash64"], 16) for sid in ids}
    order = {sid: _numeric_order_key(sid) for sid in ids}
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            distance = (dhash[left] ^ dhash[right]).bit_count()
            if distance <= 6:
                join(left, right, "dhash_within_6")
            elif (distance <= 12 and order[left] and order[right]
                    and order[left][0] == order[right][0] and abs(order[left][1] - order[right][1]) <= 2):
                join(left, right, "dhash_within_12_near_consecutive_id")
    canonical = grouped(ids, lambda sid: rows[sid]["group_id"])
    replay = grouped(ids, components.find)
    if {frozenset(g) for g in canonical.values()} != {frozenset(g) for g in replay.values()}:
        raise ValueError("Linkage replay differs from canonical development partition")
    f0 = grouped((sid for sid in ids if rows[sid]["dev_fold"] == 0), lambda sid: rows[sid]["group_id"])
    largest = set(max(f0.values(), key=len))
    source_groups = defaultdict(Counter)
    for sid, row in rows.items():
        source_groups[row["family"]][row["group_id"]] += 1
    fold_results = {}
    for fold in range(5):
        this = [sid for sid in ids if rows[sid]["dev_fold"] == fold]
        groups = Counter(rows[sid]["group_id"] for sid in this)
        count = max(groups.values())
        fold_results[str(fold)] = {
            "photos": len(this), "groups": len(groups), "largest_group_photos": count,
            "largest_group_photo_share": count / len(this),
        }
    mechanisms = (
        "initial_numeric_prefix", "exact_pixel_sha256", "pairs.json:recorded_quarantine",
        "pairs.json:geometric_threshold", "semantic-pairs.json:recorded_quarantine",
        "semantic-pairs.json:geometric_threshold", "dhash_within_6", "dhash_within_12_near_consecutive_id",
    )
    totals = Counter(edge[2] for edge in edges)
    largest_totals = Counter(kind for left, right, kind, _, _ in edges if left in largest and right in largest)
    return {
        "scope": "Development-only reconstruction of existing links; no new fold assignment",
        "input_sha256": {key.as_posix(): digest(root / key) for key in (DEV, INVENTORY, *PAIR_FILES)},
        "photos": len(ids), "groups": len(canonical),
        "successful_union_by_mechanism": {kind: totals[kind] for kind in mechanisms},
        "by_source": {
            source: {"photos": sum(counts.values()), "groups": len(counts),
                     "top_five_group_sizes_within_source": sorted(counts.values(), reverse=True)[:5]}
            for source, counts in sorted(source_groups.items())
        },
        "by_fold": fold_results,
        "largest_fold0_group": {
            "photos": len(largest), "source_photo_counts": dict(Counter(rows[sid]["family"] for sid in largest)),
            "initial_prefix_group_sizes": sorted(Counter(initial_prefix_group(sid) for sid in largest).values(), reverse=True),
            "internal_union_by_mechanism": {kind: largest_totals[kind] for kind in mechanisms},
            "post_prefix_union_sequence": [
                {"mechanism": kind, "premerge_component_sizes": sizes, "geometric_evidence": evidence}
                for left, right, kind, sizes, evidence in edges
                if left in largest and right in largest and kind != "initial_numeric_prefix"
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/clean-development-group-lineage.json"))
    args = parser.parse_args()
    output = (json.dumps(run(args.root), indent=2) + "\n").encode("utf-8")
    save_atomic(args.output, lambda stream: stream.write(output))


if __name__ == "__main__":
    main()
