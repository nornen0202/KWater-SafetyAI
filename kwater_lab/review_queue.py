"""Prepare a local, label-blind visual-review queue from the training audit.

The CSV includes all pairs within the chosen Hamming distance and every
image's nearest neighbor(s), including ties. Incident IDs require independent
review. Export confirmed decisions as left,right,decision for the audit CLI.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path


def prefix(sample_id):
    # A display/filter field only. The source does not define incident IDs.
    return re.sub(r"\d+$", "", sample_id)


def build(inventory, distance):
    if not 0 <= distance <= 64 or len(inventory) < 2:
        raise ValueError("Need at least two images and a bit distance in [0, 64]")
    keys = sorted(inventory)
    hashes = {key: int(inventory[key]["dhash64"], 16) for key in keys}
    nearest = dict.fromkeys(keys, 65)
    for a, b in combinations(keys, 2):
        bits = (hashes[a] ^ hashes[b]).bit_count()
        nearest[a] = min(nearest[a], bits)
        nearest[b] = min(nearest[b], bits)
    rows = []
    distances = Counter()
    for left, right in combinations(keys, 2):
        bits = (hashes[left] ^ hashes[right]).bit_count()
        distances[bits] += 1
        is_left = bits == nearest[left]
        is_right = bits == nearest[right]
        if bits > distance and not (is_left or is_right):
            continue
        rows.append({"left": left, "right": right, "distance": bits,
                     "nearest_to_left": int(is_left), "nearest_to_right": int(is_right),
                     "left_prefix": prefix(left), "right_prefix": prefix(right),
                     "left_image": inventory[left]["image"], "right_image": inventory[right]["image"],
                     "decision": ""})
    rows.sort(key=lambda r: (r["distance"], -r["nearest_to_left"] - r["nearest_to_right"],
                             r["left"], r["right"]))
    pair_prefix = Counter(" / ".join(sorted((r["left_prefix"], r["right_prefix"]))) for r in rows)
    summary = {
        "photos": len(keys), "all_pair_count": sum(distances.values()),
        "hamming_threshold": distance, "queue_pairs": len(rows), "queue_covered_photos": len(
            {key for row in rows for key in (row["left"], row["right"])}),
        "within_threshold_pairs": sum(v for k, v in distances.items() if k <= distance),
        "mutual_nearest_pairs": sum(r["nearest_to_left"] and r["nearest_to_right"] for r in rows),
        "same_prefix_pairs": sum(r["left_prefix"] == r["right_prefix"] for r in rows),
        "cross_prefix_pairs": sum(r["left_prefix"] != r["right_prefix"] for r in rows),
        "all_pair_distance_histogram": dict(sorted(distances.items())),
        "nearest_distance_photo_histogram": dict(sorted(Counter(nearest.values()).items())),
        "photo_counts_by_display_prefix": dict(sorted(Counter(prefix(k) for k in keys).items())),
        "candidate_pair_counts_by_display_prefix": dict(sorted(pair_prefix.items())),
        "selection_inputs": ["sample_id for stable tie order", "64-bit horizontal pixel difference hash"],
        "scope": "Review triage only. No hazard labels, notes or incident assumptions in selection. "
                 "ID prefixes describe spelling only. Distant photos may still show the same incident.",
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--distance", type=int, default=12)
    args = parser.parse_args()
    if len({p.resolve() for p in (args.audit, args.queue, args.summary)}) != 3:
        parser.error("input and outputs must have distinct paths")
    if args.queue.exists():
        parser.error("choose a new queue path to preserve existing human decisions")
    raw = args.audit.read_bytes()
    rows, summary = build(json.loads(raw)["inventory"], args.distance)
    summary["audit_sha256"] = hashlib.sha256(raw).hexdigest()
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.queue.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} candidate pairs, {summary['queue_covered_photos']} photos.")


if __name__ == "__main__":
    main()
