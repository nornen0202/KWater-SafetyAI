"""Audit official K-water training wrappers and prepare an incident-level split.

Run with `python -m kwater_lab.audit --help`. Use Pillow to decode images.
Without a known incident identifier, provide a reviewed CSV with columns
sample_id,group_id for every photo. Put the same incident, crops and consecutive
shots in the same group. A unique photo number is not an incident identifier.
Only exact pixel matches merge automatically. Review other close dHash pairs
in a CSV with columns left,right,decision (same or separate). Confirm other
related shots, including different viewpoints, through the group map.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps

from .metrics import CODES, decode_gold, reject_constant, unique_keys


RATIOS = {"train": 0.70, "dev": 0.15, "holdout": 0.15}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_photo(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("image must be a nonempty relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("image path escapes its image root")
    # Hash and decode the exact same read, including on slower external drives.
    return inspect_bytes(path.read_bytes())


def inspect_bytes(raw):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        pixel_hash = hashlib.sha256(f"RGB:{rgb.width}x{rgb.height}:".encode())
        pixel_hash.update(rgb.tobytes())
        reduced = rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        light = reduced.tobytes()
        dhash = 0
        for y in range(8):
            for x in range(8):
                i = 9 * y + x
                dhash = (dhash << 1) | int(light[i] > light[i + 1])
        return {"width": rgb.width, "height": rgb.height, "sha256": hashlib.sha256(raw).hexdigest(),
                "pixels_sha256": pixel_hash.hexdigest(), "dhash64": f"{dhash:016x}"}


def case_value(row, field):
    value = row
    for key in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if type(value) in (str, int) and str(value).strip():
        return str(value).strip()
    return None


def read_inventory(labels, root, field):
    photos, groups, errors = {}, {}, []
    with labels.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line.decode("utf-8"), object_pairs_hook=unique_keys,
                                 parse_constant=reject_constant)
                if not isinstance(row, dict) or not {"sample_id", "target", "image"} <= row.keys():
                    raise ValueError("expected sample_id, image and target")
                key, codes = decode_gold(row)
                if key in photos:
                    raise ValueError(f"duplicate sample_id {key}")
                photo = inspect_photo(root, row["image"])
                photo.update(image=row["image"], status=row["target"]["status"], codes=sorted(codes))
                photos[key] = photo
                if field:
                    group = case_value(row, field)
                    if group is not None:
                        groups[key] = group
            except (KeyError, ValueError, TypeError, OSError, Image.DecompressionBombError) as exc:
                errors.append(f"line {number}: {exc}")
    if not photos:
        errors.append("no readable labeled photo")
    return photos, groups, errors


def csv_rows(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(fields) != set(reader.fieldnames) or len(fields) != len(reader.fieldnames):
            raise ValueError(f"{path}: expected CSV header {','.join(fields)}")
        for number, row in enumerate(reader, 2):
            if None in row or any(not row[f] or not row[f].strip() for f in fields):
                raise ValueError(f"{path}:{number}: empty value or extra column")
            yield tuple(row[f].strip() for f in fields)


class Components:
    def __init__(self, ids):
        self.parent = {key: key for key in ids}

    def find(self, key):
        while key != self.parent[key]:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left, right):
        a, b = sorted((self.find(left), self.find(right)))
        self.parent[b] = a


def buckets(mapping):
    out = defaultdict(list)
    for key, value in sorted(mapping.items()):
        out[value].append(key)
    return out


def assign_groups(connected, seed, exposed=()):
    # Sorting by a seeded digest makes the split independent of JSONL order.
    def priority(item):
        key, members = item
        return -len(members), hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()

    ordered = sorted(connected.items(), key=priority)
    total = sum(map(len, connected.values()))
    sizes = Counter()
    result = {}
    exposed = set(exposed)
    ineligible = {group for group, members in connected.items() if not exposed.isdisjoint(members)}
    remaining_eligible = len(connected) - len(ineligible)
    for i, (group, members) in enumerate(ordered):
        if group not in ineligible:
            remaining_eligible -= 1
        empty = {part for part in RATIOS if not sizes[part]}
        available = []
        for part in RATIOS:
            if part == "holdout" and group in ineligible:
                continue
            still_empty = empty - {part}
            if (len(ordered) - i - 1 >= len(still_empty) and
                    ("holdout" not in still_empty or remaining_eligible)):
                available.append(part)
        # Change in weighted squared distance from the requested photo counts.
        def cost(part):
            target = total * RATIOS[part]
            n = len(members)
            return (2 * (sizes[part] - target) * n + n * n) / target

        part = min(available, key=cost)
        sizes[part] += len(members)
        for key in members:
            result[key] = {"group_id": group, "split": part}
    return dict(sorted(result.items()))


def summarize(photos, ids):
    counts = Counter(code for key in ids for code in photos[key]["codes"])
    pairs = Counter(" + ".join(pair) for key in ids
                    for pair in combinations(photos[key]["codes"], 2))
    return {"photos": len(ids),
            "status": dict(Counter(photos[key]["status"] for key in ids)),
            "hazards_per_photo": dict(Counter(len(photos[key]["codes"]) for key in ids)),
            "code_cooccurrence": dict(pairs),
            "codes": {code: counts[code] for code in CODES}}


def audit(labels, *, image_root=None, case_field=None, group_map=None,
          review_pairs=None, distance=4, seed=20261125, exclude_holdout_ids=None):
    labels = Path(labels)
    if not 0 <= distance <= 64:
        raise ValueError("dHash distance must be between 0 and 64")
    if case_field and group_map:
        raise ValueError("choose one incident identifier source")
    if case_field and case_field.split(".")[-1] in {"sample_id", "image_id", "image"}:
        raise ValueError("a photo identifier cannot serve as an incident identifier")
    root = Path(image_root or labels.parent).resolve()
    photos, groups, errors = read_inventory(labels, root, case_field)
    if group_map:
        for key, group in csv_rows(Path(group_map), ("sample_id", "group_id")):
            if key not in photos or key in groups:
                raise ValueError(f"unexpected or duplicated group mapping for {key}")
            groups[key] = group
    explicit = buckets(groups)
    exact = buckets({key: photo["pixels_sha256"] for key, photo in photos.items()})
    components = Components(photos)
    for bucket in [*explicit.values(), *exact.values()]:
        for key in bucket[1:]:
            components.join(bucket[0], key)

    decisions = {}
    if review_pairs:
        for a, b, decision in csv_rows(Path(review_pairs), ("left", "right", "decision")):
            pair = tuple(sorted((a, b)))
            if (a == b or a not in photos or b not in photos or
                    decision not in {"same", "separate"} or pair in decisions):
                raise ValueError(f"invalid pair decision: {a},{b},{decision}")
            decisions[pair] = decision
            if decision == "same":
                components.join(a, b)

    matches = []
    hashes = {key: int(photo["dhash64"], 16) for key, photo in photos.items()}
    for a, b in combinations(sorted(photos), 2):
        if components.find(a) == components.find(b):
            continue
        bits = (hashes[a] ^ hashes[b]).bit_count()
        if bits <= distance:
            matches.append({"left": a, "right": b, "distance": bits,
                            "decision": decisions.get((a, b), "review")})
    for (a, b), decision in decisions.items():
        if decision == "separate" and components.find(a) == components.find(b):
            errors.append(f"pair marked separate within a merged incident: {a},{b}")

    connected = buckets({key: components.find(key) for key in photos})
    missing = sorted(photos.keys() - groups.keys())
    pending = sum(p["decision"] == "review" for p in matches)
    codes_by_group = Counter(code for members in connected.values()
                             for code in {c for key in members for c in photos[key]["codes"]})
    report = {
        "status": "audit_only", "labels_sha256": digest(labels), "case_field": case_field,
        "group_map_sha256": digest(Path(group_map)) if group_map else None,
        "review_pairs_sha256": digest(Path(review_pairs)) if review_pairs else None,
        "dhash_max_distance": distance, "seed": seed, "target_photo_fractions": RATIOS,
        "inventory": dict(sorted(photos.items())), "summary": summarize(photos, photos),
        "exact_pixel_duplicates": [members for members in exact.values() if len(members) > 1],
        "exact_pixel_groups_with_conflicting_codes": [
            members for members in exact.values() if len({tuple(photos[key]["codes"])
                                                         for key in members}) > 1],
        "cross_group_near_pairs": matches, "pending_pair_reviews": pending,
        "unmapped_sample_ids": missing, "errors": errors,
        "warning": "dHash is a review aid. Crops and new viewpoints of one incident can have distant hashes.",
    }
    exposed = set()
    if exclude_holdout_ids:
        raw = Path(exclude_holdout_ids).read_bytes()
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_keys)
        ids = manifest["exposed_ids"]
        if (manifest["labels_sha256"] != report["labels_sha256"] or not isinstance(ids, list)
                or not all(isinstance(k, str) and k in photos for k in ids) or len(set(ids)) != len(ids)):
            raise ValueError("Invalid exposure manifest, sample IDs or label checksum")
        exposed = set(ids)
        report["holdout_exposure_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        report["explicitly_exposed_photos"] = len(exposed)
    if errors:
        report["status"] = "invalid_data"
    elif missing or pending:
        report["status"] = "needs_group_review"
    elif len(connected) < 3:
        report["status"] = "too_few_incident_groups"
    elif all(not exposed.isdisjoint(members) for members in connected.values()):
        report["status"] = "no_unexposed_holdout_group"
    else:
        assigned = assign_groups(connected, seed, exposed)
        violations = [
            members for members in [*explicit.values(), *exact.values(), *connected.values()]
            if len({assigned[key]["split"] for key in members}) != 1
        ]
        if violations or set(assigned) != set(photos) or any(assigned[k]["split"] == "holdout" for k in exposed):
            raise RuntimeError("split conservation or incident leakage check failed")
        report.update(
            status="ready", assignments=assigned, cross_split_group_violations=len(violations),
            incident_groups=len(connected),
            incident_groups_per_code={c: codes_by_group[c] for c in CODES},
            split_summary={part: summarize(photos, [key for key, row in assigned.items()
                                                   if row["split"] == part])
                           for part in RATIOS},
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_jsonl", type=Path)
    parser.add_argument("--image-root", type=Path, help="defaults to the label file's parent")
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case-field", help="documented true incident identifier, dotted path supported")
    source.add_argument("--group-map", type=Path, help="reviewed CSV: sample_id,group_id")
    parser.add_argument("--review-pairs", type=Path, help="reviewed CSV: left,right,decision")
    parser.add_argument("--exclude-holdout-ids", type=Path,
                        help="JSON with labels_sha256 and exposed_ids; exclude whole linked incidents from holdout")
    parser.add_argument("--distance", type=int, default=4, help="maximum 64-bit dHash distance for review")
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    originals = [p for p in (args.train_jsonl, args.group_map, args.review_pairs, args.exclude_holdout_ids) if p]
    if args.output.resolve() in {p.resolve() for p in originals}:
        parser.error("choose an output path separate from source files")
    report = audit(args.train_jsonl, image_root=args.image_root, case_field=args.case_field,
                   group_map=args.group_map, review_pairs=args.review_pairs,
                   distance=args.distance, seed=args.seed, exclude_holdout_ids=args.exclude_holdout_ids)
    if args.output.suffix.lower() != ".json":
        parser.error("choose a .json output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{report['status']}: {report['summary']['photos']} photos, "
          f"{len(report['unmapped_sample_ids'])} without incident IDs, "
          f"{report['pending_pair_reviews']} pending pairs. {args.output}")
    return 2 if report["status"] == "invalid_data" else 0


if __name__ == "__main__":
    raise SystemExit(main())
