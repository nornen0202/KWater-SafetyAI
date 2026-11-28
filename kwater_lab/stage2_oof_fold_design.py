"""Freeze a full-data five-fold design with conservative, reviewed photo-group blocks."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .audit import Components
from .clean_split import initial_prefix_group
from .fold4_same_verifier import digest, read, save
from .metrics import CODES, decode_gold, unique_keys

FULL = Path("data/clean-split-v1/split_manifest_full_sealed.json")
DEV = Path("data/clean-split-v1/split_manifest_dev.json")
LABELS = Path("data/train/train.jsonl")
INVENTORY = Path("results/train_audit_full.json")
SPLIT_RECEIPT = Path("results/clean_split_v1_summary.json")
F0 = Path("data/fold0-crosssplit-visual-review")
F0_MAPS = (F0 / "mapping-local.json", F0.with_name(F0.name + "-wave2") / "mapping-local.json")
F0_LEDGER = F0 / "fold0-group-exclusions-confirmed.json"
F4 = Path("data/fold4-crosssplit-visual-review")
F4_MAP = F4 / "mapping-local.json"
F4_REVIEW = F4 / "judgments-local.json"
F4_LEDGER = F4 / "fold4-group-exclusions-local.json"
F4_SCAN = Path("results/fold4-crosssplit-audit.json")
PHASE01 = Path("data/phase01-runs/phase01-20261126-0513-2155e8fa")
PHASE01_PAIRS = (PHASE01 / "pairs.json", PHASE01 / "semantic-pairs.json")
DEV_LINEAGE = Path("results/clean-development-group-lineage.json")
PROVENANCE = Path("results/official_incident_provenance.json")
RESTRICTED = Path("data/stage2-oof-design-20261128")
RESULT = Path("results/stage2-oof-design-20261128.json")
SOURCE_FILES = (FULL, DEV, LABELS, INVENTORY, SPLIT_RECEIPT, F0_LEDGER, *F0_MAPS, F4_LEDGER,
                F4_MAP, F4_REVIEW, F4_SCAN, *PHASE01_PAIRS, DEV_LINEAGE, PROVENANCE)
SEED = 20261128
N_FOLDS = 5
SUPPORT_KEYS = ("photos", "normal", "severity4", *CODES)
WEIGHT = {"photos": 8, "normal": 3, "severity4": 3}


def partition(keys, key_for):
    bins = defaultdict(list)
    for key in sorted(keys):
        bins[key_for(key)].append(key)
    return bins


def read_roster(hashes):
    prior = read(SPLIT_RECEIPT)
    inventory = read(INVENTORY)
    if ((prior["sealed_manifest_sha256"], prior["dev_manifest_sha256"], prior["labels_sha256"])
            != (hashes[FULL.as_posix()], hashes[DEV.as_posix()], hashes[LABELS.as_posix()])
            or inventory["labels_sha256"] != prior["labels_sha256"] or inventory["errors"]):
        raise ValueError("Official inventory or historical split fingerprint differs")
    full = read(FULL)
    if [row for row in full if row["split"] == "dev"] != read(DEV):
        raise ValueError("Historical full roster does not preserve the existing development split")
    roster = {row["sample_id"]: row for row in full}
    official = [
        json.loads(line, object_pairs_hook=unique_keys) for line in LABELS.read_bytes().splitlines() if line.strip()
    ]
    if (len(full) != len(roster) or len(roster) != len(official) or len(official) != 1045
            or inventory["inventory"].keys() != roster.keys()):
        raise ValueError("Expected exactly 1,045 canonical official photo IDs")
    observed = set()
    infos = {}
    for raw in official:
        sid, codes = decode_gold(raw)
        row, info = roster[sid], inventory["inventory"][sid]
        if (sid in observed or row["image"] != raw["image"] or info["image"] != row["image"]
                or row["sha256"] != info["sha256"] or set(row["codes"]) != codes
                or set(info["codes"]) != codes or row["status"] != raw["target"]["status"]):
            raise ValueError("A canonical photo or official code set differs from the prior audited inventory")
        observed.add(sid)
        severity4 = {code for code, item in raw["target"]["hazards"].items()
                     if item["risk_level"]["severity"]["score"] == 4}
        infos[sid] = {**row, "pixels_sha256": info["pixels_sha256"], "severity4": sorted(severity4)}
    if (len({row["image"] for row in infos.values()}) != 1045
            or sum(not row["codes"] for row in infos.values()) != 315):
        raise ValueError("Expected 1,045 distinct paths and 315 official normal photos")
    return infos


def reviewed_edges(rows, hashes):
    confirmed = []
    suspect = []

    def check_pair(pair):
        left, right = pair["left"], pair["right"]
        for item in (left, right):
            observed = rows[item["sample_id"]]
            if any(item[key] != observed[key] for key in ("image", "group_id", "dev_fold")):
                raise ValueError("Visual review endpoint differs from the historical roster")
            if item.get("sha256", observed["sha256"]) != observed["sha256"]:
                raise ValueError("Visual review endpoint content hash differs")
        return left, right

    ledger0, ledger4 = read(F0_LEDGER), read(F4_LEDGER)
    scan4 = read(F4_SCAN)
    if (any(ledger["canonical_manifest_sha256"] != hashes[DEV.as_posix()] for ledger in (ledger0, ledger4))
            or ledger0["visual_review_mappings_sha256"] != [hashes[path.as_posix()] for path in F0_MAPS]
            or ledger4["review_sha256"] != hashes[F4_REVIEW.as_posix()]
            or scan4["mapping_sha256"] != hashes[F4_MAP.as_posix()]
            or scan4["confirmed_mask_sha256"] != hashes[F4_LEDGER.as_posix()]):
        raise ValueError("Review maps or links differ from the confirmed ledgers")
    mapping0 = {item["number"]: item for path in F0_MAPS for item in read(path)["pair_mapping"]}
    mapping4 = {item["number"]: item for item in read(F4_MAP)["pairs"]}
    judgments4 = read(F4_REVIEW)
    if (mapping4.keys() != judgments4.keys() or
            {key for key, item in judgments4.items() if item["decision"] == "same_visible_scene"} !=
            {item["review_number"] for item in ledger4["confirmed_cross_group_links"]}):
        raise ValueError("Fold 4 review decisions and confirmed link ledger differ")
    for fold, mapping, ledger in ((0, mapping0, ledger0), (4, mapping4, ledger4)):
        for item in ledger["confirmed_cross_group_links"]:
            left, right = check_pair(mapping[item["review_number"]])
            validate, train = (left, right) if left["dev_fold"] == fold else (right, left)
            if (validate["dev_fold"] != fold or train["dev_fold"] == fold
                    or validate["group_id"] != item[f"fold{fold}_group_id"]
                    or train["group_id"] != item["train_group_id"]):
                raise ValueError("Reviewed link endpoints do not match the declared fold groups")
            confirmed.append((left["sample_id"], right["sample_id"], f"fold{fold}_reviewed_same_scene"))
    for number, item in judgments4.items():
        if item["decision"] == "possible_same_site_unproven":
            left, right = check_pair(mapping4[number])
            suspect.append((left["sample_id"], right["sample_id"], "fold4_possible_same_site"))

    for path in PHASE01_PAIRS:
        for pair in read(path):
            if pair["decision"] in {"quarantine_same_visual_source", "quarantine"}:
                left, right = pair["left"], pair["right"]
                if left not in rows or right not in rows:
                    raise ValueError("Reviewed near-duplicate is outside the official roster")
                confirmed.append((left, right, "phase01_same_visual_source_quarantine"))
    return confirmed, suspect


def support(rows, ids):
    values = Counter()
    for sid in ids:
        row = rows[sid]
        values.update(row["codes"])
        values["photos"] += 1
        values["normal"] += not row["codes"]
        values["severity4"] += bool(row["severity4"])
    return values


def assign_blocks(blocks, rows):
    if len(blocks) < N_FOLDS:
        raise ValueError("Fewer audited blocks than requested folds")
    total = support(rows, rows)
    targets = {key: total[key] / N_FOLDS for key in SUPPORT_KEYS}
    vectors = {root: support(rows, ids) for root, ids in blocks.items()}
    states = [Counter() for _ in range(N_FOLDS)]
    allocated = {}

    def priority(root):
        vector = vectors[root]
        rarity = sum(vector[code] / total[code] for code in CODES if total[code] > 0)
        seeded = sha256(f"{SEED}:{root}".encode()).hexdigest()
        return -vector["photos"], -rarity, seeded

    def cost(fold, vector):
        state = states[fold]
        return sum(WEIGHT.get(key, 1) * (2 * (state[key] - targets[key]) * vector[key] + vector[key] ** 2)
                   / max(1, targets[key]) for key in SUPPORT_KEYS)

    for index, root in enumerate(sorted(blocks, key=priority)):
        vector = vectors[root]
        empty = [f for f in range(N_FOLDS) if not states[f]]
        available = empty if len(blocks) - index <= len(empty) else range(N_FOLDS)
        chosen = min(available, key=lambda fold: (cost(fold, vector), states[fold]["photos"], fold))
        allocated[root] = chosen
        states[chosen].update(vector)
    return allocated


def join_blocks(rows, historical, exact, edges):
    components = Components(rows)

    def join_members(members):
        for other in members[1:]:
            components.join(members[0], other)

    for members in historical.values():
        join_members(members)
    for sets in exact.values():
        for members in sets.values():
            join_members(members)
    for left, right, _ in edges:
        components.join(left, right)
    return partition(rows, components.find)


def block_assignment(rows, historical, confirmed, suspect):
    exact = {name: partition(rows, lambda sid: rows[sid][name]) for name in ("sha256", "pixels_sha256")}
    blocks = join_blocks(rows, historical, exact, confirmed + suspect)
    assigned = assign_blocks(blocks, rows)
    by_id = {sid: assigned[root] for root, members in blocks.items() for sid in members}
    for groups in (historical, *exact.values()):
        if any(len({by_id[sid] for sid in members}) != 1 for members in groups.values()):
            raise ValueError("An existing block or official content digest crosses folds")
    if any(by_id[left] != by_id[right] for left, right, _ in confirmed + suspect):
        raise ValueError("A reviewed or conservatively blocked pair crosses folds")
    return blocks, by_id, exact


def prepare(out, result):
    hashes = {path.as_posix(): digest(path) for path in SOURCE_FILES}
    rows = read_roster(hashes)
    historical = partition(rows, lambda sid: rows[sid]["group_id"])
    confirmed, suspect = reviewed_edges(rows, hashes)
    blocks, folds, exact = block_assignment(rows, historical, confirmed, suspect)
    if len(historical) != 239 or any(len(items) != 1045 for items in exact.values()):
        raise ValueError("Historical block count or unique official image count changed")
    membership_hashes = {root: sha256(json.dumps(members).encode()).hexdigest() for root, members in blocks.items()}
    block_ids = {sid: membership_hashes[root] for root, members in blocks.items() for sid in members}
    manifest = [{**{key: rows[sid][key] for key in ("sample_id", "image", "sha256", "family", "status", "codes")},
                 "group_id": block_ids[sid], "split": "dev", "dev_fold": folds[sid],
                 "historical_sealed_panel": rows[sid]["split"] == "holdout"}
                for sid in sorted(rows)]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fold-manifest-local.json"
    save(path, manifest)
    fold_results = {}
    for f in range(N_FOLDS):
        test = {sid for sid in folds if folds[sid] == f}
        train = {sid for sid in folds if folds[sid] != f}
        train_groups, test_groups = ({block_ids[sid] for sid in ids} for ids in (train, test))
        if train != rows.keys() - test or train_groups & test_groups or not test:
            raise ValueError("A fold fails complement or audited block isolation")
        counts = support(rows, test)
        train_counts = support(rows, train)
        largest = max(Counter(block_ids[sid] for sid in test).values())
        fold_results[str(f)] = {
            "test_photos": len(test), "train_photos": len(train), "test_blocks": len(test_groups),
            "largest_test_block_photos": largest, "largest_test_block_share": largest / len(test),
            "test_normal": counts["normal"], "test_severity4_photos": counts["severity4"],
            "train_normal": train_counts["normal"], "train_severity4_photos": train_counts["severity4"],
            "test_codes": {code: counts[code] for code in CODES},
            "train_codes": {code: train_counts[code] for code in CODES},
        }
    if sum(fold["test_photos"] for fold in fold_results.values()) != len(rows):
        raise ValueError("Photos must have exactly one test fold")
    # Explain inherited prefix dependence without using prefix inference to allocate any photo.
    extends_prefix = sum(any(n > 1 for n in Counter(initial_prefix_group(sid) for sid in members).values())
                         for members in historical.values())
    totals = support(rows, rows)
    confirmed_blocks = join_blocks(rows, historical, exact, confirmed)

    def block_sizes(members):
        return {"blocks": len(members), "largest_block_sizes": sorted(map(len, members.values()), reverse=True)[:5]}

    summary = {
        "scope": "Design only. Existing proxy groups remain indivisible; reviewed same-scene and possible-site "
                 "pairs and exact content are blocked transitively. True event identities are unavailable.",
        "models_fitted": 0, "oof_prediction_rows_generated": 0, "folds": fold_results, "seed": SEED,
        "official_roster": {key: totals[key] for key in SUPPORT_KEYS},
        "official_unique_file_digests": len(exact["sha256"]), "official_unique_pixel_digests": len(exact["pixels_sha256"]),
        "historical_groups": len(historical), "historical_groups_with_inherited_numeric_prefix_joins": extends_prefix,
        "audited_blocks": len(blocks),
        "block_size_diagnostics": {"historical_only": block_sizes(historical),
                                   "historical_and_confirmed": block_sizes(confirmed_blocks),
                                   "historical_confirmed_and_possible_used_for_assignment": block_sizes(blocks)},
        "confirmed_or_quarantined_pair_counts": dict(Counter(kind for _, _, kind in confirmed)),
        "conservative_possible_site_pair_counts": dict(Counter(kind for _, _, kind in suspect)),
        "cross_fold_photo_content_reviewed_edge_or_historical_block_violations": 0,
        "historical_sealed_photos_in_design": sum(row["historical_sealed_panel"] for row in manifest),
        "assignment": {"ordered_by": ["block_size descending", "relative code rarity descending", "seeded SHA-256"],
                       "component_load_weights": WEIGHT, "all_other_code_weights": 1},
        "input_sha256": hashes, "restricted_manifest_bytes": path.stat().st_size,
        "restricted_manifest_sha256": digest(path),
    }
    save(result, summary)
    print(f"DESIGN_VERIFIED {len(rows)} photos, {len(blocks)} audited blocks, five folds; {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESTRICTED)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Restricted manifest must stay in data/")
    prepare(args.out, args.result)
