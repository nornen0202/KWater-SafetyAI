"""Prepare the 48-photo local ROI review packet. Run from the project root.

Restore without feature archives: --from-key <private-key-local.json> --out <new-directory>
The public results/vision-roi-panel-20261127.json binds the exact private key bytes.
"""

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

D = Path("data")
OUT = D / "vision-roi-48"
MANIFEST = D / "clean-split-v1/split_manifest_dev.json"
FEATURES = D / "clean-specialist-v1/features_multiview_2176d.npz"
CODES = ("struck_by_lifting", "fall_height", "slip_trip")


class PanelError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ReviewPhoto:
    alias: str
    family: str
    image: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class RestoreContract:
    train_root: Path
    panel_sha256: str


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def review_photos(key, train_root: Path) -> list[ReviewPhoto]:
    """Reduce a private key to blinded aliases, families, and confined image paths."""
    expected = {f"R{i:02}" for i in range(1, 49)}
    if len(key) != 48 or {row["id"] for row in key} != expected:
        raise PanelError("Expected exactly 48 unique aliases R01 through R48")
    root = train_root.resolve(strict=True)
    photos = []
    for row in key:
        relative = Path(row["source"]["image"])
        image = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not image.is_relative_to(root):
            raise PanelError("Image path escapes local training data")
        if row["family"] not in CODES:
            raise PanelError("Unknown target family")
        if sha(image) != row["source"]["sha256"]:
            raise PanelError("Canonical image SHA-256 mismatch")
        photos.append(
            ReviewPhoto(row["id"], row["family"], image, row["source"]["sha256"])
        )
    return photos


def write_review(blind: Path, photos: list[ReviewPhoto]) -> None:
    """Write the original PNG, LabelMe, and blank observations format without gold."""
    blind.mkdir()
    with (blind / "observations.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "id",
                "target_family",
                "object_visible",
                "counterpart_visible",
                "relation",
                "uncertainty_reason",
                "observation",
                "reviewer",
            ]
        )
        for photo in photos:
            if sha(photo.image) != photo.sha256:
                raise PanelError("Canonical image changed while building review")
            with Image.open(photo.image) as image:
                ImageOps.exif_transpose(image, in_place=True)
                im = image.convert("RGB")
                im.save(blind / f"{photo.alias}.png")
            save(
                blind / f"{photo.alias}.json",
                {
                    "version": "5.5.0",
                    "flags": {},
                    "shapes": [],
                    "imagePath": f"{photo.alias}.png",
                    "imageData": None,
                    "imageHeight": im.height,
                    "imageWidth": im.width,
                },
            )
            writer.writerow([photo.alias, photo.family] + [""] * 6)


def restore_from_key(key_path: Path, out: Path, contract: RestoreContract) -> None:
    """Rebuild only into a new directory after validating the exact private key."""
    if out.exists():
        raise PanelError(f"Existing review packet preserved: {out}")
    key_bytes = key_path.read_bytes()
    if hashlib.sha256(key_bytes).hexdigest() != contract.panel_sha256:
        raise PanelError("Private key SHA-256 does not match the frozen panel report")
    photos = review_photos(json.loads(key_bytes), contract.train_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix="roi-restore-") as scratch:
        packet = Path(scratch) / "packet"
        packet.mkdir()
        write_review(packet / "review", photos)
        (packet / "key-local.json").write_bytes(key_bytes)
        save(
            packet / "aggregate.json",
            {
                "photos": 48,
                "original_digests_verified": 48,
                "review_completed": 0,
                "panel_sha256": contract.panel_sha256,
                "restored_from_frozen_key": True,
            },
        )
        packet.rename(out)


def prepare(out):
    masks = [
        D / f"fold{i}-crosssplit-visual-review" / name
        for i, name in (
            (0, "fold0-group-exclusions-confirmed.json"),
            (4, "fold4-group-exclusions-local.json"),
        )
    ]
    blocked = set()
    for path in masks:
        mask = load(path)
        assert mask["canonical_manifest_sha256"] == sha(MANIFEST)
        blocked.update(x["train_group_id"] for x in mask["confirmed_cross_group_links"])
    prior = D / "curation-side-20261127/real-pairs.json"
    excluded = {m["sha256"] for p in load(prior) for m in p["members"]}
    rows = [
        r
        for r in load(MANIFEST)
        if r["split"] == "dev"
        and r["dev_fold"] in (1, 2, 3)
        and not r["exposed_prior"]
        and r["group_id"] not in blocked
        and r["sha256"] not in excluded
    ]
    with np.load(FEATURES, allow_pickle=False) as f:
        index = {v: i for i, v in enumerate(f["ids"].tolist())}
        x = f["features"][[index[r["sample_id"]] for r in rows], 0].astype(np.float32)
    sim = np.zeros((len(rows), len(rows)), dtype=np.float32)
    for part in (x[:, :1152], x[:, 1152:]):
        unit = part / np.maximum(np.linalg.norm(part, axis=1, keepdims=True), 1e-10)
        sim += (unit @ unit.T) / 2
    used, groups, pairs = set(), Counter(), []
    for k in range(8):
        for code in CODES:
            choices = []
            for i, a in enumerate(rows):
                if code not in a["codes"] or a["sha256"] in used:
                    continue
                for j, b in enumerate(rows):
                    if (
                        code in b["codes"]
                        or bool(b["codes"]) != bool(k % 2)
                        or b["sha256"] in used
                        or a["sha256"] == b["sha256"]
                        or a["family"] != b["family"]
                        or a["group_id"] == b["group_id"]
                    ):
                        continue
                    rank = (
                        max(groups[a["group_id"]], groups[b["group_id"]]),
                        groups[a["group_id"]] + groups[b["group_id"]],
                        -round(float(sim[i, j]), 6),
                        i,
                        j,
                    )
                    choices.append(rank)
            _, _, _, i, j = min(choices)
            for r in (rows[i], rows[j]):
                groups[r["group_id"]] += 1
                used.add(r["sha256"])
            pairs.append((code, rows[i], rows[j]))
    assert len(used) == 48
    entries = [
        (n, code, role, row)
        for n, (code, a, b) in enumerate(pairs, 1)
        for role, row in (("positive", a), ("control", b))
    ]
    random.Random(20261127).shuffle(entries)
    key = [
        {"id": f"R{i:02}", "pair": pair, "family": code, "role": role, "source": row}
        for i, (pair, code, role, row) in enumerate(entries, 1)
    ]
    write_review(out / "review", review_photos(key, D / "train"))
    save(out / "key-local.json", key)
    audit = {
        "eligible": len(rows),
        "photos": len(used),
        "pairs": len(pairs),
        "groups": len(groups),
        "largest_group": max(groups.values()),
        "pairs_per_code": dict(Counter(p[0] for p in pairs)),
        "control_safe": sum(not p[2]["codes"] for p in pairs),
        "original_digests_verified": len(key),
        "review_completed": 0,
        "panel_sha256": sha(out / "key-local.json"),
        "input_sha256": {
            str(p.as_posix()): sha(p) for p in [MANIFEST, FEATURES, prior, *masks]
        },
    }
    save(out / "aggregate.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-key", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Existing review packet preserved: {args.out}")
    if args.from_key is not None:
        report = load(Path("results/vision-roi-panel-20261127.json"))
        restore_from_key(
            args.from_key,
            args.out,
            RestoreContract(D / "train", report["panel_sha256"]),
        )
    else:
        with tempfile.TemporaryDirectory(dir=D, prefix="roi-build-") as scratch:
            packet = Path(scratch) / "packet"
            packet.mkdir()
            prepare(packet)
            packet.rename(args.out)
