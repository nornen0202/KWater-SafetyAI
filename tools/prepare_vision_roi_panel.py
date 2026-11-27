"""Prepare the 48-photo local ROI review packet. Run from the project root."""
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

D = Path("data")
OUT = D / "vision-roi-48"
MANIFEST = D / "clean-split-v1/split_manifest_dev.json"
FEATURES = D / "clean-specialist-v1/features_multiview_2176d.npz"
CODES = ("struck_by_lifting", "fall_height", "slip_trip")


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(out):
    masks = [D / f"fold{i}-crosssplit-visual-review" / name for i, name in (
        (0, "fold0-group-exclusions-confirmed.json"), (4, "fold4-group-exclusions-local.json"))]
    blocked = set()
    for path in masks:
        mask = load(path)
        assert mask["canonical_manifest_sha256"] == sha(MANIFEST)
        blocked.update(x["train_group_id"] for x in mask["confirmed_cross_group_links"])
    prior = D / "curation-side-20261127/real-pairs.json"
    excluded = {m["sha256"] for p in load(prior) for m in p["members"]}
    rows = [r for r in load(MANIFEST) if r["split"] == "dev" and r["dev_fold"] in (1, 2, 3)
            and not r["exposed_prior"] and r["group_id"] not in blocked and r["sha256"] not in excluded]
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
                    if (code in b["codes"] or bool(b["codes"]) != bool(k % 2)
                            or b["sha256"] in used or a["sha256"] == b["sha256"]
                            or a["family"] != b["family"] or a["group_id"] == b["group_id"]):
                        continue
                    rank = (max(groups[a["group_id"]], groups[b["group_id"]]),
                            groups[a["group_id"]] + groups[b["group_id"]], -round(float(sim[i,j]), 6), i, j)
                    choices.append(rank)
            _, _, _, i, j = min(choices)
            for r in (rows[i], rows[j]):
                groups[r["group_id"]] += 1
                used.add(r["sha256"])
            pairs.append((code, rows[i], rows[j]))
    assert len(used) == 48
    entries = [(n, code, role, row) for n, (code, a, b) in enumerate(pairs, 1)
               for role, row in (("positive", a), ("control", b))]
    random.Random(20261127).shuffle(entries)
    blind = out / "review"
    blind.mkdir()
    key = []
    with (blind / "observations.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "target_family", "object_visible", "counterpart_visible", "relation",
                         "uncertainty_reason", "observation", "reviewer"])
        for i, (pair, code, role, row) in enumerate(entries, 1):
            q = f"R{i:02}"
            src = D / "train" / row["image"]
            assert sha(src) == row["sha256"]
            with Image.open(src) as image:
                im = ImageOps.exif_transpose(image).convert("RGB")
                im.save(blind / f"{q}.png")
            save(blind / f"{q}.json", dict(version="5.5.0", flags={}, shapes=[],
                 imagePath=f"{q}.png", imageData=None, imageHeight=im.height, imageWidth=im.width))
            writer.writerow([q, code] + [""] * 6)
            key.append(dict(id=q, pair=pair, family=code, role=role, source=row))
    save(out / "key-local.json", key)
    audit = dict(eligible=len(rows), photos=len(used), pairs=len(pairs), groups=len(groups),
                 largest_group=max(groups.values()), pairs_per_code=dict(Counter(p[0] for p in pairs)),
                 control_safe=sum(not p[2]["codes"] for p in pairs), original_digests_verified=len(key),
                 review_completed=0, panel_sha256=sha(out / "key-local.json"),
                 input_sha256={str(p.as_posix()):sha(p) for p in [MANIFEST, FEATURES, prior, *masks]})
    save(out / "aggregate.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"Existing review packet preserved: {OUT}")
    with tempfile.TemporaryDirectory(dir=D, prefix="roi-build-") as scratch:
        packet = Path(scratch) / "packet"
        packet.mkdir()
        prepare(packet)
        packet.rename(OUT)
