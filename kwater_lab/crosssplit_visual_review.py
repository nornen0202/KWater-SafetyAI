"""Build a local blind Fold 0 contact sheet from unresolved semantic candidates."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .audit import digest
from .train_head import save_atomic


MANIFEST = Path("data/clean-split-v1/split_manifest_dev.json")
PAIRS = Path("data/phase01-runs/phase01-20261126-0513-2155e8fa/semantic-pairs.json")
PRIOR = Path("data/fold0-crosssplit-visual-review/mapping-local.json")
OUTPUT = Path("data/fold0-crosssplit-visual-review-wave2")
FIELDS = ("sample_id", "image", "group_id", "dev_fold")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_key(left, right):
    return tuple(sorted((left["group_id"], right["group_id"])))


def select_pairs(rows, candidates, previous, count):
    index = {r["sample_id"]: {key: r[key] for key in FIELDS} for r in rows}
    reviewed = {group_key(pair["left"], pair["right"]) for pair in previous}
    ranked = []
    for pair in candidates:
        if pair.get("decision") != "unresolved":
            continue
        left, right = index.get(pair["left"]), index.get(pair["right"])
        if left is None or right is None or ((left["dev_fold"] == 0) == (right["dev_fold"] == 0)):
            continue
        score = float(pair["cosine"])
        if not math.isfinite(score):
            raise ValueError("nonfinite similarity")
        key = group_key(left, right)
        if key not in reviewed:
            ranked.append((score, key, left, right))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]["sample_id"], row[3]["sample_id"]))
    selected = []
    seen = set(reviewed)
    for score, key, left, right in ranked:
        if key in seen:
            continue
        seen.add(key)
        number = max(int(pair["number"]) for pair in previous) + len(selected) + 1
        hashed = hashlib.sha256(f"kwater/fold0-crosssplit-wave2/{number}".encode()).digest()
        if hashed[0] % 2:
            left, right = right, left
        selected.append({"number": f"{number:02d}", "left": left, "right": right, "cosine": score})
        if len(selected) == count:
            break
    return selected, len(ranked), len({row[1] for row in ranked})


def build_sheet(root, pairs, destination):
    width, row_height, image_height = 1600, 490, 444
    sheet = Image.new("RGB", (width, row_height * len(pairs)), "#efefef")
    draw = ImageDraw.Draw(sheet)
    for row_idx, pair in enumerate(pairs):
        y = row_idx * row_height
        for col, side in enumerate(("left", "right")):
            x = col * (width // 2)
            draw.text((x + 16, y + 9), f"{pair['number']}  {'A' if col == 0 else 'B'}",
                      fill="#202020", font_size=22)
            with Image.open(root / "data/train" / pair[side]["image"]) as image:
                displayed = ImageOps.exif_transpose(image).convert("RGB")
                displayed.thumbnail((width // 2 - 30, image_height), Image.Resampling.LANCZOS)
            sheet.paste(displayed, (x + (width // 2 - displayed.width) // 2,
                                    y + 38 + (image_height - displayed.height) // 2))
    save_atomic(destination, lambda stream: sheet.save(stream, format="JPEG", quality=92))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=18)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out_dir.resolve()
    if not out.is_relative_to(root / "data"):
        parser.error("participant contact sheets and index must stay under project data/")
    if not 16 <= args.count <= 20:
        parser.error("select 16–20 review pairs")
    rows, candidates, prior = (load(root / path) for path in (MANIFEST, PAIRS, PRIOR))
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("expected development-only manifest")
    selected, records, groups = select_pairs(rows, candidates, prior["pair_mapping"], args.count)
    if len(selected) != args.count:
        raise ValueError("not enough distinct unreviewed group pairs")
    out.mkdir(parents=True, exist_ok=False)
    for start in range(0, len(selected), 2):
        page = selected[start:start + 2]
        filename = f"sheet-{page[0]['number']}-{page[-1]['number']}.jpg"
        for pair in page:
            pair["sheet"] = filename
        build_sheet(root, page, out / filename)
    mapping = {
        "rule": "unresolved dev Fold 0 cross-pairs, descending semantic cosine, previously reviewed unordered group pairs excluded",
        "input_sha256": {path.as_posix(): digest(root / path) for path in (MANIFEST, PAIRS, PRIOR)},
        "eligible_photo_pairs": records, "eligible_distinct_group_pairs": groups,
        "pair_mapping": selected,
    }
    raw = (json.dumps(mapping, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    save_atomic(out / "mapping-local.json", lambda stream: stream.write(raw))
    print(f"Selected {len(selected)} pairs of {groups} eligible group pairs across {records} candidate photo pairs.")
    print(f"Cosine range {selected[0]['cosine']:.4f}..{selected[-1]['cosine']:.4f}.")


if __name__ == "__main__":
    main()
