"""Review positive/negative state contrasts within the already exposed official pilot."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


CODES = ("fall_height", "struck_by_lifting")
BLIND_SEED = "state-pair-blind-v1/"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def select(root, extended):
    pilot = root / "data/prefix-pilot"
    exposed = read(pilot / "exposure.json")
    scoped = read(pilot / "diagnostic.json")
    excluded = set(read(root / "data/holdout-exclusions.json")["exposed_ids"])
    ids = sorted(exposed["exposed_ids"])
    gold = {r["sample_id"]: set(r["gold"]) for r in scoped["paired_out_of_prefix_predictions"]}
    if (len(ids) != len(set(ids)) or len(ids) != 170 or len(excluded) != 182 or
            not set(ids) <= excluded or set(ids) != set(gold) or
            scoped["labels_sha256"] != exposed["labels_sha256"]):
        raise ValueError("Exposed cohort, exclusions or labels changed")
    scores, provenance = [], {}
    for name in ("features.npz", "siglip2-base16-384-features.npz"):
        raw = (pilot / name).read_bytes()
        provenance[name] = sha(raw)
        with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
            source = cache["ids"].tolist()
            if len(source) != len(ids) or set(source) != set(ids):
                raise ValueError("Feature cache does not match the exposed cohort")
            x = cache["features"][[source.index(i) for i in ids]]
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.isfinite(norm).all() or (norm <= 0).any():
            raise ValueError("Invalid embeddings")
        x = x / norm
        scores.append(x @ x.T)
    similarities = (scores[0] + scores[1]) / 2
    rankings = {}
    for code in CODES:
        rankings[code] = sorted(
            ((float(similarities[i, j]), a, b) for i, a in enumerate(ids) if code in gold[a]
             for j, b in enumerate(ids) if code not in gold[b]),
            key=lambda row: (-row[0], row[1], row[2]))
    pairs, used = [], set()

    def add(code, rank, row):
        score, positive, negative = row
        pairs.append({"pair": f"{code}-{rank}", "code": code, "positive": positive,
                      "negative": negative, "mean_cosine": score})
        used.update((positive, negative))

    for rank in range(1, 7):
        for code in CODES:
            add(code, rank, next(r for r in rankings[code] if used.isdisjoint(r[1:])))
    first_ids = set(used)
    if extended:
        code = "struck_by_lifting"
        done = {p["positive"] for p in pairs if p["code"] == code}
        total = sum(code in gold[k] for k in ids)
        while len(done) < total:
            row = next(r for r in rankings[code] if r[1] not in done and r[2] not in used)
            done.add(row[1])
            add(code, len(done), row)
    order = lambda k: sha((BLIND_SEED + k).encode())
    ordered = sorted(first_ids, key=order) + sorted(used - first_ids, key=order)
    tokens = {sid: f"P{i:02d}" for i, sid in enumerate(ordered, 1)}
    for row in pairs:
        for side in ("positive", "negative"):
            row[side] = tokens[row[side]]
    return pairs, tokens, provenance


def render_sheets(folder, tokens, start):
    font = ImageFont.load_default(size=20)
    for offset in range(start, len(tokens), 4):
        canvas = Image.new("RGB", (1280, 872), (235, 235, 235))
        draw = ImageDraw.Draw(canvas)
        for i, token in enumerate(tokens[offset:offset + 4]):
            with Image.open(folder / "originals" / f"{token}.png") as source:
                view = ImageOps.exif_transpose(source).convert("RGB")
            view.thumbnail((634, 402), Image.Resampling.LANCZOS)
            x, y = (i % 2) * 640, (i // 2) * 436
            draw.text((x + 4, y), token, fill="black", font=font)
            canvas.paste(view, (x + (640 - view.width) // 2, y + 27 + (402 - view.height) // 2))
        canvas.save(folder / "sheets" / f"panel-{offset // 4 + 1}.jpg", quality=90)


def prepare(root, extended):
    pairs, tokens, provenance = select(root, extended)
    folder = root / "data/state-pairs"
    key_path = folder / "key.json"
    if key_path.exists() and not extended:
        raise FileExistsError("Keep the existing blinded panel and observations")
    previous = read(key_path) if extended else {"photos": {}, "pairs": []}
    if extended and (len(previous["pairs"]) != 12 or previous["pairs"] != pairs[:12] or
                     set(read(folder / "blind-review.json")) != set(previous["photos"])):
        raise ValueError("Extension requires the original 24 blinded observations")
    photos = dict(previous["photos"])
    pending = {sid for sid, token in tokens.items() if token not in photos}
    for name in ("originals", "sheets"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    train = (root / "data/train").resolve()
    with (train / "train.jsonl").open("rb") as stream:
        for line in stream:
            # Official wrappers place sample_id and image before target. Never
            # parse labels for new photos or open an image outside the panel.
            header = line.split(b'"target"', 1)[0]
            match = re.search(rb'"sample_id"\s*:\s*"([^"]+)"', header)
            if not match or (sid := match[1].decode("ascii")) not in pending:
                continue
            match = re.search(rb'"image"\s*:\s*"([^"]+)"', header)
            if not match:
                raise ValueError("Expected image path in official wrapper")
            rel = match[1].decode("ascii")
            src = (train / rel).resolve()
            if not src.is_relative_to(train):
                raise ValueError("Unexpected source path")
            raw = src.read_bytes()
            token = tokens[sid]
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
            (folder / "originals" / f"{token}.png").write_bytes(raw)
            photos[token] = {"sample_id": sid, "source": rel, "sha256": sha(raw),
                             "width": width, "height": height}
    if set(photos) != set(tokens.values()):
        raise ValueError("Images missing from the blinded panel")
    write(key_path, {"photos": photos, "pairs": pairs, "feature_sha256": provenance})
    render_sheets(folder, list(tokens.values()), len(previous["photos"]))
    print(f"{len(pairs)} candidates, {len(photos)} full frames")


def summarize(root):
    folder = root / "data/state-pairs"
    key = read(folder / "key.json")
    automatic, tokens, provenance = select(root, extended=True)
    if (automatic != key["pairs"] or provenance != key["feature_sha256"] or
            tokens != {r["sample_id"]: t for t, r in key["photos"].items()}):
        raise ValueError("Candidate selection or source cache changed")
    blind, decisions = read(folder / "blind-review.json"), read(folder / "adjudication.json")
    rematched = read(folder / "manual-pairs.json")
    pairs = automatic + rematched
    if (len({p["pair"] for p in pairs}) != len(pairs) or
            len({(p["code"], p["positive"], p["negative"]) for p in pairs}) != len(pairs)):
        raise ValueError("Duplicate pair")
    for pair in rematched:
        for side in ("positive", "negative"):
            if pair[side] not in {r[side] for r in automatic if r["code"] == pair["code"]}:
                raise ValueError("Post-review pair must reuse reviewed photos of the same polarity")
    if (set(blind) != set(tokens.values()) or not all(blind.values()) or
            set(decisions) != {p["pair"] for p in pairs}):
        raise ValueError("Incomplete observations or adjudication")
    for row in decisions.values():
        if (not row["reason"] or row["incident"] not in {"same", "different", "unresolved"} or
                row["state"] not in {"usable_visual_contrast", "unresolved"} or
                (row["state"] == "usable_visual_contrast" and not row.get("relation"))):
            raise ValueError("Incomplete pair decision")
    counts = {}
    for code in CODES:
        selected = [p for p in pairs if p["code"] == code]
        rows = [decisions[p["pair"]] for p in selected]
        usable = [p for p in selected if decisions[p["pair"]]["state"] == "usable_visual_contrast"]
        counts[code] = {
            "candidate_pairs": len(rows), "model_reviewed_pairs": len(rows),
            "embedding_selected": sum(p in automatic for p in selected),
            "post_review_rematched": sum(p in rematched for p in selected),
            "usable_embedding_selected": sum(p in automatic for p in usable),
            "usable_post_review_rematched": sum(p in rematched for p in usable),
            "usable_visual_contrasts": len(usable), "unresolved_state_comparisons": len(rows) - len(usable),
            "visible_relation_counts": dict(Counter(decisions[p["pair"]]["relation"] for p in usable)),
            "incident_relationship": {v: sum(r["incident"] == v for r in rows)
                                      for v in ("same", "different", "unresolved")},
            "human_reviewed_pairs": 0, "human_unreviewed_pairs": len(rows)}
    result = {"previously_exposed_pool": 170, "final_holdout_exclusions_checked": 182, "new_exposures": 0,
              "candidate_pairs": len(pairs), "model_reviewed_unique_full_frames": len(blind),
              "model_reviewed_pairs": len(decisions), "human_reviewed_pairs": 0,
              "human_unreviewed_pairs": len(decisions),
              "usable_visual_contrasts": sum(r["usable_visual_contrasts"] for r in counts.values()),
              "unresolved_state_comparisons": sum(r["unresolved_state_comparisons"] for r in counts.values()),
              "by_hazard": counts}
    write(root / "results/official_state_contrasts.json", result)
    print(f"{result['usable_visual_contrasts']} usable visual contrasts across {len(pairs)} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extend-lifting", "summarize"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.action == "summarize":
        summarize(args.root)
    else:
        prepare(args.root, args.action == "extend-lifting")
