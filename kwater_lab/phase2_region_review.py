"""Construct matched RGB views from local model output and evaluate paired decisions."""

from collections import Counter
from copy import deepcopy
import hashlib

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .phase2_paths import file_stem
from .phase2_panel import RUN, order, read, registry, write
from .phase2_regions import ROOT, image_for


ARMS = (
    "whole",
    "random_crop",
    "real_crop",
    "swapped_crop",
    "mask",
    "rectangle",
    "swapped_mask",
    "depth",
    "swapped_depth",
)
EDGE = 384


def bounds(norm, w, h):
    x0, y0, x1, y1 = map(int, np.rint(np.asarray(norm) * [w, h, w, h]))
    x0, y0 = min(w - 1, max(0, x0)), min(h - 1, max(0, y0))
    return x0, y0, max(x0 + 1, min(w, x1)), max(y0 + 1, min(h, y1))


def move_crop(rect, s):
    u, v = (int(order("region-random/" + s, a)[:12], 16) / 16**12 for a in ("x", "y"))
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    x, y = (1 - w) * u, (1 - h) * v
    return [x, y, x + w, y + h]


def tile(pic):
    return ImageOps.pad(
        pic.convert("RGB"),
        (EDGE, EDGE),
        method=Image.Resampling.BICUBIC,
        color=(105, 105, 105),
    )


def relative_overlap(field, sel):
    if not sel["pair_visible"]:
        return {"status": "missing_pair"}
    field = field.resize((EDGE, EDGE), Image.Resampling.BILINEAR)
    a = np.asarray(field)
    vals = []
    for key in ("person", "partner"):
        x, y, u, v = bounds(sel[key]["xyxy"], field.width, field.height)
        vec = a[y:v, x:u].ravel() / 255
        if len(vec) < 20:
            return {"status": "too_few_pixels"}
        vals.append(vec)
    bins = np.linspace(0, 1, 17)
    hist = [np.histogram(v, bins)[0] / len(v) for v in vals]
    return {
        "status": "both_detected",
        "person_minus_partner_median": float(np.median(vals[0]) - np.median(vals[1])),
        "histogram_intersection": float(np.minimum(*hist).sum()),
        "pixels": [len(v) for v in vals],
        "sampling_grid": [EDGE, EDGE],
    }


def map_image(folder, key):
    with Image.open(ROOT / folder / (file_stem(key) + ".png")) as im:
        return im.convert("L")


def mask_variants(s, t, sel, donor_sel, mask_folder="masks"):
    # Every overlay uses the same 384x384 coordinate grid and recipient detector box.
    side = EDGE
    out = {
        a: np.zeros((side, side), np.uint8)
        for a in ("mask", "rectangle", "swapped_mask")
    }
    details = {}
    if sel["box"] is None:
        return out, {"status": "no_recipient_box"}
    x, y, u, v = bounds(sel["box"], side, side)
    w, h = u - x, v - y
    area = (
        np.asarray(
            map_image(mask_folder, s).resize((side, side), Image.Resampling.NEAREST)
        )[y:v, x:u]
        > 0
    )
    n = int(area.sum())
    out["mask"][y:v, x:u] = area
    details["recipient_area"] = n
    if not n:
        return out, dict(details, status="empty_true_mask")
    # Centred rectangle, scaled to the true area and box aspect; resolve one last row by stable order.
    yy, xx = np.ogrid[:h, :w]
    d = np.maximum(np.abs(2 * (xx + 0.5) / w - 1), np.abs(2 * (yy + 0.5) / h - 1))
    rectangle = np.zeros((h, w), np.uint8)
    rectangle.flat[np.argsort(d.ravel(), kind="stable")[:n]] = 1
    out["rectangle"][y:v, x:u] = rectangle
    if donor_sel["box"] is None:
        return out, dict(details, status="no_donor_box")
    mask = map_image(mask_folder, t)
    mask = mask.crop(bounds(donor_sel["box"], *mask.size)).resize(
        (w, h), Image.Resampling.BILINEAR
    )
    raw = np.asarray(mask)
    if not raw.any():
        return out, dict(details, status="empty_donor_mask")
    smooth = (
        np.asarray(mask.filter(ImageFilter.GaussianBlur(max(w, h) / 16))).astype(float)
        / 256
    )
    rank = raw.astype(float) + smooth
    donor = np.zeros((h, w), np.uint8)
    donor.flat[np.argsort(-rank.ravel(), kind="stable")[:n]] = 1
    out["swapped_mask"][y:v, x:u] = donor
    return out, dict(
        details, status="matched", areas={k: int(a.sum()) for k, a in out.items()}
    )


def prepare_views(
    s, t, image, detect_folder="detect", mask_folder="masks", *, arms=ARMS
):
    sel = read(ROOT / detect_folder / (file_stem(s) + ".json"))["selection"]
    donor = read(ROOT / detect_folder / (file_stem(t) + ".json"))["selection"]
    masks, desc = mask_variants(s, t, sel, donor, mask_folder)
    truth = sel["crop"]
    rectangles = {
        "whole": [0, 0, 1, 1],
        "random_crop": move_crop(truth, s),
        "swapped_crop": donor["crop"] if sel["box"] is not None else truth,
    }
    panels = {}
    base = tile(image)
    depths = {}
    for arm in arms:
        rect = rectangles.get(arm, truth)
        region = image.crop(bounds(rect, *image.size))
        if arm in masks:
            mask = Image.fromarray(masks[arm] * np.uint8(105))
            rect2 = bounds(rect, EDGE, EDGE)
            mask_roi = mask.crop(rect2).resize(region.size, Image.Resampling.BILINEAR)
            region = Image.composite(
                Image.new("RGB", region.size, (255, 190, 10)), region, mask_roi
            )
        if arm in ("depth", "swapped_depth") and sel["box"] is not None:
            field = map_image("depth-maps", s if arm == "depth" else t)
            depths[arm] = relative_overlap(field, sel)
            field = field.crop(bounds(rect, *field.size)).resize(
                region.size, Image.Resampling.BILINEAR
            )
            region = Image.blend(region, field.convert("RGB"), 0.40)
        full = Image.new("RGB", (2 * EDGE, EDGE))
        full.paste(base, (0, 0))
        full.paste(tile(region), (EDGE, 0))
        panels[arm] = full
    return panels, {
        "recipient": s,
        "donor": t,
        "selection": sel,
        "donor_selection": donor,
        "crop_controls": rectangles,
        "masks": desc,
        "relative_depth": depths,
        "routing": "no detected recipient box uses two unmodified RGB tiles in every arm",
        "full_rgb_tile_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
        "view_sha256": {
            a: hashlib.sha256(im.tobytes()).hexdigest() for a, im in panels.items()
        },
    }


def prepare(s, t, image):
    p = ROOT / "views" / file_stem(s)
    receipt = p / "receipt.json"
    if receipt.exists():
        prior = read(receipt)
        if prior.get("routing") and prior["donor"] == t:
            views = {}
            for arm in ARMS:
                with Image.open(p / (arm + ".png")) as saved:
                    views[arm] = saved.convert("RGB")
                if (
                    hashlib.sha256(views[arm].tobytes()).hexdigest()
                    != prior["view_sha256"][arm]
                ):
                    raise ValueError(f"Region view changed for {s}, {arm}")
            return views
    views, measure = prepare_views(s, t, image)
    p.mkdir(parents=True, exist_ok=True)
    for key, im in views.items():
        im.save(p / (file_stem(key) + ".png"))
    write(p / "receipt.json", measure)
    return views


def base_depth_views(s, t, im):
    sel = read(ROOT / "detect-base" / (file_stem(s) + ".json"))["selection"]
    if not sel["pair_visible"]:
        return {}, {}
    crop = sel["crop"]
    clean = im.crop(bounds(crop, *im.size))
    full = tile(im)
    views = {}
    stats = {}
    for name, source in (("no_depth", None), ("same_depth", s), ("swapped_depth", t)):
        region = clean
        if source:
            field = map_image("depth-maps", source)
            stats[name] = relative_overlap(field, sel)
            overlay = field.crop(bounds(crop, *field.size)).resize(
                clean.size, Image.Resampling.BILINEAR
            )
            region = Image.blend(clean, overlay.convert("RGB"), 0.4)
        pic = Image.new("RGB", (2 * EDGE, EDGE))
        pic.paste(full, (0, 0))
        pic.paste(tile(region), (EDGE, 0))
        views[name] = pic
    return views, {
        "selection": sel,
        "recipient": s,
        "donor": t,
        "depth": stats,
        "rgb_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
        "views": {
            k: hashlib.sha256(im.tobytes()).hexdigest() for k, im in views.items()
        },
    }


def base_depth(model=None, proc=None):
    from .phase2_review import SYSTEM, interpret, review_prompt
    from .phase2_official import generate

    rows = registry()
    spec = read(ROOT / "design.json")
    counts = Counter()
    for s in spec["review"]:
        t = spec["donors"][s]
        views, stats = base_depth_views(s, t, image_for(rows[s]))
        if not views:
            counts["no_pair"] += 1
            continue
        counts["pair"] += 1
        folder = ROOT / "base-depth" / file_stem(s)
        write(folder / "geometry.json", stats)
        for k, im in views.items():
            if not (folder / (k + ".png")).exists():
                im.save(folder / (k + ".png"))
            if model is None or (folder / (k + ".json")).exists():
                continue
            original = read(RUN / "A" / (file_stem(s) + ".json"))["final"]
            code = spec["target"][s]
            prompt = (
                "왼쪽은 전체 현장, 오른쪽은 보조 화면이다. 거리·체결·이동을 색이나 고리만으로 단정하지 않는다. "
                + review_prompt(original["hazards"], code)
            )
            try:
                result = generate(model, proc, im, SYSTEM, prompt, 144)
                status, obj = interpret(
                    result["text"], set(original["hazards"]) - {code}, code
                )
                result.update(status=status, supported=obj is not None)
            except (RuntimeError, ValueError) as e:
                result = {"status": "failed", "error": repr(e), "supported": False}
            write(
                folder / (k + ".json"),
                dict(
                    result,
                    scope="observational reading only",
                    target=code,
                    original_codes=list(original["hazards"]),
                ),
            )
    write(
        ROOT / "base-depth" / "coverage.json",
        {
            "frozen_n": len(spec["review"]),
            "counts": dict(counts),
            "interpretation": "visible person+target noun; no proof of suspension or distance",
        },
    )


def paired(model, proc, image_source):
    from .phase2_official import generate
    from .phase2_review import SYSTEM, interpret, review_prompt
    from .submission2_output import make_row
    from .metrics import decode_target

    spec = read(ROOT / "design.json")
    for s in spec["review"]:
        views = prepare(s, spec["donors"][s], image_source(s))
        base = read(RUN / "A" / (file_stem(s) + ".json"))["final"]
        existing = set(base["hazards"])
        target = spec["target"][s]
        prompt = (
            "왼쪽은 현장의 전경, 오른쪽은 보조 화면이다. 색·경계·투영 겹침·검출 누락만으로 거리를 단정하지 않는다. "
            + review_prompt(base["hazards"], target)
        )
        for arm in sorted(ARMS, key=lambda a: order(s, a)):
            dest = ROOT / "review" / arm / (file_stem(s) + ".json")
            if dest.exists():
                continue
            final = deepcopy(base)
            record = {"condition": arm, "candidate": target, "sample_id": s}
            status = "cap_proposal_only" if len(existing) >= 3 else "already_present"
            if len(existing) < 3 and target not in existing:
                try:
                    record.update(
                        generate(model, proc, views[arm], SYSTEM, prompt, 144)
                    )
                    status, obj = interpret(record["text"], existing, target)
                    if obj:
                        item = make_row(final["image_id"], [target])["hazards"][target]
                        item.pop("references", None)
                        item.update(
                            evidence=obj["evidence"].strip(),
                            prevention=obj["prevention"].strip(),
                        )
                        final["hazards"][target] = item
                        final["status"] = "hazard"
                        final.pop("scene_evidence", None)
                        decode_target(final)
                except (RuntimeError, ValueError, TypeError) as e:
                    status, record["error"], final = "failure", repr(e), deepcopy(base)
            write(dest, dict(record, status=status, final=final))
        print("REGION_REVIEW_SAVED", s, flush=True)
    base_depth(model, proc)
    from . import phase2_large_sam

    if (phase2_large_sam.DEST / "engineering.json").is_file():
        phase2_large_sam.paired(model, proc)
    from . import phase2_sam3

    direct = {
        arm: ("detect-" + arm, "masks-" + arm)
        for arm in phase2_sam3.ARMS
        if (phase2_sam3.DEST / arm / "engineering.json").is_file()
    }
    if direct:
        phase2_large_sam.paired(
            model, proc, output_root=phase2_sam3.DEST, sources=direct
        )
        phase2_large_sam.score(output_root=phase2_sam3.DEST, arms=direct)


def score():
    from .phase2_score import readlines, summarize
    from .metrics import paired_bootstrap

    rows = registry()
    spec = read(ROOT / "design.json")
    samples = spec["recipients"]
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in samples
    }
    baseline = {s: read(RUN / "A" / (file_stem(s) + ".json"))["final"] for s in samples}
    out = {}
    preds = {}
    for arm in ARMS:
        final = dict(baseline)
        ops = Counter()
        for s in spec["review"]:
            r = read(ROOT / "review" / arm / (file_stem(s) + ".json"))
            final[s] = r["final"]
            old, new, gold = map(
                set, [baseline[s]["hazards"], final[s]["hazards"], labels[s]]
            )
            ops.update(
                recovered_tp=len((new - old) & gold),
                added_fp=len((new - old) - gold),
                lost_tp=len((old - new) & gold),
                seconds=r.get("seconds", 0),
                calls=int("seconds" in r),
                parse_failure=int(r["status"] == "parse_failure"),
            )
        preds[arm] = {s: frozenset(final[s]["hazards"]) for s in samples}
        out[arm] = {
            "review16": summarize(spec["review"], final, labels, rows),
            "fixed_policy64": summarize(samples, final, labels, rows),
            "changes": dict(ops),
        }
    paired_ci = {}
    for a, b in (
        ("real_crop", "random_crop"),
        ("real_crop", "swapped_crop"),
        ("mask", "rectangle"),
        ("mask", "swapped_mask"),
        ("depth", "swapped_depth"),
        ("depth", "real_crop"),
    ):
        paired_ci[a + "-" + b] = paired_bootstrap(
            {s: frozenset(labels[s]) for s in samples},
            preds[b],
            preds[a],
            {s: rows[s]["group"] for s in samples},
        )
    write(
        ROOT / "scores.json",
        {"arms": out, "paired_diagnostic": paired_ci, "human_evidence_quality": None},
    )


def engineering():
    from .phase2_regions import ROLES

    spec = read(ROOT / "design.json")
    results = {}
    for cohort, ids in (
        ("frozen16", spec["review"]),
        ("frozen64", spec["recipients"]),
        ("training_donors16", [spec["donors"][s] for s in spec["review"]]),
    ):
        rr = [read(ROOT / "detect" / (file_stem(s) + ".json")) for s in ids]
        valid = [r for r in rr if "selection" in r]
        summary = {
            "rows": len(rr),
            "successful_rows": len(valid),
            "photo_counts_by_noun": {
                role: sum(
                    any(b["role"] == role for b in r["detections"]) for r in valid
                )
                for role in ROLES
            },
            "no_detection": sum(not r["detections"] for r in valid),
            "any_task_relevant_box": sum(
                r["selection"]["box"] is not None for r in valid
            ),
            "person_and_target": sum(r["selection"]["pair_visible"] for r in valid),
            "detector_seconds": sum(r["seconds"] for r in rr),
            "detector_p50": float(np.median([r["seconds"] for r in rr])),
        }
        if cohort != "frozen64":
            for stage in ("segment", "depth"):
                rr2 = [read(ROOT / stage / (file_stem(s) + ".json")) for s in ids]
                summary[stage] = {
                    "statuses": dict(Counter(r.get("status", "ok") for r in rr2)),
                    "seconds": sum(r["seconds"] for r in rr2),
                }
        results[cohort] = summary
    receipts = [
        read(ROOT / "views" / file_stem(s) / "receipt.json") for s in spec["review"]
    ]
    for r in receipts:
        assert "routing" in r
        if r["selection"]["box"] is None:
            assert len(set(r["view_sha256"].values())) == 1
    results["view_controls"] = {
        "total": len(receipts),
        "mask_comparison": dict(Counter(r["masks"]["status"] for r in receipts)),
        "actually_different_from_whole": {
            a: sum(r["view_sha256"][a] != r["view_sha256"]["whole"] for r in receipts)
            for a in ARMS
        },
        "tile_pixels": [2 * EDGE, EDGE],
        "human_grounding_scores": None,
        "qwen_hazard_scores": None,
    }
    write(ROOT / "engineering.json", results)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "score", "engineering", "base-depth"))
    p.add_argument("--limit", type=int, default=16)
    args = p.parse_args()
    if args.action == "score":
        score()
    elif args.action == "engineering":
        engineering()
    elif args.action == "base-depth":
        base_depth()
    else:
        rows = registry()
        spec = read(ROOT / "design.json")
        for s in spec["recipients"][: args.limit]:
            prepare(s, spec["donors"][s], image_for(rows[s]))
