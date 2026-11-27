"""Compare Tiny and Large SAM2.1 on exactly the same frozen Grounding DINO Base boxes."""

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import time
import numpy as np
from PIL import Image, ImageOps

from .phase2_panel import RUN, file_hash, order, read, registry, write
from .phase2_paths import file_stem
from .phase2_regions import ROOT, Segmenter, image_for, pixels
from .phase2_region_review import mask_variants, prepare_views
from .visual_probe import DEFAULT_CACHE, FILES


DEST = ROOT / "base-sam-size"
MODEL_ARMS = {
    "large": RUN / "large-models/sam2-large-onnx",
    "tiny": ROOT / "models/segmenter",
}
VIEWS = ("whole", "real_crop", "mask", "rectangle", "swapped_mask")


def matched_swaps():
    """Freeze additional cross-group donors with visible masks before any Qwen region reread."""
    spec, rows = read(ROOT / "design.json"), registry()
    ids = spec["review"]
    pool = sorted({spec["donors"][s] for s in ids})
    assert not set(pool) & set(read(RUN / "plan.json")["panel"])
    boxes = {
        s: read(ROOT / "detect-base" / (file_stem(s) + ".json"))["selection"]
        for s in ids + pool
    }
    choices = {}
    for s in ids:
        if boxes[s]["box"] is None:
            continue
        options = []
        for d in pool:
            if rows[d]["dev_fold"] == rows[s]["dev_fold"] or boxes[d]["box"] is None:
                continue
            statuses = [
                mask_variants(s, d, boxes[s], boxes[d], "masks-base-" + arm)[1]
                for arm in MODEL_ARMS
            ]
            if all(
                v["status"] == "matched" and len(set(v["areas"].values())) == 1
                for v in statuses
            ):
                same_role = bool(boxes[d]["person"]) == bool(boxes[s]["person"])
                options.append(
                    (
                        d != spec["donors"][s],
                        not same_role,
                        order("p38-mask-reserve/" + s, d),
                        d,
                    )
                )
        choices[s] = [v[-1] for v in sorted(options)]
    assigned = {}

    # Maximum-cardinality matching. Fewest available alternatives first, then fixed hash order.
    def match(s, seen):
        for d in choices[s]:
            if d in seen:
                continue
            seen.add(d)
            if d not in assigned or match(assigned[d], seen):
                assigned[d] = s
                return True
        return False

    for s in sorted(
        choices, key=lambda s: (len(choices[s]), order("p38-mask-recipient", s))
    ):
        match(s, set())
    mapping = {s: d for d, s in assigned.items()}
    measures = {}
    for s, d in mapping.items():
        measures[s] = {
            a: mask_variants(s, d, boxes[s], boxes[d], "masks-base-" + a)[1]
            for a in MODEL_ARMS
        }
    manifest = dict(
        original_donors={s: spec["donors"][s] for s in ids},
        pool=pool,
        eligible=choices,
        donors=mapping,
        areas=measures,
        rules="Fixed donor pool only, never the recipient outer fold. Unique donors. Same recipient box and pixel area in both versions. No outcome labels.",
    )
    dest = DEST / "matched-swap-design.json"
    if dest.exists() and read(dest) != manifest:
        raise ValueError("Frozen matched-mask design changed")
    write(dest, manifest)
    print("MATCHED_MASKS", len(mapping), len(choices), flush=True)


def study_views(output_root):
    return VIEWS + (
        ("matched_swapped_mask",)
        if (output_root / "matched-swap-design.json").is_file()
        else ()
    )


def job():
    spec, rows = read(ROOT / "design.json"), registry()
    public = DEFAULT_CACHE / "public-example.jpg"
    assert file_hash(public) == FILES[public.name]
    recipients = spec["review"]
    ids = ["public-example"] + recipients + [spec["donors"][s] for s in recipients]
    for name, model_path in MODEL_ARMS.items():
        t = time.perf_counter()
        model = Segmenter(model_path)
        cold = time.perf_counter() - t
        write(
            DEST / name / "runtime.json",
            dict(
                provider=model.enc.get_providers(),
                load_seconds=cold,
                model_path=str(model_path),
                checkpoint_precision="int8",
                vram_bytes=0,
            ),
        )
        print("SAM_LOADED", name, cold, flush=True)
        for s in ids:
            output = DEST / name / "rows" / (file_stem(s) + ".json")
            if output.is_file() and not read(output).get("error"):
                continue
            if s == "public-example":
                with Image.open(public) as source:
                    pic = ImageOps.exif_transpose(source).convert("RGB")
                box = [0.25, 0.25, 0.75, 0.75]
                origin = (
                    "prespecified central box on the public image; runtime check only"
                )
            else:
                box_record = ROOT / "detect-base" / (file_stem(s) + ".json")
                deadline = time.monotonic() + 1800
                while not box_record.is_file() and time.monotonic() < deadline:
                    time.sleep(5)
                if not box_record.is_file() or "selection" not in read(box_record):
                    write(
                        output,
                        dict(
                            status="failed",
                            error="Missing Base detector result",
                            sample_id=s,
                            box=None,
                            seconds=0,
                        ),
                    )
                    continue
                box = read(box_record)["selection"]["box"]
                pic = image_for(rows[s])
                origin = "unchanged Grounding DINO Base selection"
            t = time.perf_counter()
            try:
                out = model.segment(
                    pic, box, ROOT / ("masks-base-" + name) / (file_stem(s) + ".png")
                )
            except Exception as err:
                out = dict(status="failed", error=repr(err))
            out.update(
                seconds=time.perf_counter() - t,
                sample_id=s,
                box=box,
                box_source=origin,
                size=pic.size,
            )
            write(output, out)
            print(
                "SAM_ROW",
                name,
                s,
                out["status"],
                out.get("area"),
                round(out["seconds"], 3),
                flush=True,
            )
        del model
        gc.collect()
    stats = {}
    for name in MODEL_ARMS:
        saved = {
            s: read(DEST / name / "rows" / (file_stem(s) + ".json"))
            for s in recipients
            if (DEST / name / "rows" / (file_stem(s) + ".json")).exists()
        }
        called = [v["seconds"] for v in saved.values() if v["box"] is not None]
        stats[name] = dict(
            rows=len(saved),
            status=dict(Counter(v["status"] for v in saved.values())),
            conditional_calls=len(called),
            conditional_seconds=sum(called),
            called_p50_p95_seconds=np.percentile(called, [50, 95]).tolist()
            if called
            else None,
            load=read(DEST / name / "runtime.json"),
        )
    overlap = []
    for s in ["public-example"] + recipients:
        paths = [
            ROOT / ("masks-base-" + key) / (file_stem(s) + ".png") for key in MODEL_ARMS
        ]
        if all(p.is_file() for p in paths):
            masks = []
            for p in paths:
                with Image.open(p) as im:
                    masks.append(np.asarray(im).astype(bool))
            union = (masks[0] | masks[1]).sum()
            overlap.append(
                dict(
                    sample_id=s,
                    areas=[int(m.sum()) for m in masks],
                    iou=float((masks[0] & masks[1]).sum() / union) if union else None,
                )
            )
    write(
        DEST / "engineering.json",
        dict(arms=stats, same_box_masks=overlap, human_mask_truth=None),
    )


def call_costs():
    spec, rows = read(ROOT / "design.json"), registry()
    report = {}
    for name, model_path in MODEL_ARMS.items():
        start = time.perf_counter()
        model = Segmenter(model_path)
        cold = time.perf_counter() - start
        samples = []
        for s in spec["review"]:
            pic = image_for(rows[s])
            box = read(ROOT / "detect-base" / (file_stem(s) + ".json"))["selection"][
                "box"
            ]
            t = time.perf_counter()
            if box is None:
                model.enc.run(None, {"pixel_values": pixels(pic, model.cfg)})
            else:
                again = model.segment(
                    pic,
                    box,
                    DEST / name / "all-encoder-masks" / (file_stem(s) + ".png"),
                )
                observed = read(DEST / name / "rows" / (file_stem(s) + ".json"))
                assert again["sha256"] == observed["sha256"]
            samples.append(
                dict(
                    sample_id=s,
                    needs_decoder=box is not None,
                    seconds=time.perf_counter() - t,
                )
            )
            write(DEST / name / "always-encoder-rows.json", samples)
        report[name] = dict(
            encoder_calls=len(samples),
            decoder_calls=sum(v["needs_decoder"] for v in samples),
            load_seconds=cold,
            seconds=sum(v["seconds"] for v in samples),
            p50_p95=np.percentile([v["seconds"] for v in samples], [50, 95]).tolist(),
        )
        del model
        gc.collect()
    write(DEST / "always-encoder-metrics.json", report)


def paired(model, proc, *, output_root=DEST, sources=None):
    from .phase2_official import generate
    from .phase2_review import SYSTEM, interpret, review_prompt
    from .submission2_output import make_row
    from .metrics import decode_target

    rows, spec = registry(), read(ROOT / "design.json")
    sources = sources or {
        name: ("detect-base", "masks-base-" + name) for name in MODEL_ARMS
    }
    variants = study_views(output_root)
    design_file = output_root / "matched-swap-design.json"
    swap_map = read(design_file)["donors"] if design_file.is_file() else {}
    for s in spec["review"]:
        t = spec["donors"][s]
        base = read(RUN / "A" / (file_stem(s) + ".json"))["final"]
        target = spec["target"][s]
        generated = {}
        for name, (boxes, masks) in sources.items():
            try:
                pic = image_for(rows[s])
                views, measure = prepare_views(s, t, pic, boxes, masks, arms=VIEWS)
                if "matched_swapped_mask" in variants:
                    if s in swap_map:
                        swap, info = prepare_views(
                            s, swap_map[s], pic, boxes, masks, arms=("swapped_mask",)
                        )
                        views["matched_swapped_mask"] = swap["swapped_mask"]
                        measure["matched_swap"] = dict(
                            donor=swap_map[s], areas=info["masks"]["areas"]
                        )
                    elif measure["selection"]["box"] is None:
                        views["matched_swapped_mask"] = views["swapped_mask"]
                        measure["matched_swap"] = dict(status="no_recipient_box")
            except (KeyError, OSError, ValueError) as err:
                for arm in variants:
                    out = output_root / name / "paired" / file_stem(s) / (arm + ".json")
                    if not out.exists():
                        write(
                            out,
                            dict(
                                sample_id=s,
                                condition=arm,
                                status="failed_visual",
                                error=repr(err),
                                final=base,
                                shared_identical_visual=False,
                                seconds_executed=0,
                            ),
                        )
                continue
            write(output_root / name / "views" / (file_stem(s) + ".json"), measure)
            for arm in sorted(variants, key=lambda a: order(s, a)):
                out = output_root / name / "paired" / file_stem(s) / (arm + ".json")
                if out.exists():
                    old = read(out)
                    if "rgb_sha256" in old:
                        generated[old["rgb_sha256"]] = old
                    continue
                if arm not in views:
                    write(
                        out,
                        dict(
                            sample_id=s,
                            condition=arm,
                            status="no_matched_donor",
                            final=base,
                            shared_identical_visual=False,
                            seconds_executed=0,
                        ),
                    )
                    continue
                im = views[arm]
                digest = hashlib.sha256(im.tobytes()).hexdigest()
                if digest in generated:
                    rec = deepcopy(generated[digest])
                    rec.update(
                        condition=arm, shared_identical_visual=True, seconds_executed=0
                    )
                    write(out, rec)
                    continue
                final = deepcopy(base)
                obj = None
                rec = {}
                if len(base["hazards"]) >= 3:
                    status = "cap_proposal_only"
                elif target in base["hazards"]:
                    status = "already_present"
                else:
                    try:
                        rec = generate(
                            model, proc, im, SYSTEM, review_prompt(base, target), 144
                        )
                        status, obj = interpret(
                            rec["text"], set(base["hazards"]), target
                        )
                        if obj:
                            item = make_row(base["image_id"], [target])["hazards"][
                                target
                            ]
                            item.pop("references", None)
                            item.update(
                                evidence=obj["evidence"].strip(),
                                prevention=obj["prevention"].strip(),
                            )
                            final["hazards"][target] = item
                            final["status"] = "hazard"
                            final.pop("scene_evidence", None)
                            decode_target(final)
                    except (ValueError, RuntimeError, TypeError) as err:
                        rec.update(error=repr(err))
                        status, final = "failed", deepcopy(base)
                rec.update(
                    condition=arm,
                    candidate=target,
                    status=status,
                    final=final,
                    rgb_sha256=digest,
                    shared_identical_visual=False,
                    seconds_executed=rec.get("seconds", 0),
                )
                write(out, rec)
                generated[digest] = rec


def score(*, output_root=DEST, arms=MODEL_ARMS):
    from .phase2_score import readlines, summarize
    from .phase2_release import changes
    from .metrics import paired_bootstrap

    rows, spec = registry(), read(ROOT / "design.json")
    ids = spec["review"]
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in ids
    }
    base = {s: read(RUN / "A" / (file_stem(s) + ".json"))["final"] for s in ids}
    summary = {}
    predictions = {}
    for model in arms:
        for v in study_views(output_root):
            saved = {
                s: read(output_root / model / "paired" / file_stem(s) / (v + ".json"))
                for s in ids
            }
            output = {s: r["final"] for s, r in saved.items()}
            predictions[model + "/" + v] = {
                s: frozenset(r["hazards"]) for s, r in output.items()
            }
            counts = Counter()
            for s in ids:
                c = changes(base[s], output[s], labels[s])
                for key in ("gained_tp", "new_fp", "lost_tp", "removed_fp"):
                    counts[key] += len(c[key])
            summary[model + "/" + v] = dict(
                metrics=summarize(ids, output, labels, rows),
                changes=dict(counts),
                statuses=dict(Counter(r["status"] for r in saved.values())),
                distinct_generation_seconds=sum(
                    r["seconds_executed"] for r in saved.values()
                ),
                identical_input_reuses=sum(
                    r["shared_identical_visual"] for r in saved.values()
                ),
                human_factual_support=None,
            )
    ci = {}
    gold, groups = (
        {s: frozenset(labels[s]) for s in ids},
        {s: rows[s]["group"] for s in ids},
    )
    for model in arms:
        for variant in study_views(output_root)[1:]:
            ci[model + "/" + variant] = paired_bootstrap(
                gold,
                predictions[model + "/whole"],
                predictions[model + "/" + variant],
                groups,
            )
    write(output_root / "paired-metrics.json", summary)
    write(output_root / "paired-groups.json", ci)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage",
        nargs="?",
        choices=("segment", "call-costs", "matched-swaps", "score"),
        default="segment",
    )
    stage = ap.parse_args().stage
    {
        "segment": job,
        "call-costs": call_costs,
        "matched-swaps": matched_swaps,
        "score": score,
    }[stage]()
