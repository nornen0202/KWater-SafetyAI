"""Paired CPU replay of same-photo Qwen text availability and fixed-code output."""

from collections import Counter
from copy import deepcopy
import time

from .adapter_scale_pilot import load_entry
from .metrics import decode_target
from .phase2_agreement import PIN, SPEC
from .phase2_panel import PROJECT, RUN, file_hash, read, registry, write
from .phase2_paths import file_stem
from .phase2_score import readlines, source_comparison, summarize
from .submission2_output import make_row


def compose(image_id, codes, capture, mapping, clip):
    fixed = make_row(image_id, list(codes))
    for item in fixed["hazards"].values():
        item.pop("references", None)
    generated = deepcopy(fixed)
    raw = capture["raw"] if capture["valid"] else {}
    texts = {}
    for h in raw.get("hazards", []):
        code = mapping.get(h["code"])
        evidence, prevention = clip(h["evidence"], 240), clip(h["prevention"], 480)
        if code and evidence and prevention and code not in texts:
            texts[code] = dict(evidence=evidence, prevention=prevention)
    disposition = {}
    for code in codes:
        if code in texts:
            generated["hazards"][code].update(texts[code])
            disposition[code] = "same_photo_and_code_generated"
        else:
            disposition[code] = "fixed_fallback_missing_same_photo_code"
    if not codes:
        scene = clip(raw.get("scene_note", ""), 240)
        if not capture["final"]["hazards"] and scene:
            generated["scene_evidence"] = scene
            disposition["no_visible_hazard"] = "same_photo_nohaz_generated"
        else:
            disposition["no_visible_hazard"] = "fixed_fallback_no_same_status_scene"
    if decode_target(generated) != decode_target(fixed):
        raise ValueError("Text replay changed the frozen codes")
    return fixed, generated, disposition


def replay():
    folder = RUN / "cached-writing-28"
    start = time.perf_counter()
    if file_hash(SPEC) != PIN:
        raise ValueError("The exposed 28-photo list changed")
    ids = read(SPEC)["already_generated"]
    rows = registry()
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
    }
    if set(rows) != labels.keys():
        raise ValueError("Expected only the 920 registered development samples")
    source = {
        r["image_id"]: r for r in readlines(RUN / "fusion-panel/submission.jsonl")
    }
    entry = load_entry(PROJECT / "data/baseline")
    cells = {f"{d}_{w}": {} for d in ("A", "S2") for w in ("fixed", "cached_generated")}
    counts = {d: Counter() for d in ("A", "S2")}
    records = []
    for sid in ids:
        path = RUN / "A" / (file_stem(sid) + ".json")
        capture = read(path)
        image_id = capture["final"]["image_id"]
        for detector, row in (("A", capture["final"]), ("S2", source[image_id])):
            fixed, generated, disposition = compose(
                image_id, row["hazards"], capture, entry.CODE_MAP, entry.clip
            )
            cells[detector + "_fixed"][sid] = fixed
            cells[detector + "_cached_generated"][sid] = generated
            counts[detector].update(disposition.values())
            records.append(
                dict(
                    sample_id=sid,
                    group=rows[sid]["group"],
                    detector=detector,
                    qwen_sha256=file_hash(path),
                    qwen_valid=capture["valid"],
                    writing_source=disposition,
                    fixed=fixed,
                    cached_generated=generated,
                )
            )
    metrics = {
        k: summarize(ids, predictions, labels, rows) for k, predictions in cells.items()
    }
    for d in counts:
        if metrics[d + "_fixed"] != metrics[d + "_cached_generated"]:
            raise ValueError("Writing method changed a code metric")
    write(folder / "rows.json", records)
    articles = source_comparison(cells, labels, rows, folder / "source-ledger.json")
    result = dict(
        scope="Previously exposed 28. Unprompted same-image raw Qwen text reused only for an exact mapped code, otherwise retain fixed text. A conditional GPU writer is a separate arm.",
        seconds=time.perf_counter() - start,
        photos=len(ids),
        groups=len({rows[s]["group"] for s in ids}),
        frozen_review_overlap=len(set(ids) & set(read(RUN / "plan.json")["review"])),
        generation_parse_failures=sum(
            not r["qwen_valid"] for r in records if r["detector"] == "A"
        ),
        code_metrics=metrics,
        writing_sources=counts,
        sources=articles,
        human_ratings=None,
        factual_support=None,
        prevention_adequacy=None,
        legal_entailment=None,
        new_gpu_calls=0,
        holdout_opened=False,
        official_score=None,
    )
    write(folder / "metrics.json", result)
    print({d: dict(v) for d, v in counts.items()}, result["seconds"], flush=True)


if __name__ == "__main__":
    replay()
