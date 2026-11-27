"""Freeze diagnostic panels and compare proposal heads on the same outer folds."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import time

import numpy as np

from .phase2_io import (
    PROJECT as PROJECT,
    PHASE1 as PHASE1,
    RUN as RUN,
    read as read,
    write as write,
)
from .adapter_scale_pilot import sample
from .metrics import CODES
from .train_head import file_hash, fit, probabilities, save_atomic


SIGNATURE = "phase01-r2-sig384-dino224-grid4-proj64-seed404-tiles60pct-training-jpeg"
ALLOWED8 = tuple(
    c
    for c in CODES
    if c not in ("drowning_inundation", "thermal_weather", "confined_space")
)


def order(key, value):
    return hashlib.sha256((key + "/" + value).encode()).hexdigest()


def registry():
    p = RUN / "dev-registry.json"
    if (
        file_hash(p)
        != "968aff06acfea68892ae1b35bce604d2fb244c4ec205f7836722e254bacefa68"
    ):
        raise ValueError("Development-only registry changed")
    return {r["sample_id"]: r for r in read(p)}


def project_development():
    source = PHASE1 / "evaluation-manifest.json"
    if (
        file_hash(source)
        != "e9b46b79042bf7ef6d33213cfbcf135423168a9acfe7f637ec49ebb0eefe567d"
    ):
        raise ValueError("Phase 1 manifest changed")
    allowed = {
        json.loads(line)["sample_id"]
        for line in (PHASE1 / "dev_fold-oof.jsonl").read_bytes().splitlines()
    }
    only = [r for r in read(source) if r["sample_id"] in allowed]
    assert len(only) == len(allowed) == 920 and all(r["dev_fold"] >= 0 for r in only)
    data = (json.dumps(only, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    gold = b"".join(
        line + b"\n"
        for line in (PROJECT / "data/train/train.jsonl").read_bytes().splitlines()
        if json.loads(line)["sample_id"] in allowed
    )
    for name, content, expected in (
        (
            "dev-registry.json",
            data,
            "968aff06acfea68892ae1b35bce604d2fb244c4ec205f7836722e254bacefa68",
        ),
        (
            "dev-train.jsonl",
            gold,
            "d098a7c397464893568d24aca4dacc8a61447b02f81ae9baa17f0f658193e7fe",
        ),
    ):
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Changed development projection: {name}")
        target = RUN / name
        if not target.exists():
            save_atomic(target, lambda f: f.write(content))
        if file_hash(target) != expected:
            raise ValueError(f"Changed development file: {name}")


def freeze():
    rows = registry()
    groups = defaultdict(list)
    for sid, row in rows.items():
        if not row["exposed"]:
            groups[row["group"]].append(sid)
    for g, ids in groups.items():
        ids.sort(key=lambda s: order("p38-panel", s))
    panel = []
    for n in range(64):
        for g in sorted(groups, key=lambda v: order("p38-groups", v)):
            if len(groups[g]) > n and len(panel) < 64:
                panel.append(groups[g][n])
    counts = Counter(rows[s]["group"] for s in panel)
    prototype = [
        s for s, _ in sample(read(PROJECT / "data/prefix-pilot/exposure.json"))
    ]
    train = sorted(
        [
            s
            for s in rows
            if rows[s]["dev_fold"] != 0 and s not in panel and s not in prototype
        ],
        key=lambda s: order("p38-train", s),
    )[:32]
    test = sorted(
        [s for s in panel if rows[s]["dev_fold"] == 0],
        key=lambda s: order("p38-test", s),
    )
    review = sorted(panel, key=lambda s: order("p38-review", s))[:16]
    plan = {
        "manifest_sha256": file_hash(PHASE1 / "evaluation-manifest.json"),
        "seed": 42,
        "development_rows": len(rows),
        "development_groups": len({r["group"] for r in rows.values()}),
        "prototype": prototype,
        "panel": panel,
        "review": review,
        "lora_train": train,
        "lora_test": test,
        "panel_probability_by_group": {
            g: counts[g] / len(ids) for g, ids in groups.items()
        },
        "review_probability_given_panel": 16 / 64,
        "eligible_by_group": {g: len(ids) for g, ids in groups.items()},
        "review_policy": "all 16 incl cap, no labels, one attempt per arm; cap=3 proposal only",
        "lora_policy": "fold 0 evaluation, 32 seed-hash-selected other-fold training rows, 16 updates accum2; 1x token weights, rank 8, alpha 16, dropout 0, last 4 language MLP down_proj, max 128 visual tokens, greedy 96 tokens; no reasoning",
        "proposal_policy": "single top new organizer-supported code, L2 .01 250 iterations, no threshold selection",
        "permutation_policy": "after A and before B/C/D; cross-group bipartite assignment, minimize duplicates and original predictions, seed-hash tie breaks; no validation labels",
        "exposure": "panel excludes 182 historically exposed photo IDs; official LoRA still training-exposed; Phase1 already evaluated development groups",
    }
    assert len(set(panel)) == 64 and not set(panel) & set(prototype)
    assert not {rows[s]["group"] for s in train} & {rows[s]["group"] for s in test}
    dest = RUN / "plan.json"
    if dest.exists() and read(dest) != plan:
        raise ValueError("Frozen plan differs")
    write(dest, plan)
    print(
        "FROZEN",
        len(panel),
        len(review),
        len(train),
        len(test),
        file_hash(dest),
        flush=True,
    )


def freeze_sft():
    plan, rows = read(RUN / "plan.json"), registry()
    pool = sorted(
        (
            s
            for s, r in rows.items()
            if r["dev_fold"] != 0
            and s not in plan["panel"]
            and s not in plan["prototype"]
        ),
        key=lambda s: order("p38-train", s),
    )
    support = Counter(c for s in pool for c in rows[s]["codes"])
    required = []
    for c in sorted(support, key=lambda k: (support[k], k)):
        if not any(c in rows[s]["codes"] for s in required):
            required.append(next(s for s in pool if c in rows[s]["codes"]))
    selected = sorted(
        (required + [s for s in pool if s not in required])[:32],
        key=lambda s: order("p38-train", s),
    )
    split = {
        "plan_sha256": file_hash(RUN / "plan.json"),
        "train": selected,
        "test": plan["lora_test"],
        "reason": "Initial label-blind 32 training photos covered only 5 codes. Before any SFT or evaluation, reserve one positive per available outer-training code, then fill the same 32-row budget by the original order.",
        "outer_training_support": support,
        "selected_support": Counter(c for s in selected for c in rows[s]["codes"]),
        "unavailable_training_codes": [c for c in CODES if c not in support],
        "original_training32": plan["lora_train"],
    }
    dest = RUN / "sft-split.json"
    if dest.exists() and read(dest) != split:
        raise ValueError("SFT split changed")
    write(dest, split)
    print("SFT_FROZEN", split["selected_support"], file_hash(dest), flush=True)


def feature_inputs(rows, extend):
    from PIL import Image
    from .clip_probe import ImageEncoder, session

    raw = {}
    times = {}
    for name, filename, output in [
        ("clip", "features.npz", "image_embeds"),
        ("sig", "siglip2-base16-features.npz", "pooler_output"),
    ]:
        with np.load(PROJECT / "data/prefix-pilot" / filename) as f:
            d = dict(zip(f["ids"].tolist(), f["features"]))
        if extend:
            bundle = PROJECT / "submission/fusion"
            sig224 = PROJECT / "data/encoders/siglip2-base-p16-224"
            cfg = read(
                sig224 / "preprocessor_config.json"
                if name == "sig"
                else bundle / "clip.json"
            )
            weights = (
                sig224 / "vision_model_quantized.onnx"
                if name == "sig"
                else bundle / "clip.onnx"
            )
            enc = ImageEncoder.__new__(ImageEncoder)
            enc.edge = cfg["size"].get("shortest_edge") or cfg["size"]["height"]
            enc.mean, enc.std = (
                np.asarray(cfg[k], np.float32) for k in ("image_mean", "image_std")
            )
            assert enc.edge == 224
            enc.output_name, enc.model = output, None
            start = time.perf_counter()
            for s, r in rows.items():
                if s in d:
                    continue
                cache = (
                    RUN
                    / "features"
                    / ("sig224" if name == "sig" else name)
                    / (r["sha256"] + ".npy")
                )
                if cache.exists():
                    d[s] = np.load(cache, allow_pickle=False)
                else:
                    if enc.model is None:
                        if (
                            name == "sig"
                            and file_hash(weights)
                            != "5f2b401c1a4fc095702a5d45348e17ad46c4f87064085365b43c6e8eaa5c0070"
                        ):
                            raise ValueError("Unexpected SigLIP2-224 weights")
                        enc.model = session(weights)
                    with Image.open(PROJECT / "data/train" / r["image"]) as im:
                        d[s] = enc.encode(im, "full_frame")
                    save_atomic(cache, lambda f: np.save(f, d[s]))
            times[name] = time.perf_counter() - start
        raw[name] = d
    ids = sorted(set(raw["sig"]) & set(raw["clip"]) & set(rows))
    for name in ("large_sig", "dino", "large_concat"):
        raw[name] = {}
    for sid in ids:
        digest = hashlib.sha256((SIGNATURE + rows[sid]["sha256"]).encode()).hexdigest()
        features = np.load(PHASE1 / "features" / (digest + ".npy"), allow_pickle=False)
        assert features.shape == (6, 17, 128)
        pooled = features[0, 0].astype(np.float32)  # full-frame projected global token
        for key, v in [
            ("large_sig", pooled[:64]),
            ("dino", pooled[64:]),
            ("large_concat", pooled),
        ]:
            raw[key][sid] = v
    return ids, raw, times


def heads(extend):
    rows = registry()
    ids, raw, times = feature_inputs(rows, extend)
    plan = read(RUN / "plan.json")
    predictions = {s: {} for s in ids}
    provenance = {}
    for fold in range(5):
        train = [s for s in ids if rows[s]["dev_fold"] != fold]
        test = [s for s in ids if rows[s]["dev_fold"] == fold]
        if not test:
            continue
        assert not {rows[s]["group"] for s in train} & {rows[s]["group"] for s in test}
        provenance[fold] = train
        y = np.array(
            [[c in rows[s]["codes"] for c in CODES] for s in train], dtype=float
        )
        for name, d in raw.items():
            start = time.perf_counter()
            x = np.stack([d[s] for s in train + test])
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            head = fit(x[: len(train)], y, 0.01, 250)
            scores = probabilities(head, x[len(train) :])
            times[name + "_fit_score"] = (
                times.get(name + "_fit_score", 0) + time.perf_counter() - start
            )
            for sid, z in zip(test, scores):
                predictions[sid][name] = dict(zip(CODES, z.tolist()))
    tag = "dev" if extend else "cache"
    write(
        RUN / (tag + "-heads.json"),
        {"train_ids_by_fold": provenance, "scores": predictions, "seconds": times},
    )
    a = {
        x["image_id"]: set(x["hazards"])
        for x in map(
            json.loads,
            (PROJECT / "data/adapter-scale-pilot/replay-1.0/out/submission.jsonl")
            .read_bytes()
            .splitlines(),
        )
    }
    stats = {}
    for name in raw:
        tp = fp = 0
        for s in plan["prototype"]:
            base = a[order("kwater/scale", s)[:32]]
            ranks = predictions[s][name]
            top = max((c for c in ALLOWED8 if c not in base), key=ranks.get)
            if len(base) < 3:
                tp += int(top in rows[s]["codes"])
                fp += int(top not in rows[s]["codes"])
        stats[name] = {"oracle_new_tp": tp, "raw_new_fp": fp}
    write(
        RUN / (tag + "-comparison.json"),
        {"n": len(ids), "prototype18": stats, "seconds": times},
    )
    print(tag, len(ids), stats, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["project-development", "freeze", "sft-split", "cache", "dev"]
    )
    args = parser.parse_args()
    if args.stage == "project-development":
        project_development()
    elif args.stage == "freeze":
        freeze()
    elif args.stage == "sft-split":
        freeze_sft()
    else:
        heads(args.stage == "dev")
