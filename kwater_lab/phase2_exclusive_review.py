"""Freeze and replay the exclusive proposal strata inside the original 64-photo panel."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import permutations

from .phase2_panel import RUN, file_hash, order, read, registry, write
from .phase2_review import SYSTEM, review_prompt
from .phase2_paths import file_stem


ROOT = RUN / "exclusive-review"
SPEC = ROOT / "design.json"
DONORS = ROOT / "training-donors.json"
HEADS = ("dino", "large_concat")
LIMIT = 144
GENERATION_FIELDS = (
    "text",
    "seconds",
    "new_tokens",
    "prompt_tokens",
    "vision_grid",
    "limit_reached",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


def frozen(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f"Frozen record changed: {path}")
    else:
        write(path, data)


def shuffled(ids, codes, head):
    def key(p):
        return (
            sum(codes[t] == codes[s] for s, t in zip(ids, p)),
            sum(s == t for s, t in zip(ids, p)),
            order("p38-exclusive-v1/" + head, "/".join(p)),
        )

    best = min(permutations(ids), key=key)
    assigned = {s: codes[t] for s, t in zip(ids, best, strict=True)}
    assert Counter(assigned.values()) == Counter(codes.values())
    return assigned


def freeze():
    plan = read(RUN / "plan.json")
    offers = read(RUN / "matched-budget-predictions.json")
    rows = registry()
    panel = plan["panel"]
    design = {}
    for head, other, expected in (
        ("dino", "large_concat", 6),
        ("large_concat", "dino", 4),
    ):
        ids = [
            s
            for s in panel
            if offers[head][s]["called"] and not offers[other][s]["called"]
        ]
        assert len(ids) == expected
        codes = {s: offers[head][s]["candidate"] for s in ids}
        swapped = shuffled(ids, codes, head)
        for s in ids:
            design[s] = dict(
                group=rows[s]["group"],
                fold=rows[s]["dev_fold"],
                head=head,
                true=codes[s],
                swapped=swapped[s],
                original16=s in plan["review"],
            )
    templates = {
        code: review_prompt(["__ORIGINAL__"], code).replace(
            "['__ORIGINAL__']", "{original_code_list}"
        )
        for code in sorted({v["true"] for v in design.values()})
    }
    spec = dict(
        panel_order=[s for s in panel if s in design],
        rows=design,
        source_sha256={
            p: file_hash(RUN / p)
            for p in (
                "plan.json",
                "matched-budget-predictions.json",
                "dev-registry.json",
            )
        },
        system=SYSTEM,
        templates=templates,
        max_new_tokens=LIMIT,
        do_sample=False,
        seed=42,
        image="original EXIF-transposed RGB; official processor pixel budget 384*32*32",
        model="original Qwen3-VL-4B official BF16 LoRA scale 1.0",
        donor_rule="Use recipient outer-training groups only; exclude frozen 64 and prototype 18; unique top-one proposal among eleven from the same head; hash tie break.",
        decision="Original 3 codes: proposals only. Otherwise apply the same 144-token read to both hints (including already present); accepted novel supported code only. Equal prompts share a generation.",
        scope="Two diagnostic strata scored separately, excluding release/promotion decisions.",
    )
    frozen(SPEC, spec)
    print("FROZEN", file_hash(SPEC), len(design), flush=True)


def donors():
    import numpy as np
    from .metrics import CODES
    from .phase2_panel import PHASE1, SIGNATURE
    from .train_head import fit, probabilities

    spec, rows = read(SPEC), registry()
    plan = read(RUN / "plan.json")
    exclude = set(plan["panel"]) | set(plan["prototype"])
    cache = read(RUN / "dev-heads.json")
    features = {}
    for s, r in rows.items():
        digest = sha256((SIGNATURE + r["sha256"]).encode()).hexdigest()
        a = np.load(
            PHASE1 / "features" / (digest + ".npy"), mmap_mode="r", allow_pickle=False
        )
        assert a.shape == (6, 17, 128)
        features[s] = a[0, 0].astype(np.float32)
    used, assignments = set(), {}
    for head in HEADS:
        batches = sorted(
            {v["fold"] for v in spec["rows"].values() if v["head"] == head}
        )
        for fold in batches:
            recipients = [
                s
                for s in spec["panel_order"]
                if spec["rows"][s]["head"] == head and spec["rows"][s]["fold"] == fold
            ]
            training = cache["train_ids_by_fold"][str(fold)]
            assert all(rows[t]["dev_fold"] != fold for t in training)
            assert not {rows[t]["group"] for t in training} & {
                rows[s]["group"] for s in recipients
            }
            x = np.stack(
                [
                    features[s][64:] if head == "dino" else features[s]
                    for s in training + recipients
                ]
            )
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            y = np.array(
                [[c in rows[t]["codes"] for c in CODES] for t in training], dtype=float
            )
            z = probabilities(fit(x[: len(training)], y, 0.01, 250), x)
            np.testing.assert_allclose(
                z[len(training) :],
                [[cache["scores"][s][head][c] for c in CODES] for s in recipients],
                rtol=1e-6,
                atol=1e-6,
            )
            top = {
                s: (CODES[int(v.argmax())], float(v.max())) for s, v in zip(training, z)
            }
            for s in recipients:
                code = spec["rows"][s]["swapped"]
                t = min(
                    (
                        k
                        for k in training
                        if k not in exclude | used and top[k][0] == code
                    ),
                    key=lambda k: order("p38-exclusive-donor/" + s, k),
                )
                used.add(t)
                assignments[s] = dict(
                    sample_id=t,
                    group=rows[t]["group"],
                    fold=rows[t]["dev_fold"],
                    recipient_fold=fold,
                    code=code,
                    rank_of_eleven=1,
                    score=top[t][1],
                    head=head,
                    fit_ids_sha256=sha256("\n".join(training).encode()).hexdigest(),
                )
    assert len(used) == len(assignments) == 10
    frozen(DONORS, dict(design_sha256=file_hash(SPEC), rows=assignments))
    print("TRAINING_DONORS", file_hash(DONORS), len(used), flush=True)


def first_large_result():
    for arm in ("sam31-frame", "sam3-frame", "sam3-image"):
        path = RUN / "regions" / ("detect-" + arm) / "public-example.json"
        if path.is_file() and read(path).get("status") == "ok":
            return str(path)
    for model in ("8b", "9b"):
        path = RUN / ("lora-" + model) / "results.json"
        if path.is_file():
            return str(path)
    return None


def paired(model=None, proc=None, image_for=None):
    from .phase2_official import generate
    from .phase2_review import apply_review

    spec = read(SPEC)
    if not DONORS.exists():
        return
    provenance = read(DONORS)
    assert provenance["design_sha256"] == file_hash(SPEC)
    first = first_large_result()
    for s in spec["panel_order"]:
        stem = file_stem(s) + ".json"
        path = RUN / "A" / stem
        if not path.is_file():
            continue
        original = read(path)["final"]
        r = spec["rows"][s]
        for arm in sorted(
            ("true", "swapped"), key=lambda a: order("exclusive-call/" + s, a)
        ):
            dest = ROOT / arm / stem
            if dest.is_file():
                continue
            hint = r[arm]
            prompt = review_prompt(original["hazards"], hint)
            assert prompt == spec["templates"][hint].replace(
                "{original_code_list}", str(list(original["hazards"]))
            )
            record = dict(
                sample_id=s,
                head=r["head"],
                arm=arm,
                hint=hint,
                base_codes=list(original["hazards"]),
                original_sha256=file_hash(path),
                system_sha256=sha256(SYSTEM.encode()).hexdigest(),
                prompt=prompt,
                max_new_tokens=LIMIT,
                additional_physical_calls=0,
                donor=provenance["rows"][s] if arm == "swapped" else None,
            )
            final = original
            status = "cap_proposal_only"
            if len(original["hazards"]) < 3:
                matches = [RUN / k / stem for k in "BCDE"] + [
                    ROOT / a / stem for a in ("true", "swapped") if a != arm
                ]
                generation = None
                for existing in matches:
                    if not existing.is_file():
                        continue
                    cached = read(existing)
                    if (
                        cached.get("prompt") != prompt
                        or cached.get("hint") != hint
                        or list(original["hazards"]) != cached.get("base_codes")
                    ):
                        continue
                    if cached.get("status") == "generation_failure":
                        # Preserve the failure for both arms when an identical call already failed.
                        record.update(
                            reused_from=existing.relative_to(RUN).as_posix(),
                            error=cached.get("error", "generation failed"),
                        )
                        break
                    if all(k in cached for k in GENERATION_FIELDS):
                        generation = {k: cached[k] for k in GENERATION_FIELDS}
                        record["reused_from"] = existing.relative_to(RUN).as_posix()
                        break
                if generation is None and "error" not in record:
                    if model is None or not first:
                        continue
                    import time

                    started = time.perf_counter()
                    record.update(
                        additional_physical_calls=1, large_model_receipt=first
                    )
                    try:
                        generation = generate(
                            model, proc, image_for(s), SYSTEM, prompt, LIMIT
                        )
                    except (ValueError, RuntimeError) as err:
                        record.update(
                            error=str(err),
                            failed_call_seconds=time.perf_counter() - started,
                        )
                if generation is None:
                    status = "generation_failure"
                else:
                    record.update(generation)
                    status, final = apply_review(original, generation, hint)
            record.update(status=status, final=final)
            write(dest, record)
    score()


def score():
    from .phase2_release import changes
    from .phase2_score import readlines, summarize

    spec = read(SPEC)
    paths = {
        s: {
            "A": RUN / "A" / (file_stem(s) + ".json"),
            **{a: ROOT / a / (file_stem(s) + ".json") for a in ("true", "swapped")},
        }
        for s in spec["panel_order"]
    }
    samples = {
        s: {a: read(p) for a, p in arms.items() if p.is_file()}
        for s, arms in paths.items()
    }
    labels = {
        v["sample_id"]: v["target"]["hazards"]
        for v in readlines(RUN / "dev-train.jsonl")
        if v["sample_id"] in paths
    }
    result = {}
    ledger = {}
    for head in HEADS:
        ids = [s for s in spec["panel_order"] if spec["rows"][s]["head"] == head]
        complete = [s for s in ids if len(samples[s]) == 3]
        result[head] = dict(
            planned=len(ids),
            originals=sum("A" in samples[s] for s in ids),
            paired=len(complete),
            groups=len({spec["rows"][s]["group"] for s in complete}),
            arms={},
            contrasts={},
        )
        for arm in ("A", "true", "swapped"):
            log = [samples[s][arm] for s in complete]
            result[head]["arms"][arm] = dict(
                saved=sum(arm in samples[s] for s in ids),
                schema_and_severity=summarize(
                    complete,
                    {s: samples[s][arm]["final"] for s in complete},
                    labels,
                    spec["rows"],
                )
                if complete
                else None,
                logical_calls=sum(v.get("status") != "cap_proposal_only" for v in log),
                marginal_new_gpu_calls=sum(
                    v.get("additional_physical_calls", 0) for v in log
                )
                if arm != "A"
                else 0,
                decision_counts=dict(Counter(v.get("status", "original") for v in log)),
                logical_generation_seconds=sum(
                    v.get("seconds", v.get("failed_call_seconds", 0)) for v in log
                ),
                marginal_new_gpu_seconds=sum(
                    v.get("seconds", v.get("failed_call_seconds", 0))
                    for v in log
                    if v.get("additional_physical_calls")
                ),
            )
        for before, after in (("A", "true"), ("A", "swapped"), ("swapped", "true")):
            totals = Counter()
            contrast = after + "_minus_" + before
            for s in complete:
                delta = changes(
                    samples[s][before]["final"], samples[s][after]["final"], labels[s]
                )
                ledger.setdefault(
                    s, dict(group=spec["rows"][s]["group"], head=head, contrasts={})
                )["contrasts"][contrast] = delta
                totals.update(
                    {
                        k: len(delta[k])
                        for k in ("gained_tp", "lost_tp", "new_fp", "removed_fp")
                    }
                )
            result[head]["contrasts"][contrast] = dict(
                **{
                    k: totals[k]
                    for k in ("gained_tp", "lost_tp", "new_fp", "removed_fp")
                },
                net_correct_recovery=totals["gained_tp"] - totals["lost_tp"],
            )
    write(ROOT / "paired-rows.json", ledger)
    write(
        ROOT / "metrics.json",
        dict(
            scope="Two separately reported diagnostic strata from the original frozen 64. Unobserved pairs excluded from deltas.",
            strata=result,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "donors", "replay", "score"))
    args = p.parse_args()
    {"freeze": freeze, "donors": donors, "replay": paired, "score": score}[args.stage]()
