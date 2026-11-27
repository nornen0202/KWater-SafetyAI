"""Independent, sealed-policy CPU replay of the manager's two frozen proposal caches."""

import argparse
import time

from .metrics import CODES, evaluate
from .phase2_panel import RUN, file_hash, read, registry, write
from .phase2_paths import file_stem
from .phase2_score import readlines


SPEC = RUN / "manager-agreement-prereg.json"
DEST = RUN / "agreement-independent"
PIN = "dbf82cc5d5dd3675bcbe5c3cc01778f96fbc86f0f682b3fcf468da92272717b0"


def additions(original, top, frozen):
    proposed = [top] if top not in original and len(original) < 3 else []
    return {
        "pristine": [],
        "all_top1": proposed,
        "confirmed": [c for c in proposed if c in frozen],
    }


def summarize(ledger, ids, arm):
    if not ids:
        return None
    gold = {s: frozenset(ledger[s]["gold"]) for s in ids}
    groups = {s: ledger[s]["group"] for s in ids}
    added = {s: set(ledger[s]["additions"][arm]) for s in ids}
    pred = {s: frozenset(ledger[s]["base"]) | v for s, v in added.items()}
    metric = evaluate(gold, pred, groups)
    return dict(
        photos=len(ids),
        groups=len(set(groups.values())),
        actual_group_ids=sorted(set(groups.values())),
        valid_base=sum(ledger[s]["base_valid"] for s in ids),
        active_codes=metric["active_codes"],
        macro_active=metric["macro_f1"],
        macro_fixed11=sum(r["f1"] or 0 for r in metric["per_code"].values())
        / len(CODES),
        per_code=metric["per_code"],
        metrics=dict(
            calls=sum(map(len, added.values())),
            new_true=sum(len(v & gold[s]) for s, v in added.items()),
            new_false=sum(len(v - gold[s]) for s, v in added.items()),
            normal_n=metric["no_hazard_photos"],
            normal_alarm=metric["no_hazard_false_alarms"],
            new_normal_alarm=sum(
                not gold[s] and not ledger[s]["base"] and bool(added[s]) for s in ids
            ),
            severe_n=sum(bool(ledger[s]["severe"]) for s in ids),
            severe_missed=sum(
                bool(ledger[s]["severe"]) and not (pred[s] & set(ledger[s]["severe"]))
                for s in ids
            ),
            exact=metric["exact_set_matches"],
        ),
        added_by_code={
            c: dict(
                true=sum(c in added[s] and c in gold[s] for s in ids),
                false=sum(c in added[s] and c not in gold[s] for s in ids),
            )
            for c in CODES
        },
    )


def replay(watch=False):
    if file_hash(SPEC) != PIN:
        raise ValueError("Agreement preregistration changed")
    spec = read(SPEC)
    for name, expected in spec["input_sha256"].items():
        if file_hash(RUN / name) != expected:
            raise ValueError(f"Frozen input differs: {name}")
    cohorts = ("already_generated", "remaining")
    fixed = {k: set(spec[k]) for k in cohorts}
    assert len(fixed[cohorts[0]]) == 28 and len(fixed[cohorts[1]]) == 36
    assert not fixed[cohorts[0]] & fixed[cohorts[1]] and set.union(
        *fixed.values()
    ) == set(spec["panel"])
    rows = registry()
    top = read(RUN / "matched-budget-predictions.json")["large_concat"]
    confirm = read(RUN / "prior-refit/predictions.json")
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
    }
    source = {s: RUN / "A" / (file_stem(s) + ".json") for s in spec["panel"]}
    ledger = {}
    while True:
        changed = False
        for s, path in source.items():
            if s in ledger or not path.is_file():
                continue
            raw = read(path)
            if raw["sample_id"] != s:
                raise ValueError("Source image differs")
            base = sorted(raw["final"]["hazards"])
            candidate = top[s]["candidate"]
            decisions = additions(base, candidate, confirm[s]["class"])
            ledger[s] = dict(
                base=base,
                base_valid=raw["valid"],
                source_sha256=file_hash(path),
                group=rows[s]["group"],
                prior_class=confirm[s]["class"],
                candidate=candidate,
                additions=decisions,
                gold=sorted(labels[s]),
                severe=sorted(
                    c
                    for c, h in labels[s].items()
                    if h["risk_level"]["severity"]["score"] == 4
                ),
            )
            changed = True
        if changed:
            report = {}
            for k in cohorts:
                known = sorted(fixed[k] & ledger.keys())
                report[k] = {
                    arm: summarize(ledger, known, arm)
                    for arm in ("pristine", "all_top1", "confirmed")
                }
            for arm, observed in report[cohorts[0]].items():
                reference = read(RUN / "manager-agreement-28.json")["cohorts"][
                    cohorts[0]
                ][arm]
                if (
                    observed["metrics"] != reference["metrics"]
                    or abs(observed["macro_fixed11"] - reference["macro_fixed11"])
                    > 1e-12
                ):
                    raise ValueError(
                        "Initial cohort differs from the preregistered comparison"
                    )
            complete = fixed["remaining"] <= ledger.keys()
            a, b = report["remaining"]["pristine"], report["remaining"]["confirmed"]
            reproduction = bool(
                complete
                and b["macro_fixed11"] > a["macro_fixed11"]
                and b["metrics"]["normal_alarm"] <= a["metrics"]["normal_alarm"]
                and b["metrics"]["severe_missed"] < a["metrics"]["severe_missed"]
            )
            planned_groups = {
                k: {rows[s]["group"] for s in ids} for k, ids in fixed.items()
            }
            actual_groups = {
                k: {rows[s]["group"] for s in ids & ledger.keys()}
                for k, ids in fixed.items()
            }
            review16 = set(read(RUN / "plan.json")["review"])
            exclusive = read(RUN / "exclusive-review/design.json")["rows"]
            coverage = {}
            for k in cohorts:
                triggers = {
                    s
                    for s in fixed[k] & ledger.keys()
                    if ledger[s]["additions"]["confirmed"]
                }
                coverage[k] = dict(
                    triggered=sorted(triggers),
                    in16=sorted(triggers & review16),
                    in10=sorted(triggers & exclusive.keys()),
                    in10_same_true_hint=sorted(
                        s
                        for s in triggers & exclusive.keys()
                        if exclusive[s]["true"] == ledger[s]["candidate"]
                    ),
                    outside_both=sorted(triggers - review16 - exclusive.keys()),
                )
            write(DEST / "rows.json", ledger)
            write(
                DEST / "metrics.json",
                dict(
                    spec_sha256=PIN,
                    cohorts=report,
                    planned_denominators=[28, 36],
                    observed_denominators=[
                        len(fixed[k] & ledger.keys()) for k in cohorts
                    ],
                    planned_shared_groups=sorted(
                        set.intersection(*planned_groups.values())
                    ),
                    observed_shared_groups=sorted(
                        set.intersection(*actual_groups.values())
                    ),
                    remaining_complete=complete,
                    evidence_plan_coverage=coverage,
                    reproduced_for_evidence_check=reproduction,
                    scope="Class codes only. Published adapter exposure; cohorts share groups.",
                ),
            )
        if not watch or len(ledger) == 64:
            return
        time.sleep(15)


def legacy():
    from .phase2_panel import PROJECT, order

    reference = read(RUN / "manager-agreement-legacy18.json")
    if reference["prereg"] != PIN or file_hash(SPEC) != PIN:
        raise ValueError("Legacy replay preregistration differs")
    plan = read(RUN / "plan.json")
    ids = plan["prototype"]
    rows = registry()
    assert not {rows[s]["group"] for s in ids} & {
        rows[s]["group"] for s in plan["panel"]
    }
    top = read(RUN / "matched-budget-predictions.json")["large_concat"]
    prior = read(RUN / "prior-refit/predictions.json")
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
    }
    results = {}
    for scale, ref in reference["scales"].items():
        path = (
            PROJECT
            / "data/adapter-scale-pilot"
            / ("replay-" + scale)
            / "out/submission.jsonl"
        )
        assert file_hash(path) == ref["archive_sha"]
        original = {r["image_id"]: r for r in readlines(path)}
        ledger = {}
        for s in ids:
            base = sorted(original[order("kwater/scale", s)[:32]]["hazards"])
            ledger[s] = dict(
                base=base,
                group=rows[s]["group"],
                base_valid=True,
                gold=sorted(labels[s]),
                severe=sorted(
                    c
                    for c, h in labels[s].items()
                    if h["risk_level"]["severity"]["score"] == 4
                ),
                additions=additions(base, top[s]["candidate"], prior[s]["class"]),
            )
        write(DEST / ("legacy-" + scale + "-rows.json"), ledger)
        results[scale] = {
            arm: summarize(ledger, ids, arm)
            for arm in ("pristine", "all_top1", "confirmed")
        }
        for arm, measured in results[scale].items():
            expected = ref["metrics"][arm]
            assert abs(expected["macro_fixed11"] - measured["macro_fixed11"]) < 1e-12
            assert (
                expected["counts"]["normal_alarm"]
                == measured["metrics"]["normal_alarm"]
            )
    write(
        DEST / "legacy-metrics.json",
        dict(
            exposed=True,
            photos=len(ids),
            groups=len({rows[s]["group"] for s in ids}),
            groups_overlapping_64=0,
            scales=results,
        ),
    )


def evidence(model, proc, image_for):
    from collections import Counter
    from hashlib import sha256
    from .phase2_official import generate
    from .phase2_review import CONDITIONS, SYSTEM, interpret, review_prompt

    replay()
    if not read(DEST / "metrics.json")["remaining_complete"]:
        return
    rows = read(DEST / "rows.json")
    plan = read(SPEC)
    groups = {}
    for cohort in ("already_generated", "remaining"):
        stats = Counter()
        by_truth = {"true": Counter(), "false": Counter()}
        elapsed = []
        for sid in plan[cohort]:
            rows[sid]["additions"]["image_confirmed"] = []
            for code in rows[sid]["additions"]["confirmed"]:
                stem = file_stem(sid)
                dest = DEST / "evidence" / (stem + "-" + code + ".json")
                if not dest.exists():
                    path = RUN / "A" / (stem + ".json")
                    if file_hash(path) != rows[sid]["source_sha256"]:
                        raise ValueError(
                            "Original generation changed before the evidence run"
                        )
                    original = read(path)["final"]["hazards"]
                    conditions = (
                        CONDITIONS
                        if code in CONDITIONS
                        else dict(
                            CONDITIONS,
                            drowning_inundation="물과 작업자 위치 및 추락·침수 경로",
                            confined_space="밀폐된 공간·출입 경로와 작업 위치",
                            thermal_weather="현장에서 확인되는 온열·한랭·기상 노출",
                        )
                    )
                    prompt = review_prompt(original, code, conditions)
                    matches = []
                    for arm in (
                        "B",
                        "C",
                        "D",
                        "E",
                        "exclusive-review/true",
                        "exclusive-review/swapped",
                    ):
                        prior_path = RUN / arm / (stem + ".json")
                        if not prior_path.is_file():
                            continue
                        prior = read(prior_path)
                        if (
                            prior.get("sample_id") == sid
                            and prior.get("hint") == code
                            and prior.get("base_codes") == list(original)
                            and prior.get("prompt") == prompt
                            and prior.get("max_new_tokens", 144) == 144
                            and prior.get(
                                "system_sha256", sha256(SYSTEM.encode()).hexdigest()
                            )
                            == sha256(SYSTEM.encode()).hexdigest()
                            and prior.get("original_sha256", rows[sid]["source_sha256"])
                            == rows[sid]["source_sha256"]
                        ):
                            matches.append((str(prior_path.relative_to(RUN)), prior))
                    before = time.perf_counter()
                    if matches:
                        generated = matches[0][1]
                        if any(
                            v.get("text") != generated.get("text")
                            or v.get("status") != generated.get("status")
                            for _, v in matches
                        ):
                            generated = {
                                "error": "The exact same original photo and prompt have inconsistent prior judgments"
                            }
                    else:
                        try:
                            generated = generate(
                                model, proc, image_for(sid), SYSTEM, prompt, 144
                            )
                        except (RuntimeError, ValueError, OSError) as e:
                            generated = {"error": repr(e)}
                    status, detail = (
                        interpret(generated["text"], set(original), code, conditions)
                        if "text" in generated
                        else ("failure", None)
                    )
                    write(
                        dest,
                        dict(
                            sample_id=sid,
                            cohort=cohort,
                            code=code,
                            base_sha256=rows[sid]["source_sha256"],
                            prompt=prompt,
                            system_sha256=sha256(SYSTEM.encode()).hexdigest(),
                            max_new_tokens=144,
                            image="original RGB",
                            status=status,
                            reused=[k for k, _ in matches],
                            evidence=detail["evidence"] if detail else None,
                            prevention=detail["prevention"] if detail else None,
                            generation={
                                k: v
                                for k, v in generated.items()
                                if k
                                in (
                                    "text",
                                    "error",
                                    "new_tokens",
                                    "limit_reached",
                                    "peak_allocated_bytes",
                                    "seconds",
                                )
                            },
                            additional_calls=int(not matches),
                            additional_seconds=0
                            if matches
                            else time.perf_counter() - before,
                        ),
                    )
                rec = read(dest)
                stats[rec["status"]] += 1
                by_truth["true" if code in rows[sid]["gold"] else "false"][
                    rec["status"]
                ] += 1
                if rec["status"] == "accept":
                    rows[sid]["additions"]["image_confirmed"].append(code)
                if rec["additional_calls"]:
                    elapsed.append(rec["additional_seconds"])
        groups[cohort] = dict(
            photo_code_pairs=sum(stats.values()),
            status=dict(stats),
            additional_calls=len(elapsed),
            additional_seconds=sum(elapsed),
            per_call_seconds=elapsed,
            decisions_by_candidate_truth={k: dict(v) for k, v in by_truth.items()},
            accepted_codes_vs_original=summarize(rows, plan[cohort], "image_confirmed"),
        )
    write(
        DEST / "evidence-cost.json",
        dict(
            cohorts=groups, human_support_rating=None, official_qualitative_score=None
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--legacy", action="store_true")
    args = p.parse_args()
    if args.legacy:
        legacy()
    else:
        replay(args.watch)
