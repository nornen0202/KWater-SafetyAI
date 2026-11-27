"""Pair all diagnostic rows, including fallbacks, and audit explanations separately."""

from collections import Counter
import json

from .law_examples import law_ids, select_articles, training_examples
from .law_retrieval import (
    ArticleIndex,
    article_texts,
    official_ranker_type,
    official_ranking,
)
from .metrics import CODES, decode_target, evaluate, paired_bootstrap
from .phase2_paths import file_stem
from .phase2_panel import (
    ALLOWED8,
    PROJECT,
    RUN,
    file_hash,
    order,
    read,
    registry,
    write,
)


def readlines(path):
    return list(map(json.loads, path.read_bytes().splitlines()))


def measured_cost(records):
    import numpy as np

    records = list(records)
    times = [r["seconds"] for r in records if "seconds" in r]
    return dict(
        registered_rows=len(records),
        timed_rows=len(times),
        seconds_p50_p95=np.percentile(times, [50, 95]).tolist() if times else None,
        max_allocated_bytes=max(
            (r["peak_allocated_bytes"] for r in records if "peak_allocated_bytes" in r),
            default=None,
        ),
        max_reserved_bytes=max(
            (r["peak_reserved_bytes"] for r in records if "peak_reserved_bytes" in r),
            default=None,
        ),
    )


def failed_attempts(ids):
    paths = {s: RUN / "A-attempts" / (file_stem(s) + ".json") for s in ids}
    return {s: read(p) for s, p in paths.items() if p.is_file()}


def deployed_comparison(ids, originals, labels, rows):
    from .phase2_release import changes

    fusion = {
        r["image_id"]: r for r in readlines(RUN / "fusion-panel/submission.jsonl")
    }
    second = {s: fusion[originals[s]["image_id"]] for s in ids}
    shifts = {s: changes(originals[s], second[s], labels[s]) for s in ids}
    write(RUN / "deployed-paired-rows.json", shifts)
    operations = ("gained_tp", "lost_tp", "new_fp", "removed_fp")
    return dict(
        rows=len(ids),
        groups=len({rows[s]["group"] for s in ids}),
        diagnostic="Same frozen images; official Qwen and submitted CLIP+SigLIP pipeline, training exposure applies.",
        A=summarize(ids, originals, labels, rows),
        S2=summarize(ids, second, labels, rows),
        S2_minus_A={k: sum(len(v[k]) for v in shifts.values()) for k in operations},
        per_code={
            k: dict(Counter(c for v in shifts.values() for c in v[k]))
            for k in operations
        },
        changed_photos=sum(v["codes_changed"] for v in shifts.values()),
    )


def summarize(ids, final, labels, rows):
    gold = {s: frozenset(labels[s]) for s in ids}
    pred = {s: decode_target(final[s])[1] for s in ids}
    groups = {s: rows[s]["group"] for s in ids}
    severe = {
        s: {
            c for c, h in labels[s].items() if h["risk_level"]["severity"]["score"] == 4
        }
        for s in ids
    }
    scored = evaluate(gold, pred, groups)
    lifting = "struck_by_lifting"
    pos = {s for s in ids if lifting in gold[s]}
    hits = {s for s in ids if lifting in pred[s]}
    severe_lifting = {s for s in pos if lifting in severe[s]}
    scored["lifting"] = dict(
        photos=len(ids),
        positive_photos=len(pos),
        positive_groups=len({groups[s] for s in pos}),
        predicted_photos=len(hits),
        true_positive=len(hits & pos),
        false_positive=len(hits - pos),
        false_negative=len(pos - hits),
        severity4_positive=len(severe_lifting),
        severity4_hits=len(severe_lifting & hits),
        severity4_misses=len(severe_lifting - hits),
        clear_photos=sum(not gold[s] for s in ids),
        clear_photos_with_lifting_alarm=sum(not gold[s] for s in hits),
    )
    scored["macro_f1_fixed_11_zero_undefined"] = sum(
        v["f1"] or 0 for v in scored["per_code"].values()
    ) / len(CODES)
    scored.update(
        severity4_photos=sum(bool(c) for c in severe.values()),
        severity4_completely_missed=sum(
            bool(severe[s]) and not (severe[s] & pred[s]) for s in ids
        ),
        schema_valid=len(pred),
    )
    return scored


def lifting_review(records, labels):
    code = "struck_by_lifting"
    counted = {"true": Counter(), "false": Counter()}
    for s, rec in records.items():
        if rec.get("hint", rec.get("candidate")) == code:
            counted["true" if code in labels[s] else "false"][rec["status"]] += 1
    return dict(
        registered_rows=len(records),
        hinted_rows=sum(sum(v.values()) for v in counted.values()),
        outcomes_by_hint_truth={k: dict(v) for k, v in counted.items()},
        denominator="Decisions on this exact lifting hint, including cap-only, reject, unsure and failure.",
    )


def paired_progress():
    """Report completed pairs with their true denominator; no adjustment of the frozen policy."""
    rows, plan = registry(), read(RUN / "plan.json")

    def entry(arm, s):
        return RUN / arm / (file_stem(s) + ".json")

    panel = [s for s in plan["panel"] if entry("A", s).is_file()]
    paired = [
        s for s in plan["review"] if all(entry(arm, s).is_file() for arm in "ABCDE")
    ]
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in panel
    }
    a = {s: read(entry("A", s)) for s in panel}
    summary = {
        "scope": "exposed official adapter; descriptive streaming pairs",
        "frozen_panel": len(plan["panel"]),
        "actual_A_rows": len(panel),
        "planned_review": len(plan["review"]),
        "paired_rows": len(paired),
        "A_failed_attempts": failed_attempts(plan["panel"]),
    }
    if panel:
        base = {s: r["final"] for s, r in a.items()}
        summary["A_so_far"] = summarize(panel, base, labels, rows)
        summary["lifting_reread"] = {}
        for arm in "BCDE":
            observed = {
                s: read(entry(arm, s))
                for s in plan["review"]
                if s in a and entry(arm, s).is_file()
            }
            summary["lifting_reread"][arm] = lifting_review(observed, labels)
        summary["A_seconds"] = sum(r["seconds"] for r in a.values())
        summary["A_measured_cost"] = measured_cost(a.values())
        summary["deployed_S2_vs_A"] = deployed_comparison(panel, base, labels, rows)
        summary["valid_generations"] = sum(r["valid"] for r in a.values())
    if paired:
        arms = {}
        for arm in "ABCDE":
            records = {s: read(entry(arm, s)) for s in paired}
            final = {s: r["final"] for s, r in records.items()}
            ops = Counter()
            for s, new in final.items():
                prior = set(base[s]["hazards"])
                now = set(new["hazards"])
                gold = set(labels[s])
                ops.update(
                    gained_tp=len((now - prior) & gold),
                    new_fp=len((now - prior) - gold),
                    lost_tp=len((prior - now) & gold),
                )
            arms[arm] = {
                "paired": summarize(paired, final, labels, rows),
                "changes": dict(ops),
                "seconds": sum(r.get("seconds", 0) for r in records.values()),
                "measured_cost": measured_cost(records.values()),
            }
        summary["arms"] = arms
    write(RUN / "review-progress.json", summary)
    return summary


def source_comparison(cells, labels, rows, ledger_path=None):
    source = PROJECT / "data/prefix-pilot"
    laws = readlines(source / "laws.jsonl")
    articles = select_articles(laws)
    make = official_ranker_type(source / "rag.py")
    body = ArticleIndex(article_texts(source / "laws.jsonl"), "bigram")
    methods = ("code_frequency", "fold_cases", "cases_body_agree")
    results = {
        name: {
            m: Counter(
                cited=0,
                correct_article_and_code=0,
                eligible_gold=0,
                wrong_code_cited=0,
                invalid=0,
            )
            for m in methods
        }
        for name in cells
    }
    ledger = []
    for fold in range(5):
        training = sorted(s for s, r in rows.items() if r["dev_fold"] != fold)
        examples, freqs = training_examples(training, labels, articles)
        retriever = make(laws, train=examples)
        for name, cell in cells.items():
            for s, output in cell.items():
                if rows[s]["dev_fold"] != fold:
                    continue
                for code, h in output["hazards"].items():
                    freq = freqs[code]
                    prior = sorted(freq, key=lambda x: (-freq[x], x))[:1]
                    learned = official_ranking(
                        retriever, h["evidence"], h["prevention"], code
                    )[:1]
                    # Agreement proxy only. No claim of legal or factual entailment.
                    agree = (
                        learned
                        if learned and learned[0] in body.rank(h["prevention"])[:3]
                        else []
                    )
                    gold = set(law_ids(labels[s].get(code, {})))
                    candidates = dict(zip(methods, (prior, learned, agree)))
                    for key, refs in candidates.items():
                        results[name][key].update(
                            cited=int(bool(refs)),
                            correct_article_and_code=int(bool(set(refs) & gold)),
                            eligible_gold=int(bool(gold)),
                            wrong_code_cited=int(bool(refs) and code not in labels[s]),
                            invalid=sum(r not in articles for r in refs),
                        )
                    ledger.append(
                        dict(
                            s=s,
                            cell=name,
                            code=code,
                            policies=candidates,
                            gold_refs=sorted(gold),
                            fold=fold,
                        )
                    )
    write(ledger_path or RUN / "source-ledger.json", ledger)
    return results


def score():
    rows, plan = registry(), read(RUN / "plan.json")
    ids = plan["panel"]
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in rows
    }
    review = plan["review"]
    baseline = {s: read(RUN / "A" / (file_stem(s) + ".json")) for s in ids}
    base = {s: r["final"] for s, r in baseline.items()}
    arms = {"A": base}
    result = {
        "run_id": RUN.name,
        "plan_sha256": file_hash(RUN / "plan.json"),
        "panel_n": len(ids),
        "review_n": len(review),
        "panel_groups": len({rows[s]["group"] for s in ids}),
        "official_adapter_scope": "training-domain diagnostic",
        "holdout_opened": False,
        "kit": read(RUN / "contracts.json"),
        "A": summarize(ids, base, labels, rows),
        "A_parse_failures": sum(not r["valid"] for r in baseline.values()),
        "A_failed_attempts": failed_attempts(ids),
        "A_seconds": sum(r["seconds"] for r in baseline.values()),
        "A_measured_cost": measured_cost(baseline.values()),
        "deployed_S2_vs_A": deployed_comparison(ids, base, labels, rows),
        "A_peak_allocated": max(r["peak_allocated_bytes"] for r in baseline.values()),
    }
    result["A_review_subset"] = summarize(review, base, labels, rows)
    progress18 = PROJECT / "data/adapter-scale-pilot/replay-1.0/out/submission.jsonl"
    cached = {r["image_id"]: r for r in readlines(progress18)}
    s = plan["prototype"][0]
    old = cached[order("kwater/scale", s)[:32]]
    new = read(RUN / "A" / (file_stem(s) + ".json"))["final"]
    result["replay_18_sentinel"] = {
        "codes_match": set(old["hazards"]) == set(new["hazards"]),
        "entire_row_match": old == new,
        "legacy_formatter": "18 cache used laws with no training examples; repeated A uses full submitted retriever",
    }
    donor = read(RUN / "review-assignment.json")
    scores = read(RUN / "dev-heads.json")["scores"]
    ledger = []
    counters = {}
    for arm in "BCDE":
        changes = Counter()
        statuses = Counter()
        final = dict(base)
        seconds = []
        peaks = []
        for s in review:
            r = read(RUN / arm / (file_stem(s) + ".json"))
            final[s] = r["final"]
            old, new, gold = (
                set(base[s]["hazards"]),
                set(final[s]["hazards"]),
                set(labels[s]),
            )
            added = new - old
            hit, fp, lost = added & gold, added - gold, (old - new) & gold
            hint = r["hint"]
            changes.update(
                recovered_tp=len(hit),
                new_fp=len(fp),
                existing_tp_lost=len(lost),
                candidate_was_missed_tp=int(hint in gold - old),
                changed=int(bool(added or old - new)),
                new_nohaz_alarm=int(not old and not gold and bool(new)),
            )
            statuses[r["status"]] += 1
            if "seconds" in r:
                seconds.append(r["seconds"])
                peaks.append(r["peak_allocated_bytes"])
            ledger.append(
                dict(
                    s=s,
                    group=rows[s]["group"],
                    fold=rows[s]["dev_fold"],
                    condition=arm,
                    old=sorted(old),
                    candidate=hint,
                    status=r["status"],
                    final=sorted(new),
                    gold=sorted(gold),
                    recovered=sorted(hit),
                    new_fp=sorted(fp),
                    lost=sorted(lost),
                    seconds=r.get("seconds"),
                    raw_file=str(RUN / arm / (file_stem(s) + ".json")),
                )
            )
        assert changes["existing_tp_lost"] == 0
        arms[arm] = final
        counters[arm] = {
            "review_subset": summarize(review, final, labels, rows),
            "all_64": summarize(ids, final, labels, rows),
            "changes": dict(changes),
            "statuses": dict(statuses),
            "generation_calls": len(donor["eligible"]),
            "timed_generations": len(seconds),
            "generation_seconds": sum(seconds),
            "peak_allocated": max(peaks, default=0),
        }
    result["rereads"] = counters
    result["paired_panel_diagnostic"] = {}
    groups = {s: rows[s]["group"] for s in ids}
    for first, second in [("A", "B"), ("B", "C"), ("D", "C"), ("B", "E")]:
        predictions = [
            {s: frozenset(arms[k][s]["hazards"]) for s in ids} for k in (first, second)
        ]
        result["paired_panel_diagnostic"][second + "-" + first] = paired_bootstrap(
            {s: frozenset(labels[s]) for s in ids}, *predictions, groups
        )
    comparison = {}
    for name in scores[ids[0]]:
        candidates = []
        for s in donor["eligible"]:
            available = set(ALLOWED8) - set(base[s]["hazards"])
            c = max((c for c in ALLOWED8 if c in available), key=scores[s][name].get)
            candidates.append((scores[s][name][c], s, c))
        ranked = sorted(candidates, reverse=True)
        hits = [int(c in labels[s]) for _, s, c in ranked]
        comparison[name] = {
            "proposals": len(hits),
            "available_tp": sum(hits),
            "raw_fp": len(hits) - sum(hits),
            "retrospective_tp_at_fp_cap": {
                k: max(
                    [
                        sum(hits[:i])
                        for i in range(len(hits) + 1)
                        if (i - sum(hits[:i])) <= k
                    ]
                )
                for k in (0, 1, 2, 4)
            },
        }
    result["equal_review_budget_proposals"] = comparison
    write(RUN / "paired-ledger.json", ledger)
    cells = {d + "_" + w: {} for d in ("A", "S2") for w in ("fixed", "generated")}
    writing = {}
    for d in ("A", "S2"):
        statuses = Counter()
        for s in review:
            r = read(RUN / "factorial" / (file_stem(s) + "-" + d + ".json"))
            statuses.update(r["writing"].values())
            for w in ("fixed", "generated"):
                cells[d + "_" + w][s] = r[w]
        writing[d] = dict(statuses)
    result["factorial"] = {
        k: summarize(review, v, labels, rows) for k, v in cells.items()
    }
    for d in ("A", "S2"):
        assert (
            result["factorial"][d + "_fixed"] == result["factorial"][d + "_generated"]
        )
    result["writing"] = writing
    blind = []
    key = []
    text_stats = {}
    for d in ("A", "S2"):
        for w in ("fixed", "generated"):
            stats = Counter()
            for s in review:
                output = cells[d + "_" + w][s]
                common = (
                    set(cells["A_" + w][s]["hazards"])
                    & set(cells["S2_" + w][s]["hazards"])
                    & set(labels[s])
                )
                for code, item in output["hazards"].items():
                    stats.update(
                        predicted_codes=1,
                        correct_codes=int(code in labels[s]),
                        shared_correct_codes=int(code in common),
                        evidence_filled=int(bool(item["evidence"].strip())),
                        prevention_filled=int(bool(item["prevention"].strip())),
                    )
                if not output["hazards"]:
                    stats.update(
                        nohaz_rows=1,
                        scene_evidence_filled=int(bool(output.get("scene_evidence"))),
                    )
            text_stats[d + "_" + w] = dict(stats)
        for s in review:
            pair_id = order("p38-blind-" + d, s)[:24]
            sides = sorted(["fixed", "generated"], key=lambda w: order(pair_id, w))
            blind.append(
                {
                    "pair": pair_id,
                    "image": str(PROJECT / "data/train" / rows[s]["image"]),
                    "left": cells[d + "_" + sides[0]][s],
                    "right": cells[d + "_" + sides[1]][s],
                    "rubric": [
                        "근거의 사실 지지 0~3",
                        "예방의 적절성 0~3",
                        "출처의 조치 지지 0~3 / 미인용·적용불가 별도",
                    ],
                    "rater": None,
                    "left_ratings": None,
                    "right_ratings": None,
                    "unsupported_assertion": None,
                    "unsafe_action": None,
                }
            )
            key.append(
                {
                    "pair": pair_id,
                    "detector": d,
                    "s": s,
                    "left": sides[0],
                    "right": sides[1],
                }
            )
    result["text_field_counts"] = text_stats
    write(RUN / "blind-pairs.json", sorted(blind, key=lambda r: r["pair"]))
    write(RUN / "blind-key.json", key)
    result["references"] = source_comparison(cells, labels, rows)
    result["human_blind_12_12_6"] = {
        "evidence": None,
        "prevention": None,
        "source_support": None,
        "raters": 0,
    }
    result["source_id_support_measured"] = False
    result["official_qualitative_score"] = None
    result["lora"] = {}
    for name in (
        "3vl",
        "35",
        "35-weight6",
        "35-eight",
        "35-resample",
        "35-state",
        "35-permuted",
        "35-pixels256",
        "35-vision",
    ):
        out = RUN / ("lora-" + name)
        if not (out / "results.json").exists():
            continue
        r = read(out / "results.json")
        for arm in ("base", "adapted"):
            outputs = {
                s: read(out / arm / (file_stem(s) + ".json"))["final"]
                for s in plan["lora_test"]
            }
            r[arm]["safety"] = summarize(plan["lora_test"], outputs, labels, rows)
        result["lora"][name] = r
        diag = out / "external-diagnostic.json"
        if diag.exists():
            r["external_state_diagnostic"] = {
                k: v for k, v in read(diag).items() if k != "rows"
            }
            q = [read(p) for p in sorted((out / "qualitative").glob("*.json"))]
            r["qualitative"] = {
                "requested": len(q),
                "structurally_usable": sum(row["usable"] for row in q),
                "human_evidence_prevention_source": None,
            }
    ext = RUN / "external/audit.json"
    if ext.exists():
        audit = read(ext)
        result["71407"] = {
            k: audit[k]
            for k in (
                "stats",
                "eligible_videos",
                "eligible_sites",
                "paired_sites",
                "candidate_pair_combinations",
                "pair_capability_by_scenario",
                "banned_videos",
            )
        }
        result["71407"].update(
            train_sites=len({pair[0]["site"] for pair in audit["selected"]}),
            train_pairs=len(audit["selected"]),
            unique_train_videos=2 * len(audit["selected"]),
            flipped_fraction=1.0,
        )
    result["external_paired"] = {}
    test = plan["lora_test"]
    gold = {s: frozenset(labels[s]) for s in test}
    g = {s: rows[s]["group"] for s in test}
    for ref in ("35", "35-resample", "35-permuted"):
        if ref not in result["lora"] or "35-state" not in result["lora"]:
            continue
        a, b = (
            {
                s: frozenset(
                    read(RUN / ("lora-" + name) / "adapted" / (file_stem(s) + ".json"))[
                        "final"
                    ]["hazards"]
                )
                for s in test
            }
            for name in (ref, "35-state")
        )
        result["external_paired"]["state-minus-" + ref] = {
            **paired_bootstrap(gold, a, b, g),
            "fall_recovered": sum(
                "fall_height" in (b[s] - a[s]) & gold[s] for s in test
            ),
            "fall_new_fp": sum("fall_height" in (b[s] - a[s]) - gold[s] for s in test),
            "exact_gains": sum(a[s] != gold[s] and b[s] == gold[s] for s in test),
            "exact_losses": sum(a[s] == gold[s] and b[s] != gold[s] for s in test),
            "per_group_changed": dict(Counter(g[s] for s in test if a[s] != b[s])),
        }
    write(RUN / "metrics.json", result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-progress", action="store_true")
    if parser.parse_args().paired_progress:
        paired_progress()
    else:
        score()
