"""Compare the cached out-of-fold proposal heads at identical label-blind call budgets."""

from collections import Counter
import time

from .metrics import CODES, evaluate, paired_bootstrap
from .phase2_panel import RUN, read, registry, order, write
from .phase2_score import readlines


def compare():
    start = time.perf_counter()
    rows = registry()
    cache = read(RUN / "dev-heads.json")
    scores = cache["scores"]
    assert set(scores) == set(rows)
    folds = {
        f: sorted(s for s, r in rows.items() if r["dev_fold"] == f) for f in range(5)
    }
    # Largest-remainder allocation matches 16 review calls per 64 frozen photos.
    calls = {f: len(ids) // 4 for f, ids in folds.items()}
    remainder_order = sorted(folds, key=lambda f: (-(len(folds[f]) % 4), f))
    for f in remainder_order[: 230 - sum(calls.values())]:
        calls[f] += 1
    assert sum(calls.values()) == 230
    gold = {s: frozenset(r["codes"]) for s, r in rows.items()}
    groups = {s: r["group"] for s, r in rows.items()}
    targets = {
        rec["sample_id"]: rec["target"]["hazards"]
        for rec in readlines(RUN / "dev-train.jsonl")
    }
    severe = {
        s: {
            c
            for c, h in targets[s].items()
            if h["risk_level"]["severity"]["score"] == 4
        }
        for s in rows
    }
    preds, ledger, report = {}, {}, {}
    for head in ("clip", "sig", "large_sig", "dino", "large_concat"):
        records = {}
        for fold, ids in folds.items():
            assert not (set(ids) & set(cache["train_ids_by_fold"][str(fold)]))
            choices = {s: max(CODES, key=lambda c: scores[s][head][c]) for s in ids}
            ranked = sorted(
                ids,
                key=lambda s: (
                    -scores[s][head][choices[s]],
                    order("p38-fixed-calls", s),
                ),
            )
            for rank, s in enumerate(ranked):
                code = choices[s]
                records[s] = dict(
                    fold=fold,
                    rank=rank + 1,
                    score=scores[s][head][code],
                    candidate=code,
                    called=rank < calls[fold],
                )
        pred = {
            s: frozenset([r["candidate"]]) if r["called"] else frozenset()
            for s, r in records.items()
        }
        preds[head] = pred
        ledger[head] = records
        metrics = evaluate(gold, pred, groups)
        tp = sum(len(gold[s] & pred[s]) for s in rows)
        ranked_offers = sorted(
            (s for s, v in pred.items() if v),
            key=lambda s: (
                records[s]["rank"] / calls[records[s]["fold"]],
                order("p38-fixed-fp", s),
            ),
        )
        fixed_fp = {}
        for budget in (10, 20, 40):
            recovered = wrong = 0
            for s in ranked_offers:
                hit = bool(gold[s] & pred[s])
                if not hit and wrong == budget:
                    break
                recovered += int(hit)
                wrong += int(not hit)
            fixed_fp[str(budget)] = dict(
                eligible_offers=230,
                retained=recovered + wrong,
                correct=recovered,
                false_positive=wrong,
            )
        report[head] = dict(
            calls=sum(bool(v) for v in pred.values()),
            true_positive_proposals=tp,
            false_positive_proposals=230 - tp,
            retrospective_equal_observed_fp=fixed_fp,
            severity4_photos=sum(bool(v) for v in severe.values()),
            severity4_completely_missed=sum(
                bool(severe[s]) and not bool(severe[s] & pred[s]) for s in rows
            ),
            metrics=metrics,
            called_code_counts=dict(Counter(next(iter(v)) for v in pred.values() if v)),
        )
    intervals = {
        name: paired_bootstrap(gold, preds["sig"], p, groups)
        for name, p in preds.items()
        if name != "sig"
    }
    write(RUN / "matched-budget-predictions.json", ledger)
    write(
        RUN / "matched-budget-metrics.json",
        dict(
            scope="Standalone cached outer-group proposal heads. No Qwen reread or submission estimate.",
            population=len(rows),
            groups=len(set(groups.values())),
            calls_by_fold=calls,
            common_budget=230,
            arms=report,
            paired_vs_light_sig=intervals,
            cpu_replay_seconds=time.perf_counter() - start,
        ),
    )


if __name__ == "__main__":
    compare()
