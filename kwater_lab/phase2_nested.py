"""920-image outer-OOF proposer diagnostic at an inner-fitted false-alarm budget."""

from collections import Counter
import hashlib
import numpy as np
from .metrics import CODES, evaluate, paired_bootstrap
from .phase2_panel import RUN, feature_inputs, order, read, registry, write
from .train_head import fit, probabilities


def folds_without_labels(groups):
    loads = [0, 0, 0]
    assignment = {}
    sizes = Counter(groups)
    for g in sorted(sizes, key=lambda v: (-sizes[v], order("p38-inner", v))):
        f = min(range(3), key=loads.__getitem__)
        assignment[g] = f
        loads[f] += sizes[g]
    return np.array([assignment[g] for g in groups])


def cutoff(z, y):
    # Largest allowed number of false alarms on gold-empty inner OOF rows; strict > excludes ties.
    empty = z[~y.any(axis=1)].max(axis=1)
    if not len(empty):
        return 1.0
    return float(np.sort(empty)[::-1][int(0.2 * len(empty))])


def run():
    rows = registry()
    ids, raw, _ = feature_inputs(rows, True)
    existing = read(RUN / "dev-heads.json")["scores"]
    outer = np.array([rows[s]["dev_fold"] for s in ids])
    groups = np.array([rows[s]["group"] for s in ids])
    y = np.array([[c in rows[s]["codes"] for c in CODES] for s in ids], dtype=float)
    assert len(ids) == 920 and len(set(groups)) == 29
    forecasts = {}
    report = {}
    train_ids = {}
    for name, d in raw.items():
        x = np.stack([d[s] for s in ids])
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        for fold in range(5):
            train = np.flatnonzero(outer != fold)
            test = np.flatnonzero(outer == fold)
            inner = folds_without_labels(groups[train])
            inner_z = np.zeros((len(train), len(CODES)))
            for j in range(3):
                t = train[inner != j]
                v = train[inner == j]
                assert not set(groups[t]) & set(groups[v]) and not set(groups[t]) & set(
                    groups[test]
                )
                train_ids[f"{fold}-{j}"] = [ids[i] for i in t]
                model = fit(x[t], y[t], 0.01, 250)
                inner_z[inner == j] = probabilities(model, x[v])
            z = np.array([[existing[ids[i]][name][c] for c in CODES] for i in test])
            for mask_rare in (False, True):
                arm = name + ("-suppress" if mask_rare else "")
                iz = inner_z.copy()
                zz = z.copy()
                if mask_rare:
                    for j in range(3):
                        t = train[inner != j]
                        sparse = [
                            len(set(groups[t[y[t, k] > 0]])) <= 1
                            for k in range(len(CODES))
                        ]
                        iz[np.ix_(inner == j, sparse)] = 0
                    sparse = [
                        len(set(groups[train[y[train, k] > 0]])) <= 1
                        for k in range(len(CODES))
                    ]
                    zz[:, sparse] = 0
                threshold = cutoff(iz, y[train])
                per = forecasts.setdefault(arm, {})
                for i, score in zip(test, zz):
                    top = int(score.argmax())
                    per[ids[i]] = {
                        "code": CODES[top] if score[top] > threshold else None,
                        "score": float(score[top]),
                        "fold": fold,
                        "threshold": threshold,
                    }
    gold = {s: frozenset(rows[s]["codes"]) for s in ids}
    g = {s: rows[s]["group"] for s in ids}
    preds = {}
    for arm, cases in forecasts.items():
        pred = {
            s: frozenset([r["code"]]) if r["code"] else frozenset()
            for s, r in cases.items()
        }
        preds[arm] = pred
        report[arm] = evaluate(gold, pred, g)
    intervals = {
        arm: paired_bootstrap(gold, preds["sig"], p, g)
        for arm, p in preds.items()
        if arm != "sig"
    }
    write(RUN / "nested-predictions.json", forecasts)
    write(RUN / "inner-train-ids.json", train_ids)
    write(
        RUN / "nested-metrics.json",
        {
            "scope": "Standalone top-one proposals; 5 outer folds, 3 inner folds, no Qwen verifier",
            "calibration": "inner no-hazard FPR <= 0.20, > threshold, identically specified for every encoder",
            "policy": "separate train-only <=1 positive group suppression; identical fitted loss for both policies",
            "training_ids_sha256": hashlib.sha256(
                (RUN / "inner-train-ids.json").read_bytes()
            ).hexdigest(),
            "conditions": report,
            "paired_vs_sig": intervals,
        },
    )


def safety_summary():
    from .phase2_score import readlines

    predictions = read(RUN / "nested-predictions.json")
    report = read(RUN / "nested-metrics.json")
    rows = registry()
    gold = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in rows
    }
    severe = {
        s: {c for c, h in hs.items() if h["risk_level"]["severity"]["score"] == 4}
        for s, hs in gold.items()
    }
    for name, ledger in predictions.items():
        m = report["conditions"][name]
        assert set(ledger) == set(rows)
        m.update(
            fixed_11_f1=sum(v["f1"] or 0 for v in m["per_code"].values()) / len(CODES),
            severity4_photos=sum(bool(v) for v in severe.values()),
            severity4_completely_missed=sum(
                bool(v) and ledger[s]["code"] not in v for s, v in severe.items()
            ),
            proposal_count=sum(r["code"] is not None for r in ledger.values()),
            valid_rows=len(ledger),
        )
    write(RUN / "nested-metrics.json", report)


if __name__ == "__main__":
    import sys

    if "--safety-only" not in sys.argv:
        run()
    safety_summary()
