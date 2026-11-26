"""Replay two stored outer-fold models and the single precommitted empty-slot rule on CPU."""

import argparse
import csv
import hashlib
import json
import mmap
from collections import Counter
from pathlib import Path
import numpy as np
from kwater_lab.metrics import CODES

BASE, MULTI, RULE = "single_fixed", "multi_class", "one_slot"


def read_dev_labels(path, ids):
    # Look up only whitelisted record prefixes. Materialize and decode only matching rows.
    labels = {}
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as index,
    ):
        for sid in ids:
            key = b'{"sample_id":' + json.dumps(sid).encode() + b","
            start = index.find(key)
            if start < 0:
                start = index.find(b'{"sample_id": ' + json.dumps(sid).encode() + b",")
            assert start >= 0 and (start == 0 or index[start - 1 : start] == b"\n")
            end = index.find(b"\n", start)
            row = json.loads(index[start : end if end >= 0 else len(index)])
            assert row["sample_id"] == sid
            labels[sid] = {
                k: v["risk_level"]["severity"]["score"]
                for k, v in row["target"]["hazards"].items()
            }
    return labels


def trace(z, cfg):
    z = np.array(z, dtype=float)
    temp = np.array(cfg["temperatures"])
    p = np.exp(-np.logaddexp(0, -z / temp))
    gate = p[-1] >= cfg["scene_gate"]
    ok = (p[:-1] >= cfg["thresholds"]) & gate
    score = p[:-1] - cfg["offset"]
    raw = sorted(range(11), key=lambda j: (-z[j], j))
    cal = sorted(range(11), key=lambda j: (-score[j], j))
    eligible = [j for j in cal if ok[j]]
    pred = [CODES[j] for j in eligible[:3]]
    rows = []
    for j, c in enumerate(CODES):
        # Rank if ONLY this code's threshold were waived; other classes keep their actual policy.
        hyp = [i for i in cal if i == j or ok[i]].index(j) + 1
        rank = eligible.index(j) + 1 if ok[j] else None
        if c in pred:
            cause = "hit"
        elif not gate:
            cause = "scene"
        elif ok[j]:
            cause = "rank_only"
        elif hyp > 3:
            cause = "threshold_and_rank"
        else:
            cause = "threshold_only"
        rows.append(
            dict(
                code=c,
                logit=z[j],
                temperature=temp[j],
                p=p[j],
                threshold=cfg["thresholds"][j],
                offset=cfg["offset"][j],
                utility=score[j],
                raw_rank=raw.index(j) + 1,
                calibrated_rank=cal.index(j) + 1,
                scene_p=p[-1],
                scene_gate=cfg["scene_gate"],
                passed=bool(ok[j]),
                eligible_rank=rank,
                rank_if_admitted=hyp,
                cause=cause,
            )
        )
    return pred, rows


def csvout(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def evaluate(y, severe, pred):
    denom = y.sum(0) + pred.sum(0)
    active = denom > 0
    return dict(
        photos=len(y),
        macro_f1=float(((2 * (y & pred).sum(0))[active] / denom[active]).mean()),
        severe_photos=int(severe.any(1).sum()),
        complete_misses=int((severe.any(1) & ~(severe & pred).any(1)).sum()),
        severe_code_instances=int(severe.sum()),
        missed_severe_codes=int((severe & ~pred).sum()),
        no_hazard=int((~y.any(1)).sum()),
        alarms=int((~y.any(1) & pred.any(1)).sum()),
    )


def paired(y, severe, forecasts, groups):
    levels = sorted(set(groups))
    draws = np.random.default_rng(20261126).multinomial(
        len(levels), np.full(len(levels), 1 / len(levels)), size=5000
    )

    def by_group(x):
        return np.stack([x[groups == g].sum(axis=0) for g in levels])

    def rate(num, den):
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    def stats(pred):
        denom = draws @ by_group(y.astype(int) + pred)
        f1 = rate(2 * draws @ by_group(y & pred), denom)
        misses = draws @ by_group(severe.any(1) & ~(severe & pred).any(1))
        alarms = draws @ by_group(~y.any(1) & pred.any(1))
        return np.column_stack(
            [
                np.nanmean(f1, axis=1),
                rate(misses, draws @ by_group(severe.any(1))),
                rate(alarms, draws @ by_group(~y.any(1))),
            ]
        )

    samples = {k: stats(v) for k, v in forecasts.items()}
    return {
        a + " minus " + b: np.nanquantile(
            samples[a] - samples[b], [0.025, 0.975], axis=0
        ).T.tolist()
        for a, b in [(MULTI, BASE), (RULE, BASE), (RULE, MULTI)]
    }


def run(args):
    run, out = args.run, args.out
    out.mkdir(parents=True, exist_ok=True)
    buf = (run / "dev_fold-oof.jsonl").read_bytes()
    assert (
        hashlib.sha256(buf).hexdigest()
        == "4d1f337ac3bad2197019a50726cbc623e868217b3e0f1b935652a2ddf65b9519"
    )
    records = [json.loads(s) for s in buf.splitlines()]
    ids = [r["sample_id"] for r in records]
    cfg = json.loads((run / "phase3-development/policy.json").read_bytes())
    assert len(set(ids)) == len(ids) == cfg["fit_photos"] == 920
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == cfg["fit_ids_sha256"]
    labels = read_dev_labels(args.labels, ids)
    m = json.loads((run / "dev_fold-metrics.json").read_bytes())
    forecasts = {k: [] for k in (BASE, MULTI, RULE)}
    fold = np.array([r["fold"] for r in records])
    groups = np.array([r["group"] for r in records])
    assert len(set(groups)) == 29 and set(fold) == set(range(5))
    signature = json.loads((run / "completed.json").read_bytes())["signature"].encode()
    for f in range(5):
        assert not set(groups[fold == f]) & set(groups[fold != f])
        train, test = (np.flatnonzero(a).astype("<i8") for a in (fold != f, fold == f))
        identity = hashlib.sha256(
            signature + train.tobytes() + b"/" + test.tobytes()
        ).hexdigest()
        for mode in ("single", "multi"):
            tag = f"dev_fold-{mode}-{f}"
            with np.load(run / "fold-cache" / (tag + ".npz"), allow_pickle=False) as d:
                assert d["identity"].item() == identity
                assert np.array_equal(
                    d["z"], [r["logits"][mode] for r in records if r["fold"] == f]
                )
                assert json.loads(d["policies"].item()) == m["policies"][tag]
                meta = json.loads(d["metadata"].item())
                assert (
                    meta == m["training_partitions"][tag]
                    and len(meta["inner_folds"]) == 3
                )
                assert (
                    meta["train_photos"]
                    == sum(a["photos"] for a in meta["inner_folds"])
                    == len(train)
                )
                assert meta["test_photos"] == len(test)
    photos, details = [], []
    by_code, by_photo, new_causes = {}, {}, {}
    y, ss = [], []
    for r in records:
        sid = r["sample_id"]
        gold = labels[sid]
        assert set(r["gold"]) == set(gold)
        severe = {c for c, score in gold.items() if score == 4}
        y.append([c in gold for c in CODES])
        ss.append([c in severe for c in CODES])
        preds, causes = {}, {}
        photo = dict(
            sample_id=sid,
            group=r["group"],
            fold=r["fold"],
            gold=" ".join(sorted(gold)),
            severe=" ".join(sorted(severe)),
        )
        for arm in (BASE, MULTI):
            mode, policy = arm.split("_")
            tag = f"dev_fold-{mode}-{r['fold']}"
            cfg = m["policies"][tag][policy]
            pred, rows = trace(r["logits"][mode], cfg)
            assert set(pred) == set(r["predictions"][arm])
            preds[arm] = pred
            for entry in rows:
                code = entry["code"]
                details.append(
                    dict(
                        sample_id=sid,
                        group=r["group"],
                        fold=r["fold"],
                        arm=arm,
                        actual_severity=gold.get(code),
                        **entry,
                    )
                )
                if code in severe:
                    by_code.setdefault(arm, Counter())[entry["cause"]] += 1
            causes[arm] = "+".join(
                sorted({e["cause"] for e in rows if e["code"] in severe})
            )
            if severe and severe.isdisjoint(pred):
                by_photo.setdefault(arm, Counter())[causes[arm]] += 1
            if arm == MULTI:
                extra = []
                if 0 < len(pred) < 3:
                    assert cfg["scene_gate"] == 0 and not any(cfg["offset"])
                    extra = [
                        max((e for e in rows if not e["passed"]), key=lambda e: e["p"])[
                            "code"
                        ]
                    ]
                preds[RULE] = pred + extra
        for arm, pred in preds.items():
            forecasts[arm].append([c in pred for c in CODES])
            photo[arm] = " ".join(pred)
        b = bool(severe & set(preds[BASE]))
        c = bool(severe & set(preds[MULTI]))
        photo["pair"] = (
            (
                "both_hit"
                if b and c
                else "baseline_hit_only"
                if b
                else "candidate_hit_only"
                if c
                else "both_miss"
            )
            if severe
            else ""
        )
        if severe and b and not c:
            new_causes.setdefault(MULTI, Counter())[causes[MULTI]] += 1
        photo["baseline_causes"] = causes[BASE]
        photo["candidate_causes"] = causes[MULTI]
        photos.append(photo)
    y, ss = np.array(y), np.array(ss)
    forecasts = {k: np.array(v) for k, v in forecasts.items()}

    def scores(ix):
        return {k: evaluate(y[ix], ss[ix], p[ix]) for k, p in forecasts.items()}

    outdata = dict(
        pooled=scores(np.ones(len(ids), bool)),
        fold={f: scores(fold == f) for f in range(5)},
        paired={k: v for k, v in Counter(p["pair"] for p in photos).items() if k},
        code_causes=by_code,
        complete_miss_photo_causes=by_photo,
        new_complete_miss_photo_causes=new_causes,
        bootstrap_ci=paired(y, ss, forecasts, groups),
        bootstrap_columns=["macro_f1", "complete_miss_rate", "no_hazard_alarm_rate"],
        bootstrap_seed=20261126,
        bootstrap_draws=5000,
    )
    assert [outdata["pooled"][a]["complete_misses"] for a in (BASE, MULTI, RULE)] == [
        96,
        109,
        104,
    ]
    csvout(out / "audit-photos.csv", photos)
    csvout(out / "audit-270.csv", [p for p in photos if p["pair"]])
    csvout(out / "audit-codes.csv", details)
    (out / "audit.json").write_text(json.dumps(outdata, indent=2), encoding="utf8")
    print(json.dumps(outdata["pooled"]))
    print(json.dumps(outdata["paired"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for arg in ("run", "out", "labels"):
        ap.add_argument("--" + arg, required=True, type=Path)
    run(ap.parse_args())
