"""Refit the single-view spatial head without the log-prior term in its actual loss."""

from functools import partial
import hashlib
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from . import phase01_experiment as experiment
from .metrics import paired_bootstrap
from .phase01_head import code_sets, decide, fit_head
from .phase2_panel import PHASE1, PROJECT, RUN, SIGNATURE, read, registry, write


def run(cpu_reference=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = registry()
    records = list(rows.values())
    ids = list(rows)
    ref = {
        r["sample_id"]: r
        for r in map(
            json.loads, (PHASE1 / "dev_fold-oof.jsonl").read_bytes().splitlines()
        )
    }
    assert len(ids) == 920 and set(ids) == set(ref)
    xs = []
    for r in records:
        key = hashlib.sha256((SIGNATURE + r["sha256"]).encode()).hexdigest()
        x = np.load(PHASE1 / "features" / (key + ".npy"), allow_pickle=False)
        assert x.shape == (6, 17, 128)
        xs.append(x)
    x = torch.from_numpy(np.stack(xs))
    del xs
    labels, ev, severe = experiment.supervision(
        PROJECT, records, RUN / "dev-train.jsonl"
    )
    y, e = torch.from_numpy(labels), torch.from_numpy(ev)
    groups = np.array([r["group"] for r in records])
    outer = np.array([r["dev_fold"] for r in records])
    seed, epochs = 20261126, 14
    strength = 0.5 if cpu_reference else 0.0
    dest = RUN / ("prior-refit-cpu-reference" if cpu_reference else "prior-refit")
    forecasts = {}
    prior = read(PHASE1 / "completed.json")
    assert (prior["seed"], prior["epochs"]) == (seed, epochs)
    for f in range(5):
        test, train = np.flatnonzero(outer == f), np.flatnonzero(outer != f)
        assert set(groups[test]).isdisjoint(groups[train])
        path = dest / (f"fold-{f}.json")
        if not path.exists():
            start = time.perf_counter()
            with patch.object(
                experiment, "fit_head", partial(fit_head, prior_strength=strength)
            ):
                z, policies, meta = experiment.solve_partition(
                    x, y, e, records, groups, train, test, False, seed, epochs
                )
            write(
                path,
                dict(
                    test=test.tolist(),
                    z=z.tolist(),
                    policies=policies,
                    meta=meta,
                    seconds=time.perf_counter() - start,
                    train_ids=[ids[i] for i in train],
                ),
            )
        rec = read(path)
        assert rec["test"] == test.tolist()
        z = np.asarray(rec["z"], np.float32)
        for key in ("fixed", "class"):
            m = forecasts.setdefault(key, np.zeros((len(ids), 11), bool))
            m[test] = decide(z, rec["policies"][key])
        print("PRIOR_FOLD", f, rec["seconds"], flush=True)
    gold = {s: frozenset(rows[s]["codes"]) for s in ids}
    g = {s: rows[s]["group"] for s in ids}
    all_z = np.empty((len(ids), 12), np.float32)
    for f in range(5):
        rec = read(dest / (f"fold-{f}.json"))
        all_z[rec["test"]] = rec["z"]
    result = {}
    ledger = {s: {} for s in ids}
    for key, arr in forecasts.items():
        preds = code_sets(ids, arr)
        old = {s: frozenset(ref[s]["predictions"]["single_" + key]) for s in ids}
        for s in ids:
            ledger[s][key] = sorted(preds[s])
        result[key] = {
            ("cpu_reference" if cpu_reference else "no_adjustment"): experiment.metrics(
                records, np.arange(len(ids)), all_z, arr, groups, severe
            ),
            "reference": read(PHASE1 / "dev_fold-metrics.json")["arms"][
                "single_" + key
            ],
            "paired_delta": paired_bootstrap(gold, old, preds, g),
        }
    write(
        dest / "metrics.json",
        {
            "prior_strength": strength,
            "reference_strength": 0.5,
            "training_device": "cpu",
            "reference_device": "RTX 3060 GPU",
            "epochs": epochs,
            "outer_folds": 5,
            "inner_folds": 3,
            "fresh_fits": 20,
            "conditions": result,
        },
    )
    write(dest / "predictions.json", ledger)
    if cpu_reference:
        zero = read(RUN / "prior-refit/predictions.json")
        equal_device = {}
        for key in forecasts:
            previous = {s: frozenset(ledger[s][key]) for s in ids}
            current = {s: frozenset(zero[s][key]) for s in ids}
            equal_device[key] = dict(
                delta_no_prior_minus_half=paired_bootstrap(gold, previous, current, g),
                changed_sets=sum(previous[s] != current[s] for s in ids),
            )
        write(dest / "cpu-paired.json", equal_device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference", action="store_true")
    run(parser.parse_args().cpu_reference)
