"""Verify frozen scale-up memberships and score new adapters on their unused groups."""

import argparse
from collections import Counter

from .metrics import paired_bootstrap
from .phase2_panel import RUN, file_hash, order, read, registry, write
from .phase2_paths import file_stem
from .phase2_release import changes
from .phase2_score import readlines, summarize


SPLITS = {
    "panel170": (
        "expansion-split.json",
        "f53889e11ca8f8e6ba248a7a07c9eae752f4a1fe4fc5c11835379ab15398e498",
    ),
    "outer1": (
        "expansion-outer1.json",
        "2810a7ebaef3c5631d794352666267e69e0794c26e5f4da9a99f888add3cc297",
    ),
}


def verify():
    rows, plan = registry(), read(RUN / "plan.json")
    frozen = {}
    for stage, (name, digest) in SPLITS.items():
        p = RUN / "large-models" / name
        if file_hash(p) != digest:
            raise ValueError(f"{stage} membership or order changed")
        d = read(p)
        train, test = d["train"], d["test"]
        tg = {rows[s]["group"] for s in train}
        eg = {rows[s]["group"] for s in test}
        assert (
            len(train) == len(set(train))
            and len(test) == len(set(test))
            and not tg & eg
        )
        if stage == "panel170":
            assert test == plan["panel"] and set(train) == {
                s for s, r in rows.items() if r["group"] not in eg
            }
            assert set(d["evaluation_without_pilot_groups"]) == {
                s
                for s in test
                if rows[s]["group"] not in {rows[u]["group"] for u in d["first_test"]}
            }
        else:
            assert set(train) == {
                s for s, r in rows.items() if r["dev_fold"] != d["outer_fold"]
            }
            assert set(test) == {
                s
                for s, r in rows.items()
                if r["dev_fold"] == d["outer_fold"]
                and not r["exposed"]
                and s not in plan["panel"]
            }
        frozen[stage] = dict(
            train_images=len(train),
            train_groups=len(tg),
            validation_images=len(test),
            validation_groups=len(eg),
            split_sha256=digest,
        )
    write(RUN / "large-models/verified-expansion.json", frozen)
    return frozen


def freeze_inner():
    """Keep model-family comparison and training tuning entirely inside outer fold 1's training groups."""
    from .phase2_nested import folds_without_labels

    verify()
    rows = registry()
    root = RUN / "large-models"
    outer = read(root / SPLITS["outer1"][0])
    pool = sorted(outer["train"])
    folds = folds_without_labels([rows[s]["group"] for s in pool])
    for fold in range(3):
        training = sorted(
            (s for s, f in zip(pool, folds) if f != fold),
            key=lambda s: order("p38-inner1/train", s),
        )
        val = [s for s, f in zip(pool, folds) if f == fold]
        unique = []
        for code in sorted({c for s in training for c in rows[s]["codes"]}):
            if any(code in rows[s]["codes"] for s in unique):
                continue
            unique.append(next(s for s in training if code in rows[s]["codes"]))
        train = (unique + [s for s in training if s not in unique])[:32]
        groups = sorted(
            {rows[s]["group"] for s in val}, key=lambda g: order("p38-inner1/val", g)
        )
        by_group = [
            sorted(
                (s for s in val if rows[s]["group"] == g),
                key=lambda s: order("p38-inner1/val", s),
            )
            for g in groups
        ]
        test = [group[i] for i in range(11) for group in by_group if len(group) > i][
            :11
        ]
        assert not {rows[s]["group"] for s in train} & {rows[s]["group"] for s in test}
        assert all(rows[s]["dev_fold"] != 1 for s in train + test)
        dest = root / f"outer1-inner{fold}.json"
        spec = dict(
            outer_fold=1,
            inner_fold=fold,
            outer_train_sha256=SPLITS["outer1"][1],
            train=train,
            test=test,
            train_groups=sorted({rows[s]["group"] for s in train}),
            test_groups=sorted({rows[s]["group"] for s in test}),
            steps=16,
            selection="Compare three inner folds only; fixed training-side code coverage and label-blind validation group order.",
        )
        if dest.exists() and read(dest) != spec:
            raise ValueError("Existing inner selection split differs")
        write(dest, spec)


def score(stage, name):
    verify()
    control = {"8b": "3vl", "9b": "35", "35": "3vl"}[name]
    rows = registry()
    split = read(RUN / "large-models" / SPLITS[stage][0])
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in split["test"]
    }
    results = {}
    subsets = {"all": split["test"]}
    if stage == "panel170":
        subsets["outside_pilot_groups"] = split["evaluation_without_pilot_groups"]
    for group, ids in subsets.items():
        outputs, metrics = {}, {}
        for model in (control, name):
            folder = RUN / ("lora-" + model + "-" + stage)
            evidence = read(folder / "results.json")
            lineage = read(folder / "lineage.json")
            if (
                lineage["parent_adapters"]
                or lineage["cumulative_group_overlap"]["same_groups"]
                or set(lineage["evaluation_ids"]) != set(split["test"])
            ):
                raise ValueError(
                    "Expanded evaluation ancestry does not match its frozen groups"
                )
            if evidence["training_split_sha256"] != SPLITS[stage][1] or evidence[
                "adapter_file_sha256"
            ] != file_hash(folder / "adapter/adapter_model.safetensors"):
                raise ValueError("Adapter or split receipt differs")
            for mode in ("base", "adapted"):
                key = model + "/" + mode
                recs = {s: read(folder / mode / (file_stem(s) + ".json")) for s in ids}
                outputs[key] = {s: r["final"] for s, r in recs.items()}
                metrics[key] = dict(
                    scores=summarize(ids, outputs[key], labels, rows),
                    fit_lineage_sha256=file_hash(folder / "lineage.json"),
                    train_support=lineage["train_support"],
                    raw_valid=sum(r["valid"] for r in recs.values()),
                    truncated=sum(r.get("limit_reached", False) for r in recs.values()),
                    seconds=sum(r["seconds"] for r in recs.values()),
                )
        comparisons = {}
        pairs = [
            (control + "/base", name + "/base"),
            (control + "/adapted", name + "/adapted"),
            (name + "/base", name + "/adapted"),
            (control + "/base", control + "/adapted"),
        ]
        for first, second in pairs:
            counts = Counter()
            for s in ids:
                delta = changes(outputs[first][s], outputs[second][s], labels[s])
                counts.update(
                    {
                        k: len(delta[k])
                        for k in ("gained_tp", "new_fp", "lost_tp", "removed_fp")
                    }
                )
            p = [
                {s: frozenset(outputs[k][s]["hazards"]) for s in ids}
                for k in (first, second)
            ]
            interval = paired_bootstrap(
                {s: frozenset(labels[s]) for s in ids},
                *p,
                {s: rows[s]["group"] for s in ids},
            )
            comparisons[second + " minus " + first] = dict(
                move_counts=dict(counts), paired=interval
            )
        results[group] = dict(
            photos=len(ids),
            groups=len({rows[s]["group"] for s in ids}),
            metrics=metrics,
            comparisons=comparisons,
        )
    write(RUN / "large-models" / f"{stage}-{name}-paired.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("verify", "freeze-inner", *SPLITS))
    p.add_argument("--challenger", choices=("35", "8b", "9b"))
    args = p.parse_args()
    if args.stage == "verify":
        verify()
    elif args.stage == "freeze-inner":
        freeze_inner()
    elif args.challenger:
        score(args.stage, args.challenger)
    else:
        p.error("Choose the trained challenger")
