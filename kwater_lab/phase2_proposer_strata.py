"""Summarize real frozen Qwen rows by the prespecified DINO and combined proposal overlap."""

import argparse
from collections import Counter
import time

from .phase2_panel import RUN, read, registry, write
from .phase2_paths import file_stem
from .phase2_release import changes
from .phase2_score import readlines


STRATA = ("same_code", "different_codes", "dino_only", "combined_only", "both_silent")
CONTRASTS = (("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("D", "C"))


def classify(dino, combined):
    if dino["called"] and combined["called"]:
        return (
            "same_code"
            if dino["candidate"] == combined["candidate"]
            else "different_codes"
        )
    if dino["called"]:
        return "dino_only"
    return "combined_only" if combined["called"] else "both_silent"


def replay(watch=False):
    inputs = read(RUN / "matched-budget-predictions.json")
    spec, rows = read(RUN / "plan.json"), registry()
    dino, combo = inputs["dino"], inputs["large_concat"]
    split = {s: classify(dino[s], combo[s]) for s in rows}
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
    }
    membership = {}
    for flag in STRATA:
        ids = [s for s in split if split[s] == flag]
        membership[flag] = dict(
            population=len(ids),
            groups=len({rows[s]["group"] for s in ids}),
            hazard_images=sum(bool(labels[s]) for s in ids),
            dino_correct=sum(
                dino[s]["called"] and dino[s]["candidate"] in labels[s] for s in ids
            ),
            combined_correct=sum(
                combo[s]["called"] and combo[s]["candidate"] in labels[s] for s in ids
            ),
            both_correct=sum(
                dino[s]["called"]
                and combo[s]["called"]
                and dino[s]["candidate"] in labels[s]
                and combo[s]["candidate"] in labels[s]
                for s in ids
            ),
            frozen64=sum(s in spec["panel"] for s in ids),
            frozen16=sum(s in spec["review"] for s in ids),
        )
    previous = None
    while True:
        available = {}
        for arm in "ABCDE":
            ids = spec["panel"] if arm == "A" else spec["review"]
            paths = {s: RUN / arm / (file_stem(s) + ".json") for s in ids}
            available[arm] = {s: p for s, p in paths.items() if p.is_file()}
        signature = tuple(tuple(v) for v in available.values())
        if signature != previous:
            actual = {
                arm: {s: read(p) for s, p in paths.items()}
                for arm, paths in available.items()
            }
            from .phase2_score import paired_progress

            paired_progress()
            ledger = {}
            summary = {}
            for flag in STRATA:
                originals = {s: r for s, r in actual["A"].items() if split[s] == flag}
                summary[flag] = dict(
                    membership[flag],
                    observed_A=len(originals),
                    A_valid=sum(r["valid"] for r in originals.values()),
                    comparisons={},
                )
                for a, b in CONTRASTS:
                    comparison = b + "_minus_" + a
                    ids = [
                        s
                        for s in spec["review"]
                        if split[s] == flag and s in actual[a] and s in actual[b]
                    ]
                    counts = Counter()
                    for s in ids:
                        before, after = actual[a][s], actual[b][s]
                        delta = changes(before["final"], after["final"], labels[s])
                        counts.update(
                            {
                                k: len(delta[k])
                                for k in (
                                    "gained_tp",
                                    "new_fp",
                                    "lost_tp",
                                    "removed_fp",
                                )
                            }
                        )
                        info = ledger.setdefault(
                            s,
                            dict(
                                stratum=flag,
                                group=rows[s]["group"],
                                dino=dino[s],
                                combined=combo[s],
                                contrasts={},
                            ),
                        )
                        info["contrasts"][comparison] = dict(
                            actual_hint=after.get("hint"),
                            status=after.get("status"),
                            changes=delta,
                        )
                    summary[flag]["comparisons"][comparison] = dict(
                        photos=len(ids),
                        groups=len({rows[s]["group"] for s in ids}),
                        **{
                            k: counts[k]
                            for k in ("gained_tp", "new_fp", "lost_tp", "removed_fp")
                        },
                        decision_status=dict(
                            Counter(actual[b][s].get("status", "unknown") for s in ids)
                        ),
                    )
            write(RUN / "proposer-strata-rows.json", ledger)
            write(
                RUN / "proposer-strata-metrics.json",
                dict(
                    scope="Observed official-Qwen diagnostic only. Stratification uses fixed 230/920 standalone offers; no extra generator or combined-head reread.",
                    rows_by_arm={arm: len(v) for arm, v in actual.items()},
                    strata=summary,
                ),
            )
            previous = signature
        if not watch or (
            len(available["A"]) == 64 and all(len(available[k]) == 16 for k in "BCDE")
        ):
            break
        time.sleep(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch", action="store_true")
    replay(p.parse_args().watch)
