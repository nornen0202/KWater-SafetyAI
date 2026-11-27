"""Paired evaluation of completed local SFT arms on the same held-out development groups."""

from collections import Counter

from .metrics import paired_bootstrap
from .phase2_panel import RUN, read, registry, write
from .phase2_paths import file_stem
from .phase2_release import changes
from .phase2_score import measured_cost, readlines, summarize


def compare():
    rows = registry()
    ids = read(RUN / "sft-split.json")["test"]
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in ids
    }
    gold = {s: frozenset(labels[s]) for s in ids}
    groups = {s: rows[s]["group"] for s in ids}
    outputs, arms, external = {}, {}, {}
    for result in sorted(RUN.glob("lora-*/results.json")):
        folder = result.parent
        info = read(result)
        if (
            info.get("stage", "pilot") != "pilot"
            or info.get("task") == "image_proposal_verification"
        ):
            continue
        lineage = read(folder / "lineage.json")
        if (
            lineage["parent_adapters"]
            or lineage["cumulative_group_overlap"]["same_groups"]
        ):
            raise ValueError("Pilot checkpoint ancestry overlaps its test groups")
        for mode in ("base", "adapted"):
            key = folder.name.removeprefix("lora-") + "/" + mode
            records = {s: read(folder / mode / (file_stem(s) + ".json")) for s in ids}
            final = {s: v["final"] for s, v in records.items()}
            outputs[key] = final
            arms[key] = dict(
                metrics=summarize(ids, final, labels, rows),
                train_support=lineage["train_support"],
                raw_valid=sum(v["valid"] for v in records.values()),
                truncated=sum(v.get("limit_reached", False) for v in records.values()),
                thinking_tokens=sum(
                    v.get("delimited_reasoning_tokens") or 0 for v in records.values()
                ),
                thinking_measured_rows=sum(
                    v.get("delimited_reasoning_tokens") is not None
                    for v in records.values()
                ),
                seconds=sum(v["seconds"] for v in records.values()),
                measured_cost=measured_cost(records.values()),
                peak_allocated_bytes=max(
                    v["peak_allocated_bytes"] for v in records.values()
                ),
                optimizer_file=str(folder / "optimizer.json"),
                adapter_file_sha256=info["adapter_file_sha256"],
            )
        if (folder / "external-diagnostic.json").is_file():
            diagnostic = read(folder / "external-diagnostic.json")
            qualitative = [read(p) for p in (folder / "qualitative").glob("*.json")]
            external[folder.name] = dict(
                site_count=len({r["site"] for r in diagnostic["rows"]}),
                photos=diagnostic["n"],
                correct=diagnostic["correct"],
                valid=diagnostic["valid"],
                pairs=diagnostic["n"] // 2,
                both_states_correct=diagnostic["both_states_correct"],
                writing_rows=len(qualitative),
                writing_usable=sum(v["usable"] for v in qualitative),
                human_ratings=None,
            )
    comparisons = {}
    ledger = {}
    contrasts = [
        ("3vl/base", "35/base"),
        ("3vl/adapted", "35/adapted"),
        ("3vl/base", "8b/base"),
        ("3vl/adapted", "8b/adapted"),
        ("35/base", "9b/base"),
        ("35/adapted", "9b/adapted"),
        ("8b/base", "9b/base"),
        ("8b/adapted", "9b/adapted"),
        ("35-permuted/adapted", "35-state/adapted"),
        ("35-resample/adapted", "35-state/adapted"),
    ]
    for a in arms:
        if a.endswith("/adapted"):
            contrasts.append((a.replace("/adapted", "/base"), a))
            if a.startswith("35-"):
                contrasts.append(("35/adapted", a))
    for first, second in contrasts:
        if first not in outputs or second not in outputs:
            continue
        key = second + " minus " + first
        paired = {
            s: changes(outputs[first][s], outputs[second][s], labels[s]) for s in ids
        }
        ledger[key] = paired
        counts = Counter()
        for item in paired.values():
            for field in ("gained_tp", "new_fp", "lost_tp", "removed_fp"):
                counts[field] += len(item[field])
        predictions = [
            {s: frozenset(outputs[arm][s]["hazards"]) for s in ids}
            for arm in (first, second)
        ]
        comparisons[key] = dict(
            changes=dict(counts),
            group_interval=paired_bootstrap(gold, *predictions, groups)
            if len(set(groups.values())) >= 2
            else None,
        )
    if not arms:
        raise ValueError("No fully trained and reloaded adapter available to pair")
    write(RUN / "lora-paired-rows.json", ledger)
    write(
        RUN / "lora-paired-metrics.json",
        dict(
            photos=len(ids),
            groups=len(set(groups.values())),
            arms=arms,
            comparisons=comparisons,
            external=external,
            scope="Frozen development fold pilot; deployment threshold requires full development-group confirmation.",
        ),
    )


if __name__ == "__main__":
    compare()
