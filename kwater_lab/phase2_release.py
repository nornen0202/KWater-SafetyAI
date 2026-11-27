"""Replay the two frozen submission formatters on identical, locally generated raw rows."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from types import ModuleType
import time
from zipfile import ZipFile

from .adapter_scale_pilot import load_entry
from .metrics import decode_target
from .phase2_panel import PROJECT, RUN, order, read, registry, write
from .phase2_paths import file_stem
from .phase2_score import readlines, summarize


DEST = RUN / "release-replay"
ARMS = ("pristine", "scope", "mechanism")
FROZEN_SOURCES = {
    "scope": (
        "f238ab3aca25feb82bfaf00d26f82fb709598d01a7bd594f61455e64c6d8567f",
        "16d9c2cc1aa84e9ec596dd66d5aebabc7a9ad104e976e3876443a23ffadfa4c9",
    ),
    "mechanism": (
        "529d232cdbcaae118bd6d27fd4c916fb380dea04630a156335bf5fbcbc438825",
        "c7e3f3aec5360a97711062120e837246fb6b030513dd728080c15cb058fbad4d",
    ),
}


def frozen_modules():
    original = PROJECT / "data/packages/kwater-baseline.zip"
    receipts, modules = {}, {}
    with ZipFile(original) as pristine:
        for arm, source in (
            ("scope", "article_scope.py"),
            ("mechanism", "risk_text.py"),
        ):
            path = PROJECT / f"data/submissions/kwater-qwen4b-lora1-{arm}-20261126.zip"
            with ZipFile(path) as archive:
                assert set(archive.namelist()) - set(pristine.namelist()) == {source}
                changed = []
                for name in pristine.namelist():
                    a, b = pristine.getinfo(name), archive.getinfo(name)
                    if (a.CRC, a.file_size) != (b.CRC, b.file_size):
                        changed.append(name)
                assert changed == ["main.py"]
                module = ModuleType(source.removesuffix(".py"))
                code = archive.read(source)
                fingerprints = (
                    sha256(code).hexdigest(),
                    sha256(archive.read("main.py")).hexdigest(),
                )
                if fingerprints != FROZEN_SOURCES[arm]:
                    raise ValueError(
                        f"{arm} release changed since the initial frozen replay"
                    )
                exec(compile(code, f"{path}!{source}", "exec"), module.__dict__)
                modules[arm] = module
                receipts[arm] = dict(
                    path=str(path),
                    bytes=path.stat().st_size,
                    module=source,
                    module_sha256=fingerprints[0],
                    main_sha256=fingerprints[1],
                    matching_original_members_by_crc_and_size=len(pristine.namelist())
                    - 1,
                )
    write(DEST / "sources.json", receipts)
    return modules


def changes(before, after, gold):
    old, new = set(before["hazards"]), set(after["hazards"])
    severe = {c for c, h in gold.items() if h["risk_level"]["severity"]["score"] == 4}

    def refs(row):
        return {
            c: [r["id"] for r in h.get("references", [])]
            for c, h in row["hazards"].items()
        }

    return dict(
        before=sorted(old),
        after=sorted(new),
        gained_tp=sorted((new - old) & gold.keys()),
        new_fp=sorted((new - old) - gold.keys()),
        lost_tp=sorted((old - new) & gold.keys()),
        removed_fp=sorted((old - new) - gold.keys()),
        codes_changed=old != new,
        output_changed=before != after,
        references_before=refs(before),
        references_after=refs(after),
        no_hazard_false_alarm=[not gold and bool(v) for v in (old, new)],
        severity4_completely_missed=[
            bool(severe) and not (v & severe) for v in (old, new)
        ],
    )


def replay(rows, raw_rows, source, entry, modules, laws, cases, labels, exposed_groups):
    records = {}
    active = modules["scope"].active_articles(laws)
    started = time.perf_counter()
    for arm in ("pristine", "mechanism"):
        valid = []
        for sid, raw in raw_rows.items():
            prediction = deepcopy(raw["raw"])
            if not entry.validate_prediction(prediction):
                if arm == "mechanism":
                    modules[arm].reconcile(prediction)
                valid.append(
                    {
                        "sample_id": order("kwater/scale", sid)[:32],
                        "prediction": prediction,
                    }
                )
        formatted = {}
        if valid:
            output = DEST / f"format-{arm}.jsonl"
            entry.format_submission(valid, laws, cases, output)
            formatted = {
                x["sample_id"]: x["prediction"] for x in entry.load_rows(output)
            }
        for sid, raw in raw_rows.items():
            anon = order("kwater/scale", sid)[:32]
            result = entry.convert(anon, formatted.get(anon))
            decode_target(result)
            records.setdefault(sid, {})[arm] = result
            if arm == "pristine" and "final" in raw and raw["final"] != result:
                raise ValueError(
                    f"{sid}: new pristine CPU output differs from saved organizer output"
                )
    for sid, outputs in records.items():
        outputs["scope"] = modules["scope"].apply(deepcopy(outputs["pristine"]), active)
        decode_target(outputs["scope"])
        gold = labels[sid]
        record = dict(
            sample_id=sid,
            group=rows[sid]["group"],
            source=source,
            overlaps_prototype_group=rows[sid]["group"] in exposed_groups,
            gold=sorted(gold),
            raw_valid=not entry.validate_prediction(raw_rows[sid]["raw"]),
            outputs=outputs,
            changes={
                a: changes(outputs["pristine"], outputs[a], gold) for a in ARMS[1:]
            },
        )
        write(DEST / source / (file_stem(sid) + ".json"), record)
        for arm, item in record["changes"].items():
            if item["output_changed"]:
                print(
                    source,
                    sid,
                    arm,
                    "codes",
                    item["before"],
                    item["after"],
                    "gold",
                    sorted(gold),
                    flush=True,
                )
    print(
        "REPLAY",
        source,
        len(records),
        round(time.perf_counter() - started, 2),
        flush=True,
    )


def totals(all_records, labels, rows):
    parts = {}
    for key, records in all_records.items():
        parts[key] = {
            "rows": len(records),
            "groups": len({v["group"] for v in records.values()}),
        }
        if not records:
            continue
        for arm in ARMS:
            finals = {s: r["outputs"][arm] for s, r in records.items()}
            metric = summarize(list(records), finals, labels, rows)
            transitions = Counter()
            per_code = Counter()
            for r in records.values():
                if arm == "pristine":
                    continue
                delta = r["changes"][arm]
                for k in ("output_changed", "codes_changed"):
                    transitions[k] += int(delta[k])
                for k in ("gained_tp", "new_fp", "lost_tp", "removed_fp"):
                    transitions[k] += len(delta[k])
                    per_code.update((k + ":" + c) for c in delta[k])
            parts[key][arm] = dict(
                metrics=metric, changes=dict(transitions), by_code=dict(per_code)
            )
    return parts


def run(watch=False):
    plan, rows = read(RUN / "plan.json"), registry()
    selected = set(plan["panel"] + plan["prototype"])
    labels = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in selected
    }
    exposed_groups = {rows[s]["group"] for s in plan["prototype"]}
    modules = frozen_modules()
    base = PROJECT / "data/baseline"
    entry = load_entry(base)
    laws, cases = (
        entry.load_rows(base / f"assets/{name}.jsonl") for name in ("laws", "train")
    )
    prototypes = {
        r["sample_id"]: r["prediction"]
        for r in read(PROJECT / "data/adapter-scale-pilot/generated-scale-1.0.json")
    }
    aggregated = False
    while True:
        added = False
        for source, ids in (
            ("fresh64", plan["panel"]),
            ("prototype18", plan["prototype"]),
        ):
            raw_rows = {}
            for sid in ids:
                name = file_stem(sid) + ".json"
                if (DEST / source / name).is_file():
                    continue
                if source == "prototype18":
                    raw_rows[sid] = {"raw": prototypes[order("kwater/scale", sid)[:32]]}
                elif (RUN / "A" / name).is_file():
                    raw_rows[sid] = read(RUN / "A" / name)
            if raw_rows:
                added = True
                replay(
                    rows,
                    raw_rows,
                    source,
                    entry,
                    modules,
                    laws,
                    cases,
                    labels,
                    exposed_groups,
                )

        if watch and aggregated and not added:
            time.sleep(20)
            continue

        def load(source):
            records = [read(p) for p in (DEST / source).glob("*.json")]
            return {r["sample_id"]: r for r in records}

        fresh = load("fresh64")
        all_records = dict(prototype18=load("prototype18"), fresh64=fresh)
        for name, overlap in (
            ("fresh_same_prototype_group", True),
            ("fresh_other_groups", False),
        ):
            all_records[name] = {
                s: r
                for s, r in fresh.items()
                if r["overlaps_prototype_group"] == overlap
            }
        write(
            DEST / "progress.json",
            dict(
                planned_fresh=64,
                pair_scope="diagnostic; released adapter saw training",
                strata=totals(all_records, labels, rows),
            ),
        )
        aggregated = True
        if not watch or len(fresh) == 64:
            return
        time.sleep(20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    run(ap.parse_args().watch)
