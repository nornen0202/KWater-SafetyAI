"""Audit development split exposure and record the effective ancestry of each image fit."""

from .metrics import CODES
from .phase2_panel import PROJECT, RUN, file_hash, read, registry, write


def support(ids, rows):
    ids = set(ids)
    result = {}
    for code in CODES:
        positive = [s for s in ids if code in rows[s].get("codes", ())]
        result[code] = dict(
            photos=len(positive), groups=len({rows[s]["group"] for s in positive})
        )
    return dict(
        positive_by_type=result,
        unsupported=[c for c in CODES if not result[c]["photos"]],
    )


def overlap(train, test, rows):
    known = {rows[s]["group"] for s in train if "group" in rows[s]}
    shared = [s for s in test if rows[s]["group"] in known]
    return dict(
        same_photos=sorted(set(train) & set(test)),
        photos_from_training_groups=sorted(set(shared)),
        same_groups=sorted({rows[s]["group"] for s in shared}),
    )


def preflight():
    rows = registry()
    plans = {
        name: read(path)
        for name, path in (
            ("pilot32", RUN / "sft-split.json"),
            ("panel170", RUN / "large-models/expansion-split.json"),
            ("outer727", RUN / "large-models/expansion-outer1.json"),
        )
    }
    comparisons = {}
    for name, plan in plans.items():
        splits = {}
        for other, v in plans.items():
            splits[other] = overlap(plan["train"], v["test"], rows)
        if splits[name]["same_groups"]:
            raise ValueError(f"Train/evaluation group leak in {name}")
        comparisons[name] = dict(
            train_photos=len(plan["train"]),
            evaluation_photos=len(plan["test"]),
            train=support(plan["train"], rows),
            test=support(plan["test"], rows),
            overlap_by_test=splits,
        )
    write(RUN / "development-ancestry-audit.json", comparisons)
    for name, r in comparisons.items():
        print(
            name,
            r["train_photos"],
            r["evaluation_photos"],
            len(CODES) - len(r["train"]["unsupported"]),
            {
                k: (
                    len(v["same_photos"]),
                    len(v["photos_from_training_groups"]),
                    len(v["same_groups"]),
                )
                for k, v in r["overlap_by_test"].items()
            },
            flush=True,
        )


def record(folder, name, base_path, train, test, rows, split_path, model):
    from pathlib import Path

    if getattr(model, "peft_config", None) or any(
        "lora_" in k for k, _ in model.named_parameters()
    ):
        raise ValueError(
            "Fresh fitting requires an unloaded base and zero inherited adapters"
        )
    if any(p.requires_grad for p in model.parameters()) and getattr(
        model, "_hf_peft_config_loaded", False
    ):
        raise ValueError("Base model reports an integrated PEFT adapter")
    parent = []
    group_check = overlap(train, test, rows)
    if group_check["same_groups"] or group_check["same_photos"]:
        raise ValueError("Effective training ancestry overlaps the evaluation split")
    local = Path(base_path)
    if name in ("8b", "9b"):
        source = read(local / "source.json")
        repo, revision = source["repo"], source["revision"]
        identity = read(local / "complete.json")
    else:
        repo = {"3vl": "Qwen/Qwen3-VL-4B-Instruct", "35": "Qwen/Qwen3.5-4B"}[name]
        source = read(RUN / "public-sources.json")[repo]
        revision = source["revision"]
        identity = dict(
            published_lfs={
                f["rfilename"]: f["lfs"]["sha256"]
                for f in source["files"]
                if f.get("lfs")
            }
        )
        if name == "3vl":
            identity["organizer_archive_sha256"] = read(
                PROJECT / "results/official_baseline_archive.json"
            )["original_sha256"]
    train_groups = sorted({rows[s]["group"] for s in train if "group" in rows[s]})
    write(
        folder / "lineage.json",
        dict(
            base_repo=repo,
            model_card_revision=revision,
            local_base=str(local),
            base_config_sha256=file_hash(local / "config.json"),
            source_identity=identity,
            parent_adapters=parent,
            initialized_from_base=True,
            split_sha256=file_hash(split_path),
            actual_ordered_training_ids=train,
            unique_training_groups=train_groups,
            evaluation_ids=test,
            evaluation_groups=sorted({rows[s]["group"] for s in test}),
            cumulative_training_ids=sorted(set(train)),
            cumulative_group_overlap=group_check,
            train_support=support(train, rows),
            evaluation_support=support(test, rows),
            external_sites=sorted(
                {rows[s]["external_site"] for s in train if "external_site" in rows[s]}
            ),
            external_videos=sorted(
                {
                    rows[s]["external_video"]
                    for s in train
                    if "external_video" in rows[s]
                }
            ),
            independence_scope="Independence from this checkpoint's fitting history; prior project contact is tracked separately.",
        ),
    )


if __name__ == "__main__":
    preflight()
