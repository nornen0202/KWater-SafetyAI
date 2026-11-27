"""Create localized image/code verification examples from the fixed training-only matches."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .metrics import CODES, unique_keys
from .phase2_panel import PROJECT, RUN, file_hash, order, read, registry, write
from .phase2_score import readlines
from .train_head import save_atomic

ROOT = RUN / "verifier-16pairs"
SPEC = RUN / "manager-verifier-train16pairs.json"
PIN = "826bdd0f4ca4b48f0f0b4256f869da7dcebac50cc9052f090297f41e3397ba84"


def question(code):
    if code not in CODES:
        raise ValueError("Unknown hazard proposal")
    return (
        f"사진에서 제안 코드 {code}에 해당하는 위험을 검증한다. "
        "사진의 상태를 관찰하고 확인되면 accept, 근거가 없으면 reject, 판별하기 어려우면 unsure로 쓴다. "
        "evidence에 확인한 위치와 상태를 쓰고 accept이면 prevention에 구체적인 조치를, 그 밖에는 빈 문자열을 쓴다. "
        '출력은 {"decision":"accept/reject/unsure","evidence":"관찰 근거","prevention":"조치 또는 빈 문자열"} 형식 JSON 한 개다.'
    )


def parse(text):
    obj = json.loads(text.strip(), object_pairs_hook=unique_keys)
    if not isinstance(obj, dict) or set(obj) != {"decision", "evidence", "prevention"}:
        raise ValueError("Wrong verifier object")
    status = obj["decision"]
    if (
        status not in ("accept", "reject", "unsure")
        or not isinstance(obj["evidence"], str)
        or not isinstance(obj["prevention"], str)
    ):
        raise ValueError("Invalid verifier fields")
    if not obj["evidence"].strip() or bool(obj["prevention"].strip()) != (
        status == "accept"
    ):
        raise ValueError("Missing observation or invalid prevention")
    return obj


def prepare():
    if file_hash(SPEC) != PIN:
        raise ValueError("The manager's fixed training IDs changed")
    spec = read(SPEC)
    for rel, h in spec["source_files_sha256"].items():
        if file_hash(PROJECT / Path(rel.replace("\\", "/"))) != h:
            raise ValueError(f"Changed source {rel}")
    rows = registry()
    split = read(RUN / "large-models/expansion-split.json")
    allowed = set(split["train"])
    panel = set(split["test"])
    historical = set(read(RUN / "plan.json")["prototype"])
    anchors = read(PROJECT / "data/prompt-pilot/design.json")["anchors"]
    historical.update(a["sid"] for a in anchors)
    old_hashes = {rows[s]["sha256"] for s in historical if s in rows} | {
        a["sha256"] for a in anchors
    }
    allowed = {
        s
        for s in allowed
        if s not in historical and rows[s]["sha256"] not in old_hashes
    }
    if len(allowed) != 146:
        raise ValueError("Unexpected number of unaffected training images")
    labels = {r["sample_id"]: r["target"] for r in readlines(RUN / "dev-train.jsonl")}
    data, provenance = {}, {}
    seen_hash = set()
    for pair in spec["pairs"]:
        code = pair["candidate"]
        for field in ("positive_id", "safe_id"):
            s = pair[field]
            r = rows[s]
            target = labels[s]
            if (
                s not in allowed
                or s in data
                or r["sha256"] in seen_hash
                or r["group"] != pair["group"]
            ):
                raise ValueError("Duplicated or ineligible matched training image")
            path = PROJECT / "data/train" / r["image"]
            if file_hash(path) != r["sha256"]:
                raise ValueError("Original training image hash differs")
            seen_hash.add(r["sha256"])
            if field == "positive_id":
                h = target["hazards"][code]
                out = dict(
                    decision="accept",
                    evidence=h["evidence"],
                    prevention=h["prevention"],
                )
            else:
                if target.get("status") != "no_visible_hazard" or target["hazards"]:
                    raise ValueError("Expected a labeled clear image")
                out = dict(
                    decision="reject", evidence=target["scene_evidence"], prevention=""
                )
            parse(json.dumps(out, ensure_ascii=False))
            data[s] = dict(input=dict(image=r["image"], candidate=code), target=out)
            provenance[s] = dict(
                source_id=s,
                sha256=r["sha256"],
                group=r["group"],
                pair_id=pair["pair_id"],
                candidate=code,
            )
    train_groups = {v["group"] for v in provenance.values()}
    if (
        len(data) != 32
        or seen_hash & {rows[s]["sha256"] for s in panel}
        or train_groups & {rows[s]["group"] for s in panel}
    ):
        raise ValueError("Verifier training images/groups overlap the frozen panel")
    ids = sorted(data, key=lambda s: order("p38-verifier-sft-order", s))
    payload = b"".join(
        (json.dumps(data[s], ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for s in ids
    )
    save_atomic(ROOT / "training.jsonl", lambda f: f.write(payload))
    write(ROOT / "provenance.json", [provenance[s] for s in ids])
    write(
        ROOT / "audit.json",
        dict(
            spec_sha256=PIN,
            training_sha256=hashlib.sha256(payload).hexdigest(),
            rows=len(ids),
            pairs=len(spec["pairs"]),
            decisions=dict(Counter(v["target"]["decision"] for v in data.values())),
            unique_groups=len(train_groups),
            by_code=dict(Counter(v["input"]["candidate"] for v in data.values())),
            unsupported=sorted(
                set(CODES) - {v["input"]["candidate"] for v in data.values()}
            ),
            eval_panel=len(panel),
            shared_hashes=0,
            shared_groups=0,
            input_fields=["image", "candidate"],
            historical_excluded_photos=spec["excluded_historical_ids_or_hashes"],
            source_files_sha256=spec["source_files_sha256"],
            optimizer_steps_completed=0,
            scope="Verified supervision records only; no preference optimization.",
        ),
    )


def condition():
    evidence = RUN / "agreement-independent/evidence-cost.json"
    if not evidence.is_file():
        return None
    observations = read(RUN / "agreement-independent/rows.json")
    false_accepts = []
    missing_support = []
    for p in (evidence.parent / "evidence").glob("*.json"):
        r = read(p)
        s = r["sample_id"]
        if not observations[s]["gold"] and r["status"] == "accept":
            false_accepts.append(s)
        if r["status"] in (
            "failure",
            "parse_failure",
            "missing_or_invalid_support",
            "schema_failure",
        ):
            missing_support.append(s)
    verdict = dict(
        eligible=bool(false_accepts or missing_support),
        false_accepts_on_clear_photos=false_accepts,
        missing_parseable_support=missing_support,
        source_sha256=file_hash(evidence),
        scope="Conditional diagnostic pilot; evaluation cohort previously examined to initiate this test.",
    )
    write(ROOT / "trigger.json", verdict)
    return verdict


def fit_data(rows):
    if not (ROOT / "audit.json").exists():
        prepare()
    audit = read(ROOT / "audit.json")
    if audit["training_sha256"] != file_hash(ROOT / "training.jsonl"):
        raise ValueError("Verifier supervision changed")
    data = readlines(ROOT / "training.jsonl")
    origins = read(ROOT / "provenance.json")
    train = [o["source_id"] for o in origins]
    updated = dict(rows)
    for s, v in zip(train, data, strict=True):
        updated[s] = dict(
            rows[s],
            question=question(v["input"]["candidate"]),
            verifier_target=v["target"],
        )
    observed = read(RUN / "agreement-independent/rows.json")
    panel = read(RUN / "plan.json")["panel"]
    test = [s for s in panel if observed[s]["additions"]["confirmed"]]
    if len(observed) != 64 or not test:
        raise ValueError("The full frozen verification cohort is not available")
    from .phase2_paths import file_stem

    for s in test:
        code = observed[s]["candidate"]
        original = read(RUN / "A" / (file_stem(s) + ".json"))["final"]
        updated[s] = dict(
            rows[s], question=question(code), candidate=code, original=original
        )
    split = ROOT / "fit-split.json"
    spec = dict(
        train=train,
        test=test,
        spec_sha256=PIN,
        source_rows_sha256=file_hash(RUN / "agreement-independent/rows.json"),
        selection="All triggered candidates in the two predeclared photo cohorts, independent of candidate correctness.",
    )
    if split.exists() and read(split) != spec:
        raise ValueError("Verifier fit split differs")
    write(split, spec)
    return train, test, updated, split


def evaluate_verifier(model, proc, rows, ids, folder, *unused):
    from copy import deepcopy
    import time
    import torch
    from .phase2_lora import inputs, observed_thinking_tokens
    from .phase2_score import summarize, lifting_review
    from .phase2_paths import file_stem
    from .submission2_output import make_row
    from .phase2_release import changes
    from .metrics import decode_target

    gold = {
        r["sample_id"]: r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] in ids
    }
    model.eval()
    pred = {}
    records = {}
    counts = Counter()
    for s in ids:
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        candidate = rows[s]["candidate"]
        final = deepcopy(rows[s]["original"])
        rec = dict(candidate=candidate, valid=False)
        try:
            b, vision = inputs(proc, rows[s])
            rec["visual_tokens"] = vision
            with torch.inference_mode():
                t = model.generate(
                    **b, max_new_tokens=256, do_sample=False, use_cache=True
                )[0, b["input_ids"].shape[1] :]
            out = proc.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            rec.update(
                text=out,
                limit_reached=len(t) == 256,
                new_tokens=len(t),
                **observed_thinking_tokens(
                    proc.tokenizer, b["input_ids"][0].tolist(), t.tolist()
                ),
            )
            parsed = parse(out)
            if parsed["decision"] == "accept":
                h = make_row(final["image_id"], [candidate])["hazards"][candidate]
                h.pop("references", None)
                h.update(evidence=parsed["evidence"], prevention=parsed["prevention"])
                final["hazards"][candidate] = h
                final["status"] = "hazard"
                final.pop("scene_evidence", None)
                decode_target(final)
            rec.update(valid=True, status=parsed["decision"], response=parsed)
        except (ValueError, RuntimeError, OSError, TypeError) as e:
            rec.update(error=repr(e), status="failure")
            final = rows[s]["original"]
        torch.cuda.synchronize()
        rec.update(
            final=final,
            seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            changes=changes(rows[s]["original"], final, gold[s]),
        )
        write(folder / (file_stem(s) + ".json"), rec)
        records[s] = rec
        pred[s] = final
        counts[
            ("correct_proposal/" if candidate in gold[s] else "incorrect_proposal/")
            + rec["status"]
        ] += 1
    return dict(
        metrics=summarize(ids, pred, gold, rows),
        rows=len(ids),
        valid=sum(r["valid"] for r in records.values()),
        verdicts=dict(counts),
        lifting_decisions=lifting_review(records, gold),
        seconds=sum(r["seconds"] for r in records.values()),
        peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in records.values()),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "train"], nargs="?", default="prepare")
    if p.parse_args().stage == "train":
        if not (condition() or {}).get("eligible"):
            raise SystemExit(
                "Awaiting observed unsupported or false acceptance in the fixed source verifier"
            )
        from .phase2_lora import run

        run("3vl", verifier=True)
    else:
        prepare()
