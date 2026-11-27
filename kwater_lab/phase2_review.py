"""Frozen paired rereads and image-conditioned writing for fixed code sets."""

from copy import deepcopy
from functools import lru_cache
import json
import os

from .metrics import decode_target, unique_keys
from .phase2_paths import file_stem
from .phase2_panel import ALLOWED8, RUN, order, read, registry, write
from .submission2_output import make_row


CONDITIONS = {
    "fall_height": "작업자와 단부·개구부의 위치 및 추락 방호",
    "slip_trip": "실제 이동 경로의 바닥·장애물·미끄러운 상태",
    "struck_by_lifting": "들린 하중·낙하 경로와 사람의 위치",
    "caught_in_machine": "기계 가동부·협착점과 신체 접근",
    "electrical_energy": "노출된 전기 위험원과 접촉 가능한 위치",
    "fire_explosion": "점화원과 연소·폭발 가능한 물질",
    "chemical_exposure": "확인되는 물질·누출 또는 노출 경로",
    "other": "위 항목 밖의 구체적인 위험원과 노출 상황",
}
SYSTEM = "당신은 현장사진의 위험을 재검토한다. 사진에서 확인한 위치와 상태만 쓴다. 보이지 않으면 판단 곤란을 선택한다. 반드시 JSON 한 개만 출력한다."


def assign(ids, candidate, base, rows):
    ids = list(ids)

    @lru_cache(maxsize=None)
    def solve(mask):
        i = mask.bit_count()
        if i == len(ids):
            return 0, ()
        s = ids[i]
        best = (float("inf"), ())
        for j, t in enumerate(ids):
            if (
                mask & (1 << j)
                or rows[s]["group"] == rows[t]["group"]
                or (
                    "dev_fold" in rows[s] and rows[s]["dev_fold"] == rows[t]["dev_fold"]
                )
            ):
                continue
            cost, path = solve(mask | (1 << j))
            cost += (
                (candidate[t] in base[s]) * 100000
                + (candidate[t] == candidate[s]) * 100
                + int(order(s, t)[:3], 16) / 4096
            )
            best = min(best, (cost, (j,) + path))
        return best

    cost, path = solve(0)
    if cost == float("inf"):
        raise ValueError("No cross-group candidate permutation")
    return {s: ids[j] for s, j in zip(ids, path, strict=True)}


def review_prompt(original, hint, conditions=CONDITIONS):
    choices = " / ".join(f"{k} ({v})" for k, v in conditions.items())
    cue = (
        "후보 없음. 스스로 다른 위험을 찾는다."
        if hint is None
        else "확인할 후보 " + hint + " (" + conditions[hint] + ")."
    )
    return (
        f"기존 코드 {list(original)}. 기존 판독을 유지하며 추가 위험 하나만 검토한다. {cue} "
        f"허용 위험 {choices}. 후보나 시설 이름만으로 채택하지 않는다. 위치와 위험을 뒷받침하는 상태를 모두 확인한다. "
        'JSON {"decision":"accept/reject/unsure 중 하나","code":"허용 코드 또는 빈 문자열","evidence":"사진 근거","prevention":"구체적 안전조치"}.'
    )


def interpret(text, existing, hint, allowed=ALLOWED8):
    try:
        obj = json.loads(text.strip(), object_pairs_hook=unique_keys)
        if not isinstance(obj, dict) or obj.get("decision") not in (
            "accept",
            "reject",
            "unsure",
        ):
            return "parse_failure", None
        if obj["decision"] != "accept":
            return obj["decision"], None
        c = obj.get("code")
        if c not in allowed or (hint is not None and c != hint) or c in existing:
            return "duplicate_or_wrong_code", None
        if any(
            not isinstance(obj.get(k), str) or not obj[k].strip() or len(obj[k]) > lim
            for k, lim in [("evidence", 240), ("prevention", 480)]
        ):
            return "missing_or_invalid_support", None
        return "accept", obj
    except (ValueError, TypeError):
        return "parse_failure", None


def apply_review(original, generation, hint):
    final = deepcopy(original)
    status, obj = interpret(generation["text"], set(final["hazards"]), hint)
    if obj is not None:
        c = obj["code"]
        item = make_row(final["image_id"], [c])["hazards"][c]
        item.pop("references", None)
        item.update(
            evidence=obj["evidence"].strip(), prevention=obj["prevention"].strip()
        )
        final["hazards"][c] = item
        final["status"] = "hazard"
        final.pop("scene_evidence", None)
        try:
            decode_target(final)
        except ValueError:
            return "schema_failure", deepcopy(original)
    return status, final


def repeat_sample_kit(model, proc):
    from .adapter_scale_pilot import load_entry
    from .phase2_contract import check
    from .phase2_panel import PROJECT, file_hash
    from unittest.mock import patch
    import time

    dest = RUN / "fresh-qwen-kit"
    if (dest / "receipt.json").exists():
        return
    dest.mkdir(exist_ok=True)
    inp = PROJECT / "data/local-kit/sample-input"
    base = PROJECT / "data/baseline"
    entry = load_entry(base)
    rows = [
        {"sample_id": r["image_id"], "image": "images/" + r["image_id"] + ".jpg"}
        for r in entry.load_rows(inp / "manifest.jsonl")
    ]
    texts = []
    parse = entry.parse_prediction

    def capture(txt):
        texts.append(txt)
        return parse(txt)

    start = time.perf_counter()
    with (
        patch.object(entry, "load_model", return_value=(model, proc)),
        patch.object(entry, "parse_prediction", capture),
    ):
        raw = entry.infer_batched(rows, inp)
    valid = [r for r in raw if not entry.validate_prediction(r["prediction"])]
    examples, laws = (
        entry.load_rows(base / f"assets/{f}.jsonl") for f in ("train", "laws")
    )
    formatted = {}
    if valid:
        entry.format_submission(valid, laws, examples, dest / "format.jsonl")
        formatted = {
            r["sample_id"]: r["prediction"]
            for r in entry.load_rows(dest / "format.jsonl")
        }
    final = [entry.convert(r["sample_id"], formatted.get(r["sample_id"])) for r in raw]
    out = dest / "out"
    out.mkdir(exist_ok=True)
    entry.write_jsonl(out / "submission.jsonl", final)
    write(dest / "raw.json", {"parsed": raw, "texts": texts})
    past = entry.load_rows(
        PROJECT / "data/official-baseline-10/output/submission.jsonl"
    )
    assert [r["image_id"] for r in past] == [r["image_id"] for r in final]
    write(
        dest / "receipt.json",
        {
            "images": len(raw),
            "valid_generations": len(valid),
            "seconds": time.perf_counter() - start,
            "checker": check(inp, out),
            "matching_rows": sum(x == y for x, y in zip(final, past)),
            "matching_code_sets": sum(
                set(x["hazards"]) == set(y["hazards"]) for x, y in zip(final, past)
            ),
            "submission_sha256": file_hash(out / "submission.jsonl"),
        },
    )


def review(model, proc, image_for):
    from .phase2_official import generate, save_missing

    plan, rows = read(RUN / "plan.json"), registry()
    pending = [
        s for s in plan["panel"] if not (RUN / "A" / (file_stem(s) + ".json")).exists()
    ]
    if pending:
        save_missing(model, proc, pending, rows)
    first_fit = RUN / "first-fit-after-a64.json"
    if first_fit.is_file() and not (RUN / "lora-3vl/results.json").is_file():
        remaining = [
            s
            for s in plan["panel"]
            if not (RUN / "A" / (file_stem(s) + ".json")).is_file()
        ]
        write(
            RUN / "a64-first-fit-handoff.json",
            dict(
                saved=len(plan["panel"]) - len(remaining),
                missing=remaining,
                next_model="Qwen3-VL-4B fresh QLoRA",
                remaining_frozen_review=plan["review"],
                kit_done=(RUN / "fresh-qwen-kit/receipt.json").is_file(),
            ),
        )
        if remaining:
            raise RuntimeError(
                f"Original frozen generation still missing {len(remaining)} photos"
            )
        return
    ids = plan["review"]
    scores = read(RUN / "dev-heads.json")["scores"]
    base = {s: read(RUN / "A" / (file_stem(s) + ".json"))["final"] for s in ids}
    codes = {s: set(base[s]["hazards"]) for s in ids}

    def proposals(name):
        return {
            s: max((c for c in ALLOWED8 if c not in codes[s]), key=scores[s][name].get)
            for s in ids
        }

    true, alternative = proposals("sig"), proposals("dino")
    eligible = [s for s in ids if len(codes[s]) < 3]
    donor = assign(eligible, true, codes, rows)
    swapped = {s: true[t] for s, t in donor.items()}
    capped = [s for s in ids if s not in donor]
    for s in capped:
        # These hints are recorded only. Match the capped C frequencies and use another training photo.
        code = true[s]
        candidates = [
            t
            for t in ids
            if rows[t]["dev_fold"] != rows[s]["dev_fold"] and true[t] == code
        ]
        if not candidates:
            candidates = [
                t
                for t in scores
                if t not in ids
                and rows[t]["dev_fold"] != rows[s]["dev_fold"]
                and max(ALLOWED8, key=scores[t]["sig"].get) == code
            ]
        if candidates:
            donor[s] = min(candidates, key=lambda t: order("cap-donor/" + s, t))
            swapped[s] = code
    design = {
        "ids": ids,
        "eligible": eligible,
        "C": true,
        "D_donor": donor,
        "D_hint": swapped,
        "D_cap_only": capped,
        "D_unmatched_cap_only": [s for s in capped if s not in swapped],
        "E_dino": alternative,
    }
    from .phase2_donors import gallery, sources

    if not (RUN / "sig224-training-donor-scores.json").is_file():
        gallery()
    design["D_training_sources"] = sources(ids, swapped, rows)
    path = RUN / "review-assignment.json"
    if path.exists() and read(path) != design:
        raise ValueError("Review assignment changed")
    write(path, design)
    for i, s in enumerate(ids):
        pic = image_for(s)
        for arm in sorted("BCDE", key=lambda k: order(s, k)):
            dest = RUN / arm / (file_stem(s) + ".json")
            if dest.exists():
                continue
            hint = {
                "B": None,
                "C": true[s],
                "D": swapped.get(s),
                "E": alternative[s],
            }[arm]
            final = deepcopy(base[s])
            record = {
                "sample_id": s,
                "condition": arm,
                "hint": hint,
                "base_codes": list(final["hazards"]),
            }
            status = "cap_proposal_only"
            if s in eligible:
                prompt = review_prompt(final["hazards"], hint)
                record.update(prompt=prompt)
                try:
                    generation = generate(model, proc, pic, SYSTEM, prompt, 144)
                    record.update(generation)
                    status, final = apply_review(base[s], generation, hint)
                except (RuntimeError, ValueError) as err:
                    status, record["error"], final = (
                        "generation_failure",
                        str(err),
                        deepcopy(base[s]),
                    )
            record.update(status=status, final=final)
            write(dest, record)
        print("REVIEW_SAVED", i + 1, len(ids), flush=True)
    repeat_sample_kit(model, proc)
    if (
        os.environ.get("KW_PHASE2_CORE_ONLY") == "1"
        or (RUN / "first-qwen-training-priority.json").is_file()
        and not (RUN / "lora-3vl/results.json").is_file()
    ):
        write(
            RUN / "deferred-auxiliary.json",
            dict(
                source="msg_96933c1e63a7"
                if first_fit.is_file()
                else "msg_50cf42cdb128",
                core_completed=["A64", "BCDE16", "official-kit10"],
                remaining=["factorial", "region", "exclusive"],
                resume="same official runner reuses all saved core rows",
            ),
        )
        return
    factorial(model, proc, image_for, base)
    from .phase2_region_review import paired

    paired(model, proc, image_for)
    from .phase2_exclusive_review import SPEC, paired as exclusive_pairs

    if SPEC.is_file():
        exclusive_pairs(model, proc, image_for)
    from . import phase2_agreement

    if phase2_agreement.SPEC.is_file():
        phase2_agreement.evidence(model, proc, image_for)


def image_writing(model, proc, pic, code, cache):
    from .phase2_official import generate

    empty = code == "no_visible_hazard"
    if not cache.exists():
        prompt = (
            f"지정 코드 {code}. "
            + (
                "사진에서 보이는 작업장 상태 한 문장을 scene_evidence에 쓴다."
                if empty
                else "위험원의 위치·상태를 evidence에, 이를 해결할 구체적 조치를 prevention에 쓴다."
            )
            + " 사진이 지지하지 않으면 unsure=true, 문장을 비운다. 지정한 필드와 unsure만 있는 JSON 하나를 작성한다."
        )
        record = {}
        try:
            record = generate(model, proc, pic, SYSTEM, prompt, 160)
            obj = json.loads(record["text"], object_pairs_hook=unique_keys)
            keys = ["scene_evidence"] if empty else ["evidence", "prevention"]
            if not isinstance(obj, dict) or type(obj.get("unsure")) is not bool:
                raise ValueError("missing explicit support decision")
            status = "unsupported" if obj["unsure"] else "generated"
            if status == "generated" and any(
                not isinstance(obj.get(k), str)
                or not obj[k].strip()
                or len(obj[k]) > (480 if k == "prevention" else 240)
                for k in keys
            ):
                raise ValueError("missing or overlong text")
            record.update(
                status=status,
                fields={k: obj[k] for k in keys} if status == "generated" else {},
            )
        except (ValueError, TypeError, RuntimeError) as err:
            record.update(status="failed", error=str(err), fields={})
        write(cache, record)
    return read(cache)


def factorial(model, proc, image_for, base):

    second = {
        r["image_id"]: r
        for r in map(
            json.loads,
            (RUN / "fusion-panel/submission.jsonl").read_bytes().splitlines(),
        )
    }
    for s, first in base.items():
        pic = image_for(s)
        for detector, original in [("A", first), ("S2", second[first["image_id"]])]:
            dest = RUN / "factorial" / (file_stem(s) + "-" + detector + ".json")
            if dest.exists():
                continue
            fixed = make_row(first["image_id"], list(original["hazards"]))
            for item in fixed["hazards"].values():
                item.pop("references", None)
            generated = deepcopy(fixed)
            records = {}
            for code in list(fixed["hazards"]) or ["no_visible_hazard"]:
                cache = RUN / "writing" / (file_stem(s) + "-" + code + ".json")
                empty = code == "no_visible_hazard"
                record = image_writing(model, proc, pic, code, cache)
                records[code] = record["status"]
                if record["status"] == "generated":
                    (generated if empty else generated["hazards"][code]).update(
                        record["fields"]
                    )
                elif empty:
                    generated["scene_evidence"] = (
                        "사진에서 현장 상태를 확인하기 어렵다."
                    )
                else:
                    generated["hazards"][code].update(
                        evidence="사진에서 이 코드의 근거를 확인하기 어렵다.",
                        prevention="작업 전 해당 위험원의 존재와 방호 상태를 현장에서 확인한다.",
                    )
            assert decode_target(generated) == decode_target(fixed)
            write(dest, {"fixed": fixed, "generated": generated, "writing": records})
        print("WRITING_SAVED", s, flush=True)
