"""Audit 71407 training archives for localized, observable protection-state pairs."""

from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
import json
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
from .audit import inspect_bytes
from .phase2_paths import file_stem
from .phase2_panel import (
    PROJECT,
    RUN,
    read,
    write,
    order,
    registry,
    file_hash,
)

FOLDER = PROJECT / "data/external/aihub/71407"
# Both state polygons must have an explicitly named counterpart in the publisher's object dictionary.
OBJECTS = {
    1: ("SO-03", "SO-04"),
    2: ("SO-05", "SO-06"),
    3: ("SO-07", "SO-19"),
    5: ("SO-02", "SO-14"),
}
QUESTIONS = {
    1: "비계 작업발판의 설치 상태",
    2: "개폐형 작업발판 해치가 닫혀 있는 상태",
    3: "개구부 덮개의 설치 상태",
    5: "작업 단부의 난간 방호 상태",
    7: "이동식 비계 작업면의 난간 상태",
    8: "이동식 비계 발판 위 간이 사다리 사용 여부",
    9: "사다리 하단 지지면의 설치 상태",
    10: "사다리 작업자의 최상단 발판 사용 여부",
    19: "이동식 크레인 작업에서 신호수 배치 상태",
}


def prompt(scenario):
    return (
        QUESTIONS[scenario]
        + "를 사진으로 판독한다. 해당 부분에 방호가 갖춰졌으면 protected, 불량이면 deficient, "
        '사진으로 확인하기 어려우면 unknown. {"state":"값"}만 출력한다.'
    )


def receipt(path):
    st = path.stat()
    record = read(path.with_name(path.name + ".integrity.json"))
    stamp = record["stamp"]
    prior = read(path.with_name(path.name + ".json"))
    assert (stamp["bytes"], stamp["mtime_ns"], stamp["sha256"]) == (
        st.st_size,
        st.st_mtime_ns,
        prior["sha256"],
    )
    assert record["members_crc_checked"] > 0
    return stamp["sha256"]


def in_region(poly, box):
    if len(poly) < 6 or len(poly) % 2 or len(box) != 4:
        return False
    points = np.asarray(poly, dtype=float).reshape(-1, 2)
    x, y, w, h = box
    if not np.isfinite(points).all() or not all(np.isfinite(box)) or min(w, h) <= 0:
        return False
    lo = points.min(0)
    hi = points.max(0)
    intersection = np.maximum(
        0, np.minimum(hi, [x + w, y + h]) - np.maximum(lo, [x, y])
    ).prod()
    # Discard tiny objects, malformed polygons, and unrelated objects elsewhere in the frame.
    return float((hi - lo).prod()) >= 32**2 and intersection >= 0.25 * float(
        (hi - lo).prod()
    )


def inventory():
    old = read(FOLDER / "verified-state-pairs.json")["pairs"]
    banned_sites = {r["site"] for r in old}
    banned_clips = {r[state]["clip"] for r in old for state in ("normal", "abnormal")}
    stats = Counter()
    archives = {}
    clips = {}
    folder = FOLDER / "packages"
    for n, classes in OBJECTS.items():
        for state, kind, cls in zip(("Y", "N"), ("정상", "비정상"), classes):
            code = f"{state}-{n:02}"
            pattern = f"_5대사고유형_추락_{kind}_{code}.zip"
            paths = [list(folder.glob("*-" + x + pattern)) for x in ("TS", "TL")]
            if not all(len(pp) == 1 for pp in paths):
                stats["missing_complete_archive_pairs"] += 1
                continue
            photos, labels = (p[0] for p in paths)
            for p in (photos, labels):
                archives[p.name] = receipt(p)
            with ZipFile(photos) as src, ZipFile(labels) as lab:
                images = {
                    PurePosixPath(m).stem: m
                    for m in src.namelist()
                    if m.lower().endswith((".jpg", ".png", ".jpeg"))
                }
                for member in lab.infolist():
                    if not member.filename.lower().endswith(".json"):
                        continue
                    stats["annotations"] += 1
                    sid = PurePosixPath(member.filename).stem
                    row = json.loads(lab.read(member))
                    raw = row["Raw_Data_Info."]
                    anns = row["Learning_Data_Info."]["Annotations"]
                    site, clip = str(raw["Location_ID"]), raw["Raw_data_ID"]
                    if (
                        sid not in images
                        or row["Source_Data_Info."]["Source_Data"] != sid
                        or not sid.startswith(clip + "_")
                        or raw["Situation_ID"] != code
                    ):
                        stats["image_annotation_id_mismatch"] += 1
                        continue
                    if site in banned_sites or clip in banned_clips:
                        stats["reserved_external_site_or_clip"] += 1
                        continue
                    boxes = [
                        a["value"]
                        for a in anns
                        if a.get("type") == "bbox" and a.get("class_ID") == code
                    ]
                    poly = [
                        a["value"]
                        for a in anns
                        if a.get("type") == "polygon" and a.get("class_ID") == cls
                    ]
                    if not any(in_region(p, b) for p in poly for b in boxes):
                        stats["no_localized_state_polygon"] += 1
                        continue
                    start, end = [
                        datetime.fromisoformat(raw["Date"] + "T" + raw[k]).isoformat()
                        for k in ("Start_Time", "End_Time")
                    ]
                    if end <= start:
                        stats["bad_video_time"] += 1
                        continue
                    stats["eligible_frames"] += 1
                    frame = dict(
                        id=sid,
                        scenario=n,
                        site=site,
                        clip=clip,
                        process=str(raw["process_ID"]),
                        state=state,
                        archive=photos.name,
                        member=images[sid],
                        annotation_archive=labels.name,
                        annotation=member.filename,
                        class_polygon=cls,
                        scenario_box=code,
                        start=start,
                        end=end,
                    )
                    if clip not in clips or order("p38-frame", sid) < order(
                        "p38-frame", clips[clip]["id"]
                    ):
                        clips[clip] = frame
    strata = defaultdict(lambda: {"Y": [], "N": []})
    for frame in clips.values():
        strata[(frame["scenario"], frame["site"], frame["process"])][
            frame["state"]
        ].append(frame)
    pairs = []
    for sides in strata.values():
        for y, n in product(sides["Y"], sides["N"]):
            if y["end"] < n["start"] or n["end"] < y["start"]:
                pairs.append((y, n))
    official = read(PROJECT / "results/train_audit_full.json")["inventory"]
    hashes = [int(official[s]["dhash64"], 16) for s in registry()]
    used = set()
    used_clips = set()
    chosen = []
    audit_file = RUN / "external" / "audit.json"
    dest = RUN / "external" / "photos"
    dest.mkdir(parents=True, exist_ok=True)
    # Prefer distinct sites, then fill the two-pair budget with unused videos.
    ranked = sorted(
        pairs, key=lambda pair: order("p38-pair", pair[0]["id"] + pair[1]["id"])
    )
    for prefer_distinct_site, (y, n) in [
        (strict, pair) for strict in (True, False) for pair in ranked
    ]:
        if (prefer_distinct_site and y["site"] in used) or {
            y["clip"],
            n["clip"],
        } & used_clips:
            continue
        saved = []
        for f in (y, n):
            with ZipFile(folder / f["archive"]) as z:
                data = z.read(f["member"])
            info = inspect_bytes(data)
            h = int(info["dhash64"], 16)
            if any((a ^ h).bit_count() <= 4 for a in hashes + [i[2] for i in saved]):
                stats["selected_pair_near_duplicate"] += 1
                break
            saved.append((f, data, h, info))
        if len(saved) != 2:
            continue
        pair = []
        for f, data, h, info in saved:
            p = dest / (info["sha256"] + ".jpg")
            p.write_bytes(data)
            pair.append(
                dict(f, image=str(p), sha256=info["sha256"], dhash64=info["dhash64"])
            )
            hashes.append(h)
        used.add(y["site"])
        used_clips.update([y["clip"], n["clip"]])
        chosen.append(pair)
        if len(chosen) == 2:
            break
    if not chosen:
        write(
            RUN / "external/unmatched.json",
            {
                "stats": stats,
                "eligible_videos": len(clips),
                "candidate_pairs": len(pairs),
            },
        )
        raise ValueError("No verified training pair")
    diagnostic = []
    for pair in old:
        for key, state in (("normal", "Y"), ("abnormal", "N")):
            f = pair[key]
            with ZipFile(FOLDER / f["archive"]) as z:
                data = z.read(f["member"])
            import hashlib

            assert hashlib.sha256(data).hexdigest() == f["sha256"]
            p = dest / (f["sha256"] + ".jpg")
            p.write_bytes(data)
            diagnostic.append(
                dict(
                    scenario=pair["scenario"],
                    site=pair["site"],
                    clip=f["clip"],
                    id=f["id"],
                    state=state,
                    image=str(p),
                )
            )
    write(
        audit_file,
        {
            "scope": "71407 supplier training split; four fall scenarios with localized paired object taxonomy; one representative frame per clip",
            "checked_archives": archives,
            "stats": stats,
            "eligible_videos": len(clips),
            "eligible_sites": len({f["site"] for f in clips.values()}),
            "candidate_pair_combinations": len(pairs),
            "paired_sites": len({y["site"] for y, n in pairs}),
            "pair_capability_by_scenario": Counter(
                str(y["scenario"]) for y, n in pairs
            ),
            "banned_sites": sorted(banned_sites),
            "banned_videos": len(banned_clips),
            "selected": chosen,
            "diagnostic": diagnostic,
            "external_diagnostic_sha256": file_hash(
                FOLDER / "verified-state-pairs.json"
            ),
            "authority": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407",
            "signals": "Published frame-level situation box plus matching named object polygon. Human factual rating unmeasured; other risks unobserved.",
        },
    )
    print("EXTERNAL_READY", len(clips), len(pairs), len(chosen), flush=True)


def training_rows(treatment, train, rows):
    audit = read(RUN / "external/audit.json")
    count = 2 * len(audit["selected"])
    suffix = list(train[count:])
    if treatment == "resample":
        aux = sorted(train, key=lambda s: order("p38-resample", s))[:count]
        return aux + suffix, rows
    augmented = dict(rows)
    aux = []
    for pair in audit["selected"]:
        for f in pair:
            sid = "71407-" + f["id"]
            truth = f["state"] == "Y"
            if treatment == "permuted":
                truth = not truth  # swap each matched pair's two responses, keeping photo/question fixed.
            augmented[sid] = {
                "image": f["image"],
                "question": prompt(f["scenario"]),
                "state_answer": "protected" if truth else "deficient",
                "external_site": f["site"],
                "external_video": f["clip"],
            }
            aux.append(sid)
    assert len(aux) == count
    return aux + suffix, augmented


def evaluate_states(model, proc, out):
    import torch
    from .phase2_lora import inputs
    import time

    audit = read(RUN / "external/audit.json")
    cases = []
    model.eval()
    for f in audit["diagnostic"]:
        b, _ = inputs(proc, {"image": f["image"], "question": prompt(f["scenario"])})
        rec = dict(
            site=f["site"],
            scenario=f["scenario"],
            id=f["id"],
            gold="protected" if f["state"] == "Y" else "deficient",
        )
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                t = model.generate(
                    **b, do_sample=False, max_new_tokens=96, use_cache=True
                )[0, b["input_ids"].shape[-1] :]
            rec["text"] = proc.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            obj = json.loads(rec["text"])
            if (
                not isinstance(obj, dict)
                or set(obj) != {"state"}
                or obj["state"] not in ("protected", "deficient", "unknown")
            ):
                raise ValueError("invalid response")
            rec["pred"] = obj["state"]
        except (ValueError, RuntimeError, TypeError) as err:
            rec.update(error=str(err), pred=None)
        rec["seconds"] = time.perf_counter() - start
        cases.append(rec)
    write(
        out / "external-diagnostic.json",
        {
            "rows": cases,
            "n": len(cases),
            "correct": sum(r["gold"] == r["pred"] for r in cases),
            "valid": sum(r["pred"] is not None for r in cases),
            "both_states_correct": sum(
                all(x["gold"] == x["pred"] for x in cases[i : i + 2])
                for i in range(0, len(cases), 2)
            ),
        },
    )


def write_qualitative(model, proc, out):
    from PIL import Image, ImageOps
    from .phase2_official import generate

    rows = registry()
    split = read(RUN / "sft-split.json")
    for sid in sorted(split["test"], key=lambda s: order("p38-external-writing", s))[
        :6
    ]:
        final = read(out / "adapted" / (file_stem(sid) + ".json"))["final"]
        codes = list(final["hazards"])
        question = (
            f"확정된 판독 코드 {codes}. 코드는 바꾸지 않는다. 사진 속 위치와 상태를 evidence, 적용할 예방 행동을 prevention에 한 문장씩 쓴다. "
            '빈 코드이면 현장 상태만 evidence에 쓰고 prevention은 빈 문자열. 사실을 확인할 수 없으면 unsure=true. JSON {"evidence":"",'
            '"prevention":"","unsure":false} 형식을 쓴다. 법 조문은 제시하지 않는다.'
        )
        with Image.open(PROJECT / "data/train" / rows[sid]["image"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        rec = {}
        try:
            rec = generate(
                model,
                proc,
                image,
                "사진에서 직접 확인되는 정보만 기록한다.",
                question,
                144,
                enable_thinking=False,
            )
            obj = json.loads(rec["text"])
            if (
                not isinstance(obj, dict)
                or type(obj.get("unsure")) is not bool
                or any(
                    not isinstance(obj.get(k), str) for k in ("evidence", "prevention")
                )
            ):
                raise ValueError("wrong writing schema")
            rec.update(
                usable=not obj["unsure"]
                and bool(obj["evidence"].strip())
                and (not codes or bool(obj["prevention"].strip())),
                fields=obj,
            )
        except (RuntimeError, ValueError, TypeError) as err:
            rec.update(usable=False, error=str(err))
        write(
            out / "qualitative" / (file_stem(sid) + ".json"),
            {**rec, "codes": codes, "human_support_rating": None},
        )


if __name__ == "__main__":
    inventory()
