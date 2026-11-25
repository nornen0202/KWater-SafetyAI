"""Identify local provider-annotated normal/abnormal pairs in distinct source videos."""

from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np

from .aihub_ablation import AUDIT, EXPOSURE, ROOT, read
from .audit import inspect_bytes
from .train_head import save_atomic


FOLDER = ROOT / "71407"
LOCAL = FOLDER / "verified-state-pairs.json"
SCENARIOS = {*range(1, 11), 19, 20}
READY_RAW = {"487915", "487939", "487941", "487942", "487943", "487944", "487945",
             "487946", "487949", "487950", "487952", "487953", "487954", "487955", "487956"}


def stable(value):
    return sha256(("kwater-71407-state-pairs-v1|" + value).encode()).digest()


def save(path, doc):
    save_atomic(path, lambda f: f.write((json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode()))


def image_index(archive):
    members = [n for n in archive.namelist() if PurePosixPath(n).suffix.lower() in {".jpg", ".jpeg", ".png"}]
    paths = {PurePosixPath(n).stem: n for n in members}
    if len(paths) != len(members):
        raise ValueError("Duplicate source image ID")
    return paths


def collect():
    catalog = {(r["name"]): r["filekey"] for r in read(FOLDER / "catalog.json")}
    files = [("sample.zip", "sample.zip", None)]
    for name, key in sorted(catalog.items()):
        part = re.fullmatch(r"VL_5대사고유형_(?:추락|낙하)_(?:정상|비정상)_([YN]-[0-9]{2})\.zip", name)
        if not part or int(part[1][2:]) not in SCENARIOS:
            continue
        raw = "VS" + name[2:]
        rawkey = catalog[raw]
        rawpath = f"packages/{rawkey}-{raw}" if rawkey in READY_RAW else None
        files.append((f"packages/{key}-{name}", rawpath, part[1]))
    if len(files) != 25 or sum(bool(x[1]) for x in files) != 16:
        raise ValueError("Frozen 24 label, 15 raw and one sample snapshot differs")

    videos, accessible, catalog_clips = {}, {}, defaultdict(set)
    counters, inconsistent = Counter(), set()
    source_checksums = {}
    for label_file, raw_file, expected in files:
        receipt = read((FOLDER / label_file).with_name(Path(label_file).name + ".json"))
        source_checksums[label_file] = receipt["sha256"]
        if raw_file:
            r = read((FOLDER / raw_file).with_name(Path(raw_file).name + ".json"))
            source_checksums[raw_file] = r["sha256"]
            with ZipFile(FOLDER / raw_file) as raw_z:
                by_id = image_index(raw_z)
        else:
            by_id = {}
        label_src = FOLDER / label_file
        # Label packages are small. The sample also contains images and stays on disk.
        with ZipFile(BytesIO(label_src.read_bytes()) if expected else label_src) as labels:
            for entry in sorted(labels.infolist(), key=lambda z: z.header_offset):
                path = PurePosixPath(entry.filename)
                if path.suffix.lower() != ".json":
                    continue
                row = json.loads(labels.read(entry))
                raw = row["Raw_Data_Info."]
                token = str(raw.get("Situation_ID", ""))
                match = re.fullmatch(r"([YN])-([0-9]{2})", token)
                if not match or int(match[2]) not in SCENARIOS:
                    continue
                number, state = int(match[2]), match[1]
                family = "sample" if expected is None else "validation"
                counters[family + "_target_annotations"] += 1
                if (expected and expected != token) or row["Source_Data_Info."]["Source_Data"] != path.stem:
                    counters["source_id_or_scenario_mismatch"] += 1
                    continue
                if raw["Type_Description"] != ("추락" if number <= 10 else "낙하"):
                    counters["provider_type_mismatch"] += 1
                    continue
                annotations = row["Learning_Data_Info."]["Annotations"]
                box = any(a.get("class_ID") == token and a.get("type") == "bbox"
                          and len(a.get("value", [])) == 4 and np.isfinite(a["value"]).all() for a in annotations)
                if not box:
                    counters["no_frame_level_scenario_box"] += 1
                    continue
                try:
                    start, end = (datetime.fromisoformat(f"{raw['Date']}T{raw[k]}") for k in ("Start_Time", "End_Time"))
                except (KeyError, ValueError):
                    counters["unusable_video_time"] += 1
                    continue
                if end <= start:
                    counters["unusable_video_time"] += 1
                    continue
                clip = raw["Raw_data_ID"]
                meta = (number, state, str(raw["Location_ID"]), str(raw["process_ID"]), start.isoformat(), end.isoformat())
                catalog_clips[family].add(clip)
                if clip in videos and videos[clip] != meta:
                    inconsistent.add(clip)
                videos[clip] = meta
                if path.stem in by_id:
                    counters[family + "_physically_paired_frames"] += 1
                    image = {"id": path.stem, "archive": raw_file, "member": by_id[path.stem], "clip": clip}
                    rank = (family == "sample", stable(path.stem))
                    if clip not in accessible or rank < accessible[clip][0]:
                        accessible[clip] = (rank, image)
                elif raw_file:
                    counters["missing_image_in_frozen_raw"] += 1
    counters["inconsistent_video_metadata"] = len(inconsistent)
    for clip in inconsistent:
        videos.pop(clip, None)
        accessible.pop(clip, None)
    audit = {"frames": dict(counters), "sample_videos": len(catalog_clips["sample"]),
             "validation_videos": len(catalog_clips["validation"]),
             "sample_and_validation_shared_videos": len(catalog_clips["sample"] & catalog_clips["validation"]),
             "usable_distinct_videos": len(videos), "distinct_videos_with_source_photos": len(accessible)}
    return videos, {k: v[1] for k, v in accessible.items()}, audit, source_checksums


def overlapping(a, b):
    return a[2] == b[2] and a[4] <= b[5] and b[4] <= a[5]


def source_image(info):
    with ZipFile(FOLDER / info["archive"]) as z:
        return z.read(info["member"])


def select_pairs():
    videos, photos, audit, sums = collect()
    exposure = read(EXPOSURE)["exposed_ids"]
    inventory = read(AUDIT)["inventory"]
    official_hashes = [int(inventory[s]["dhash64"], 16) for s in exposure]
    strata = defaultdict(lambda: {"Y": [], "N": []})
    for clip, (number, state, site, process, *_) in videos.items():
        strata[(number, site, process)][state].append(clip)
    photo_hashes, accepted_hashes, used_videos, pairs = {}, [], [], []
    counts, selected_sites, by_num = Counter(), defaultdict(set), defaultdict(Counter)
    for scope in sorted(strata, key=lambda k: (k[0], stable(str(k)))):
        number, site, process = scope
        sides = strata[scope]
        candidates = [(a, b) for a in sides["Y"] for b in sides["N"] if a != b and not overlapping(videos[a], videos[b])]
        if candidates:
            by_num[number]["eligible_metadata_strata"] += 1
        available = [(a, b) for a, b in candidates if a in photos and b in photos]
        if available:
            by_num[number]["strata_with_both_images"] += 1
        if site in selected_sites[number] or len(selected_sites[number]) >= 2:
            continue
        for a, b in sorted(available, key=lambda pair: stable("|".join(pair))):
            if any(overlapping(videos[c], videos[old]) for c in (a, b) for old in used_videos):
                counts["overlapping_previous_time_range"] += 1
                continue
            for c in (a, b):
                if c not in photo_hashes:
                    info = inspect_bytes(source_image(photos[c]))
                    photo_hashes[c] = (info["sha256"], int(info["dhash64"], 16))
            h, j = [photo_hashes[c][1] for c in (a, b)]
            if ((h ^ j).bit_count() <= 4 or any((v ^ prior).bit_count() <= 4
                    for v in (h, j) for prior in [*official_hashes, *accepted_hashes])):
                counts["near_photo"] += 1
                continue
            accepted_hashes.extend((h, j))
            used_videos.extend((a, b))
            selected_sites[number].add(site)
            examples = [{**photos[c], "sha256": photo_hashes[c][0], "start": videos[c][4], "end": videos[c][5]}
                        for c in (a, b)]
            pairs.append({"scenario": number, "site": site, "process": process,
                          "normal": examples[0], "abnormal": examples[1]})
            by_num[number]["selected_pairs"] += 1
            break
    if len(used_videos) != len(set(used_videos)) or len(accepted_hashes) != 2 * len(pairs):
        raise ValueError("Selected source reused across pairs")
    stats = {"scope": "AI-Hub 71407 frame-level provider normal/abnormal scenario boxes; no injury outcome labels",
             "design": "Same scenario, site and construction process; distinct source videos, nonoverlapping times at same site; at most two sites per scenario. Future split by site.",
             "frozen_validation_raw_keys": sorted(READY_RAW), "validation_label_archives": 24, **audit,
             "selected_pairs": len(pairs), "unique_images": len(accepted_hashes), "unique_videos": len(set(used_videos)),
             "unique_selected_sites": len({p["site"] for p in pairs}),
             "fall_pairs": sum(p["scenario"] <= 10 for p in pairs), "crane_pairs": sum(p["scenario"] >= 19 for p in pairs),
             "rejections_during_selection": dict(counts),
             "by_scenario": {str(n): dict(by_num[n]) for n in sorted(SCENARIOS)},
             "source": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407"}
    save(LOCAL, {"source_receipt_sha256": sums, "pairs": pairs})
    stats["local_pair_selection_sha256"] = sha256(LOCAL.read_bytes()).hexdigest()
    save(Path("results/aihub_state_pairs.json"), stats)
    return stats


if __name__ == "__main__":
    result = select_pairs()
    print("Pairs", result["selected_pairs"], "distinct videos", result["unique_videos"], flush=True)
