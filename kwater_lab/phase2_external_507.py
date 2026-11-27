"""Audit the local rope-work image/annotation pairs and explicit observable labels."""

from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from .aihub_ablation import paired
from .phase2_io import PROJECT, RUN, read, write


def archive(folder, key):
    (path,) = folder.glob(f"{key}-*.zip")
    receipt = read(Path(str(path) + ".integrity.json"))
    if (
        path.stat().st_size != receipt["stamp"]["bytes"]
        or not receipt["members_crc_checked"]
    ):
        raise ValueError("The original archive no longer matches its integrity ledger")
    return path, receipt


def check():
    root = PROJECT / "data/external/aihub/507/packages"
    folder = RUN / "external/507-rope"
    report, evidence = {}, []
    for split, image_key, label_key in (
        ("training", 68016, 68029),
        ("validation", 68035, 68023),
    ):
        src, src_receipt = archive(root, image_key)
        lab, lab_receipt = archive(root, label_key)
        by_video, sites, labels, situations = (
            defaultdict(list),
            set(),
            Counter(),
            Counter(),
        )
        excluded = Counter()
        with ZipFile(src) as images, ZipFile(BytesIO(lab.read_bytes())) as annotations:
            photos, notes = paired(images, annotations)
            for sid, member in notes.items():
                row = json.loads(annotations.read(member))
                raw, original, learning = (
                    row["Raw Data Info."],
                    row["Source Data Info."],
                    row["Learning Data Info."],
                )
                video, site, scene = (
                    raw["raw_data_ID"],
                    raw["location_ID"],
                    raw["situation_ID"],
                )
                if (
                    original["source_data_ID"] != sid
                    or Path(photos[sid]).parent.name != video
                ):
                    excluded["original_id_or_video"] += 1
                    continue
                width, height = raw["resolution"]
                codes = set()
                for item in learning["annotation"]:
                    if "box" not in item:
                        continue
                    x, y, w, h = item["box"]
                    if (
                        not all(map(math.isfinite, (x, y, w, h)))
                        or x < 0
                        or y < 0
                        or w <= 0
                        or h <= 0
                        or x + w > width + 1
                        or y + h > height + 1
                    ):
                        excluded["invalid_box_instances"] += 1
                        continue
                    codes.add(item["class_id"])
                labels.update(codes)
                situations[scene] += 1
                by_video[video].append((sid, site, width, height, sorted(codes)))
                sites.add(site)
            for video, frames in sorted(by_video.items()):
                sid, site, width, height, codes = min(
                    frames,
                    key=lambda x: sha256(("p38/507/frame/" + x[0]).encode()).digest(),
                )
                payload = images.read(photos[sid])
                with Image.open(BytesIO(payload)) as pic:
                    pic.load()
                    valid = pic.format == "JPEG" and pic.size == (width, height)
                evidence.append(
                    dict(
                        split=split,
                        sample_id=sid,
                        video=video,
                        site=site,
                        classes=codes,
                        jpeg_and_dimensions_valid=valid,
                        sha256=sha256(payload).hexdigest(),
                        member_crc=images.getinfo(photos[sid]).CRC,
                    )
                )
            report[split] = dict(
                archive_members_paired=len(photos),
                linked=sum(map(len, by_video.values())),
                site_ids=sorted(sites),
                video_ids=sorted(by_video),
                situations=situations,
                image_counts_by_box_class=labels,
                excluded=excluded,
                source_zip_sha256_from_ledger=src_receipt["stamp"]["sha256"],
                label_zip_sha256_from_ledger=lab_receipt["stamp"]["sha256"],
            )
    a, b = report["training"], report["validation"]
    report.update(
        shared_videos=sorted(set(a["video_ids"]) & set(b["video_ids"])),
        shared_sites=sorted(set(a["site_ids"]) & set(b["site_ids"])),
        jpeg_verified=sum(r["jpeg_and_dimensions_valid"] for r in evidence),
        jpeg_checked=len(evidence),
        stage="source audit only",
        trained_images=0,
        source="https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=507",
        ontology={
            "WO-01": "작업자",
            "SO-06": "로프",
            "WO-06": "안전고리 미착용",
            "UA-01": "안전고리 미체결",
        },
        explicit_fastening_positive_not_inferred_from_absence=True,
    )
    write(folder / "jpeg-ledger.json", evidence)
    write(folder / "inventory.json", report)
    print(
        "507",
        report["jpeg_verified"],
        report["jpeg_checked"],
        len(report["shared_videos"]),
        flush=True,
    )


def common_hook():
    """Check the available common-work diagnostic archive for explicit unfastened scenarios."""
    root = PROJECT / "data/external/aihub/507/packages"
    (src,) = root.glob("68034-*.zip")
    src_download = read(Path(str(src) + ".json"))
    if src_download["bytes"] != src.stat().st_size or len(src_download["sha256"]) != 64:
        raise ValueError("Common-work diagnostic source differs from its finished download")
    lab, lab_stamp = archive(root, 68022)
    folder = RUN / "external/507-common-hook"
    counts, excluded, by_video = Counter(), Counter(), defaultdict(list)
    matched = []
    with ZipFile(src) as images, ZipFile(BytesIO(lab.read_bytes())) as annotations:
        if len(images.infolist()) != src_download["zip_members"]:
            raise ValueError("Common-work central directory changed since the download")
        photos, notes = paired(images, annotations)
        for sid in sorted(notes):
            if "_UA-01_" not in sid:
                continue
            row = json.loads(annotations.read(notes[sid]))
            raw = row["Raw Data Info."]
            video = raw["raw_data_ID"]
            if (
                raw["situation_ID"] != "UA-01"
                or row["Source Data Info."]["source_data_ID"] != sid
                or Path(photos[sid]).parent.name != video
            ):
                excluded["internal_id_or_situation"] += 1
                continue
            boxes = defaultdict(list)
            width, height = raw["resolution"]
            for item in row["Learning Data Info."]["annotation"]:
                if item.get("class_id") not in {"WO-01", "WO-06"} or "box" not in item:
                    continue
                x, y, w, h = item["box"]
                if (
                    all(map(math.isfinite, (x, y, w, h)))
                    and x >= 0
                    and y >= 0
                    and w > 0
                    and h > 0
                    and x + w <= width + 1
                    and y + h <= height + 1
                ):
                    boxes[item["class_id"]].append(item["box"])
            counts.update(
                verified_source_and_scenario=1,
                with_worker_box=int(bool(boxes["WO-01"])),
                with_nonwear_box=int(bool(boxes["WO-06"])),
                with_both=int(bool(boxes["WO-01"] and boxes["WO-06"])),
            )
            by_video[video].append(
                (sid, raw["location_ID"], width, height, dict(boxes))
            )
        for video, frames in sorted(by_video.items()):
            sid, site, width, height, boxes = min(
                frames, key=lambda x: sha256(("p38/507/hook/" + x[0]).encode()).digest()
            )
            payload = images.read(photos[sid])
            with Image.open(BytesIO(payload)) as pic:
                pic.load()
                valid = pic.format == "JPEG" and pic.size == (width, height)
            matched.append(
                dict(
                    video=video,
                    sample_id=sid,
                    site=site,
                    boxes=boxes,
                    jpeg_and_dimensions_valid=valid,
                    sha256=sha256(payload).hexdigest(),
                )
            )
    catalog = read(root.parent / "catalog.json")
    train_parts = {
        e["name"]: bool(list(root.glob(e["filekey"] + "-*.zip")))
        for e in catalog
        if e["name"].startswith("TS_03_")
    }
    write(folder / "jpeg-ledger.json", matched)
    write(
        folder / "inventory.json",
        dict(
            provider_split="validation",
            source_archive_members_paired=len(photos),
            selected_by="UA-01 raw situation only",
            counts=counts,
            excluded=excluded,
            videos=len(by_video),
            sites=sorted({x["site"] for x in matched}),
            jpeg_checked=len(matched),
            jpeg_verified=sum(x["jpeg_and_dimensions_valid"] for x in matched),
            supplier_training_image_archives_local=train_parts,
            trained_images=0,
            source_zip_sha256_from_download_receipt=src_download["sha256"],
            full_source_member_crc_receipt=Path(str(src) + ".integrity.json").is_file(),
            jpeg_crc_verified_by_zipfile_read=len(matched),
            label_zip_sha256_from_ledger=lab_stamp["stamp"]["sha256"],
            scope="Supplier-declared scenario and object boxes. Visibility of actual fastening, paired fastened positives and K-water hazard ground truth not established.",
        ),
    )
    print("507 common", counts, len(matched), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subset", choices=["rope", "common-hook"], nargs="?", default="rope"
    )
    if parser.parse_args().subset == "rope":
        check()
    else:
        common_hook()
