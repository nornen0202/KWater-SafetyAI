"""Sample real AI-Hub 71407 frames before seeing v3 predictions, grouped by site."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from .citation_review_panel import ROOT, freeze, read

SOURCE = Path("data/external/aihub/71407")
SEED = "kwater-s1-71407-v1"
TYPES = {"전도": "slip_trip", "협착": "caught_in_machine", "화재": "fire_explosion"}


def rank(value):
    return sha256(f"{SEED}/{value}".encode()).hexdigest()


def main():
    known = {int(x["site"]) for x in read(SOURCE / "verified-state-pairs.json")["pairs"]}
    for pair in read(Path("data/curation-side-20261127/71407-shortlist.json"))["pairs"]:
        known.update(int(x["site"]) for x in pair["members"])
    candidates = defaultdict(list)
    for category, code in TYPES.items():
        for pictures in sorted((SOURCE / "packages").glob(f"*-TS_5대사고유형_{category}_비정상_N-*.zip")):
            suffix = pictures.name.split("-TS_", 1)[1]
            annotations = next((SOURCE / "packages").glob("*-TL_" + suffix))
            with ZipFile(pictures) as images, ZipFile(annotations) as labels:
                by_stem = {PurePosixPath(n).stem: n for n in images.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
                by_clip = defaultdict(list)
                for name in labels.namelist():
                    stem = PurePosixPath(name).stem
                    if name.endswith(".json") and stem in by_stem:
                        by_clip[stem.rsplit("_", 1)[0]].append((name, by_stem[stem]))
                for clip, frames in by_clip.items():
                    member, photo = sorted(frames)[len(frames)//2]
                    annotation = json.loads(labels.read(member))
                    meta = annotation["Raw_Data_Info."]
                    site = int(meta["Location_ID"])
                    if site in known or meta["Raw_data_ID"] != clip or meta["Type_Description"] != category:
                        continue
                    if not re.fullmatch("N-[0-9]+", meta["Situation_ID"]):
                        continue
                    candidates[code, site].append(dict(clip=clip, site=site, provider_family=code,
                                                        archive=pictures.name, photo_member=photo))
    # Round-robin across site and provider family. At most two clips per site/family.
    selected = []
    for values in candidates.values():
        values.sort(key=lambda row: rank(row["clip"]))
    for depth in range(2):
        for key in sorted(candidates, key=lambda k: rank(str(k))):
            if len(candidates[key]) > depth:
                selected.append(candidates[key][depth])
    selected = selected[:120]
    if not selected or len({r["clip"] for r in selected}) != len(selected):
        raise ValueError("No independent clip set")
    directory = ROOT / "prospective-input"
    images_dir = directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        sid = rank(row["clip"])[:32]
        row["sample_id"] = sid
        with ZipFile(SOURCE / "packages" / row["archive"]) as z:
            image = z.read(row["photo_member"])
        row["sha256"] = sha256(image).hexdigest()
        row["image"] = f"images/{sid}{PurePosixPath(row['photo_member']).suffix.lower()}"
        path = directory / row["image"]
        if path.exists() and sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Previously frozen pixels changed")
        if not path.exists():
            path.write_bytes(image)
    if len({r["sha256"] for r in selected}) != len(selected):
        raise ValueError("Duplicate content across clips")
    receipt = dict(photos=len(selected), sites=len({r["site"] for r in selected}), excluded_prior_sites=len(known),
                   provider_families=dict(Counter(r["provider_family"] for r in selected)),
                   source="AI-Hub 71407 training partition", seed=SEED,
                   local_manifest_sha256=freeze(directory / "selection.json", selected))
    freeze(Path("results/citation-prospective-site-inputs.json"), receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
