"""Compare three local AI-Hub sources on the registered, previously exposed cohort."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from .clip_probe import ImageEncoder, REVISION, unit_rows
from .metrics import CODES, evaluate
from .prefix_pilot import filtered_training
from .train_head import decisions, file_hash, fit, probabilities, save_atomic
from .weight_pilot import pilot_data


ROOT = Path("data/external/aihub")
AUDIT = Path("results/train_audit_full.json")
EXPOSURE = Path("data/prefix-pilot/exposure.json")
FEATURES = Path("data/prefix-pilot/features.npz")
REFERENCE = Path("results/prefix_pilot_fpr20.json")
SOURCES = ("507", "71407", "71793")
PROXIES = {"추락": "fall_height", "전도": "slip_trip", "협착": "caught_in_machine",
           "낙하": "struck_by_lifting", "화재": "fire_explosion"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def order(key):
    return sha256(("kwater-aihub-ablation-v1|" + key).encode()).digest()


@dataclass(frozen=True)
class Frame:
    photo: str
    site: str
    group: str
    code: str = ""
    positive: bool = False


def paired(images, labels):
    def members(archive, suffix):
        rows = [n for n in archive.namelist() if n.lower().endswith(suffix)]
        mapping = {PurePosixPath(n).stem: n for n in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate archive member stem")
        return mapping

    photos = members(images, ".jpg")
    annotations = members(labels, ".json")
    if photos.keys() != annotations.keys():
        raise ValueError("Image and annotation ID sets differ")
    return photos, annotations


def spread(frames):
    """Cycle through sites; draw from each original source group at most once later."""
    sites = defaultdict(list)
    for frame in sorted(frames, key=lambda f: order(f.photo)):
        sites[frame.site].append(frame)
    names = sorted(sites, key=order)
    for offset in range(max(len(rows) for rows in sites.values())):
        for site in names:
            if offset < len(sites[site]):
                yield sites[site][offset]


def source_frames(source, photos, labels, annotations):
    rows = []
    for key, name in photos.items():
        parts = PurePosixPath(name).parts
        if source == "71407" and parts[2] == "5대사고유형":
            annotation = json.loads(labels.read(annotations[key]))
            raw = annotation["Raw_Data_Info."]
            if annotation["Source_Data_Info."]["Source_Data"] != key or parts[4] not in {"정상", "비정상"}:
                raise ValueError("Construction sample metadata mismatch")
            rows.append(Frame(key, str(raw["Location_ID"]), str(raw["Raw_data_ID"]),
                              PROXIES[parts[3]], parts[4] == "비정상"))
        elif source == "507":
            group = parts[-2]
            rows.append(Frame(key, group.split("_")[1], group))
        elif source == "71793" and "지역 라벨링" in parts[2]:
            rows.append(Frame(key, parts[4], "/".join(parts[3:-1])))
    return rows


def select_features(source, img, lab, encoder, official_hashes):
    photos, annotations = paired(img, lab)
    frames = source_frames(source, photos, lab, annotations)
    accepted, vectors, hashes, groups = [], [], [], set()
    rejected = Counter()
    for frame in spread(frames):
        if frame.group in groups:
            continue
        if source != "71407":
            meta = json.loads(lab.read(annotations[frame.photo]))
            if source == "71793" and not any(PurePosixPath(i["file_name"]).stem == frame.photo for i in meta["images"]):
                rejected["filename_mismatch_checked"] += 1
                continue
            if source == "507":
                raw = meta["Raw Data Info."]
                if ((raw["raw_data_ID"], raw["location_ID"]) != (frame.group, frame.site)
                        or meta["Source Data Info."]["source_data_ID"] != frame.photo
                        or not raw["situation_ID"].startswith("WS-")):
                    raise ValueError("High-work source ID or metadata mismatch")
        with Image.open(BytesIO(img.read(photos[frame.photo]))) as original:
            picture = ImageOps.exif_transpose(original).convert("RGB")
        grid = np.asarray(picture.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        dhash = int.from_bytes(np.packbits(grid[:, :-1] > grid[:, 1:]).tobytes(), "big")
        if any((h ^ dhash).bit_count() <= 4 for h in official_hashes):
            rejected["near_official"] += 1
            continue
        if any((h ^ dhash).bit_count() <= 4 for h in hashes):
            rejected["near_same_source"] += 1
            continue
        groups.add(frame.group)
        hashes.append(dhash)
        accepted.append(frame)
        vectors.append(encoder.encode(picture, "full_frame"))
        if source != "71407" and len(accepted) == 12:
            break
    if len(vectors) < 4:
        raise ValueError("Each source needs at least four nonduplicate groups")
    info = {"paired_photos": len(photos), "eligible_photos": len(frames),
            "eligible_sites": len({f.site for f in frames}), "eligible_original_groups": len({f.group for f in frames}),
            "encoded_groups": len(accepted), "rejections_during_selection": dict(rejected)}
    return accepted, np.stack(vectors), info


def packages(source):
    folder = ROOT / source
    if source == "507":
        return [next((folder / "packages").glob(code + "-*.zip")) for code in ("68016", "68029")]
    return [folder / "sample.zip"]


def source_cache(source, official_hashes, model_path, encoders):
    paths = packages(source)
    receipts = [read(p.with_name(p.name + ".json")) for p in paths]
    if any(p.stat().st_size != rec["bytes"] for p, rec in zip(paths, receipts)):
        raise ValueError("Archive size differs from the verified receipt")
    provenance = {"source_sha256": [r["sha256"] for r in receipts], "vision_revision": REVISION,
                  "official_feature_sha256": file_hash(FEATURES), "method": file_hash(Path(__file__))}
    cache = ROOT / source / "pilot-ablation-features.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved["provenance"])) == provenance:
                rows = [Frame(**row) for row in json.loads(str(saved["rows"]))]
                features = saved["features"].copy()
                if features.shape != (len(rows), 512) or not np.isfinite(features).all():
                    raise ValueError("Invalid cached external features")
                return rows, features, json.loads(str(saved["info"]))
    if not encoders:
        encoders.append(ImageEncoder(model_path, verify_vision=True))
    with ZipFile(paths[0]) as img:
        if source == "507":
            with ZipFile(paths[1]) as lab:
                rows, features, info = select_features(source, img, lab, encoders[0], official_hashes)
            val = next((ROOT / source / "packages").glob("68035-*.zip"))
            with ZipFile(val) as other:
                train_groups, val_groups = [{PurePosixPath(n).parent.name for n in z.namelist() if n.lower().endswith(".jpg")}
                                            for z in (img, other)]
            info["provider_rope_video_overlap"] = {"train": len(train_groups), "validation": len(val_groups),
                                                   "shared": len(train_groups & val_groups)}
        else:
            rows, features, info = select_features(source, img, img, encoders[0], official_hashes)
    info["source_sha256"] = provenance["source_sha256"]
    info["source_files"] = [p.name for p in paths]
    payload = {"provenance": json.dumps(provenance), "rows": json.dumps([vars(f) for f in rows]),
               "features": features, "info": json.dumps(info)}
    save_atomic(cache, lambda stream: np.savez_compressed(stream, **payload))
    return rows, features, info


def centers(vectors):
    points = unit_rows(vectors)
    selected = [0]
    while len(selected) < 4:
        similarities = (points @ points[selected].T).max(axis=1)
        similarities[selected] = np.inf
        selected.append(int(np.argmin(similarities)))
    result = points[selected].copy()
    for _ in range(10):
        assign = (points @ result.T).argmax(axis=1)
        for k in range(4):
            if np.any(assign == k):
                result[k] = unit_rows(points[assign == k].mean(axis=0, keepdims=True))[0]
    return result


def official_refs(train, groups, hashes, count):
    candidates = [Frame(s, groups[s], s) for s in train]
    result = []
    for frame in spread(candidates):
        if all((hashes[frame.photo] ^ hashes[t]).bit_count() > 4 for t in result):
            result.append(frame.photo)
        if len(result) == count:
            return result
    raise ValueError("Insufficient unique official reference images")


def balanced_weak(rows, vectors):
    batches = {}
    for code in PROXIES.values():
        positive, negative = [[i for i, f in enumerate(rows) if f.code == code and f.positive == value]
                              for value in (True, False)]
        n = min(len(positive), len(negative))
        if n:
            batches[code] = (vectors[positive[:n] + negative[:n]], np.repeat([1., 0.], n))
    return batches


def weak_scores(x, y, test, base, batches, fold):
    scores = {k: base.copy() for k in ("official_resample", "71407_weak_labels", "71407_shuffled")}
    applied = []
    for code, (extra, target) in batches.items():
        col = CODES.index(code)
        orig = y[:, col]
        if len(np.unique(orig)) != 2:
            continue
        n = len(target) // 2
        rng = np.random.default_rng(int.from_bytes(order(fold + code)[:8], "big"))
        ids = np.concatenate([rng.choice(np.flatnonzero(orig == state), n, replace=True) for state in (1, 0)])
        assignments = {"official_resample": (x[ids], target), "71407_weak_labels": (extra, target),
                       "71407_shuffled": (extra, np.random.default_rng(42).permutation(target))}
        for name, (added, labels) in assignments.items():
            truth = np.zeros((len(y) + len(labels), len(CODES)))
            truth[:, col] = np.concatenate([orig, labels])
            head = fit(np.vstack([x, added]), truth, .01, 250)
            scores[name][:, col] = probabilities(head, test)[:, col]
        applied.append(code)
    return scores, applied


def run(model):
    ref = read(REFERENCE)
    folds, features, gold, linked = pilot_data(AUDIT, EXPOSURE, FEATURES, ref)
    inventory = read(AUDIT)["inventory"]
    hashes = {s: int(inventory[s]["dhash64"], 16) for s in gold}
    encoders, sources = [], {}
    for source in SOURCES:
        print("Preparing source", source, flush=True)
        sources[source] = source_cache(source, list(hashes.values()), model, encoders)
    count = min(12, *(len(r[0]) for r in sources.values()))
    extra_centers = {key: centers(source[1][:count]) for key, source in sources.items()}
    batches = balanced_weak(*sources["71407"][:2])
    groups = {s: f for f, keys in folds.items() for s in keys}
    predictions, trained = defaultdict(dict), Counter()
    for fold, valid in folds.items():
        train = filtered_training(gold, valid, linked)
        x, test = [np.stack([features[s] for s in keys]) for keys in (train, valid)]
        y = np.array([[c in gold[s] for c in CODES] for s in train], dtype=float)
        base = probabilities(fit(x, y, .01, 250), test)
        scores = {"official_only": base}
        ids = official_refs(train, groups, hashes, count)
        for key, bank in {"official_references": centers(np.stack([features[s] for s in ids])),
                          **extra_centers}.items():
            augmented = np.column_stack([x, x @ bank.T])
            probe = np.column_stack([test, test @ bank.T])
            scores[key] = probabilities(fit(augmented, y, .01, 250), probe)
        weak, applied = weak_scores(x, y, test, base, batches, fold)
        scores.update(weak)
        trained.update(applied)
        for key, score in scores.items():
            predictions[key].update(decisions(valid, score, .5))
    metrics = {key: evaluate(gold, pred, groups) for key, pred in predictions.items()}
    for result in metrics.values():
        codes = list(result["per_code"].values())
        result["macro_f1_fixed11"] = sum(r["f1"] or 0 for r in codes) / len(CODES)
        result["unsupported_code_false_alarms"] = sum(r["fp"] for r in codes if not r["support"])
    for key, (rows, _, info) in sources.items():
        info["reference_groups"] = count
        info["reference_sites"] = len({r.site for r in rows[:count]})
    return {"scope": "exploratory 170 exposed photos; six ID prefixes unverified as independent incidents",
            "protocol": "docs/aihub-ablation-protocol.md", "official_feature_sha256": file_hash(FEATURES),
            "model_revision": REVISION, "source_diagnostics": {k: v[2] for k, v in sources.items()},
            "reference_budget": {"pictures": count, "centers": 4, "features": 516, "kmeans_iterations": 10},
            "training": {"l2": .01, "steps": 250, "threshold": .5, "max_codes": 3},
            "weak_label_support": {k: {"added_positive": len(v[1])//2, "added_negative": len(v[1])//2,
                                       "trained_outer_folds": trained[k]} for k, v in batches.items()},
            "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32"))
    args = parser.parse_args()
    result = run(args.model)
    target = Path("results/aihub_ablation.json")
    save_atomic(target, lambda stream: stream.write((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()))
    print("Saved", target, flush=True)
