"""Evaluate official-only risk heads on provider-labelled pairs from distinct videos."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .aihub_ablation import AUDIT, EXPOSURE, FEATURES, REFERENCE, read
from .clip_probe import ImageEncoder, REVISION
from .depth_pilot import CACHE, EXPORT, MODEL, MODES, TARGETS, DepthEncoder, unrelated
from .metrics import CODES
from .state_pairs import LOCAL, save, source_image
from .train_head import decisions, file_hash, fit, probabilities
from .weight_pilot import pilot_data


CLIP = Path("G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32")
DEPTH_RESULT = Path("results/depth_pilot.json")
PAIR_RESULT = Path("results/aihub_state_pairs.json")


def external_features(pairs):
    rows = [pair[state] for pair in pairs for state in ("normal", "abnormal")]
    photos = []
    for row in rows:
        raw = source_image(row)
        if sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("State-pair image changed since selection")
        photos.append(raw)
    images, maps = [], []
    clip = ImageEncoder(CLIP, verify_vision=True)
    for content in photos:
        with Image.open(BytesIO(content)) as image:
            images.append(clip.encode(image, "full_frame"))
    del clip
    depth = DepthEncoder(MODEL)
    for i, content in enumerate(photos, 1):
        with Image.open(BytesIO(content)) as image:
            maps.append(depth.encode(image))
        if i % 8 == 0 or i == len(photos):
            print(f"Paired images {i}/{len(photos)}", flush=True)
    return np.stack(images), np.stack(maps)


def targeted_scores(pairs, scores, prediction, target):
    subset = [i for i, pair in enumerate(pairs)
              if (TARGETS[0] if pair["scenario"] <= 10 else TARGETS[1]) == target]
    margins = []
    true_pos = false_pos = 0
    for i in subset:
        normal, abnormal = 2 * i, 2 * i + 1
        j = CODES.index(target)
        margins.append(float(scores[abnormal, j] - scores[normal, j]))
        false_pos += target in prediction[normal]
        true_pos += target in prediction[abnormal]
    return {"pairs": len(subset), "sites": len({pairs[i]["site"] for i in subset}),
            "normal_images": len(subset), "abnormal_images": len(subset),
            "tp": true_pos, "fp": false_pos, "fn": len(subset) - true_pos, "tn": len(subset) - false_pos,
            "abnormal_scored_higher": sum(m > 0 for m in margins),
            "equal_scores": sum(m == 0 for m in margins),
            "abnormal_scored_lower": sum(m < 0 for m in margins),
            "mean_abnormal_minus_normal": float(np.mean(margins)) if margins else None,
            "median_abnormal_minus_normal": float(np.median(margins)) if margins else None}


def run():
    selection = read(PAIR_RESULT)
    digest = file_hash(LOCAL)
    if digest != selection["local_pair_selection_sha256"]:
        raise ValueError("Local pair selection differs from the frozen manifest")
    pairs = read(LOCAL)["pairs"]
    if not pairs or len(pairs) != selection["selected_pairs"]:
        raise ValueError("External evaluation requires the previously selected pairs")
    buckets, rgb, gold, linked = pilot_data(AUDIT, EXPOSURE, FEATURES, read(REFERENCE))
    official = read(DEPTH_RESULT)
    if file_hash(CACHE) != official["feature_cache_sha256"]:
        raise ValueError("Official depth descriptor cache changed")
    keys = sorted(gold)
    with np.load(CACHE, allow_pickle=False) as cache:
        ids = cache["ids"].tolist()
        if ids != keys or len(ids) != official["photos"]:
            raise ValueError("Official depth descriptors do not cover the training set")
        maps = cache["vectors"]
    external_rgb, external_maps = external_features(pairs)
    size = 2 * len(pairs)
    official_rgb = np.stack([rgb[k] for k in keys])
    labels = np.array([[c in gold[k] for c in CODES] for k in keys], dtype=float)
    groups = {k: g for g, members in buckets.items() for k in members}
    training_wrong = unrelated(keys, keys, groups, linked, "external-state-pair|training")
    index = {k: i for i, k in enumerate(keys)}
    train_donor_rows = [index[training_wrong[k]] for k in keys]
    # All test images were screened against the training set for near duplicates.
    # Draw unique training donors in a frozen order independent of the external labels.
    pool = sorted(keys, key=lambda k: sha256(("external-state-pair|test|" + k).encode()).digest())
    external_ids = [p[state]["sha256"] for p in pairs for state in ("normal", "abnormal")]
    if size > len(pool) or len(set(external_ids)) != size:
        raise ValueError("Cannot allocate distinct wrong-depth maps")
    donor = dict(zip(sorted(external_ids), pool))
    test_donor_rows = [index[donor[k]] for k in external_ids]
    metrics = {}
    for method in MODES:
        train, test = official_rgb, external_rgb
        if method != "rgb":
            ix = {"brightness": 0, "edge": 1, "depth": 2, "wrong_depth": 2}[method]
            train_map, test_map = ((maps[train_donor_rows], maps[test_donor_rows])
                                   if method == "wrong_depth" else (maps, external_maps))
            train = np.column_stack((train, train_map[:, ix]))
            test = np.column_stack((test, test_map[:, ix]))
        scores = probabilities(fit(train, labels, .01, 250), test)
        pred = decisions(range(size), scores, .5)
        metrics[method] = {t: targeted_scores(pairs, scores, pred, t) for t in TARGETS}
    return {"scope": "External contrast of provider-annotated safety conditions in selected frames. "
                     "Only the target of each pair is scored. Other hazards are unlabelled.",
            "training": {"official_photos": len(keys), "external_photos": 0, **official["training"]},
            "clip_revision": REVISION, "depth_export_revision": EXPORT,
            "official_feature_cache_sha256": official["official_rgb_feature_sha256"],
            "official_depth_cache_sha256": official["feature_cache_sha256"],
            "selection_sha256": digest, "pairs": len(pairs), "photos": size,
            "source_videos": selection["unique_videos"], "sites": selection["unique_selected_sites"],
            "wrong_depth_donors": {"training": len(set(train_donor_rows)), "external_test": len(set(test_donor_rows)),
                                   "source": "official training photos"},
            "metrics": metrics}


if __name__ == "__main__":
    result = run()
    save(Path("results/depth_state_pairs.json"), result)
    print("Saved results/depth_state_pairs.json", flush=True)
