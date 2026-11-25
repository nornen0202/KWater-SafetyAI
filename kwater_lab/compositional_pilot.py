"""Hide a common safety code, then compare text, shared cues and 0/1/5 support images."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .clip_probe import DESCRIPTIONS, FILES, REVISION, unit_rows
from .observation_pilot import load_cohort, ridge_projection, write_aggregate
from .prefix_pilot import filtered_training
from .rare_probe import RARE_CODES
from .train_head import file_hash


# Paired visible descriptions supply the zero-shot prior. Korean patterns supervise
# whether training observations mention a cue, not whether an unmentioned hazard is absent.
CUES = {
    "drop": ("an opening or an abrupt drop to a lower level", "a flat continuous work surface", r"개구부|단차|고소|가장자리|작업발판|고가|높은 곳"),
    "approach": ("a walking or working position next to the feature", "a separated area away from the walking route", r"통로|동선|보행|이동|접근|작업자|작업 위치"),
    "gap": ("an open edge or opening without a cover or barrier", "an opening protected by a cover or continuous guardrail", r"덮개.{0,10}(?:없|미|열)|난간.{0,10}(?:없|미|개방)|방호.{0,8}(?:없|미)|무방호|개방|열린"),
    "obstacle": ("a cable or loose material across the walking surface", "an unobstructed walking surface with materials stored aside", r"전선|케이블|호스|장애물|적치|돌출"),
    "slippery": ("a wet or slippery walking surface", "a dry clear walking surface", r"젖|물기|고인|미끄|기름|결빙"),
    "load": ("a raised suspended or precariously stacked load", "a stable load secured on the ground", r"인양|적재|매달|중량물|크레인|적층"),
    "path": ("an exposed working position beneath or beside the load path", "a physically separated working position outside the load path", r"하부|아래|하단|작업반경|이동 경로|주변.{0,6}작업"),
    "water": ("open water or incoming water beside the work surface", "a dry drained area away from open water", r"수면|수로|침수|수조|물가|유입수|저수|하천"),
    "enclosed": ("an enclosed vessel or chamber with a constricted entry", "a wide open unobstructed outdoor area", r"맨홀|탱크|수조 내부|저류조|밀폐|집수정|설비 내부"),
    "entry": ("workers or entry equipment at a narrow access opening", "a closed access opening isolated from workers", r"진입|출입|들어가|사다리|내부.{0,8}작업"),
    "weather": ("workers exposed to intense sunlight snow or ice", "workers in a sheltered temperate work area", r"강한 햇|직사광|눈이|눈길|얼음|결빙|땡볕|그늘"),
}
CLAUSES = {
    "fall_height": (("drop", "gap"),),
    "slip_trip": (("approach", "obstacle"), ("approach", "slippery")),
    "struck_by_lifting": (("load", "path"),),
    "drowning_inundation": (("water", "approach", "gap"),),
    "confined_space": (("enclosed", "entry"),),
    "thermal_weather": (("weather", "approach"),),
}
SEEDS = (11, 23, 37, 53, 71)


def encode_prompts(model_dir):
    names = ["text_model_quantized.onnx", "tokenizer.json"]
    for name, key in zip(names, ("onnx/text_model_quantized.onnx", "tokenizer.json"), strict=True):
        if file_hash(model_dir / name) != FILES[key]:
            raise ValueError(f"Unexpected CLIP asset {name}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(model_dir / names[0]), sess_options=opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / names[1]))
    descriptions = [DESCRIPTIONS[c] for c in CLAUSES] + [txt for pos, neg, _ in CUES.values() for txt in (pos, neg)]
    vectors = []
    for text in descriptions:
        ids = np.array([tokenizer.encode(f"A photo of an industrial worksite with {text}.").ids], dtype=np.int64)
        vectors.append(encoder.run(["text_embeds"], {"input_ids": ids})[0][0])
    matrix = unit_rows(np.stack(vectors))
    pairs = matrix[len(CLAUSES):].reshape(len(CUES), 2, -1)
    return matrix[:len(CLAUSES)], pairs[:, 0] - pairs[:, 1]


def standard(values, reference):
    return (values - reference.mean(axis=0)) / np.maximum(reference.std(axis=0), 1e-6)


def compose(attrs, code):
    indices = {name: i for i, name in enumerate(CUES)}
    return np.maximum.reduce([attrs[:, [indices[name] for name in clause]].min(axis=1) for clause in CLAUSES[code]])


def support_order(positives, groups, code, outer, seed):
    def key(value):
        return hashlib.sha256(f"kwater-support/{code}/{outer}/{seed}/{value}".encode()).digest()
    by_group = defaultdict(list)
    for sid in positives:
        by_group[groups[sid]].append(sid)
    return [min(by_group[g], key=key) for g in sorted(by_group, key=key)]


def binary_metrics(y, scores, alarms, normal):
    y = np.asarray(y, dtype=bool)
    scores, alarms, normal = np.asarray(scores), np.asarray(alarms, dtype=bool), np.asarray(normal, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.where(np.diff(scores[order]) != 0)[0], len(y) - 1]
    cumulative = np.cumsum(y[order])[ends]
    ap = float(np.sum(np.diff(np.r_[0, cumulative]) * cumulative / (ends + 1)) / y.sum()) if y.any() else None
    return {"positives": int(y.sum()), "negatives": int((~y).sum()), "average_precision": ap,
            "tp": int((alarms & y).sum()), "fp": int((alarms & ~y).sum()),
            "normal_photos": int(normal.sum()), "normal_alerts": int((alarms & normal).sum())}


def run(root, model_dir):
    data = load_cohort(root)
    keys = sorted(data.gold)
    ix = {s: i for i, s in enumerate(keys)}
    groups = {s: g for g, ids in data.buckets.items() for s in ids}
    images = np.stack([data.clip[s] for s in keys])
    code_vectors, cue_vectors = encode_prompts(model_dir)
    code_scores, cue_scores = images @ code_vectors.T, images @ cue_vectors.T
    observations = [" ".join(data.observations[s].values()) for s in keys]
    mentions = np.asarray([[bool(re.search(pattern, text)) for _, _, pattern in CUES.values()] for text in observations], dtype=float)
    report = {"scope": "Previously exposed 170 photos, six unverified prefixes. Binary retrieval and screening, not the official 11-code score.",
              "input_sha256": data.hashes, "CLIP_revision": REVISION, "support_seeds": SEEDS,
              "calibration": "strictly above the 90th percentile (higher order statistic) of outer-training target negatives",
              "support_unit": "one positive photo per distinct training prefix proxy, nested 1 and 5; no confirmed incidents",
              "by_code": {}, "mention_transfer_support": {}}
    for col, code in enumerate(CLAUSES):
        y = np.asarray([code in data.gold[s] for s in keys])
        normal = np.asarray([not data.gold[s] for s in keys])
        records = defaultdict(lambda: (np.empty(len(keys)), np.zeros(len(keys), dtype=bool)))
        report["mention_transfer_support"][code] = {}
        for prefix, valid in data.buckets.items():
            train = filtered_training(keys, valid, data.linked)
            negative = [ix[s] for s in train if code not in data.gold[s]]
            positive = [s for s in train if code in data.gold[s]]
            vi = [ix[s] for s in valid]
            # No target-positive photo or its text enters the transfer model or its calibration.
            y_bg = mentions[negative]
            query, loo = ridge_projection(images[negative], y_bg, images[vi])
            count = y_bg.sum(axis=0)
            active = (count >= 3) & (len(negative) - count >= 3)
            report["mention_transfer_support"][code][prefix] = {name: int(v) for name, v in zip(CUES, count, strict=True)}
            cue = standard(cue_scores, cue_scores[negative])
            learned_neg, learned_val = cue[negative].copy(), cue[vi].copy()
            learned_neg[:, active] = (learned_neg[:, active] + standard(loo, loo)[:, active]) / 2
            learned_val[:, active] = (learned_val[:, active] + standard(query, loo)[:, active]) / 2
            raw = {"code_text": (code_scores[negative, col], code_scores[vi, col]),
                   "composition": (compose(cue[negative], code), compose(cue[vi], code)),
                   "transfer": (compose(learned_neg, code), compose(learned_val, code))}
            adjusted = {name: (standard(n, n), standard(v, n)) for name, (n, v) in raw.items()}

            def record(shots, seed, name, pair):
                background, predicted = pair
                output, alerts = records[shots, seed, name]
                output[vi] = predicted
                alerts[vi] = predicted > np.quantile(background, 0.9, method="higher")

            for name, pair in adjusted.items():
                record(0, 0, name, pair)
            if code in RARE_CODES:
                if positive or y[vi].any():
                    raise ValueError("Expected zero rare positives in this diagnostic")
                continue
            for seed in SEEDS:
                picks = support_order(positive, groups, code, prefix, seed)
                if len(picks) < 5:
                    raise ValueError("Five separate positive support prefixes are required for this fixed comparison")
                for shots in (1, 5):
                    support = images[[ix[s] for s in picks[:shots]]]
                    nn = (images @ support.T).max(axis=1)
                    nn = standard(nn, nn[negative])
                    record(shots, seed, "nearest", (nn[negative], nn[vi]))
                    for name, (n, v) in adjusted.items():
                        record(shots, seed, name + "_with_support", ((n + nn[negative]) / 2, (v + nn[vi]) / 2))
        by_shot = {}
        for (shots, seed, name), (values, alerts) in records.items():
            by_shot.setdefault(str(shots), {}).setdefault(name, {})[str(seed)] = binary_metrics(y, values, alerts, normal)
        report["by_code"][code] = by_shot
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clip", type=Path, default=Path("G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32"))
    args = parser.parse_args()
    result = run(args.root, args.clip)
    write_aggregate(args.root / "results/compositional_pilot.json", result)
    print("Evaluated three pseudo-rare hazards and three negative-only rare hazards.")
