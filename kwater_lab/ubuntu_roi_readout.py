"""Matched two-view, fixed-candidate Qwen diagnostic; predict before opening gold.

Run: .venv/bin/python -m kwater_lab.ubuntu_roi_readout predict
Then: .venv/bin/python -m kwater_lab.ubuntu_roi_readout score --run <printed-run>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment

ROOT: Final = Path(__file__).resolve().parents[1]
LOCAL: Final = ROOT / "data/ubuntu-roi-20261127"
CODES: Final = ("slip_trip", "fall_height", "struck_by_lifting")
ARMS: Final = ("global", "fixed", "box", "mask", "sham", "caption", "swapped")
SIDE: Final = 384
PREFIX: Final = '{"decision":"'


@dataclass(frozen=True, slots=True)
class ContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class Photo:
    sample_id: str
    image: str
    sha256: str
    prefix: str


@dataclass(frozen=True, slots=True)
class View:
    sample_id: str
    image_sha256: str
    fixed: str
    box: str
    mask: str
    sham: str
    caption: str
    detected: bool


@dataclass(frozen=True, slots=True)
class Prediction:
    sample_id: str
    arm: str
    code: str
    margin: float
    input_sha256: str
    grid: list[list[int]]
    visual_tokens: int
    input_tokens: int
    reused: bool
    seconds: float


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


class CodeMetrics(TypedDict):
    tp: int
    fp: int
    fn: int
    support: int
    precision: float
    recall: float
    f1: float


class Metrics(TypedDict):
    photos: int
    per_code: dict[str, CodeMetrics]
    fixed_3code_macro_f1: float
    active_code_macro_f1: float | None
    active_codes: int
    tp: int
    fp: int
    fn: int
    precision: float
    normal_photos: int
    normal_false_alarm_photos: int
    normal_false_alarm_rate: float | None
    positive_photo_coverage: float
    scored_photo_coverage: float


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def load_inputs(cohort: Path, index: Path) -> tuple[list[Photo], list[View]]:
    """Parse only the frozen, label-free image identities and independently built views."""
    photos = [Photo(**row) for row in json.loads(cohort.read_text())["rows"]]
    groups = Counter(photo.prefix for photo in photos)
    if len(photos) != 24 or len(groups) != 6 or set(groups.values()) != {4}:
        raise ContractError(
            "Expected frozen 24-photo, six-prefix, four-per-prefix design"
        )
    if (
        len({p.sample_id for p in photos}) != 24
        or len({p.sha256 for p in photos}) != 24
    ):
        raise ContractError("Duplicate sample ID or image content")
    fields = tuple(View.__dataclass_fields__)
    views = [
        View(**{key: row[key] for key in fields})
        for row in json.loads(index.read_text())["rows"]
    ]
    if {v.sample_id for v in views} != {p.sample_id for p in photos} or len(
        views
    ) != 24:
        raise ContractError("View coverage differs from the frozen cohort")
    mapped = {view.sample_id: view for view in views}
    for photo in photos:
        if digest(ROOT / "data/train" / photo.image) != photo.sha256:
            raise ContractError("Original image hash mismatch")
        if mapped[photo.sample_id].image_sha256 != photo.sha256:
            raise ContractError("ROI source image hash mismatch")
    return photos, [mapped[p.sample_id] for p in photos]


def caption_pairs(
    captions: list[str], tokenizer: Tokenizer
) -> list[tuple[int, str, str]]:
    """Derange distinct-image captions, minimizing token mismatch before truncation."""
    encoded = [
        tokenizer.encode(text.strip(), add_special_tokens=False) for text in captions
    ]
    if len(captions) < 2 or any(not tokens for tokens in encoded):
        raise ContractError(
            "Caption permutation requires at least two nonempty captions"
        )
    lengths = np.asarray([len(tokens) for tokens in encoded])
    cost = np.abs(lengths[:, None] - lengths[None, :]).astype(float)
    np.fill_diagonal(cost, 1e9)
    for i, caption in enumerate(captions):
        for j, donor in enumerate(captions):
            if caption == donor and i != j:
                cost[i, j] += 1e6
    _, donors = linear_sum_assignment(cost)
    pairs: list[tuple[int, str, str]] = []
    for i, donor in enumerate(donors.tolist()):
        if donor == i:
            raise ContractError("Caption permutation contains a self donor")
        size = int(min(lengths[i], lengths[donor]))
        true, swapped = "", ""
        while size > 0:
            true = tokenizer.decode(encoded[i][:size], skip_special_tokens=True)
            swapped = tokenizer.decode(encoded[donor][:size], skip_special_tokens=True)
            if len(tokenizer.encode(true, add_special_tokens=False)) == len(
                tokenizer.encode(swapped, add_special_tokens=False)
            ):
                break
            size -= 1
        if size == 0:
            raise ContractError("Unable to equalize caption token lengths")
        pairs.append((donor, true, swapped))
    return pairs


def check_grid(grid: list[list[int]], visual_tokens: int) -> None:
    """Fail closed when two-view geometry or merged visual token parity drifts."""
    expected = [[1, SIDE // 16, SIDE // 16]] * 2
    if grid != expected or visual_tokens != 2 * (SIDE // 32) ** 2:
        raise ContractError(
            f"Unexpected visual budget: grid={grid}, tokens={visual_tokens}"
        )


def load_pixels(path: Path) -> Image.Image:
    with Image.open(path) as image:
        ImageOps.exif_transpose(image, in_place=True)
        return image.convert("RGB").resize((SIDE, SIDE), Image.Resampling.LANCZOS)


def predict(args: argparse.Namespace) -> None:
    """Cache actual next-token margins; never open a training target or label file."""
    import torch
    from torch.version import cuda

    from .vlm_11code import load_qwen_model, load_qwen_processor, verify_user_text

    photos, views = load_inputs(args.cohort, args.index)
    processor = load_qwen_processor(args.base, SIDE * SIDE)
    captions = caption_pairs([v.caption for v in views], processor.tokenizer)
    system = args.prompt.read_text()
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("vlm_11code.py"),
        args.prompt,
    ]
    model_paths = sorted(
        p
        for folder in (args.base, args.adapter)
        for p in folder.rglob("*")
        if p.is_file()
    )
    identity = {
        "schema": "roi-conditional-verifier-v1",
        "cohort": digest(args.cohort),
        "views": digest(args.index),
        "train_source_sha256": json.loads(args.cohort.read_text())[
            "train_source_sha256"
        ],
        "files": {
            str(p.relative_to(ROOT)): digest(p) for p in model_paths + source_paths
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "Pillow")
        },
        "device": torch.cuda.get_device_name(0),
        "cuda": cuda,
        "side": SIDE,
        "codes": CODES,
        "arms": ARMS,
        "threshold": 0.0,
        "quantization": "NF4 double quant, bf16 compute",
        "caption_pairs": captions,
        "view_bytes": [
            {
                key: digest(Path(getattr(v, key)))
                for key in ("fixed", "box", "mask", "sham")
            }
            for v in views
        ],
    }
    fingerprint = stable_hash(identity)
    run = args.out / fingerprint[:20]
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "identity.json", identity)
    if (run / "seal.json").exists():
        verify_seal(run)
        print(f"sealed run already complete: {run}", flush=True)
        return
    tokenizer = processor.tokenizer
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    decision_ids = []
    for decision in ("accept", "reject"):
        tokens = tokenizer.encode(PREFIX + decision, add_special_tokens=False)
        if tokens[:-1] != prefix_tokens:
            raise ContractError(
                "Accept/reject must each extend JSON prefix by exactly one token"
            )
        decision_ids.append(tokens[-1])
    model = load_qwen_model(args.base, use_4bit=True, adapter_dir=args.adapter).eval()
    torch.manual_seed(20261127)
    records = []
    forwards = 0
    started = time.perf_counter()
    caption_lengths = []
    for number, (photo, view, pair) in enumerate(
        zip(photos, views, captions, strict=True)
    ):
        global_image = load_pixels(ROOT / "data/train" / photo.image)
        donor, true_caption, swapped_caption = pair
        paired_lengths = []
        for arm in ARMS:
            selected = "box" if arm in ("caption", "swapped") else arm
            second = (
                global_image
                if selected == "global"
                else load_pixels(Path(getattr(view, selected)))
            )
            caption = {"caption": true_caption, "swapped": swapped_caption}.get(arm, "")
            for code in CODES:
                user = verify_user_text(code)
                if caption:
                    user += "\n보조 이미지 설명(오류 가능): " + caption
                messages = [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": user},
                        ],
                    },
                ]
                text = (
                    processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    + PREFIX
                )
                batch = processor(
                    text=[text], images=[global_image, second], return_tensors="pt"
                )
                grid = batch["image_grid_thw"].tolist()
                visual_tokens = int(
                    (batch["input_ids"] == model.config.image_token_id).sum()
                )
                check_grid(grid, visual_tokens)
                if arm in ("caption", "swapped"):
                    paired_lengths.append(
                        (arm, code, int(batch["input_ids"].shape[-1]))
                    )
                tensor_hashes = {
                    key: hashlib.sha256(
                        tensor.contiguous().numpy().tobytes()
                    ).hexdigest()
                    for key, tensor in batch.items()
                }
                key = stable_hash((fingerprint, tensor_hashes, decision_ids))
                cache = run / "cache" / f"{key}.json"
                reused = cache.exists()
                seconds = 0.0
                if reused:
                    cached = json.loads(cache.read_text())
                    if cached["input_sha256"] != key or not math.isfinite(
                        cached["margin"]
                    ):
                        raise ContractError("Corrupt cached margin")
                    margin = float(cached["margin"])
                else:
                    if args.limit_forwards and forwards >= args.limit_forwards:
                        print(
                            json.dumps(
                                {
                                    "paused_run": str(run),
                                    "forwards": forwards,
                                    "seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                        return
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    with torch.inference_mode():
                        output = model(
                            **batch.to("cuda"), use_cache=False, logits_to_keep=1
                        )
                    logits = output.logits[0, -1].float()
                    margin = float(
                        (logits[decision_ids[0]] - logits[decision_ids[1]]).item()
                    )
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - before
                    del output, logits
                    write_json(
                        cache,
                        {"input_sha256": key, "margin": margin, "seconds": seconds},
                    )
                    forwards += 1
                records.append(
                    Prediction(
                        photo.sample_id,
                        arm,
                        code,
                        margin,
                        key,
                        grid,
                        visual_tokens,
                        int(batch["input_ids"].shape[-1]),
                        reused,
                        seconds,
                    )
                )
                print(
                    json.dumps(
                        {
                            "photo": number + 1,
                            "arm": arm,
                            "code": code,
                            "records": len(records),
                            "forwards_this_process": forwards,
                            "reused": reused,
                            "seconds": seconds,
                        }
                    ),
                    flush=True,
                )
        left = [length for arm, _, length in paired_lengths if arm == "caption"]
        right = [length for arm, _, length in paired_lengths if arm == "swapped"]
        caption_lengths.append(
            {
                "sample_id": photo.sample_id,
                "donor": photos[donor].sample_id,
                "true_input_tokens": left,
                "swapped_input_tokens": right,
                "matched": left == right,
                "identical_caption": true_caption == swapped_caption,
            }
        )
    payload = {
        "identity_sha256": fingerprint,
        "photos": [asdict(p) for p in photos],
        "predictions": [asdict(r) for r in records],
        "caption_parity": caption_lengths,
        "detected_photos": sum(v.detected for v in views),
        "elapsed_seconds": time.perf_counter() - started,
        "detected_sample_ids": [v.sample_id for v in views if v.detected],
        "forwards_this_process": forwards,
        "unique_inputs": len({r.input_sha256 for r in records}),
        "reused_records_this_process": sum(r.reused for r in records),
    }
    write_json(run / "predictions.json", payload)
    write_json(
        run / "seal.json",
        {
            "predictions_sha256": digest(run / "predictions.json"),
            "identity_sha256": digest(run / "identity.json"),
            "sealed_unix": time.time(),
        },
    )
    verify_seal(run)
    print(f"SEALED_RUN={run}", flush=True)


def verify_seal(run: Path):
    """Refuse scoring incomplete, edited, duplicate, or non-finite predictions."""
    seal = json.loads((run / "seal.json").read_text())
    if (
        digest(run / "predictions.json") != seal["predictions_sha256"]
        or digest(run / "identity.json") != seal["identity_sha256"]
    ):
        raise ContractError("Sealed input hashes changed")
    payload = json.loads((run / "predictions.json").read_text())
    identity = json.loads((run / "identity.json").read_text())
    if stable_hash(identity) != payload["identity_sha256"]:
        raise ContractError("Prediction identity changed")
    photos = [Photo(**row) for row in payload["photos"]]
    predictions = [Prediction(**row) for row in payload["predictions"]]
    expected = {
        (p.sample_id, arm, code) for p in photos for arm in ARMS for code in CODES
    }
    actual = {(p.sample_id, p.arm, p.code) for p in predictions}
    if len(photos) != 24 or actual != expected or len(predictions) != len(expected):
        raise ContractError(
            "Predictions must cover all 504 unique image/arm/code conditions"
        )
    for record in predictions:
        if not math.isfinite(record.margin):
            raise ContractError("Non-finite margin")
        check_grid(record.grid, record.visual_tokens)
    return payload, identity, seal


def classification_metrics(
    gold: list[frozenset[str]], predicted: list[frozenset[str]]
) -> Metrics:
    """Fixed three-code denominator; normal means no annotated hazard of any code."""
    if len(gold) != len(predicted) or not gold:
        raise ContractError("Metric inputs require equal nonempty coverage")
    classes: dict[str, CodeMetrics] = {}
    for code in CODES:
        tp = sum(code in g and code in p for g, p in zip(gold, predicted, strict=True))
        fp = sum(
            code not in g and code in p for g, p in zip(gold, predicted, strict=True)
        )
        fn = sum(
            code in g and code not in p for g, p in zip(gold, predicted, strict=True)
        )
        classes[code] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    normal = sum(not g for g in gold)
    alarms = sum(not g and bool(p) for g, p in zip(gold, predicted, strict=True))
    active = [r["f1"] for r in classes.values() if r["support"]]
    tp = sum(r["tp"] for r in classes.values())
    fp = sum(r["fp"] for r in classes.values())
    fn = sum(r["fn"] for r in classes.values())
    return {
        "photos": len(gold),
        "per_code": classes,
        "fixed_3code_macro_f1": sum(r["f1"] for r in classes.values()) / 3,
        "active_code_macro_f1": sum(active) / len(active) if active else None,
        "active_codes": len(active),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "normal_photos": normal,
        "normal_false_alarm_photos": alarms,
        "normal_false_alarm_rate": alarms / normal if normal else None,
        "positive_photo_coverage": sum(bool(p) for p in predicted) / len(predicted),
        "scored_photo_coverage": 1.0,
    }


def paired_summary(deltas: list[float]):
    positive, negative = sum(x > 0 for x in deltas), sum(x < 0 for x in deltas)
    n = positive + negative
    p = (
        min(
            1.0,
            2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n,
        )
        if n
        else 1.0
    )
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(deltas) - n,
        "two_sided_sign_p": p,
        "deltas": deltas,
    }


def score(args: argparse.Namespace) -> None:
    """Only this stage reads gold, after the complete predictions seal is checked."""
    payload, identity, seal = verify_seal(args.run)
    if digest(args.gold) != identity["train_source_sha256"]:
        raise ContractError("Gold bytes differ from the preselected cohort source")
    photos = [Photo(**row) for row in payload["photos"]]
    selected = {p.sample_id for p in photos}
    gold = {}
    with args.gold.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] in selected:
                if row["sample_id"] in gold:
                    raise ContractError("Duplicate gold sample ID")
                gold[row["sample_id"]] = frozenset(row["target"]["hazards"])
    if set(gold) != selected:
        raise ContractError("Gold coverage mismatch")
    predictions = {arm: {sid: set() for sid in selected} for arm in ARMS}
    for row in payload["predictions"]:
        if row["margin"] > 0.0:
            predictions[row["arm"]][row["sample_id"]].add(row["code"])

    def metrics(arm: str, subset: list[str]):
        return classification_metrics(
            [gold[s] for s in subset], [frozenset(predictions[arm][s]) for s in subset]
        )

    ids = [p.sample_id for p in photos]
    groups = sorted({p.prefix for p in photos})
    totals = {arm: metrics(arm, ids) for arm in ARMS}
    detected_ids = payload["detected_sample_ids"]
    detected_metrics = (
        {arm: metrics(arm, detected_ids) for arm in ARMS} if detected_ids else {}
    )
    contrasts = {}
    for arm, control in (
        ("fixed", "global"),
        ("box", "global"),
        ("box", "fixed"),
        ("mask", "box"),
        ("mask", "sham"),
        ("caption", "box"),
        ("caption", "swapped"),
    ):
        group_deltas, leave_one_out = [], []
        for group in groups:
            same = [p.sample_id for p in photos if p.prefix == group]
            other = [p.sample_id for p in photos if p.prefix != group]
            group_deltas.append(
                metrics(arm, same)["fixed_3code_macro_f1"]
                - metrics(control, same)["fixed_3code_macro_f1"]
            )
            leave_one_out.append(
                metrics(arm, other)["fixed_3code_macro_f1"]
                - metrics(control, other)["fixed_3code_macro_f1"]
            )
        contrasts[f"{arm}_minus_{control}"] = {
            "macro_f1_delta": totals[arm]["fixed_3code_macro_f1"]
            - totals[control]["fixed_3code_macro_f1"],
            "six_prefix_sign": paired_summary(group_deltas),
            "leave_one_prefix_out_deltas": leave_one_out,
        }
    report = {
        "scope": "Development-only fixed-candidate conditional verifier; official adapter saw training data; not end-to-end or independent performance.",
        "threshold": "accept iff next-token accept logit minus reject logit > 0; no positive cap",
        "candidate_codes": CODES,
        "photos": 24,
        "prefix_groups": 6,
        "prefix_independence_verified": False,
        "arms": totals,
        "detected_only_arms": detected_metrics,
        "contrasts": contrasts,
        "normal_definition": "No annotated hazard across all 11 codes",
        "visual_parity": {
            "pass": True,
            "grid": [[1, 24, 24], [1, 24, 24]],
            "tokens": 288,
        },
        "caption_parity": {
            "matched_photos": sum(p["matched"] for p in payload["caption_parity"]),
            "identical_caption_pairs": sum(
                p["identical_caption"] for p in payload["caption_parity"]
            ),
        },
        "detected_photos": payload["detected_photos"],
        "unique_inputs": payload["unique_inputs"],
        "prediction_conditions": 504,
        "prediction_seal": seal,
        "gold_sha256": digest(args.gold),
        "input_fingerprint": payload["identity_sha256"],
        "source_and_model_hashes": identity["files"],
        "elapsed_seconds_final_process": payload["elapsed_seconds"],
        "scored_unix": time.time(),
    }
    write_json(args.report, report)
    print(f"AGGREGATE_REPORT={args.report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--cohort", type=Path, default=LOCAL / "cohort.json")
    prediction.add_argument("--index", type=Path, default=LOCAL / "views/index.json")
    prediction.add_argument(
        "--base", type=Path, default=ROOT / "data/baseline/base_model"
    )
    prediction.add_argument(
        "--adapter", type=Path, default=ROOT / "data/baseline/adapter"
    )
    prediction.add_argument(
        "--prompt", type=Path, default=ROOT / "kwater_lab/system_prompt_11code.txt"
    )
    prediction.add_argument("--out", type=Path, default=LOCAL / "readout")
    prediction.add_argument("--limit-forwards", type=int, default=0)
    prediction.set_defaults(action=predict)
    scoring = sub.add_parser("score")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--gold", type=Path, default=ROOT / "data/train/train.jsonl")
    scoring.add_argument(
        "--report", type=Path, default=LOCAL / "readout/aggregate.json"
    )
    scoring.set_defaults(action=score)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
