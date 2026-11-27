"""Frozen-base Qwen3-VL spatial-token probe; not an end-to-end submission model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable
import uuid

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as functional

from .metrics import CODES
from .vlm_11code import load_qwen_processor


GRID_SIDE = 4
ALLOWED_FOLDS = (1, 2, 3)
CACHE_ALGORITHM = "qwen3vl-visual-4x4-adaptive-v3-effective-processor"


@dataclass(frozen=True, slots=True)
class Row:
    sample_id: str
    image: str
    sha256: str
    group_id: str
    fold: int
    codes: frozenset[str]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp, **arrays)
        temporary = Path(tmp.name)
    temporary.replace(path)


def load_rows(manifest_path: Path, summary_path: Path) -> list[Row]:
    """Load only the canonical development rows after its recorded hash checks."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual = digest(manifest_path)
    if summary.get("dev_manifest_sha256") != actual:
        raise ValueError("development manifest SHA-256 differs from clean_split_v1_summary")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected a non-empty development manifest list")
    rows: list[Row] = []
    for item in raw:
        if item.get("split") != "dev":
            raise ValueError("sealed or non-development row supplied to spatial probe")
        fold = int(item["dev_fold"])
        if fold not in ALLOWED_FOLDS:
            continue
        codes = item.get("codes")
        if not isinstance(codes, list) or len(codes) != len(set(codes)):
            raise ValueError(f"invalid code list for {item.get('sample_id')}")
        rows.append(Row(str(item["sample_id"]), str(item["image"]), str(item["sha256"]),
                        str(item["group_id"]), fold, frozenset(map(str, codes))))
    if not rows:
        raise ValueError("no folds 1--3 remain after Fold0/Fold4 exclusion")
    return rows


def encoder_fingerprint(base_model: Path, max_pixels: int, precision: str) -> str:
    """Pin the base visual encoder and processor files without loading an adapter."""
    names = ("config.json", "preprocessor_config.json", "model.safetensors.index.json")
    missing = [name for name in names if not (base_model / name).is_file()]
    if missing:
        raise ValueError(f"base model lacks required files: {', '.join(missing)}")
    payload = {name: digest(base_model / name) for name in names}
    visual_shard = base_model / "model-00002-of-00002.safetensors"
    expected = "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6"
    if not visual_shard.is_file() or digest(visual_shard) != expected:
        raise ValueError("official Qwen3-VL visual-weight shard identity check failed")
    processor = load_qwen_processor(base_model, max_pixels)
    effective_processor = json.loads(processor.image_processor.to_json_string())
    payload.update({"effective_image_processor": effective_processor,
                    "official_visual_shard_sha256": expected, "max_pixels": max_pixels,
                    "precision": precision, "algorithm": CACHE_ALGORITHM})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_path(cache_dir: Path, source_sha256: str, encoder_id: str) -> Path:
    return cache_dir / f"{source_sha256}-{encoder_id[:16]}.npz"


def read_cached(row: Row, cache_dir: Path, encoder_id: str) -> np.ndarray | None:
    """Return a label-blind cached grid only when source and encoder identities match."""
    path = _cache_path(cache_dir, row.sha256, encoder_id)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if archive["source_sha256"].item() != row.sha256 or archive["encoder_sha256"].item() != encoder_id:
            return None
        features = archive["features"]
        if features.dtype != np.float32 or features.ndim != 2 or features.shape[0] != GRID_SIDE * GRID_SIDE:
            return None
        if not np.isfinite(features).all():
            return None
        return features


def _valid_cached(source_sha256: str, cache_dir: Path, encoder_id: str) -> bool:
    path = _cache_path(cache_dir, source_sha256, encoder_id)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            features = archive["features"]
            valid = (archive["source_sha256"].item() == source_sha256 and archive["encoder_sha256"].item() == encoder_id
                     and features.dtype == np.float32 and features.ndim == 2 and features.shape[0] == GRID_SIDE * GRID_SIDE
                     and np.isfinite(features).all())
    except (KeyError, OSError, ValueError):
        valid = False
    if not valid:
        path.replace(path.with_name(f"{path.stem}.corrupt-{uuid.uuid4().hex}.npz"))
    return valid


def _grid_tokens(tokens: torch.Tensor, grid_thw: torch.Tensor, merge_size: int) -> np.ndarray:
    height, width = (int(grid_thw[0, 1]), int(grid_thw[0, 2]))
    height //= merge_size
    width //= merge_size
    expected = height * width
    if tokens.ndim != 2 or tokens.shape[0] != expected:
        raise ValueError(f"unexpected visual token shape {tuple(tokens.shape)} for merged grid {height}x{width}")
    image = tokens.reshape(height, width, tokens.shape[1]).permute(2, 0, 1).unsqueeze(0)
    pooled = functional.adaptive_avg_pool2d(image.float(), (GRID_SIDE, GRID_SIDE))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(GRID_SIDE * GRID_SIDE, -1).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def extract_images(image_paths: Iterable[Path], cache_dir: Path, base_model: Path, device_name: str,
                   max_pixels: int) -> tuple[int, int, str]:
    """Label-blind content-addressed extraction from image bytes and official base weights only."""
    precision = "bf16" if device_name.startswith("cuda") else "fp32"
    encoder_id = encoder_fingerprint(base_model, max_pixels, precision)
    sources = [(path, digest(path)) for path in image_paths]
    missing = [(path, source_sha256) for path, source_sha256 in sources
               if not _valid_cached(source_sha256, cache_dir, encoder_id)]
    if not missing:
        return 0, len(sources), encoder_id
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    device = torch.device(device_name)
    config = Qwen3VLConfig.from_pretrained(str(base_model), local_files_only=True)
    visual = Qwen3VLVisionModel(config.vision_config)
    with safe_open(base_model / "model-00002-of-00002.safetensors", framework="pt", device="cpu") as shard:
        state = {key.removeprefix("model.visual."): shard.get_tensor(key) for key in shard.keys()
                 if key.startswith("model.visual.")}
    visual.load_state_dict(state, strict=True)
    visual = visual.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    processor = load_qwen_processor(base_model, max_pixels)
    merge_size = int(config.vision_config.spatial_merge_size)
    for image_path, source_sha256 in missing:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            encoded = processor.image_processor(images=image, return_tensors="pt")
        pixels = encoded["pixel_values"].to(device)
        grid = encoded["image_grid_thw"].to(device)
        output = visual(pixels, grid_thw=grid)
        tokens = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        features = _grid_tokens(tokens, grid.cpu(), merge_size)
        _atomic_npz(_cache_path(cache_dir, source_sha256, encoder_id), features=features,
                    source_sha256=np.asarray(source_sha256), encoder_sha256=np.asarray(encoder_id))
    return len(missing), len(sources), encoder_id


def extract_missing(rows: Iterable[Row], image_root: Path, cache_dir: Path, base_model: Path,
                    device_name: str, max_pixels: int) -> tuple[int, str]:
    """Extract development rows after their manifest has passed the supervised boundary."""
    rows = list(rows)
    paths = [image_root / row.image for row in rows]
    for row, path in zip(rows, paths, strict=True):
        if digest(path) != row.sha256:
            raise ValueError(f"image hash differs from manifest: {row.sample_id}")
    extracted, _, encoder_id = extract_images(paths, cache_dir, base_model, device_name, max_pixels)
    return extracted, encoder_id


def _sigmoid(values: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(values)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return float((precision * ranked).sum() / positives)


def _fit_scores(train_features: np.ndarray, train_labels: np.ndarray, valid_features: np.ndarray,
                *, regional: bool, top_k: int, seed: int, steps: int, device_name: str) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device(device_name)
    train = torch.from_numpy(train_features).to(device)
    valid = torch.from_numpy(valid_features).to(device)
    labels = torch.from_numpy(train_labels).to(device)
    dimension, code_count = train.shape[-1], labels.shape[1]
    weight = torch.nn.Parameter(train.new_zeros(dimension, code_count))
    bias = torch.nn.Parameter(train.new_zeros(code_count))
    optimizer = torch.optim.AdamW((weight, bias), lr=0.03, weight_decay=0.02)
    for _ in range(steps):
        token_logits = train @ weight + bias
        logits = token_logits.topk(top_k, dim=1).values.mean(dim=1) if regional else token_logits.mean(dim=1)
        loss = functional.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    token_logits = valid @ weight + bias
    scores = token_logits.topk(top_k, dim=1).values.mean(dim=1) if regional else token_logits.mean(dim=1)
    return scores.detach().cpu().numpy()


def evaluate_cached(rows: list[Row], cache_dir: Path, encoder_id: str, *, top_k: int, seed: int,
                    steps: int, device_name: str = "cpu") -> dict[str, object]:
    """Group-disjoint 1--3-fold comparison of equal-parameter global and regional heads."""
    grids = [read_cached(row, cache_dir, encoder_id) for row in rows]
    if any(grid is None for grid in grids):
        raise ValueError("all selected development images require a matching feature cache")
    features = np.stack([grid for grid in grids if grid is not None])
    codes = list(CODES)
    labels = np.asarray([[code in row.codes for code in codes] for row in rows], dtype=np.float32)
    fold_results: dict[str, object] = {}
    for fold in ALLOWED_FOLDS:
        valid = np.asarray([row.fold == fold for row in rows])
        train = ~valid
        if set(row.group_id for row, use in zip(rows, train, strict=True) if use) & set(row.group_id for row, use in zip(rows, valid, strict=True) if use):
            raise ValueError(f"incident group crosses fit/validation for fold {fold}")
        global_scores = _fit_scores(features[train], labels[train], features[valid], regional=False,
                                    top_k=top_k, seed=seed + fold, steps=steps, device_name=device_name)
        regional_scores = _fit_scores(features[train], labels[train], features[valid], regional=True,
                                      top_k=top_k, seed=seed + fold, steps=steps, device_name=device_name)
        valid_labels = labels[valid].astype(bool)
        usable = np.asarray([(valid_labels[:, col].any() and (~valid_labels[:, col]).any() and labels[train, col].any())
                             for col in range(len(codes))])
        ranking_eligible = labels[train].any(axis=0)
        names = [code for code, use in zip(codes, usable, strict=True) if use]
        def metrics(scores: np.ndarray) -> dict[str, object]:
            ap = {_code: _average_precision(valid_labels[:, col], scores[:, col]) for col, _code in enumerate(codes) if usable[col]}
            chosen = scores[:, ranking_eligible].argmax(axis=1)
            chosen_columns = np.flatnonzero(ranking_eligible)[chosen]
            recall_denominator = int(valid_labels.any(axis=1).sum())
            recall_numerator = int(sum(valid_labels[index, column] for index, column in enumerate(chosen_columns)))
            return {"macro_common_code_ap": float(np.mean(list(ap.values()))) if ap else None, "common_codes": names,
                    "ranking_eligible_codes_from_fit_only": [code for code, use in zip(codes, ranking_eligible, strict=True) if use],
                    "ap_denominator_codes": len(names), "recall_at_1": recall_numerator / recall_denominator if recall_denominator else None,
                    "recall_at_1_numerator": recall_numerator, "recall_at_1_denominator": recall_denominator}
        fold_results[str(fold)] = {"fit_photos": int(train.sum()), "validation_photos": int(valid.sum()),
                                   "fit_groups": len({row.group_id for row, use in zip(rows, train, strict=True) if use}),
                                   "validation_groups": len({row.group_id for row, use in zip(rows, valid, strict=True) if use}),
                                   "global_linear": metrics(global_scores), "regional_mil_topk": metrics(regional_scores)}
    return {"scope": "Frozen official base Qwen3-VL visual encoder probe only; not an end-to-end competition score.",
            "folds": list(ALLOWED_FOLDS), "excluded": "Fold0, Fold4, and all sealed/holdout rows", "grid": "fixed 4x4 adaptive pooled visual-token grid",
            "regional_pooling": f"top-{top_k} of 16 token logits", "proposal_budget": "one top-scoring common code per image for recall@1",
            "seed": seed, "fit_steps": steps, "fit_device": device_name,
            "parameter_budget": "each head has D*C+C trainable parameters", "per_fold": fold_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path, default=Path("results/clean_split_v1_summary.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/train"))
    parser.add_argument("--base-model", type=Path, default=Path("data/baseline/base_model"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".runtime/spatial-qwen3vl-cache"))
    parser.add_argument("--report", type=Path, default=Path("results/ubuntu_spatial_probe.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precompute-images", type=Path,
                        help="label-blind directory of official image files; does not read a manifest or labels")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if not 1 <= args.top_k <= GRID_SIDE * GRID_SIDE or args.steps < 1 or args.max_pixels < 65536:
        raise ValueError("top-k must be 1..16, steps positive, and max-pixels at least 65536")
    if args.precompute_images is not None:
        if args.manifest is not None or args.extract_only:
            raise ValueError("precompute mode accepts only image bytes, not manifest or extract-only options")
        images = sorted(path for path in args.precompute_images.rglob("*")
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError("precompute image directory contains no supported image files")
        extracted, total, encoder_id = extract_images(images, args.cache_dir, args.base_model, args.device, args.max_pixels)
        print(json.dumps({"status": "label-blind cache complete", "scanned_images": total,
                          "new_images": extracted, "resumed_images": total - extracted,
                          "cache_directory": str(args.cache_dir), "encoder_sha256": encoder_id}))
        return
    if args.manifest is None:
        raise ValueError("--manifest is required for supervised fit/report mode")
    rows = load_rows(args.manifest, args.summary)
    extracted, encoder_id = extract_missing(rows, args.image_root, args.cache_dir, args.base_model,
                                            args.device, args.max_pixels)
    if args.extract_only:
        print(json.dumps({"extracted": extracted, "encoder_sha256": encoder_id, "status": "label-blind cache complete"}))
        return
    report = evaluate_cached(rows, args.cache_dir, encoder_id, top_k=args.top_k, seed=args.seed, steps=args.steps,
                             device_name=args.device)
    report["cache"] = {"directory": str(args.cache_dir), "new_images": extracted, "encoder_sha256": encoder_id}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
