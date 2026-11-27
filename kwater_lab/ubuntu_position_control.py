"""Kit-only control: can frozen Qwen 4x4 visual tokens reveal their tile quadrant?

This is a fallback diagnostic, not a hazard model or submission component.  It
uses only the ten local-kit photographs, extracts the same adaptive-pooled Qwen
visual grids as ``spatial_probe``, and discards both tiny classifier heads after
reporting held-out tile accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
import torch.nn.functional as functional

from .spatial_probe import GRID_SIDE, Row, encoder_fingerprint, extract_missing, read_cached


KIT_PHOTO_COUNT: Final = 10
QUADRANT_COUNT: Final = 4


@dataclass(frozen=True, slots=True)
class KitPhoto:
    """Label-free local-kit image identity."""

    sample_id: str
    image: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FitOptions:
    """Shared optimizer conditions for true and shuffled controls."""

    seed: int
    steps: int
    learning_rate: float
    weight_decay: float
    device: str


def digest(path: Path) -> str:
    """Hash the exact kit photo bytes used for feature extraction."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_kit_photos(manifest: Path, image_root: Path) -> list[KitPhoto]:
    """Load exactly the local-kit manifest without opening any training labels."""
    raw = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw) != KIT_PHOTO_COUNT or any(set(item) != {"image_id"} for item in raw):
        raise ValueError("Expected exactly ten label-free local-kit manifest rows")
    photos = []
    for item in raw:
        sample_id = item["image_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("local-kit image_id must be a nonempty string")
        image = f"{sample_id}.jpg"
        image_path = image_root / image
        if not image_path.is_file():
            raise ValueError(f"Kit image missing: {image}")
        photos.append(KitPhoto(sample_id, image, digest(image_path)))
    if len({photo.sample_id for photo in photos}) != KIT_PHOTO_COUNT or len({photo.sha256 for photo in photos}) != KIT_PHOTO_COUNT:
        raise ValueError("Kit photos must have unique IDs and byte hashes")
    return photos


def split_photos(photos: list[KitPhoto], seed: int) -> tuple[list[KitPhoto], list[KitPhoto]]:
    """Freeze a content-hash-derived 6/4 split before looking at cached features."""
    ordered = sorted(photos, key=lambda photo: hashlib.sha256(f"{seed}:{photo.sha256}".encode()).hexdigest())
    return ordered[:6], ordered[6:]


def quadrants(photo_count: int) -> np.ndarray:
    """Return native top-left/top-right/bottom-left/bottom-right labels for 4x4 tiles."""
    one_grid = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3], dtype=np.int64)
    return np.tile(one_grid, photo_count)


def deterministic_shuffle(labels: np.ndarray, photo_ids: list[str], seed: int) -> np.ndarray:
    """Create train-only hash-shuffled labels with the same class counts."""
    if labels.shape != (len(photo_ids) * GRID_SIDE * GRID_SIDE,):
        raise ValueError("Tile labels and training photos have incompatible sizes")
    keys = [hashlib.sha256(f"{seed}:{photo_id}:{tile}".encode()).hexdigest()
            for photo_id in photo_ids for tile in range(GRID_SIDE * GRID_SIDE)]
    return labels[np.argsort(keys, kind="stable")]


def cached_features(photos: list[KitPhoto], cache_dir: Path, encoder_id: str) -> np.ndarray:
    """Read valid 16-token grids and fail rather than silently changing feature conditions."""
    rows = [Row(photo.sample_id, photo.image, photo.sha256, f"kit:{photo.sha256}", 1, frozenset()) for photo in photos]
    grids = [read_cached(row, cache_dir, encoder_id) for row in rows]
    if any(grid is None for grid in grids):
        raise ValueError("All kit photos require matching 4x4 visual caches")
    return np.concatenate([grid for grid in grids if grid is not None], axis=0)


def fit_accuracy(train_features: np.ndarray, train_labels: np.ndarray, heldout_features: np.ndarray,
                 heldout_labels: np.ndarray, options: FitOptions) -> float:
    """Fit one disposable linear tile classifier on the requested device."""
    if train_features.ndim != 2 or heldout_features.ndim != 2 or train_features.shape[1] != heldout_features.shape[1]:
        raise ValueError("Features must be compatible two-dimensional matrices")
    device = torch.device(options.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fitting was requested but CUDA is unavailable")
    torch.manual_seed(options.seed)
    train_x = torch.from_numpy(train_features).float().to(device)
    heldout_x = torch.from_numpy(heldout_features).float().to(device)
    train_y = torch.from_numpy(train_labels).to(device)
    weight = torch.nn.Parameter(torch.zeros((train_x.shape[1], QUADRANT_COUNT), device=device))
    bias = torch.nn.Parameter(torch.zeros(QUADRANT_COUNT, device=device))
    optimizer = torch.optim.AdamW((weight, bias), lr=options.learning_rate, weight_decay=options.weight_decay)
    for _ in range(options.steps):
        loss = functional.cross_entropy(train_x @ weight + bias, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = float(((heldout_x @ weight + bias).argmax(dim=1) == torch.from_numpy(heldout_labels).to(device)).float().mean())
    del weight, bias, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return accuracy


def evaluate(photos: list[KitPhoto], cache_dir: Path, encoder_id: str, options: FitOptions) -> dict[str, object]:
    """Compare true and train-only-shuffled native quadrant labels on held-out photos."""
    train, heldout = split_photos(photos, options.seed)
    train_features = cached_features(train, cache_dir, encoder_id)
    heldout_features = cached_features(heldout, cache_dir, encoder_id)
    train_labels = quadrants(len(train))
    heldout_labels = quadrants(len(heldout))
    shuffled = deterministic_shuffle(train_labels, [photo.sample_id for photo in train], options.seed)
    true_accuracy = fit_accuracy(train_features, train_labels, heldout_features, heldout_labels, options)
    shuffled_accuracy = fit_accuracy(train_features, shuffled, heldout_features, heldout_labels, options)
    return {
        "scope": "Kit-only visual-position control; not a hazard, quality, or submission score.",
        "photo_count": len(photos), "tile_count": len(photos) * GRID_SIDE * GRID_SIDE,
        "grid": "Qwen visual encoder adaptive-pooled 4x4 grid, spatial_probe-compatible cache",
        "split": {"method": "SHA-256(seed:content_sha256)", "train_photo_hashes": [photo.sha256 for photo in train],
                  "heldout_photo_hashes": [photo.sha256 for photo in heldout]},
        "classes": ["top_left", "top_right", "bottom_left", "bottom_right"], "chance_accuracy": 0.25,
        "fit_conditions": asdict(options), "true_quadrant_heldout_accuracy": true_accuracy,
        "train_only_shuffled_quadrant_heldout_accuracy": shuffled_accuracy,
        "accuracy_difference": true_accuracy - shuffled_accuracy,
        "comparison_gate": "ADVANTAGE" if true_accuracy > shuffled_accuracy else "NO_ADVANTAGE",
        "heldout_photo_count": len(heldout), "heldout_correlated_tile_count": len(heldout_labels),
        "learned_head_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/local-kit/sample-input/manifest.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/local-kit/sample-input/images"))
    parser.add_argument("--base-model", type=Path, default=Path("data/baseline/base_model"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".runtime/spatial-qwen3vl-cache"))
    parser.add_argument("--out", type=Path, default=Path("data/ubuntu-position-control/report.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit-device", default="cpu")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if args.steps < 1 or args.max_pixels < 65536 or not args.out.resolve().is_relative_to((Path.cwd() / "data").resolve()):
        raise ValueError("steps must be positive, max-pixels at least 65536, and --out under ignored data/")
    photos = load_kit_photos(args.manifest, args.image_root)
    rows = [Row(photo.sample_id, photo.image, photo.sha256, f"kit:{photo.sha256}", 1, frozenset()) for photo in photos]
    extracted, encoder_id = extract_missing(rows, args.image_root, args.cache_dir, args.base_model, args.device, args.max_pixels)
    if args.extract_only:
        print(json.dumps({"extracted": extracted, "encoder_sha256": encoder_id, "kit_photos": len(photos)}))
        return
    options = FitOptions(args.seed, args.steps, 0.03, 0.02, args.fit_device)
    report = evaluate(photos, args.cache_dir, encoder_id, options)
    report["cache"] = {"new_images": extracted, "encoder_sha256": encoder_id}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
