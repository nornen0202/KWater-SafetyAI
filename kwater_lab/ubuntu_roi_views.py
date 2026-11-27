"""Local, label-free pretrained ROI views and independent BLIP captions.

Run: .venv/bin/python -m kwater_lab.ubuntu_roi_views
Model files are downloaded separately from official vendors and verified here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch
import torchvision
import transformers
from PIL import Image, ImageFilter, ImageOps
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import pil_to_tensor
from transformers import BlipConfig, BlipForConditionalGeneration, BlipProcessor

ROOT: Final = Path(__file__).resolve().parents[1]
WORK: Final = ROOT / "data/ubuntu-roi-20261127"
SIDE: Final = 512
BLIP_REVISION: Final = "82a37760796d32b1411fe092ab5d4e227313294b"


@dataclass(frozen=True, slots=True)
class Photo:
    sample_id: str
    image: str
    sha256: str
    prefix: str


@dataclass(frozen=True, slots=True)
class Selection:
    index: int | None
    reason: str


class Prediction(TypedDict):
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    masks: torch.Tensor


class ViewRow(TypedDict):
    sample_id: str
    image_sha256: str
    fixed: str
    box: str
    mask: str
    sham: str
    caption: str
    detected: bool
    selection_reason: str
    class_name: str
    score: float
    bbox_xyxy: list[int]
    original_size: list[int]
    contour: str
    sham_contour: str
    contour_pixels: int
    sham_pixels: int
    contour_sham_iou: float
    overlay_color: list[int]
    detection_seconds: float
    caption_seconds: float


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose_detection(prediction: Prediction) -> Selection:
    """Prefer the largest confident person, then strongest other detection."""
    confident = torch.where(prediction["scores"] >= 0.5)[0].tolist()
    people = [index for index in confident if prediction["labels"][index].item() == 1]
    if people:
        boxes = prediction["boxes"]
        index = max(
            people,
            key=lambda i: float(
                (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            ),
        )
        return Selection(index, "largest_person")
    if confident:
        return Selection(
            max(confident, key=lambda i: float(prediction["scores"][i])),
            "strongest_nonperson",
        )
    return Selection(None, "center_fallback_no_score_ge_0.5")


def center_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        round(width * 0.15),
        round(height * 0.15),
        round(width * 0.85),
        round(height * 0.85),
    )


def padded_box(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Expand each bbox side by 20 percent of its corresponding dimension."""
    left, top, right, bottom = box
    dx, dy = (right - left) * 0.2, (bottom - top) * 0.2
    return (
        max(0, int(left - dx)),
        max(0, int(top - dy)),
        min(size[0], int(np.ceil(right + dx))),
        min(size[1], int(np.ceil(bottom + dy))),
    )


def canvas(image: Image.Image, *, nearest: bool = False) -> Image.Image:
    return ImageOps.pad(
        image,
        (SIDE, SIDE),
        method=Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS,
        color=0,
    )


def contours(binary: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Circular translation is a bijection: equal touched pixels, no clipping loss."""
    dilated = np.asarray(binary.filter(ImageFilter.MaxFilter(3))) > 0
    eroded = np.asarray(binary.filter(ImageFilter.MinFilter(3))) > 0
    real = dilated & ~eroded
    sham = np.roll(real, shift=(SIDE // 2, SIDE // 2), axis=(0, 1))
    return real, sham


def overlay_pair(
    base: Image.Image, pair: tuple[np.ndarray, np.ndarray]
) -> tuple[Image.Image, Image.Image, list[int]]:
    """Select one colour absent from the entire base, guaranteeing equal changes."""
    pixels = np.asarray(base)
    channels = pixels.astype(np.uint32)
    occupied = set(
        ((channels[:, :, 0] << 16) | (channels[:, :, 1] << 8) | channels[:, :, 2])
        .ravel()
        .tolist()
    )
    packed_color = 0xFF00FF
    while packed_color in occupied:
        packed_color = (packed_color + 1) % (1 << 24)
    color = [packed_color >> 16, (packed_color >> 8) & 255, packed_color & 255]
    rendered = []
    for contour in pair:
        modified = pixels.copy()
        modified[contour] = color
        rendered.append(Image.fromarray(modified))
    return rendered[0], rendered[1], color


def build_views(photo: Photo, prediction: Prediction, seconds: float) -> ViewRow:
    """Create comparable RGB images without consulting hazard annotations."""
    folder = WORK / "views" / photo.sample_id
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/train" / photo.image
    if digest(source) != photo.sha256:
        raise RuntimeError(f"Input digest mismatch: {photo.sample_id}")
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    selection = choose_detection(prediction)
    index = selection.index
    bounds = (
        center_box(image.size)
        if index is None
        else padded_box(prediction["boxes"][index].tolist(), image.size)
    )
    crop = canvas(image.crop(bounds))
    if index is None:
        binary = Image.new("L", image.size)
    else:
        binary = Image.fromarray(
            (prediction["masks"][index, 0].numpy() >= 0.5).astype(np.uint8) * 255
        )
    real, sham = contours(canvas(binary.crop(bounds), nearest=True))
    masked, placebo, color = overlay_pair(crop, (real, sham))
    images = {
        "fixed": canvas(image.crop(center_box(image.size))),
        "box": crop,
        "mask": masked,
        "sham": placebo,
        "contour": Image.fromarray(real.astype(np.uint8) * 255),
        "sham_contour": Image.fromarray(sham.astype(np.uint8) * 255),
    }
    paths = {}
    for name, rendered in images.items():
        path = folder / f"{name}.png"
        rendered.save(path)
        paths[name] = str(path)
    union = int(np.logical_or(real, sham).sum())
    return ViewRow(
        sample_id=photo.sample_id,
        image_sha256=photo.sha256,
        fixed=paths["fixed"],
        box=paths["box"],
        mask=paths["mask"],
        sham=paths["sham"],
        caption="",
        detected=index is not None,
        selection_reason=selection.reason,
        class_name="none"
        if index is None
        else MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"][
            int(prediction["labels"][index])
        ],
        score=0.0 if index is None else float(prediction["scores"][index]),
        bbox_xyxy=list(bounds),
        original_size=list(image.size),
        contour=paths["contour"],
        sham_contour=paths["sham_contour"],
        contour_pixels=int(real.sum()),
        sham_pixels=int(sham.sum()),
        contour_sham_iou=float(np.logical_and(real, sham).sum()) / union
        if union
        else 1.0,
        overlay_color=color,
        detection_seconds=seconds,
        caption_seconds=0.0,
    )


def main() -> None:
    """Verify official weights, serialize GPU work, then persist complete rows."""
    model_dir = WORK / "models"
    detector_path = model_dir / "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
    caption_path = model_dir / "blip/pytorch_model.bin"
    detector_sha, caption_sha = digest(detector_path), digest(caption_path)
    tree = json.loads((model_dir / "blip-tree.json").read_text())
    published_caption_sha = next(
        item["lfs"]["oid"] for item in tree if item["path"] == "pytorch_model.bin"
    )
    if not detector_sha.startswith("bf2d0c1e") or caption_sha != published_caption_sha:
        raise RuntimeError("Official model digest verification failed")
    raw = json.loads((WORK / "cohort.json").read_text())
    photos = [Photo(**row) for row in raw["rows"]]
    if len({photo.sample_id for photo in photos}) != len(photos):
        raise RuntimeError("Duplicate sample identity")
    torch.manual_seed(20261127)
    torch.set_num_threads(4)
    rows = []
    started = time.perf_counter()
    lock = ROOT / ".runtime/ubuntu-weekend-gpu.lock"
    with lock.open("a") as handle:
        print("Waiting for shared GPU lock", flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print("Acquired GPU lock", flush=True)
        torch.cuda.reset_peak_memory_stats()
        detector = maskrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, min_size=640, max_size=960
        )
        detector.load_state_dict(
            torch.load(detector_path, map_location="cpu", weights_only=True)
        )
        detector.eval().to("cuda")
        for number, photo in enumerate(photos):
            source = ROOT / "data/train" / photo.image
            if digest(source) != photo.sha256:
                raise RuntimeError(f"Input digest mismatch: {photo.sample_id}")
            with Image.open(source) as opened:
                tensor = (
                    pil_to_tensor(opened.convert("RGB")).float().div(255).to("cuda")
                )
            tick = time.perf_counter()
            with torch.inference_mode():
                output = detector([tensor])[0]
            prediction = {key: value.cpu() for key, value in output.items()}
            torch.cuda.synchronize()
            rows.append(build_views(photo, prediction, time.perf_counter() - tick))
            del output, tensor, prediction
            print(f"Detection {number + 1}/{len(photos)}", flush=True)
        detector_peak = torch.cuda.max_memory_allocated()
        del detector
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        processor = BlipProcessor.from_pretrained(
            model_dir / "blip", local_files_only=True
        )
        config = BlipConfig.from_pretrained(model_dir / "blip", local_files_only=True)
        captioner = BlipForConditionalGeneration(config)
        state = torch.load(caption_path, map_location="cpu", weights_only=True)
        # Older BLIP checkpoints persist this deterministic buffer; current HF does not.
        legacy_positions = state.pop("text_decoder.bert.embeddings.position_ids")
        if not torch.equal(legacy_positions, torch.arange(512).expand((1, -1))):
            raise RuntimeError("Unexpected legacy BLIP position_ids buffer")
        captioner.load_state_dict(state, strict=True)
        del state
        captioner.eval().to("cuda")
        for number, row in enumerate(rows):
            with Image.open(row["box"]) as opened:
                inputs = processor(
                    images=opened.convert("RGB"), return_tensors="pt"
                ).to("cuda")
            tick = time.perf_counter()
            with torch.inference_mode():
                output = captioner.generate(
                    **inputs, max_new_tokens=40, num_beams=3, do_sample=False
                )
            row["caption"] = processor.decode(
                output[0], skip_special_tokens=True
            ).strip()
            torch.cuda.synchronize()
            row["caption_seconds"] = time.perf_counter() - tick
            if not row["caption"]:
                raise RuntimeError(f"Empty caption: {row['sample_id']}")
            print(f"Caption {number + 1}/{len(rows)}", flush=True)
        caption_peak = torch.cuda.max_memory_allocated()
        del captioner, inputs, output
        torch.cuda.empty_cache()
    metadata = {
        "rows": rows,
        "canvas": [SIDE, SIDE],
        "cohort_sha256": digest(WORK / "cohort.json"),
        "detector": {
            "name": "torchvision MaskRCNN ResNet50 FPN COCO_V1",
            "sha256": detector_sha,
            "published_sha256_prefix": "bf2d0c1e",
            "min_size": 640,
            "max_size": 960,
            "score_threshold": 0.5,
            "mask_threshold": 0.5,
        },
        "captioner": {
            "name": "Salesforce/blip-image-captioning-base",
            "revision": BLIP_REVISION,
            "sha256": caption_sha,
            "published_lfs_sha256": published_caption_sha,
            "independent_of_readout": True,
            "max_new_tokens": 40,
            "num_beams": 3,
            "do_sample": False,
            "checkpoint_compatibility": "Verified and omitted legacy deterministic position_ids buffer; all parameters loaded strict=True",
        },
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
        "geometry": {
            "crop_padding_fraction_per_side": 0.2,
            "fixed_center_fraction_per_dimension": 0.7,
            "sham": "circular translation by (256,256) pixels; may wrap across canvas edges",
            "overlay": "opaque two-sided morphological contour, equal changed pixel count and identical colour",
            "no_detection": "all ROI arms use center fallback; contours empty",
        },
        "detector_peak_allocated_bytes": detector_peak,
        "captioner_peak_allocated_bytes": caption_peak,
        "wall_seconds_including_lock_wait": time.perf_counter() - started,
        "model_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(model_dir.rglob("*"))
            if path.is_file()
        ],
    }
    index = WORK / "views/index.json"
    index.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Complete: {len(rows)} rows; {index}", flush=True)


if __name__ == "__main__":
    main()
