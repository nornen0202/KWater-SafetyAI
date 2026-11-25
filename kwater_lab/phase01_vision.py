"""Frozen SigLIP 2 So400m and DINOv2-L spatial tokens for global and four overlapping views."""

import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F


SIGNATURE = "phase01-r2-sig384-dino224-grid4-proj64-seed404-tiles60pct-training-jpeg"
MODEL_IDS = {
    "sig": (
        "google/siglip2-so400m-patch14-384",
        "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    ),
    "dino": ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"),
}


def views(image):
    w, h = image.size
    tw, th = max(1, round(0.6 * w)), max(1, round(0.6 * h))
    return [image] + [
        image.crop((x, y, x + tw, y + th)) for y in (0, h - th) for x in (0, w - tw)
    ]


class Vision:
    def __init__(self, folder, device):
        from transformers import Dinov2Model, SiglipVisionModel

        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.models = {}
        self.projection = {}
        self.norm = {}
        for name, klass, edge, width in (
            ("sig", SiglipVisionModel, 384, 1152),
            ("dino", Dinov2Model, 224, 1024),
        ):
            started = time.monotonic()
            path = Path(folder) / name
            print("loading", name, flush=True)
            model, info = klass.from_pretrained(
                path,
                local_files_only=True,
                output_loading_info=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map={"": str(self.device)},
            )
            if info["missing_keys"] or info["mismatched_keys"]:
                raise ValueError(
                    f"Uninitialized {name} backbone: {info['missing_keys']}, {info['mismatched_keys']}"
                )
            self.models[name] = model.eval()
            cfg = json.loads((path / "preprocessor_config.json").read_bytes())
            self.norm[name] = (
                edge,
                np.array(cfg["image_mean"], np.float32),
                np.array(cfg["image_std"], np.float32),
                Image.Resampling(int(cfg["resample"])),
            )
            rng = np.random.default_rng(404 + width)
            matrix = rng.standard_normal((width, 64), dtype=np.float32) / np.sqrt(
                np.float32(width)
            )
            self.projection[name] = torch.as_tensor(
                matrix, device=self.device, dtype=self.dtype
            )
            print("loaded", name, round(time.monotonic() - started, 2), flush=True)

    @torch.inference_mode()
    def encode(self, image, multiscale=True, augmentation_seed=None):
        original = ImageOps.exif_transpose(image).convert("RGB")
        frames = views(original) if multiscale else [original]
        if augmentation_seed is not None:
            rng = np.random.default_rng(augmentation_seed)
            # Use the same photo-specific nuisance transform in both scale arms, only on training rows.
            small = ImageOps.contain(original, (768, 768), Image.Resampling.BILINEAR)
            small = ImageEnhance.Brightness(small).enhance(float(rng.uniform(0.9, 1.1)))
            warm = float(rng.uniform(0.95, 1.05))
            red, green, blue = small.split()
            small = Image.merge(
                "RGB",
                (red.point(lambda p: p * warm), green, blue.point(lambda p: p / warm)),
            )
            stream = io.BytesIO()
            small.save(stream, format="JPEG", quality=int(rng.integers(75, 96)))
            stream.seek(0)
            with Image.open(stream) as jpeg:
                frames.append(jpeg.convert("RGB"))
        emb = []
        for name, model in self.models.items():
            edge, mean, std, method = self.norm[name]
            fill = tuple(int(round(float(m) * 255)) for m in mean)
            pixels = np.stack(
                [
                    np.asarray(
                        ImageOps.pad(v, (edge, edge), method=method, color=fill),
                        np.float32,
                    )
                    for v in frames
                ]
            )
            pixels = np.ascontiguousarray(
                ((pixels / 255.0 - mean) / std).transpose(0, 3, 1, 2)
            )
            out = model(
                pixel_values=torch.from_numpy(pixels).to(self.device, dtype=self.dtype)
            )
            patches = (
                out.last_hidden_state[:, 1:]
                if name == "dino"
                else out.last_hidden_state
            )
            size = int(patches.shape[1] ** 0.5)
            if size * size != patches.shape[1]:
                raise ValueError("Unexpected patch grid")
            grid = F.adaptive_avg_pool2d(
                patches.transpose(1, 2).reshape(len(frames), -1, size, size), (4, 4)
            )
            tok = torch.cat(
                [out.pooler_output[:, None, :], grid.flatten(2).transpose(1, 2)], dim=1
            )
            tok = F.normalize((tok @ self.projection[name]).float(), dim=-1)
            emb.append(tok.cpu().numpy())
        data = np.concatenate(emb, axis=-1).astype(np.float16)
        if data.shape != (len(frames), 17, 128) or not np.isfinite(data).all():
            raise ValueError("Invalid vision tokens")
        return data


def feature_cache(vision, train, run, rows, limit=None):
    cache = run / "features"
    cache.mkdir(exist_ok=True)
    start = time.monotonic()
    result = []
    for i, row in enumerate(rows[:limit]):
        name = hashlib.sha256((SIGNATURE + row["sha256"]).encode()).hexdigest() + ".npy"
        path = cache / name
        if path.exists():
            with path.open("rb") as f:
                data = np.load(f, allow_pickle=False)
        else:
            image_path = train / row["image"]
            raw = image_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Changed training image")
            with Image.open(io.BytesIO(raw)) as pic:
                data = vision.encode(pic, augmentation_seed=int(row["sha256"][:16], 16))
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.save(f, data, allow_pickle=False)
            tmp.replace(path)
        if (
            data.shape != (6, 17, 128)
            or data.dtype != np.float16
            or not np.isfinite(data).all()
        ):
            raise ValueError(f"Invalid cache: {path}")
        result.append(data)
        if (i + 1) % 25 == 0:
            print(
                "features",
                i + 1,
                "/",
                len(rows[:limit]),
                round(time.monotonic() - start, 1),
                flush=True,
            )
    return np.stack(result)
