"""Count actual Qwen3-VL image tokens on local photos without loading weights."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
from struct import unpack
from zipfile import ZipFile


MIN_PIXELS = 128 * 32 * 32
PIXEL_CAPS = (393216, 655360, 786432)
BASE = Path("data/submissions/research-8b-fold4-e1-stage-20261128/base_model")
IMAGES = Path("data/train/images")
PROMPT = Path(__file__).with_name("system_prompt_v3.txt")
TRANSFORMERS_WHEEL_SHA256 = "b10d05da8fa67dc41644dbbf9bc45a44cb86ae33da6f9295f5fbf5b7890bd267"
FULL_MANIFEST = Path("data/v3-11code-full1045/manifest_all_1045_train.json")
SIDECAR_DIRECTORY = Path("data/stage1-image-tokens")


def distribution(values: list[int]) -> dict:
    values = sorted(values)
    return {
        "min": values[0],
        "p05": values[math.ceil(0.05 * len(values)) - 1],
        "p25": values[math.ceil(0.25 * len(values)) - 1],
        "median": statistics.median(values),
        "p75": values[math.ceil(0.75 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
        "mean": round(statistics.mean(values), 4),
    }


def summarize(counts: dict[int, list[int]]) -> dict:
    reference = counts[PIXEL_CAPS[0]]
    return {
        str(cap): {
            "image_tokens": distribution(values),
            "histogram": dict(sorted(Counter(values).items())),
            "delta_tokens_vs_393216": distribution([value - baseline for value, baseline in zip(values, reference, strict=True)]),
            "photos_with_more_tokens_vs_393216": sum(value > baseline for value, baseline in zip(values, reference, strict=True)),
        }
        for cap, values in counts.items()
    }


def audit(base: Path, paths: list[Path], prompt_path: Path) -> dict:
    from PIL import Image, ImageOps
    import torch
    from transformers import AutoProcessor

    from .vlm_11code import INSPECT_USER_TEXT

    prompt = prompt_path.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]},
    ]
    processors = {}
    for cap in PIXEL_CAPS:
        processor = AutoProcessor.from_pretrained(str(base), min_pixels=MIN_PIXELS, max_pixels=cap, local_files_only=True)
        image_processor = processor.image_processor
        image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": cap}
        for attr, size in (("min_pixels", MIN_PIXELS), ("max_pixels", cap)):
            if hasattr(image_processor, attr):
                setattr(image_processor, attr, size)
        processor.tokenizer.padding_side = "left"
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processors[cap] = (processor, prefix)
    print(f"Loaded the processor for {len(paths)} photos at {len(processors)} caps", flush=True)

    counts: dict[int, list[int]] = {cap: [] for cap in PIXEL_CAPS}
    batch_size = 4
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            photos = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as source:
                    photos.append(ImageOps.exif_transpose(source).convert("RGB"))
            for cap, (processor, prefix) in processors.items():
                batch = processor(text=[prefix] * len(photos), images=photos, padding=True, return_tensors="pt")
                tokens = batch["input_ids"].eq(processor.image_token_id).sum(dim=1).tolist()
                merge_area = processor.image_processor.merge_size ** 2
                grids = batch["image_grid_thw"].prod(dim=1).tolist()
                if tokens != [patches // merge_area for patches in grids] or any(patches % merge_area for patches in grids):
                    raise ValueError("Tokenized image placeholders differ from the processor grid")
                if any(token < 1 for token in tokens):
                    raise ValueError("An image produced zero image tokens")
                counts[cap].extend(tokens)
                del batch
            if (start + batch_size) % 128 == 0:
                print(f"Processed {start + batch_size}/{len(paths)} photos", flush=True)

    primary = processors[PIXEL_CAPS[0]][0].image_processor
    bytes_bound = [base / name for name in ("chat_template.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")]
    return {
        "version": "stage1-image-token-audit-v1",
        "photo_count": len(paths),
        "min_pixels": MIN_PIXELS,
        "processor_class": type(primary).__name__,
        "patch_size": primary.patch_size,
        "spatial_merge_size": primary.merge_size,
        "batch_size": batch_size,
        "libraries": {package: version(package) for package in ("transformers", "torch", "torchvision", "Pillow")},
        "processor_files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bytes_bound},
        "system_prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "method": "CPU full processor after EXIF transpose and RGB conversion. Count input_ids equal to image_token_id; verify against image_grid_thw product divided by spatial merge area. Quartiles and tails use nearest rank, median uses middle value or middle pair average.",
        "by_max_pixels": summarize(counts),
    }


def png_orientation_and_size(path: Path) -> tuple[int, int, int]:
    from PIL import Image

    height = width = None
    orientation = 1
    with path.open("rb") as source:
        if source.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Geometry audit requires PNG input")
        while header := source.read(8):
            length, kind = unpack(">I4s", header)
            if kind == b"IHDR" and length == 13:
                width, height = unpack(">II", source.read(8))
                length -= 8
            elif kind == b"eXIf":
                exif = Image.Exif()
                exif.load(source.read(length))
                orientation = int(exif.get(274, 1))
                length = 0
            elif kind in (b"tEXt", b"zTXt", b"iTXt"):
                keyword = source.read(min(80, length))
                if keyword.startswith(b"Raw profile type exif\x00"):
                    raise ValueError("Legacy text EXIF requires the full processor audit")
                length -= len(keyword)
            elif kind == b"IEND":
                break
            source.seek(length + 4, 1)
        else:
            raise ValueError("PNG geometry audit requires a complete IEND marker")
    if height is None or width is None:
        raise ValueError("PNG geometry audit is missing the required IHDR dimensions")
    if orientation in (5, 6, 7, 8):
        height, width = width, height
    return orientation, height, width


def audited_image_geometry(paths: list[Path], inventory: Path) -> tuple[list[str], list[tuple[int, int]]]:
    source = json.loads(inventory.read_bytes())
    if source["errors"]:
        raise ValueError("The decoded-photo inventory contains errors")
    full_manifest = {row["image"]: row["sha256"] for row in json.loads(FULL_MANIFEST.read_bytes())}
    audited = {row["image"]: row for row in source["inventory"].values()}
    if {name: row["sha256"] for name, row in audited.items()} != full_manifest:
        raise ValueError("The decoded-photo digests differ from the frozen full1045 manifest")
    if set(audited) != {f"images/{path.name}" for path in paths}:
        raise ValueError("The current file roster differs from the decoded-photo inventory")
    ordered = [audited[f"images/{path.name}"] for path in paths]
    hashes = [row["sha256"] for row in ordered]
    if len(hashes) != len(set(hashes)):
        raise ValueError("The decoded-photo inventory needs one row per photo content digest")
    return hashes, [(row["height"], row["width"]) for row in ordered]


def audit_geometry(base: Path, paths: list[Path], wheel: Path, inventory: Path | None = None,
                   local_sidecar: Path | None = None) -> dict:
    with wheel.open("rb") as source:
        wheel_sha = hashlib.file_digest(source, "sha256").hexdigest()
    if wheel_sha != TRANSFORMERS_WHEEL_SHA256:
        raise ValueError("Geometry audit requires the submitted Transformers 4.57.1 wheel")
    module_name = "transformers/models/qwen2_vl/image_processing_qwen2_vl.py"
    with ZipFile(wheel) as archive:
        module = ast.parse(archive.read(module_name))
    resize = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "smart_resize")
    namespace = {"math": math}
    exec(compile(ast.Module(body=[resize], type_ignores=[]), module_name, "exec"), namespace)

    processor_file = base / "preprocessor_config.json"
    settings = json.loads(processor_file.read_bytes())
    if settings["image_processor_type"] != "Qwen2VLImageProcessorFast":
        raise ValueError("Expected the Qwen2VL fast image processor used by Qwen3-VL")
    factor = settings["patch_size"] * settings["merge_size"]
    if inventory is None:
        oriented = [png_orientation_and_size(path) for path in paths]
        dimensions = [(h, w) for _, h, w in oriented]
        method = "Read PNG IHDR and EXIF orientation."
    else:
        hashes, dimensions = audited_image_geometry(paths, inventory)
        method = "Reuse EXIF-normalized decoded dimensions, bound to the frozen full1045 manifest by raw-content SHA-256."
    counts: dict[int, list[int]] = {cap: [] for cap in PIXEL_CAPS}
    for cap, values in counts.items():
        for height, width in dimensions:
            h, w = namespace["smart_resize"](height, width, factor=factor, min_pixels=MIN_PIXELS, max_pixels=cap)
            values.append(h * w // factor**2)
    report = {
        "version": "stage1-image-token-geometry-audit-v1",
        "photo_count": len(paths),
        "min_pixels": MIN_PIXELS,
        "processor_class": settings["image_processor_type"],
        "patch_size": settings["patch_size"],
        "spatial_merge_size": settings["merge_size"],
        "processor_file_sha256": hashlib.sha256(processor_file.read_bytes()).hexdigest(),
        "transformers_wheel_sha256": wheel_sha,
        "physical_image_resize_and_tokenizer_run": False,
        "method": method + " Run the submitted wheel's exact smart_resize, followed by per-photo resized area divided by patch and spatial merge area. Quartiles and tails use nearest rank, median uses middle value or middle pair average.",
        "by_max_pixels": summarize(counts),
    }
    if inventory is None:
        report["exif_orientation_swapped_photos"] = sum(orientation in (5, 6, 7, 8) for orientation, _, _ in oriented)
    elif local_sidecar is not None:
        entries = sorted(zip(hashes, *counts.values(), strict=True))
        records = (json.dumps({"sha256": sha, "image_tokens_by_max_pixels": dict(zip(PIXEL_CAPS, values, strict=True))},
                              separators=(",", ":")) + "\n" for sha, *values in entries)
        sidecar_bytes = "".join(records).encode("utf-8")
        local_sidecar.parent.mkdir(parents=True, exist_ok=True)
        local_sidecar.write_bytes(sidecar_bytes)
        report["local_per_photo_sidecar"] = {
            "sha256": hashlib.sha256(sidecar_bytes).hexdigest(), "bytes": len(sidecar_bytes), "photos": len(entries),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    parser.add_argument("--expected-count", type=int, default=1045)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-wheel", type=Path, help="Use exact pinned resize geometry without decoding PNG pixels or importing Torch")
    parser.add_argument("--decoded-inventory", type=Path, help="Reuse EXIF-normalized dimensions from a prior local photo audit")
    parser.add_argument("--local-sidecar", type=Path, help="Write per-photo digest and token counts under data/stage1-image-tokens")
    args = parser.parse_args()
    if args.decoded_inventory and not args.geometry_wheel:
        parser.error("--decoded-inventory requires --geometry-wheel")
    if args.local_sidecar and (not args.decoded_inventory or not args.local_sidecar.resolve().is_relative_to(SIDECAR_DIRECTORY.resolve())):
        parser.error("--local-sidecar requires --decoded-inventory and an output under data/stage1-image-tokens")
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file())
    if args.expected_count < 1 or len(paths) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} local photos, found {len(paths)}")
    if args.geometry_wheel:
        result = audit_geometry(args.base, paths, args.geometry_wheel, args.decoded_inventory, args.local_sidecar)
    else:
        import torch

        torch.set_num_threads(1)
        result = audit(args.base, paths, args.prompt)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
