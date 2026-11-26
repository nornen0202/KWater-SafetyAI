"""Resume the organizer's original model, parser and formatter on fold-0 inputs."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile
from zlib import crc32

from .metrics import HEX_ID, unique_keys
from .run_official_baseline import PACKAGES, PROJECT, check_packaged_code, sha256, summarize_rows


def write_atomic(path: Path, contents: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, newline="\n", encoding="utf-8") as stream:
        scratch = Path(stream.name)
        stream.write(contents)
    try:
        os.replace(scratch, path)
    finally:
        scratch.unlink(missing_ok=True)


def read_inputs(inputs: Path) -> list[dict]:
    ids = [json.loads(line, object_pairs_hook=unique_keys)["image_id"]
           for line in (inputs / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    image_paths = sorted((inputs / "images").iterdir())
    images = {p.stem: p for p in image_paths if p.is_file()}
    if not ids or len(set(ids)) != len(ids) or any(not HEX_ID.fullmatch(i) or i not in images for i in ids):
        raise ValueError("Input manifest must have unique valid image IDs and files")
    if len(images) != len(image_paths):
        raise ValueError("Input image names have duplicate stems or non-files")
    return [{"sample_id": i, "image": images[i].relative_to(inputs).as_posix()} for i in ids]


def baseline_fingerprint(base: Path, archive_path: Path) -> dict:
    if not check_packaged_code(archive_path, base):
        raise FileNotFoundError("The originally submitted organizer archive is required")
    with ZipFile(archive_path) as archive:
        watched = {info.filename: info for info in archive.infolist()
                   if info.filename.endswith((".py", ".json", ".jsonl", ".txt", ".safetensors"))}
    fingerprints = {}
    for name, info in sorted(watched.items()):
        path = base / name
        fingerprint = hashlib.sha256()
        checksum = 0
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                checksum = crc32(chunk, checksum)
                fingerprint.update(chunk)
        if path.stat().st_size != info.file_size or checksum != info.CRC:
            raise ValueError(f"Organizer asset differs from the archived copy: {name}")
        fingerprints[name] = fingerprint.hexdigest()
    return fingerprints


def infer_cached(entry, rows, inputs, output, batch_size, timings=None):
    raw_text = []
    timings = {} if timings is None else timings
    parse_prediction = entry.parse_prediction
    original_load = entry.load_model

    @lru_cache(maxsize=1)
    def load_once(*args):
        start = time.perf_counter()
        model_and_processor = original_load(*args)
        timings["model_load_seconds"] = time.perf_counter() - start
        return model_and_processor

    entry.load_model = load_once

    def capture(text):
        raw_text.append(text)
        return parse_prediction(text)

    entry.parse_prediction = capture
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        ids = [row["sample_id"] for row in batch]
        path = output / f"{start:06d}.json"
        if path.is_file():
            records = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        else:
            raw_text.clear()
            started = time.perf_counter()
            parsed = entry.infer_batched(batch, inputs)
            timings["generation_and_load_seconds"] = timings.get("generation_and_load_seconds", 0.0) + time.perf_counter() - started
            timings["generated_photos"] = timings.get("generated_photos", 0) + len(batch)
            records = [{**row, "raw_text": text} for row, text in zip(parsed, raw_text, strict=True)]
            if [row["sample_id"] for row in records] != ids:
                raise ValueError("Official inference did not return the requested ordered IDs")
            write_atomic(path, json.dumps(records) + "\n")
            print(f"saved official raw {start + len(batch)}/{len(rows)}", flush=True)
        if [row["sample_id"] for row in records] != ids:
            raise ValueError(f"Saved batch does not match requested IDs: {path}")
        yield from records
    entry.load_model.cache_clear()


def cuda_identity() -> dict:
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability()[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The organizer's model needs one CUDA GPU with native BF16 support")
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "compute_capability": [props.major, props.minor],
            "memory_bytes": props.total_memory, "torch_cuda_version": torch.version.cuda}


def run(inputs: Path, run_dir: Path, batch_size: int, root: Path) -> dict:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not run_dir.is_relative_to((root / "data").resolve()):
        raise ValueError("The raw replay cache must stay under data/")
    before = time.perf_counter()
    rows = read_inputs(inputs)
    base = root / "data/baseline"
    archive = root / "data/packages/kwater-baseline.zip"
    versions = {pkg: importlib.metadata.version(pkg) for pkg in PACKAGES}
    gpu = cuda_identity()
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "TOKENIZERS_PARALLELISM": "false", "KW_BATCH": str(batch_size), "KWATER_SEED": "42"})
    print(f"Validating fingerprints for the organizer files and {len(rows)} images", flush=True)
    fingerprint = {"version": 2, "batch_size": batch_size, "versions": versions, "gpu": gpu,
                   "files": baseline_fingerprint(base, archive), "manifest": sha256(inputs / "manifest.jsonl"),
                   "inputs": [[r["sample_id"], r["image"], sha256(inputs / r["image"])] for r in rows]}
    provenance_seconds = time.perf_counter() - before
    cache = run_dir / "raw-batches"
    cache.mkdir(parents=True, exist_ok=True)
    config_path = cache / "config.json"
    configuration = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True) + "\n"
    if config_path.is_file():
        if config_path.read_text(encoding="utf-8") != configuration:
            raise ValueError("Resume guards differ. Use a fresh run directory.")
    else:
        if any(cache.iterdir()):
            raise ValueError("Unbound raw cache: missing configuration")
        write_atomic(config_path, configuration)

    spec = importlib.util.spec_from_file_location("official_kwater_entrypoint", base / "main.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    timings = {}
    parsed = list(infer_cached(entry, rows, inputs, cache, batch_size, timings=timings))
    if len(parsed) != len(rows):
        raise ValueError("Official inference coverage is incomplete")
    valid = [{"sample_id": r["sample_id"], "prediction": r["prediction"]}
             for r in parsed if not entry.validate_prediction(r.get("prediction"))]
    final = {}
    if valid:
        destination = run_dir / "formatted_internal.jsonl"
        entry.format_submission(valid, entry.load_rows(base / "assets/laws.jsonl"),
                                entry.load_rows(base / "assets/train.jsonl"), destination)
        final = {r["sample_id"]: r["prediction"] for r in entry.load_rows(destination)}
    output = run_dir / "output"
    output.mkdir(exist_ok=True)
    submission = output / "submission.jsonl"
    write_atomic(submission, "".join(json.dumps(entry.convert(row["sample_id"], final.get(row["sample_id"])), ensure_ascii=False) + "\n" for row in rows))
    checker = root / "data/local-kit/kit/check_output.py"
    checked = subprocess.run([sys.executable, str(checker), str(inputs), str(output)], capture_output=True,
                             text=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
    write_atomic(run_dir / "checker.log", checked.stdout + checked.stderr)
    matched = re.search(r"유효 결과\s+(\d+)/(\d+)", checked.stdout)
    if checked.returncode or not matched or int(matched[1]) != len(rows) or int(matched[2]) != len(rows):
        raise RuntimeError(f"Official output checker did not accept every photo: {run_dir / 'checker.log'}")
    model_load_seconds = timings.get("model_load_seconds", 0.0)
    generated_photos = timings.get("generated_photos", 0)
    result = {"mode": "official stock inference and formatter, batch-resumable native host wrapper", "gpu": gpu,
              "competition_container_tested": False, "native_versions": versions, "kw_batch": batch_size,
              "generation_settings": {"precision": "BF16", "max_pixels": 384 * 32 * 32,
                                      "max_new_tokens": 900, "batch_size": batch_size,
                                      "organizer_default_batch_size": 16},
              "input_count": len(rows), "official_model_schema_valid_count": len(valid),
              "official_checker_valid_rows": int(matched[1]), "current_invocation_seconds": round(time.perf_counter() - before, 2),
              "current_invocation_provenance_seconds": round(provenance_seconds, 2),
              "current_invocation_model_load_seconds": round(model_load_seconds, 2),
              "current_invocation_generation_seconds": round(timings.get("generation_and_load_seconds", 0.0) - model_load_seconds, 2),
              "current_invocation_generated_photos": generated_photos,
              "source_manifest_sha256": fingerprint["manifest"], "main_sha256": fingerprint["files"]["main.py"],
              **summarize_rows(submission, base / "assets/laws.jsonl")}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input_dir.resolve(), args.run_dir.resolve(), args.batch_size, PROJECT)
    write_atomic(args.out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
