"""Run the original K-water baseline and its official output checker on a local NVIDIA GPU."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torchvision", "transformers", "peft", "accelerate", "huggingface-hub",
            "tokenizers", "numpy", "Pillow", "safetensors", "jsonschema")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_used_mib():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.DEVNULL)
        return int(result.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def check_packaged_code(archive_path, baseline):
    if not archive_path.is_file():
        return None
    with ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".py") or name == "system_prompt.txt"]
        for name in names:
            archived = hashlib.sha256(archive.read(name)).hexdigest()
            if sha256(baseline / name) != archived:
                raise ValueError(f"The local file differs from the organizer archive: {name}")
    return len(names)


def summarize_rows(output, laws):
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_ids = set()
    with laws.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            article = str(json.loads(line).get("article", "")).strip()
            match = re.fullmatch(r"(?:제)?(\d+)(?:조)?(?:의(\d+))?", article)
            if match:
                suffix = f"-{int(match[2])}" if match[2] else ""
                source_ids.add(f"OSH-{int(match[1])}{suffix}")
    hazards = [item for row in rows for item in row.get("hazards", {}).values()]
    references = [ref for item in hazards for ref in item.get("references", [])]
    law_ids = [ref["id"] for ref in references if ref["kind"] == "law"]
    return {
        "n_rows": len(rows), "n_hazard": sum(row.get("status") == "hazard" for row in rows),
        "n_no_visible_hazard": sum(row.get("status") == "no_visible_hazard" for row in rows),
        "predicted_code_counts": dict(Counter(key for row in rows for key in row.get("hazards", {}))),
        "reference_count": len(references), "law_reference_count": len(law_ids),
        "law_reference_ids_resolved_in_official_corpus": sum(key in source_ids for key in law_ids),
        "output_sha256": sha256(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT / "data/baseline")
    parser.add_argument("--kit-dir", type=Path, default=PROJECT / "data/local-kit")
    parser.add_argument("--input-dir", type=Path, help="Defaults to the official kit's sample-input")
    parser.add_argument("--batch-size", type=int, default=1, help="Official KW_BATCH setting (organizer default 16)")
    parser.add_argument("--time-limit", type=int, default=1800)
    parser.add_argument("--run-name", default="official-baseline-10")
    args = parser.parse_args()
    if args.batch_size < 1 or args.time_limit < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("Specify a positive batch size and time limit, and a single run name")
    baseline, kit = args.baseline_dir.resolve(), args.kit_dir.resolve()
    inputs = (args.input_dir or kit / "sample-input").resolve()
    manifest = inputs / "manifest.jsonl"
    total = sum(bool(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines())
    versions = {package: importlib.metadata.version(package) for package in PACKAGES}
    original_files = check_packaged_code(PROJECT / "data/packages/kwater-baseline.zip", baseline)
    run = PROJECT / "data" / args.run_name
    report = PROJECT / "results" / (args.run_name.replace("-", "_") + ".json")
    if run.exists() or report.exists():
        parser.error("The run already exists. Set --run-name to retain both experiments")
    output = run / "output"
    output.mkdir(parents=True)
    scratch = PROJECT / ".runtime" / "temp"
    scratch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(PROJECT / ".runtime/hf"), "TORCH_HOME": str(PROJECT / ".runtime/torch"),
        "CUDA_CACHE_PATH": str(PROJECT / ".runtime/cuda-cache"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "TEMP": str(scratch), "TMP": str(scratch), "TMPDIR": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "KWATER_SEED": "42", "PYTHONHASHSEED": "42", "KW_BATCH": str(args.batch_size),
    })
    command = [sys.executable, str(baseline / "main.py"), "--input-dir", str(inputs), "--output-dir", str(output)]
    before = gpu_used_mib()
    samples = [] if before is None else [before]
    log = run / "model.log"
    start = time.perf_counter()
    timed_out = False
    with log.open("wb") as stream:
        print(f"Starting original main.py for {total} photos, KW_BATCH={args.batch_size}", flush=True)
        process = subprocess.Popen(command, cwd=baseline, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while process.poll() is None:
                used = gpu_used_mib()
                if used is not None:
                    samples.append(used)
                if time.perf_counter() - start > args.time_limit:
                    timed_out = True
                    break
                time.sleep(2)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    elapsed = round(time.perf_counter() - start, 3)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    parsed = re.search(r"(?m)^valid model outputs (\d+)/(\d+)", log_text)
    batches = re.findall(r"(?m)^infer (\d+)/(\d+) ([0-9.]+)s", log_text)
    archive_state = PROJECT / ".runtime/official-pack-state.json"
    archive = json.loads(archive_state.read_text(encoding="utf-8")) if archive_state.exists() else {}
    summary = {
        "finished_utc": datetime.now(timezone.utc).isoformat(), "mode": "official_code_native_host",
        "competition_container_tested": False, "external_network_blocked_at_os_level": False,
        "offline_environment": {key: env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
        "main_sha256": sha256(baseline / "main.py"), "archive_sha256": archive.get("archive_sha256"),
        "code_and_prompt_files_matching_archive": original_files,
        "input_count": total, "source_manifest_sha256": sha256(manifest), "native_versions": versions,
        "command": command, "kw_batch": args.batch_size, "seed": 42, "return_code": process.returncode,
        "native_timeout_seconds": args.time_limit, "time_limit_reached": timed_out,
        "elapsed_seconds_including_native_model_startup": elapsed,
        "gpu_system_used_mib_before": before, "gpu_system_peak_used_mib": max(samples) if samples else None,
        "any_traceback": "Traceback (most recent call last):" in log_text, "full_model_log": str(log),
        "official_model_schema_valid_count": int(parsed[1]) if parsed else None,
        "internal_invalid_fallback_count": int(parsed[2]) - int(parsed[1]) if parsed else None,
        "infer_progress": [{"cumulative_images": int(n), "total": int(t), "generation_seconds": float(sec)}
                           for n, t, sec in batches],
    }
    if process.returncode == 0:
        check = subprocess.run([sys.executable, str(kit / "kit/check_output.py"), str(inputs), str(output)],
                               env=env, capture_output=True, encoding="utf-8", errors="replace")
        (run / "checker.log").write_text(check.stdout + check.stderr, encoding="utf-8")
        matched = re.search(r"유효 결과\s+(\d+)/(\d+)", check.stdout)
        summary.update(official_checker_exit=check.returncode,
                       official_checker_valid_rows=int(matched[1]) if matched else None)
        if check.returncode == 0:
            summary.update(summarize_rows(output / "submission.jsonl", baseline / "assets/laws.jsonl"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Native exit:", process.returncode, "| elapsed seconds:", elapsed,
          "| valid output rows:", summary.get("official_checker_valid_rows"), flush=True)
    if process.returncode or summary.get("official_checker_exit") != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
