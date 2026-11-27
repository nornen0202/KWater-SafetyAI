"""Prepare and verify a 200-photo timing run of the unmodified official kit."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zipfile import ZipFile

from .fold0_production_compare import image_id, prepare_inputs
from .fold4_official_output_check import DEFAULT_MANIFEST, sha256_file
from .metrics import read_index
from .offline_archive_audit import run_output_checker

COUNT = 200
SEED = "kwater-public200-latency-v1"
PHASES = (
    "specialist_seconds", "vlm_load_seconds", "inspect_seconds", "margin_gate_seconds",
    "verify_generate_seconds", "postprocess_seconds", "total_seconds",
)


def panel(manifest):
    rows = [r for r in json.loads(manifest.read_bytes()) if r["split"] == "dev"]
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}/{r['sample_id']}".encode()).digest())
    seen = set()
    first, rest = [], []
    for r in ordered:
        group = r["group_id"]
        (rest if group in seen else first).append(r)
        seen.add(group)
    chosen = (first + rest)[:COUNT]
    if len(chosen) != COUNT or len({r["sample_id"] for r in chosen}) != COUNT:
        raise ValueError("Expected 200 unique development timing photos")
    return chosen


def prepare(args):
    if not args.input_dir.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Timing inputs must remain in the local data directory")
    rows = panel(args.manifest)
    prepare_inputs(rows, args.train_dir.resolve(), args.input_dir.resolve())
    return {
        "scope": "Timing-only, development images, available to final training. No accuracy estimate.",
        "photos": len(rows), "groups": len({r["group_id"] for r in rows}), "selection_seed": SEED,
        "dev_manifest_sha256": sha256_file(args.manifest),
        "input_manifest_sha256": sha256_file(args.input_dir / "manifest.jsonl"),
    }


def summarize(args):
    log = args.log.read_text(encoding="utf-8")
    input_manifest = args.input_dir / "manifest.jsonl"
    evidence = json.loads(args.panel_witness.read_bytes())
    if evidence["input_manifest_sha256"] != sha256_file(input_manifest):
        raise ValueError("Timing run input differs from the fixed panel")
    inputs = read_index(input_manifest, lambda r: (r["image_id"], None))
    output = args.output_dir / "submission.jsonl"
    outputs = read_index(output, lambda r: (r["image_id"], None), limit_size=True) if output.is_file() else {}
    elapsed = re.findall(r"종료 코드 (\d+), 실행 (\d+)초 \+ 설치 (\d+)초 = (\d+)초 / 상한 (\d+)초", log)
    checked = re.findall(r"^유효 결과\s+(\d+)/(\d+)\s*$", log, re.MULTILINE)
    hashes = re.findall(r"^ARCHIVE_SHA256 ([a-f0-9]{64})$", log, re.MULTILINE)
    results = re.findall(r"^RESULT (\w+)", log, re.MULTILINE)
    telemetry_lines = re.findall(r"^MAIN_PY_TELEMETRY (\{.*\})$", log, re.MULTILINE)
    metrics = json.loads(telemetry_lines[0]) if len(telemetry_lines) == 1 else {}
    official_clock = dict(zip(("return_code", "execution_seconds", "install_seconds", "total_seconds", "limit_seconds"),
                              map(int, elapsed[0]))) if len(elapsed) == 1 else {}
    sha = sha256_file(output) if output.is_file() else None
    runtime_phases_present = all(isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k])
                                 and metrics[k] >= 0 for k in PHASES)
    production_policy = (metrics.get("enable_hybrid_booster") is True and metrics.get("max_candidates_per_photo") == 1
                         and metrics.get("inspect_initial_batch_size") == 4
                         and metrics.get("verify_initial_batch_size") == 2)
    complete = (hashes == [args.archive_sha256] and results == ["OK"] and len(inputs) == len(outputs) == COUNT and inputs.keys() == outputs.keys()
                and checked == [(str(COUNT), str(COUNT))] and metrics.get("input_photos") == COUNT
                and metrics.get("output_submission_sha256") == sha and runtime_phases_present and production_policy
                and official_clock.get("return_code") == 0 and official_clock.get("limit_seconds") == 1800
                and official_clock["total_seconds"] == official_clock["install_seconds"] + official_clock["execution_seconds"]
                and official_clock["total_seconds"] <= 1800
                and "네트워크 차단" in log and "이 컴퓨터에서는 보안 제한 없이" not in log)
    return {
        "verified_local_200_photo_kit_run": complete,
        "hardware_observation": args.hardware, "archive_sha256": hashes[0] if len(hashes) == 1 else None,
        "kit_result": results, "kit_valid_row_report": checked,
        "input_rows": len(inputs), "output_unique_rows": len(outputs), "output_sha256": sha,
        "kit_clock": official_clock, "runtime_phases": metrics,
        "kit_log_sha256": sha256_file(args.log), "input_manifest_sha256": sha256_file(input_manifest),
    }



def check_panel_files(input_dir, manifest, witness):
    if witness["dev_manifest_sha256"] != sha256_file(manifest):
        raise ValueError("Development timing manifest differs")
    if witness["input_manifest_sha256"] != sha256_file(input_dir / "manifest.jsonl"):
        raise ValueError("Timing run input manifest differs")
    rows = panel(manifest)
    manifest_ids = read_index(input_dir / "manifest.jsonl", lambda row: (row["image_id"], None))
    if list(manifest_ids) != [image_id(row["sample_id"]) for row in rows]:
        raise ValueError("Timing run IDs do not match the ordered 200-photo panel")
    images = input_dir / "images"
    for row in rows:
        file = images / f"{image_id(row['sample_id'])}{Path(row['image']).suffix}"
        if sha256_file(file) != row["sha256"]:
            raise ValueError("Timing image bytes differ from the original development manifest")
    return len(rows)


def native(args):
    # This entry point is the actual archive execution and can use GPU. The audit commands above run on CPU.
    archive, inputs, workspace = (p.resolve() for p in (args.archive, args.input_dir, args.workspace))
    kit = args.kit.resolve()
    if not workspace.is_relative_to(Path("data").resolve()):
        raise ValueError("Native extraction must remain in the local data directory")
    with ZipFile(archive) as bundle:
        cfg = json.loads(bundle.read("pipeline_config.json"))
        main_sha = hashlib.sha256(bundle.read("main.py")).hexdigest()
    if (cfg.get("enable_hybrid_booster") is not True or not cfg.get("margin_gate")
            or (cfg.get("batch_size"), cfg.get("verify_batch_size"), cfg.get("max_candidates_per_photo")) != (4, 2, 1)):
        raise ValueError("Expected the frozen hybrid settings (4, 2, one candidate, active gate)")
    source_sha = sha256_file(archive)
    if source_sha != args.archive_sha256:
        raise ValueError("Native archive bytes differ from the frozen SHA-256")
    witness = json.loads(args.panel_witness.read_bytes())
    original_files_verified = check_panel_files(inputs, args.manifest, witness)
    workspace.mkdir(parents=True, exist_ok=False)
    code, output = workspace / "code", workspace / "output"
    code.mkdir()
    output.mkdir()
    started = time.perf_counter()
    with (workspace / "check_zip.log").open("xb") as handle:
        archive_check = subprocess.run([sys.executable, "-X", "utf8", "-I", str(kit / "check_zip.py"),
                                       str(archive), str(code)], stdout=handle, stderr=subprocess.STDOUT)
    archive_seconds = round(time.perf_counter() - started, 3)
    report = {
        "scope": "Windows/Linux native diagnostic; no offline wheel install, Linux seccomp or Public environment claim",
        "archive_sha256": source_sha, "archive_main_sha256": main_sha,
        "original_input_images_sha256_verified": original_files_verified,
        "input_manifest_sha256": witness["input_manifest_sha256"],
        "archive_check_exit": archive_check.returncode, "archive_check_extract_seconds": archive_seconds,
    }
    if archive_check.returncode:
        return {**report, "passed": False}
    if sha256_file(code / "main.py") != main_sha:
        raise ValueError("Official extraction changed the archive entrypoint")
    env = {k: v for k, v in os.environ.items() if k not in {"KW_BATCH", "KW_VERIFY_BATCH"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", KWATER_SEED="42", PYTHONHASHSEED="42")
    started = time.perf_counter()
    timeout = False
    with (workspace / "main.log").open("xb") as handle:
        try:
            run = subprocess.run([sys.executable, "-X", "utf8", str(code / "main.py"),
                                  "--input-dir", str(inputs), "--output-dir", str(output)],
                                 cwd=code, env=env, timeout=args.timeout, stdout=handle, stderr=subprocess.STDOUT)
            code_return = run.returncode
        except subprocess.TimeoutExpired:
            timeout, code_return = True, None
    main_seconds = round(time.perf_counter() - started, 3)
    log = (workspace / "main.log").read_text(encoding="utf-8", errors="replace")
    lines = re.findall(r"^MAIN_PY_TELEMETRY (\{.*\})$", log, re.MULTILINE)
    phases = json.loads(lines[0]) if len(lines) == 1 else {}
    outcome = run_output_checker(kit, inputs, output, Path(sys.executable))
    jsonl = output / "submission.jsonl"
    output_hash = sha256_file(jsonl) if jsonl.is_file() else None
    expected = read_index(inputs / "manifest.jsonl", lambda row: (row["image_id"], None))
    exact_ids = False
    if jsonl.is_file():
        try:
            actual = read_index(jsonl, lambda row: (row["image_id"], None), limit_size=True)
            exact_ids = actual.keys() == expected.keys()
        except (ValueError, TypeError):
            pass
    return {
        **report, "native_main_wall_seconds": main_seconds, "native_main_exit": code_return,
        "native_main_timeout": timeout, "official_output_check": outcome, "exact_output_ids": exact_ids, "runtime_phases": phases,
        "output_sha256": output_hash, "native_main_log_sha256": sha256_file(workspace / "main.log"),
        "passed": (code_return == 0 and exact_ids and outcome["all_rows_valid"] and outcome["valid_rows"] == COUNT
                   and phases.get("input_photos") == COUNT and phases.get("output_submission_sha256") == output_hash),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prep.add_argument("--train-dir", type=Path, default=Path("data/train"))
    prep.add_argument("--input-dir", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("summarize")
    for name in ("input-dir", "output-dir", "panel-witness", "log", "out"):
        audit.add_argument("--" + name, type=Path, required=True)
    audit.add_argument("--hardware", required=True)
    audit.add_argument("--archive-sha256", required=True)
    local = sub.add_parser("native")
    for name in ("archive", "input-dir", "workspace", "panel-witness", "out"):
        local.add_argument("--" + name, type=Path, required=True)
    local.add_argument("--archive-sha256", required=True)
    local.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    local.add_argument("--kit", type=Path, default=Path("data/local-kit/kit"))
    local.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh aggregate output")
    result = {"prepare": prepare, "summarize": summarize, "native": native}[args.step](args)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    passed = result.get("verified_local_200_photo_kit_run", result.get("passed"))
    return 0 if args.step == "prepare" or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
