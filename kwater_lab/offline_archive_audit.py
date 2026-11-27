"""Read-only K-water bundle audit. No candidate imports or extract of large members."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


COMPETITION_LIMIT = 20_000_000_000
MEDIA = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".tiff", ".bmp"}
LOCK_LINE = re.compile(r"([A-Za-z0-9_][\w.-]*)==([^\s;]+)((?:\s+--hash=sha256:[0-9a-fA-F]{64})+)")


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def entries(zip_file):
    paths = {}
    for info in zip_file.infolist():
        raw = info.orig_filename
        is_dir = info.is_dir()
        name = raw[:-1] if is_dir else raw
        kind = stat.S_IFMT(info.external_attr >> 16)
        if (info.flag_bits & 1 or re.search(r"[\\\x00]|^[A-Za-z]:/", raw)
            or any(part in ("", ".", "..") for part in name.split("/"))
            or kind not in (0, stat.S_IFDIR if is_dir else stat.S_IFREG)
            or name in paths):
            raise ValueError("Unsafe, encrypted or duplicate ZIP member path")
        paths[name] = None if is_dir else info.file_size
    for name in paths:
        if any(paths.get(parent.as_posix(), None) is not None for parent in Path(name).parents):
            raise ValueError("ZIP file and directory paths collide")
    return {name: size for name, size in paths.items() if size is not None}


def stage_entries(stage):
    paths = {}
    for path in stage.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Staged links must not redirect reads outside the bundle")
        if path.is_file():
            paths[path.relative_to(stage).as_posix()] = path.stat().st_size
    return paths


def official_manifest(kit, scratch, files, open_file):
    spec = importlib.util.spec_from_file_location("kwater_kit_zipcheck", kit / "zipcheck.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TemporaryDirectory(dir=scratch) as folder:
        stage = Path(folder)
        for name in ("assets.json", "requirements.lock"):
            if name in files:
                if files[name] > 1_048_576:
                    raise ValueError(f"Manifest or lock file is unexpectedly large: {name}")
                with open_file(name) as stream:
                    (stage / name).write_bytes(stream.read())
        if any(k.startswith("wheelhouse/") for k in files):
            (stage / "wheelhouse").mkdir()
        return module.manifest(stage)


def linux312_tag(filename):
    interpreter, abi, platform = filename.removesuffix(".whl").rsplit("-", 3)[-3:]
    for py in interpreter.split("."):
        if py in {"py3", "py312", "cp312"} or (abi == "abi3" and re.fullmatch(r"cp3\d+", py) and 2 <= int(py[3:]) <= 12):
            if platform == "any" or any(p.startswith(("manylinux", "linux")) and p.endswith("x86_64") for p in platform.split(".")):
                return True
    return False


def wheel_audit(files, open_file):
    if "requirements.lock" not in files:
        return []
    with open_file("requirements.lock") as stream:
        raw = stream.read().decode("utf-8")
    wheels = [name for name in files if name.startswith("wheelhouse/") and name.count("/") == 1 and name.endswith(".whl")]
    normalized = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    results = []
    for line in raw.splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        matched = LOCK_LINE.fullmatch(line)
        if not matched:
            raise ValueError("Lock requires pinned hashed offline entries")
        package, version, hashes = matched.groups()
        expected = {h.lower() for h in re.findall(r"--hash=sha256:([0-9a-fA-F]{64})", hashes)}
        verified = []
        for name in wheels:
            leaf = Path(name).name
            dist, wheel_version = leaf.split("-")[:2]
            if normalized(dist) != normalized(package) or wheel_version != version or not linux312_tag(leaf):
                continue
            with open_file(name) as stream:
                sha = digest(stream)
            if sha in expected:
                verified.append({"filename": leaf, "sha256": sha})
        if not verified:
            raise ValueError(f"No hash-matching Linux x86-64 Python 3.12 wheel for {package}=={version}")
        results.append({"package": package, "version": version, "matching_wheels": verified})
    return results


def static_indicators(files, open_file):
    signals = []
    for path, length in files.items():
        if not path.endswith(".py") or length > 2_097_152:
            continue
        with open_file(path) as source:
            tree = ast.parse(source.read(), filename=path, feature_version=(3, 12))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name.startswith(("requests.", "httpx.", "socket.", "urllib.request.", "subprocess.")) or name == "os.system":
                signals.append({"member": path, "line": node.lineno, "call": name})
    return signals


def inspect_bundle(files, open_file, kit, scratch, expected, restricted_sources=()):
    if not {"main.py", "assets.json"} <= files.keys() or sum(files.values()) > 64 * 1024**3:
        raise ValueError("Top-level entrypoint, assets or official unpacked size limit failed")
    seed, locked = official_manifest(kit, scratch, files, open_file)
    with open_file("main.py") as stream:
        compile(stream.read(), "main.py", "exec")
    with open_file("assets.json") as stream:
        assets = json.load(stream)
    hashes = {}
    for name in ("main.py", "assets.json", *expected):
        with open_file(name) as stream:
            hashes[name] = digest(stream)
        if name in expected and hashes[name] != expected[name]:
            raise ValueError(f"Expected asset SHA-256 does not match: {name}")
    adapter = "adapter/adapter_model.safetensors"
    declared = [entry["revision"].lower().removeprefix("sha256:") for entry in assets["assets"]
                if entry["kind"] == "model" and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", entry["revision"])
                and "adapter" in entry["name"].lower()]
    if declared:
        with open_file(adapter) as stream:
            hashes[adapter] = digest(stream)
        if hashes[adapter] not in declared:
            raise ValueError("Adapter bytes differ from the SHA-256 declared in assets.json")
    forbidden = [name for name in files if Path(name).suffix.lower() in MEDIA
                 or any(part.lower() in {"__pycache__", "sample-input", "raw-batches", "images", "logs"} for part in name.split("/"))
                 or Path(name).name in {"training_state.pt", "train.zip", "train.jsonl", "submission.jsonl", "steps.jsonl"}
                 or re.search(r"\.(?:log|stderr|stdout)(?:\.txt)?$", name, re.IGNORECASE)]
    byte_matches = set()
    for source in restricted_sources:
        candidates = [name for name, size in files.items() if size == source.stat().st_size]
        if not candidates:
            continue
        with source.open("rb") as stream:
            source_sha = digest(stream)
        for name in candidates:
            with open_file(name) as stream:
                if digest(stream) == source_sha:
                    byte_matches.add(name)
    return {"files": len(files), "unpacked_bytes": sum(files.values()), "seed": seed, "offline_lock": locked,
            "asset_sha256": hashes, "declared_asset_count": len(assets["assets"]),
            "offline_wheels": wheel_audit(files, open_file), "restricted_or_unneeded_members": forbidden,
            "restricted_source_exact_content_matches": sorted(byte_matches),
            "network_or_process_static_indicators": static_indicators(files, open_file),
            "linux_seccomp_executed": False, "offline_install_executed": False}


def run_output_checker(kit, inputs, output, executable):
    check = subprocess.run([str(executable), "-X", "utf8", str(kit / "check_output.py"), str(inputs), str(output)],
                           capture_output=True, text=True, encoding="utf-8")
    counted = re.search(r"유효 결과\s+(\d+)/(\d+)", check.stdout + check.stderr)
    valid, total = (map(int, counted.groups()) if counted else (None, None))
    missing = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", check.stderr)
    return {"return_code": check.returncode, "valid_rows": valid, "total_rows": total,
            "checker_dependency_missing": missing[1] if missing else None,
            "all_rows_valid": check.returncode == 0 and valid == total and total is not None and total > 0}


def log_summary(path):
    longest = count = large = running = total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(65_536):
            total += len(chunk)
            *lines, remainder = chunk.split(b"\n")
            for line in lines:
                length = running + len(line) + 1
                longest = max(longest, length)
                count += 1
                large += length > 4096
                running = 0
            running += len(remainder)
    return {"bytes": total, "lines": count + bool(running), "longest_line_bytes": max(longest, running),
            "lines_above_4096_bytes": large + (running > 4096)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", type=Path)
    mode.add_argument("--stage", type=Path)
    parser.add_argument("--kit", type=Path, default=Path("data/local-kit/kit"))
    parser.add_argument("--scratch", type=Path, default=Path(".runtime/temp"))
    parser.add_argument("--stream-crc", action="store_true")
    parser.add_argument("--expected-sha", action="append", default=[])
    parser.add_argument("--restricted-source", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--existing-output", type=Path)
    parser.add_argument("--checker-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    expected = dict(item.rsplit("=", 1) for item in args.expected_sha)
    if bool(args.input_dir) != bool(args.existing_output) or (args.stream_crc and not args.archive):
        parser.error("Use input/output directories together; full CRC requires an archive")
    args.scratch.mkdir(parents=True, exist_ok=True)
    sources = [Path("data/train/train.jsonl"), Path("data/packages/kwater-train.zip"), *args.restricted_source]
    sources = [path for path in sources if path.is_file() or path in args.restricted_source]
    if args.archive:
        size = args.archive.stat().st_size
        if size > COMPETITION_LIMIT:
            raise ValueError("Competition archive exceeds 20 GB")
        with ZipFile(args.archive) as archive:
            report = inspect_bundle(entries(archive), archive.open, args.kit, args.scratch, expected, sources)
            if args.stream_crc and archive.testzip() is not None:
                raise ValueError("ZIP payload CRC verification failed")
        report.update(archive_bytes=size, stream_crc_checked=args.stream_crc)
    else:
        report = inspect_bundle(stage_entries(args.stage), lambda name: (args.stage / name).open("rb"),
                                args.kit, args.scratch, expected, sources)
    report["official_full_zip_extraction_executed"] = False
    report["official_existing_output_check"] = (run_output_checker(args.kit, args.input_dir, args.existing_output, args.checker_python)
                                                 if args.existing_output else None)
    if args.runtime_log:
        report["existing_runtime_log"] = log_summary(args.runtime_log)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
