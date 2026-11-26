"""Benchmark the packaged entry and official checker inside a network-isolated Linux runtime."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time

SIZES = (
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (960, 1280),
    (4032, 3024),
    (2048, 1536),
    (720, 1280),
    (1024, 768),
)


def synthetic(root, count, seed):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    images = root / "images"
    images.mkdir(parents=True)
    digests = set()
    with (
        (root / "manifest.jsonl").open("w") as manifest,
        (root.parent / "synthetic-provenance.csv").open("w", newline="") as f,
    ):
        writer = csv.writer(f)
        writer.writerow(["image_id", "width", "height", "jpeg_sha256"])
        for i in range(count):
            w, h = SIZES[i % len(SIZES)]
            small = Image.frombytes("RGB", (96, 72), rng.randbytes(96 * 72 * 3))
            img = small.resize((w, h), Image.Resampling.BILINEAR)
            d = ImageDraw.Draw(img)
            for _ in range(23):
                x, y = rng.randrange(w), rng.randrange(h)
                x2, y2 = (
                    x + rng.randrange(1, max(2, w // 3)),
                    y + rng.randrange(1, max(2, h // 3)),
                )
                color = tuple(rng.randrange(256) for _ in range(3))
                if rng.randrange(2):
                    d.rectangle((x, y, x2, y2), fill=color)
                else:
                    d.line((x, y, x2, y2), fill=color, width=rng.randrange(1, 12))
            sid = hashlib.sha256(f"kwater39-synthetic-{seed}-{i}".encode()).hexdigest()[
                :32
            ]
            p = images / (sid + ".jpg")
            img.save(p, quality=rng.randrange(74, 94), optimize=False)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest not in digests
            digests.add(digest)
            manifest.write(json.dumps({"image_id": sid}) + "\n")
            writer.writerow((sid, w, h, digest))


def timed(cmd, log, env, cwd=None):
    start = time.monotonic()
    p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    log.flush()
    if p.returncode:
        raise RuntimeError(
            f"{cmd[0]} failed with status {p.returncode}; see {log.name}"
        )
    return time.monotonic() - start


def run(a):
    job = a.out
    job.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        HOME=str(job / "cache"),
        TMPDIR=str(job / "scratch"),
        HF_HOME=str(job / "cache/hf"),
        XDG_CACHE_HOME=str(job / "cache/xdg"),
        PIP_NO_INDEX="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        KWATER_PROFILE="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        KWATER_SEED="20261126",
        PYTHONHASHSEED="20261126",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    for p in ("cache", "scratch", "output"):
        (job / p).mkdir()
    assert not any((job / "cache").iterdir()) and os.geteuid() != 0
    clocks = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": platform.machine(),
        "kernel": platform.release(),
        "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)),
        "meminfo": Path("/proc/meminfo").read_text().splitlines()[:3],
        "netdev": Path("/proc/net/dev").read_text(),
        "network_namespace": os.readlink("/proc/self/ns/net"),
        "uid": os.geteuid(),
        "input_count": a.n,
        "seed": a.seed,
        "library_base": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "tokenizers",
                "huggingface-hub",
                "accelerate",
                "numpy",
                "Pillow",
                "safetensors",
            )
        },
    }
    assert platform.machine() == "x86_64"
    assert {
        line.split(":")[0].strip() for line in clocks["netdev"].splitlines()[2:]
    } <= {"lo"}
    in_dir = job / "input"
    synthetic(in_dir, a.n, a.seed)
    artifacts = a.verified_code or job / "code"
    if a.verified_code:
        assert os.statvfs(artifacts).f_flag & os.ST_RDONLY
        (job / "zip-check.log").write_text(
            "Reused read-only official-checker extraction from the cold smoke.\n"
        )
        clocks["official_extract_check_s"] = None
    else:
        with (job / "zip-check.log").open("w") as log:
            clocks["official_extract_check_s"] = timed(
                [
                    sys.executable,
                    "-I",
                    str(a.kit / "check_zip.py"),
                    str(a.zip),
                    str(artifacts),
                ],
                log,
                env,
            )
    # The official extraction has just written the weights. Release the guest's clean
    # file-cache pages before timing a first model load in this fresh process/venv.
    os.sync()
    for weight in artifacts.rglob("*.safetensors"):
        with weight.open("rb") as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    clocks["guest_weights_fadvise_dontneed"] = True
    venv = job / "venv"
    py = venv / "bin/python"
    with (job / "install.log").open("w") as log:
        print("BENCH_EVENT", a.n, "install_start", flush=True)
        t = time.monotonic()
        clocks["venv_s"] = timed(
            [
                sys.executable,
                "-m",
                "venv",
                "--system-site-packages",
                "--without-pip",
                str(venv),
            ],
            log,
            env,
        )
        clocks["pip_s"] = timed(
            [
                sys.executable,
                "-m",
                "pip",
                "--python",
                str(py),
                "install",
                "--no-index",
                "--find-links",
                str(artifacts / "wheelhouse"),
                "--only-binary=:all:",
                "--require-hashes",
                "--no-cache-dir",
                "-r",
                str(artifacts / "requirements.lock"),
            ],
            log,
            env,
        )
        clocks["install_s"] = time.monotonic() - t
        print("BENCH_EVENT", a.n, "install_end", flush=True)
    cpu_list = ",".join(map(str, clocks["affinity"]))
    cmd = [
        sys.executable,
        "-I",
        str(a.kit / "sandbox_exec.py"),
        "--mode",
        "seccomp",
        "--cpus",
        cpu_list,
        "--",
        str(py),
        "main.py",
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(job / "output"),
    ]
    phases = {}
    photos = []
    print("BENCH_EVENT", a.n, "main_start", flush=True)
    t = time.monotonic()
    last_close = None
    with (job / "main.log").open("w") as log:
        with subprocess.Popen(
            cmd,
            cwd=artifacts,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf8",
            errors="replace",
        ) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                tag, _, s = line.rstrip().partition(" ")
                if tag == "PROFILE_IMPORT":
                    phases["entry_import_s"] = float(s)
                    phases["startup_until_import_s"] = time.monotonic() - t
                elif tag == "PROFILE_LOAD":
                    phases.update(json.loads(s))
                elif tag == "loaded" and s.startswith(("sig ", "dino ")):
                    name, seconds = s.split()
                    phases[name + "_load_s_rounded"] = float(seconds)
                elif tag == "PROFILE_PHOTO":
                    photos.append(json.loads(s))
                elif tag == "PROFILE_CLOSE":
                    phases["output_close_rename_s"] = float(s)
                    last_close = time.monotonic()
            rc = proc.wait()
    end = time.monotonic()
    clocks["main_process_s"] = end - t
    print("BENCH_EVENT", a.n, "main_exit", flush=True)
    if rc or len(photos) != a.n or last_close is None:
        raise RuntimeError(f"main rc={rc} rows={len(photos)}; see main.log")
    phases["after_close_until_exit_s"] = end - last_close
    phases["vision_import_and_setup_residual_s"] = (
        phases["backbones_s"]
        - phases["sig_load_s_rounded"]
        - phases["dino_load_s_rounded"]
    )
    with (job / "photos.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=photos[0].keys())
        w.writeheader()
        w.writerows(photos)
    rest = [p["total_s"] for p in photos[1:]]
    phases.update(
        first_photo_s=photos[0]["total_s"],
        subsequent_photos_s=sum(rest),
        rest_p50_s=statistics.median(rest) if rest else None,
        rest_p95_s=statistics.quantiles(rest, n=100, method="inclusive")[94]
        if len(rest) > 1
        else (rest or [None])[0],
        all_photo_s=sum(p["total_s"] for p in photos),
    )
    phases["all_preprocess_s"] = sum(p["open_s"] + p["preprocess_s"] for p in photos)
    phases["all_writes_s"] = (
        sum(p["write_s"] for p in photos) + phases["output_close_rename_s"]
    )
    with (job / "output-check.log").open("w") as log:
        clocks["checker_s"] = timed(
            [
                sys.executable,
                "-I",
                str(a.kit / "check_output.py"),
                str(in_dir),
                str(job / "output"),
            ],
            log,
            env,
        )
    assert f"{a.n}/{a.n}" in (job / "output-check.log").read_text()
    clocks.update(phases)
    clocks["install_plus_process_s"] = clocks["install_s"] + clocks["main_process_s"]
    clocks["max_child_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    probe = (
        "import json,sys,torch; from importlib.metadata import version; "
        "print(json.dumps(dict(versions={p:version(p) for p in sys.argv[1:]}, "
        "cuda_available=torch.cuda.is_available(), "
        "gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "vram_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None)))"
    )
    clocks["post_run_environment"] = json.loads(
        subprocess.check_output(
            [str(py), "-c", probe, *clocks["library_base"]], env=env, text=True
        )
    )
    (job / "timing.json").write_text(json.dumps(clocks, indent=2))
    print(json.dumps(clocks), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for name in ("zip", "kit", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--verified-code", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    run(ap.parse_args())
