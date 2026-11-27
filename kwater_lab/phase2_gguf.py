"""Local file-only mtmd CLI image test, including rollback-free raw logs and VRAM sampling."""

import argparse
from pathlib import Path
import os
import re
import subprocess
import threading
import time
import numpy as np

from .phase2_lora import TASK, parse_codes
from .phase2_panel import PROJECT, RUN, file_hash, order, read, registry, write
from .phase2_paths import file_stem
from .submission2_output import make_row
from .visual_probe import DEFAULT_CACHE, FILES, RUNTIME


# Same explicit non-reasoning prefix for both Qwen GGUFs. mtmd supplies their own image markers.
CHATML = "{% for m in messages %}{{ '<|im_start|>' ~ m.role ~ '\\n' ~ m.content ~ '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
RESPONSE_MARKER = "KWATER_MTMD_OUTPUT_49BB82CE"


def smoke(key, count, runtime=None):
    if key not in ("9b-gguf", "27b-gguf"):
        raise ValueError("Choose a pinned GGUF bundle")
    bundle = RUN / "large-models" / key
    if not (bundle / "complete.json").is_file():
        raise ValueError(
            "Language weights and BF16 image projector must both be verified"
        )
    weights = [
        f
        for f in read(bundle / "source.json")["files"]
        if f["rfilename"].endswith(".gguf")
    ]
    language = next(f for f in weights if not f["rfilename"].startswith("mmproj"))
    projector = next(f for f in weights if f["rfilename"].startswith("mmproj"))
    exe = (
        Path(runtime)
        if runtime
        else DEFAULT_CACHE
        / RUNTIME
        / ("llama-mtmd-cli.exe" if os.name == "nt" else "llama-mtmd-cli")
    )
    out = RUN / "large-models" / (key + "-mtmd")
    out.mkdir(parents=True, exist_ok=True)
    rows = registry()
    ids = ["public-example"] + read(RUN / "plan.json")["review"][:count]
    for sid in ids:
        stem = file_stem(sid)
        dest = out / (stem + ".json")
        if dest.exists() and read(dest).get("exit_code") == 0:
            continue
        if sid == "public-example":
            image = DEFAULT_CACHE / "public-example.jpg"
            assert file_hash(image) == FILES[image.name]
        else:
            image = PROJECT / "data/train" / rows[sid]["image"]
        argv = [
            str(exe),
            "--offline",
            "--log-colors",
            "off",
            "-m",
            str(bundle / language["rfilename"]),
            "--mmproj",
            str(bundle / projector["rfilename"]),
            "--image",
            str(image),
            "-p",
            TASK,
            "--jinja",
            "--chat-template",
            CHATML,
            "-c",
            "2048",
            "-n",
            "96",
            "--image-min-tokens",
            "64",
            "--image-max-tokens",
            "128",
            "--temp",
            "0",
            "--seed",
            "42",
            "-t",
            "4",
            "-ngl",
            "99" if key == "9b-gguf" else "28",
        ]
        samples = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    q = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=10,
                        capture_output=True,
                        text=True,
                    )
                    if q.returncode == 0:
                        samples.append([time.time(), int(q.stdout.splitlines()[0])])
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                stop.wait(2)

        watcher = threading.Thread(target=monitor, daemon=True)
        start = time.perf_counter()
        watcher.start()
        response = out / (stem + ".stdout")
        logging = out / (stem + ".stderr")
        rec = {}
        try:
            with response.open("wb") as stdout, logging.open("wb") as stderr:
                result = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env={**os.environ, "MTMD_TEST_RESPONSE_MARKER": RESPONSE_MARKER},
                    timeout=7200,
                )
            rec["exit_code"] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as err:
            rec.update(exit_code=None, error=repr(err))
        finally:
            stop.set()
            watcher.join(timeout=12)
        raw = (
            response.read_bytes().decode("utf8", "replace") if response.exists() else ""
        )
        logs = (
            logging.read_bytes().decode("utf8", "replace") if logging.exists() else ""
        )
        # b12721 emits a marker at eval_message, generated tokens on stdout and timings on stderr.
        _, marker, answer = raw.replace("\r\n", "\n").rpartition(RESPONSE_MARKER + "\n")
        measured = False
        pred = []
        if rec.get("exit_code") == 0 and marker:
            try:
                pred = parse_codes(answer)
                measured = True
            except (ValueError, TypeError):
                pass
        perf = re.search(r"(?<!prompt )eval time\s*=.*?/\s*(\d+)\s+runs", logs)
        rec.update(
            sample=sid,
            argv=argv,
            runner_sha256=file_hash(exe),
            seconds=time.perf_counter() - start,
            valid=measured,
            generated_text=answer if marker else None,
            sampled_after_prefill=int(perf.group(1)) if perf else None,
            output_token_cap=96,
            explicit_think_markers=sum(
                answer.count(v) for v in ("<think>", "</think>")
            ),
            final=make_row(order("kwater/scale", sid)[:32], pred),
            source_revision=read(bundle / "source.json")["revision"],
            language=language,
            projector=projector,
            global_gpu_mib_samples=samples,
            global_gpu_mib_peak=max((v for _, v in samples), default=None),
            socket_transport=False,
            reasoning_prefix="closed empty think block",
            bf16_original_provided=False,
        )
        write(dest, rec)
        print(
            "MTMD_IMAGE",
            key,
            sid,
            measured,
            rec["seconds"],
            rec["global_gpu_mib_peak"],
            flush=True,
        )
        if rec.get("exit_code") != 0:
            raise RuntimeError(f"mtmd failed; see {logging}")
    saved = [read(out / (file_stem(s) + ".json")) for s in ids]
    dev = [v for v in saved if v["sample"] != "public-example"]
    write(
        out / "summary.json",
        dict(
            engine="b12721, one file-only native process per photo",
            photos=len(saved),
            dev_photos=len(dev),
            dev_valid=sum(v["valid"] for v in dev),
            per_photo_cold_p50_p95=np.percentile(
                [v["seconds"] for v in dev], [50, 95]
            ).tolist()
            if len(dev) >= 2
            else None,
            warm_photo_percentiles=None,
            L4_measured=False,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=("9b-gguf", "27b-gguf"))
    ap.add_argument("--dev-count", type=int, choices=range(17), default=1)
    ap.add_argument("--runtime", type=Path)
    args = ap.parse_args()
    smoke(args.model, args.dev_count, args.runtime)
