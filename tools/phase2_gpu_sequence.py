"""Schedule one GPU-owning child at a time on the 12 GB development machine."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from kwater_lab.phase2_io import RUN, read, write
from kwater_lab.phase2_paths import file_stem


def wait_official(pid):
    import psutil

    try:
        process = psutil.Process(pid)
        if "kwater_lab.phase2_official" in process.cmdline():
            process.wait()
        elif not (RUN / "a64-first-fit-handoff.json").is_file():
            raise RuntimeError(
                "Expected original official generation process before its completed receipt"
            )
    except psutil.NoSuchProcess:
        pass
    spec = read(RUN / "plan.json")
    if not all((RUN / "A" / (file_stem(s) + ".json")).is_file() for s in spec["panel"]):
        raise RuntimeError(
            "Official generation ended before all frozen panel rows; resume that run first"
        )


def source_core_complete():
    plan = read(RUN / "plan.json")
    return (RUN / "fresh-qwen-kit/receipt.json").is_file() and all(
        (RUN / arm / (file_stem(s) + ".json")).is_file()
        for arm in "BCDE"
        for s in plan["review"]
    )


def resume_official(core_only):
    prefix = "post-first-fit-core" if core_only else "auxiliary-resumed"
    wrapper = Path(__file__).with_name("run_phase2_study.ps1").resolve()
    env = dict(os.environ, KW_PHASE2_CORE_ONLY="1" if core_only else "0")
    with (
        (RUN / (prefix + ".out.log")).open("ab") as out,
        (RUN / (prefix + ".err.log")).open("ab") as err,
    ):
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "-Stage",
                "Official",
                "-Run",
                str(RUN),
            ],
            stdout=out,
            stderr=err,
            env=env,
            check=True,
        )
    if core_only and not source_core_complete():
        raise RuntimeError("Restored frozen core is incomplete")
    write(RUN / (prefix + ".json"), dict(completed_at=time.time()))


def run_meta3_if_ready():
    if not (RUN / "large-models/sam3/complete.json").is_file():
        return
    record = RUN / "regions/sam3-static/sam3-both-handoff.json"
    if record.is_file():
        return
    wrapper = Path(__file__).with_name("run_phase2_sam.ps1").resolve()
    with (
        (RUN / "sam3-both.out.log").open("ab") as stdout,
        (RUN / "sam3-both.err.log").open("ab") as stderr,
    ):
        outcome = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(wrapper),
                "-Arm",
                "sam-suite",
                "-Run",
                str(RUN),
            ],
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    write(record, dict(returncode=outcome.returncode, at=time.time()))


def launch(pid, sam_pid=None):
    wait_official(pid)
    if sam_pid:
        import psutil

        try:
            process = psutil.Process(sam_pid)
            if not any("run_phase2_sam.ps1" in a.lower() for a in process.cmdline()):
                raise RuntimeError(
                    "Expected the reserved SAM wrapper at the GPU handoff"
                )
            process.wait()
        except psutil.NoSuchProcess:
            pass
        write(
            RUN / "training-sam-handoff.json",
            dict(
                pid=sam_pid,
                at=time.time(),
                completed=(
                    RUN / "regions/sam3-static/sam31-frame/engineering.json"
                ).exists(),
            ),
        )
    import importlib.metadata

    write(
        RUN / "training-queue-ready.json",
        {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "bitsandbytes")
        },
    )

    def gpu_child(module, *args):
        started = time.time()
        with subprocess.Popen(
            [sys.executable, "-B", "-u", "-m", module, *args]
        ) as child:
            receipt = dict(
                parent=os.getpid(),
                child=child.pid,
                module=module,
                arguments=list(args),
                started=started,
            )
            dest = RUN / "gpu-children" / f"{time.time_ns()}-{child.pid}.json"
            write(dest, receipt)
            result = child.wait()
            write(dest, dict(receipt, ended=time.time(), returncode=result))
            if result:
                raise subprocess.CalledProcessError(result, child.args)

    while True:
        if (RUN / "lora-3vl/results.json").is_file() and not source_core_complete():
            resume_official(True)
            continue
        pending = [
            n
            for n in ("3vl", "35", "8b", "9b")
            if not (
                RUN
                / ("lora-" + n)
                / ("standalone-nf4-check.json" if n in ("8b", "9b") else "results.json")
            ).is_file()
        ]
        available = [
            n
            for n in pending
            if n in ("3vl", "35")
            or (RUN / "large-models" / n / "complete.json").is_file()
        ]
        if available:
            gpu_child("kwater_lab.phase2_lora", available[0])
            from kwater_lab.phase2_lora_score import compare

            compare()
            continue
        large_ready = any(
            (RUN / ("lora-" + n) / "results.json").is_file() for n in ("8b", "9b")
        )
        if large_ready and (RUN / "large-models/sam3/complete.json").is_file():
            run_meta3_if_ready()
        for quant in ("9b-gguf", "27b-gguf"):
            from kwater_lab.phase2_gguf import smoke

            dest = RUN / "large-models" / quant
            runtime = RUN / "large-models" / (quant + "-mtmd")
            if (
                (dest / "complete.json").is_file()
                and not (runtime / "summary.json").is_file()
                and not (runtime / "attempt-error.json").is_file()
            ):
                try:
                    smoke(quant, 1, None)
                except (
                    ValueError,
                    RuntimeError,
                    OSError,
                    subprocess.SubprocessError,
                ) as exc:
                    write(
                        runtime / "attempt-error.json",
                        dict(error=repr(exc), at=time.time()),
                    )
        if not (RUN / "lora-35-vision/results.json").is_file():
            gpu_child(
                "kwater_lab.phase2_lora",
                "35",
                "--series",
                "--external",
                "--yield-to-large",
            )
            continue
        if (
            large_ready
            and (RUN / "regions/sam3-static/sam3-both-handoff.json").is_file()
            and not (RUN / "auxiliary-resumed.json").is_file()
        ):
            resume_official(False)
        if (RUN / "auxiliary-resumed.json").is_file():
            from kwater_lab.phase2_verifier import condition

            if (condition() or {}).get("eligible") and not (
                RUN / "lora-3vl-verifier/results.json"
            ).is_file():
                gpu_child("kwater_lab.phase2_lora", "3vl", "--verifier")
                continue
            image_towers = RUN / "agreement-heads/image-towers"

            if (
                (RUN / "agreement-heads/shared-predictions.json").is_file()
                and (image_towers / "bf16/dino/weights.json").is_file()
                and not (image_towers / "comparison.json").is_file()
            ):
                gpu_child("tools.phase2_image_towers", "compare", "--device", "cuda")
        waiting_quant = [
            q
            for q in ("9b-gguf", "27b-gguf")
            if not (RUN / "large-models" / q / "complete.json").is_file()
        ]
        write(
            RUN / "training-queue-state.json",
            dict(pending_weights=pending, waiting_quant=waiting_quant, at=time.time()),
        )
        if not pending and not waiting_quant:
            break
        time.sleep(30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-pid", required=True, type=int)
    ap.add_argument("--sam-wrapper-pid", type=int)
    args = ap.parse_args()
    launch(args.official_pid, args.sam_wrapper_pid)
