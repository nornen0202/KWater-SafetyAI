"""Reserve CPU replay and the next download without importing model libraries while idle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kwater_lab.phase2_paths import file_stem


def observe():
    from kwater_lab.phase2_proposer_strata import replay
    from kwater_lab.phase2_release import run
    from kwater_lab.phase2_agreement import replay as agreement

    with ThreadPoolExecutor(3) as pool:
        observers = [pool.submit(f, True) for f in (run, replay, agreement)]
        for job in observers:
            job.result()


def reserve(root):
    ids = json.loads((root / "plan.json").read_bytes())["panel"]
    names = {file_stem(s) + ".json" for s in ids}
    waiting = {"replay", "transfer"}
    while waiting:
        starts = []
        if "replay" in waiting:
            available = names.intersection(os.listdir(root / "A"))
            if available.difference(os.listdir(root / "release-replay/fresh64")):
                starts.append(
                    ("replay", ["-m", "tools.phase2_deferred_jobs", "observe"])
                )
        if (
            "transfer" in waiting
            and (root / "large-models/sam-sequence-complete.json").is_file()
        ):
            starts.append(("transfer", ["-m", "tools.phase2_qwen_transfers"]))
        for name, args in starts:
            with (
                (root / f"deferred-{name}.out.log").open("ab") as out,
                (root / f"deferred-{name}.err.log").open("ab") as err,
            ):
                child = subprocess.Popen(
                    [sys.executable, "-B", "-X", "utf8", "-u", *args],
                    stdout=out,
                    stderr=err,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            print("START", name, child.pid, time.time(), flush=True)
            waiting.remove(name)
        if waiting:
            time.sleep(15)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["reserve", "observe"])
    p.add_argument("--run", type=Path)
    args = p.parse_args()
    if args.command == "observe":
        observe()
    else:
        root = args.run or Path(os.environ["KW_PHASE2_RUN"])
        os.environ["KW_PHASE2_RUN"] = str(root)
        reserve(root)
