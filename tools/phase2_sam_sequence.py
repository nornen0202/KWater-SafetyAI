"""Continue the authorized sequence after the first active SAM 3.1 transfer."""

import argparse
import os
import subprocess
import time

from kwater_lab.phase2_io import RUN, read, write
from tools.phase2_sam_transfer import transfer


def run(previous):
    root = RUN / "large-models"
    policy = root / "transfer-policy.json"
    if read(root / "sam31/transfer-progress.json")["pid"] != previous:
        raise RuntimeError("Expected the original SAM 3.1 transfer's measured PID")
    if os.name == "nt":
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Wait-Process -Id {int(previous)} -ErrorAction SilentlyContinue",
            ],
            check=False,
        )
    else:
        while os.path.isdir(f"/proc/{previous}"):
            time.sleep(5)
    if not (root / "sam31/complete.json").is_file():
        transfer("sam31")
    steps = [
        ("sam3", "sam3.pt"),
        ("8b", "model-00002-of-00004.safetensors"),
        ("8b", "model-00001-of-00004.safetensors"),
    ]
    for key, name in steps:
        if (
            key == "8b"
            and not (root / key / (name + ".partial")).is_file()
            and not (root / key / name).is_file()
        ):
            raise RuntimeError("The authorized existing partial is missing")
        current = read(policy)
        if (
            current.get("model") not in ("sam31", "sam3", "8b")
            or not current.get("mib_s", 0) > 0
        ):
            raise RuntimeError("No live network allowance")
        if current.get("next") != ["sam3", "8b"]:
            raise RuntimeError("The manager changed the ordered transfer scope")
        current.update(model=key, file=name, switched_at=time.time())
        write(policy, current)
        print("NEXT_RANGE", key, name, current["mib_s"], flush=True)
        transfer(key, name)
    write(root / "sam-sequence-complete.json", dict(at=time.time(), completed=steps))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", required=True, type=int)
    run(p.parse_args().previous)
