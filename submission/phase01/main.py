"""Entry for the local Phase 1 bundle prepared by phase01_experiment."""

from pathlib import Path
import os
import time

started = time.perf_counter()
from kwater_lab.phase01_infer import main

if __name__ == "__main__":
    if os.environ.get("KWATER_PROFILE") == "1":
        print("PROFILE_IMPORT", time.perf_counter() - started, flush=True)
    main(Path(__file__).resolve().parent)
