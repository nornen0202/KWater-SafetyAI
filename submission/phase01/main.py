"""Entry for the local Phase 1 bundle prepared by phase01_experiment."""

from pathlib import Path

from kwater_lab.phase01_infer import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
