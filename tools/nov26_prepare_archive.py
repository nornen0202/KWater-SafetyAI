"""Copy the scored baseline to an unpublished staging archive, reserving main.py."""

import argparse
import copy
import shutil
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("staging", type=Path)
    args = parser.parse_args()
    with ZipFile(args.baseline) as source, ZipFile(args.staging, "x") as dest:
        for entry in source.infolist():
            if entry.filename in {"main.py", "assets.json"}:
                continue
            with source.open(entry) as src, dest.open(copy.copy(entry), "w") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)


if __name__ == "__main__":
    main()
