"""Prepare the pinned, small Meta Python inference source and tokenizer on G:."""

import argparse
import hashlib
from pathlib import Path
import requests

from kwater_lab.phase2_panel import RUN, read, write


REVISION = "31cb7e624e4fe95301b1be1d4e809111593d2e45"


def fetch():
    dest = RUN / "large-models/sam-source"
    client = requests.Session()
    index = dest / "tree.json"
    if not index.exists():
        response = client.get(
            f"https://api.github.com/repos/facebookresearch/sam3/git/trees/{REVISION}",
            params={"recursive": "1"},
            timeout=30,
        )
        response.raise_for_status()
        tree = response.json()
        if tree["truncated"]:
            raise ValueError("Incomplete official source tree")
        write(index, tree["tree"])
    files = {}
    for item in read(index):
        relative = item["path"]
        if item["type"] != "blob" or not (
            relative
            in {"pyproject.toml", "README.md", "LICENSE", "scripts/bench_compile.py"}
            or (
                relative.startswith("sam3/")
                and Path(relative).suffix in {".py", ".gz", ".yaml", ".txt"}
            )
        ):
            continue
        path = dest / relative
        data = path.read_bytes() if path.is_file() else b""
        prefix = f"blob {item['size']}\0".encode()
        if hashlib.sha1(prefix + data).hexdigest() != item["sha"]:
            response = client.get(
                f"https://raw.githubusercontent.com/facebookresearch/sam3/{REVISION}/{relative}",
                timeout=30,
            )
            response.raise_for_status()
            data = response.content
            if (
                len(data) != item["size"]
                or hashlib.sha1(prefix + data).hexdigest() != item["sha"]
            ):
                raise ValueError(f"Source digest differs for {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        files[relative] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    write(
        dest / "verified-source.json",
        {
            "revision": REVISION,
            "bytes": sum(f["bytes"] for f in files.values()),
            "files": files,
        },
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    fetch()
