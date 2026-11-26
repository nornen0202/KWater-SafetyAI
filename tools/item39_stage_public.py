"""Stage the exact public wheels, SigLIP tokenizer and license used in the audited ZIP."""

import argparse
import hashlib
import json
from pathlib import Path

import requests


def fetch(url, path, digest):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temp.open("wb") as out:
            for block in response.iter_content(1024 * 1024):
                h.update(block)
                out.write(block)
    if h.hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[1] / "results/item39_zip_manifest.json"
    members = json.loads(reference.read_bytes())["members"]
    revision = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    for name, info in members.items():
        leaf = Path(name).name
        if name.startswith("wheelhouse/"):
            project, version = leaf.split("-")[:2]
            result = requests.get(
                f"https://pypi.org/pypi/{project}/{version}/json", timeout=20
            )
            result.raise_for_status()
            item = next(f for f in result.json()["urls"] if f["filename"] == leaf)
            if item["digests"]["sha256"] != info["sha256"]:
                raise ValueError(f"Unexpected PyPI digest for {leaf}")
            url = item["url"]
        elif name.startswith("models/sig/") and "token" in leaf:
            url = f"https://huggingface.co/google/siglip2-so400m-patch14-384/resolve/{revision}/{leaf}"
        elif name == "licenses/tokenizers/LICENSE":
            url = "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.1/LICENSE"
        else:
            continue
        fetch(url, args.stage / name, info["sha256"])
        print(info["sha256"], name)


if __name__ == "__main__":
    main()
