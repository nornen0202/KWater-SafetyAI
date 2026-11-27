"""Acquire pinned public backbones. No participant data is opened or transmitted."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time

import requests


MODELS = {
    "sig": (
        "google/siglip2-so400m-patch14-384",
        "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        "9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2",
    ),
    "dino": (
        "facebook/dinov2-large",
        "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    ),
}


def acquire(folder, key):
    repo, revision, digest = MODELS[key]
    dest = folder / key
    dest.mkdir(exist_ok=True, parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        target = dest / name
        partial = target.with_suffix(target.suffix + ".partial")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        if target.exists():
            with target.open("rb") as stream:
                if (
                    name == "model.safetensors"
                    and hashlib.file_digest(stream, "sha256").hexdigest() != digest
                ):
                    raise ValueError(f"Unexpected cached model {target}")
            continue
        for attempt in range(5):
            offset = partial.stat().st_size if partial.is_file() else 0
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    if (
                        response.status_code == 416
                        and offset
                        and response.headers.get("Content-Range") == f"bytes */{offset}"
                        and name == "model.safetensors"
                    ):
                        break  # Complete partial file. Its digest is checked before promotion.
                    response.raise_for_status()
                    append = offset and response.status_code == 206
                    if append and not response.headers.get(
                        "Content-Range", ""
                    ).startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected range")
                    with partial.open("ab" if append else "wb") as out:
                        for chunk in response.iter_content(2**20):
                            out.write(chunk)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        if name == "model.safetensors":
            with partial.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Weight digest mismatch {target}")
        partial.replace(target)
        print(key, name, target.stat().st_size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda key: acquire(args.directory, key), MODELS))
