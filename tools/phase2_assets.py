"""Resolve public source identities and acquire only the pinned 4B challenger."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
import sys
import time

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REPOS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-Flash-Next",
    "facebook/PE-Core-B16-224",
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "unsloth/Qwen3.8-27B-GGUF",
    "unsloth/Qwen3.5-9B-GGUF",
]


def acquire(item, dest, repo, revision, bytes_per_second=None):
    name = item["rfilename"]
    path = dest / name
    if name.startswith("."):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            partial = path.with_suffix(path.suffix + ".partial")
            if not path.exists() and (
                not partial.exists() or partial.stat().st_size != item["size"]
            ):
                off = partial.stat().st_size if partial.exists() else 0
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
                with requests.get(
                    url,
                    headers={"Range": f"bytes={off}-"} if off else {},
                    stream=True,
                    timeout=(20, 100),
                ) as response:
                    response.raise_for_status()
                    if off and response.status_code != 206:
                        raise requests.RequestException(
                            "The server did not honor the resume range"
                        )
                    append = off and response.status_code == 206
                    if append and not response.headers["Content-Range"].startswith(
                        f"bytes {off}-"
                    ):
                        raise ValueError("range mismatch")
                    with partial.open("ab" if append else "wb") as f:
                        started = time.monotonic()
                        received = 0
                        for block in response.iter_content(
                            2**18 if bytes_per_second else 2**20
                        ):
                            f.write(block)
                            if bytes_per_second:
                                received += len(block)
                                wait = received / bytes_per_second - (
                                    time.monotonic() - started
                                )
                                if wait > 0:
                                    time.sleep(wait)
            current = path if path.exists() else partial
            if current.stat().st_size != item["size"]:
                raise IOError("size mismatch")
            if item.get("lfs"):
                with current.open("rb") as f:
                    if (
                        hashlib.file_digest(f, "sha256").hexdigest()
                        != item["lfs"]["sha256"]
                    ):
                        raise ValueError("hash mismatch")
            if current != path:
                current.replace(path)
            print("verified", name, item["size"], flush=True)
            return
        except (requests.RequestException, IOError):
            if attempt == 5:
                raise
            time.sleep(2**attempt)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    dest = run / "qwen35"
    dest.mkdir(parents=True, exist_ok=True)
    sources = {}
    for repo in REPOS:
        rev = "/revision/" + REVISION if repo == REPOS[0] else ""
        r = requests.get(
            f"https://huggingface.co/api/models/{repo}{rev}?blobs=true", timeout=30
        )
        if not r.ok and repo.startswith("unsloth"):
            sources[repo] = {"http": r.status_code}
            continue
        r.raise_for_status()
        card = r.json()
        sources[repo] = {
            "revision": card["sha"],
            "license": card.get("cardData", {}).get("license"),
            "files": [
                {k: v for k, v in x.items() if k in ("rfilename", "size", "lfs")}
                for x in card["siblings"]
            ],
        }
    (run / "public-sources.json").write_text(
        json.dumps(sources, indent=2), encoding="utf8"
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda item: acquire(item, dest, REPOS[0], REVISION),
                sources[REPOS[0]]["files"],
            )
        )
