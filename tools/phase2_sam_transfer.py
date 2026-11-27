"""Resume one authorized official Meta file at the manager's live bandwidth limit."""

import argparse
import hashlib
import json
import os
import time

import requests

from kwater_lab.phase2_io import RUN, read, write


def disk_sha256(path):
    def sequential(p, flags):
        return os.open(p, flags | getattr(os, "O_SEQUENTIAL", 0))

    digest = hashlib.sha256()
    with open(path, "rb", buffering=0, opener=sequential) as stream:
        buffer = bytearray(4 * 1024 * 1024)
        view = memoryview(buffer)
        while n := stream.readinto(buffer):
            digest.update(view[:n])
    return digest.hexdigest()


def transfer(key, requested=None):
    root = RUN / "large-models"
    asset = root / key
    if key in ("sam3", "sam31"):
        info = read(asset / "access-latest.json")
        item = info["files"][0]
        name = info["weight_name"]
        assert info["access_state"] == "approved" and name == item["rfilename"]
        assert info["weight_bytes"] == item["size"]
    else:
        if key not in ("8b", "9b", "9b-gguf", "27b-gguf") or not requested:
            raise ValueError("Unknown pinned Qwen transfer")
        info = read(asset / "source.json")
        from tools.fetch_phase2_large import SPECS

        assert (info["repo"], info["revision"]) == SPECS[key]
        item = next(
            f for f in info["files"] if f["rfilename"] == requested and "lfs" in f
        )
        name = requested
    target = asset / name
    temp = asset / (name + ".partial")
    url = f"https://huggingface.co/{info['repo']}/resolve/{info['revision']}/{name}"
    headers = {}
    if key in ("sam3", "sam31"):
        from huggingface_hub.utils import build_hf_headers

        headers = build_hf_headers(token=True)
    policy = root / "transfer-policy.json"

    def limit():
        for retry in range(10):
            try:
                p = read(policy)
                break
            except (PermissionError, FileNotFoundError, json.JSONDecodeError):
                if retry == 9:
                    raise
                time.sleep(0.15)
        if p["model"] != key or p["file"] != name or p["mib_s"] <= 0:
            raise RuntimeError("Current official transfer allowance changed")
        return float(p["mib_s"])

    limit()
    for attempt in range(6):
        offset = temp.stat().st_size if temp.exists() else 0
        if target.is_file() or offset == item["size"]:
            break
        try:
            with requests.get(
                url,
                headers={**headers, "Range": f"bytes={offset}-"},
                stream=True,
                timeout=(20, 90),
            ) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range")
                if (
                    response.status_code != 206
                    or content_range
                    != f"bytes {offset}-{item['size'] - 1}/{item['size']}"
                ):
                    raise requests.RequestException(
                        f"Unexpected official byte range {response.status_code} {content_range}"
                    )
                write(
                    (asset if key in ("sam3", "sam31") else asset / "ranges" / name)
                    / f"range-{offset}.json",
                    dict(
                        pid=os.getpid(),
                        status=206,
                        content_range=content_range,
                        start=time.time(),
                        allowance_mib_s=limit(),
                    ),
                )
                print("RANGE_START", key, offset, content_range, flush=True)
                persisted = offset
                last = time.monotonic()
                with temp.open("ab", buffering=0) as output:
                    for chunk in response.iter_content(chunk_size=512 * 1024):
                        started = time.monotonic()
                        rate = limit()
                        if not chunk:
                            continue
                        if output.write(chunk) != len(chunk):
                            raise OSError("Short checkpoint write")
                        persisted += len(chunk)
                        if time.monotonic() - last >= 30:
                            write(
                                asset / "transfer-progress.json",
                                dict(
                                    bytes=persisted,
                                    at=time.time(),
                                    pid=os.getpid(),
                                    mib_s=rate,
                                ),
                            )
                            last = time.monotonic()
                        time.sleep(
                            max(
                                0,
                                len(chunk) / (rate * 2**20)
                                - (time.monotonic() - started),
                            )
                        )
            if persisted != item["size"]:
                raise requests.RequestException(
                    "Incomplete response; resume from persisted offset"
                )
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
            continue
        break
    verified = target if target.exists() else temp
    if (
        verified.stat().st_size != item["size"]
        or disk_sha256(verified) != item["lfs"]["sha256"]
    ):
        raise ValueError(
            "Downloaded checkpoint differs from the official LFS size or digest"
        )
    if verified == temp:
        os.replace(temp, target)
    receipt = (
        asset / "complete.json"
        if key in ("sam3", "sam31")
        else asset / "transfers" / (name + ".json")
    )
    write(
        receipt,
        dict(
            revision=info["revision"],
            weight=name,
            bytes=item["size"],
            sha256=item["lfs"]["sha256"],
        ),
    )
    print("VERIFIED", key, item["size"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=("sam31", "sam3"))
    transfer(p.parse_args().model)
