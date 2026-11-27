"""Prepare the Windows Triton kernel wheel for a local torch 2.5 compatibility run."""

import hashlib
import os
import time
from zipfile import ZipFile
import requests

from kwater_lab.phase2_panel import RUN, write


def main():
    dest = RUN / "large-models/sam-wheels"
    dest.mkdir(parents=True, exist_ok=True)
    package, version = "triton-windows", "3.1.0.post17"
    response = requests.get(
        f"https://pypi.org/pypi/{package}/{version}/json", timeout=20
    )
    response.raise_for_status()
    item = next(
        v
        for v in response.json()["urls"]
        if v["filename"].endswith("cp312-cp312-win_amd64.whl")
    )
    wheel = dest / item["filename"]
    partial = wheel.with_suffix(".partial")
    if not wheel.is_file():
        off = partial.stat().st_size if partial.exists() else 0
        with requests.get(
            item["url"], headers={"Range": f"bytes={off}-"}, stream=True, timeout=60
        ) as resp:
            resp.raise_for_status()
            if off and (
                resp.status_code != 206
                or not resp.headers.get("Content-Range", "").startswith(f"bytes {off}-")
            ):
                raise ValueError("Triton wheel range did not match")
            with partial.open("ab") as output:
                for chunk in resp.iter_content(128 * 1024):
                    start = time.monotonic()
                    output.write(chunk)
                    time.sleep(
                        max(0, len(chunk) / (0.25 * 2**20) - (time.monotonic() - start))
                    )
    local = wheel if wheel.exists() else partial
    with local.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if local.stat().st_size != item["size"] or digest != item["digests"]["sha256"]:
        raise ValueError("Triton wheel differs from PyPI")
    if local == partial:
        os.replace(partial, wheel)
    with ZipFile(wheel) as archive:
        archive.extractall(RUN / "sam-kernel")
    write(
        dest / "runtime.json",
        dict(
            package=package,
            version=version,
            bytes=item["size"],
            sha256=item["digests"]["sha256"],
            url=item["url"],
            purpose="local Windows adaptation of Meta's CUDA kernels; official torch >=2.7 target separately recorded",
        ),
    )


if __name__ == "__main__":
    main()
