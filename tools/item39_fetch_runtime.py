import hashlib
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import requests

path = pathlib.Path(sys.argv[1])
path.mkdir(exist_ok=True, parents=True)
repo = "jiwoochris/kwater-runtime"
pin = "sha256:2e924d941c41876e9eb1de79a1af7eaf796aacb1210fb86f6ff3345cc7c58d64"


def get(endpoint, stream=False, headers=None):
    auth = requests.get(
        "https://ghcr.io/token",
        params={"scope": "repository:" + repo + ":pull"},
        timeout=40,
    )
    auth.raise_for_status()
    r = requests.get(
        "https://ghcr.io/v2/" + repo + "/" + endpoint,
        headers={
            "Authorization": "Bearer " + auth.json()["token"],
            "Accept": "application/vnd.oci.image.manifest.v1+json",
            **(headers or {}),
        },
        params={"t": time.time_ns()},
        stream=stream,
        timeout=(30, 120),
    )
    r.raise_for_status()
    return r


m = get("manifests/" + pin).content
assert hashlib.sha256(m).hexdigest() == pin[7:]
(path / "manifest.json").write_bytes(m)
manifest = json.loads(m)


def fetch(layer):
    digest = layer["digest"][7:]
    dest = path / digest
    if dest.exists():
        with dest.open("rb") as f:
            valid = (
                dest.stat().st_size == layer["size"]
                and hashlib.file_digest(f, "sha256").hexdigest() == digest
            )
        if valid:
            return
        dest.replace(dest.with_name(dest.name + ".invalid-" + str(time.time_ns())))
    part = dest.with_suffix(".part")
    for attempt in range(10):
        try:
            while (start := part.stat().st_size if part.exists() else 0) < layer[
                "size"
            ]:
                end = min(start + 128 * 2**20, layer["size"]) - 1
                with get(
                    "blobs/sha256:" + digest, True, {"Range": f"bytes={start}-{end}"}
                ) as r:
                    if r.status_code == 206:
                        assert (
                            r.headers["Content-Range"]
                            == f"bytes {start}-{end}/{layer['size']}"
                        )
                    elif start != 0:
                        part.replace(
                            part.with_name(
                                part.name + ".no-range-" + str(time.time_ns())
                            )
                        )
                        raise ValueError("registry ignored resume range")
                    with part.open("ab") as f:
                        for buf in r.iter_content(2**20):
                            f.write(buf)
            with part.open("rb") as f:
                valid = hashlib.file_digest(f, "sha256").hexdigest() == digest
            if not valid:
                part.replace(
                    part.with_name(part.name + ".invalid-" + str(time.time_ns()))
                )
                raise ValueError("content digest mismatch")
            part.replace(dest)
            print(digest, dest.stat().st_size, flush=True)
            return
        except (requests.RequestException, OSError, ValueError) as err:
            print("retry", digest, type(err).__name__, flush=True)
            if attempt == 9:
                raise


print("bytes", sum(x["size"] for x in manifest["layers"]), flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, [manifest["config"], *manifest["layers"]]))
