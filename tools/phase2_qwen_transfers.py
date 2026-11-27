"""Continue one verified public Qwen transfer at a time after the two retained 8B partials."""

import hashlib
import time

from kwater_lab.phase2_io import RUN, read, write
from tools.fetch_phase2_large import SPECS
from tools.phase2_assets import acquire
from tools.phase2_sam_transfer import transfer


ROOT = RUN / "large-models"
ORDER = ("8b", "9b-gguf", "9b", "27b-gguf")


def source_receipt(key, item):
    folder = ROOT / key
    name = item["rfilename"]
    path = folder / name
    rec = folder / "transfers" / (name + ".json")
    if not path.is_file() or path.stat().st_size != item["size"] or not rec.is_file():
        return None
    r = read(rec)
    if (
        r.get("revision") != SPECS[key][1]
        or r.get("bytes") != item["size"]
        or r.get("weight") != name
    ):
        return None
    if (
        r.get("sha256") != item["lfs"]["sha256"]
        or rec.stat().st_mtime_ns < path.stat().st_mtime_ns
    ):
        return None
    return r


def run():
    while not (ROOT / "sam-sequence-complete.json").is_file():
        time.sleep(15)
    prior = read(ROOT / "sam-sequence-complete.json")["completed"]
    assert [key for key, name in prior] == ["sam3", "8b", "8b"]
    policy_path = ROOT / "transfer-policy.json"
    for key in ORDER:
        dest = ROOT / key
        info = read(dest / "source.json")
        assert (info["repo"], info["revision"]) == SPECS[key]
        items = sorted(
            info["files"], key=lambda f: (bool(f.get("lfs")), f["rfilename"])
        )
        verified = {}
        for item in items:
            name = item["rfilename"]
            policy = read(policy_path)
            if (
                policy.get("after_retained_partials") != list(ORDER)
                or not 0 < policy["mib_s"] <= 6
            ):
                raise RuntimeError("Public transfer order or allowed bandwidth changed")
            policy.update(model=key, file=name, switched_at=time.time())
            write(policy_path, policy)
            if "lfs" in item:
                receipt = source_receipt(key, item)
                if receipt is None:
                    transfer(key, name)
                    receipt = source_receipt(key, item)
                assert receipt
                verified[name] = dict(bytes=item["size"], sha256=receipt["sha256"])
            else:
                # Small git blobs use the publisher's git object identity alongside a local SHA-256.
                if not (dest / name).is_file():
                    acquire(
                        item,
                        dest,
                        info["repo"],
                        info["revision"],
                        bytes_per_second=policy["mib_s"] * 2**20,
                    )
                data = (dest / name).read_bytes()
                object_id = hashlib.sha1(
                    f"blob {len(data)}\0".encode() + data
                ).hexdigest()
                if len(data) != item["size"] or object_id != item["blobId"]:
                    raise ValueError(f"Publisher git blob mismatch: {key}/{name}")
                verified[name] = dict(
                    bytes=len(data),
                    git_blob=object_id,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
        write(
            dest / "complete.json",
            dict(
                revision=info["revision"],
                verified_files=len(verified),
                bytes=sum(x["bytes"] for x in verified.values()),
                files=verified,
                at=time.time(),
            ),
        )
        print("QWEN_COMPLETE", key, len(verified), flush=True)
    write(ROOT / "qwen-sequence-complete.json", dict(order=ORDER, at=time.time()))


if __name__ == "__main__":
    run()
