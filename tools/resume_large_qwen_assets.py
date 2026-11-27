"""Resume the two pinned model downloads one file at a time, then verify their digests."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1] / "data/phase2-runs/p38-20261126-49bb82ce/large-models"


def receive(folder, source, entry):
    name = entry["rfilename"]
    if Path(name).name != name:
        raise ValueError(name)
    dest = folder / name
    part = dest.with_name(name + ".partial")
    target = dest if dest.exists() else part
    size = entry["size"]
    if target == part and (not part.exists() or part.stat().st_size < size):
        url = f'https://huggingface.co/{source["repo"]}/resolve/{source["revision"]}/{quote(name)}?download=true'
        print("Resuming", part, flush=True)
        subprocess.run(["curl.exe", "-fLsS", "--retry", "4", "--limit-rate", "3M", "--connect-timeout", "30",
                        "--speed-time", "90", "--speed-limit", "1024", "-C", "-", "-o", str(part), url], check=True)
    if target.stat().st_size != size:
        raise ValueError(f"Size mismatch: {target}")
    lfs = entry.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1(f"blob {size}\0".encode())
    with target.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            time.sleep(0.02)
    if digest.hexdigest() != (lfs["sha256"] if lfs else entry["blobId"]):
        raise ValueError(f"Digest mismatch: {target}")
    if target == part:
        part.rename(dest)
    print("Verified", name, size, digest.hexdigest(), flush=True)


if __name__ == "__main__":
    for model in ("27b-gguf", "8b"):
        folder = ROOT / model
        source = json.loads((folder / "source.json").read_text(encoding="utf-8-sig"))
        for item in sorted(source["files"], key=lambda entry: entry["size"]):
            receive(folder, source, item)
