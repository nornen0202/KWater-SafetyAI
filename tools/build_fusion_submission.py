"""Rebuild the exact November 25 fusion submission from source, pinned models and Linux wheels."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZipInfo
import zlib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission/fusion"


def entry_stream(source, item):
    # Git can normalize text newlines. Restore the recorded bytes of small source and notice files.
    if source.suffix in {".onnx", ".whl", ".npz"}:
        return source.open("rb")
    data = source.read_bytes()
    lf = data.replace(b"\r\n", b"\n")
    for candidate in (data, lf, lf.replace(b"\n", b"\r\n")):
        if len(candidate) == item["size"] and f"{zlib.crc32(candidate):08x}" == item["crc32"]:
            return io.BytesIO(candidate)
    raise ValueError(f"Source differs from the accepted revision: {source.name}")


def build(output, clip, sig, wheelhouse):
    receipt = json.loads((ROOT / "results/public_submission_002.json").read_text(encoding="utf-8"))
    layout = json.loads((SOURCE / "zip-layout.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False, suffix=".partial") as f:
        tmp = Path(f.name)
    paths = {"clip.onnx": clip, "sig.onnx": sig}
    try:
        with ZipFile(tmp, "w") as archive:
            for item in layout:
                name = item["name"]
                source = wheelhouse / Path(name).name if name.startswith("wheelhouse/") else paths.get(name, SOURCE / name)
                info = ZipInfo(name, tuple(item["date_time"]))
                for field in ("create_system", "create_version", "extract_version", "external_attr", "internal_attr"):
                    setattr(info, field, item[field])
                info.compress_type = item["method"]
                info.extra = bytes.fromhex(item["extra_hex"])
                info.comment = bytes.fromhex(item["comment_hex"])
                with entry_stream(source, item) as contents, archive.open(info, "w") as out:
                    shutil.copyfileobj(contents, out, length=2**20)
                if info.file_size != item["size"] or f"{info.CRC:08x}" != item["crc32"]:
                    raise ValueError(f"Packaged content differs: {name}")
        with tmp.open("rb") as data:
            checksum = hashlib.file_digest(data, "sha256").hexdigest()
        if checksum != receipt["sha256"] or tmp.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Archive differs from the official submission receipt: {checksum}")
        tmp.replace(output)
        print(f"{output}  SHA-256 {checksum}")
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--output", type=Path, default=ROOT / ".runtime/submission2-rebuilt.zip")
    arg.add_argument("--clip", type=Path, default=SOURCE / "clip.onnx")
    arg.add_argument("--sig", type=Path, default=SOURCE / "sig.onnx")
    arg.add_argument("--wheelhouse", type=Path, default=SOURCE / "wheelhouse")
    args = arg.parse_args()
    build(args.output, args.clip, args.sig, args.wheelhouse)
