"""Read GGUF v3 metadata and tensor descriptors from the retained local byte ranges."""

from collections import Counter
import hashlib
import struct

from kwater_lab.phase2_panel import RUN, read, write


FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def number(stream, fmt):
    return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))[0]


def string(stream, retain):
    size = number(stream, "Q")
    if retain and size <= 8192:
        return stream.read(size).decode("utf8")
    stream.seek(size, 1)


def value(stream, kind, retain):
    if kind == 8:
        return string(stream, retain)
    if kind == 9:
        element = number(stream, "I")
        count = number(stream, "Q")
        if element in FORMATS:
            stream.seek(count * struct.calcsize("<" + FORMATS[element]), 1)
        elif element == 8:
            for _ in range(count):
                string(stream, False)
        else:
            raise ValueError("Invalid GGUF array element type")
        return {"elements": count, "type": element}
    return number(stream, FORMATS[kind])


def inspect(path, expected):
    allowed = {
        "general.architecture",
        "general.name",
        "general.type",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "clip.projector_type",
        "clip.has_vision_encoder",
    }
    metadata = {}
    with path.open("rb", buffering=1024 * 1024) as f:
        magic, version, tensors, fields = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF" or version != 3:
            raise ValueError("Expected pinned GGUF v3")
        for _ in range(fields):
            name = string(f, True)
            retain = name in allowed or name.endswith(
                (
                    ".block_count",
                    ".embedding_length",
                    ".image_size",
                    ".projection_dim",
                    ".patch_size",
                )
            )
            observed = value(f, number(f, "I"), retain)
            if retain:
                metadata[name] = observed
        types = Counter()
        for _ in range(tensors):
            string(f, False)
            dims = number(f, "I")
            f.seek(8 * dims, 1)
            types[number(f, "I")] += 1
            f.seek(8, 1)
        end = f.tell()
        if end > path.stat().st_size:
            raise ValueError("Truncated tensor descriptor table")
        f.seek(0)
        digest = hashlib.sha256()
        remaining = end
        while remaining:
            chunk = f.read(min(remaining, 256 * 1024))
            if not chunk:
                raise ValueError("Incomplete GGUF header")
            digest.update(chunk)
            remaining -= len(chunk)
    return dict(
        gguf_version=version,
        metadata_fields=fields,
        tensors=tensors,
        tensor_types=dict(types),
        header_and_descriptors_bytes=end,
        header_and_descriptors_sha256=digest.hexdigest(),
        metadata=metadata,
        available_bytes=path.stat().st_size,
        expected_full_bytes=expected,
        incomplete_source=path.suffix == ".partial",
    )


def run():
    root = RUN / "large-models"
    record = {}
    for key in ("9b-gguf", "27b-gguf"):
        folder = root / key
        info = read(folder / "source.json")
        items = {}
        for item in info["files"]:
            name = item["rfilename"]
            if not name.endswith(".gguf"):
                continue
            p = folder / name
            if not p.is_file():
                p = folder / (name + ".partial")
            items[name] = inspect(p, item["size"])
        record[key] = dict(revision=info["revision"], files=items)
    write(
        root / "gguf-headers.json",
        dict(
            source="https://github.com/ggml-org/llama.cpp/blob/b12721/ggml/include/gguf.h",
            scope="Metadata bytes and tensor tables only; full-file integrity and model execution tracked separately.",
            models=record,
        ),
    )
    for k, v in record.items():
        print(
            k,
            [
                (
                    n,
                    x["tensors"],
                    x["metadata"].get("general.architecture"),
                    x["metadata"].get("clip.projector_type"),
                )
                for n, x in v["files"].items()
            ],
        )


if __name__ == "__main__":
    run()
