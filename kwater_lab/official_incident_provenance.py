"""Count source fields without decoding pixels, opening targets, or exporting metadata values."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct

from PIL.Image import Exif


SIGNATURE = b"\x89PNG\r\n\x1a\n"
TOKENS = re.compile(rb'"(?:\\.|[^"\\])*"|[{}\[\],:]|[^\s{}\[\],:"]+')
EXIF_FIELDS = {
    306: "file_datetime", 36867: "capture_datetime_original", 36868: "digitized_datetime",
    42016: "image_unique_id", 269: "document_name", 285: "page_name",
    315: "artist", 33432: "copyright", 42033: "body_serial_number", 34853: "gps_ifd",
}
TEXT_CHUNKS = {"tEXt", "iTXt", "zTXt"}
STANDARD = {"IHDR", "PLTE", "IDAT", "IEND", "tRNS", "cHRM", "gAMA", "iCCP", "sBIT",
            "sRGB", "bKGD", "hIST", "pHYs", "sPLT", "tIME", "eXIf", *TEXT_CHUNKS}


def png_fields(contents):
    found = set()
    if contents[:8] != SIGNATURE:
        raise ValueError("Invalid PNG signature")
    offset = 8
    while offset + 12 <= len(contents):
        length, tag = struct.unpack_from(">I4s", contents, offset)
        end = offset + 12 + length
        if end > len(contents):
            raise ValueError("Truncated PNG chunk")
        name = tag.decode("ascii")
        found.add("chunk_" + (name if name in STANDARD else "other"))
        if name in TEXT_CHUNKS:
            found.add("has_text_container")
        elif name == "eXIf":
            # EXIF metadata is the sole chunk payload examined. Never parse
            # pixel samples, PNG text values, maker notes, or GPS coordinates.
            exif = Exif()
            exif.load(contents[offset + 8:end - 4])
            names = set(exif.keys())
            if 34665 in names:
                names.update(exif.get_ifd(34665).keys())
            found.update("tag_" + label for tag_id, label in EXIF_FIELDS.items() if tag_id in names)
        offset = end
        if name == "IEND":
            if offset != len(contents):
                found.add("has_trailing_bytes")
            break
    if not {"chunk_IHDR", "chunk_IDAT", "chunk_IEND"} <= found:
        raise ValueError("Incomplete PNG")
    return found


def png_header_fields(path):
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as contents:
        return png_fields(contents)


def wrapper_fields(line):
    # Containers under all other fields, including target, are skipped as a
    # unit. Only actual keys in the root object and notes object are decoded.
    tokens = (m.group() for m in TOKENS.finditer(line))
    result = {"wrapper": set(), "notes": set()}

    def skip(first):
        depth = int(first in (b"{", b"["))
        while depth:
            token = next(tokens)
            depth += int(token in (b"{", b"[")) - int(token in (b"}", b"]"))

    def fields(where):
        key = next(tokens)
        while key != b"}":
            if not key.startswith(b'"') or next(tokens) != b":":
                raise ValueError("Invalid object field")
            name = json.loads(key)
            result[where].add(name)
            value = next(tokens)
            if where == "wrapper" and name == "notes" and value == b"{":
                fields("notes")
            else:
                skip(value)
            split = next(tokens)
            if split == b"}":
                return
            if split != b",":
                raise ValueError("Invalid object separator")
            key = next(tokens)

    if next(tokens) != b"{":
        raise ValueError("Expected training wrapper")
    fields("wrapper")
    if next(tokens, None) is not None:
        raise ValueError("Trailing token")
    return result


def run(root):
    train = root / "data/train"
    previous = json.loads((root / "results/train_audit_summary.json").read_text(encoding="utf-8"))
    counts = Counter()
    paths = sorted((train / "images").glob("*.png"))
    count = len(paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fields in pool.map(png_header_fields, paths):
            counts.update(fields)
    schema = {"wrapper": Counter(), "notes": Counter()}
    label_hash = hashlib.sha256()
    n = 0
    with (train / "train.jsonl").open("rb") as f:
        for line in f:
            label_hash.update(line)
            fields = wrapper_fields(line)
            for group in schema:
                schema[group].update(fields[group])
            n += 1
    agree = (label_hash.hexdigest() == previous["label_sha256"] and n == previous["rows"] and
             count == previous["unique_image_paths"] and
             set(schema["wrapper"]) == set(previous["observed_top_level_keys"]) and
             set(schema["notes"]) == set(previous["observed_note_keys"]) and
             counts["chunk_eXIf"] == previous["image_metadata"]["exif_tagged_images"] and
             counts["chunk_iCCP"] == previous["image_metadata"]["png_keys_counts"]["icc_profile"] and
             counts["chunk_pHYs"] == previous["image_metadata"]["png_keys_counts"]["dpi"])
    if not agree:
        raise ValueError("Audit differs from the previous official package inventory; inspect locally")
    result = {
        "png_files": count, "jsonl_rows": n, "previous_inventory_matches": agree,
        "png_chunk_presence_in_files": {k[6:]: v for k, v in sorted(counts.items()) if k.startswith("chunk_")},
        "png_with_exif": counts["chunk_eXIf"], "png_with_text_container": counts["has_text_container"],
        "png_with_tIME_modification_only": counts["chunk_tIME"], "png_with_trailing_bytes": counts["has_trailing_bytes"],
        "images_with_exif_field": {label: counts["tag_" + label] for label in EXIF_FIELDS.values()},
        "jsonl_field_presence_in_rows": {name: dict(sorted(c.items())) for name, c in schema.items()},
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = run(args.root)
    out = args.root / "results/official_incident_provenance.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['png_files']} PNG files, {report['jsonl_rows']} schema records, "
          f"{report['png_with_exif']} EXIF containers.")
