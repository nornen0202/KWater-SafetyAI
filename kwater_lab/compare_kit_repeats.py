"""Check two collected official-kit runs. Each receipt accompanies its own submission.jsonl."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


def inspect(receipt):
    meta = json.loads(receipt.read_bytes())
    if not meta["run_id"] or meta["seed"] != 42 or meta["kit_return_code"] != 0 or meta["result"] != "OK":
        raise ValueError("Expected a successful seed-42 official run")
    for key in ("archive_sha256", "input_manifest_sha256", "output_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", meta[key]):
            raise ValueError(key)
    if not re.search("@sha256:[0-9a-f]{64}$", meta["official_image"]):
        raise ValueError("Expected digest-pinned official image")
    path = receipt.parent / "submission.jsonl"
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    if digest != meta["output_sha256"]:
        raise ValueError("Recorded output SHA does not match the collected file")
    rows = [json.loads(line) for line in content.splitlines()]
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != 200 or len(ids) != 200 or meta["valid_rows"] != [200, 200]:
        raise ValueError("Expected 200 unique officially valid outputs")
    return meta, path, ids, rows


def compare(first, second):
    a, path_a, ids_a, rows_a = inspect(first)
    b, path_b, ids_b, rows_b = inspect(second)
    if a["run_id"] == b["run_id"] or path_a.samefile(path_b):
        raise ValueError("Independent run receipts and output files are required")
    for key in ("archive_sha256", "input_manifest_sha256", "official_image"):
        if a[key] != b[key]:
            raise ValueError(f"Conditions differ: {key}")
    return {"same_ids_and_order": ids_a == ids_b, "exact_output": a["output_sha256"] == b["output_sha256"],
            "different_rows_at_same_position": sum(x != y for x, y in zip(rows_a, rows_b)),
            "hashes": [a["output_sha256"], b["output_sha256"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["exact_output"] else 1)
