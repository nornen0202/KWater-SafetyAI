"""Build a development-exposure ledger solely to protect the final holdout.

The organizer's local-kit README identifies the ten smoke inputs as training
photos. This routine compares image hashes only, never returns their labels,
and does not use the smoke images to train or score a model.
"""

import argparse
import json
from pathlib import Path

from .audit import digest, inspect_photo
from .train_head import save_atomic


def build(audit_path, exposure_path, kit_input, public_ids, output):
    kit_input = kit_input.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    photos = audit["inventory"]
    initial = set(exposure["exposed_ids"]) | set(public_ids)
    if (audit["errors"] or audit["labels_sha256"] != exposure["labels_sha256"]
            or initial - photos.keys() or len(exposure["exposed_ids"]) != len(set(exposure["exposed_ids"]))):
        raise ValueError("Audit, exposure manifest or explicit published IDs are inconsistent")
    manifest = kit_input / "manifest.jsonl"
    inputs = [json.loads(line)["image_id"] for line in manifest.read_text(encoding="utf8").splitlines() if line]
    if not inputs or len(inputs) != len(set(inputs)):
        raise ValueError("Empty or duplicate local-kit input manifest")
    possible_matches = set()
    coverage = 0
    for key in inputs:
        sample = inspect_photo(kit_input, f"images/{key}.jpg")
        h = int(sample["dhash64"], 16)
        matches = {sid for sid, record in photos.items()
                   if record["pixels_sha256"] == sample["pixels_sha256"]
                   or (int(record["dhash64"], 16) ^ h).bit_count() <= 4}
        coverage += bool(matches)
        possible_matches.update(matches)
    record = {
        "purpose": "Final holdout exclusion only. Exclude whole reviewed incidents containing these IDs.",
        "labels_sha256": audit["labels_sha256"], "audit_sha256": digest(audit_path),
        "pilot_exposure_sha256": digest(exposure_path), "pilot_count": len(exposure["exposed_ids"]),
        "published_example_ids": sorted(public_ids), "kit_manifest_sha256": digest(manifest),
        "kit_count": len(inputs), "kit_inputs_with_candidates": coverage,
        "kit_candidate_training_photos": len(possible_matches),
        "kit_method": "Organizer local-kit training examples only; equal RGB SHA256 or dHash64 distance <=4.",
        "limitation": "Near matches are conservatively excluded. Detection cannot establish incident independence.",
        "exposed_ids": sorted(initial | possible_matches),
    }
    save_atomic(output, lambda stream: stream.write((json.dumps(record, indent=2) + "\n").encode("utf8")))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("audit", "pilot", "kit-input", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--published-id", action="append", default=[])
    args = parser.parse_args()
    result = build(args.audit, args.pilot, args.kit_input, args.published_id, args.output)
    print(f"Excluded IDs: {len(result['exposed_ids'])}. Kit hash-candidate coverage: "
          f"{result['kit_inputs_with_candidates']}/{result['kit_count']}.")
