"""Replay the cached research output through the updated prevention serializer.

    python -m kwater_lab.prevention_replay
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .metrics import decode_target, unique_keys
from .prototype import ELECTRICAL_PREVENTION, assemble


def replay(source_path):
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes, object_pairs_hook=unique_keys)
    before = source["generated_row"]
    decode_target(before)
    raw = source["model_response"]["raw_output"]
    content = json.loads(raw, object_pairs_hook=unique_keys)
    after = assemble(before["image_id"], content)
    original = next(item for item in content["hazards"] if item["code"] == "electrical_energy")
    before_text = before["hazards"]["electrical_energy"]["prevention"]
    if original["prevention"] != before_text:
        raise ValueError("Cached baseline does not match its raw electrical prevention output")
    expected = deepcopy(before)
    expected["hazards"]["electrical_energy"]["prevention"] = ELECTRICAL_PREVENTION
    if before == after or expected != after:
        raise ValueError("Expected exactly the electrical prevention field to change")
    if any(item["references"] for item in after["hazards"].values()):
        raise ValueError("Unverified references appeared in the replay")
    return {
        "experiment": "Electrical prevention serialization replay",
        "source_file": source_path.name, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "sample_id": source["sample_id"], "image_sha256": source["image_sha256"],
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "new_inference_requests": 0, "changed_fields": ["hazards.electrical_energy.prevention"],
        "before_prevention_sha256": hashlib.sha256(before_text.encode("utf-8")).hexdigest(),
        "all_other_fields_equal": True, "local_schema_valid": {"before": True, "after": True},
        "official_law_corpus_used": False, "after_row": after,
        "scope": "One saved response. Classification, risk estimates and other controls retain their original errors.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/prototype.json"))
    parser.add_argument("--output", type=Path, default=Path("results/prevention_replay.json"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Replay output must use a different path from the archived source")
    report = replay(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Replay verified. One field changed, both rows passed local validation.")


if __name__ == "__main__":
    main()
