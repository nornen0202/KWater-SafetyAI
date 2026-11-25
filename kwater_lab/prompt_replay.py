"""Check the five locally stored prompt arms with the organizer's output checker."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

from .adapter_scale_pilot import write_json
from .prompt_pilot import ARMS, digest, parse_generation
from .train_head import file_hash


def jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def replay(root, checker_python):
    local = root / "data/prompt-pilot"
    design = json.loads((local / "design.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "results/prompt_pilot.json").read_text(encoding="utf-8"))
    signature = digest(design)
    anchors = design["anchors"]
    if summary["signature"] != signature or tuple(design["arms"]) != ARMS:
        raise ValueError("Generation report and local protocol differ")
    receipts = local / signature[:12]
    checker = root / "data/local-kit/kit/check_output.py"
    conditions = {}
    for arm in ARMS:
        folder = receipts / "format-replay" / arm
        out = folder / "out"
        out.mkdir(parents=True, exist_ok=True)
        rows, hashes, outcomes = [], [], Counter()
        for index in range(len(anchors)):
            case = anchors[(index + int(arm == "swapped_11")) % len(anchors)]
            path = receipts / f"{index}-{arm}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["signature"] != signature or record["image_sha256"] != case["sha256"]:
                raise ValueError("Prompt receipt does not match its input")
            hashes.append(file_hash(path))
            result = record["generation"]
            outcomes[result["finish_reason"]] += 1
            _, row = parse_generation(arm, case["sid"], result)
            if row is not None:
                rows.append(row)
        if len(rows) != summary["conditions"][arm]["valid"]:
            raise ValueError("Replay and inference disagree on parseable outputs")
        manifest = [{"image_id": digest(a["sid"])[:32]} for a in anchors]
        if len({row["image_id"] for row in manifest}) != len(anchors):
            raise ValueError("Expected distinct anchor IDs")
        jsonl(folder / "manifest.jsonl", manifest)
        submission = out / "submission.jsonl"
        jsonl(submission, rows)
        completed = subprocess.run(
            [str(checker_python), "-X", "utf8", str(checker), str(folder), str(out)],
            capture_output=True, encoding="utf-8", timeout=300,
        )
        (folder / "checker.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        matched = re.search(r"^유효 결과 (\d+)/(\d+)\s*$", completed.stdout, re.MULTILINE)
        if matched is None:
            raise RuntimeError(f"Official checker did not return counts for {arm}; see its local log")
        valid, total = map(int, matched.groups())
        if total != len(anchors) or valid > len(rows) or completed.returncode != (0 if valid else 20):
            raise ValueError("Official checker status differs from its counts")
        conditions[arm] = {
            "expected": total, "generation_finish": dict(outcomes), "model_assembled": len(rows),
            "kit_valid": valid, "checker_exit": completed.returncode, "receipts_digest": digest(hashes),
            "submission_sha256": file_hash(submission),
        }
    result = {"scope": "Organizer schema and manifest replay. Unusable generations omitted; each "
                       "arm checked against all six images. No quality or leaderboard score.",
              "signature": signature, "checker_sha256": file_hash(checker),
              "schema_sha256": file_hash(checker.with_name("prediction.schema.json")),
              "conditions": conditions}
    write_json(root / "results/prompt_format_replay.json", result)
    for arm, counts in conditions.items():
        print(f"{arm} {counts['kit_valid']}/{counts['expected']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checker-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    replay(args.root.resolve(), args.checker_python.resolve())
