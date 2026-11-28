"""Validate original Fold 2 BF16 prefix recovery and record exact batch boundaries."""
import re

from .fold2_v3_epoch1 import DEFAULT_RUN, split
from .fold4_same_verifier import digest, read, save
from .metrics import decode_target, read_index
from pathlib import Path


def main():
    run = DEFAULT_RUN
    preserved = run / "epoch1-bf16-original-44-cache.json"
    final = run / "epoch1-bf16-raw-cache.json"
    predictions = run / "epoch1-bf16-predictions.jsonl"
    old, current = read(preserved), read(final)
    ids = [r["sample_id"] for r in split()[1]]
    if (old["identity"] != current["identity"] or len(old["outputs"]) != 44
            or list(old["outputs"]) != ids[:44] or list(current["outputs"]) != ids
            or any(current["outputs"][sid] != row for sid, row in old["outputs"].items())):
        raise ValueError("Preserved cache is not the exact unchanged prefix of complete Fold 2 BF16 cache")
    receipt = read(run / "epoch1-bf16-inference-receipt.json")
    if (receipt["stats"]["photos"] != len(ids) or receipt["prediction_sha256"] != digest(predictions)
            or list(read_index(predictions, lambda r: decode_target(r, local_ids=True))) != ids):
        raise ValueError("Scored file and inference receipt do not cover the exact canonical cohort")
    events = {0: old["identity"]["batch_size"]}
    for path in sorted(run.glob("*-infer.stdout.log")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"\s*vlm_batch_backoff (\d+) after (\d+) completed images", line)
            if match:
                size, offset = map(int, match.groups())
                events[offset] = min(size, events.get(offset, size))
    if events.get(0) != 4 or events.get(44) != 2 or events[max(events)] != current["effective_batch_size"]:
        raise ValueError("Batch history missing or disagrees with the immutable cache")
    batches = []
    position = 0
    while position < len(ids):
        start = max(point for point in events if point <= position)
        count = min(events[start], len(ids) - position)
        if any(position < event < position + count for event in events):
            raise ValueError("Backoff happened inside a completed batch")
        batches.append({"start_1based": position + 1, "stop_1based": position + count, "photo_ids": ids[position:position+count]})
        position += count
    private = run / "epoch1-bf16-batches-local.json"
    save(private, {"predictions_sha256": digest(predictions), "prefix_sha256": digest(preserved),
                   "raw_cache_sha256": digest(final), "batches": batches})
    public = Path("results/fold2-v3-4b-epoch1-batch-witness-20261128.json")
    save(public, {"photos": len(ids), "exact_canonical_coverage": True, "predictions_sha256": digest(predictions),
                  "preserved_prefix_rows_unchanged": len(old["outputs"]), "prefix_sha256": digest(preserved),
                  "raw_cache_sha256": digest(final), "private_batch_manifest_sha256": digest(private),
                  "batch_transitions": [{"rows_already_complete": p, "effective_batch_size": events[p]} for p in sorted(events)],
                  "batch_count_by_rows": {n: sum(len(b["photo_ids"]) == n for b in batches) for n in (1, 2, 4)},
                  "matching_limit": "Declared BF16, prompt, image limits and order match. Measurements combine two batch sizes after an actual CUDA OOM. This does not establish uniform-batch output parity."})


if __name__ == "__main__":
    main()
