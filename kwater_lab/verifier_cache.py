"""Atomic ordered raw verification cache for a frozen candidate list."""

from __future__ import annotations

import json
from pathlib import Path
import time

from .metrics import unique_keys
from .train_head import save_atomic


def verify_with_cache(candidates, verify_batch, path: Path, identity: dict, batch_size: int):
    if batch_size < 1:
        raise ValueError("Verifier batch size must be positive")
    order = [(row["sample_id"], row["code"]) for row in candidates]
    if len(set(order)) != len(order):
        raise ValueError("Verification candidates must have unique photo-code pairs")
    fingerprint = {"schema_version": 1, "inputs": identity, "batch_size": batch_size, "candidates": candidates}
    results = []
    seconds = 0.0
    if path.is_file():
        cache = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        if cache["fingerprint"] != fingerprint:
            raise ValueError(f"Verifier settings or candidate order differ from {path}")
        results = cache["results"]
        seconds = cache["verification_seconds"]
        if [(row["sample_id"], row["code"]) for row in results] != order[:len(results)]:
            raise ValueError(f"Verifier cache is not an ordered candidate prefix: {path}")
        if len(results) < len(candidates) and len(results) % batch_size:
            raise ValueError(f"Verifier cache ends in a partial batch: {path}")
    reused = len(results)
    for start in range(reused, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        before = time.perf_counter()
        responses = verify_batch(batch)
        seconds += time.perf_counter() - before
        if [(r["sample_id"], r["code"]) for r in responses] != order[start:start + len(batch)]:
            raise ValueError("Verifier did not return the requested candidate batch in order")
        results.extend(responses)
        receipt = {"fingerprint": fingerprint, "results": results, "verification_seconds": seconds}
        save_atomic(path, lambda f: f.write(json.dumps(receipt).encode("utf-8")))
        print(f"  verifier_cache {len(results)}/{len(candidates)}", flush=True)
    return results, {"cache_reused_candidates": reused, "generation_and_model_load_seconds": round(seconds, 3)}
