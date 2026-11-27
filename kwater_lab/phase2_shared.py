"""Audit a shallow head on the same Qwen image tokens already computed by the model."""

from contextlib import contextmanager
import time
import numpy as np

from .metrics import CODES, evaluate
from .phase2_paths import file_stem
from .phase2_panel import RUN, feature_inputs, read, registry, write
from .train_head import fit, probabilities, save_atomic


@contextmanager
def capture(model, dest, sid):
    if dest is None:
        yield
        return
    path = dest / (file_stem(sid) + ".npy")
    if path.exists():
        yield
        return
    mergers = [m for n, m in model.named_modules() if n.endswith("visual.merger")]
    if len(mergers) != 1:
        write(
            dest / "unavailable.json",
            {"reason": "no unique visual.merger", "matches": len(mergers)},
        )
        yield
        return

    def store(_module, _inputs, output):
        if path.exists():
            return
        pooled = (
            output.detach().float().reshape(-1, output.shape[-1]).mean(0).cpu().numpy()
        )
        if not np.isfinite(pooled).all():
            raise ValueError("nonfinite shared vision embedding")
        save_atomic(path, lambda stream: np.save(stream, pooled, allow_pickle=False))

    handle = mergers[0].register_forward_hook(store)
    try:
        yield
    finally:
        handle.remove()


def compare(name):
    out = RUN / ("lora-" + name)
    split = read(RUN / "sft-split.json")
    train, test = split["train"], split["test"]
    ids = train + test
    rows = registry()
    _, alternatives, _ = feature_inputs({s: rows[s] for s in ids}, True)
    alternatives["shared_qwen"] = {
        s: np.load(out / "shared-vision" / (file_stem(s) + ".npy"), allow_pickle=False)
        for s in ids
    }
    y = np.array([[c in rows[s]["codes"] for c in CODES] for s in train], dtype=float)
    gold = {s: frozenset(rows[s]["codes"]) for s in test}
    existing = {
        s: set(read(out / "base" / (file_stem(s) + ".json"))["final"]["hazards"])
        for s in test
    }
    result = {}
    ledger = {}
    for key, d in alternatives.items():
        x = np.stack([d[s] for s in ids]).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
        start = time.perf_counter()
        head = fit(x[: len(train)], y, 0.01, 250)
        scores = probabilities(head, x[len(train) :])
        elapsed = time.perf_counter() - start
        stand = {}
        proposals = {}
        tp = fp = 0
        for s, z in zip(test, scores):
            stand[s] = frozenset([CODES[int(z.argmax())]])
            c = max(
                (c for c in CODES if c not in existing[s]),
                key=lambda c: z[CODES.index(c)],
            )
            proposals[s] = {
                "candidate": c,
                "base": sorted(existing[s]),
                "eligible": len(existing[s]) < 3,
            }
            if len(existing[s]) < 3:
                tp += int(c in gold[s])
                fp += int(c not in gold[s])
        result[key] = {
            "dimension": int(x.shape[1]),
            "train": len(train),
            "test": len(test),
            "standalone_top1": evaluate(gold, stand),
            "missing_tp_proposals": tp,
            "additional_fp_proposals": fp,
            "fit_and_score_seconds": elapsed,
        }
        ledger[key] = proposals
    write(
        out / "shared-vision-comparison.json",
        {
            "models": result,
            "scope": "outer group-disjoint probe, base Qwen codes, one proposal, no re-read or final acceptance",
        },
    )
    write(out / "shared-vision-proposals.json", ledger)


if __name__ == "__main__":
    import sys

    compare(sys.argv[1])
