"""Restore both agreement heads from the original outer-training groups using shared cached visual tokens."""

import argparse
import hashlib
import time
import numpy as np
from .metrics import CODES
from .phase2_panel import (
    RUN,
    PROJECT,
    PHASE1,
    SIGNATURE,
    registry,
    read,
    write,
    file_hash,
)
from .train_head import fit, probabilities, save_atomic

ROOT = RUN / "agreement-heads"


def matrix(rows):
    path = ROOT / "development-fp16.npy"
    ids = list(rows)
    manifest = ROOT / "features.json"
    if not path.is_file() or not manifest.is_file():
        ROOT.mkdir(exist_ok=True)
        x = np.empty((len(ids), 6, 17, 128), np.float16)
        for i, s in enumerate(ids):
            key = hashlib.sha256((SIGNATURE + rows[s]["sha256"]).encode()).hexdigest()
            with (PHASE1 / "features" / (key + ".npy")).open("rb") as f:
                v = np.load(f, allow_pickle=False)
            if v.shape != (6, 17, 128) or v.dtype != np.float16:
                raise ValueError("Existing image feature layout differs")
            x[i] = v
        save_atomic(path, lambda f: np.save(f, x, allow_pickle=False))
        write(
            ROOT / "features.json",
            dict(
                signature=SIGNATURE,
                sha256=file_hash(path),
                ids=ids,
                inference_view=0,
                pooled_token=0,
                spatial_tokens=list(range(17)),
                training_only_augmentation_view=5,
            ),
        )
    receipt = read(manifest)
    if receipt["ids"] != ids or file_hash(path) != receipt["sha256"]:
        raise ValueError("The consolidated feature order changed")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def linear():
    rows = registry()
    x = matrix(rows)
    ids = list(rows)
    idx = {s: i for i, s in enumerate(ids)}
    provenance = read(RUN / "dev-heads.json")
    target = read(RUN / "matched-budget-predictions.json")["large_concat"]
    checks = []
    predictions = {}
    for fold in range(5):
        train = provenance["train_ids_by_fold"][str(fold)]
        test = sorted(s for s in ids if rows[s]["dev_fold"] == fold)
        if {rows[s]["group"] for s in train} & {rows[s]["group"] for s in test}:
            raise ValueError("Fold group overlap")
        features = x[[idx[s] for s in train + test], 0, 0, :].astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        y = np.array(
            [[c in rows[s]["codes"] for c in CODES] for s in train], dtype=float
        )
        start = time.perf_counter()
        head = fit(features[: len(train)], y, 0.01, 250)
        path = ROOT / f"fold-{fold}-linear.npz"
        save_atomic(path, lambda f: np.savez(f, **head))
        with np.load(path, allow_pickle=False) as model:
            z = probabilities(model, features[len(train) :])
        expected = np.array(
            [[provenance["scores"][s]["large_concat"][c] for c in CODES] for s in test]
        )
        current = [CODES[i] for i in np.argmax(z, axis=1)]
        equal = sum(c == target[s]["candidate"] for c, s in zip(current, test))
        checks.append(
            dict(
                fold=fold,
                photos=len(test),
                top1_matches=equal,
                max_score_error=float(abs(z - expected).max()),
                seconds=time.perf_counter() - start,
                file_sha256=file_hash(path),
                train_ids=train,
            )
        )
        predictions.update(zip(test, current))
        write(ROOT / "linear-progress.json", checks)
    if not all(
        v["photos"] == v["top1_matches"] and v["max_score_error"] < 1e-6 for v in checks
    ):
        raise ValueError("Restored linear heads disagree with outer-fold outputs")
    write(ROOT / "linear-predictions.json", predictions)


def policy_replay():
    """Check serialized policies on original held-out logits before spatial weights finish."""
    rows = registry()
    ids = list(rows)
    reference = read(RUN / "prior-refit/predictions.json")
    top = read(ROOT / "linear-predictions.json")
    folds = []
    ledger = {}
    for fold in range(5):
        data = read(RUN / "prior-refit" / f"fold-{fold}.json")
        policy_file = ROOT / f"fold-{fold}-policy.json"
        write(policy_file, data["policies"]["class"])
        p = read(policy_file)
        z = np.asarray(data["z"], dtype=np.float64)
        prob = np.exp(-np.logaddexp(0, -z / np.asarray(p["temperatures"])))
        mask = (prob[:, :11] >= p["thresholds"]) & (
            prob[:, 11, None] >= p["scene_gate"]
        )
        rank = np.argsort(
            -np.where(mask, prob[:, :11] - p["offset"], -np.inf), axis=1, kind="stable"
        )[:, :3]
        match = 0
        for j, i in enumerate(data["test"]):
            sid = ids[i]
            assert rows[sid]["dev_fold"] == fold
            codes = sorted(CODES[k] for k in rank[j] if mask[j, k])
            match += codes == sorted(reference[sid]["class"])
            ledger[sid] = dict(
                candidate=top[sid], prior_class=codes, agreed=top[sid] in codes
            )
        folds.append(
            dict(fold=fold, n=len(z), equal=match, policy_sha256=file_hash(policy_file))
        )
    if any(v["equal"] != v["n"] for v in folds):
        raise ValueError("Archived-logit policy replay differs")
    write(
        ROOT / "policy-replay.json",
        dict(
            folds=folds,
            all=sum(v["n"] for v in folds),
            candidate_in_class=sum(r["agreed"] for r in ledger.values()),
            scope="Actual saved/reloaded top-one weights and policies; class logits still read from historical OOF.",
        ),
    )
    write(ROOT / "policy-replay-rows.json", ledger)


def spatial():
    import torch
    from .phase01_head import SpatialHead, fit_head, logits, decide
    from .phase01_experiment import supervision

    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    rows = registry()
    ids = list(rows)
    x = torch.from_numpy(np.array(matrix(rows)))
    y, e, _ = supervision(PROJECT, list(rows.values()), RUN / "dev-train.jsonl")
    y, e = torch.from_numpy(y), torch.from_numpy(e)
    prior = read(RUN / "prior-refit/predictions.json")
    top = read(ROOT / "linear-predictions.json")
    checks = []
    forecasts = {}
    for fold in range(5):
        before = time.perf_counter()
        cache = read(RUN / "prior-refit" / f"fold-{fold}.json")
        train = np.asarray(
            [i for i, s in enumerate(ids) if rows[s]["dev_fold"] != fold]
        )
        test = np.asarray([i for i, s in enumerate(ids) if rows[s]["dev_fold"] == fold])
        if cache["test"] != test.tolist() or cache["train_ids"] != [
            ids[i] for i in train
        ]:
            raise ValueError("Prior=0 outer split changed")
        fitted = fit_head(x, y, e, train, False, 20261126, 14, prior_strength=0)
        path = ROOT / f"fold-{fold}-spatial.pt"
        save_atomic(path, lambda f: torch.save(fitted.state_dict(), f))
        policies = ROOT / f"fold-{fold}-policy.json"
        write(policies, cache["policies"]["class"])
        net = SpatialHead(False).eval()
        net.load_state_dict(
            torch.load(path, weights_only=True, map_location="cpu"), strict=True
        )
        z = logits(net, x[test, :1])
        keep = decide(z, read(policies))
        matches = 0
        for i, mask in zip(test, keep):
            s = ids[i]
            codes = [c for c, v in zip(CODES, mask) if v]
            matches += set(codes) == set(prior[s]["class"])
            forecasts[s] = dict(top1=top[s], spatial=codes, confirmed=top[s] in codes)
        checks.append(
            dict(
                fold=fold,
                photos=len(test),
                set_matches=matches,
                max_logit_error=float(np.abs(z - np.asarray(cache["z"])).max()),
                seconds=time.perf_counter() - before,
                state_sha256=file_hash(path),
                policy_sha256=file_hash(policies),
                train_ids=cache["train_ids"],
            )
        )
        write(ROOT / "spatial-progress.json", checks)
    if any(v["set_matches"] != v["photos"] for v in checks):
        raise ValueError(
            "Restored spatial thresholds disagree with the frozen outer-fold pass sets"
        )
    write(ROOT / "shared-predictions.json", forecasts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["linear", "policy", "spatial"])
    {"linear": linear, "policy": policy_replay, "spatial": spatial}[
        p.parse_args().stage
    ]()
