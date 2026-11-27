"""Score possible negative-control donor photos using only the recipient fold's training groups."""

import numpy as np

from .metrics import CODES
from .phase2_panel import ALLOWED8, PROJECT, RUN, order, read, registry, write
from .train_head import fit, probabilities


def gallery():
    rows = registry()
    dev = read(RUN / "dev-heads.json")
    with np.load(PROJECT / "data/prefix-pilot/siglip2-base16-features.npz") as cache:
        features = dict(zip(cache["ids"].tolist(), cache["features"]))
    for sid, r in rows.items():
        if sid not in features:
            features[sid] = np.load(
                RUN / "features/sig224" / (r["sha256"] + ".npy"), allow_pickle=False
            )
    score_by_fold = {}
    for fold, training in dev["train_ids_by_fold"].items():
        held = sorted(s for s in rows if rows[s]["dev_fold"] == int(fold))
        assert set(training).isdisjoint(held)
        y = np.array(
            [[c in rows[s]["codes"] for c in CODES] for s in training], dtype=float
        )
        x = np.stack([features[s] for s in training + held])
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        head = fit(x[: len(training)], y, 0.01, 250)
        z = probabilities(head, x)
        expected = [[dev["scores"][s]["sig"][c] for c in CODES] for s in held]
        np.testing.assert_allclose(z[len(training) :], expected, rtol=1e-6, atol=1e-6)
        score_by_fold[fold] = {
            s: dict(zip(CODES, v.tolist())) for s, v in zip(training, z)
        }
    write(RUN / "sig224-training-donor-scores.json", score_by_fold)


def sources(ids, assigned, rows):
    predictions = read(RUN / "sig224-training-donor-scores.json")
    used, selected = set(), {}
    for sid in ids:
        if sid not in assigned:
            continue
        code = assigned[sid]
        scores = predictions[str(rows[sid]["dev_fold"])]

        def ranking(donor):
            z = scores[donor]
            rank = sorted(ALLOWED8, key=lambda c: -z[c]).index(code) + 1
            return rank, order("p38-gallery/" + sid, donor)

        donor = min((s for s in scores if s not in used and s not in ids), key=ranking)
        rank = ranking(donor)[0]
        assert (
            rows[donor]["group"] != rows[sid]["group"]
            and rows[donor]["dev_fold"] != rows[sid]["dev_fold"]
        )
        used.add(donor)
        selected[sid] = dict(
            sample_id=donor,
            group=rows[donor]["group"],
            fold=rows[donor]["dev_fold"],
            recipient_fold=rows[sid]["dev_fold"],
            code=code,
            rank_of_eight=rank,
            score=scores[donor][code],
        )
    return selected


if __name__ == "__main__":
    gallery()
