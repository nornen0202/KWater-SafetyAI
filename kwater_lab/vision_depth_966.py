"""Five group-held-out folds for relative depth. Run extract, kernel, then specialist."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .fold4_code_diagnosis import DEV
from .metrics import CODES
from .train_head import save_atomic
from .vision_gate_depth_probe import RGB, auxiliary_kernel, counts, predict, rgb_kernel


PRIVATE = Path("data/local-reports/vision-depth-966")
CACHE = PRIVATE / "descriptors.npz"
OOF = Path("data/clean-specialist-v1/dev_oof_predictions.npz")
CODE = "struck_by_lifting"
SEEDS = (42, 1042, 2042)


def manifest():
    rows = [r for r in json.loads(DEV.read_bytes()) if r["split"] == "dev"]
    folds = np.array([r["dev_fold"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    if len(rows) != 966 or len({r["sample_id"] for r in rows}) != 966 or set(folds) != set(range(5)):
        raise ValueError("Unexpected development cohort")
    for f in range(5):
        if set(groups[folds == f]) & set(groups[folds != f]):
            raise ValueError("A visual group crosses folds")
    return rows, folds, groups


def save_report(name, obj):
    path = Path(f"results/vision-depth-966-{name}-20261128.json")
    content = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode()
    save_atomic(path, lambda f: f.write(content))


def extract():
    from .depth_pilot import MODEL, collect
    rows, _, _ = manifest()
    inventory = {r["sample_id"]: r for r in rows}
    collect(sorted(inventory), inventory, Path("data/train"), MODEL, CACHE)


def load_inputs(rows):
    ids = [r["sample_id"] for r in rows]
    with np.load(CACHE) as c:
        pos = {sid: i for i, sid in enumerate(c["ids"].tolist())}
        if len(pos) != len(c["ids"]) or set(pos) != set(ids):
            raise ValueError("Incomplete depth descriptors")
        order = [pos[s] for s in ids]
        depth = c["vectors"][order]
        times = c["seconds"][order]
        signature = str(c["signature"])
    with np.load(RGB) as c:
        pos = {sid: i for i, sid in enumerate(c["ids"].tolist())}
        x = c["features"][[pos[s] for s in ids]]
        fingerprint = str(c["digest_key"])
    if depth.shape != (966, 3, 32) or x.shape != (966, 7, 2176) or not np.isfinite(depth).all():
        raise ValueError("Invalid feature dimensions or contents")
    return x, depth, times, {"depth_signature": signature, "rgb_signature": fingerprint,
                             "manifest_sha256": sha256(DEV.read_bytes()).hexdigest(),
                             "depth_cache_sha256": sha256(CACHE.read_bytes()).hexdigest()}


def shuffled(values, train, groups, salt):
    rng = np.random.default_rng(salt)
    donors = [rng.choice(train[groups[train] != g]) for g in groups]
    return values[donors]


def incident_links(groups):
    from .fold4_code_diagnosis import MASK
    paths = (("fold0_group_id", Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")),
             ("fold4_group_id", MASK))
    # Collapse confirmed visual aliases transitively, while retaining all 966 outer test photos.
    root = {g: g for g in set(groups)}
    for key, path in paths:
        for pair in json.loads(path.read_bytes())["confirmed_cross_group_links"]:
            a, b = root[pair[key]], root[pair["train_group_id"]]
            root = {g: (a if v == b else v) for g, v in root.items()}
    alias = np.array([root[g] for g in groups])
    return alias


def group_interval(y, left, right, groups):
    unique, inverse = np.unique(groups, return_inverse=True)
    def table(p):
        return np.stack([np.bincount(inverse, weights=m, minlength=len(unique))
                         for m in (p & y, p & ~y, ~p & y)], axis=1)
    def score(t):
        tp, fp, fn = t.T
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    rng = np.random.default_rng(20261128)
    samples = rng.integers(len(unique), size=(2000, len(unique)))
    delta = score(table(right)[samples].sum(1)) - score(table(left)[samples].sum(1))
    return np.percentile(delta, [2.5, 97.5]).tolist()


def kernel():
    rows, folds, groups = manifest()
    x, depth, times, fingerprint = load_inputs(rows)
    y = np.array([CODE in r["codes"] for r in rows])
    normal = np.array([not r["codes"] for r in rows])
    k = rgb_kernel(x)
    modes = ["rgb", "brightness", "edge", "depth"] + [f"wrong_{i}" for i in range(20)]
    preds = {m: np.zeros(len(rows), bool) for m in modes}
    by_fold = {}
    alias = incident_links(groups)
    purged = {m: np.zeros(len(rows), bool) for m in ("rgb", "depth", "wrong_0")}
    for f in range(5):
        train, test = np.where(folds != f)[0], np.where(folds == f)[0]
        for m in modes:
            combined = k
            if m != "rgb":
                values = depth[:, {"brightness": 0, "edge": 1}.get(m, 2)]
                if m.startswith("wrong"):
                    values = shuffled(values, train, groups, 1000 * f + int(m.split("_")[1]))
                combined = k + .25 * auxiliary_kernel(values, train)
            preds[m][test] = predict(combined, y, train, test)
        by_fold[str(f)] = {m: counts(y[test], preds[m][test], normal[test]) for m in modes[:4]}
        clean_train = train[~np.isin(alias[train], alias[test])]
        if set(alias[clean_train]) & set(alias[test]):
            raise ValueError("Known related scene remains in training")
        by_fold[str(f)]["purged_training_photos"] = len(train) - len(clean_train)
        for m in purged:
            gram = k
            if m != "rgb":
                values = depth[:, 2]
                if m == "wrong_0":
                    values = shuffled(values, clean_train, alias, 1000*f)
                gram = k + .25 * auxiliary_kernel(values, clean_train)
            purged[m][test] = predict(gram, y, clean_train, test)
    pooled = {m: counts(y, p, normal) for m, p in preds.items()}
    save_report("kernel", dict(photos=len(rows), groups=len(set(groups)), positive=int(y.sum()),
        normal=int(normal.sum()), inputs=fingerprint, pooled=pooled, by_fold=by_fold,
        alias_purged_pooled={m: counts(y,p,normal) for m,p in purged.items()},
        alias_purged_groups=len(set(alias)),
        depth_beats_wrong=sum(pooled["depth"]["f1"] > pooled[m]["f1"] for m in modes[4:]),
        group_bootstrap_delta_f1_95pct=group_interval(y, preds["rgb"], preds["depth"], groups),
        cpu_seconds_all_966=float(sum(times)), cpu_seconds_mean=float(times.mean()),
        cpu_seconds_median=float(np.median(times)), cpu_seconds_p95=float(np.percentile(times, 95)),
        timing_scope="Read, SHA, decode, CPU ONNX, three descriptors. Excludes model load and checkpoint writes."))


def geometry(values):
    # Fixed four-by-four relative depth and gradient grids. Reuse one depth pass.
    # Local views receive their quadrant at the same scale as the global grid.
    grid = values.reshape(-1, 2, 4, 4)
    crops = [grid] + [grid[:, :, i:i+2, j:j+2] for i, j in ((0, 0), (0, 2), (2, 0), (2, 2), (1, 1))] + [grid]
    views = []
    for crop in crops:
        if crop.shape[-1] == 2:
            crop = crop.repeat(2, axis=2).repeat(2, axis=3)
        diff = crop - crop[:, :, ::-1, :]
        views.append(np.concatenate([crop.reshape(-1, 32), diff.reshape(-1, 32)], axis=1))
    return np.stack(views, axis=1)


def specialist():
    import torch
    from .clean_specialist import (apply_calibrated_policy, build_fold_text_svd_targets,
                                   fit_nested_calibration, predict_probs, train_single_head)
    from .clean_development_audit import point
    from .law_reranker_11code import read_selected_cases

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    rows, folds, groups = manifest()
    x, depth, _, fingerprint = load_inputs(rows)
    ids = [r["sample_id"] for r in rows]
    y = np.array([[code in r["codes"] for code in CODES] + [bool(r["codes"])] for r in rows])
    rationales = {}
    for row in read_selected_cases(Path("data/train/train.jsonl"), set(ids)):
        t = row["target"]
        text = " ".join(f"{c}: {h['evidence']} {h['prevention']}" for c, h in sorted(t["hazards"].items()))
        rationales[row["sample_id"]] = text if t["hazards"] else "무위험 안전 장면 " + str(t.get("scene_evidence", ""))
    with np.load(OOF) as a:
        if a["ids"].tolist() != ids or not np.array_equal(a["y_codes"], y[:, :-1]):
            raise ValueError("Frozen original specialist output IDs or labels differ")
        old_c, old_s = a["code_probs"], a["scene_probs"]
    modes = ("rgb", "depth", "wrong")
    preds = {mode: np.zeros_like(y[:, :-1]) for mode in (*modes, "archived_rgb")}
    by_fold = {}
    for fold in range(5):
        train, test = np.where(folds != fold)[0], np.where(folds == fold)[0]
        with np.load(f"data/clean-specialist-v1/strict-inner-oof/fold-{fold}.npz") as a:
            if a["train_ids"].tolist() != [ids[i] for i in train]:
                raise ValueError("Calibration archive and outer training order differ")
            policy = fit_nested_calibration(a["code_scores"], a["scene_scores"], y[train, :-1], groups[train].tolist())
        def record(mode, code, scene):
            result = apply_calibrated_policy(code, scene, policy)
            preds[mode][test] = [[c in r for c in CODES] for r in result]
        record("archived_rgb", old_c[test], old_s[test])
        text = build_fold_text_svd_targets([rationales[ids[i]] for i in train])
        for mode in modes:
            views = x
            if mode != "rgb":
                values = depth[:, 2]
                if mode == "wrong":
                    values = shuffled(values, train, groups, fold * 1000)
                a = geometry(values)
                z = np.clip((a - a[train].mean(0)) / np.maximum(a[train].std(0), 1e-5), -4, 4)
                # Both RGB encoders are unit normalized. Match per-coordinate RMS.
                views = np.concatenate([x, (z * np.sqrt(2 / 2176)).astype(x.dtype)], axis=2)
            votes = []
            for seed in SEEDS:
                stamp = json.dumps([fingerprint, fold, mode, seed, 55, "geometry-v1",
                                    sha256(Path(__file__).read_bytes()).hexdigest()], sort_keys=True)
                path = PRIVATE / f"fold-{fold}-{mode}-{seed}.npz"
                if path.exists():
                    with np.load(path) as a:
                        if str(a["stamp"]) != stamp:
                            raise ValueError(f"Stale cache: {path}")
                        votes.append((a["code"], a["scene"]))
                    continue
                start = perf_counter()
                model = train_single_head(views[train], y[train], text, seed + 100 * fold, torch.device("cpu"))
                c, s = predict_probs([model], views[test], torch.device("cpu"))
                save_atomic(path, lambda f: np.savez(f, stamp=stamp, code=c, scene=s))
                votes.append((c, s))
                print(f"fold={fold} {mode} seed={seed} seconds={perf_counter()-start:.1f}", flush=True)
            record(mode, np.mean([v[0] for v in votes], 0), np.mean([v[1] for v in votes], 0))
        by_fold[str(fold)] = {m: point(y[test, :-1], pred[test]) for m, pred in preds.items()}
    col = CODES.index(CODE)
    summary = {m: point(y[:, :-1], pr) for m, pr in preds.items()}
    alias = incident_links(groups)
    clean = np.ones(len(ids), bool)
    for f in range(5):
        test = folds == f
        clean[test] = ~np.isin(alias[test], alias[~test])
    save_report("specialist", dict(photos=len(ids), groups=len(set(groups)), seeds=SEEDS, epochs=55,
        inputs=fingerprint, pooled=summary, by_fold=by_fold,
        alias_excluded_photos=int(clean.sum()), alias_excluded_normal=int((clean & ~y[:,-1]).sum()),
        alias_excluded={m: point(y[clean,:-1],p[clean]) for m,p in preds.items()},
        lifting={m: counts(y[:, col], pred[:, col], ~y[:, -1]) for m, pred in preds.items()},
        group_bootstrap_delta_f1_95pct=group_interval(y[:, col], preds["rgb"][:, col], preds["depth"][:, col], groups),
        calibration="Identical strict-inner RGB OOF temperatures and thresholds fixed across all arms in each outer fold."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "kernel", "specialist"))
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    {"extract": extract, "kernel": kernel, "specialist": specialist}[args.stage]()
