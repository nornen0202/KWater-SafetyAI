"""Audit the same Gemini neural outer scores with two calibration data sources."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .clean_development_audit import FEATURES, GEMINI, ID_FIELD, LABELS, MANIFEST, comparison, severity4, sha256, top3
from .clean_specialist import (
    apply_calibrated_policy,
    build_fold_text_svd_targets,
    fit_nested_calibration,
    predict_probs,
    train_single_head,
)
from .metrics import CODES


def evaluate_policies(root, policies):
    rows = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("expected development rows only")
    ids = [row["sample_id"] for row in rows]
    folds = np.array([row["dev_fold"] for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    labels = np.array([[code in row["codes"] for code in CODES] + [row["status"] == "hazard"] for row in rows])
    severe = severity4(root / LABELS, ids)
    archive_hash = sha256(root / GEMINI)
    with np.load(root / GEMINI, allow_pickle=False) as archive:
        if (archive["ids"].tolist() != ids or not np.array_equal(archive["folds"], folds)
                or not np.array_equal(archive["y_codes"], labels[:, :-1])):
            raise ValueError("probability archive disagrees with development manifest")
        probs, scene = archive["code_probs"], archive["scene_probs"]
    fixed = top3(probs, np.full(len(CODES), 0.5))
    if set(policies) != set(folds):
        raise ValueError("pooled validation requires one policy for each outer fold")
    reference = np.zeros_like(fixed)
    corrected = np.zeros_like(fixed)
    for fold, policy in policies.items():
        if policy["outer_probability_sha256"] != archive_hash or policy["outer_fold"] != fold:
            raise ValueError("policy does not belong to requested neural OOF probabilities and fold")
        mask = folds == fold
        for output, field in ((reference, "reference_cross_outer_policy"), (corrected, "correct_inner_oof_policy")):
            selected = apply_calibrated_policy(probs[mask], scene[mask], policy[field])
            output[mask] = np.array([[code in row for code in CODES] for row in selected])
    return {
        "calibration_by_fold": {str(k): v["correct_inner_oof_policy"] for k, v in policies.items()},
        "cross_outer_vs_correct_nested": comparison(labels, reference, corrected, groups, np.ones(len(rows), bool), severe),
        "fixed05_vs_correct_nested": comparison(labels, fixed, corrected, groups, np.ones(len(rows), bool), severe),
        "by_fold": {str(k): {key: v[key] for key in ("cross_outer_vs_correct_nested", "fixed05_vs_correct_nested")}
                    for k, v in policies.items()},
        "inner_epochs": policies[0]["inner_epochs"], "inner_seeds": policies[0]["inner_seeds"],
        "outer_probability_sha256": archive_hash,
    }


def inner_predictions(root, ids, folds, groups, labels, features, train, test, outer, epochs, seeds, cache):
    fingerprint = {
        "version": "clean-neural-inner-oof-v1",
        "inputs": {str(path): sha256(root / path) for path in (MANIFEST, FEATURES, LABELS)},
        "specialist_implementation_sha256": sha256(Path(__file__).with_name("clean_specialist.py")),
        "epochs": epochs, "seeds": list(seeds), "outer": outer,
        "torch": str(torch.__version__), "numpy": np.__version__, "cpu_threads": torch.get_num_threads(),
    }
    inner_ids = np.array(ids)[train]
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if (json.loads(str(stored["fingerprint"].item())) == fingerprint
                    and np.array_equal(stored["train_ids"], inner_ids)):
                inner_code, inner_scene = stored["code_scores"], stored["scene_scores"]
                if inner_code.shape == (len(inner_ids), len(CODES)) and inner_scene.shape == (len(inner_ids),):
                    return inner_code, inner_scene

    # Parse rationale supervision only for the outer training IDs.
    allowed_ids = set(inner_ids)
    rationale = {}
    with (root / LABELS).open("rb") as stream:
        for line in stream:
            match = ID_FIELD.search(line)
            if not match or match[1].decode() not in allowed_ids:
                continue
            row = json.loads(line)
            target = row["target"]
            if target["status"] == "no_visible_hazard":
                text = "무위험 안전 장면 " + str(target.get("scene_evidence", ""))
            else:
                text = " ".join(f"{c}: {item['evidence']} {item['prevention']}"
                                for c, item in sorted(target["hazards"].items()))
            rationale[row["sample_id"]] = text

    cross_prob = np.zeros((len(ids), len(CODES)))
    cross_scene = np.zeros(len(ids))
    start = time.perf_counter()
    for inner in sorted(set(folds[train])):
        inner_test = train & (folds == inner)
        inner_train = train & ~inner_test
        if set(groups[inner_train]) & set(groups[inner_test | test]):
            raise ValueError("inner split contains a group outside the training partition")
        text = build_fold_text_svd_targets([rationale[sid] for sid, usable in zip(ids, inner_train) if usable])
        print(f"outer={outer} inner={inner} start elapsed={time.perf_counter()-start:.1f}s", flush=True)
        models = []
        for seed in seeds:
            models.append(train_single_head(
                features[inner_train], labels[inner_train], text,
                seed=seed + int(inner) * 100, device=torch.device("cpu"), epochs=epochs,
            ))
            print(f"outer={outer} inner={inner} seed={seed} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        cross_prob[inner_test], cross_scene[inner_test] = predict_probs(models, features[inner_test], torch.device("cpu"))

    inner_code, inner_scene = cross_prob[train], cross_scene[train]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez(stream, fingerprint=np.array(json.dumps(fingerprint)), train_ids=inner_ids,
                     code_scores=inner_code, scene_scores=inner_scene)
        temp.replace(cache)
    return inner_code, inner_scene


def run(root, outer, epochs, seeds, cache=None, inner_only=False):
    rows = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if any(row["split"] != "dev" for row in rows):
        raise ValueError("expected development rows only")
    ids = [row["sample_id"] for row in rows]
    folds = np.array([row["dev_fold"] for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    labels = np.array([[code in row["codes"] for code in CODES] + [row["status"] == "hazard"] for row in rows])
    with np.load(root / FEATURES, allow_pickle=False) as archive:
        positions = {sid: i for i, sid in enumerate(archive["ids"].tolist())}
        features = archive["features"][[positions[sid] for sid in ids]]
    with np.load(root / GEMINI, allow_pickle=False) as archive:
        if (archive["ids"].tolist() != ids or not np.array_equal(archive["folds"], folds)
                or not np.array_equal(archive["y_codes"], labels[:, :-1])):
            raise ValueError("probability archive disagrees with development manifest")
        outer_prob, outer_scene = archive["code_probs"], archive["scene_probs"]
    test = folds == outer
    train = ~test
    if not test.any() or set(groups[train]) & set(groups[test]):
        raise ValueError("invalid outer split")

    cross_prob, cross_scene = inner_predictions(root, ids, folds, groups, labels, features, train, test, outer, epochs, seeds, cache)
    if inner_only:
        return cross_prob, cross_scene
    reference_calib = fit_nested_calibration(outer_prob[train], outer_scene[train], labels[train, :-1], groups[train])
    correct_calib = fit_nested_calibration(cross_prob, cross_scene, labels[train, :-1], groups[train])

    def decisions(calibration):
        predicted = np.zeros_like(labels[:, :-1])
        code_sets = apply_calibrated_policy(outer_prob[test], outer_scene[test], calibration)
        predicted[test] = np.array([[code in row for code in CODES] for row in code_sets])
        return predicted

    reference = decisions(reference_calib)
    corrected = decisions(correct_calib)
    fixed = top3(outer_prob, np.full(len(CODES), 0.5))
    severe = severity4(root / LABELS, ids)
    return {
        "outer_fold": outer, "inner_epochs": epochs, "inner_seeds": list(seeds),
        "outer_probability_sha256": sha256(root / GEMINI),
        "reference_cross_outer_policy": reference_calib,
        "correct_inner_oof_policy": correct_calib,
        "cross_outer_vs_correct_nested": comparison(labels, reference, corrected, groups, test, severe),
        "fixed05_vs_correct_nested": comparison(labels, fixed, corrected, groups, test, severe),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1042, 2042])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/clean-neural-calibration-fold0.json"))
    parser.add_argument("--pool-from-fold0", type=Path,
                        help="reuse a completed matching fold-0 audit and refit outer folds 1–4")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("set one to four CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.pool_from_fold0:
        first = json.loads(args.pool_from_fold0.read_text(encoding="utf-8"))
        if (first["inner_epochs"] != args.epochs or first["inner_seeds"] != args.seeds
                or first["outer_fold"] != 0 or first["outer_probability_sha256"] != sha256(args.root / GEMINI)):
            raise ValueError("cached fold-0 calibration does not match epoch count, seeds or inputs")
        policies = {0: first}
        cache_folder = args.root / "data/clean-specialist-v1/calibration-audit-folds"
        cache_folder.mkdir(parents=True, exist_ok=True)
        for fold in range(1, 5):
            cache = cache_folder / f"fold-{fold}.json"
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
            if all(cached.get(key) == first[key] for key in ("inner_epochs", "inner_seeds", "outer_probability_sha256")):
                policies[fold] = cached
            else:
                policies[fold] = run(args.root, fold, args.epochs, args.seeds)
                temp = cache.with_suffix(".tmp")
                temp.write_text(json.dumps(policies[fold], allow_nan=False), encoding="utf-8")
                temp.replace(cache)
        result = evaluate_policies(args.root, policies)
    else:
        result = run(args.root, args.outer_fold, args.epochs, args.seeds)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
