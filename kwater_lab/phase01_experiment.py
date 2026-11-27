"""Run a local smoke, group OOF factorial and final all-photo fit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

from .evidence_text import text_vector
from .metrics import CODES, decode_target, evaluate
from .phase01_data import (
    balanced_folds,
    semantic_quarantine,
    sha,
    source_family,
    write_json,
)
from .phase01_head import (
    SpatialHead,
    calibrate,
    code_sets,
    decide,
    fit_head,
    fixed,
    logits,
)
from .phase01_infer import run as infer
from .phase01_verify import checked_output
from .phase01_vision import MODEL_IDS, SIGNATURE, Vision, feature_cache
from .submission2_output import make_row


def read(path):
    return json.loads(Path(path).read_bytes())


def write_rows(path, rows):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def supervision(project, records, target_file=None):
    known = {r["sample_id"]: i for i, r in enumerate(records)}
    labels = np.zeros((len(known), 12), np.float32)
    evidence = np.zeros((len(known), 12, 64), np.float32)
    severe = {}
    source = target_file if target_file is not None else project / "data/train/train.jsonl"
    for line in source.read_bytes().splitlines():
        row = json.loads(line)
        sid = row["sample_id"]
        if sid not in known:
            continue
        i = known[sid]
        target = row["target"]
        if "codes" in records[i] and set(records[i]["codes"]) != set(target["hazards"]):
            raise ValueError("Target and frozen manifest differ")
        severe[sid] = set()
        for j, c in enumerate(CODES):
            if c in target["hazards"]:
                h = target["hazards"][c]
                labels[i, j] = 1
                evidence[i, j] = text_vector(h["evidence"]).reshape(8, 64).sum(0)
                if h["risk_level"]["severity"]["score"] == 4:
                    severe[sid].add(c)
        labels[i, 11] = bool(target["hazards"])
        obs = row.get("notes", {}).get("scene_note") or target.get("scene_evidence", "")
        evidence[i, 11] = text_vector(obs).reshape(8, 64).sum(0)
    if len(severe) != len(known):
        raise ValueError("Missing supervised rows")
    return labels, evidence, severe


def policy_cfg(head_name, multiscale, policy):
    return dict(
        codes=list(CODES),
        vision_signature=SIGNATURE,
        multiscale=multiscale,
        head=head_name,
        policy=policy,
    )


def official_check(project, inputs, output, checker):
    n = len((inputs / "manifest.jsonl").read_bytes().splitlines())
    return checked_output(checker, project, inputs, output, n)


def smoke(project, run, vision, records, device, checker, seed):
    sub = [r for r in records if r["exposed"]][:12]
    start = time.monotonic()
    xt = torch.from_numpy(feature_cache(vision, project / "data/train", run, sub)).to(
        device
    )
    yy, ee, _ = supervision(project, sub)
    model = fit_head(
        xt,
        torch.tensor(yy, device=device),
        torch.tensor(ee, device=device),
        np.arange(len(sub)),
        True,
        seed,
        2,
    )
    directory = run / "smoke"
    directory.mkdir(exist_ok=True)
    write_json(directory / "policy.json", policy_cfg("smoke-head.pt", True, fixed()))
    inputs = project / "data/local-kit/sample-input"
    infer(directory, inputs, directory / "output", device, loaded=(vision, model))
    checked = official_check(project, inputs, directory / "output", checker)
    write_json(
        directory / "smoke.json",
        dict(
            training=len(sub),
            kit=checked,
            seconds=time.monotonic() - start,
            output_sha256=sha(directory / "output/submission.jsonl"),
        ),
    )
    print("SMOKE_OK", checked, flush=True)


def metrics(records, indices, z, pred, groups, severe):
    ids = [records[i]["sample_id"] for i in indices]
    gold = {
        r["sample_id"]: frozenset(r["codes"]) for r in (records[i] for i in indices)
    }
    sets = code_sets(ids, pred)
    m = evaluate(gold, sets, dict(zip(ids, groups[indices])))
    for row in m["per_code"].values():
        row["precision"] = row["tp"] / row["predicted"] if row["predicted"] else None
        row["recall"] = row["tp"] / row["support"] if row["support"] else None
    rank = np.argsort(-z[:, :11], axis=1, kind="stable")[:, :3]
    candidates = {k: {CODES[j] for j in row} for k, row in zip(ids, rank)}
    top3 = sum(len(gold[k] & candidates[k]) for k in ids)
    lost_inside = sum(len((gold[k] - sets[k]) & candidates[k]) for k in ids)
    lost_outside = sum(len(gold[k] - sets[k] - candidates[k]) for k in ids)
    m.update(
        raw_top3_hits=top3,
        final_misses_inside_raw_top3=lost_inside,
        final_misses_outside_raw_top3=lost_outside,
        final_hits_from_outside_raw_top3=sum(
            len((gold[k] & sets[k]) - candidates[k]) for k in ids
        ),
        gold_instances=sum(map(len, gold.values())),
        valid_rows=len(ids),
        invalid_rows=0,
        severity4_completely_missed=sum(
            bool(severe[k]) and severe[k].isdisjoint(sets[k]) for k in ids
        ),
        severity4_photos=sum(bool(severe[k]) for k in ids),
        cardinality={str(n): int((pred.sum(1) == n).sum()) for n in range(4)},
    )
    for k in ids:
        mock_id = hashlib.md5(k.encode(), usedforsecurity=False).hexdigest()
        decode_target(make_row(mock_id, [c for c in CODES if c in sets[k]]))
    return m


def solve_partition(x, y, e, records, groups, train, test, multi, seed, epochs):
    family = [source_family(records[i]["sample_id"]) for i in train]
    inner = balanced_folds(groups[train], y[train, :11].cpu().numpy(), 3, seed, family)
    inner_z = np.empty((len(train), 12), np.float32)
    for f in sorted(set(inner)):
        tr, va = train[inner != f], train[inner == f]
        assert set(groups[tr]).isdisjoint(groups[va])
        net = fit_head(x, y, e, tr, multi, seed, epochs)
        inner_z[inner == f] = logits(net, x[va])
    policies = calibrate(inner_z, y[train, :11].cpu().numpy(), groups[train])
    net = fit_head(x, y, e, train, multi, seed, epochs)
    policies_metadata = {
        "train_photos": len(train),
        "test_photos": len(test),
        "train_groups": len(set(groups[train])),
        "test_groups": len(set(groups[test])),
        "train_support": y[train, :11].sum(0).cpu().int().tolist(),
        "positive_groups": [
            len(set(groups[train][mask]))
            for mask in y[train, :11].bool().cpu().numpy().T
        ],
        "inner_fold_sha256": hashlib.sha256(inner.astype("<i8").tobytes()).hexdigest(),
        "inner_folds": [
            {
                "fold": int(f),
                "photos": int((inner == f).sum()),
                "groups": len(set(groups[train][inner == f])),
            }
            for f in sorted(set(inner))
        ],
    }
    return logits(net, x[test]), policies, policies_metadata


def partition_cache(run, tag, signature, train, test, solve):
    path = run / "fold-cache" / (tag + ".npz")
    path.parent.mkdir(exist_ok=True)
    identity = hashlib.sha256(
        signature.encode()
        + train.astype("<i8").tobytes()
        + b"/"
        + test.astype("<i8").tobytes()
    ).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as d:
            if d["identity"].item() != identity:
                raise ValueError("Changed protocol; use a new run directory")
            return (
                d["z"],
                json.loads(d["policies"].item()),
                json.loads(d["metadata"].item()),
            )
    z, policies, metadata = solve()
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(
            f,
            identity=identity,
            z=z,
            policies=json.dumps(policies),
            metadata=json.dumps(metadata),
        )
    tmp.replace(path)
    return z, policies, metadata


def bootstrap_arms(y, forecasts, groups, resamples=2000, seed=42):
    """Resample identical whole groups for every arm; aggregate integer sufficient statistics."""
    cases = sorted(set(groups))
    rng = random.Random(seed)
    draws = np.array(
        [
            np.bincount(
                rng.choices(range(len(cases)), k=len(cases)), minlength=len(cases)
            )
            for _ in range(resamples)
        ]
    )

    def by_group(values):
        return np.stack([values[groups == group].sum(0) for group in cases])

    actual_by_group = by_group(y)
    actual = draws @ actual_by_group

    def score(hits, denom):
        active = denom > 0
        f1 = np.divide(
            2 * hits, denom, out=np.zeros_like(hits, dtype=float), where=active
        )
        return np.divide(
            f1.sum(-1),
            active.sum(-1),
            out=np.full(f1.shape[:-1], np.nan),
            where=active.any(-1),
        )

    samples, pooled = {}, {}
    for arm, pred in forecasts.items():
        hits, frequency = by_group(y & pred), by_group(pred)
        samples[arm] = score(draws @ hits, actual + draws @ frequency)
        pooled[arm] = float(score(hits.sum(0), (actual_by_group + frequency).sum(0)))
    result = {}
    for arm in forecasts:
        if arm == "single_fixed":
            continue
        delta = samples[arm] - samples["single_fixed"]
        valid = delta[np.isfinite(delta)]
        result[arm] = dict(
            groups=len(cases),
            seed=seed,
            resamples=resamples,
            defined_resamples=len(valid),
            undefined_resamples=resamples - len(valid),
            macro_f1_delta_b_minus_a=pooled[arm] - pooled["single_fixed"],
            percentile_95_ci=np.quantile(valid, [0.025, 0.975]).tolist()
            if len(valid)
            else None,
        )
    return result


def evaluate_split(axis, records, x, y, e, severe, run, signature, seed, epochs):
    groups = np.array([r["group"] for r in records])
    assignments = np.array([r[axis] for r in records])
    use = assignments >= 0 if axis == "dev_fold" else np.ones(len(records), bool)
    indices = np.flatnonzero(use)
    arms = {}
    all_z = {}
    predmat = {}
    fold_policies = {}
    fold_metadata = {}
    for mode, multi in (("single", False), ("multi", True)):
        combined = np.full((len(records), 12), np.nan, np.float32)
        forecasts = {
            k: np.zeros((len(records), 11), bool)
            for k in (
                "fixed",
                "common",
                "class",
                "class_utility",
                "class_utility_scene",
            )
        }
        for f in sorted(set(assignments[use])):
            test = np.flatnonzero(use & (assignments == f))
            train = np.flatnonzero(use & (assignments != f))
            assert (
                set(groups[train]).isdisjoint(groups[test]) and len(test) and len(train)
            )
            tag = f"{axis}-{mode}-{f}"
            started = time.monotonic()
            z, policies, metadata = partition_cache(
                run,
                tag,
                signature,
                train,
                test,
                lambda: solve_partition(
                    x, y, e, records, groups, train, test, multi, seed, epochs
                ),
            )
            fold_policies[tag] = policies
            fold_metadata[tag] = metadata
            combined[test] = z
            for key, pol in policies.items():
                forecasts[key][test] = decide(z, pol)
            print(
                "fold",
                tag,
                len(train),
                len(test),
                round(time.monotonic() - started, 2),
                flush=True,
            )
        if not np.isfinite(combined[indices]).all():
            raise ValueError("Missing OOF logits")
        all_z[mode] = combined[indices]
        for key in forecasts:
            name = mode + "_" + key
            pred = forecasts[key][indices]
            arms[name] = metrics(
                records, indices, combined[indices], pred, groups, severe
            )
            predmat[name] = pred
    ids = [records[i]["sample_id"] for i in indices]
    paired = bootstrap_arms(
        y[indices, :11].cpu().numpy().astype(bool), predmat, groups[indices]
    )
    write_rows(
        run / (axis + "-oof.jsonl"),
        (
            dict(
                sample_id=sid,
                group=records[i]["group"],
                fold=int(assignments[i]),
                gold=records[i]["codes"],
                logits={k: v[n].tolist() for k, v in all_z.items()},
                predictions={
                    k: [c for c, flag in zip(CODES, mat[n]) if flag]
                    for k, mat in predmat.items()
                },
            )
            for n, (i, sid) in enumerate(zip(indices, ids))
        ),
    )
    write_json(
        run / (axis + "-metrics.json"),
        dict(
            arms=arms,
            paired_vs_single_fixed=paired,
            policies=fold_policies,
            training_partitions=fold_metadata,
        ),
    )
    return arms, paired, all_z


def choose(arms, paired):
    base = arms["single_fixed"]
    candidates = [
        k
        for k, v in arms.items()
        if k != "single_fixed"
        and v["macro_f1"] >= base["macro_f1"] + 0.02
        and v["no_hazard_false_alarm_rate"]
        <= max(0.20, base["no_hazard_false_alarm_rate"])
        and v["severity4_completely_missed"] < base["severity4_completely_missed"]
        and paired[k]["percentile_95_ci"][0] > 0
    ]
    return max(["single_fixed", *candidates], key=lambda k: arms[k]["macro_f1"])


def stage_bundle(run, seed, name="inference", policy_name="policy.json"):
    root = Path(__file__).resolve().parents[1]
    bundle = run / name
    package = bundle / "kwater_lab"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()
    for name in (
        "phase01_infer",
        "phase01_head",
        "phase01_vision",
        "metrics",
        "submission2_output",
    ):
        shutil.copyfile(root / "kwater_lab" / (name + ".py"), package / (name + ".py"))
    shutil.copyfile(root / "submission/phase01/main.py", bundle / "main.py")
    cfg = read(run / policy_name)
    shutil.copyfile(run / cfg["head"], bundle / cfg["head"])
    shutil.copyfile(run / policy_name, bundle / "policy.json")
    for key in MODEL_IDS:
        target = bundle / "models" / key
        target.mkdir(parents=True, exist_ok=True)
        for filename in (
            "preprocessor_config.json",
            "config.json",
            "model.safetensors",
        ):
            src = run / "models" / key / filename
            dst = target / filename
            if not dst.exists():
                os.link(src, dst)
            elif not src.samefile(dst):
                raise ValueError(f"Unrelated existing bundle file {dst}")
    shutil.copyfile(
        root / "submission/fusion/licenses/siglip2-APACHE-2.0.txt",
        bundle / "Apache-2.0.txt",
    )
    (bundle / "README.md").write_text(
        "# Local Phase 1 inference\n\nOffline entry on Linux or Windows with PyTorch, transformers, accelerate, safetensors, NumPy and Pillow installed.\n\n`python main.py --input-dir /input --output-dir /output`\n\nBackbones and fitted head are local. The K-water Linux L4 runtime has not been benchmarked.\n",
        encoding="utf8",
    )
    assets = [
        dict(
            kind="model",
            name=repo,
            source="https://huggingface.co/" + repo,
            revision=rev,
            license="Apache-2.0",
            use="frozen visual features",
        )
        for repo, rev in MODEL_IDS.values()
    ]
    assets.append(
        dict(
            kind="data",
            name="official train",
            source="K-water participant training download",
            revision="2026-11-25",
            license="organizer competition-only terms",
            use=f"fit on {cfg['fit_photos']} photos; policy fixed using development OOF only",
        )
    )
    assets.append(
        dict(
            kind="model",
            name="Phase 1 spatial head",
            source="Local team training",
            revision=sha(bundle / cfg["head"]),
            license="Team authored for competition",
            use="hazard and scene scores",
        )
    )
    for name, license_id in (
        ("torch", "BSD-3-Clause"),
        ("transformers", "Apache-2.0"),
        ("accelerate", "Apache-2.0"),
        ("safetensors", "Apache-2.0"),
        ("numpy", "BSD-3-Clause"),
        ("Pillow", "MIT-CMU"),
    ):
        from importlib.metadata import version

        assets.append(
            dict(
                kind="library",
                name=name,
                source="https://pypi.org/project/" + name + "/",
                revision=version(name),
                license=license_id,
                use="local offline inference",
            )
        )
    write_json(
        bundle / "assets.json", dict(schema_version="1.0", seed=seed, assets=assets)
    )
    return bundle


def save_head(model, path):
    tmp = path.with_suffix(".tmp")
    torch.save(model.cpu().state_dict(), tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checker-python", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    start = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    records = read(a.run / "manifest.json")
    meta = read(a.run / "data-summary.json")
    assert (
        len(records) == meta["photos"]
        and sha(a.run / "manifest.json") == meta["manifest_sha256"]
    )
    for row in records:
        if row["dev_fold"] < 0:
            row.pop("codes")
            row.pop("status")
    modules = (
        "phase01_vision.py",
        "phase01_head.py",
        "phase01_data.py",
        "phase01_experiment.py",
        "evidence_text.py",
    )
    signature = ":".join(
        [
            meta["manifest_sha256"],
            SIGNATURE,
            str(a.seed),
            str(a.epochs),
            *(sha(Path(__file__).with_name(m)) for m in modules),
        ]
    )
    vision = Vision(a.run / "models", a.device)
    smoke(a.project, a.run, vision, records, a.device, a.checker_python, a.seed)
    print("FULL_START", len(records), meta["groups"], flush=True)
    features = feature_cache(vision, a.project / "data/train", a.run, records)
    split = semantic_quarantine(a.project, a.run, records, features, a.seed)
    signature += ":" + sha(a.run / "evaluation-manifest.json")
    xt = torch.from_numpy(features).to(a.device)
    del features
    have_features = time.monotonic()
    dev_idx = np.array([i for i, r in enumerate(records) if r["dev_fold"] >= 0])
    dev_rows = [records[i] for i in dev_idx]
    dev_x = xt[dev_idx]
    labels, ev, severe = supervision(a.project, dev_rows)
    yy = torch.from_numpy(labels).to(a.device)
    ee = torch.from_numpy(ev).to(a.device)
    dev, paired, dev_z = evaluate_split(
        "dev_fold", dev_rows, dev_x, yy, ee, severe, a.run, signature, a.seed, a.epochs
    )
    winner = choose(dev, paired)
    write_json(
        a.run / "frozen-selection.json",
        dict(
            arm=winner,
            basis="development OOF only, delta >= .02, FPR <= max(base FPR, .20), severity4 complete misses strictly reduced, paired lower bound >0",
        ),
    )
    dev_all = np.arange(len(dev_rows))
    dev_groups = np.array([r["group"] for r in dev_rows])
    for mode, multi in (("single", False), ("multi", True)):
        model = fit_head(dev_x, yy, ee, dev_all, multi, a.seed, a.epochs)
        filename = mode + "-dev.pt"
        save_head(model, a.run / filename)
        policies = calibrate(dev_z[mode], labels[:, :11], dev_groups)
        write_json(a.run / (mode + "-dev-policies.json"), policies)
        if winner.startswith(mode + "_"):
            dev_policy = policy_cfg(filename, multi, policies[winner[len(mode) + 1 :]])
    dev_policy.update(
        fit_photos=len(dev_rows),
        calibration_photos=len(dev_rows),
        fit_ids_sha256=hashlib.sha256(
            "\n".join(r["sample_id"] for r in dev_rows).encode()
        ).hexdigest(),
    )
    write_json(a.run / "phase3-dev-policy.json", dev_policy)
    stage_bundle(a.run, a.seed, "phase3-development", "phase3-dev-policy.json")
    # Only after fixing the policy and the Phase 3 development-only weights, open
    # all labels for the independent full-data fit. No holdout scoring takes place.
    labels, ev, _ = supervision(a.project, records)
    full = np.arange(len(records))
    yy, ee = torch.from_numpy(labels).to(a.device), torch.from_numpy(ev).to(a.device)
    for mode, multi in (("single", False), ("multi", True)):
        model = fit_head(xt, yy, ee, full, multi, a.seed, a.epochs)
        save_head(model, a.run / (mode + "-all.pt"))
    final_policy = dict(
        dev_policy,
        head=winner.split("_", 1)[0] + "-all.pt",
        fit_photos=len(records),
        fit_ids_sha256=hashlib.sha256(
            "\n".join(r["sample_id"] for r in records).encode()
        ).hexdigest(),
    )
    write_json(a.run / "policy.json", final_policy)
    bundle = stage_bundle(a.run, a.seed)
    final_head = SpatialHead(final_policy["multiscale"]).to(a.device)
    final_head.load_state_dict(
        torch.load(
            a.run / final_policy["head"], map_location=a.device, weights_only=True
        )
    )
    inputs = a.project / "data/local-kit/sample-input"
    infer(
        bundle,
        inputs,
        a.run / "kit-output",
        a.device,
        loaded=(vision, final_head.eval()),
    )
    check = official_check(a.project, inputs, a.run / "kit-output", a.checker_python)
    write_json(
        a.run / "completed.json",
        dict(
            photos=len(records),
            oof_photos=len(dev_rows),
            holdout_photos=len(records) - len(dev_rows),
            fit_only_full=True,
            groups=split["groups"],
            winner=winner,
            epochs=a.epochs,
            seed=a.seed,
            elapsed_seconds=time.monotonic() - start,
            through_features_seconds=have_features - start,
            checker=check,
            gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
            peak_torch_mib=torch.cuda.max_memory_allocated() / 2**20
            if a.device == "cuda"
            else None,
            torch=torch.__version__,
            numpy=np.__version__,
            python=sys.version,
            signature=signature,
            files={
                k: sha(a.run / k)
                for k in (
                    "manifest.json",
                    "dev_fold-oof.jsonl",
                    "single-all.pt",
                    "multi-all.pt",
                    "single-dev.pt",
                    "multi-dev.pt",
                    "phase3-dev-policy.json",
                    "policy.json",
                )
            },
        ),
    )
    print("COMPLETE", winner, check, flush=True)


if __name__ == "__main__":
    main()
