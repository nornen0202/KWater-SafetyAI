"""CPU fit-only Clean-Cal specialist and law witnesses for the frozen v2 split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import torch

from .clean_specialist import build_fold_text_svd_targets, predict_probs, train_single_head
from .hybrid_pipeline import select_booster_candidates
from .law_reranker_11code import export_compact_case_bank, read_selected_cases
from .metrics import CODES
from .sub78_clean_cal_design import DEV, LABELS, WORK, known_aliases

CACHE = Path("data/clean-specialist-v1/features_multiview_2176d.npz")
OUT = Path("data/local-reports/clean-cal-v2-fit-specialist")
RECIPE = {"inner_folds": 4, "epochs": 55, "batch_size": 64, "base_seeds": [42, 1042, 2042]}
EXPECTED = {4: (709, 82, 175), 2: (631, 138, 197)}


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes((json.dumps(value, indent=2) + "\n").encode())
    temp.replace(path)


def manifest_inputs(fold):
    folder = WORK / f"fold-{fold}"
    contract_path = folder / "contract.json"
    contract = load(contract_path)
    paths = [folder / f"{kind}-manifest.json" for kind in ("fit", "calibration", "outer")]
    for kind, path in zip(("fit", "calibration", "outer"), paths, strict=True):
        if sha(path) != contract["inputs"][kind]:
            raise ValueError(f"Frozen manifest changed: {kind}")
    for name, expected in contract["confirmed_link_ledger_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Frozen alias ledger changed")
    fit, cal = (load(p) for p in paths[:2])
    outer_ids = {r["sample_id"] for r in load(paths[2])}
    aliases, _, _ = known_aliases(load(DEV))
    fit_ids, cal_ids = ({r["sample_id"] for r in rows} for rows in (fit, cal))
    if (len(fit_ids), len(cal_ids), len(outer_ids)) != EXPECTED[fold]:
        raise ValueError("Unexpected frozen scope size")
    if fit_ids & cal_ids or (fit_ids | cal_ids) & outer_ids:
        raise ValueError("Fit/calibration/outer IDs overlap")
    groups = [aliases[r["group_id"]] for r in fit]
    if set(groups) & {aliases[r["group_id"]] for r in cal}:
        raise ValueError("Fit and calibration share a confirmed alias component")
    return fit, cal, groups, sha(contract_path)


def fold_splits(groups):
    folds = list(GroupKFold(n_splits=RECIPE["inner_folds"]).split(np.zeros(len(groups)), groups=groups))
    if Counter(i for _, test in folds for i in test.tolist()) != Counter(range(len(groups))):
        raise ValueError("Every fit row must receive exactly one OOF prediction")
    for train, test in folds:
        if {groups[i] for i in train} & {groups[i] for i in test}:
            raise ValueError("An alias component crosses inner fit and OOF prediction")
    return folds


def texts_and_targets(rows, labels):
    text, targets = [], []
    for row in rows:
        target = labels[row["sample_id"]]["target"]
        hazards = target["hazards"]
        if set(row["codes"]) != hazards.keys() or row["status"] != target["status"]:
            raise ValueError("Fit manifest and original target disagree")
        if not hazards:
            text.append("무위험 안전 장면 " + str(target.get("scene_evidence", "")))
        else:
            text.append(" ".join(f"{code}: {item['evidence']} {item['prevention']}" for code, item in sorted(hazards.items())))
        targets.append([code in hazards for code in CODES] + [bool(hazards)])
    return text, np.asarray(targets, dtype=np.float32)


def proposals(rows, codes, scenes):
    fixed = {"temperatures": [1.0] * len(CODES), "thresholds": [0.50] * len(CODES), "scene_threshold": 0.50}
    return select_booster_candidates(rows, [{"parsed": None}] * len(rows), codes, scenes, fixed,
                                    code_precisions=None, max_candidates_per_photo=1)


def proposal_support(rows, candidates):
    ids = {row["sample_id"] for row in rows}
    labels = {row["sample_id"]: row["target"]["hazards"] for row in read_selected_cases(LABELS, ids)}
    if labels.keys() != ids:
        raise ValueError("Missing calibration or fit support rows")
    proposed = {row["sample_id"]: row["code"] for row in candidates}
    severe = {sid: {c for c, h in target.items() if h["risk_level"]["severity"]["score"] == 4} for sid, target in labels.items()}
    false = sum(proposed.get(sid) is not None for sid, target in labels.items() if not target)
    return {"photos": len(rows), "proposals": len(candidates), "true_proposed_pairs": sum(c in labels[s] for s,c in proposed.items()),
            "false_proposed_pairs": sum(c not in labels[s] for s,c in proposed.items()),
            "normal_photos": sum(not h for h in labels.values()), "normal_false_proposals": false,
            "severity4_photos": sum(bool(v) for v in severe.values()), "severity4_photos_with_proposals": sum(bool(v) and sid in proposed for sid,v in severe.items()),
            "severity4_matching_proposals": sum(code in severe[sid] for sid,code in proposed.items()),
            "per_code_proposals": dict(Counter(proposed.values()))}


def run(folds, limit):
    with np.load(CACHE, allow_pickle=False) as source:
        cache_ids = source["ids"].tolist()
        features = source["features"]
        cache_key = str(source["digest_key"].item())
    if len(cache_ids) != len(set(cache_ids)) or features.shape != (len(cache_ids), 7, 2176):
        raise ValueError("Unexpected cached feature coverage")
    position = {key: i for i,key in enumerate(cache_ids)}
    source_code = Path(__file__)
    common = {"feature_cache_sha256": sha(CACHE), "feature_cache_key": cache_key,
              "training_source_sha256": sha(Path("kwater_lab/clean_specialist.py")),
              "script_sha256": sha(source_code), "recipe": RECIPE, "device": "CPU", "torch_threads": torch.get_num_threads()}
    for fold in folds:
        started = time.monotonic()
        fit, cal, groups, contract_sha = manifest_inputs(fold)
        fit_ids = [row["sample_id"] for row in fit]
        cal_ids = [row["sample_id"] for row in cal]
        x_fit, x_cal = (features[[position[sid] for sid in ids]] for ids in (fit_ids, cal_ids))
        if not np.isfinite(x_fit).all() or not np.isfinite(x_cal).all():
            raise ValueError("Nonfinite fit/calibration features")
        labels = {row["sample_id"]: row for row in read_selected_cases(LABELS, set(fit_ids))}
        if labels.keys() != set(fit_ids):
            raise ValueError("Only exact fit labels may contribute")
        text, y = texts_and_targets(fit, labels)
        work = OUT / f"fold-{fold}"
        work.mkdir(parents=True, exist_ok=True)
        bank = work / "fit-only-cases.jsonl"
        export_compact_case_bank([labels[sid] for sid in fit_ids], bank)
        splits = fold_splits(groups)
        witness = {**common, "contract_sha256": contract_sha, "fit_ids_law": fit_ids, "fit_only_label_sha256": hashlib.sha256(json.dumps([labels[sid] for sid in fit_ids], sort_keys=True).encode()).hexdigest(),
                   "calibration_predict_ids": cal_ids, "law_case_sha256": sha(bank),
                   "inner": [{"fit_ids": [fit_ids[i] for i in tr], "predict_ids": [fit_ids[i] for i in va]} for tr,va in splits]}
        witness_path = work / "inputs-local.json"
        if witness_path.exists() and load(witness_path)!=witness:
            raise ValueError("Cannot resume scores with changed fit witness")
        save_json(witness_path,witness)
        oof_code=np.full((len(fit),len(CODES)),np.nan); oof_scene=np.full(len(fit),np.nan)
        ensemble_code=[]; ensemble_scene=[]; rows=[]
        for inner,(tr,va) in enumerate(splits):
            models=[]; times=[]
            vectors=None
            for seed in RECIPE["base_seeds"]:
                path=work/f"inner-{inner}-seed-{seed}.pt"
                from .clean_specialist import SpatialDualSpecialistHead
                if path.is_file():
                    saved=torch.load(path,map_location="cpu",weights_only=True)
                    if saved["witness_sha256"]!=sha(witness_path) or saved["inner"]!=inner or saved["seed"]!=seed:
                        raise ValueError("Head checkpoint differs from its fit witness")
                    model=SpatialDualSpecialistHead(feat_dim=x_fit.shape[-1]);model.load_state_dict(saved["state"])
                    model.eval();elapsed=saved["seconds"]
                else:
                    if limit==0:
                        return
                    if vectors is None:
                        vectors=build_fold_text_svd_targets([text[i] for i in tr])
                    t=time.monotonic()
                    model=train_single_head(x_fit[tr],y[tr],vectors,seed=seed+inner*100,device=torch.device("cpu"),
                                            epochs=RECIPE["epochs"],batch_size=RECIPE["batch_size"])
                    elapsed=round(time.monotonic()-t,2)
                    temp=path.with_suffix(".tmp")
                    torch.save({"state":model.state_dict(),"witness_sha256":sha(witness_path),"inner":inner,"seed":seed,"seconds":elapsed},temp)
                    temp.replace(path)
                    if limit is not None:
                        limit-=1
                    print(f"Fold {fold} inner {inner} seed {seed}: {len(tr)} fit, {len(va)} predict, {elapsed}s head",flush=True)
                models.append(model); times.append(elapsed)
            a,b=predict_probs(models,x_fit[va],torch.device("cpu"));oof_code[va]=a;oof_scene[va]=b
            a,b=predict_probs(models,x_cal,torch.device("cpu"));ensemble_code.append(a);ensemble_scene.append(b)
            rows.append({"inner_fold":inner,"fit_count":len(tr),"oof_count":len(va),"seed_seconds":times})
        cal_code,cal_scene=np.mean(ensemble_code,axis=0),np.mean(ensemble_scene,axis=0)
        if not all(np.isfinite(a).all() for a in (oof_code,oof_scene,cal_code,cal_scene)):
            raise ValueError("Incomplete score coverage")
        scores=work/"fit-oof-and-calibration-local.npz"
        np.savez(scores,fit_ids=np.asarray(fit_ids),calibration_ids=np.asarray(cal_ids),fit_code=oof_code,fit_scene=oof_scene,
                 calibration_code=cal_code,calibration_scene=cal_scene)
        # Freeze raw scores before accessing calibration annotations for descriptive diagnostics.
        result={"fold":fold,"scope":"Fit-only actual specialist heads; fixed empty-draft diagnostic, no thresholds fitted or candidate verifier run",
                **common,"contract_sha256":contract_sha,"fit_and_prediction_witness_sha256":sha(witness_path),
                "fit_only_law_case_sha256":sha(bank),"fit_only_law_cases":len(bank.read_bytes().splitlines()),
                "fit_only_law_source_photos":len(fit_ids),"scores_sha256":sha(scores),"inner_folds":rows,
                "fit_oof_covered_once":len(fit_ids),"calibration_ensemble_heads":len(rows)*len(RECIPE["base_seeds"]),
                "policy":{"temperatures":1.0,"code_threshold":0.50,"scene_threshold":0.50,"empty_code_bonus":0.12,
                          "additional_scene_requirement":0.08,"max_proposals_per_photo":1,"code_precision_gate":False,"draft":"empty"},
                "fit_oof_proposal_support":proposal_support(fit,proposals(fit,oof_code,oof_scene)),
                "calibration_proposal_support":proposal_support(cal,proposals(cal,cal_code,cal_scene)),
                "current_process_seconds_including_loaded_heads":round(time.monotonic()-started,2)}
        save_json(Path(f"results/clean-cal-v2-fit-specialist-fold{fold}-20261128.json"),result)
        print(f"Fold {fold} complete: {len(fit)} OOF and {len(cal)} calibration rows",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",type=int,nargs="+",choices=[2,4],default=[4,2])
    parser.add_argument("--new-head-limit",type=int)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=2):
        run(args.fold,args.new_head_limit)
