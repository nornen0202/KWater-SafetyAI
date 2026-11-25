"""Audit CCTV source/label paths and repeat the reference test after removing mismatches."""

import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

from .aihub_ablation import AUDIT, EXPOSURE, FEATURES, REFERENCE, ROOT, SOURCES, centers, official_refs, paired, read
from .metrics import CODES, evaluate
from .prefix_pilot import filtered_training
from .train_head import decisions, file_hash, fit, probabilities, save_atomic
from .weight_pilot import pilot_data


def run():
    primary = read(Path("results/aihub_ablation.json"))
    folder = ROOT / "71793"
    with ZipFile(folder / "sample.zip") as archive:
        photos, labels = paired(archive, archive)
        rows = [s for s, path in photos.items() if "지역 라벨링" in PurePosixPath(path).parts[2]]
        def group(path):
            return PurePosixPath(path).parts[3:-1]
        folder_mismatch = {s for s in rows if group(photos[s]) != group(labels[s])}
        name_mismatch = {s for s in rows if not any(PurePosixPath(i["file_name"]).stem == s
                         for i in json.loads(archive.read(labels[s]))["images"])}
        audit = {"cctv": len(rows), "source_folders": len({group(photos[s]) for s in rows}),
                 "annotation_folders": len({group(labels[s]) for s in rows}),
                 "different_folder": len(folder_mismatch), "different_internal_filename": len(name_mismatch),
                 "both_issues": len(folder_mismatch & name_mismatch)}
    selected = {}
    for source in SOURCES:
        with np.load(ROOT / source / "pilot-ablation-features.npz", allow_pickle=False) as cache:
            proof = json.loads(str(cache["provenance"]))
            expected = primary["source_diagnostics"][source]
            if proof["source_sha256"] != expected["source_sha256"] or proof["vision_revision"] != primary["model_revision"]:
                raise ValueError("Source or image model changed since the primary comparison")
            ids = [r["photo"] for r in json.loads(str(cache["rows"]))[:expected["reference_groups"]]]
            mask = [source != "71793" or s not in folder_mismatch | name_mismatch for s in ids]
            selected[source] = cache["features"][:len(ids)][mask]
            if source == "71793":
                audit["affected_initial_references"] = len(ids) - sum(mask)
    n = min(map(len, selected.values()))
    banks = {source: centers(x[:n]) for source, x in selected.items()}
    folds, feat, gold, linked = pilot_data(AUDIT, EXPOSURE, FEATURES, read(REFERENCE))
    inventory = read(AUDIT)["inventory"]
    hashes = {s: int(inventory[s]["dhash64"], 16) for s in gold}
    groups = {s: g for g, ids in folds.items() for s in ids}
    predictions = {k: {} for k in ("official_references", *SOURCES)}
    for valid in folds.values():
        train = filtered_training(gold, valid, linked)
        x, test = [np.stack([feat[s] for s in keys]) for keys in (train, valid)]
        y = np.array([[c in gold[s] for c in CODES] for s in train], dtype=float)
        subset = official_refs(train, groups, hashes, n)
        matched = {"official_references": centers(np.stack([feat[s] for s in subset])), **banks}
        for key, bank in matched.items():
            head = fit(np.column_stack([x, x @ bank.T]), y, .01, 250)
            score = probabilities(head, np.column_stack([test, test @ bank.T]))
            predictions[key].update(decisions(valid, score, .5))
    return {"scope": "post-hoc sensitivity on the same exposed 170; reference budget matched after filtering",
            "primary_result_sha256": file_hash(Path("results/aihub_ablation.json")),
            "audit": audit, "matched_reference_count": n,
            "metrics": {k: evaluate(gold, p, groups) for k, p in predictions.items()}}


if __name__ == "__main__":
    result = run()
    save_atomic(Path("results/busan_alignment.json"),
                lambda f: f.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()))
    print(result["audit"], "reference count", result["matched_reference_count"], flush=True)
