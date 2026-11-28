"""Replay the prospectively defined, but not ID-frozen, Fold 2 exposure sensitivity."""
from hashlib import sha256
import json
from pathlib import Path

from .fold0_group_sensitivity import arm_metrics
from .fold2_v3_epoch1 import DEFAULT_RUN, MASK, digest, read, save, split
from .metrics import decode_target, read_index
from .sub78_fold2_readiness import CURATION_PAIRS, MANIFEST, RESULTS, ROI, ROI_AGG, annotations


def main():
    readiness = read(RESULTS / "sub78-fold2-prospective-readiness-20261128.json")
    main_result = read(RESULTS / "fold2-v3-4b-epoch1-20261128.json")
    expected = readiness["sensitivity_mask"]
    if digest(MASK) != expected["recipe_sha256"]:
        raise ValueError("Original 55-photo mask changed")
    for path in (ROI, ROI_AGG, CURATION_PAIRS, MANIFEST):
        if digest(path) != readiness["known_source_sha256"][path.as_posix()]:
            raise ValueError(f"Original source changed: {path.name}")
    dev = {r["sample_id"]: r for r in read(MANIFEST)}
    _, outer = split()
    groups = {r["sample_id"]: r["group_id"] for r in outer}
    kept = set(read(MASK)["prospective_mask_retained_ids"])
    sources = [entry["source"] for entry in read(ROI)]
    sources.extend(row for pair in read(CURATION_PAIRS) for row in pair["members"])
    if any(dev[row["sample_id"]]["sha256"] != row["sha256"] for row in sources):
        raise ValueError("Original photo-hash join changed")
    exposed_groups = {dev[row["sample_id"]]["group_id"] for row in sources}
    chosen = [sid for sid, group in groups.items() if sid in kept and group not in exposed_groups]
    gold = {r["sample_id"]: set(r["codes"]) for r in outer}
    ann = annotations(set(gold))
    if any(set(ann[sid]) != codes for sid, codes in gold.items()):
        raise ValueError("Current labels do not match the original Fold 2 manifest")
    severe = {sid: {code for code, item in row.items() if item["risk_level"]["severity"]["score"] == 4}
              for sid, row in ann.items()}

    def support(ids):
        return {"photos": len(ids), "groups": len({groups[sid] for sid in ids}),
                "normal": sum(not gold[sid] for sid in ids), "severity4_photos": sum(bool(severe[sid]) for sid in ids),
                "severity4_code_instances": sum(len(severe[sid]) for sid in ids)}

    frozen = [(list(groups), readiness["primary_197"]),
              (kept, expected["retained"]),
              (chosen, expected["optional_stricter_visual_exposure_sensitivity_roi_and_curation"])]
    if any(any(reference[k] != value for k, value in support(ids).items()) for ids, reference in frozen):
        raise ValueError("Original 197/55/19 denominators did not reproduce")
    path = DEFAULT_RUN / "epoch1-bf16-predictions.jsonl"
    prediction_hash = digest(path)
    pred = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    if prediction_hash != main_result["inference"]["prediction_sha256"] or list(pred) != list(groups):
        raise ValueError("Prediction file disagrees with original 197-row receipt or order")
    frozen_names = sha256(json.dumps(chosen, separators=(",", ":")).encode()).hexdigest()
    save(DEFAULT_RUN / "exposure19-reconstructed-ids-local.json",
         {"ordered_ids": chosen, "canonical_ordered_id_list_sha256": frozen_names})
    metrics = arm_metrics({sid: gold[sid] for sid in chosen}, {sid: pred[sid] for sid in chosen},
                          {sid: groups[sid] for sid in chosen}, severe)
    save(Path("results/fold2-v3-4b-epoch1-exposure19-replay-20261128.json"),
         {"scope": "Exploratory replay of the pre-inference rule, with membership reconstructed after inference. Original 19 IDs were not separately frozen. No effect on primary results or gates.",
          "support": support(chosen), "original_197_row_prediction_sha256": prediction_hash,
          "reconstructed_ordered_ids_sha256": frozen_names, "input_mask_sha256": digest(MASK),
          "input_source_sha256": {path.as_posix(): digest(path) for path in (ROI, ROI_AGG, CURATION_PAIRS, MANIFEST)},
          "three_published_denominators_reproduced": True, "metrics": metrics})


if __name__ == "__main__":
    main()
