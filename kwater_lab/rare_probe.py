"""Measure fixed CLIP text-rank false candidate rates on the exposed 170-photo pilot."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from . import clip_probe


RARE_CODES = ("confined_space", "drowning_inundation", "thermal_weather")


def checked_texts(model_dir, prior_probe):
    codes = (*clip_probe.CODES, "no_visible_hazard")
    prompts = [f"A photo of an industrial worksite with {clip_probe.DESCRIPTIONS[k]}." for k in codes]
    if (prior_probe["revision"] != clip_probe.REVISION or
            dict(zip(codes, prompts)) != prior_probe["prompt_by_code"]):
        raise ValueError("Fixed prompt set differs from the recorded CLIP pilot")
    model = (model_dir / "text_model_quantized.onnx").read_bytes()
    tokenizer_bytes = (model_dir / "tokenizer.json").read_bytes()
    for name, raw in [("onnx/text_model_quantized.onnx", model), ("tokenizer.json", tokenizer_bytes)]:
        if hashlib.sha256(raw).hexdigest() != clip_probe.FILES[name]:
            raise ValueError(f"Pinned model asset changed: {name}")
    encoder = clip_probe.session(model)
    tokenizer = Tokenizer.from_buffer(tokenizer_bytes)
    vectors = []
    for prompt in prompts:
        tokens = np.array([tokenizer.encode(prompt).ids], dtype=np.int64)
        vectors.append(encoder.run(["text_embeds"], {"input_ids": tokens})[0][0])
    return codes, clip_probe.unit_rows(np.stack(vectors))


def fractions(rankings, codes):
    rows = len(rankings)
    indices = [codes.index(code) for code in RARE_CODES]
    results = {}
    for depth in (1, 3):
        top = rankings[:, :depth]
        hits = {name: np.any(top == ix, axis=1) for name, ix in zip(RARE_CODES, indices)}
        hits["any_of_three"] = np.any(np.stack(list(hits.values())), axis=0)
        results[f"top{depth}"] = {name: {"images": int(mask.sum()), "denominator": rows,
                                           "rate": int(mask.sum()) / rows} for name, mask in hits.items()}
    return results


def run(features, diagnostic, exposure, old_prompts, model_dir):
    paths = {"features": features, "diagnostic": diagnostic, "exposure": exposure,
             "prompt_record": old_prompts}
    raw = {key: path.read_bytes() for key, path in paths.items()}
    diag = json.loads(raw["diagnostic"])
    cohort = json.loads(raw["exposure"])
    ids = cohort["exposed_ids"]
    rows = diag["paired_out_of_prefix_predictions"]
    if (len(ids) != 170 or len(set(ids)) != 170 or len(rows) != 170 or
            {r["sample_id"] for r in rows} != set(ids) or set(diag["exposed_ids"]) != set(ids)):
        raise ValueError("Expected exactly the same 170 previously exposed IDs")
    positives = {code: sum(code in row["gold"] for row in rows) for code in RARE_CODES}
    if any(positives.values()):
        raise ValueError("The exposed slice contains a rare-code positive")
    with np.load(io.BytesIO(raw["features"]), allow_pickle=False) as archive:
        if (archive["scope"].item() != "exploratory_prefix_subset" or
                archive["framing"].item() != diag["framing"] or diag["framing"] != "full_frame" or
                archive["revision"].item() != diag["encoder_revision"] or
                diag["encoder_revision"] != clip_probe.REVISION or
                archive["labels_sha256"].item() != cohort["labels_sha256"] or
                archive["labels_sha256"].item() != diag["labels_sha256"] or
                archive["audit_sha256"].item() != diag["audit_sha256"]):
            raise ValueError("Audit, label source, framing or encoder provenance mismatch")
        cached_ids = archive["ids"].tolist()
        images = archive["features"]
        if (len(cached_ids) != 170 or set(cached_ids) != set(ids) or
                images.dtype != np.float32 or images.shape != (170, 512) or
                not np.allclose(np.linalg.norm(images, axis=1), 1, atol=2e-5)):
            raise ValueError("Invalid fully covered unit image-feature matrix")
    codes, texts = checked_texts(model_dir, json.loads(raw["prompt_record"]))
    cosine = images @ texts.T
    rankings = np.argsort(-cosine, axis=1, kind="stable")
    sorted_cosine = np.sort(cosine, axis=1)
    return {"scope": "previously exposed 170-photo prefix pilot, false candidate rates among negatives",
            "photos": len(ids), "rare_gold_counts": positives, "image_files_opened": 0,
            "code_order": list(codes), "framing": diag["framing"], "revision": clip_probe.REVISION,
            "encoder": clip_probe.REPO, "text_runtime": "ONNX Runtime CPUExecutionProvider, 4 threads",
            "text_model_sha256": clip_probe.FILES["onnx/text_model_quantized.onnx"],
            "tokenizer_sha256": clip_probe.FILES["tokenizer.json"],
            "versions": {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__},
            "ranking": "unit image and text cosine, all 11 hazards and no_visible_hazard together, "
                       "descending with published code order resolving equal scores",
            "ties_seen_at_top1_boundary": int((sorted_cosine[:, -2] == sorted_cosine[:, -1]).sum()),
            "ties_seen_at_top3_boundary": int((sorted_cosine[:, -4] == sorted_cosine[:, -3]).sum()),
            "threshold_tuning": False, "incident_independence_verified": False,
            "false_candidate_rates": fractions(rankings, codes), "positive_recall": None,
            "limitation": "No rare-class positives are present. These candidate rankings do not "
                          "measure positive recall, final submitted false alarms, or an official score.",
            "input_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "diagnostic", "exposure", "old-prompts", "model-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.features, args.diagnostic, args.exposure, args.old_prompts, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["false_candidate_rates"]))


if __name__ == "__main__":
    main()
