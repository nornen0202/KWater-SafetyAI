"""Paired, in-memory LoRA scale diagnostic on an ID-selected slice of the exposed pilot."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from .metrics import evaluate
from .train_head import file_hash, save_atomic


def sample(exposure):
    selected = []
    for prefix, ids in exposure["by_prefix"].items():
        order = sorted(ids, key=lambda s: hashlib.sha256(("kwater-scale-a-v1/" + s).encode()).digest())
        selected.extend((s, prefix) for s in order[:3])
    return selected


@contextmanager
def scale_default_adapter(model, factor):
    layers = [m for m in model.modules() if "default" in getattr(m, "lora_A", ())
              and "default" in getattr(m, "scaling", {})]
    if not layers:
        raise ValueError("No active default LoRA layers")
    original = [m.scaling["default"] for m in layers]
    try:
        for m, value in zip(layers, original):
            m.scaling["default"] = factor * value
        yield len(layers)
    finally:
        for m, value in zip(layers, original):
            m.scaling["default"] = value


def write_json(path, record):
    save_atomic(path, lambda f: f.write((json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()))


def competition_codes(raw, mapping):
    """Map validated, pre-format model hazards in the same order as the submission wrapper."""
    codes = []
    for item in raw.get("hazards", []):
        code = mapping.get(item["code"])
        if code and code not in codes:
            codes.append(code)
            if len(codes) == 3:
                break
    return frozenset(codes)


def run(root, output, local):
    base = root / "data/baseline"
    audit_path, exposure_path = root / "results/train_audit_full.json", root / "data/prefix-pilot/exposure.json"
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = json.loads((root / "results/official_baseline_10.json").read_text(encoding="utf-8"))
    if (audit["labels_sha256"] != exposure["labels_sha256"] or
            file_hash(base / "main.py") != expected["main_sha256"]):
        raise ValueError("Baseline or pilot provenance mismatch")
    selected = sample(exposure)
    if len(selected) != 18 or len(set(s for s, _ in selected)) != 18:
        raise ValueError("Expected the registered six-prefix pilot, three images per prefix")
    inputs, gold, groups = [], {}, {}
    for sid, prefix in selected:
        row = audit["inventory"][sid]
        if file_hash(root / "data/train" / row["image"]) != row["sha256"]:
            raise ValueError("Image differs from the audited training package")
        anon = hashlib.sha256(("kwater/scale/" + sid).encode()).hexdigest()[:32]
        inputs.append({"sample_id": anon, "image": row["image"]})
        gold[anon] = frozenset(row["codes"])
        groups[anon] = prefix
    os.environ["KW_BATCH"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_HUB_OFFLINE"] = "1"
    spec = importlib.util.spec_from_file_location("competition_scale_probe", base / "main.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.seed_all(42)
    t0 = time.perf_counter()
    print("Loading official adapter for paired scale test", flush=True)
    model, processor = entry.load_model(str(base / "base_model"), "sdpa", str(base / "adapter"), 384*32*32)
    load_seconds = time.perf_counter() - t0
    metrics, codes = {}, {}
    for factor in (1.0, 0.75):
        start = time.perf_counter()
        with scale_default_adapter(model, factor) as layer_count, patch.object(
                entry, "load_model", return_value=(model, processor)):
            rows = entry.infer_batched(inputs, root / "data/train")
        elapsed = time.perf_counter() - start
        if [r["sample_id"] for r in rows] != [r["sample_id"] for r in inputs]:
            raise ValueError("Unexpected model output IDs")
        path = local / f"generated-scale-{factor}.json"
        write_json(path, rows)
        usable = {r["sample_id"]: not entry.validate_prediction(r["prediction"]) for r in rows}
        pred = {r["sample_id"]: competition_codes(r["prediction"], entry.CODE_MAP)
                if usable[r["sample_id"]] else frozenset() for r in rows}
        codes[factor] = pred
        metrics[str(factor)] = {"inference_seconds": elapsed, "model_parsed": sum(usable.values()),
                                "fallback_rows": len(rows)-sum(usable.values()), "scaled_layers": layer_count,
                                "raw_sha256": file_hash(path), "metrics": evaluate(gold, pred, groups)}
    first, second = codes[1.0], codes[0.75]
    results = {"scope": "Training-domain diagnostic, organizer LoRA was fit on training material, no independent generalization claim.",
               "design": "Three SHA-ordered known exposed samples from each of six ID prefixes, labels unused for selection. Single GPU load. Original inference function; no law retrieval or LLM scoring.",
               "photos": len(selected), "cohort_sha256": hashlib.sha256("\n".join(sorted(gold)).encode()).hexdigest(),
               "exposure_sha256": file_hash(exposure_path), "main_sha256": expected["main_sha256"],
               "adapter_sha256": file_hash(base / "adapter/adapter_model.safetensors"),
               "batch_size": 1, "seed": 42, "load_seconds_after_import": load_seconds, "conditions": metrics,
               "code_set_changed": sum(first[k] != second[k] for k in gold),
               "exact_set_gains": sum(first[k] != gold[k] and second[k] == gold[k] for k in gold),
               "exact_set_losses": sum(first[k] == gold[k] and second[k] != gold[k] for k in gold)}
    write_json(output, results)
    print({k: v["metrics"]["macro_f1"] for k,v in metrics.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/adapter_scale_pilot.json"))
    parser.add_argument("--local", type=Path, default=Path("data/adapter-scale-pilot"))
    args = parser.parse_args()
    run(args.root.resolve(), args.output, args.local)
