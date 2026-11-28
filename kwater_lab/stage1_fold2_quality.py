"""Bind real generation controls to Fold 2 output and score a single quality contrast."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from .fold2_8b_baseline import BASE_REVISION, EVAL_SEGMENTS, order_digest, sha_json, write_json
from .fold2_8b_paired_score import score_panel
from .fold2_v3_epoch1 import LABELS, LAWS, MANIFEST, MASK, split
from .fold4_same_verifier import digest, read
from .law_reranker_11code import read_selected_cases
from .metrics import CODES, decode_target, read_index
from .sub78_public200_compare import match_changes

FP8_REVISION = "9cdc6310a8cb770ce18efaf4e9935334512aee45"
REVISION = {"bf16": BASE_REVISION, "fp8": FP8_REVISION}
FACTORS = {"prompt": {"prompt_sha256"}, "precision": {"precision", "base_revision"}, "pixels": {"max_pixels"}}
FORMAT = "kwater-stage1-fold2-generation-v1"
READINESS = Path("results/sub78-fold2-prospective-readiness-20261128.json")
RISK_FOCUS = ("fall_height", "struck_by_lifting", "caught_in_machine")
MIN_PIXELS = 131072
MAX_NEW_TOKENS = 620
PACKAGES = ("torch", "transformers", "peft", "accelerate", "tokenizers", "Pillow", "triton",
            "pytorch-triton", "bitsandbytes", "safetensors")
CONTROL_KEYS = frozenset({
    "precision", "base_revision", "base_nonquant_sha256", "tokenizer_sha256", "processor_assets_sha256",
    "generation_assets_sha256", "adapter_sha256", "adapter_config_sha256", "train_configuration_sha256",
    "prompt_sha256", "min_pixels", "max_pixels", "max_new_tokens", "do_sample", "use_cache",
    "runtime_sha256", "formatter_sha256", "engine_sha256", "law_corpus_sha256", "train_case_bank_sha256",
})


def runtime_fingerprint():
    import torch

    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    device = torch.cuda.get_device_properties(0)
    return sha_json({"platform": sys.platform, "python": list(sys.version_info[:3]), "packages": packages,
                     "cuda": torch.version.cuda, "gpu": device.name, "cc": [device.major, device.minor]})


def generation_controls(*, base, base_revision, precision, adapter, train_configuration, prompt_path, max_pixels,
                        formatter_path, engine_path, train_cases):
    """Measure semantic metadata and exact inputs before invoking the common generator."""
    config = read(base / "config.json")
    quant = config.pop("quantization_config", None)
    if (precision not in REVISION or REVISION[precision] != base_revision
            or (quant is not None) != (precision == "fp8")
            or config.get("model_type") != "qwen3_vl"):
        raise ValueError("Model metadata do not match the pinned Stage 1 precision")
    tokenizer = {name: read(base / name) for name in
                 ("tokenizer.json", "tokenizer_config.json", "chat_template.json", "vocab.json")}
    merges_path = base / "merges.txt"
    if merges_path.is_file():
        merges = merges_path.read_text(encoding="utf-8").splitlines()
        if merges and merges[0].startswith("#version:"):
            merges = merges[1:]
        embedded = tokenizer["tokenizer.json"]["model"]["merges"]
        if merges != [" ".join(row) if isinstance(row, list) else row for row in embedded]:
            raise ValueError("Supplemental BPE merges differ from the fast tokenizer")
    processor = {name: read(base / name) for name in ("preprocessor_config.json", "video_preprocessor_config.json")}
    return {
        "precision": precision, "base_revision": base_revision, "base_nonquant_sha256": sha_json(config),
        "tokenizer_sha256": sha_json(tokenizer), "processor_assets_sha256": sha_json(processor),
        "generation_assets_sha256": sha_json(read(base / "generation_config.json")),
        "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": digest(adapter / "adapter_config.json"),
        "train_configuration_sha256": sha_json(train_configuration), "prompt_sha256": digest(prompt_path),
        "min_pixels": MIN_PIXELS, "max_pixels": max_pixels, "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False, "use_cache": True, "runtime_sha256": runtime_fingerprint(),
        "formatter_sha256": digest(formatter_path), "engine_sha256": digest(engine_path),
        "law_corpus_sha256": digest(LAWS), "train_case_bank_sha256": sha_json(train_cases),
    }


def receipt_cache_key(controls):
    return f"stage1:fold2:{sha_json(controls)}"


def write_generation_receipt(path, predictions, outer, segments, controls, *, loaded_adapter_verified=None, extras=None):
    """Fingerprint emitted predictions, ordered input and actual generation cache files."""
    if loaded_adapter_verified is not None and type(loaded_adapter_verified) is not bool:
        raise ValueError("Loaded adapter evidence must be a Boolean or unknown")
    write_json(path, {**(extras or {}), "format": FORMAT, "controls": controls,
                      "loaded_adapter_verified": loaded_adapter_verified, "prediction_sha256": digest(predictions),
                      "eval_order_sha256": order_digest(outer), "cache_key": receipt_cache_key(controls),
                      "segments": segments})


def validate_arm(path, receipt_path, outer):
    receipt = read(receipt_path)
    controls = receipt.get("controls", {})
    if receipt.get("format") != FORMAT or controls.keys() != CONTROL_KEYS:
        raise ValueError("A generation receipt with prospective Stage 1 controls is required")
    precision = controls["precision"]
    if (precision not in REVISION or REVISION[precision] != controls["base_revision"]
            or controls["min_pixels"] != MIN_PIXELS or controls["max_new_tokens"] != MAX_NEW_TOKENS
            or controls["max_pixels"] not in (393216, 655360, 786432)
            or controls["do_sample"] is not False or controls["use_cache"] is not True):
        raise ValueError("Precision, pixels or decoding differ from registered Stage 1 settings")
    for name, value in controls.items():
        if name.endswith("_sha256") and (not isinstance(value, str) or len(value) != 64
                                         or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"Invalid fingerprint for generation control {name}")
    cache_key = receipt_cache_key(controls)
    if (digest(path) != receipt["prediction_sha256"] or receipt["eval_order_sha256"] != order_digest(outer)
            or receipt["cache_key"] != cache_key):
        raise ValueError("Prediction or cohort differs from the upstream generation receipt")
    decoded = read_index(path, lambda row: decode_target(row, local_ids=True), limit_size=True)
    if list(decoded) != [row["sample_id"] for row in outer]:
        raise ValueError("Both predictions must cover all 197 photos in canonical order")
    segments, prefix = [], None
    for segment, (start, end, batch) in zip(receipt["segments"], EVAL_SEGMENTS, strict=True):
        raw = receipt_path.parent / f"raw-{start}-{end}.json"
        raw_sha = digest(raw)
        if (segment["start"], segment["end"], segment["raw_sha256"]) != (start, end, raw_sha):
            raise ValueError("Raw generation cache differs from the upstream receipt")
        cache, stats = read(raw), segment["stats"]
        identity = cache["identity"]
        items = outer[start:end]
        if (identity["samples"] != [[row["sample_id"], row["image"]] for row in items]
                or identity["image_sha256"] != {row["image"]: row["sha256"] for row in items}
                or identity["processor_size"] != {"shortest_edge": MIN_PIXELS, "longest_edge": controls["max_pixels"]}
                or identity["model_key"] != cache_key or identity["max_new_tokens"] != MAX_NEW_TOKENS
                or identity["batch_size"] != batch or cache["effective_batch_size"] != batch
                or (stats["photos"], stats["initial_batch_size"], stats["final_effective_batch_size"]) !=
                (end - start, batch, batch)
                or list(cache["outputs"]) != [row["sample_id"] for row in items]
                or any(row["sample_id"] != key for key, row in cache["outputs"].items())):
            raise ValueError("Observed generation order, photo identity, pixels or batches differ from the contract")
        if prefix is not None and prefix != identity["prefix_sha256"]:
            raise ValueError("Rendered prompt differs between batches")
        prefix = identity["prefix_sha256"]
        observed = {"parsed": sum(row["parse_error"] is None and row["parsed"] is not None
                                  for row in cache["outputs"].values()),
                    "strict_schema": sum(not row["vlm_schema_errors"] for row in cache["outputs"].values()),
                    "token_cap_hits": sum(row["token_cap_hit"] for row in cache["outputs"].values())}
        if observed != {"parsed": stats["raw_json_parse_success"], "strict_schema": stats["strict_vlm_schema_success"],
                        "token_cap_hits": stats["token_cap_hits"]}:
            raise ValueError("Generation parse or token-cap totals differ from raw outputs")
        segments.append({"start": start, "end": end, "batch_size": batch, "raw_sha256": raw_sha, **observed})
    evidence = receipt.get("loaded_adapter_verified")
    if evidence is not None and type(evidence) is not bool:
        raise ValueError("Invalid loaded-adapter evidence")
    return decoded, {"controls": controls, "prediction_sha256": receipt["prediction_sha256"],
                     "receipt_sha256": digest(receipt_path), "rendered_prefix_sha256": prefix,
                     "loaded_adapter_verified": evidence, "segments": segments}


def audit_changed_factor(first, second, factor):
    changed = {name for name in CONTROL_KEYS if first["controls"][name] != second["controls"][name]}
    if changed != FACTORS[factor] or (
        (first["rendered_prefix_sha256"] != second["rendered_prefix_sha256"]) != (factor == "prompt")
    ):
        raise ValueError(f"Expected exactly the {factor} contrast, observed config changes: {', '.join(sorted(changed))}")
    return sorted(changed)


def compare_panel(ids, groups, gold, first, second, severe):
    panel = score_panel(ids, groups, gold, first, second, severe)
    panel["metrics"] = {key: panel["metrics"][old] for key, old in (("control", "4b"), ("candidate", "8b"))}
    panel["active_codes_by_arm"] = {key: [code for code in CODES if arm["per_code"][code]["f1"] is not None]
                                    for key, arm in panel["metrics"].items()}
    panel["shared_active_macro_f1"] = {key: panel["shared_active_macro_f1"][old]
                                     for key, old in (("control", "4b"), ("candidate", "8b"))}
    for name in ("supplemental_shared_active", "fixed_8"):
        item = panel["paired_group_bootstrap"][name]
        item["delta_candidate_minus_control"] = item.pop("delta_8b_minus_4b")
    multi = [sid for sid in ids if len(gold[sid]) > 1]
    possible_codes = {(sid, code) for sid in multi for code in gold[sid]}
    panel["paired_outcomes"].update(
        multi_hazard_true_codes=match_changes(
            {(sid, code) for sid, code in possible_codes if code in first[sid]},
            {(sid, code) for sid, code in possible_codes if code in second[sid]}, len(possible_codes)),
        multi_hazard_exact_sets=match_changes({sid for sid in multi if first[sid] == gold[sid]},
                                             {sid for sid in multi if second[sid] == gold[sid]}, len(multi)),
    )
    panel["focused_code_changes"] = {}
    for code in RISK_FOCUS:
        positives = [sid for sid in ids if code in gold[sid]]
        negatives = [sid for sid in ids if code not in gold[sid]]
        panel["focused_code_changes"][code] = {
            "target_photos": len(positives),
            "correct_detection": match_changes({sid for sid in positives if code in first[sid]},
                                               {sid for sid in positives if code in second[sid]}, len(positives)),
            "no_false_positive": match_changes({sid for sid in negatives if code not in first[sid]},
                                               {sid for sid in negatives if code not in second[sid]}, len(negatives)),
        }
    return panel


def compare(control, candidate, control_receipt, candidate_receipt, factor):
    train, outer = split()
    ids = [row["sample_id"] for row in outer]
    mask_sha = digest(MASK)
    if mask_sha != read(READINESS)["sensitivity_mask"]["recipe_sha256"]:
        raise ValueError("The prospective 55-photo mask changed")
    selected = set(read(MASK)["prospective_mask_retained_ids"])
    first, first_receipt = validate_arm(control, control_receipt, outer)
    second, second_receipt = validate_arm(candidate, candidate_receipt, outer)
    changed = audit_changed_factor(first_receipt, second_receipt, factor)
    train_ids = {row["sample_id"] for row in train}
    cases = {row["sample_id"]: row for row in read_selected_cases(LABELS, train_ids)}
    if cases.keys() != train_ids:
        raise ValueError("Missing canonical training cases")
    local_sources = {"train_case_bank_sha256": sha_json([cases[row["sample_id"]] for row in train]),
                     "law_corpus_sha256": digest(LAWS)}
    if any(arm["controls"][key] != expected for arm in (first_receipt, second_receipt)
           for key, expected in local_sources.items()):
        raise ValueError("The scored law corpus or training case bank differs from generation")
    gold = {row["sample_id"]: set(row["codes"]) for row in outer}
    official = {row["sample_id"]: row["target"]["hazards"] for row in read_selected_cases(LABELS, set(ids))}
    if official.keys() != gold.keys() or any(set(official[sid]) != gold[sid] for sid in ids):
        raise ValueError("Official labels disagree with the frozen Fold 2 roster")
    severe = {sid: {code for code, record in items.items() if record["risk_level"]["severity"]["score"] == 4}
              for sid, items in official.items()}
    groups = {row["sample_id"]: row["group_id"] for row in outer}
    panels = {name: compare_panel(members, groups, gold, first, second, severe) for name, members in (
        ("primary_197", ids), ("sensitivity_55", [sid for sid in ids if sid in selected]))}
    if factor == "precision":
        loaded = ("verified" if all(arm["loaded_adapter_verified"] is True for arm in (first_receipt, second_receipt))
                  else "unverified")
        precision_gates = {}
        for name, panel in panels.items():
            delta = panel["paired_group_bootstrap"]["active_per_arm"]["macro_f1_delta_b_minus_a"]
            precision_gates[name] = {"threshold": 0.005, "absolute_active_macro_f1_delta": abs(delta),
                                     "point_threshold_met": abs(delta) <= 0.005}
    else:
        loaded, precision_gates = None, None
    return {"scope": "Paired Fold 2 local development quality. The historical 55-photo sensitivity includes prior exposure.",
            "changed_factor": factor, "changed_control_fields": changed, "manifest_sha256": digest(MANIFEST),
            "sensitivity_mask_sha256": mask_sha, "cohort_order_sha256": order_digest(outer),
            "generation": {"control": first_receipt, "candidate": second_receipt}, "panels": panels,
            "precision_point_gates": precision_gates, "precision_loaded_adapter_evidence": loaded,
            "runtime_parity_status": "unmeasured_by_this_scorer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("control", "candidate", "control-receipt", "candidate-receipt", "result"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--factor", choices=tuple(FACTORS), required=True)
    args = parser.parse_args()
    if any(not path.resolve().is_relative_to(Path("data").resolve())
           for path in (args.control, args.candidate, args.control_receipt, args.candidate_receipt)):
        parser.error("Predictions, raw generations and source receipts must stay in data/")
    write_json(args.result, compare(args.control, args.candidate, args.control_receipt, args.candidate_receipt, args.factor))


if __name__ == "__main__":
    main()
