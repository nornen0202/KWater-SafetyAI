"""Measure a paired local RTX 3060 NF4 inference-pixel contrast on frozen Fold 2 photos."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

from .fold2_8b_baseline import BASE_CONFIG_SHA, BASE_REVISION, order_digest, write_json
from .fold2_v3_epoch1 import LABELS, LAWS, MANIFEST, MASK, PROMPT, split
from .fold4_same_verifier import digest, read


BASE = Path(".runtime/stage1-known-bf16")
ADAPTER = Path("data/fold2-8b-matched-epoch1/fit/epoch-1")
MATCHED = Path("results/fold2-matched-4b-8b-epoch1-20261128.json")
IMAGE_ROOT = Path("data/train")
WORK = Path("data/stage1-local-nf4-resolution-55")
RESULT = Path("results/stage1-local-nf4-resolution-55-20261129.json")
V4_WORK = Path("data/stage1-local-nf4-prompt-v4-55")
V4_RESULT = Path("results/stage1-local-nf4-prompt-v4-55-20261129.json")
V4_PROMPT = Path("kwater_lab/system_prompt_v4.txt")
V4_PROMPT_SHA = "023304292db08d0a14c358fa731842a50f21bf579d5b7030093692c4cc86c176"
BASE_RECEIPT = Path("results/fold4-8b-epoch1-research-package-20261128.json")
BASE_REFERENCE = Path("data/submissions/research-8b-fold4-e1-stage-20261128/base_model")
CAPS = (393216, 655360)
BATCH_SIZE, TOKEN_LIMIT, SEED = 1, 620, 42


def fixed_cohort():
    matched = read(MATCHED)
    frozen = matched["recipe"]["training"]
    if digest(MANIFEST) != frozen["manifest_sha256"] or digest(MASK) != frozen["mask_sha256"]:
        raise ValueError("Fold 2 roster or fixed sensitivity mask differs from the paired baseline")
    train, outer = split()
    kept = set(read(MASK)["prospective_mask_retained_ids"])
    rows = [row for row in outer if row["sample_id"] in kept]
    if len(rows) != 55 or len({row["group_id"] for row in rows}) != 34:
        raise ValueError("Expected the fixed 55-photo/34-group Fold 2 cohort")
    return matched, train, rows


def identity(matched, rows):
    frozen = matched["recipe"]["training"]
    source = read(BASE / "source.json")
    if source["revision"] != BASE_REVISION or digest(BASE / "config.json") != BASE_CONFIG_SHA:
        raise ValueError("Expected the pinned resident 8B BF16 basis for NF4 loading")
    for item in source["files"]:
        if item["rfilename"].endswith(".safetensors") and (BASE / item["rfilename"]).stat().st_size != item["size"]:
            raise ValueError("Pinned base shard size differs")
    adapter_sha = digest(ADAPTER / "adapter_model.safetensors")
    prompt_sha = digest(PROMPT)
    if adapter_sha != matched["generated"]["adapter_sha256"] or prompt_sha != frozen["prompt_sha256"]:
        raise ValueError("Expected the actual matched 8B first-epoch adapter and original v3 prompt")
    return {"precision": "nf4_double_quant_bf16_compute", "base_revision": BASE_REVISION,
            "base_config_sha256": BASE_CONFIG_SHA, "adapter_sha256": adapter_sha,
            "adapter_config_sha256": digest(ADAPTER / "adapter_config.json"), "prompt_sha256": prompt_sha,
            "cohort_order_sha256": order_digest(rows), "mask_sha256": frozen["mask_sha256"], "photos": len(rows),
            "batch_size": BATCH_SIZE, "max_new_tokens": TOKEN_LIMIT, "seed": SEED}


def generate(work):
    import msvcrt

    WORK.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with (WORK / "process.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        print(f"LOCAL_NF4_RESOLUTION_PID {os.getpid()}", flush=True)
        run_arms(work)


def run_arms(work):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch

    from .vlm_11code import infer_batched_vlm, load_qwen_model, load_qwen_processor, seed_all

    matched, _, rows = fixed_cohort()
    controls = identity(matched, rows)
    if (torch.cuda.device_count() != 1 or "RTX 3060" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("This local NF4 comparison requires one RTX 3060 with BF16 support")
    controls["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(),
                           **{name: version(name) for name in ("transformers", "peft", "accelerate", "bitsandbytes", "Pillow")}}
    frozen = work / "generation-controls.json"
    if frozen.exists() and read(frozen) != controls:
        raise ValueError("Previously started generation has different controls")
    write_json(frozen, controls)
    print(f"LOCAL_NF4_RESOLUTION_VERIFYING {len(rows)} photo digests", flush=True)
    for row in rows:
        if digest(IMAGE_ROOT / row["image"]) != row["sha256"]:
            raise ValueError("A fixed cohort photo differs from its frozen digest")
    print("LOCAL_NF4_RESOLUTION_PHOTOS_VERIFIED", flush=True)
    seed_all(SEED)
    model = None
    context = {}

    def load_model():
        nonlocal model
        if model is None:
            print("LOCAL_NF4_RESOLUTION_LOADING", flush=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            model = load_qwen_model(BASE, use_4bit=True, adapter_dir=ADAPTER)
            torch.cuda.synchronize()
            if (not getattr(model, "is_loaded_in_4bit", False)
                    or any(p.device != torch.device("cuda:0") for p in model.parameters())):
                raise RuntimeError("Expected the complete NF4 model on cuda:0 without offload")
            context.update(model_load_seconds=round(time.perf_counter() - start, 3),
                           load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           load_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            print(f"LOCAL_NF4_RESOLUTION_LOADED {context}", flush=True)
        return model

    prompt = PROMPT.read_text(encoding="utf-8")
    for cap in CAPS:
        receipt_path = work / f"arm-{cap}.json"
        cache = work / f"raw-{cap}.json"
        if receipt_path.is_file():
            receipt = read(receipt_path)
            if (receipt["controls_sha256"] != digest(frozen) or receipt["max_pixels"] != cap
                    or receipt["raw_sha256"] != digest(cache)):
                raise ValueError("Completed arm receipt differs from frozen generation")
            continue
        print(f"LOCAL_NF4_RESOLUTION_PROCESSOR max_pixels={cap}", flush=True)
        processor = load_qwen_processor(BASE, cap)
        load_model()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_all(SEED)
        print(f"LOCAL_NF4_RESOLUTION_ARM max_pixels={cap} batch={BATCH_SIZE}", flush=True)
        _, stats = infer_batched_vlm(
            model, processor, rows, IMAGE_ROOT, prompt, batch_size=BATCH_SIZE, max_new_tokens=TOKEN_LIMIT,
            cache_path=cache, cache_key=f"stage1-local-nf4:{BASE_REVISION}:{controls['adapter_sha256']}:px{cap}",
        )
        torch.cuda.synchronize()
        identity_record = read(cache)["identity"]
        if identity_record["image_sha256"] != {row["image"]: row["sha256"] for row in rows}:
            raise ValueError("Raw generation cache differs from fixed photos")
        write_json(receipt_path, {"controls_sha256": digest(frozen), "max_pixels": cap, "raw_sha256": digest(cache),
                                  "stats": stats, **context, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def load_arm(work, cap, rows, controls, frozen_sha):
    receipt = read(work / f"arm-{cap}.json")
    raw = work / f"raw-{cap}.json"
    if (receipt["controls_sha256"] != frozen_sha or receipt["raw_sha256"] != digest(raw)
            or receipt["max_pixels"] != cap):
        raise ValueError("Arm receipt does not bind the raw output and frozen generation settings")
    cache = read(raw)
    cache_identity = cache["identity"]
    stats = receipt["stats"]
    outputs = cache["outputs"]
    if (cache_identity["samples"] != [[row["sample_id"], row["image"]] for row in rows]
            or cache_identity["image_sha256"] != {row["image"]: row["sha256"] for row in rows}
            or cache_identity["batch_size"] != BATCH_SIZE or cache["effective_batch_size"] != BATCH_SIZE
            or cache_identity["max_new_tokens"] != TOKEN_LIMIT
            or cache_identity["processor_size"] != {"shortest_edge": 131072, "longest_edge": cap}
            or cache_identity["model_key"] != f"stage1-local-nf4:{BASE_REVISION}:{controls['adapter_sha256']}:px{cap}"
            or (stats["photos"], stats["initial_batch_size"], stats["final_effective_batch_size"]) != (len(rows), BATCH_SIZE, BATCH_SIZE)
            or list(outputs) != [row["sample_id"] for row in rows]
            or any(key != value["sample_id"] for key, value in outputs.items())
            or stats["raw_json_parse_success"] != sum(row["parse_error"] is None and row["parsed"] is not None for row in outputs.values())
            or stats["strict_vlm_schema_success"] != sum(not row["vlm_schema_errors"] for row in outputs.values())
            or stats["token_cap_hits"] != sum(row["token_cap_hit"] for row in outputs.values())
            or not 0 < receipt["cuda_peak_allocated_bytes"] <= receipt["cuda_peak_reserved_bytes"]):
        raise ValueError("Raw outputs have incomplete or incomparable settings")
    return receipt, cache_identity, outputs


def relabel_score(report, labels=tuple(str(cap) for cap in CAPS), delta_name="delta_655360_minus_393216"):
    # The established paired scorer names its first and second arms 4b and 8b.
    for section in ("metrics", "shared_active_macro_f1"):
        report[section] = {label: report[section][name] for label, name in zip(labels, ("4b", "8b"), strict=True)}
    for name in ("supplemental_shared_active", "fixed_8"):
        interval = report["paired_group_bootstrap"][name]
        interval[delta_name] = interval.pop("delta_8b_minus_4b")
    return report


def verify_base_hardlinks(base=BASE):
    files = read(BASE_RECEIPT)["base_member_sha256"]
    large, small = 0, 0
    for archive_name, expected in files.items():
        name = Path(archive_name).name
        path = base / name
        if not path.samefile(BASE_REFERENCE / name):
            raise ValueError(f"Local model is no longer the hardlink to the verified staged base: {name}")
        if path.suffix == ".safetensors":
            large += 1
        else:
            if digest(path) != expected:
                raise ValueError(f"Current small model file differs from the verified stage: {name}")
            small += 1
    return {"prior_archive_file_receipt_sha256": digest(BASE_RECEIPT), "matching_file_identities": large + small,
            "hardlinked_weight_shards": large, "fresh_small_file_hashes": small, "fresh_whole_weight_hashes": False}


def cap_failure_crosscheck(ids, raw_outputs, predictions, gold, severe):
    low, high = CAPS
    capped = {sid for sid in ids if any(raw_outputs[cap][sid]["token_cap_hit"] for cap in CAPS)}
    high_failed = {sid for sid in ids if raw_outputs[high][sid]["vlm_schema_errors"]}
    result = {"any_token_cap_hit_photos": len(capped), "high_strict_schema_failed_photos": len(high_failed),
              "cap_hit_and_high_strict_schema_failed_photos": len(capped & high_failed),
              "high_cap_hit_records_formatted_as_no_hazard": sum(not predictions[high][sid] for sid in capped)}
    before, after = (predictions[cap] for cap in CAPS)
    for label, members in (("token_cap_hit", capped), ("no_token_cap_hit", set(ids) - capped)):
        correct_before = {sid: gold[sid] & before[sid] for sid in members}
        correct_after = {sid: gold[sid] & after[sid] for sid in members}
        normals = {sid for sid in members if not gold[sid]}
        severe_rows = {sid for sid in members if severe[sid]}
        result[label] = {
            "photos": len(members), "predicted_code_set_changed_photos": sum(before[sid] != after[sid] for sid in members),
            "true_code_instances_recovered": sum(len(correct_after[sid] - correct_before[sid]) for sid in members),
            "true_code_instances_lost": sum(len(correct_before[sid] - correct_after[sid]) for sid in members),
            "normal_photos": len(normals),
            "normal_false_alarms_added": sum(not before[sid] and bool(after[sid]) for sid in normals),
            "normal_false_alarms_resolved": sum(bool(before[sid]) and not after[sid] for sid in normals),
            "severity4_photos": len(severe_rows),
            "severity4_complete_misses_added": sum(bool(severe[sid] & before[sid]) and not (severe[sid] & after[sid]) for sid in severe_rows),
            "severity4_complete_misses_resolved": sum(not (severe[sid] & before[sid]) and bool(severe[sid] & after[sid]) for sid in severe_rows),
        }
    return result


def rendered_prefix_sha256(processor, system_prompt):
    from .vlm_11code import INSPECT_USER_TEXT

    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]},
    ]
    prefix = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return sha256(prefix.encode()).hexdigest()


def run_v4_with_model(model, processor, *, base, adapter, feature_controls):
    """Continue the restored B2/B1 loader only after exact first-two v3 replay parity."""
    import msvcrt
    import torch

    from . import stage2_representation_head as b2
    from .fold2_8b_baseline import sha_json
    from .stage2_rare_base_probe import resident_layers
    from .vlm_11code import infer_batched_vlm, seed_all

    matched, _, rows = fixed_cohort()
    expected = identity(matched, rows)
    v3_frozen = WORK / "generation-controls.json"
    old = read(v3_frozen)
    _, archived, archived_raw = load_arm(WORK, CAPS[0], rows, old, digest(v3_frozen))
    if old != {**expected, "runtime": old["runtime"]}:
        raise ValueError("Archived v3 controls differ from the matched 55-photo baseline")
    base_check = verify_base_hardlinks(base)
    runtime_controls = ("precision", "base_revision", "base_config_sha256", "adapter_sha256", "adapter_config_sha256",
                        "prompt_sha256", "runtime")
    processor_files = ("tokenizer.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json")
    old_code = matched["recipe"]["training"]["source"]["vlm_11code.py"]
    if (any(feature_controls[key] != old[key] for key in runtime_controls)
            or not all((adapter / name).samefile(ADAPTER / name) for name in ("adapter_config.json", "adapter_model.safetensors"))
            or feature_controls["feature_source_sha256"] != digest(Path(b2.__file__))
            or feature_controls["engine_sha256"] != old_code
            or digest(Path(__file__).with_name("vlm_11code.py")) != old_code
            or feature_controls["tokenizer_processor_sha256"] != {name: digest(BASE / name) for name in processor_files}
            or feature_controls["preprocessing"] != archived["processor_size"]
            or dict(processor.image_processor.size) != archived["processor_size"]
            or processor.tokenizer.padding_side != "left" or torch.get_num_threads() != 1
            or b2.SEED != SEED or digest(V4_PROMPT) != V4_PROMPT_SHA):
        raise ValueError("Resident loader, runtime, same adapter, processor or pinned v4 prompt differs")
    v3_prompt = PROMPT.read_text(encoding="utf-8")
    v4_prompt = V4_PROMPT.read_text(encoding="utf-8")
    v3_prefix = rendered_prefix_sha256(processor, v3_prompt)
    v4_prefix = rendered_prefix_sha256(processor, v4_prompt)
    layers = resident_layers(model, base)
    if (v3_prefix != archived["prefix_sha256"] or v3_prefix != feature_controls["prefix_sha256"] or v4_prefix == v3_prefix
            or model.active_adapter != "default" or model.training or model.generation_config.do_sample is not False
            or any(getattr(model.generation_config, name) is not None for name in ("temperature", "top_p", "top_k"))
            or len(layers) != 260 or any(layer.disable_adapters or layer.active_adapters != ["default"] for layer in layers)
            or any(layer.lora_A["default"].weight.dtype != torch.float32
                   or layer.lora_B["default"].weight.dtype != torch.float32 for layer in layers)):
        raise ValueError("Expected original active FP32 LoRA, archived v3 prefix, and greedy generation after B1")

    V4_WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "process.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        raw_v4 = V4_WORK / f"raw-{CAPS[0]}.json"
        v4_receipt = V4_WORK / f"arm-{CAPS[0]}.json"
        frozen_v4 = V4_WORK / "generation-controls.json"
        controls_v4 = {**old, "prompt_sha256": V4_PROMPT_SHA}
        if frozen_v4.exists() and read(frozen_v4) != controls_v4:
            raise ValueError("Existing v4 cache has different controls")
        for row in rows:
            if digest(IMAGE_ROOT / row["image"]) != row["sha256"]:
                raise ValueError("A fixed v4 photo differs from the archived v3 bytes")
        write_json(frozen_v4, controls_v4)
        torch.cuda.empty_cache()
        seed_all(SEED)
        print("STAGE1_V4_V3_REPLAY_GATE photos=2", flush=True)
        replay, replay_stats = infer_batched_vlm(model, processor, rows[:2], IMAGE_ROOT, v3_prompt,
                                                batch_size=BATCH_SIZE, max_new_tokens=TOKEN_LIMIT)
        original = [archived_raw[row["sample_id"]]["raw_text"] for row in rows[:2]]
        observed = [row["raw_text"] for row in replay]
        gate = {"passed": original == observed, "photos": 2, "order_sha256": order_digest(rows[:2]),
                "archived_raw_pair_sha256": sha_json(original), "replayed_raw_pair_sha256": sha_json(observed),
                "archived_v3_full_raw_sha256": digest(WORK / f"raw-{CAPS[0]}.json"),
                "v3_prefix_sha256": v3_prefix, "v4_prefix_sha256": v4_prefix,
                "resident_active_fp32_lora_layers": len(layers), "base_file_check": base_check,
                "b2_feature_controls_sha256": sha_json(feature_controls), "same_checkpoint_and_loader": True,
                "cold_load_seed": SEED, "same_runtime_and_processor": True, "stats": replay_stats}
        write_json(V4_WORK / "v3-replay-gate.json", gate)
        if not gate["passed"]:
            raise ValueError("Resident v3 replay differs from the original fixed first two rows. V4 is blocked")
        if v4_receipt.is_file():
            load_arm(V4_WORK, CAPS[0], rows, controls_v4, digest(frozen_v4))
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed_all(SEED)
        print("STAGE1_V4_GENERATE photos=55", flush=True)
        _, stats = infer_batched_vlm(model, processor, rows, IMAGE_ROOT, v4_prompt,
                                    batch_size=BATCH_SIZE, max_new_tokens=TOKEN_LIMIT, cache_path=raw_v4,
                                    cache_key=f"stage1-local-nf4:{BASE_REVISION}:{old['adapter_sha256']}:px{CAPS[0]}")
        torch.cuda.synchronize()
        write_json(v4_receipt, {"controls_sha256": digest(frozen_v4), "max_pixels": CAPS[0], "raw_sha256": digest(raw_v4),
                               "stats": stats, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def score(work, result, *, prompt_v4=False):
    arms = (("v3", work, CAPS[0]), ("v4", V4_WORK, CAPS[0])) if prompt_v4 else tuple(
        (str(cap), work, cap) for cap in CAPS)
    if not all((folder / f"arm-{cap}.json").is_file() for _, folder, cap in arms):
        raise ValueError("Both frozen 55-photo arms must finish before scoring")

    from .fold2_8b_baseline import sha_json
    from .fold2_8b_paired_score import score_panel
    from .law_reranker_11code import read_selected_cases
    from .metrics import decode_target, read_index
    from .v3_turnaround import ActionSupportedLawRetriever, format_v3_record

    matched, train, rows = fixed_cohort()
    expected = identity(matched, rows)
    frozen = work / "generation-controls.json"
    controls = read(frozen)
    if controls != {**expected, "runtime": controls["runtime"]}:
        raise ValueError("Local generation controls differ from the matched source")
    if prompt_v4:
        gate = read(V4_WORK / "v3-replay-gate.json")
        if (gate["passed"] is not True or gate["photos"] != 2
                or gate["order_sha256"] != order_digest(rows[:2])
                or gate["archived_raw_pair_sha256"] != gate["replayed_raw_pair_sha256"]
                or gate["archived_v3_full_raw_sha256"] != digest(work / f"raw-{CAPS[0]}.json")
                or digest(V4_PROMPT) != V4_PROMPT_SHA):
            raise ValueError("V4 requires the measured fixed v3 replay gate")
    base_check = verify_base_hardlinks()
    settings = matched["recipe"]["training"]
    if digest(LABELS) != settings["label_source_sha256"] or digest(LAWS) != settings["law_corpus_sha256"]:
        raise ValueError("Label or law bytes differ from the matched source")
    train_ids = {row["sample_id"] for row in train}
    training = {row["sample_id"]: row for row in read_selected_cases(LABELS, train_ids)}
    if training.keys() != train_ids:
        raise ValueError("Training-only case bank is incomplete")
    laws = [json.loads(line) for line in LAWS.read_bytes().splitlines() if line.strip()]
    train_cases = [training[row["sample_id"]] for row in train]
    ranker = ActionSupportedLawRetriever(laws, train_cases)
    receipts, raw_identities, raw_outputs, predictions = {}, {}, {}, {}
    for label, folder, cap in arms:
        setting_file = folder / "generation-controls.json"
        arm_controls = {**controls, "prompt_sha256": V4_PROMPT_SHA} if label == "v4" else controls
        if read(setting_file) != arm_controls:
            raise ValueError("An arm has different controls beyond the requested factor")
        receipt, raw_identity, outputs = load_arm(folder, cap, rows, arm_controls, digest(setting_file))
        path = folder / f"predictions-{cap}.jsonl"
        staging = path.with_suffix(".part")
        staging.write_bytes("".join(json.dumps(format_v3_record(sid, item["parsed"], ranker), ensure_ascii=False) + "\n"
                                    for sid, item in outputs.items()).encode("utf-8"))
        staging.replace(path)
        predictions[label] = read_index(path, lambda value: decode_target(value, local_ids=True), limit_size=True)
        receipt["predictions_sha256"] = digest(path)
        receipts[label] = receipt
        raw_identities[label] = raw_identity
        raw_outputs[label] = outputs
    first, second = (label for label, _, _ in arms)
    expected_changes = {"prefix_sha256"} if prompt_v4 else {"processor_size", "model_key"}
    id_a, id_b = (raw_identities[label] for label in (first, second))
    if {key for key in id_a.keys() | id_b.keys() if id_a.get(key) != id_b.get(key)} != expected_changes:
        raise ValueError("Raw arms differ in prompt, photo identity, order or batch settings")

    ids = [row["sample_id"] for row in rows]
    truth = {row["sample_id"]: row["target"]["hazards"] for row in read_selected_cases(LABELS, set(ids))}
    gold = {row["sample_id"]: set(row["codes"]) for row in rows}
    if truth.keys() != gold.keys() or any(set(truth[sid]) != codes for sid, codes in gold.items()):
        raise ValueError("Official labels differ from the frozen cohort")
    severe = {sid: {code for code, item in labels.items() if item["risk_level"]["severity"]["score"] == 4}
              for sid, labels in truth.items()}
    groups = {row["sample_id"]: row["group_id"] for row in rows}
    paired = score_panel(ids, groups, gold, predictions[first], predictions[second], severe)
    if prompt_v4:
        if (id_a["prefix_sha256"], id_b["prefix_sha256"]) != (gate["v3_prefix_sha256"], gate["v4_prefix_sha256"]):
            raise ValueError("V4 prefixes differ from the tested replay and replacement")
        measured_v3 = read(RESULT)["measurement"][str(CAPS[0])]
        if any(receipts[first][key] != measured_v3[key] for key in ("raw_sha256", "predictions_sha256")):
            raise ValueError("The comparison requires the original measured v3 output and formatter")
        report = relabel_score(paired, ("v3", "v4"), "delta_v4_minus_v3")
        slip_absent = {sid for sid in ids if "slip_trip" not in gold[sid]}
        ancillary = [{sid for sid in slip_absent if "slip_trip" in predictions[label][sid] and len(predictions[label][sid]) > 1}
                     for label in (first, second)]
        write_json(result, {"scope": "Local RTX 3060 NF4 prompt-only comparison with fixed 55-photo sensitivity, including 16 prior ROI reviews.",
                            "max_pixels": CAPS[0], "shared_controls": {k: v for k, v in controls.items() if k != "prompt_sha256"},
                            "prompt_sha256": {first: controls["prompt_sha256"], second: V4_PROMPT_SHA},
                            "changed_cache_field": "prefix_sha256", "base_file_check": base_check, "replay_gate": gate,
                            "shared_law_case_bank_sha256": sha_json(train_cases), "law_corpus_sha256": digest(LAWS),
                            "ancillary_slip_false_positive": {"photos_without_gold_slip": len(slip_absent),
                                "v3": len(ancillary[0]), "v4": len(ancillary[1]),
                                "added": len(ancillary[1] - ancillary[0]), "resolved": len(ancillary[0] - ancillary[1])},
                            "measurement": receipts, "quality": report})
        return
    report = relabel_score(paired)
    changed = read_token_change()
    diagnostic = {}
    for stratum, increased in (("token_increase", True), ("unchanged_tokens", False)):
        members = [row["sample_id"] for row in rows if changed[row["sha256"]] == increased]
        diagnostic[stratum] = {"photos": len(members), "exact_raw_text_matches": sum(
            raw_outputs[first][sid]["raw_text"] == raw_outputs[second][sid]["raw_text"] for sid in members),
            "exact_predicted_code_set_matches": sum(predictions[first][sid] == predictions[second][sid] for sid in members)}
    raw_by_cap = {cap: raw_outputs[str(cap)] for cap in CAPS}
    predictions_by_cap = {cap: predictions[str(cap)] for cap in CAPS}
    write_json(result, {"scope": "Local RTX 3060 NF4 paired pixel-cap contrast. The nested Fold 2 sensitivity still includes 16 prior ROI review photos.",
                        "caps": CAPS, "controls": controls, "base_file_check": base_check, "identical_nonpixel_cache_identity": True,
                        "active_prefix_sha256": id_a["prefix_sha256"], "token_stratum_agreement": diagnostic,
                        "cap_failure_crosscheck": cap_failure_crosscheck(ids, raw_by_cap, predictions_by_cap, gold, severe),
                        "measurement": receipts, "quality": report})


def read_token_change():
    from .metrics import read_index
    from .stage1_resolution_opportunity import SIDECAR, TOKEN_RECEIPT, require_digest, token_counts

    require_digest(SIDECAR, read(TOKEN_RECEIPT)["local_per_photo_sidecar"]["sha256"])
    return read_index(SIDECAR, token_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--score", action="store_true")
    action.add_argument("--score-v4", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if not args.work.resolve().is_relative_to(Path("data").resolve()):
        parser.error("Raw generations must remain under data/")
    if args.score or args.score_v4:
        result = args.result or (V4_RESULT if args.score_v4 else RESULT)
        if not result.resolve().is_relative_to(Path("results").resolve()):
            parser.error("The scored aggregate must be under results/")
        score(args.work, result, prompt_v4=args.score_v4)
    else:
        generate(args.work)


if __name__ == "__main__":
    main()
