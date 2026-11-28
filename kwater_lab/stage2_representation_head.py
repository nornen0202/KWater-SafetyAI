"""One Fold 2 diagnostic with a frozen NF4 8B prefill and small visual-token head."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import io
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .clean_specialist import asymmetric_focal_bce
from .fold0_group_sensitivity import COMMON, arm_metrics
from .fold2_8b_baseline import BASE_CONFIG_SHA, BASE_REVISION, RUN as BASELINE, order_digest, sha_json, write_json
from .fold2_v3_epoch1 import LABELS, MASK, PROMPT, split
from .fold4_same_verifier import digest, read
from .law_reranker_11code import read_selected_cases
from .metrics import CODES
from .sub78_clean_cal_design import WORK as CLEAN_CAL
from .train_head import save_atomic

WORK = Path("data/stage2-representation-head-fold2")
BASE = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
ADAPTER = BASELINE / "fit/epoch-1"
MATCHED = Path("results/fold2-matched-4b-8b-epoch1-20261128.json")
BLOCKS = Path("data/stage2-oof-design-20261128/fold-manifest-local.json")
BLOCK_PROOF = Path("results/stage2-oof-design-20261128.json")
LOCAL_NF4 = Path("data/stage1-local-nf4-resolution-55")
PREP_RESULT = Path("results/stage2-representation-head-fold2-preflight-20261129.json")
SCORE_RESULT = Path("results/stage2-representation-head-fold2-20261129.json")
IMAGE_ROOT = Path("data/train")
ROLES = ("fit", "calibration", "outer")
MAX_PIXELS, HIDDEN, WIDTH, EPOCHS, BATCH, SEED = 393216, 4096, 128, 12, 8, 42
PRECISION = "nf4_double_quant_bf16_compute"
COMMON_INDEX = tuple(CODES.index(code) for code in COMMON)
FPR_CAP = 0.14
CUTS = (.15, .30, .45, .50, .60, .75, .90, 1.01)
WEIGHT_CAP = 5_000_000


class VisualTokenHead(nn.Module):
    def __init__(self, hidden=HIDDEN, width=WIDTH):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden, width), nn.LayerNorm(width))
        self.queries = nn.Parameter(torch.randn(len(CODES) + 1, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.output = nn.Linear(width, 1)
        self.bias = nn.Parameter(torch.zeros(len(CODES) + 1))

    def forward(self, tokens, valid):
        features = self.project(tokens)
        queries = self.queries.expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(queries, features, features, key_padding_mask=~valid, need_weights=False)
        return self.output(attended + queries).squeeze(-1) + self.bias


def selected_gold(rows):
    ids = {row["sample_id"] for row in rows}
    records = {row["sample_id"]: row["target"]["hazards"] for row in read_selected_cases(LABELS, ids)}
    if ids != records.keys():
        raise ValueError("Official labels do not cover the selected rows")
    return records


def freeze(work, result):
    train, outer = split()
    archived = CLEAN_CAL / "fold-2"
    contract = read(archived / "contract.json")
    roster = {role: archived / f"{role}-manifest.json" for role in ROLES}
    block_sha = digest(BLOCKS)
    if (block_sha != read(BLOCK_PROOF)["restricted_manifest_sha256"]
            or any(digest(path) != contract["inputs"][role] for role, path in roster.items())
            or any(digest(Path(path)) != expected for path, expected in contract["confirmed_link_ledger_sha256"].items())
            or digest(MASK) != read(MATCHED)["recipe"]["training"]["mask_sha256"]):
        raise ValueError("A frozen Fold 2 roster, component or review source changed")
    roster = {role: read(path) for role, path in roster.items()}
    if (roster["outer"] != outer or {row["sample_id"]: row for role in ROLES[:2] for row in roster[role]}
            != {row["sample_id"]: row for row in train}):
        raise ValueError("Clean-Cal split differs from the canonical Fold 2 scope")
    known = {row["sample_id"]: row for row in read(BLOCKS)}
    for row in train + outer:
        if any(known[row["sample_id"]][key] != row[key] for key in ("image", "sha256")):
            raise ValueError("Conservative block roster points to different photo bytes")
    outer_blocks = {known[row["sample_id"]]["group_id"] for row in outer}
    rows = {role: [{key: row[key] for key in ("sample_id", "image", "sha256", "group_id")}
                   | {"component": known[row["sample_id"]]["group_id"]}
                   for row in roster[role] if role == "outer" or known[row["sample_id"]]["group_id"] not in outer_blocks]
            for role in ROLES}
    components = {role: {row["component"] for row in rows[role]} for role in ROLES}
    sizes = tuple(len(rows[role]) for role in ROLES)
    if (sizes != (215, 42, 197) or components["fit"] & components["calibration"]
            or (components["fit"] | components["calibration"]) & outer_blocks):
        raise ValueError("The predeclared conservative Fold 2 pilot split changed")
    keep = set(read(MASK)["prospective_mask_retained_ids"])
    exposed = {known[row["sample_id"]]["group_id"] for row in train}
    if (len(keep) != 55 or not keep <= {row["sample_id"] for row in rows["outer"]}
            or any(row["sample_id"] in keep and row["component"] in exposed for row in rows["outer"])):
        raise ValueError("Sensitivity cohort overlaps the conservative historic training blocks")
    support = {}
    for role in ROLES[:2]:
        labels = selected_gold(rows[role])
        support[role] = {"photos": len(labels), "components": len(components[role]),
                         "normal": sum(not target for target in labels.values()),
                         "severity4": sum(any(item["risk_level"]["severity"]["score"] == 4 for item in r.values())
                                          for r in labels.values()),
                         "codes": {code: sum(code in target for target in labels.values()) for code in CODES}}
    rules = {"epochs": EPOCHS, "batch": BATCH, "seed": SEED, "width": WIDTH, "hidden": HIDDEN,
             "min_pixels": 131072, "max_pixels": MAX_PIXELS, "storage_dtype": "bfloat16", "optimizer": "AdamW",
             "learning_rate": .001, "weight_decay": .01, "loss": "repo asymmetric focal BCE, smooth=0",
             "code_and_scene_grid": CUTS, "fpr_max": FPR_CAP, "unsupported_common_code_cut": .50,
             "rare_output_codes_enabled": False}
    rules["code_and_scene_grid"] = list(CUTS)
    witness = {"roles": rows, "sensitivity_ids": [row["sample_id"] for row in outer if row["sample_id"] in keep],
               "mask_sha256": digest(MASK), "blocks_sha256": block_sha, "labels_sha256": digest(LABELS),
               "mixed_score_sha256": digest(MATCHED), "rules": rules}
    work.mkdir(parents=True, exist_ok=True)
    path = work / "frozen-local.json"
    freeze_json(path, witness)
    write_json(result, {
        "scope": "One Fold 2 diagnostic design. Frozen VLM already trained on the full 769-photo training side.",
        "role_support": support, "outer_photos": sizes[2], "sensitivity_photos": len(keep),
        "old_training_rows_dropped_by_outer_components": len(train) - sizes[0] - sizes[1],
        "outer_in_historical_training_components": sum(row["component"] in exposed for row in rows["outer"]),
        "head_fit_calibration_outer_component_overlaps": 0, "rules": rules,
        "calibration_fp_limit": math.floor(FPR_CAP * support["calibration"]["normal"]),
        "feature_cache_max_bf16_tensor_bytes": sum(sizes) * 384 * HIDDEN * 2,
        "head_tensor_bytes": sum(t.numel() * t.element_size() for t in VisualTokenHead().parameters()),
        "preflight_frozen_local_sha256": digest(path), "blocks_sha256": block_sha,
        "fitted_models": 0, "extracted_photos": 0, "generated_predictions": 0,
    })


def freeze_json(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Frozen {path.name} differs from the current contract")
    else:
        write_json(path, payload)


def visual_prefill(body, inputs, image_token_id, merge, feature_width=HIDDEN):
    ids, attention, grid = (inputs[key] for key in ("input_ids", "attention_mask", "image_grid_thw"))
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != attention.shape or tuple(grid.shape) != (1, 3)
            or set(inputs) != {"input_ids", "attention_mask", "image_grid_thw", "pixel_values"}):
        raise ValueError("Expected exactly one image with an inspection-only prompt")
    positions = ids.eq(image_token_id)
    mask = positions & attention.bool()
    thw = grid[0].tolist()
    expected, remainder = divmod(math.prod(thw), merge * merge)
    if any(value < 1 for value in thw) or remainder or not expected or int(mask.sum()) != expected or not torch.equal(positions, mask):
        raise ValueError("Image grid and unpadded visual-token mask disagree")
    hidden = body(**inputs, use_cache=False).last_hidden_state
    if hidden.shape != (*ids.shape, feature_width):
        raise ValueError("Unexpected last-layer hidden-state dimensions")
    features = hidden[mask].to(device="cpu", dtype=torch.bfloat16).contiguous()
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite visual tokens")
    return {"features": features, "image_grid_thw": thw, "visual_tokens": expected,
            "input_tokens": ids.shape[1], "input_ids_sha256": sha_json(ids.tolist()),
            "visual_positions_sha256": sha_json(mask[0].nonzero().flatten().tolist())}


def feature_path(work, row):
    return work / "features" / (sha256(row["sample_id"].encode()).hexdigest() + ".pt")


def decode_verified_photo(row):
    from PIL import Image, ImageOps

    payload = (IMAGE_ROOT / row["image"]).read_bytes()
    if sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("A registered photo differs from the frozen image digest")
    with Image.open(io.BytesIO(payload)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def prefetched_photos(rows):
    if not rows:
        return
    with ThreadPoolExecutor(max_workers=1) as reader:
        next_photo = reader.submit(decode_verified_photo, rows[0])
        for index, row in enumerate(rows):
            image = next_photo.result()
            if index+1 < len(rows):
                next_photo = reader.submit(decode_verified_photo, rows[index+1])
            yield row, image


def read_feature(work, row, context):
    path = feature_path(work, row)
    artifact = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    value = artifact["features"]
    if (artifact["context_sha256"] != context or artifact["photo_sha256"] != row["sha256"]
            or artifact["sample_id"] != row["sample_id"] or value.dtype != torch.bfloat16
            or value.shape != (artifact["visual_tokens"], HIDDEN)
            or math.prod(artifact["image_grid_thw"]) != 4 * artifact["visual_tokens"]):
        raise ValueError("A visual-token cache differs from the photo or NF4 context")
    return artifact


@torch.inference_mode()
def extract(work, base, part, limit, *, after_extract=None):
    if after_extract is not None and (part != "all" or limit is not None):
        raise ValueError("A resident-model continuation requires complete feature coverage")
    import msvcrt
    from .vlm_11code import INSPECT_USER_TEXT, load_qwen_model, load_qwen_processor, seed_all

    mapped = read(work / "frozen-local.json")
    roles = ROLES if part == "all" else (ROLES[:2] if part == "train" else ROLES[2:])
    rows = [row for role in roles for row in mapped["roles"][role]]
    if (torch.cuda.device_count() != 1 or "RTX 3060" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Expected one RTX 3060 with BF16 support for the NF4 feature run")
    with (work / "extract.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        matched = read(MATCHED)
        source = read(base / "source.json")
        adapter_sha = digest(ADAPTER / "adapter_model.safetensors")
        if (digest(MATCHED) != mapped["mixed_score_sha256"] or source["revision"] != BASE_REVISION
                or digest(base / "config.json") != BASE_CONFIG_SHA or adapter_sha != matched["generated"]["adapter_sha256"]
                or digest(PROMPT) != matched["recipe"]["training"]["prompt_sha256"]
                or any((base / item["rfilename"]).stat().st_size != item["size"]
                       for item in source["files"] if item["rfilename"].endswith(".safetensors"))):
            raise ValueError("The feature source differs from pinned, measured Fold 2 8B scope")
        processor = load_qwen_processor(base, MAX_PIXELS)
        processor.tokenizer.padding_side = "left"
        prefix = processor.apply_chat_template([
            {"role": "system", "content": PROMPT.read_text(encoding="utf-8")},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]},
        ], tokenize=False, add_generation_prompt=True)
        base_config = read(base / "config.json")
        token_id = base_config["image_token_id"]
        merge = base_config["vision_config"]["spatial_merge_size"]
        controls = {
            "precision": PRECISION, "base_revision": BASE_REVISION, "base_config_sha256": BASE_CONFIG_SHA,
            "adapter_sha256": adapter_sha, "adapter_config_sha256": digest(ADAPTER / "adapter_config.json"),
            "prompt_sha256": digest(PROMPT), "prefix_sha256": sha256(prefix.encode()).hexdigest(),
            "preprocessing": dict(processor.image_processor.size), "frozen_local_sha256": digest(work / "frozen-local.json"),
            "image_token_id": token_id, "spatial_merge": merge, "storage_dtype": "bfloat16",
            "feature_source_sha256": digest(Path(__file__)), "engine_sha256": digest(Path(__file__).with_name("vlm_11code.py")),
            "tokenizer_processor_sha256": {file: digest(base / file) for file in (
                "tokenizer.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json")},
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                        **{pkg: version(pkg) for pkg in ("transformers", "peft", "accelerate", "bitsandbytes", "Pillow")}},
        }
        freeze_json(work / "feature-context.json", controls)
        context = sha_json(controls)
        pending = []
        for row in rows:
            if feature_path(work, row).is_file():
                read_feature(work, row, context)
            else:
                pending.append(row)
        if (not pending and after_extract is None) or limit == 0:
            print(f"FEATURE_CACHE_REUSED {len(rows)-len(pending)}/{len(rows)}", flush=True)
            return
        torch.cuda.reset_peak_memory_stats()
        seed_all(SEED)
        start = time.perf_counter()
        model = load_qwen_model(base, use_4bit=True, adapter_dir=ADAPTER).eval()
        if (not getattr(model, "is_loaded_in_4bit", False)
                or any(p.device != torch.device("cuda:0") for p in model.parameters())):
            raise RuntimeError("The NF4 backbone must reside on CUDA without offload")
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        print(f"FEATURE_MODEL_LOADED {load_seconds:.1f}s", flush=True)
        body = model.get_base_model().model
        selected = pending[:limit]
        forward_seconds = 0.
        tokens = 0
        for count, (row, rgb) in enumerate(prefetched_photos(selected), start=1):
            inputs = processor(text=[prefix], images=[rgb], padding=True, return_tensors="pt").to("cuda")
            t0 = time.perf_counter()
            artifact = visual_prefill(body, inputs, token_id, merge)
            forward_seconds += time.perf_counter() - t0
            tokens += artifact["visual_tokens"]
            artifact.update(context_sha256=context, photo_sha256=row["sha256"], sample_id=row["sample_id"])
            save_atomic(feature_path(work, row), lambda stream: torch.save(artifact, stream))
            if count in (1, 4) or count % 16 == 0 or count == len(selected):
                print(f"FEATURES {count}/{len(selected)} elapsed={time.perf_counter()-start:.1f}s "
                      f"forward_and_copy={forward_seconds:.2f}s "
                      f"peak_allocated_gib={torch.cuda.max_memory_allocated()/2**30:.3f} "
                      f"peak_reserved_gib={torch.cuda.max_memory_reserved()/2**30:.3f}", flush=True)
            del inputs, artifact
        receipt_dir = work / "extract-receipts"
        receipt_dir.mkdir(exist_ok=True)
        write_json(receipt_dir / f"{time.time_ns()}.json", {
            "context_sha256": context, "part": part, "reused_photos": len(rows)-len(pending), "new_photos": len(selected),
            "new_visual_tokens": tokens, "model_load_seconds": round(load_seconds, 3),
            "forward_and_copy_seconds": round(forward_seconds, 3), "extract_wall_seconds": round(time.perf_counter()-start, 3),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        if len(selected) == len(pending):
            print(f"FEATURE_EXTRACTION_COMPLETE {len(rows)}", flush=True)
        if after_extract is not None:
            torch.cuda.empty_cache()
            after_extract(model, processor, controls)


def padded(samples):
    width = samples[0].shape[1]
    tokens = torch.zeros((len(samples), max(value.shape[0] for value in samples), width), dtype=torch.float32)
    valid = torch.zeros(tokens.shape[:2], dtype=torch.bool)
    for i, value in enumerate(samples):
        if not value.shape[0] or value.shape[1] != width or not torch.isfinite(value).all():
            raise ValueError("Empty or non-finite visual-token batch")
        tokens[i, :len(value)] = value
        valid[i, :len(value)] = True
    return tokens, valid


def rows_and_context(work, role):
    plan_path = work / "frozen-local.json"
    plan = read(plan_path)
    context = read(work / "feature-context.json")
    if (context["frozen_local_sha256"] != digest(plan_path) or context["precision"] != PRECISION
            or context["feature_source_sha256"] != digest(Path(__file__))
            or plan["labels_sha256"] != digest(LABELS)):
        raise ValueError("Feature or labels belong to another frozen pilot")
    return plan["roles"][role], sha_json(context)


def load_batch(work, rows, context):
    return padded([read_feature(work, row, context)["features"] for row in rows])


def fit(work):
    rows, context = rows_and_context(work, "fit")
    path = work / "head.pt"
    if path.exists():
        load_head(work, context)
        return
    torch.manual_seed(SEED)
    model = VisualTokenHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    labels = selected_gold(rows)
    targets = torch.tensor([[code in labels[row["sample_id"]] for code in CODES] + [bool(labels[row["sample_id"]])]
                            for row in rows], dtype=torch.float32)
    support = targets.sum(0)
    weights = ((len(rows) - support) / support.clamp_min(1)).sqrt().clamp(1, 5)
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for start in range(0, len(rows), BATCH):
            batch = order[start:start+BATCH]
            optimizer.zero_grad(set_to_none=True)
            features = load_batch(work, [rows[i] for i in batch], context)
            loss = asymmetric_focal_bce(model(*features), targets[batch], weights, smooth=0.)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite asymmetric classification loss")
            loss.backward()
            optimizer.step()
        print(f"HEAD_EPOCH {epoch+1}/{EPOCHS}", flush=True)
    save_atomic(path, lambda stream: torch.save({"context_sha256": context, "state": model.state_dict()}, stream))
    if path.stat().st_size >= WEIGHT_CAP:
        raise ValueError("Head serialization must be smaller than 5,000,000 bytes")


def load_head(work, context):
    path = work / "head.pt"
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if saved["context_sha256"] != context or path.stat().st_size >= WEIGHT_CAP:
        raise ValueError("Head identity or size differs from the frozen NF4 pilot")
    model = VisualTokenHead()
    model.load_state_dict(saved["state"])
    return model.eval()


@torch.inference_mode()
def logits_for(model, work, rows, context):
    return torch.cat([model(*load_batch(work, rows[i:i+BATCH], context)) for i in range(0, len(rows), BATCH)])


def decisions(probabilities, cuts, scene_cut):
    return (probabilities[:, COMMON_INDEX] >= cuts) & (probabilities[:, -1:] >= scene_cut)


def choose_cuts(probabilities, truth, normal):
    if (truth.shape != (len(normal), len(COMMON)) or probabilities.shape != (len(normal), len(CODES)+1)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
            or np.any(truth[normal])):
        raise ValueError("Unaligned, nonfinite or invalid train-side calibration inputs")
    support = truth.sum(0)
    if not normal.any():
        raise ValueError("Normal calibration support is required")
    max_false = math.floor(FPR_CAP * normal.sum())

    def rank(cuts, scene):
        pred = decisions(probabilities, cuts, scene)
        false = int(pred[normal].any(1).sum())
        denominator = support + pred.sum(0)
        f1 = np.divide(2*(truth & pred).sum(0), denominator, out=np.zeros(len(COMMON)), where=denominator != 0).mean()
        return false <= max_false, float(f1) if false <= max_false else -false, -false, -sum(abs(c-.5) for c in cuts), -scene

    best = None
    for scene in CUTS:
        cuts = np.full(len(COMMON), .50)
        for _ in range(3):
            for index in np.flatnonzero(support):
                trial = cuts.copy()
                best_cut = cuts[index]
                best_rank = rank(trial, scene)
                for choice in CUTS:
                    trial[index] = choice
                    choice_rank = rank(trial, scene)
                    if choice_rank > best_rank:
                        best_cut, best_rank = choice, choice_rank
                cuts[index] = best_cut
        option = (rank(cuts, scene), cuts.tolist(), scene)
        if best is None or option[0] > best[0]:
            best = option
    choice_rank, cuts, scene = best
    return {"common_codes": list(COMMON), "thresholds": cuts, "scene_threshold": scene,
            "unsupported_common_codes_fixed_at_0_50": [code for code, n in zip(COMMON, support) if not n],
            "normal_photos": int(normal.sum()), "normal_false_alarms": -choice_rank[2],
            "normal_false_alarm_maximum": max_false, "calibration_fixed8_macro_f1": choice_rank[1]}


def calibrate(work):
    rows, context = rows_and_context(work, "calibration")
    logits = logits_for(load_head(work, context), work, rows, context)
    labels = selected_gold(rows)
    truth = np.asarray([[code in labels[row["sample_id"]] for code in COMMON] for row in rows], dtype=bool)
    normal = np.asarray([not labels[row["sample_id"]] for row in rows], dtype=bool)
    cuts = choose_cuts(logits.sigmoid().numpy(), truth, normal)
    cuts.update(context_sha256=context, head_sha256=digest(work / "head.pt"),
                calibration_order_sha256=order_digest(rows), fallback_for_missing_code=.50)
    freeze_json(work / "thresholds.json", cuts)


def same_precision_55(work, nf4, rows, context, gold, predicted, severe):
    from .fold2_8b_paired_score import format_received_predictions
    from .stage1_fold2_quality import compare_panel

    path = nf4 / "generation-controls.json"
    source = read(path)
    raw_path = nf4 / f"raw-{MAX_PIXELS}.json"
    receipt = read(nf4 / f"arm-{MAX_PIXELS}.json")
    raw = read(raw_path)
    frozen = read(work / "feature-context.json")
    identity = raw["identity"]
    ids = [row["sample_id"] for row in rows]
    if (source["cohort_order_sha256"] != order_digest(rows) or receipt["controls_sha256"] != digest(path)
            or receipt["max_pixels"] != MAX_PIXELS or receipt["raw_sha256"] != digest(raw_path)
            or any(source[key] != frozen[key] for key in (
                "precision", "base_revision", "base_config_sha256", "adapter_sha256", "adapter_config_sha256",
                "prompt_sha256", "runtime"))
            or context != sha_json(frozen) or identity["prefix_sha256"] != frozen["prefix_sha256"]
            or identity["processor_size"] != frozen["preprocessing"]
            or identity["image_sha256"] != {row["image"]: row["sha256"] for row in rows}
            or identity["samples"] != [[row["sample_id"], row["image"]] for row in rows]
            or identity["model_key"] != f"stage1-local-nf4:{BASE_REVISION}:{frozen['adapter_sha256']}:px{MAX_PIXELS}"
            or identity["batch_size"] != 1 or identity["max_new_tokens"] != 620
            or list(raw["outputs"]) != ids or receipt["stats"]["photos"] != 55
            or receipt["stats"]["final_effective_batch_size"] != 1):
        raise ValueError("The paired comparator must be the same frozen NF4 393216-pixel 55-photo run")
    train, _ = split()
    baseline_path = work / "nf4-generate-55-formatted-local.jsonl"
    control = format_received_predictions(train, ids, raw["outputs"], baseline_path)
    return {"arms": {"control": "same NF4 autoregressive generation", "candidate": "NF4 visual-token head"},
            "generation_receipt_sha256": digest(nf4 / f"arm-{MAX_PIXELS}.json"),
            "panel": compare_panel(ids, {row["sample_id"]: row["group_id"] for row in rows}, gold, control, predicted, severe)}


def score(work, result, nf4):
    rows, context = rows_and_context(work, "outer")
    plan = read(work / "frozen-local.json")
    thresholds = read(work / "thresholds.json")
    if (thresholds["context_sha256"] != context or thresholds["head_sha256"] != digest(work / "head.pt")
            or thresholds["calibration_order_sha256"] != order_digest(plan["roles"]["calibration"])
            or thresholds["calibration_order_sha256"] == order_digest(rows)
            or thresholds["normal_false_alarms"] > thresholds["normal_false_alarm_maximum"]):
        raise ValueError("Outer scoring requires the fixed train-side calibration policy")
    logits = logits_for(load_head(work, context), work, rows, context)
    target = selected_gold(rows)
    gold = {key: set(value) for key, value in target.items()}
    severe = {key: {code for code, h in hazards.items() if h["risk_level"]["severity"]["score"] == 4}
              for key, hazards in target.items()}
    keep = set(plan["sensitivity_ids"])
    flags = decisions(logits.sigmoid().numpy(), np.asarray(thresholds["thresholds"]), thresholds["scene_threshold"])
    prediction = {row["sample_id"]: {code for code, on in zip(COMMON, output) if on}
                  for row, output in zip(rows, flags, strict=True)}
    path = work / "outer-logits-local.pt"
    save_atomic(path, lambda stream: torch.save({"ids": list(prediction), "logits": logits}, stream))
    panels = {}
    for key, portion in (("primary_197", rows), ("sensitivity_55", [row for row in rows if row["sample_id"] in keep])):
        ids = [row["sample_id"] for row in portion]
        a, b = ({sid: item[sid] for sid in ids} for item in (gold, prediction))
        panels[key] = arm_metrics(a, b, {row["sample_id"]: row["group_id"] for row in portion}, severe)
    pair = None
    pair_status = "matching NF4 receipt unavailable"
    if (nf4 / f"arm-{MAX_PIXELS}.json").is_file():
        try:
            pair = same_precision_55(work, nf4, [row for row in rows if row["sample_id"] in keep], context,
                                     gold, prediction, severe)
            pair_status = "verified same NF4 generation"
        except (KeyError, OSError, ValueError):
            pair_status = "generation receipt or cohort did not pass same-precision verification"
    receipts = [read(path) for path in sorted((work / "extract-receipts").glob("*.json"))]
    if any(receipt["context_sha256"] != context for receipt in receipts):
        raise ValueError("A prefill receipt belongs to another feature context")
    write_json(result, {
        "scope": "Single mixed Fold 2 NF4 VLM representation pilot. Head fits 215 VLM-exposed photos and calibrates on 42.",
        "precision": PRECISION, "head_weight_bytes": (work / "head.pt").stat().st_size, "head_sha256": thresholds["head_sha256"],
        "features_origin_sha256": context, "local_logits_sha256": digest(path), "calibration": thresholds,
        "pilot_panels": panels, "same_precision_55_pair": pair, "same_precision_55_pair_status": pair_status,
        "rare_proposals_enabled": False,
        "prefill_runs": receipts,
        "five_fold_oof_predictions": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "extract", "fit", "calibrate", "score"))
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--part", choices=("train", "outer", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--nf4", type=Path, default=LOCAL_NF4)
    args = parser.parse_args()
    if any(not path.resolve().is_relative_to(Path("data").resolve()) for path in (args.work, args.nf4)):
        parser.error("Private cache, model and prediction paths must stay under data/")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    torch.set_num_threads(2)
    if args.stage == "prepare":
        freeze(args.work, args.result or PREP_RESULT)
    elif args.stage == "extract":
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        extract(args.work, args.base, args.part, args.limit)
    elif args.stage == "fit":
        fit(args.work)
    elif args.stage == "calibrate":
        calibrate(args.work)
    else:
        score(args.work, args.result or SCORE_RESULT, args.nf4)
