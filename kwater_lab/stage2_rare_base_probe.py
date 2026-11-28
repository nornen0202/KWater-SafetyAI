"""Run the image-only, disabled-adapter 8B NF4 rare-hazard probe locally."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
import io
import math
from pathlib import Path
import random
import time

from .fold2_8b_baseline import BASE_CONFIG_SHA, BASE_REVISION, RUN, sha_json, write_json
from .fold4_same_verifier import MANIFEST, TRAIN, digest, read


BASE = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
ADAPTER = RUN / "fit/epoch-1"
MATCHED = Path("results/fold2-matched-4b-8b-epoch1-20261128.json")
WORK = Path("data/stage2-rare-base-nf4-20261129")
RESULT = Path("results/stage2-rare-base-nf4-20261129.json")
MANIFEST_SHA = "3864848fe26b951ec070710499d2694078b17c76ffd2aaf38d312b99ad3b2f60"
PRECISION = "nf4_double_quant_bf16_compute"
PIXELS = {"shortest_edge": 131072, "longest_edge": 393216}
PHOTOS, PILOT_PHOTOS, PILOT_SEED = 966, 20, 20261129
CODES = ("drowning_inundation", "confined_space", "thermal_weather")
MARGIN = math.log(9)
SYSTEM = "사진에 보이는 것만 확인한다. 사람이나 사물의 용도와 사진 밖 상황을 추정하지 않는다. 주어진 대문자 한 글자로 답한다."
ANSWER_PREFIX = "답변:"
PICK = (
    ("A", (), "세 환경 모두 보이지 않는다"),
    ("B", (0,), "수변만"), ("C", (1,), "밀폐공간 입구만"), ("D", (2,), "고열·한랭 환경만"),
    ("E", (0, 1), "수변과 밀폐공간 입구"), ("F", (0, 2), "수변과 고열·한랭 환경"),
    ("G", (1, 2), "밀폐공간 입구와 고열·한랭 환경"),
    ("H", (0, 1, 2), "세 환경 모두"), ("I", (), "어느 환경인지 판단 불가"),
)
ENVIRONMENTS = (
    "수변: 사람이 닿을 수 있는 수역의 가장자리, 물이 찬 수로 또는 침수 구역",
    "밀폐공간 입구: 사람이 들어갈 수 있고 출입이 제한된 맨홀, 피트, 탱크 또는 유사 공간의 개구부",
    "고열·한랭 환경: 노출 구역과 연결된 열원, 증기, 냉동설비, 결빙 등 시각 단서가 있는 곳",
)
STAGE_ONE = "눈에 들어오는 환경을 고르세요. 이 단계에서는 위험이나 방호 여부를 판단하지 않습니다.\n" + "\n".join(
    (*ENVIRONMENTS, "단순 배경과 사진 판독이 어려운 경우도 구분합니다. 가능한 조합 중 하나만 고르세요.",
     *(f"{option}. {text}" for option, _, text in PICK))
)
GATES = (
    (
        "사진에서 사람이 접근할 수 있는 물과 작업 면이 맞닿는 수변·침수 경계가 명확히 보입니까? 먼 배경의 물이나 마른 수로만으로는 예가 아닙니다.",
        "보이는 작업·통행 위치에서 물로 빠지는 경로가 열려 있고, 관련 구간의 추락 방지 방호가 명백히 빠져 있습니까? 관련 경계가 가려졌거나 화면 밖 방호를 확인할 수 없으면 판단 불가입니다.",
    ),
    (
        "사람이 들어갈 수 있는 제한 출입 공간의 입구와 내부로 이어지는 개구부가 명확히 보입니까? 단순 뚜껑, 배수구 또는 표면 구멍만으로는 예가 아닙니다.",
        "그 입구로 진입 중인 사람이나 이용 중인 접근 경로가 보이며, 그 위치의 출입통제 또는 진입자 보호가 명백히 결여되어 있습니까? 가스 농도, 환기 상태, 점검 절차와 화면 밖 보호구는 사진만으로 판정하지 않습니다.",
    ),
    (
        "고열·한랭원과 작업자가 노출되는 구역의 관계가 뚜렷이 보입니까? 단순 실외 작업, 햇빛, 계절 또는 작업복만으로는 예가 아닙니다.",
        "사진 속 작업자가 고열·한랭원에 노출되어 있고, 해당 위치의 차열·차폐 또는 노출 부위 방호 결여가 명확히 보입니까? 온도, 노출시간, 휴식 제도는 추정하지 않습니다. 관련 방호 위치가 보이지 않으면 판단 불가입니다.",
    ),
)
GATE_CHOICES = "\nA. 예, 명백히 보임\nB. 아니요, 반대 단서가 보임\nC. 불명확하거나 확인할 수 없음"


@dataclass(frozen=True)
class Query:
    prefix: str
    last_prefix_token: int
    tokens: tuple[int, ...]


def render_queries(processor):
    specifications = {"select": (STAGE_ONE, tuple(option for option, *_ in PICK))}
    for code, (environment, protection) in zip(CODES, GATES, strict=True):
        specifications[f"{code}:environment"] = (environment + GATE_CHOICES, ("A", "B", "C"))
        specifications[f"{code}:protection"] = (protection + GATE_CHOICES, ("A", "B", "C"))
    queries = {}
    encode = processor.tokenizer.encode
    for name, (question, choices) in specifications.items():
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
        prefix_ids = encode(prefix, add_special_tokens=False)
        completions = [encode(f"{prefix} {option}", add_special_tokens=False) for option in choices]
        if (prefix.count("<|image_pad|>") != 1 or not prefix_ids
                or any(ids[:-1] != prefix_ids for ids in completions)
                or len({ids[-1] for ids in completions}) != len(choices)):
            raise ValueError("The actual multimodal prefix does not have distinct single-token choices")
        queries[name] = Query(prefix, prefix_ids[-1], tuple(ids[-1] for ids in completions))
    return queries


def image_rows(manifest):
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError("Expected the frozen 966-photo development manifest")
    rows = tuple((row["image"], row["sha256"]) for row in read(manifest))
    if len(rows) != PHOTOS:
        raise ValueError("Incorrect development geometry")
    return rows


def frozen_recipe(base, adapter, manifest, processor):
    rows = image_rows(manifest)
    source = read(base / "source.json")
    matched = read(MATCHED)
    adapter_sha = digest(adapter / "adapter_model.safetensors")
    weight_files = [item for item in source["files"] if item["rfilename"].endswith(".safetensors")]
    if (source["revision"] != BASE_REVISION or digest(base / "config.json") != BASE_CONFIG_SHA
            or adapter_sha != matched["generated"]["adapter_sha256"] or len(weight_files) != 4
            or any((base / item["rfilename"]).stat().st_size != item["size"] for item in weight_files)):
        raise ValueError("Expected the pinned 8B base and measured mixed Fold 2 adapter")
    if dict(processor.image_processor.size) != PIXELS or processor.tokenizer.padding_side != "left":
        raise ValueError("Expected the B2 processor geometry and left padding")
    queries = render_queries(processor)
    recipe = {
        "probe_source_sha256": digest(Path(__file__)), "manifest_sha256": MANIFEST_SHA,
        "base_revision": BASE_REVISION, "base_config_sha256": BASE_CONFIG_SHA,
        "model_source_sha256": digest(base / "source.json"), "precision": PRECISION,
        "resident_adapter_sha256": adapter_sha,
        "resident_adapter_config_sha256": digest(adapter / "adapter_config.json"),
        "resident_adapter_usage": "disabled for every forward", "training_photos": 0, "training_updates": 0,
        "source_inputs": "one EXIF-transposed full-frame RGB image and fixed visual questions",
        "tokenizer_processor_sha256": {name: digest(base / name) for name in (
            "tokenizer.json", "tokenizer_config.json", "chat_template.json", "preprocessor_config.json")},
        "pixels": PIXELS, "batch_size": 1, "logits_to_keep": 1, "use_cache": False, "generated_tokens": 0,
        "gate_margin": MARGIN, "gate_rule": "both margins >= ln(9); yes minus max(no, unknown)",
        "select_options": {letter: [CODES[j] for j in options] for letter, options, _ in PICK},
        "select_ties": "abstain", "gate_options": ["yes", "no", "unknown"],
        "pilot": {"photos": PILOT_PHOTOS, "seed": PILOT_SEED, "selection": "random ordered row positions"},
        "queries": {name: {"prefix_sha256": sha256(query.prefix.encode()).hexdigest(),
                           "last_prefix_token": query.last_prefix_token, "choice_tokens": list(query.tokens)}
                    for name, query in queries.items()},
    }
    return rows, recipe, queries


def lock_json(path, expected):
    if path.exists():
        if read(path) != expected:
            raise ValueError(f"Frozen {path.name} differs from the current contract")
    else:
        write_json(path, expected)


def work_root(out):
    out = out.resolve()
    data = Path("data").resolve()
    if out == data or not out.is_relative_to(data):
        raise ValueError("Use an ignored project data subdirectory for the per-photo cache")
    out.mkdir(parents=True, exist_ok=True)
    return out


def pilot_order(count):
    return random.Random(PILOT_SEED).sample(range(count), PILOT_PHOTOS)


def finite_logits(values, size):
    if len(values) != size or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError("Invalid single-token choice logits")
    return values


def picked(values):
    values = finite_logits(values, len(PICK))
    best = max(values)
    if values.count(best) != 1:
        return "I", ()
    choice, indices, _ = PICK[values.index(best)]
    return choice, tuple(CODES[i] for i in indices)


def margin(values):
    yes, no, unknown = finite_logits(values, 3)
    return yes - max(no, unknown)


def outcome(record):
    choice, candidates = picked(record["select_logits"])
    if set(record["gates"]) != set(candidates):
        raise ValueError("The gate cache must cover exactly the stage-one candidate codes")
    accepted = set()
    for code in candidates:
        environment, protection = record["gates"][code]
        if min(margin(environment), margin(protection)) >= MARGIN:
            accepted.add(code)
    return choice, set(candidates), accepted


def query_logits(model, processor, picture, query):
    inputs = processor(text=[query.prefix], images=[picture], padding=True, return_tensors="pt")
    if inputs["image_grid_thw"].shape != (1, 3) or int(inputs["input_ids"][0, -1]) != query.last_prefix_token:
        raise ValueError("Processor changed one-image layout or the scored assistant boundary")
    result = model(**inputs.to("cuda:0"), logits_to_keep=1, use_cache=False)
    if result.logits.ndim != 3 or result.logits.shape[:2] != (1, 1):
        raise RuntimeError("Expected exactly one prefill position from the pinned Qwen model")
    return finite_logits(result.logits[0, 0, list(query.tokens)].float().tolist(), len(query.tokens))


def measure_image(model, processor, queries, image_root, image, image_sha):
    from PIL import Image, ImageOps

    start = time.perf_counter()
    path = (image_root / image).resolve()
    if not path.is_relative_to(image_root):
        raise ValueError("A frozen image leaves the official train root")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != image_sha:
        raise ValueError("Frozen input photo bytes differ")
    with Image.open(io.BytesIO(raw)) as photo:
        picture = ImageOps.exif_transpose(photo).convert("RGB")
    decoded = time.perf_counter()
    select = query_logits(model, processor, picture, queries["select"])
    selected = time.perf_counter()
    _, candidates = picked(select)
    gates = {code: [query_logits(model, processor, picture, queries[f"{code}:{question}"])
                    for question in ("environment", "protection")] for code in candidates}
    return {"select_logits": select, "gates": gates,
            "seconds": {"input": decoded-start, "select": selected-decoded, "gates": time.perf_counter()-selected}}


def row_file(out, index):
    return out / "rows" / f"{index:04d}.json"


def read_row(path, index, image_sha, contract):
    row = read(path)
    if (row["index"] != index or row["image_sha256"] != image_sha or row["context_sha256"] != contract
            or set(row["seconds"]) != {"input", "select", "gates"}
            or any(type(s) not in (int, float) or not math.isfinite(s) or s < 0 for s in row["seconds"].values())
            or not 0 < row["cuda_peak_allocated_bytes"] <= row["cuda_peak_reserved_bytes"]):
        raise ValueError("A saved row does not match the frozen image, context or timing")
    outcome(row)
    return row


def resident_layers(model, base):
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer

    cfg = model.get_base_model().config
    quant = cfg.quantization_config
    quant = quant if isinstance(quant, dict) else quant.to_dict()
    layers = tuple(module for module in model.modules() if isinstance(module, BaseTunerLayer))
    if (not getattr(model, "is_loaded_in_4bit", False) or cfg.model_type != "qwen3_vl"
            or Path(cfg._name_or_path).resolve() != base.resolve() or cfg._attn_implementation != "sdpa"
            or not quant.get("load_in_4bit") or quant.get("bnb_4bit_quant_type") != "nf4"
            or quant.get("bnb_4bit_use_double_quant") is not True
            or str(quant.get("bnb_4bit_compute_dtype")).removeprefix("torch.") != "bfloat16"
            or set(quant.get("llm_int8_skip_modules") or ()) != {"visual", "lm_head"}
            or not layers or any(layer.merged for layer in layers)
            or any(param.device != torch.device("cuda:0") for param in model.parameters())):
        raise RuntimeError("Expected unmerged PEFT over pinned resident 8B NF4, SDPA and BF16 compute")
    return layers


def pilot_summary(records):
    parts = {name: sum(row["seconds"][name] for row in records) for name in ("input", "select", "gates")}
    seconds = sum(parts.values())
    return {"photos": len(records), "candidate_code_queries": sum(len(row["gates"]) for row in records),
            "seconds": {name: round(value, 3) for name, value in parts.items()},
            "measured_row_seconds": round(seconds, 3),
            "estimated_full_row_seconds_excluding_cold_load": round(seconds * PHOTOS / len(records), 1)}


def run_with_model(model, processor, *, base, adapter, manifest, image_root, out, pilot_only=False):
    """Continue from B2's resident loader. Never use adapted results or KV."""
    import msvcrt
    import torch

    out = work_root(out)
    images, recipe, queries = frozen_recipe(base, adapter, manifest, processor)
    image_root = image_root.resolve()
    if (torch.cuda.device_count() != 1 or "RTX 3060" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Expected one local RTX 3060")
    layers = resident_layers(model, base)
    runtime = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
               **{name: version(name) for name in ("transformers", "peft", "bitsandbytes", "Pillow")}}
    context = {"recipe_sha256": sha_json(recipe), "runtime": runtime, "unmerged_adapter_layers": len(layers)}
    context_sha = sha_json(context)
    pilot = pilot_order(len(images))
    selected = pilot if pilot_only else pilot + sorted(set(range(len(images))) - set(pilot))
    reused, written, pilot_records = 0, 0, []
    with (out / "run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        lock_json(out / "recipe.json", recipe)
        lock_json(out / "context.json", context)
        (out / "rows").mkdir(exist_ok=True)
        (out / "receipts").mkdir(exist_ok=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.eval()
        try:
            with torch.inference_mode(), model.disable_adapter():
                if not all(layer.disable_adapters and not layer.merged for layer in layers):
                    raise RuntimeError("Could not prove the unmerged LoRA is disabled during probe inference")
                for done, index in enumerate(selected, start=1):
                    image, image_sha = images[index]
                    path = row_file(out, index)
                    if path.exists():
                        row = read_row(path, index, image_sha, context_sha)
                        reused += 1
                    else:
                        row = measure_image(model, processor, queries, image_root, image, image_sha)
                        row.update(index=index, image_sha256=image_sha, context_sha256=context_sha,
                                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                        outcome(row)
                        write_json(path, row)
                        written += 1
                    if done <= PILOT_PHOTOS:
                        pilot_records.append(row)
                    if done == PILOT_PHOTOS:
                        pilot_result = pilot_summary(pilot_records)
                        write_json(out / "pilot-timing.json", pilot_result)
                        print(f"B1_PILOT {pilot_result}", flush=True)
                    if done % 25 == 0 or done == len(selected):
                        print(f"B1_ROWS {done}/{len(selected)} reused={reused} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        finally:
            write_json(out / "receipts" / f"{time.time_ns()}.json", {
                "context_sha256": context_sha, "selected_photos": len(selected), "new_photos": written,
                "reused_photos": reused, "complete": written + reused == len(selected),
                "resident_probe_wall_seconds": round(time.perf_counter()-start, 3),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            })


def metrics(truth, outputs):
    false_any_candidate = false_any = false_normal = empty_selection = unknown_selection = other_choice = 0
    stage_one_codes, stage_two_codes = Counter(), Counter()
    code_counts = {code: Counter() for code in CODES}
    nonrare = rare = candidate_restored = confirmed_restored = 0
    for (gold, normal), (choice, candidates, confirmed) in zip(truth, outputs, strict=True):
        stage_one_codes.update(candidates)
        stage_two_codes.update(confirmed)
        empty_selection += choice == "A"
        unknown_selection += choice == "I"
        other_choice += bool(candidates)
        nonrare += not gold
        rare += bool(gold)
        candidate_restored += bool(gold and gold <= candidates)
        confirmed_restored += bool(gold and gold <= confirmed)
        false_any_candidate += not gold and bool(candidates)
        false_any += not gold and bool(confirmed)
        false_normal += normal and bool(confirmed)
        for code in CODES:
            c = code_counts[code]
            c["official_positives"] += code in gold
            c["official_negatives"] += code not in gold
            c["candidate_tp"] += code in candidates and code in gold
            c["candidate_fn"] += code not in candidates and code in gold
            c["candidate_fp"] += code in candidates and code not in gold
            c["final_tp"] += code in confirmed and code in gold
            c["final_fn"] += code not in confirmed and code in gold
            c["final_fp"] += code in confirmed and code not in gold
            c["final_fp_among_nonrare"] += code in confirmed and not gold
    return {"by_code": code_counts, "rare_positive_photos": rare, "nonrare_photos": nonrare,
            "rare_positive_photos_fully_recovered_in_candidates": candidate_restored,
            "rare_positive_photos_fully_recovered_in_final": confirmed_restored,
            "nonrare_photos_with_any_candidate_rare_fp": false_any_candidate,
            "nonrare_photos_with_any_final_rare_fp": false_any,
            "normal_photos": sum(normal for _, normal in truth), "normal_photos_with_any_final_rare_fp": false_normal,
            "selection_background_photos": empty_selection, "selection_unknown_photos": unknown_selection,
            "selection_candidate_photos": other_choice,
            "candidate_code_counts": {c: stage_one_codes[c] for c in CODES},
            "confirmed_code_counts": {c: stage_two_codes[c] for c in CODES}}


def score(out, manifest, result):
    images = image_rows(manifest)
    recipe = read(out / "recipe.json")
    context = read(out / "context.json")
    if (recipe["probe_source_sha256"] != digest(Path(__file__)) or recipe["manifest_sha256"] != MANIFEST_SHA
            or context["recipe_sha256"] != sha_json(recipe)):
        raise ValueError("Probe source, manifest or frozen recipe changed")
    context_sha = sha_json(context)
    state = [read_row(row_file(out, i), i, image_sha, context_sha) for i, (_, image_sha) in enumerate(images)]
    truth = [(set(CODES) & set(row["codes"]), not row["codes"]) for row in read(manifest)]
    outputs = [outcome(row) for row in state]
    result_payload = metrics(truth, outputs)
    if result_payload["nonrare_photos"] != 963 or result_payload["rare_positive_photos"] != 3:
        raise ValueError("Incorrect official development target support")
    receipts = [read(path) for path in sorted((out / "receipts").glob("*.json"))]
    counted_rows = sum(r["new_photos"] for r in receipts)
    if any(r["context_sha256"] != context_sha for r in receipts) or counted_rows > PHOTOS:
        raise ValueError("Inconsistent probe telemetry for the frozen candidate rows")
    write_json(result, {
        "scope": "original 8B NF4 development probe; independent in-domain external controls not available",
        "promotion": "hold_external_controls_missing", "training_photos": 0, "training_updates": 0,
        "generated_tokens": 0, "recipe": recipe, "runtime_context_sha256": context_sha, "runtime": context["runtime"],
        "predictions_sha256": sha_json([(choice, sorted(confirmed)) for choice, _, confirmed in outputs]),
        "ordered_row_cache_sha256": sha_json(state),
        "metrics": result_payload, "pilot": pilot_summary([state[i] for i in pilot_order(PHOTOS)]),
        "telemetry": {"recorded_sessions": len(receipts), "model_load_included": False,
                      "rows_without_closed_session_telemetry": PHOTOS-counted_rows,
                      "resident_probe_wall_seconds": round(sum(r["resident_probe_wall_seconds"] for r in receipts), 3),
                      "measured_row_seconds": round(sum(sum(row["seconds"].values()) for row in state), 3),
                      "cuda_peak_allocated_bytes": max(state + receipts, key=lambda r: r["cuda_peak_allocated_bytes"])["cuda_peak_allocated_bytes"],
                      "cuda_peak_reserved_bytes": max(state + receipts, key=lambda r: r["cuda_peak_reserved_bytes"])["cuda_peak_reserved_bytes"]},
    })
    print(f"B1_SCORE photos={PHOTOS} union_fp={result_payload['nonrare_photos_with_any_final_rare_fp']} "
          f"pos_recovered={result_payload['rare_positive_photos_fully_recovered_in_final']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "run"))
    parser.add_argument("--out", type=Path, default=WORK)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    if args.pilot_only and args.phase != "run":
        parser.error("--pilot-only applies to the standalone run phase")
    if args.phase == "score":
        score(args.out, MANIFEST, args.result)
        return
    from .vlm_11code import load_qwen_processor

    processor = load_qwen_processor(BASE, PIXELS["longest_edge"])
    processor.tokenizer.padding_side = "left"
    if args.phase == "freeze":
        _, recipe, _ = frozen_recipe(BASE, ADAPTER, MANIFEST, processor)
        lock_json(work_root(args.out) / "recipe.json", recipe)
        print(f"B1_RECIPE {sha_json(recipe)}", flush=True)
        return
    from .vlm_11code import load_qwen_model

    start = time.perf_counter()
    model = load_qwen_model(BASE, use_4bit=True, adapter_dir=ADAPTER).eval()
    print(f"B1_MODEL_LOADED {time.perf_counter()-start:.1f}s", flush=True)
    run_with_model(model, processor, base=BASE, adapter=ADAPTER, manifest=MANIFEST, image_root=TRAIN,
                   out=args.out, pilot_only=args.pilot_only)


if __name__ == "__main__":
    main()
