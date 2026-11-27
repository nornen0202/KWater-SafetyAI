"""Paired offline image SFT. Train a fresh adapter per model on the locked groups."""

import argparse
import hashlib
import importlib.metadata
import json
import time

from .metrics import CODES, decode_target, unique_keys
from .phase2_shared import capture
from .phase2_paths import file_stem
from .phase2_panel import (
    PROJECT,
    RUN,
    file_hash,
    order,
    read,
    registry,
    write,
)
from .submission2_output import make_row


TASK = (
    "이 현장사진에서 위치와 상태로 확인되는 위험을 최대 세 개 판독한다. "
    "허용 코드: fall_height 높이차 추락, slip_trip 미끄러짐·걸림, caught_in_machine 끼임, "
    "struck_by_lifting 낙하·인양물, electrical_energy 감전, drowning_inundation 익수·침수, "
    "fire_explosion 화재·폭발, confined_space 밀폐공간, chemical_exposure 화학물질, "
    "thermal_weather 온열·한랭·기상, other 위 분류 밖의 실제 위험. other는 차량에 한정되지 않는다. "
    '위험이 없으면 빈 배열. 설명 없이 {"codes":["허용코드"]} JSON만 쓴다.'
)
SEED = 42
INTERNAL8 = {
    "fall_height": "fall_height",
    "slip_trip": "slip_trip",
    "caught_in_machine": "caught_in_machine",
    "struck_crushed": "struck_by_lifting",
    "electrical_energy": "electrical_energy",
    "fire_thermal": "fire_explosion",
    "hazardous_exposure": "chemical_exposure",
    "collision_contact": "other",
}
TASK8 = (
    "이 현장사진에서 위치와 상태로 확인되는 위험을 최대 세 개 판독한다. 허용 코드: fall_height 높이차 추락, "
    "slip_trip 미끄러짐·걸림, caught_in_machine 끼임, struck_crushed 낙하·충돌, electrical_energy 감전, "
    "fire_thermal 화재·열, hazardous_exposure 유해물질 노출, collision_contact 차량·장비 접촉. "
    '위험이 없으면 빈 배열. 설명 없이 {"codes":["허용코드"]} JSON만 쓴다.'
)


def parse_codes(text, eight=False):
    obj = json.loads(text.strip(), object_pairs_hook=unique_keys)
    if not isinstance(obj, dict) or set(obj) != {"codes"}:
        raise ValueError("wrong object")
    codes = obj["codes"]
    if (
        not isinstance(codes, list)
        or len(codes) > 3
        or any(c not in (INTERNAL8 if eight else CODES) for c in codes)
        or len(set(codes)) != len(codes)
    ):
        raise ValueError("invalid code list")
    return [INTERNAL8[c] for c in codes] if eight else codes


def fingerprint(model):
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if "lora_" in name:
            h.update(name.encode())
            h.update(p.detach().float().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def observed_thinking_tokens(tokenizer, prompt_ids, new_ids):
    start = tokenizer.encode("<think>", add_special_tokens=False)
    stop = tokenizer.encode("</think>", add_special_tokens=False)
    if not start or not stop:
        return {"delimited_reasoning_tokens": None}

    # The fixed task text contains no tags. Some chat templates close an empty block in the prefix.
    def scan(seq, active):
        i = count = 0
        while i < len(seq):
            if seq[i : i + len(start)] == start:
                active = True
                i += len(start)
            elif seq[i : i + len(stop)] == stop:
                active = False
                i += len(stop)
            else:
                count += int(active)
                i += 1
        return active, count

    active, _ = scan(prompt_ids, False)
    unclosed, count = scan(new_ids, active)
    return {
        "delimited_reasoning_tokens": count,
        "reasoning_open_at_generation_start": active,
        "reasoning_open_at_stop": unclosed,
    }


def inputs(proc, row, eight=False):
    from PIL import Image, ImageOps

    with Image.open(PROJECT / "data/train" / row["image"]) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": row.get("question", TASK8 if eight else TASK)},
            ],
        }
    ]
    prefix = proc.apply_chat_template(
        msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    b = proc(text=[prefix], images=[image], return_tensors="pt")
    visual = int(b["image_grid_thw"][0].prod()) // 4
    if visual > proc.image_processor.size["longest_edge"] // (32**2):
        raise ValueError(f"Pixel budget exceeded: {visual}")
    return b.to("cuda"), visual


def evaluate_model(model, proc, rows, ids, folder, eight=False, shared_features=None):
    import torch

    model.eval()
    result = {}
    for s in ids:
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        rec = {}
        visual = None
        try:
            b, visual = inputs(proc, rows[s], eight)
            with torch.inference_mode(), capture(model, shared_features, s):
                tokens = model.generate(
                    **b, max_new_tokens=96, do_sample=False, use_cache=True
                )[0, b["input_ids"].shape[-1] :]
            text = proc.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_with_special = proc.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            rec.update(
                text=text,
                raw_with_special=raw_with_special,
                new_tokens=len(tokens),
                limit_reached=len(tokens) == 96,
                emitted_think_tags=any(
                    t in raw_with_special for t in ("<think>", "</think>")
                ),
                **observed_thinking_tokens(
                    proc.tokenizer, b["input_ids"][0].tolist(), tokens.tolist()
                ),
            )
            codes = parse_codes(text, eight)
            rec["valid"] = True
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            codes = []
            rec.update(valid=False, error=str(e))
        final = make_row(order("kwater/scale", s)[:32], codes)
        decode_target(final)
        torch.cuda.synchronize()
        rec.update(
            final=final,
            seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            visual_tokens=visual,
        )
        write(folder / (file_stem(s) + ".json"), rec)
        result[s] = rec
    return evaluation_summary(rows, result)


def evaluation_summary(rows, result):
    from .metrics import evaluate

    gold = {s: frozenset(rows[s]["codes"]) for s in result}
    pred = {s: frozenset(r["final"]["hazards"]) for s, r in result.items()}
    return {
        "metrics": evaluate(gold, pred),
        "rows": len(result),
        "valid": sum(v["valid"] for v in result.values()),
        "seconds": sum(v["seconds"] for v in result.values()),
        "peak_allocated_bytes": max(v["peak_allocated_bytes"] for v in result.values()),
    }


def first_checkpoint(model, proc, rows, train, test, folder, initial, step, batch):
    import torch
    from peft import get_peft_model_state_dict
    from .phase2_release import changes
    from .phase2_score import readlines

    checkpoint = folder / "first-update" / step["sha256"][:16]
    model.save_pretrained(
        checkpoint, selected_adapters=["default"], save_embedding_layers=False
    )
    sid = test[0]
    assert rows[sid]["group"] not in {rows[s]["group"] for s in train}
    from safetensors.torch import load_file

    prior_file = folder / "initial-adapter/adapter_model.safetensors"
    prior = load_file(str(prior_file))
    updated = load_file(str(checkpoint / "adapter_model.safetensors"))
    if prior.keys() != updated.keys():
        raise ValueError("The adapter tensor identities changed")
    movement = {}
    for key, tensor in prior.items():
        delta = updated[key].float() - tensor.float()
        movement[key] = dict(
            parameters=delta.numel(),
            changed=int(torch.count_nonzero(delta)),
            l2=float(torch.linalg.vector_norm(delta)),
            max_abs=float(delta.abs().max()),
        )
    if not any(v["changed"] for v in movement.values()):
        raise ValueError("Saved first-step tensor files are identical")
    receipt = dict(
        initial=initial,
        initial_adapter_file=str(prior_file),
        initial_adapter_sha256=file_hash(prior_file),
        parameter_changes=movement,
        optimizer_step=step,
        observed_batch=batch,
        planned_training_images=len(train),
        training_groups=sorted({rows[s]["group"] for s in train}),
        adapter_sha256=file_hash(checkpoint / "adapter_model.safetensors"),
        adapter=str(checkpoint),
        evaluation_id=sid,
        evaluation_group=rows[sid]["group"],
    )
    write(folder / "first-checkpoint.json", receipt)
    print("FIRST_CHECKPOINT", folder.name, step["sha256"][:16], flush=True)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        name = "p38_first_reloaded"
        model.load_adapter(
            str(checkpoint),
            adapter_name=name,
            is_trainable=False,
            local_files_only=True,
        )
        try:
            original = get_peft_model_state_dict(
                model, adapter_name="default", save_embedding_layers=False
            )
            copy = get_peft_model_state_dict(
                model, adapter_name=name, save_embedding_layers=False
            )
            if original.keys() != copy.keys() or not all(
                torch.equal(v, copy[k]) for k, v in original.items()
            ):
                raise ValueError("The first independently reloaded adapter differs")
            model.set_adapter(name, inference_mode=True)
            evaluate_model(model, proc, rows, [sid], checkpoint / "evaluation")
        finally:
            model.set_adapter("default")
            model.delete_adapter(name)
            model.train()
    if fingerprint(model) != step["sha256"]:
        raise ValueError("The held-out first-step check changed the training adapter")
    before = read(folder / "base" / (file_stem(sid) + ".json"))
    after = read(checkpoint / "evaluation" / (file_stem(sid) + ".json"))
    gold = next(
        r["target"]["hazards"]
        for r in readlines(RUN / "dev-train.jsonl")
        if r["sample_id"] == sid
    )
    write(
        folder / "first-paired.json",
        dict(
            checkpoint=receipt,
            unadapted=before,
            reloaded_first_step=after,
            changes=changes(before["final"], after["final"], gold),
            equal_loaded_parameters=True,
            scope="Prespecified first held-out photo after one update; one diagnostic observation.",
        ),
    )
    print("FIRST_HELD_GROUP_PAIR", folder.name, sid, flush=True)


def run(
    name,
    weight=1,
    eight=False,
    series=False,
    external=False,
    yield_to_large=False,
    verifier=False,
):
    import ipaddress
    import sys
    from collections import Counter

    network = Counter()
    active = True

    def local_socket(event, args):
        if not active or event not in (
            "socket.connect",
            "socket.getaddrinfo",
            "socket.sendto",
        ):
            return
        addr = args[0] if event == "socket.getaddrinfo" else args[-1]
        host = addr[0] if isinstance(addr, tuple) else addr
        if not isinstance(host, (str, bytes)) or (
            event != "socket.getaddrinfo" and not isinstance(addr, tuple)
        ):
            return
        host = host.decode() if isinstance(host, bytes) else host
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host in ("localhost", "", "ip6-localhost")
        network[event + ("_loopback" if local else "_blocked")] += 1
        if not local:
            raise PermissionError("Network disabled during the local image study")

    sys.addaudithook(local_socket)
    try:
        fit_series(name, weight, eight, series, external, yield_to_large, verifier)
    finally:
        active = False
        write(
            RUN / ("lora-" + name) / "offline-network.json",
            {
                "python_socket_audit": dict(network),
                "scope": "Local weights and processors. Python non-loopback connects and DNS blocked in this process.",
            },
        )


def fit_series(
    name, weight, eight, series, external, yield_to_large=False, verifier=False
):
    import torch
    import transformers
    from importlib import import_module
    from unittest.mock import patch

    # This isolated training process only runs greedy generation without a draft model.
    # Match the Linux wheel bundle which omits the optional assisted-decoding ROC dependency.
    with patch.object(transformers.utils, "is_sklearn_available", return_value=False):
        candidate_module = import_module("transformers.generation.candidate_generator")
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    path = {
        "3vl": PROJECT / "data/baseline/base_model",
        "35": RUN / "qwen35",
        "8b": RUN / "large-models/8b",
        "9b": RUN / "large-models/9b",
    }[name]
    if name in ("8b", "9b") and not (path / "complete.json").is_file():
        raise ValueError("Pinned official source download has not finished")
    versions = {
        p: importlib.metadata.version(p)
        for p in (
            "torch",
            "torchvision",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "peft",
            "safetensors",
            "tokenizers",
            "pillow",
        )
    }
    transformers.set_seed(SEED)
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    proc.image_processor.size = {
        "shortest_edge": 64 * 32**2,
        "longest_edge": 128 * 32**2,
    }
    start = time.perf_counter()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["model.visual", "visual", "lm_head"],
    )
    from .adapter_scale_pilot import weight_reader
    import os

    reader = "tensor-stream" if os.name == "nt" else "default"
    with weight_reader(reader):
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=quant,
            device_map={"": 0},
            local_files_only=True,
        )
    quantized_names = [
        n for n, m in model.named_modules() if type(m).__name__ == "Linear4bit"
    ]
    if not quantized_names or any(
        "visual" in n.split(".") or n.endswith("lm_head") for n in quantized_names
    ):
        raise ValueError("NF4 must be confined to the language backbone")
    write(
        RUN / ("lora-" + name) / "quantized-modules.json",
        {"quantized_modules": quantized_names, "source": str(path), "reader": reader},
    )
    write(
        RUN / ("lora-" + name) / "greedy-dependencies.json",
        dict(
            assisted_generation=False,
            optional_assistant_roc_imported=hasattr(candidate_module, "roc_curve"),
            required_inference_wheels=read(RUN / "static-package.json")["linux_wheels"],
        ),
    )
    model.generation_config.do_sample = False
    for key in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, key, None)
    load = time.perf_counter() - start
    arms = (
        [dict(weight=weight, eight=eight)]
        if not series
        else [dict(weight=w, eight=e) for w, e in [(1, False), (6, False), (1, True)]]
    )
    if external:
        for treatment in ("resample", "state", "permuted"):
            arms.append(dict(weight=1, eight=False, treatment=treatment))
    if verifier:
        if series or external or weight != 1 or eight or name != "3vl":
            raise ValueError(
                "The verifier pilot starts from one frozen Qwen3-VL-4B base"
            )
        arms = [dict(weight=1, eight=False, treatment="verifier")]
    if series:
        arms.extend(
            [
                dict(weight=1, eight=False, visual_limit=256),
                dict(weight=1, eight=False, vision_lora=True),
            ]
        )
    for arm in arms:
        if yield_to_large and any(
            (RUN / "large-models" / large / "complete.json").is_file()
            and not (RUN / ("lora-" + large) / "standalone-nf4-check.json").is_file()
            for large in ("8b", "9b")
        ):
            return
        model = train_adapter(
            model, proc, name, load=load, versions=versions, base_path=path, **arm
        )
    if (
        name in ("8b", "9b")
        and os.environ.get("KW_PHASE2_SFT_STAGE", "pilot") == "pilot"
    ):
        import gc
        from peft import PeftModel

        out = RUN / ("lora-" + name)
        packed = out / "standalone-nf4"
        model.save_pretrained(packed, max_shard_size="1GB")
        proc.save_pretrained(packed)
        if not read(packed / "config.json").get("quantization_config"):
            raise ValueError("Serialized base omitted its quantization metadata")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        from unittest.mock import patch
        from transformers import modeling_utils
        from pathlib import Path

        opened = []
        with weight_reader(reader):
            original_open = modeling_utils.safe_open

            def isolated_open(filename, *args, **kwargs):
                path_used = Path(filename).resolve()
                if not path_used.is_relative_to(packed.resolve()):
                    raise PermissionError(
                        "Standalone reload attempted to read an external weight shard"
                    )
                opened.append(path_used.name)
                return original_open(filename, *args, **kwargs)

            with patch.object(modeling_utils, "safe_open", isolated_open):
                standalone = AutoModelForImageTextToText.from_pretrained(
                    packed,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map={"": 0},
                )
        if not opened:
            raise ValueError("The safetensors weight read audit observed no shards")
        standalone = PeftModel.from_pretrained(
            standalone, out / "adapter", is_trainable=False
        )
        processor = AutoProcessor.from_pretrained(packed, local_files_only=True)
        processor.image_processor.size = proc.image_processor.size
        cold = time.perf_counter() - t0
        sid = read(RUN / "sft-split.json")["test"][0]
        expected = read(out / "adapted" / (file_stem(sid) + ".json"))
        check = evaluate_model(
            standalone, processor, registry(), [sid], out / "standalone-check"
        )
        observed = read(out / "standalone-check" / (file_stem(sid) + ".json"))
        write(
            out / "standalone-nf4-check.json",
            dict(
                root=str(packed),
                inputs=[str(packed), str(out / "adapter")],
                serialized_files={
                    p.name: p.stat().st_size for p in packed.iterdir() if p.is_file()
                },
                isolated_shards_opened=sorted(set(opened)),
                original_bf16_argument=False,
                local_files_only=True,
                load_seconds=cold,
                sample_id=sid,
                valid=check["valid"],
                same_text=observed.get("text") == expected.get("text"),
                same_final=observed["final"] == expected["final"],
                metrics=check,
            ),
        )


def train_adapter(
    model,
    proc,
    name,
    weight,
    eight,
    load,
    versions,
    treatment=None,
    visual_limit=128,
    vision_lora=False,
    base_path=None,
):
    import torch
    from transformers import set_seed
    from peft import LoraConfig, PeftModel, get_peft_model

    set_seed(SEED)
    proc.image_processor.size = {
        "shortest_edge": 64 * 32**2,
        "longest_edge": visual_limit * 32**2,
    }
    import os

    stage = os.environ.get("KW_PHASE2_SFT_STAGE", "pilot")
    plans = {
        "pilot": RUN / "sft-split.json",
        "panel170": RUN / "large-models/expansion-split.json",
        "outer1": RUN / "large-models/expansion-outer1.json",
        **{
            f"outer1-inner{i}": RUN / f"large-models/outer1-inner{i}.json"
            for i in range(3)
        },
    }
    split_path = plans[stage]
    split, rows = read(split_path), registry()
    if stage != "pilot" and (
        weight != 1 or eight or treatment or vision_lora or visual_limit != 128
    ):
        raise ValueError(
            "Expanded comparisons use the frozen primary 11-code treatment"
        )
    tokenizers_for_budget = []
    if stage != "pilot":
        from tokenizers import Tokenizer

        tokenizers_for_budget = [
            Tokenizer.from_file(str(folder / "tokenizer.json"))
            for folder in (PROJECT / "data/baseline/base_model", RUN / "qwen35")
        ]
    train, test = split["train"], split["test"]
    assert not {rows[s]["group"] for s in train} & {rows[s]["group"] for s in test}
    out = RUN / (
        "lora-"
        + name
        + ("-weight6" if weight == 6 else "")
        + ("-eight" if eight else "")
        + ("-" + treatment if treatment else "")
        + ("-pixels256" if visual_limit != 128 else "")
        + ("-vision" if vision_lora else "")
        + ("-" + stage if stage != "pilot" else "")
    )
    evaluator = evaluate_model
    if treatment == "verifier":
        from .phase2_verifier import fit_data, evaluate_verifier

        train, test, rows, split_path = fit_data(rows)
        evaluator = evaluate_verifier
    if (out / "results.json").exists():
        previous = read(out / "results.json")
        if previous["training_split_sha256"] != file_hash(split_path) or previous[
            "adapter_file_sha256"
        ] != file_hash(out / "adapter/adapter_model.safetensors"):
            raise ValueError("Previously completed fit differs from the frozen study")
        return model
    if treatment and treatment != "verifier":
        from .phase2_external import training_rows

        train, rows = training_rows(treatment, train, rows)
    from .phase2_lineage import record

    record(out, name, base_path, train, test, rows, split_path, model)
    modules = [
        n
        for n, _ in model.named_modules()
        if "language_model.layers." in n and n.endswith(".mlp.down_proj")
    ]
    modules = sorted(modules, key=lambda n: int(n.split(".layers.")[1].split(".")[0]))[
        -4:
    ]
    if len(modules) != 4:
        raise ValueError(f"Unexpected targets: {modules}")
    if vision_lora:
        visual_modules = sorted(
            (
                n
                for n, _ in model.named_modules()
                if ".visual.blocks." in n and n.endswith(".mlp.linear_fc2")
            ),
            key=lambda n: int(n.split(".blocks.")[1].split(".")[0]),
        )[-4:]
        if len(visual_modules) != 4:
            raise ValueError("Missing late visual MLPs")
        modules += visual_modules
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    init = fingerprint(model)
    primary = RUN / ("lora-" + name)
    shared_features = out / "shared-vision" if out == primary else None
    first_step_proof = out == primary and stage == "pilot"
    if first_step_proof:
        model.save_pretrained(
            out / "initial-adapter",
            selected_adapters=["default"],
            save_embedding_layers=False,
        )
    if (
        (treatment or weight != 1 or vision_lora)
        and treatment != "verifier"
        and (primary / "results.json").exists()
        and not eight
    ):
        import shutil

        before = read(primary / "results.json")["base"]
        (out / "base").mkdir(parents=True, exist_ok=True)
        for sid in test:
            shutil.copyfile(
                primary / "base" / (file_stem(sid) + ".json"),
                out / "base" / (file_stem(sid) + ".json"),
            )
        before = dict(before, shared_unadapted_cache=str(primary / "base"))
    else:
        with model.disable_adapter():
            before = evaluator(
                model,
                proc,
                rows,
                test[:1] if first_step_proof else test,
                out / "base",
                eight,
                shared_features,
            )
    print("BASE_EVALUATED", name, before["rows"], before["valid"], flush=True)
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    steps = []
    rowlog = []
    start = time.perf_counter()
    batch_start = start
    proof_seconds = 0.0
    remaining_base_seconds = 0.0
    for i, s in enumerate(train):
        b, visual = inputs(proc, rows[s], eight)
        prompt_n = b["input_ids"].shape[1]
        if treatment == "verifier":
            target_object = rows[s]["verifier_target"]
        elif "state_answer" in rows[s]:
            target_object = {"state": rows[s]["state_answer"]}
        else:
            target_object = {
                "codes": (
                    [c for c, v in INTERNAL8.items() if v in rows[s]["codes"]]
                    if eight
                    else [c for c in CODES if c in rows[s]["codes"]]
                )
            }
        answer = json.dumps(
            target_object, separators=(",", ":"), ensure_ascii=treatment != "verifier"
        )
        answer_ids = proc.tokenizer.encode(answer, add_special_tokens=False) + [
            proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
        target = torch.tensor([answer_ids], device=model.device)
        b["input_ids"] = torch.cat([b["input_ids"], target], dim=1)
        b["attention_mask"] = torch.cat(
            [b["attention_mask"], torch.ones_like(target)], dim=1
        )
        if "mm_token_type_ids" in b:
            b["mm_token_type_ids"] = torch.cat(
                [b["mm_token_type_ids"], torch.zeros_like(target)], dim=1
            )
        labels = b["input_ids"].clone()
        labels[:, :prompt_n] = -100
        supervised = len(answer_ids)
        if tokenizers_for_budget:
            supervised = min(
                len(tok.encode(answer, add_special_tokens=False).ids) + 1
                for tok in tokenizers_for_budget
            )
            if supervised > len(answer_ids):
                raise ValueError(
                    "New tokenizer has fewer decision tokens than the frozen shared budget"
                )
            if supervised < len(answer_ids):
                keep = (
                    torch.linspace(
                        0, len(answer_ids) - 1, supervised, device=model.device
                    )
                    .round()
                    .long()
                )
                missing = torch.ones(
                    len(answer_ids), dtype=torch.bool, device=model.device
                )
                missing[keep] = False
                labels[:, prompt_n:][:, missing] = -100
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        if weight == 1:
            with capture(model, shared_features, s):
                loss = model(**b, labels=labels, use_cache=False).loss
        else:
            logits = model(**b, use_cache=False).logits[0, prompt_n - 1 : -1]
            losses = torch.nn.functional.cross_entropy(
                logits.float(), target[0], reduction="none"
            )
            # Decision region includes code names or the empty-list decision, excluding JSON envelope.
            offsets = proc.tokenizer(
                answer, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            begin, end = answer.index("["), answer.rindex("]") + 1
            scale = losses.new_tensor(
                [weight if hi > begin and lo < end else 1 for lo, hi in offsets] + [1]
            )
            loss = (losses * scale).sum() / scale.sum()
        if not torch.isfinite(loss):
            raise ValueError("non-finite loss")
        accumulation = min(2, len(train) - (i // 2) * 2)
        (loss / accumulation).backward()
        rowlog.append(
            {
                "id": s,
                "loss": float(loss.detach()),
                "visual_tokens": visual,
                "target_tokens": len(answer_ids),
                "supervised_tokens": supervised,
                "seconds": time.perf_counter() - t,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        if (i + 1) % 2 == 0 or i + 1 == len(train):
            optimizer_start = time.perf_counter()
            grad = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if not (0 < grad < float("inf")):
                raise ValueError("invalid gradient")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updated = fingerprint(model)
            if updated == (steps[-1]["sha256"] if steps else init):
                raise ValueError("optimizer did not update adapter")
            batch = rowlog[(i // 2) * 2 : i + 1]
            steps.append(
                dict(
                    step=len(steps) + 1,
                    gradient_norm=grad,
                    sha256=updated,
                    images=len(batch),
                    wall_batch_seconds=time.perf_counter() - batch_start,
                    optimizer_and_fingerprint_seconds=time.perf_counter()
                    - optimizer_start,
                    peak_allocated_bytes=max(x["peak_allocated_bytes"] for x in batch),
                    peak_reserved_bytes=max(x["peak_reserved_bytes"] for x in batch),
                )
            )
            write(
                out / "optimizer.json",
                {"initial": init, "steps": steps, "rows": rowlog},
            )
            print(
                "OPTIMIZER_STEP",
                name,
                len(steps),
                rowlog[-1]["loss"],
                updated[:12],
                flush=True,
            )
            if len(steps) == 1 and first_step_proof:
                proof_start = time.perf_counter()
                first_checkpoint(
                    model, proc, rows, train, test, out, init, steps[0], batch
                )
                proof_seconds = time.perf_counter() - proof_start
                remaining_start = time.perf_counter()
                # With every adapter disabled these are the same frozen base weights.
                with (
                    torch.random.fork_rng(devices=[torch.cuda.current_device()]),
                    model.disable_adapter(),
                ):
                    evaluate_model(
                        model,
                        proc,
                        rows,
                        test[1:],
                        out / "base",
                        eight,
                        shared_features,
                    )
                model.train()
                before = evaluation_summary(
                    rows,
                    {
                        sid: read(out / "base" / (file_stem(sid) + ".json"))
                        for sid in test
                    },
                )
                remaining_base_seconds = time.perf_counter() - remaining_start
            batch_start = time.perf_counter()
    train_seconds = time.perf_counter() - start - proof_seconds - remaining_base_seconds
    checkpoint = out / "adapter"
    model.save_pretrained(checkpoint)
    expected = fingerprint(model)
    backbone = model.unload()
    reloaded = PeftModel.from_pretrained(backbone, checkpoint, is_trainable=False)
    if fingerprint(reloaded) != expected:
        raise ValueError("adapter reload mismatch")
    after = evaluator(reloaded, proc, rows, test, out / "adapted", eight)
    if (
        name == "35"
        and stage == "pilot"
        and weight == 1
        and not eight
        and visual_limit == 128
        and not vision_lora
        and (RUN / "external/audit.json").exists()
    ):
        from .phase2_external import evaluate_states, write_qualitative

        evaluate_states(reloaded, proc, out)
        write_qualitative(reloaded, proc, out)
    write(
        out / "results.json",
        {
            "model": name,
            "seed": SEED,
            "versions": versions,
            "rank": 8,
            "alpha": 16,
            "dropout": 0,
            "modules": modules,
            "trainable": sum(p.numel() for p in params),
            "lr": 2e-4,
            "accumulation": 2,
            "decision_token_weight": weight,
            "taxonomy": 8 if eight else 11,
            "external_or_resample_treatment": treatment,
            "task": "image_proposal_verification"
            if treatment == "verifier"
            else "hazard_codes",
            "maximum_visual_tokens": visual_limit,
            "vision_lora": vision_lora,
            "training_precision": "NF4 double quant, bf16 compute, visual and lm_head excluded",
            "training_split_sha256": file_hash(split_path),
            "stage": stage,
            "train_groups": len(
                {rows[s]["group"] for s in train if "group" in rows[s]}
            ),
            "effective_target_tokens": sum(r["supervised_tokens"] for r in rowlog),
            "updates_per_second": len(steps) / train_seconds,
            "evaluation_precision": "same NF4 base",
            "train_rows": len(train),
            "updates": len(steps),
            "load_seconds": load,
            "training_seconds": train_seconds,
            "first_checkpoint_save_reload_and_eval_seconds": proof_seconds,
            "remaining_base_evaluation_seconds": remaining_base_seconds,
            "optimizer_steps_per_training_second": len(steps) / train_seconds,
            "reasoning": "enable_thinking=False, greedy, max_new_tokens="
            + str(256 if treatment == "verifier" else 96),
            "initial": init,
            "reloaded": expected,
            "adapter_file_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
            "base": before,
            "adapted": after,
        },
    )
    print(
        "COMPLETED",
        name,
        before["metrics"]["macro_f1"],
        after["metrics"]["macro_f1"],
        flush=True,
    )
    return reloaded.unload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["3vl", "35", "8b", "9b", "both"])
    parser.add_argument(
        "--stage",
        choices=[
            "pilot",
            "panel170",
            "outer1",
            *(f"outer1-inner{i}" for i in range(3)),
        ],
        default="pilot",
    )
    parser.add_argument("--weight", type=int, choices=[1, 6], default=1)
    parser.add_argument("--eight", action="store_true")
    parser.add_argument(
        "--yield-to-large",
        action="store_true",
        help="Return between treatment arms when a larger frozen checkpoint is ready",
    )
    parser.add_argument(
        "--verifier",
        action="store_true",
        help="Fit the frozen matched accept/reject image pairs on the 3-VL base",
    )
    parser.add_argument(
        "--series",
        action="store_true",
        help="Also test decision weighting and original eight-code taxonomy, one factor per fit",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run matched resampling / localized 71407 states / swapped-state treatments",
    )
    args = parser.parse_args()
    import os

    os.environ["KW_PHASE2_SFT_STAGE"] = args.stage
    if args.stage != "pilot" and (
        args.model == "both"
        or args.series
        or args.external
        or args.eight
        or args.verifier
        or args.yield_to_large
        or args.weight != 1
    ):
        parser.error("Expansion requires exactly one frozen primary model")
    if args.model == "both":
        if args.verifier or args.yield_to_large:
            parser.error(
                "Use a single model for a verifier or resumable treatment series"
            )
        import gc

        run("3vl")
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        write(
            RUN / "between-lora-models.json",
            dict(
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
            ),
        )
        run("35", args.weight, args.eight, args.series, args.external)
    else:
        run(
            args.model,
            args.weight,
            args.eight,
            args.series,
            args.external,
            args.yield_to_large,
            args.verifier,
        )
