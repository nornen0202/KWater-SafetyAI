"""Reproduce the single-8B verifier design audit without importing PyTorch."""

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import struct
from typing import Any
from zipfile import ZipFile

import numpy as np
from tokenizers import Tokenizer

from .fold4_frozen_verifier import extract, load
from .law_reranker_11code import read_selected_cases
from .metrics import CODES


def header_audit():
    base4 = Path("data/baseline/base_model")
    base8 = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
    a8 = Path("data/vlm-scale-seoul-8b-fold4-20261128/fit")
    zip7 = Path("data/submissions/kwater-safeground-v3-full1045-hybrid-20261128.zip")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def inspect(p):
        with p.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if not 0 < n < 40 * 1024 * 1024:
                raise ValueError(p)
            h = json.loads(f.read(n))
            d = Counter()
            for k, v in h.items():
                if k != "__metadata__":
                    d[v["dtype"]] += v["data_offsets"][1] - v["data_offsets"][0]
            if sum(d.values()) + n + 8 != p.stat().st_size:
                raise ValueError(f"incomplete {p.name}")
            return {"bytes": p.stat().st_size, "tensor_bytes_by_dtype": dict(d)}

    def info(base):
        model = [inspect(p) for p in sorted(base.glob("model-*.safetensors"))]
        return {
            "config_sha256": sha(base / "config.json"),
            "tokenizer_sha256": sha(base / "tokenizer.json"),
            "shards": model,
            "tensor_shard_sizes": sum(x["bytes"] for x in model),
        }

    m4, m8 = info(base4), info(base8)
    actual8 = json.loads(Path("data/vlm-scale-seoul-8b-fold4-20261128/eval1/generation.json").read_bytes())
    if m8["config_sha256"] != actual8["base_config_sha256"]:
        raise RuntimeError("8B base revision differs")
    adapters = {}
    for i in (1, 2):
        x = a8 / f"epoch-{i}"
        w = inspect(x / "adapter_model.safetensors")
        cfg = json.loads((x / "adapter_config.json").read_bytes())
        adapters[str(i)] = {
            **w,
            "config_bytes": (x / "adapter_config.json").stat().st_size,
            "target_modules": cfg["target_modules"],
            "config_sha256": sha(x / "adapter_config.json"),
        }
    with ZipFile(zip7) as z:
        index = z.infolist()
        members = {p.filename: p.file_size for p in index}
        stored = all(i.compress_type == 0 for i in index)
        tensors4 = sum(
            v for k, v in members.items() if k.startswith("base_model/") and k.endswith(".safetensors")
        )
        old_adapter = sum(v for k, v in members.items() if k.startswith("adapter/"))
        zipped_base = sum(v for k, v in members.items() if k.startswith("base_model/"))
        spec = sum(v for k, v in members.items() if k.startswith("specialist/"))
        overhead = zip7.stat().st_size - sum(members.values())
        new8_model_files = [
            p
            for p in base8.iterdir()
            if p.is_file()
            and (
                p.name in {Path(k).name for k in members if k.startswith("base_model/")}
                or p.name.endswith(".safetensors")
                or p.name == "chat_template.json"
            )
        ]
        new8_bytes = sum(p.stat().st_size for p in new8_model_files)
        assert tensors4 == m4["tensor_shard_sizes"] and stored

    def tok(p):
        t = Tokenizer.from_file(str(p))
        s = '<|im_start|>assistant\n{"decision":"'
        pref = t.encode(s, add_special_tokens=False).ids
        d = {}
        for ans in ("accept", "reject"):
            full = t.encode(s + ans + '","x":"y"}', add_special_tokens=False).ids
            d[ans] = {
                "standalone": t.encode(ans, add_special_tokens=False).ids,
                "next": full[len(pref)],
                "prefix_matches": full[: len(pref)] == pref,
            }
        return d

    tok4 = tok(base4 / "tokenizer.json")
    tok8 = tok(base8 / "tokenizer.json")
    one = adapters["1"]["bytes"] + adapters["1"]["config_bytes"]
    two = adapters["2"]["bytes"] + adapters["2"]["config_bytes"]
    subtotal = zip7.stat().st_size - zipped_base - old_adapter + new8_bytes
    raw_only = zip7.stat().st_size - tensors4 - old_adapter + m8["tensor_shard_sizes"]
    out = {
        "models": {"4b": m4, "8b": m8},
        "adapters": adapters,
        "tokens": {"4b": tok4, "8b": tok8, "same": tok4 == tok8},
        "archive": {
            "seventh_bytes": zip7.stat().st_size,
            "stored": stored,
            "framing_bytes": overhead,
            "specialist_bytes": spec,
            "old_4b_complete_base_bytes": zipped_base,
            "old_4b_adapter_bytes": old_adapter,
            "new_8b_base_bytes": new8_bytes,
            "one_adapter_substitution_model_shards_only": raw_only + one,
            "two_adapters_substitution_model_shards_only": raw_only + one + two,
            "one_adapter_substitution_complete_base_files": subtotal + one,
            "two_adapters_substitution_complete_base_files": subtotal + one + two,
        },
        "scope": "Measured local safetensor headers and existing ZIP central directory. Replacement arithmetic includes existing archive framing, not a rebuilt 8B ZIP or measured VRAM.",
    }
    return out


def exposure_audit():
    context = {
        "Counter": Counter,
        "defaultdict": defaultdict,
        "random": random,
        "CODES": CODES,
        "Any": Any,
        "re": re,
        "json": json,
        "np": np,
    }

    def load_nodes(path, functions, constants=()):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        body = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in functions
            or isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id in constants
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)
        ]
        exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), context)

    vf = "kwater_lab/vlm_11code.py"
    v3 = "kwater_lab/v3_turnaround.py"
    load_nodes(
        vf,
        {"verify_user_text", "sanitize_text", "gold_row_to_vlm_target"},
        {"KOR_CODE_NAMES", "MINORITY_CODES", "INSPECT_USER_TEXT"},
    )
    load_nodes(v3, {"build_v3_curriculum_and_inner_val"})
    mp = "data/clean-split-v1/split_manifest_full_sealed.json"
    tp = "data/train/train.jsonl"
    op = "data/clean-specialist-v1/strict-inner-oof/fold-4.npz"
    cp = "data/v3-11code-fold4/inner_margin_cal_pairs.json"
    ts = "data/v3-11code-fold4/training_summary.json"
    rows = [
        r
        for r in json.loads(Path(mp).read_text(encoding="utf-8"))
        if r["split"] == "dev" and int(r["dev_fold"]) != 4
    ]
    train_ids = {row["sample_id"] for row in rows}
    gold = {row["sample_id"]: row for row in read_selected_cases(Path(tp), train_ids)}
    with np.load(op, allow_pickle=False) as arc:
        ids = arc["train_ids"].tolist()
        if len(ids) != 791 or len(train_ids) != 791 or set(ids) != train_ids or gold.keys() != train_ids:
            raise ValueError("Unexpected Fold 4 inner training membership")
        c, cal = context["build_v3_curriculum_and_inner_val"](rows, gold, arc["code_scores"], ids, seed=42)
    on_disk = json.loads(Path(cp).read_bytes())
    assert on_disk == cal, (len(on_disk), len(cal))
    summary = json.loads(Path(ts).read_bytes())
    curriculum_sha = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    assert summary["configuration"]["curriculum_sha256"] == curriculum_sha
    assert len(c) == summary["curriculum_examples"] == 1092 and len(cal) == 89
    by_im = {r["image"]: (r["sample_id"], r["group_id"]) for r in rows}
    assert len(by_im) == len(rows)

    def sets(seq):
        photos = {by_im[r["image"]][0] for r in seq}
        groups = {by_im[r["image"]][1] for r in seq}
        return photos, groups

    def aggregate(seq):
        return {"rows": len(seq), "photos": len(sets(seq)[0]), "groups": len(sets(seq)[1])}

    kinds = {r["kind"] for r in c}
    parts = {k: [r for r in c if r["kind"] == k] for k in sorted(kinds)}
    parts["all_verify"] = [r for r in c if r["kind"].startswith("verify_")]
    cal_group_shared = sets(parts["all_verify"])[1] & sets(cal)[1]
    c1 = [r for r in cal if r["group_id"] not in cal_group_shared]
    outer = {
        r["sample_id"]
        for r in json.loads(Path(mp).read_text(encoding="utf-8"))
        if r["split"] == "dev" and r["dev_fold"] == 4
    }
    if outer & sets(cal)[0]:
        raise ValueError("Outer photo exposure")
    return {
        "actual_inner": aggregate(cal),
        "c1_after_exclude_two_verifier_train_groups": aggregate(c1),
        "c1_by_kind": dict(Counter(r["kind"] for r in c1)),
        "c1_exact_inspect_exposure_photos": len(sets(c1)[0] & sets(parts["inspect"])[0]),
        "c1_exact_inspect_exposure_groups": len(sets(c1)[1] & sets(parts["inspect"])[1]),
        "c2_full_inspect_exposure_photos": len(sets(cal)[0] & sets(parts["inspect"])[0]),
        "outer_photo_overlap_at_calibration": len(outer & sets(cal)[0]),
        "specialist_precision_outer_labels_used": False,
        "actual_verifier_curriculum_rows": len(parts["all_verify"]),
        "actual_verifier_curriculum_by_kind": {
            k: len(v) for k, v in parts.items() if k.startswith("verify_")
        },
        "verified_curriculum_hash": curriculum_sha,
    }


def template_audit():
    from jinja2 import StrictUndefined
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    base4 = Path("data/baseline/base_model")
    base8 = Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
    prompt = Path("kwater_lab/system_prompt_v3.txt").read_text(encoding="utf-8")
    source = Path("kwater_lab/vlm_11code.py")
    scope = {"Any": Any}
    nodes = ast.parse(source.read_text(encoding="utf-8")).body
    names = [
        n
        for n in nodes
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "KOR_CODE_NAMES"
    ]
    exec(compile(ast.Module(body=names, type_ignores=[]), str(source), "exec"), scope)
    extract(source, {"verify_user_text"}, scope)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
    paths = (base4, base8)
    templates = [
        env.from_string(json.loads((base / "chat_template.json").read_bytes())["chat_template"])
        for base in paths
    ]
    tokenizers = [Tokenizer.from_file(str(base / "tokenizer.json")) for base in paths]
    stats = []
    for code in CODES:
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": scope["verify_user_text"](code)}],
            },
        ]
        renders = [
            template.render(messages=messages, add_generation_prompt=True, add_vision_id=False, tools=None)
            for template in templates
        ]
        decisions = []
        for tokenizer, rendered in zip(tokenizers, renders, strict=True):
            prefix = rendered + '{"decision":"'
            pref = tokenizer.encode(prefix, add_special_tokens=False).ids
            tokens = {}
            for word in ("accept", "reject"):
                full = tokenizer.encode(prefix + word + '","other":null}', add_special_tokens=False).ids
                tokens[word] = [full[len(pref)], full[: len(pref)] == pref]
            decisions.append(tokens)
        stats.append(
            {
                "code": code,
                "rendered_prefix_equal": renders[0] == renders[1],
                "prefix_sha256": [hashlib.sha256(r.encode()).hexdigest() for r in renders],
                "decision_token_and_prefix_check": decisions,
            }
        )
    return {
        "template_sha256": [
            hashlib.sha256((base / "chat_template.json").read_bytes()).hexdigest() for base in paths
        ],
        "rendered_equal_codes": sum(r["rendered_prefix_equal"] for r in stats),
        "codes_checked": len(stats),
        "codes": stats,
    }


def run():
    headers = header_audit()
    exposure = exposure_audit()
    templates = template_audit()
    scale = load(Path("results/paired-qwen-4b-8b-fold4-20261128.json"))
    proof = load(Path("results/seoul-l4-roadmap-experiments-20261128.json"))
    phase = proof["full1045_seventh_public200"]["runtime_phases"]
    spot = next(
        x for x in proof["resources"] if x["machine"] == "g2-standard-8" and x["provisioning"] == "SPOT"
    )
    vat = 1 + proof["cost_basis"]["vat_fraction"]
    exchange = proof["cost_basis"]["planning_krw_per_usd"]
    unit = spot["machine_hourly_usd"] * vat * exchange
    prior_full = (
        spot["planning_krw_including_vat_before_egress"]
        / spot["elapsed_through_absence_check_seconds"]
        * 3600
    )
    margin_rate = phase["margin_gate_seconds"] / phase["raw_specialist_candidates"]
    gen_rate = phase["verify_generate_seconds"] / phase["margin_gate_passed_candidates"]
    archive = headers["archive"]
    archive["model_shards_only_vs_complete_file_delta_bytes"] = (
        archive["one_adapter_substitution_complete_base_files"]
        - archive["one_adapter_substitution_model_shards_only"]
    )
    archive["one_adapter_complete_files_margin_to_20gb_bytes"] = (
        20_000_000_000 - archive["one_adapter_substitution_complete_base_files"]
    )
    archive["two_adapters_complete_files_margin_to_20gb_bytes"] = (
        20_000_000_000 - archive["two_adapters_substitution_complete_base_files"]
    )
    tensors = headers["adapters"]["1"]["tensor_bytes_by_dtype"]["F32"]
    exposure["matched_4b_8b_training_configuration"] = scale["training_controls"]
    exposure["full_inner_original_gate_accept_count"] = 50
    exposure["full_inner_original_gate_total"] = 89
    exposure["c2_safe_candidate_accepted_count_verified"] = False
    exposure["c1_safe_rows_at_most_q100_accept_by_construction"] = 1
    exposure["zero_accepts_from_one_safe_candidate_one_sided_95pct_upper"] = 0.95
    return {
        "scope": "Independent CPU-only review. Measurements are local data/provenance and archived 4B timings; the 8B verifier, adapter switch, VRAM, ZIP, and L4 runtime were not executed.",
        "archive": archive,
        "models": headers["models"],
        "eight_b_lora": headers["adapters"],
        "extra_loaded_tensor_mib_if_dtype_unchanged": tensors / 2**20,
        "tokenizer": {
            "short_fixture": headers["tokens"],
            "actual_jinja_verifier_prefix": templates,
            "pixel_settings_stock_preprocessor_equal": (
                (Path("data/baseline/base_model") / "preprocessor_config.json").read_bytes()
                == (
                    Path("data/phase2-runs/p38-20261126-49bb82ce/large-models/8b")
                    / "preprocessor_config.json"
                ).read_bytes()
            ),
        },
        "calibration": exposure,
        "time_cost": {
            "proposed_gpu_minutes_range": [6.5, 7.5],
            "draft_outer_candidates_per_arm": 52,
            "auditable_c1_inner_pairs_per_arm": 34,
            "c2_inner_pairs_per_arm": 89,
            "gpu_arms": 2,
            "c1_margin_rows_both_arms": 172,
            "c2_margin_rows_both_arms": 282,
            "proposed_outer_generations_both_arms_range": [30, 48],
            "historic_seventh_4b_seconds_per_margin": margin_rate,
            "historic_seventh_4b_seconds_per_generated_verdict": gen_rate,
            "4b_reference_arithmetic_c1_seconds_excluding_load_or_transfer": [
                172 * margin_rate + 30 * gen_rate,
                172 * margin_rate + 48 * gen_rate,
            ],
            "4b_reference_arithmetic_c2_seconds_excluding_load_or_transfer": [
                282 * margin_rate + 30 * gen_rate,
                282 * margin_rate + 48 * gen_rate,
            ],
            "historical_spot_compute_hourly_krw_with_vat": unit,
            "historical_spot_vm_disk_ip_hourly_krw_with_vat_before_egress": prior_full,
            "seven_minutes_compute_only_krw": unit * 7 / 60,
            "seven_minutes_vm_disk_ip_krw_before_egress": prior_full * 7 / 60,
            "quoted_cost_planning_range_krw": [250, 650],
            "observed_single_8b_runtime_or_invoice": False,
            "cost_note": "Seven minutes of GPU arithmetic is not startup-to-deletion billing. Provisioning, base/adapter availability, independent parity, transfer, idle and cleanup are not included in this arithmetic.",
        },
    }


if __name__ == "__main__":
    output = Path("results/sub78-single-8b-verifier-design-audit-20261128.json")
    result = run()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
