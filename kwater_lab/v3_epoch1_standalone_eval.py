"""Resume-safe Epoch-1 Boundary Standalone (`--vlm-only`) Evaluator, Raw-Generation Saver, & `main.py` Parity Verifier.

Addresses all 4 review items prior to Epoch-1 completion:
  1. Unifies `package` and `infer` `batch_size=4` and explicitly logs OOM backoff (`initial` vs `final_effective_batch_size`).
  2. Preserves the true 2-epoch cosine schedule (`configured_epochs=2, completed_epochs=1, 69/138, next_epoch=2, next_offset=0`)
     in the candidate ledger, distinguishing an epoch-boundary snapshot from a 1-epoch schedule.
  3. Never deletes existing `stage_dir` or `zip_path` (`rmtree`/`unlink` removed); reuses only explicitly verified artifacts or requires fresh paths.
  4. Strictly parses `check_output.py` stdout (`10/10` and `175/175` with `0` error lines), verifies exact `175` IDs / `39` groups
     and batch-cache original IDs, and avoids redundant 9 GB extraction/CRC/re-hashing so GPU inference is never blocked.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch

from .audit import digest
from .fold0_production_compare import image_id as anon_hex_id
from .hybrid_pipeline import severity4_miss_stats
from .law_reranker_11code import export_compact_case_bank, read_selected_cases
from .metrics import decode_target, evaluate, paired_bootstrap, unique_keys
from .package_submission import build_submission_package
from .train_head import save_atomic
from .v3_turnaround import (
    ActionSupportedLawRetriever,
    TRAIN_EVAL_MAX_PIXELS,
    _load_baseline_helpers,
    format_v3_record,
)
from .vlm_11code import (
    INSPECT_MAX_NEW_TOKENS,
    INSPECT_USER_TEXT,
    infer_batched_vlm,
    load_qwen_4bit_or_bf16,
    seed_all,
)


def parse_and_assert_strict_check_output(stdout_text: str, stderr_text: str, returncode: int, expected_n: int) -> dict[str, Any]:
    """Ensure official check_output.py reported exact `valid == total == expected_n` AND zero error lines."""
    if returncode != 0:
        raise RuntimeError(f"check_output.py failed (exit {returncode}): {stdout_text}\n{stderr_text}")
    m = re.search(r"^유효 결과\s+(\d+)/(\d+)\s*$", stdout_text, flags=re.MULTILINE)
    if m is None:
        raise RuntimeError(f"Missing '유효 결과 valid/total' line in check_output.py stdout: {stdout_text!r}")
    valid_n, total_n = int(m.group(1)), int(m.group(2))
    error_lines = [ln.strip() for ln in stdout_text.splitlines() if ln.startswith("  - ")]
    if valid_n != expected_n or total_n != expected_n or error_lines:
        raise RuntimeError(
            f"check_output.py did not pass {expected_n}/{expected_n} cleanly: valid={valid_n}/{total_n}, errors={error_lines}"
        )
    return {
        "return_code": returncode,
        "valid_rows": valid_n,
        "total_rows": total_n,
        "error_count": len(error_lines),
        "errors": error_lines,
        "stdout": stdout_text.strip(),
    }


def verify_fold4_cohort_and_manifests(root: Path, dev_manifest_path: Path) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Verify exact 175 Fold-4 IDs / 39 groups and agreement with full sealed manifest."""
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    eval_rows = [r for r in dev_manifest if r["split"] == "dev" and int(r["dev_fold"]) == 4]
    train_rows = [r for r in dev_manifest if r["split"] == "dev" and int(r["dev_fold"]) != 4]

    eval_ids = [r["sample_id"] for r in eval_rows]
    train_ids = {r["sample_id"] for r in train_rows}
    eval_groups = {r["group_id"] for r in eval_rows}
    train_groups = {r["group_id"] for r in train_rows}

    if len(eval_rows) != 175 or len(set(eval_ids)) != 175:
        raise ValueError(f"Expected 175 unique Fold-4 sample_ids, got {len(eval_rows)} ({len(set(eval_ids))} unique)")
    if len(eval_groups) != 39:
        raise ValueError(f"Expected 39 Fold-4 groups, got {len(eval_groups)}")
    if len(train_rows) != 791 or len(train_ids) != 791:
        raise ValueError(f"Expected 791 Fold 0-3 training photos, got {len(train_rows)}")
    if (set(eval_ids) & train_ids) or (eval_groups & train_groups):
        raise ValueError("Train/eval ID or group leakage detected in Fold 4!")

    full_path = root / "data/clean-split-v1/split_manifest_full_sealed.json"
    full_manifest = json.loads(full_path.read_text(encoding="utf-8"))
    full_eval_ids = [r["sample_id"] for r in full_manifest if r["split"] == "dev" and int(r["dev_fold"]) == 4]
    if eval_ids != full_eval_ids:
        raise ValueError("Fold-4 sample_id order differs between split_manifest_dev.json and split_manifest_full_sealed.json")

    cohort_audit = {
        "eval_photos": len(eval_rows),
        "eval_unique_ids": len(set(eval_ids)),
        "eval_groups": len(eval_groups),
        "train_photos": len(train_rows),
        "train_groups": len(train_groups),
        "eval_ids_order_sha256": hashlib.sha256(json.dumps(eval_ids).encode("utf-8")).hexdigest(),
        "dev_and_full_manifest_fold4_identical": True,
    }
    return eval_rows, train_rows, cohort_audit


def prepare_epoch1_candidate(
    root: Path,
    train_dir: Path,
    cand_dir: Path,
    dev_manifest_path: Path,
) -> dict[str, Any]:
    """Copy ONLY epoch-1 adapter weights into an isolated candidate folder while preserving exact 2-epoch schedule & resume state."""
    if cand_dir.exists() and (cand_dir / "training_summary.json").is_file():
        existing = json.loads((cand_dir / "training_summary.json").read_text(encoding="utf-8"))
        if existing.get("adapter_safetensors_sha256") == digest(cand_dir / "adapter/adapter_model.safetensors"):
            print("EPOCH1_CANDIDATE_REUSED " + json.dumps({"adapter_sha256": existing["adapter_safetensors_sha256"]}), flush=True)
            return existing
        raise FileExistsError(f"Candidate directory already exists with mismatched state: {cand_dir}")

    ep1_dir = train_dir / "epoch-1"
    state_path = ep1_dir / "training_state.pt"
    weights_path = ep1_dir / "adapter_model.safetensors"
    cfg_path = ep1_dir / "adapter_config.json"
    hist_path = train_dir / "history.json"
    for p in (ep1_dir, state_path, weights_path, cfg_path, hist_path):
        if not p.exists():
            raise FileNotFoundError(f"Epoch-1 artifact not ready yet: {p}")

    eval_rows, train_rows, cohort_audit = verify_fold4_cohort_and_manifests(root, dev_manifest_path)

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    history = json.loads(hist_path.read_text(encoding="utf-8"))
    if not history or history[0].get("epoch") != 1:
        raise ValueError(f"Unexpected history.json in {train_dir}: {history}")

    # Preserve original 2-epoch configuration verbatim (do NOT overwrite configuration["epochs"] to 1!)
    configuration = dict(state["configuration"])
    configured_epochs = int(configuration["epochs"])
    grad_accum = int(configuration["grad_accum"])
    exp_meta = configuration.get("experiment") or {}
    curriculum_n = int(exp_meta.get("curriculum_n", 0))
    if curriculum_n <= 0:
        raise ValueError(f"Missing curriculum_n in configuration: {configuration}")
    steps_per_epoch = (curriculum_n + grad_accum - 1) // grad_accum
    scheduled_total_steps = configured_epochs * steps_per_epoch

    global_step = int(state["global_step"])
    next_epoch = int(state["next_epoch"])
    next_offset = int(state["next_offset"])
    if (global_step != steps_per_epoch or next_epoch != 2 or next_offset != 0 or configured_epochs != 2):
        raise ValueError(
            f"Expected exact Epoch 1/2 boundary (step={steps_per_epoch}, next_epoch=2, next_offset=0), "
            f"got step={global_step}, next_epoch={next_epoch}, next_offset={next_offset}, configured_epochs={configured_epochs}"
        )

    cand_adapter = cand_dir / "adapter"
    cand_adapter.mkdir(parents=True, exist_ok=False)
    shutil.copy2(cfg_path, cand_adapter / "adapter_config.json")
    shutil.copy2(weights_path, cand_adapter / "adapter_model.safetensors")

    weights_sha = digest(cand_adapter / "adapter_model.safetensors")
    if weights_sha != digest(weights_path):
        raise RuntimeError("Copied epoch-1 adapter SHA-256 mismatch")

    records = state["records"]
    summary = {
        "version": "vlm-11code-multimodal-qlora-v2-epoch-boundary",
        "checkpoint_kind": "epoch_boundary_snapshot_of_multi_epoch_schedule",
        "configuration": configuration,
        "val_fold": int(configuration["val_fold"]),
        "train_photos": len(train_rows),
        "curriculum_examples": curriculum_n,
        "configured_epochs": configured_epochs,
        "completed_epochs": 1,
        "grad_accum": grad_accum,
        "lr": float(configuration["lr"]),
        "max_pixels": int(configuration["max_pixels"]),
        "trainable_params": 33980416,
        "optimizer_steps": global_step,
        "scheduled_total_steps": scheduled_total_steps,
        "resume_cursor": {
            "next_epoch": next_epoch,
            "next_offset": next_offset,
            "global_step": global_step,
        },
        "total_seconds": round(float(state.get("elapsed", history[0].get("epoch_seconds", 0.0))), 1),
        "history": [history[0]],
        "source_epoch1_checkpoint": str(ep1_dir.resolve()),
        "source_training_state_preserved": True,
        "source_training_state_bytes": state_path.stat().st_size,
        "final_adapter": str(cand_adapter.resolve()),
        "adapter_safetensors_sha256": weights_sha,
        "epoch1_last_step_record": records[-1] if records else None,
        "fold4_cohort_audit": cohort_audit,
    }
    save_atomic(
        cand_dir / "training_summary.json",
        lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")),
    )

    # Materialize canonical-order clean_8code_baseline.jsonl from existing raw8_generations_cache.json (CPU-only, no re-inference!)
    local_eval_dir = root / "data/local-eval-predictions/fold4-v3-benchmark"
    raw8_cache = local_eval_dir / "raw8_generations_cache.json"
    out_8code = local_eval_dir / "clean_8code_baseline.jsonl"
    if raw8_cache.is_file() and not out_8code.is_file():
        bh = _load_baseline_helpers(root)
        raw_train_by_id = {
            r["sample_id"]: r
            for r in (
                json.loads(l, object_pairs_hook=unique_keys)
                for l in (root / "data/train/train.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()
            )
        }
        laws = [json.loads(l) for l in (root / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        cached8 = json.loads(raw8_cache.read_text(encoding="utf-8"))
        valid8 = []
        for r, o in zip(eval_rows, cached8["raw8_outs"], strict=True):
            obj8, err8 = bh["parse_8_pred"](o["raw_text"])
            if err8 is None and not bh["validate_8_pred"](obj8):
                valid8.append({"sample_id": r["sample_id"], "prediction": obj8})
        train8_cases = [c for r in train_rows if (c := bh["convert_gold_to_8"](raw_train_by_id[r["sample_id"]])) is not None]
        tmp8 = local_eval_dir / "tmp_8code_canonical.jsonl"
        bh["format_8_sub"](valid8, laws, train8_cases, tmp8)
        fmt_map = {json.loads(l)["sample_id"]: json.loads(l)["prediction"] for l in tmp8.read_text(encoding="utf-8").splitlines() if l.strip()}
        comp8_recs = [bh["convert_8_to_comp"](r["sample_id"], fmt_map.get(r["sample_id"])) for r in eval_rows]
        save_atomic(
            out_8code,
            lambda f: f.write(("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in comp8_recs)).encode("utf-8")),
        )

    print(
        "EPOCH1_CANDIDATE_PREPARED "
        + json.dumps({"adapter_sha256": weights_sha, "step": f"{global_step}/{scheduled_total_steps}", "resume_cursor": summary["resume_cursor"]}),
        flush=True,
    )
    return summary


def package_and_audit_vlmonly(
    root: Path,
    cand_dir: Path,
    stage_dir: Path,
    zip_path: Path,
    audit_out: Path,
    batch_size: int = 4,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Build --vlm-only ZIP on CPU without rmtree/unlink and run lightweight stage audit (no 9 GB extraction/CRC)."""
    expected_adapter_sha = digest(cand_dir / "adapter/adapter_model.safetensors")

    if stage_dir.exists() or zip_path.exists():
        if not reuse_existing:
            raise FileExistsError(
                f"Refusing to overwrite existing stage_dir ({stage_dir}) or zip_path ({zip_path}); "
                "pass a fresh path or --reuse-existing after verifying."
            )
        if not (stage_dir.is_dir() and zip_path.is_file()):
            raise FileExistsError("Partial existing stage_dir/zip_path cannot be reused; pass fresh paths.")
        staged_cfg = json.loads((stage_dir / "pipeline_config.json").read_text(encoding="utf-8"))
        if (digest(stage_dir / "adapter/adapter_model.safetensors") != expected_adapter_sha
            or staged_cfg.get("batch_size") != batch_size
            or staged_cfg.get("enable_hybrid_booster") is not False):
            raise ValueError("Existing stage_dir does not match expected Epoch-1 adapter or batch_size")
        pkg_info = {
            "dest_dir": str(stage_dir),
            "zip_path": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "zip_gib": round(zip_path.stat().st_size / 2**30, 3),
            "batch_size": batch_size,
            "reused_existing": True,
        }
    else:
        pkg_info = build_submission_package(
            root=root,
            adapter_dir=cand_dir / "adapter",
            specialist_ckpt=root / "data/clean-specialist-v1/specialist_5fold_ensemble.pt",
            dest_dir=stage_dir,
            zip_path=zip_path,
            enable_hybrid=False,
            batch_size=batch_size,
            prompt_path=root / "kwater_lab/system_prompt_v3.txt",
        )
        pkg_info["reused_existing"] = False

    # Lightweight stage audit (verifies manifest seed=42, lock=True, Linux cp312 wheel hashes, AST static calls, adapter SHA)
    # without extracting or re-reading 9 GB of base_model shards!
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "kwater_lab.offline_archive_audit",
            "--stage",
            str(stage_dir),
            "--out",
            str(audit_out),
        ],
        check=True,
    )
    pkg_info["offline_stage_audit_path"] = str(audit_out)
    print("PACKAGE_VLMONLY_STAGE_AUDIT_OK " + json.dumps({"zip_gib": pkg_info["zip_gib"], "batch_size": batch_size}), flush=True)
    return pkg_info


@torch.inference_mode()
def infer_and_verify_epoch1(
    root: Path,
    cand_dir: Path,
    dev_manifest_path: Path,
    prompt_path: Path,
    stage_dir: Path,
    sample_out_dir: Path,
    benchmark_out: Path,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Single BF16 model load -> strict 10/10 kit sample check + 175/175 Fold-4 raw/formatted save + cache ID & parity verification."""
    seed_all(42)
    eval_rows, train_rows, cohort_audit = verify_fold4_cohort_and_manifests(root, dev_manifest_path)
    eval_ids = [r["sample_id"] for r in eval_rows]
    train_ids = {r["sample_id"] for r in train_rows}

    base_dir = root / "data/baseline/base_model"
    adapter_dir = cand_dir / "adapter"
    adapter_sha = digest(adapter_dir / "adapter_model.safetensors")
    prompt_sha = digest(prompt_path)
    prompt_text = prompt_path.read_text(encoding="utf-8")

    t_load0 = time.perf_counter()
    model, proc = load_qwen_4bit_or_bf16(
        base_dir,
        max_pixels=TRAIN_EVAL_MAX_PIXELS,
        use_4bit=False,
        adapter_dir=adapter_dir,
    )
    model_load_seconds = round(time.perf_counter() - t_load0, 2)
    print(f"MODEL_LOADED_BF16 seconds={model_load_seconds} adapter_sha256={adapter_sha}", flush=True)

    msgs = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSPECT_USER_TEXT}]},
    ]
    prefix_str = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefix_sha = hashlib.sha256(prefix_str.encode("utf-8")).hexdigest()

    laws = [json.loads(l) for l in (root / "data/baseline/assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    train_case_rows = list(read_selected_cases(root / "data/train/train.jsonl", train_ids))
    compact_cases_tmp = cand_dir / "cases_11code_fold4_train.jsonl"
    export_compact_case_bank(train_case_rows, compact_cases_tmp)
    compact_cases = [json.loads(l) for l in compact_cases_tmp.read_text(encoding="utf-8").splitlines() if l.strip()]
    retriever = ActionSupportedLawRetriever(laws, compact_cases)

    # 1. Decode & strict 10/10 check_output on official `data/local-kit/sample-input`
    sample_in_dir = root / "data/local-kit/sample-input"
    s_manifest = [json.loads(l) for l in (sample_in_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(s_manifest) != 10:
        raise ValueError(f"Expected 10 sample-input rows, got {len(s_manifest)}")
    img_by_stem = {p.stem: p.name for p in (sample_in_dir / "images").iterdir() if p.is_file()}
    sample_rows = [
        {"sample_id": r["image_id"], "image": r.get("image") or f"images/{img_by_stem[r['image_id']]}"}
        for r in s_manifest
    ]
    sample_outs, sample_gen_stats = infer_batched_vlm(
        model, proc, sample_rows, sample_in_dir, prompt_text, batch_size=batch_size, max_new_tokens=INSPECT_MAX_NEW_TOKENS
    )
    sample_gen_stats["backoff_occurred"] = sample_gen_stats["initial_batch_size"] != sample_gen_stats["final_effective_batch_size"]

    if sample_out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample_out_dir: {sample_out_dir}")
    sample_out_dir.mkdir(parents=True, exist_ok=False)
    sample_sub_lines = [
        json.dumps(format_v3_record(r["sample_id"], o["parsed"], retriever), ensure_ascii=False, separators=(",", ":")) + "\n"
        for r, o in zip(sample_rows, sample_outs, strict=True)
    ]
    (sample_out_dir / "submission.jsonl").write_bytes("".join(sample_sub_lines).encode("utf-8"))
    chk_s_proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(root / "data/local-kit/kit/check_output.py"), str(sample_in_dir), str(sample_out_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    sample_chk_info = parse_and_assert_strict_check_output(chk_s_proc.stdout, chk_s_proc.stderr, chk_s_proc.returncode, expected_n=10)
    print(f"SAMPLE_10_STRICT_OK | {sample_chk_info['valid_rows']}/10 errors=0 | stats={json.dumps(sample_gen_stats)}", flush=True)

    # 2. Decode unseen Fold-4 175 photos with per-batch atomic cache + immediate separate raw JSONL save!
    local_eval_dir = root / "data/local-eval-predictions/fold4-v3-benchmark"
    local_eval_dir.mkdir(parents=True, exist_ok=True)
    infer_cache_path = local_eval_dir / "v3_epoch1_infer_cache.json"
    cache_key = f"bf16:{adapter_sha}:{prompt_sha}:px{TRAIN_EVAL_MAX_PIXELS}"

    raw_v3_outs, gen_v3_stats = infer_batched_vlm(
        model,
        proc,
        eval_rows,
        root / "data/train",
        prompt_text,
        batch_size=batch_size,
        max_new_tokens=INSPECT_MAX_NEW_TOKENS,
        cache_path=infer_cache_path,
        cache_key=cache_key,
    )
    gen_v3_stats["backoff_occurred"] = gen_v3_stats["initial_batch_size"] != gen_v3_stats["final_effective_batch_size"]
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()

    # Reload batch cache and verify original 175 canonical IDs & order!
    disk_cache = json.loads(infer_cache_path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    cache_sample_ids = [pair[0] for pair in disk_cache["identity"]["samples"]]
    cache_output_ids = list(disk_cache["outputs"].keys())
    if cache_sample_ids != eval_ids or cache_output_ids != eval_ids:
        raise RuntimeError("Batch cache sample IDs do not match canonical 175 Fold-4 IDs!")

    # Save raw_text + sample_id JSONL immediately!
    raw_jsonl_path = local_eval_dir / "v3_epoch1_raw_175.jsonl"
    raw_lines = [
        json.dumps(
            {
                "sample_id": r["sample_id"],
                "image": r["image"],
                "raw_text": o["raw_text"],
                "parsed": o["parsed"],
                "parse_error": o["parse_error"],
                "vlm_schema_errors": o["vlm_schema_errors"],
                "token_cap_hit": o["token_cap_hit"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for r, o in zip(eval_rows, raw_v3_outs, strict=True)
    ]
    save_atomic(raw_jsonl_path, lambda f: f.write("".join(raw_lines).encode("utf-8")))

    # Save canonical local-ID formatted 175 rows AND 32-hex competition formatted 175 rows!
    local175_path = local_eval_dir / "v3_epoch1_11code_standalone.jsonl"
    hex175_path = local_eval_dir / "v3_epoch1_11code_standalone_hex175.jsonl"
    comp_local_recs = [format_v3_record(r["sample_id"], o["parsed"], retriever) for r, o in zip(eval_rows, raw_v3_outs, strict=True)]
    comp_hex_recs = [format_v3_record(anon_hex_id(r["sample_id"]), o["parsed"], retriever) for r, o in zip(eval_rows, raw_v3_outs, strict=True)]

    save_atomic(
        local175_path,
        lambda f: f.write(("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in comp_local_recs)).encode("utf-8")),
    )
    save_atomic(
        hex175_path,
        lambda f: f.write(("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in comp_hex_recs)).encode("utf-8")),
    )

    # Strict 175/175 error==0 check via official check_output.py!
    scratch = root / ".runtime/temp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as td:
        td_p = Path(td)
        (td_p / "in").mkdir()
        (td_p / "out").mkdir()
        (td_p / "in/manifest.jsonl").write_text(
            "".join(json.dumps({"image_id": anon_hex_id(r["sample_id"])}) + "\n" for r in eval_rows),
            encoding="utf-8",
        )
        # Write fresh regular file (st_nlink == 1 required by check_output.py!)
        (td_p / "out/submission.jsonl").write_bytes(hex175_path.read_bytes())
        chk_175_proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(root / "data/local-kit/kit/check_output.py"), str(td_p / "in"), str(td_p / "out")],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        fold4_chk_info = parse_and_assert_strict_check_output(
            chk_175_proc.stdout, chk_175_proc.stderr, chk_175_proc.returncode, expected_n=175
        )

    # Verify post-processing parity against staged `main.py` if stage_dir is ready!
    parity_ok = False
    if (stage_dir / "main.py").is_file() and (stage_dir / "assets/cases_11code.jsonl").is_file():
        ns = runpy.run_path(str(stage_dir / "main.py"))
        st_laws = [json.loads(l) for l in (stage_dir / "assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        st_cases = [json.loads(l) for l in (stage_dir / "assets/cases_11code.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        st_ret = ns["ActionSupportedLawRetriever"](st_laws, st_cases)
        replayed_hex = []
        for r, o in zip(eval_rows, raw_v3_outs, strict=True):
            p_obj, _ = ns["parse_json_loose"](o["raw_text"])
            replayed_hex.append(ns["format_v3_record"](anon_hex_id(r["sample_id"]), p_obj, st_ret))
        parity_ok = replayed_hex == comp_hex_recs
        if not parity_ok:
            raise RuntimeError("Staged main.py post-processing parity check failed against hex175!")

    # Compute primary_175 metrics vs clean_8code_baseline
    raw_train_by_id = {
        r["sample_id"]: r
        for r in (
            json.loads(l, object_pairs_hook=unique_keys)
            for l in (root / "data/train/train.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    gold_codes = {r["sample_id"]: frozenset(r["codes"]) for r in eval_rows}
    group_map = {r["sample_id"]: r["group_id"] for r in eval_rows}
    preds_v3_ep1 = {rec["image_id"]: decode_target(rec, local_ids=True)[1] for rec in comp_local_recs}
    eval_v3_ep1 = evaluate(gold_codes, preds_v3_ep1, group_map)
    sev4_v3_ep1 = severity4_miss_stats({k: raw_train_by_id[k] for k in gold_codes}, preds_v3_ep1)

    out_8code = local_eval_dir / "clean_8code_baseline.jsonl"
    preds_8code = {
        json.loads(l)["image_id"]: decode_target(json.loads(l), local_ids=True)[1]
        for l in out_8code.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    eval_8code = evaluate(gold_codes, preds_8code, group_map)
    sev4_8code = severity4_miss_stats({k: raw_train_by_id[k] for k in gold_codes}, preds_8code)

    cand_summary = json.loads((cand_dir / "training_summary.json").read_text(encoding="utf-8"))
    summary = {
        "protocol": "Fold-4 Unseen Standalone Comparison: Clean 8-Code (2ep) vs SafeGround v3 11-Code Standalone (Epoch 1/2 boundary, 69/138 steps, --vlm-only)",
        "checkpoint_schedule_provenance": {
            "checkpoint_kind": cand_summary["checkpoint_kind"],
            "configured_epochs": cand_summary["configured_epochs"],
            "completed_epochs": cand_summary["completed_epochs"],
            "optimizer_steps": cand_summary["optimizer_steps"],
            "scheduled_total_steps": cand_summary["scheduled_total_steps"],
            "resume_cursor": cand_summary["resume_cursor"],
        },
        "cohort_and_cache_id_audit": {
            **cohort_audit,
            "infer_cache_identity_samples_verified_175": True,
            "infer_cache_outputs_keys_verified_175": True,
            "infer_cache_effective_batch_size": disk_cache["effective_batch_size"],
        },
        "provenance_sha256": {
            "dev_manifest_sha256": digest(dev_manifest_path),
            "full_sealed_manifest_sha256": digest(root / "data/clean-split-v1/split_manifest_full_sealed.json"),
            "base_config_sha256": digest(base_dir / "config.json"),
            "adapter_safetensors_sha256": adapter_sha,
            "adapter_config_sha256": digest(adapter_dir / "adapter_config.json"),
            "prompt_path": "kwater_lab/system_prompt_v3.txt",
            "prompt_sha256": prompt_sha,
            "chat_prefix_sha256": prefix_sha,
            "infer_cache_sha256": digest(infer_cache_path),
            "raw_175_jsonl_sha256": digest(raw_jsonl_path),
            "formatted_local175_jsonl_sha256": digest(local175_path),
            "formatted_hex175_jsonl_sha256": digest(hex175_path),
            "clean_8code_local175_jsonl_sha256": digest(out_8code),
        },
        "decoding_conditions": {
            "precision": "torch.bfloat16",
            "use_4bit": False,
            "attn_implementation": "sdpa",
            "min_pixels": 128 * 32 * 32,
            "max_pixels": TRAIN_EVAL_MAX_PIXELS,
            "max_new_tokens": INSPECT_MAX_NEW_TOKENS,
            "do_sample": False,
            "requested_batch_size": batch_size,
            "final_effective_batch_size": gen_v3_stats["final_effective_batch_size"],
            "backoff_occurred": gen_v3_stats["backoff_occurred"],
            "model_load_seconds": model_load_seconds,
        },
        "kit_sample_10_verification": {
            "generation": sample_gen_stats,
            "check_output": sample_chk_info,
            "output_sha256": digest(sample_out_dir / "submission.jsonl"),
        },
        "fold4_175_kit_checker": fold4_chk_info,
        "staged_main_py_exact_175_parity": parity_ok,
        "arms": {
            "clean_8code_baseline_2ep": {
                "metrics": eval_8code,
                "severity4": sev4_8code,
            },
            "v3_epoch1_11code_standalone": {
                "generation": gen_v3_stats,
                "metrics": eval_v3_ep1,
                "severity4": sev4_v3_ep1,
            },
        },
        "paired_bootstrap_v3_epoch1_minus_clean_8code": paired_bootstrap(
            gold_codes, preds_8code, preds_v3_ep1, group_map, resamples=2000, seed=42
        ),
    }
    benchmark_out.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(benchmark_out, lambda f: f.write((json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    print(
        f"EPOCH1_FOLD4_DONE | Clean8 F1={eval_8code['macro_f1']:.4f} (FA={eval_8code['no_hazard_false_alarms']}/49) "
        f"| v3-Ep1 F1={eval_v3_ep1['macro_f1']:.4f} (FA={eval_v3_ep1['no_hazard_false_alarms']}/49) "
        f"| 175/175 errors=0 | backoff={gen_v3_stats['backoff_occurred']} | main.py parity={parity_ok}",
        flush=True,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", choices=["prepare", "package", "infer", "verify-parity"], required=True)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--train-dir", type=Path, default=Path("data/v3-11code-fold4"))
    ap.add_argument("--cand-dir", type=Path, default=Path("data/v3-11code-fold4-epoch1-standalone"))
    ap.add_argument("--dev-manifest", type=Path, default=Path("data/clean-split-v1/split_manifest_dev.json"))
    ap.add_argument("--prompt", type=Path, default=Path("kwater_lab/system_prompt_v3.txt"))
    ap.add_argument("--stage-dir", type=Path, default=Path("data/submissions/safeground_v3_epoch1_vlm_only_dir"))
    ap.add_argument("--zip-path", type=Path, default=Path("data/submissions/kwater-safeground-v3-epoch1-vlmonly-20261127.zip"))
    ap.add_argument("--audit-out", type=Path, default=Path("results/v3_epoch1_vlmonly_stage_audit.json"))
    ap.add_argument("--sample-out", type=Path, default=Path("data/submissions/verify_v3_epoch1_sample_out"))
    ap.add_argument("--benchmark-out", type=Path, default=Path("results/fold4_v3_epoch1_standalone_benchmark.json"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    if args.step == "prepare":
        prepare_epoch1_candidate(root, args.train_dir.resolve(), args.cand_dir.resolve(), args.dev_manifest.resolve())
    elif args.step == "package":
        package_and_audit_vlmonly(
            root,
            args.cand_dir.resolve(),
            args.stage_dir.resolve(),
            args.zip_path.resolve(),
            args.audit_out.resolve(),
            batch_size=args.batch_size,
            reuse_existing=args.reuse_existing,
        )
    elif args.step == "infer":
        infer_and_verify_epoch1(
            root,
            args.cand_dir.resolve(),
            args.dev_manifest.resolve(),
            args.prompt.resolve(),
            args.stage_dir.resolve(),
            args.sample_out.resolve(),
            args.benchmark_out.resolve(),
            batch_size=args.batch_size,
        )
    elif args.step == "verify-parity":
        stage_dir = args.stage_dir.resolve()
        local_eval_dir = root / "data/local-eval-predictions/fold4-v3-benchmark"
        raw_rows = [json.loads(l) for l in (local_eval_dir / "v3_epoch1_raw_175.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        hex_rows = [json.loads(l) for l in (local_eval_dir / "v3_epoch1_11code_standalone_hex175.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        ns = runpy.run_path(str(stage_dir / "main.py"))
        st_laws = [json.loads(l) for l in (stage_dir / "assets/laws.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        st_cases = [json.loads(l) for l in (stage_dir / "assets/cases_11code.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        st_ret = ns["ActionSupportedLawRetriever"](st_laws, st_cases)
        replayed = []
        for r in raw_rows:
            p_obj, _ = ns["parse_json_loose"](r["raw_text"])
            replayed.append(ns["format_v3_record"](anon_hex_id(r["sample_id"]), p_obj, st_ret))
        if replayed != hex_rows:
            raise RuntimeError("Staged main.py post-processing parity mismatch!")
        print(f"PARITY_VERIFIED_OK rows={len(replayed)}", flush=True)


if __name__ == "__main__":
    main()
