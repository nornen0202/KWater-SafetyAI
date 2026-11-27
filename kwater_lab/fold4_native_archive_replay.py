"""Replay real Fold 4 generations through a packaged standalone native 11-code entrypoint."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from .fold4_official_output_check import DEFAULT_MANIFEST, EXPECTED_ROWS, sha256_file, write_jsonl
from .metrics import read_index
from .offline_archive_audit import COMPETITION_LIMIT, entries, inspect_bundle, run_output_checker


def raw_index(path):
    if path.suffix == ".jsonl":
        return read_index(path, lambda row: (row["sample_id"], row["raw_text"]), limit_size=True), None
    records = json.loads(path.read_bytes())
    identity = None
    if isinstance(records, dict) and "raw_v3_outs" in records:
        records = records["raw_v3_outs"]
    elif isinstance(records, dict) and "outputs" in records:
        identity = records["identity"]
        if any(key != row["sample_id"] for key, row in records["outputs"].items()):
            raise ValueError("Raw cache IDs do not match their indexed records")
        records = list(records["outputs"].values())
    if not isinstance(records, list):
        raise ValueError("Expected a raw JSONL, list, or object with raw_v3_outs")
    index = {r["sample_id"]: r["raw_text"] for r in records}
    if len(index) != len(records):
        raise ValueError("Duplicate real raw generation IDs")
    return index, identity


class ReplayBatch(dict):
    def to(self, device):
        return self


class ReplayModel:
    def generate(self, *, input_ids, do_sample, max_new_tokens, use_cache):
        if do_sample or max_new_tokens != 620 or not use_cache:
            raise ValueError("Packaged generation conditions differ from the frozen replay")
        return self

    def __getitem__(self, key):
        return key[0]


class ReplayProcessor:
    def __init__(self, texts, batch_size, prompt):
        self.texts = iter(texts)
        self.batch_size = batch_size
        self.prompt = prompt
        self.decoded = 0
        self.observed_batches = []

    def apply_chat_template(self, messages, **options):
        if messages[0]["content"] != self.prompt:
            raise ValueError("Packaged prompt differs from the frozen model prompt")
        return "CPU generation replay"

    def __call__(self, *, text, images, padding, return_tensors):
        if len(text) != len(images) or len(images) > self.batch_size:
            raise ValueError("Packaged visual generation batch has unexpected size")
        self.observed_batches.append(len(images))
        return ReplayBatch(input_ids=SimpleNamespace(shape=(len(images), 1)))

    def decode(self, *args, **kwargs):
        self.decoded += 1
        return next(self.texts)


def load_packaged_entrypoint(path):
    torch, nn, functional = (ModuleType(name) for name in ("torch", "torch.nn", "torch.nn.functional"))
    @contextmanager
    def inference_context():
        yield
    torch.inference_mode = inference_context
    torch.no_grad = inference_context
    torch.device = str
    torch.nn = nn
    nn.Module = object
    nn.functional = functional
    spec = importlib.util.spec_from_file_location("_fold4_packaged_v3", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import the packaged entrypoint")
    module = importlib.util.module_from_spec(spec)
    modules = {item.__name__: item for item in (torch, nn, functional, module)}
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    module.seed_all = lambda seed: None
    return module


def compact_case_witness(manifest, labels):
    training = {r["sample_id"] for r in manifest if r["split"] == "dev" and r["dev_fold"] != 4}
    cases = []
    with labels.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sample_id"] not in training or row["target"]["status"] != "hazard":
                continue
            scene = str((row.get("notes") or {}).get("scene_note") or "")
            for code, hazard in row["target"]["hazards"].items():
                law_ids = [r["id"] for r in hazard.get("references", ()) if r.get("kind") == "law"]
                if law_ids:
                    cases.append({"code": code, "osh_ids": law_ids, "evidence": hazard["evidence"],
                                  "prevention": hazard["prevention"], "scene_note": scene})
    if len(training) != 791 or not cases:
        raise ValueError("Expected compact statutory cases from exactly 791 eligible training images")
    return cases


def check(args):
    if args.archive.stat().st_size > COMPETITION_LIMIT:
        raise ValueError("Candidate exceeds the 20 GB archive limit")
    manifest = json.loads(args.manifest.read_bytes())
    rows = [r for r in manifest if r["split"] == "dev" and r["dev_fold"] == 4]
    ids = [r["sample_id"] for r in rows]
    raw, raw_identity = raw_index(args.raw)
    reference = read_index(args.reference, lambda row: (row["image_id"], row), limit_size=True)
    ids_set = set(ids)
    if (len(ids) != EXPECTED_ROWS or len(ids_set) != EXPECTED_ROWS
            or set(raw) != ids_set or reference.keys() != ids_set):
        raise ValueError("Real raw and reference predictions must both cover the exact 175 original IDs")
    if not all(isinstance(txt, str) for txt in raw.values()):
        raise ValueError("Every persisted generation must include its raw text")
    hashes = {key: sha256_file(getattr(args, key)) for key in ("raw", "reference", "manifest", "adapter", "prompt")}
    witness = json.loads(args.benchmark_witness.read_bytes())
    provenance = witness["provenance_sha256"]
    measured = witness["decoding_conditions"]
    identity_fields = {
        "manifest": "dev_manifest_sha256", "adapter": "adapter_safetensors_sha256",
        "prompt": "prompt_sha256", "reference": "formatted_local175_jsonl_sha256",
        "raw": "infer_cache_sha256" if raw_identity is not None else "raw_175_jsonl_sha256",
    }
    cohort = witness["cohort_and_cache_id_audit"]
    schedule = witness["checkpoint_schedule_provenance"]
    if (schedule["configured_epochs"], schedule["completed_epochs"], schedule["optimizer_steps"],
            schedule["scheduled_total_steps"], schedule["resume_cursor"]) != (
            2, 1, 69, 138, {"next_epoch": 2, "next_offset": 0, "global_step": 69}):
        raise ValueError("Expected the actual 69/138 epoch-boundary checkpoint witness")
    if (cohort["eval_photos"] != EXPECTED_ROWS or cohort["eval_unique_ids"] != EXPECTED_ROWS or not all(
            hashes[key] == provenance[field] for key, field in identity_fields.items())):
        raise ValueError("Epoch-1 generation witness does not identify these actual files")
    expected_conditions = {
        "precision": "torch.bfloat16", "use_4bit": False, "attn_implementation": "sdpa",
        "min_pixels": 128 * 32 * 32, "max_pixels": 384 * 32 * 32, "max_new_tokens": 620,
        "do_sample": False, "requested_batch_size": args.batch_size,
        "final_effective_batch_size": args.batch_size,
    }
    if any(measured[key] != value for key, value in expected_conditions.items()):
        raise ValueError("Measured v3 decoding conditions differ from the package replay")
    sample_check = witness["kit_sample_10_verification"]["check_output"]
    if (sample_check["return_code"], sample_check["valid_rows"], sample_check["total_rows"],
            sample_check["error_count"]) != (0, 10, 10, 0):
        raise ValueError("Epoch-1 GPU-generated official sample output did not pass 10/10")
    hashes["benchmark_witness"] = sha256_file(args.benchmark_witness)
    if raw_identity is not None:
        image_hashes = {r["image"]: r["sha256"] for r in rows}
        if (raw_identity["samples"] != [[r["sample_id"], r["image"]] for r in rows]
                or raw_identity["image_sha256"] != image_hashes
                or raw_identity["prefix_sha256"] != provenance["chat_prefix_sha256"]
                or raw_identity["batch_size"] != args.batch_size
                or raw_identity["max_new_tokens"] != 620
                or raw_identity["processor_size"]["longest_edge"] != 384 * 32 * 32):
            raise ValueError("Native raw cache identity differs from the official Fold 4 replay configuration")
    args.workspace.mkdir(parents=True, exist_ok=False)
    code, inputs, outputs = (args.workspace / name for name in ("code", "input", "output"))
    with ZipFile(args.archive) as bundle:
        info = inspect_bundle(entries(bundle), bundle.open, args.kit, args.workspace,
                              {"adapter/adapter_model.safetensors": hashes["adapter"]})
        if info["restricted_or_unneeded_members"]:
            raise ValueError("Native candidate contains restricted or unneeded members")
        cfg = json.loads(bundle.read("pipeline_config.json"))
        if hashlib.sha256(bundle.read("base_model/config.json")).hexdigest() != provenance["base_config_sha256"]:
            raise ValueError("Packaged base configuration does not match the real-generation witness")
        packed_cases = [json.loads(line) for line in bundle.read("assets/cases_11code.jsonl").splitlines() if line]
        expected_cases = compact_case_witness(manifest, args.labels)
        canonical = lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False)
        if Counter(map(canonical, packed_cases)) != Counter(map(canonical, expected_cases)):
            raise ValueError("Bundled cases do not match the train-only Fold 4 legal case bank")
        prompt_name = cfg["prompt_filename"]
        if len(PurePosixPath(prompt_name).parts) != 1:
            raise ValueError("Expected a top-level packaged prompt")
        if (cfg["enable_hybrid_booster"] or not cfg["use_v3_formatter"] or cfg["batch_size"] != args.batch_size
                or cfg["max_pixels_infer"] != 384 * 32 * 32 or bundle.read(prompt_name) != args.prompt.read_bytes()):
            raise ValueError("Standalone package inference configuration or prompt differs from the Fold 4 run")
        for name in ("main.py", "pipeline_config.json", prompt_name, "assets/laws.jsonl", "assets/cases_11code.jsonl"):
            bundle.extract(name, code)
    module = load_packaged_entrypoint(code / "main.py")
    mapping = {sid: hashlib.sha256(f"kwater/fold4-native-parity/{sid}".encode()).hexdigest()[:32] for sid in ids}
    if len(set(mapping.values())) != EXPECTED_ROWS:
        raise ValueError("Package checker ID collision")
    (inputs / "images").mkdir(parents=True)
    write_jsonl(inputs / "manifest.jsonl", ({"image_id": mapping[sid]} for sid in ids))
    for image_id in mapping.values():
        (inputs / "images" / f"{image_id}.png").touch()
    proc = ReplayProcessor((raw[sid] for sid in ids), args.batch_size, args.prompt.read_text(encoding="utf-8"))
    module.load_vlm = lambda *a, **kw: (ReplayModel(), proc)
    # Generation text comes from the real raw cache. The model and pixels are bypassed in this CPU replay.
    with patch.object(module.Image, "open", lambda path: module.Image.new("RGB", (1, 1))):
        with patch.object(sys, "argv", ["main.py", "--input-dir", str(inputs), "--output-dir", str(outputs)]):
            with patch.dict(module.os.environ, {"KW_BATCH": str(args.batch_size)}):
                module.main()
    replay_path = outputs / "submission.jsonl"
    replay = read_index(replay_path, lambda row: (row["image_id"], row), limit_size=True)
    mismatches = sum(replay.get(mapping[sid]) != {**reference[sid], "image_id": mapping[sid]} for sid in ids)
    outcome = run_output_checker(args.kit, inputs, outputs, args.checker_python)
    complete_batches, remainder = divmod(EXPECTED_ROWS, args.batch_size)
    expected_batches = [args.batch_size] * complete_batches + ([remainder] if remainder else [])
    passed = (outcome["all_rows_valid"] and outcome["valid_rows"] == EXPECTED_ROWS
              and mismatches == 0 and proc.decoded == EXPECTED_ROWS and proc.observed_batches == expected_batches)
    source_observations = None
    if args.source_observations is not None:
        events = [json.loads(line) for line in args.source_observations.read_bytes().splitlines() if line]
        if not events or not any(e["epoch1_checkpoint_present"] for e in events):
            raise ValueError("No epoch-boundary source observation is available")
        source_observations = {
            "log_sha256": sha256_file(args.source_observations),
            "observed_phases": [e["phase"] for e in events],
            "identical_source_digests_at_every_observed_phase": all(
                e["source_files"] == events[0]["source_files"] for e in events
            ),
            "last_observed_source_sha256": events[-1]["source_files"],
            "scope": "On-disk source observations; Python process-loaded bytes are not independently attested",
        }
    return {"scope": "Real generation CPU replay through archive main; GPU and Linux sandbox not executed",
            "source_observations": source_observations, "gpu_generated_kit_sample_rows": sample_check["valid_rows"],
            "passed": passed, "reference_mismatched_rows": mismatches, "real_generations_replayed": proc.decoded,
            "observed_generation_batches": proc.observed_batches, "official_output_check": outcome, "input_sha256": hashes, "runtime_config": cfg,
            "raw_cache_identity_verified": raw_identity is not None,
            "archive_bytes": args.archive.stat().st_size, "archive_main_sha256": info["asset_sha256"]["main.py"],
            "offline_wheels": len(info["offline_wheels"]), "train_only_cases": len(expected_cases), "hashed_replayed_result": sha256_file(replay_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "raw", "reference", "adapter", "prompt", "benchmark-witness", "workspace", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-observations", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("data/train/train.jsonl"))
    parser.add_argument("--kit", type=Path, default=Path("data/local-kit/kit"))
    parser.add_argument("--checker-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if (args.batch_size < 1 or not args.workspace.resolve().is_relative_to(Path("data").resolve()) or args.workspace.exists()
            or args.out.exists()):
        parser.error("Use a fresh local data workspace and a new summary output")
    report = check(args)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
