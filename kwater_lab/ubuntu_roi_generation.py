"""Preregister, generate, and score a local official-format ROI anchor.

Run sequentially: .venv/bin/python -m kwater_lab.ubuntu_roi_generation prepare
Then predict [--max-calls N], then score. Prediction never reads ground truth.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageOps

from .fold0_official_resume import write_atomic
from .metrics import CODES, evaluate, load_predictions
from .run_official_baseline import check_packaged_code, sha256
from .ubuntu_roi_views import Photo

ROOT: Final = Path(__file__).resolve().parents[1]
WORK: Final = ROOT / "data/ubuntu-roi-20261127"
RUN: Final = WORK / "generation-anchor"
BASE: Final = ROOT / "data/baseline"
ARMS: Final = ("native_full", "fixed", "box")
MAX_TOKENS: Final = 900


@dataclass(frozen=True, slots=True)
class AnchorError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def select_cohort(rows: list[Photo]) -> list[Photo]:
    """Take first two in frozen input order per prefix without reading labels."""
    counts: Counter[str] = Counter()
    selected = []
    for row in rows:
        if counts[row.prefix] < 2:
            selected.append(row)
            counts[row.prefix] += 1
    if len(selected) != 12 or len(counts) != 6 or set(counts.values()) != {2}:
        raise AnchorError("Anchor requires exactly six prefixes with two photos each")
    return selected


def prepare() -> None:
    """Freeze sources, pixels, official prompt and model bytes before generation."""
    rows = select_cohort(
        [Photo(**row) for row in json.loads((WORK / "cohort.json").read_text())["rows"]]
    )
    views = {
        row["sample_id"]: row
        for row in json.loads((WORK / "views/index.json").read_text())["rows"]
    }
    RUN.mkdir(exist_ok=True)
    plan_path = RUN / "plan.json"
    if plan_path.exists():
        print(f"Existing frozen plan retained: {plan_path}", flush=True)
        return
    checked = check_packaged_code(ROOT / "data/packages/kwater-baseline.zip", BASE)
    if not checked:
        raise AnchorError("Official source archive is required")
    inputs = RUN / "inputs"
    (inputs / "images").mkdir(parents=True, exist_ok=True)
    planned = []
    for photo in rows:
        source = ROOT / "data/train" / photo.image
        if (
            sha256(source) != photo.sha256
            or views[photo.sample_id]["image_sha256"] != photo.sha256
        ):
            raise AnchorError("Cohort and view image digests differ")
        image_id = hashlib.sha256(f"roi-anchor:{photo.sample_id}".encode()).hexdigest()[
            :32
        ]
        native = inputs / "images" / f"{image_id}{source.suffix}"
        if not native.exists():
            native.symlink_to(source)
        full = inputs / f"{image_id}-full.png"
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            ImageOps.exif_transpose(image, in_place=True)
            ImageOps.pad(
                image, (512, 512), method=Image.Resampling.LANCZOS, color=0
            ).save(full)
        paths = {
            "native_full": [str(native)],
            "fixed": [str(full), views[photo.sample_id]["fixed"]],
            "box": [str(full), views[photo.sample_id]["box"]],
        }
        planned.append(
            {
                "sample_id": photo.sample_id,
                "image_id": image_id,
                "prefix": photo.prefix,
                "detected": views[photo.sample_id]["detected"],
                "paths": paths,
                "sha256": {
                    arm: [sha256(Path(path)) for path in values]
                    for arm, values in paths.items()
                },
            }
        )
    watched = [
        *BASE.glob("*.py"),
        BASE / "system_prompt.txt",
        *BASE.glob("scripts/*.py"),
        *BASE.glob("base_model/*.json"),
        *BASE.glob("base_model/*.safetensors"),
        *BASE.glob("adapter/*.json"),
        *BASE.glob("adapter/*.safetensors"),
        BASE / "assets/laws.jsonl",
        BASE / "assets/train.jsonl",
    ]
    plan = {
        "schema": "roi-official-generation-v1",
        "rows": planned,
        "arms": ARMS,
        "selection": "first two rows in frozen cohort order per prefix; no labels used",
        "cohort_sha256": sha256(WORK / "cohort.json"),
        "views_sha256": sha256(WORK / "views/index.json"),
        "official_code_files_matching_archive": checked,
        "implementation_sha256": sha256(Path(__file__)),
        "official_file_sha256": {
            str(path.relative_to(BASE)): sha256(path) for path in sorted(watched)
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "peft", "bitsandbytes")
        },
        "generation": {
            "max_new_tokens": MAX_TOKENS,
            "batch_size": 1,
            "do_sample": False,
            "precision": "NF4 language model, BF16 visual, official adapter",
            "min_pixels": 128 * 32 * 32,
            "max_pixels": 384 * 32 * 32,
            "two_view_canvas": [512, 512],
        },
        "interpretation": "fixed vs box matched two-view image grids; native full is a different image-token budget reference; official adapter and retrieval examples include training data, not independent holdout",
    }
    write_atomic(
        inputs / "manifest.jsonl",
        "".join(json.dumps({"image_id": row["image_id"]}) + "\n" for row in planned),
    )
    write_atomic(
        plan_path, json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(
        f"Prepared 12 photos, 36 conditions, no labels opened: {plan_path}", flush=True
    )


def load_entry():
    """Import the checked organizer parser, formatter, and conversion boundary."""
    spec = importlib.util.spec_from_file_location(
        "kwater_roi_official_entry", BASE / "main.py"
    )
    if spec is None or spec.loader is None:
        raise AnchorError("Official entrypoint could not be imported")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def seal_outputs(run: Path) -> None:
    """Seal all expected generated conditions; incomplete runs cannot be scored."""
    plan = json.loads((run / "plan.json").read_text())
    paths = [
        run / "records" / f"{row['image_id']}-{arm}.json"
        for row in plan["rows"]
        for arm in plan["arms"]
    ]
    if any(not path.is_file() for path in paths):
        raise AnchorError("Cannot seal incomplete prediction conditions")
    seal = {
        "plan_sha256": sha256(run / "plan.json"),
        "records": {path.name: sha256(path) for path in paths},
    }
    write_atomic(run / "sealed.json", json.dumps(seal, sort_keys=True, indent=2) + "\n")


def verify_seal(run: Path) -> None:
    """Reject changed/incomplete outputs before any evaluation target is opened."""
    if not (run / "sealed.json").is_file():
        raise AnchorError("Predictions must be complete and sealed before opening gold")
    seal = json.loads((run / "sealed.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    expected = {
        f"{row['image_id']}-{arm}.json" for row in plan["rows"] for arm in plan["arms"]
    }
    if (
        sha256(run / "plan.json") != seal["plan_sha256"]
        or set(seal["records"]) != expected
    ):
        raise AnchorError("Plan or sealed coverage changed")
    for name, digest in seal["records"].items():
        if sha256(run / "records" / name) != digest:
            raise AnchorError("Sealed predictions changed")


def predict(max_calls: int, run: Path = RUN) -> None:
    """Offline greedy batch-one generation, with exact-input resumable caches."""
    import torch

    from .vlm_11code import load_qwen_model, load_qwen_processor

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    plan = json.loads((run / "plan.json").read_text())
    for relative, digest in plan["official_file_sha256"].items():
        if sha256(BASE / relative) != digest:
            raise AnchorError(
                "Official model, parser, prompt, or formatter inputs changed"
            )
    for package, version in plan["versions"].items():
        if importlib.metadata.version(package) != version:
            raise AnchorError("Runtime version changed after preregistration")
    for row in plan["rows"]:
        for arm in plan["arms"]:
            if [sha256(Path(path)) for path in row["paths"][arm]] != row["sha256"][arm]:
                raise AnchorError("Frozen input pixels changed")
    if (run / "sealed.json").exists():
        verify_seal(run)
        print("Already sealed; unchanged inputs verified; no model loaded", flush=True)
        return
    if sha256(Path(__file__)) != plan["implementation_sha256"]:
        raise AnchorError("Anchor implementation changed after preregistration")
    plan_sha = sha256(run / "plan.json")
    records = run / "records"
    cache = run / "cache"
    records.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    entry = load_entry()
    prompt = (BASE / "system_prompt.txt").read_text()
    calls = 0
    with (ROOT / ".runtime/ubuntu-weekend-gpu.lock").open("a") as lock:
        print("Waiting for exclusive GPU lock", flush=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        print("Acquired exclusive GPU lock", flush=True)
        entry.seed_all(42)
        processor = load_qwen_processor(BASE / "base_model", 384 * 32 * 32)
        model = load_qwen_model(
            BASE / "base_model", use_4bit=True, adapter_dir=BASE / "adapter"
        )
        model.eval()
        generation_config = model.generation_config
        if generation_config is None:
            raise AnchorError("Loaded model has no decoding configuration")
        torch.cuda.reset_peak_memory_stats()
        for row in plan["rows"]:
            paired_grids = {}
            for arm in plan["arms"]:
                destination = records / f"{row['image_id']}-{arm}.json"
                image_paths = [Path(value) for value in row["paths"][arm]]
                if [sha256(path) for path in image_paths] != row["sha256"][arm]:
                    raise AnchorError("Frozen input pixels changed")
                contents = [{"type": "image"} for _ in image_paths] + [
                    {
                        "type": "text",
                        "text": row.get("user_text", {}).get(arm, entry.USER_TEXT),
                    }
                ]
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": contents},
                ]
                prefix = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                identity = hashlib.sha256(
                    json.dumps([plan_sha, prefix, row["sha256"][arm]]).encode()
                ).hexdigest()
                cached = cache / f"{identity}.json"
                if destination.is_file():
                    record = json.loads(destination.read_text())
                    if record["input_identity"] != identity:
                        raise AnchorError("Resume record identity mismatch")
                elif cached.is_file():
                    record = {
                        **json.loads(cached.read_text()),
                        "sample_id": row["sample_id"],
                        "image_id": row["image_id"],
                        "arm": arm,
                        "reused": True,
                    }
                    write_atomic(
                        destination, json.dumps(record, ensure_ascii=False) + "\n"
                    )
                else:
                    if max_calls and calls >= max_calls:
                        print(
                            f"Paused after {calls} new generations; outputs remain unsealed",
                            flush=True,
                        )
                        return
                    images = []
                    for path in image_paths:
                        with Image.open(path) as opened:
                            image = opened.convert("RGB")
                            ImageOps.exif_transpose(image, in_place=True)
                            images.append(image)
                    batch = processor(
                        text=[prefix], images=images, return_tensors="pt"
                    ).to("cuda")
                    grids = batch["image_grid_thw"].tolist()
                    tick = time.perf_counter()
                    with torch.inference_mode():
                        ids = model.generate(
                            **batch,
                            do_sample=False,
                            max_new_tokens=MAX_TOKENS,
                            use_cache=True,
                        )
                    new_ids = ids[0, batch["input_ids"].shape[-1] :]
                    text = processor.decode(
                        new_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    parsed, error = entry.parse_prediction(text)
                    errors = [] if error else entry.schema_errors(parsed)
                    torch.cuda.synchronize()
                    record = {
                        "sample_id": row["sample_id"],
                        "image_id": row["image_id"],
                        "arm": arm,
                        "input_identity": identity,
                        "text": text,
                        "prediction": {} if error or errors else parsed,
                        "parse_error": error,
                        "schema_errors": errors,
                        "new_tokens": len(new_ids),
                        "token_cap_hit": len(new_ids) >= MAX_TOKENS,
                        "prompt_tokens": int(batch["input_ids"].shape[-1]),
                        "image_grid_thw": grids,
                        "visual_tokens": [
                            int(t * h * w) // processor.image_processor.merge_size**2
                            for t, h, w in grids
                        ],
                        "seconds": time.perf_counter() - tick,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "reused": False,
                    }
                    eos = generation_config.eos_token_id
                    eos_ids = eos if isinstance(eos, list) else [eos]
                    record["ended_with_eos"] = int(new_ids[-1]) in eos_ids
                    record["truncated_at_cap"] = (
                        record["token_cap_hit"] and not record["ended_with_eos"]
                    )
                    write_atomic(cached, json.dumps(record, ensure_ascii=False) + "\n")
                    write_atomic(
                        destination, json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    calls += 1
                    print(
                        f"Generated {calls}: {arm}; {record['new_tokens']} tokens; {record['seconds']:.1f}s",
                        flush=True,
                    )
                    del batch, ids, new_ids
                if arm != "native_full":
                    paired_grids[arm] = record["image_grid_thw"]
            if len({json.dumps(grid) for grid in paired_grids.values()}) > 1:
                raise AnchorError("Fixed and detector crops have unequal visual grids")
        del model
        torch.cuda.empty_cache()
    seal_outputs(run)
    print("All planned conditions sealed; score may now open gold", flush=True)


def write_checked_submission(text: str, diagnostics: Path, inputs: Path) -> Path:
    """Keep the checker directory submission-only, including on repeated scoring."""
    output = diagnostics / "official-output"
    output.mkdir(parents=True, exist_ok=True)
    submission = output / "submission.jsonl"
    write_atomic(submission, text)
    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "data/local-kit/kit/check_output.py"),
            str(inputs),
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    write_atomic(diagnostics / "checker.log", checked.stdout + checked.stderr)
    expected = sum(
        bool(line.strip())
        for line in (inputs / "manifest.jsonl").read_text().splitlines()
    )
    accepted = re.search(r"유효 결과\s+(\d+)/(\d+)", checked.stdout)
    if (
        checked.returncode
        or accepted is None
        or (int(accepted[1]), int(accepted[2])) != (expected, expected)
    ):
        raise AnchorError(f"Official checker failed: {diagnostics.name}")
    return submission


def score(run: Path = RUN) -> None:
    """Run official formatting/checker, then exact local code metrics after seal."""
    verify_seal(run)
    plan = json.loads((run / "plan.json").read_text())
    entry = load_entry()
    summaries, predictions = {}, {}
    mapping = {row["sample_id"]: row["image_id"] for row in plan["rows"]}
    all_records = []
    for arm in plan["arms"]:
        records = [
            json.loads((run / "records" / f"{row['image_id']}-{arm}.json").read_text())
            for row in plan["rows"]
        ]
        all_records.extend(records)
        valid = [
            {"sample_id": row["image_id"], "prediction": row["prediction"]}
            for row in records
            if not entry.validate_prediction(row["prediction"])
        ]
        output = run / arm
        output.mkdir(exist_ok=True)
        formatted = {}
        if valid:
            internal = output / "internal.jsonl"
            entry.format_submission(
                valid,
                entry.load_rows(BASE / "assets/laws.jsonl"),
                entry.load_rows(BASE / "assets/train.jsonl"),
                internal,
            )
            formatted = {
                row["sample_id"]: row["prediction"] for row in entry.load_rows(internal)
            }
        submission = write_checked_submission(
            "".join(
                json.dumps(
                    entry.convert(row["image_id"], formatted.get(row["image_id"])),
                    ensure_ascii=False,
                )
                + "\n"
                for row in plan["rows"]
            ),
            output,
            run / "inputs",
        )
        summaries[arm] = {
            "parse_failures": sum(row["parse_error"] is not None for row in records),
            "schema_failures": sum(bool(row["schema_errors"]) for row in records),
            "fallbacks": len(records) - len(valid),
            "token_cap_hits": sum(row["token_cap_hit"] for row in records),
            "truncated_at_cap": sum(row["truncated_at_cap"] for row in records),
            "new_tokens": [row["new_tokens"] for row in records],
            "visual_tokens": [row["visual_tokens"] for row in records],
            "generation_seconds": sum(
                row["seconds"] for row in records if not row["reused"]
            ),
            "reused": sum(row["reused"] for row in records),
            "checker_exit": 0,
            "checker_valid_rows": len(records),
            "submission_path": str(submission.relative_to(ROOT)),
            "submission_sha256": sha256(submission),
        }
    # This is deliberately the first code path that opens evaluation targets.
    gold = {}
    with (ROOT / "data/train/train.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] in mapping:
                gold[mapping[row["sample_id"]]] = frozenset(row["target"]["hazards"])
    if len(gold) != len(mapping):
        raise AnchorError("Ground truth does not cover the frozen twelve photos")
    for arm in plan["arms"]:
        predictions[arm] = load_predictions(
            ROOT / summaries[arm]["submission_path"], gold
        )
        metrics = evaluate(gold, predictions[arm])
        metrics.pop("quantitative_points_out_of_70")
        summaries[arm]["metrics"] = metrics
        fixed_scores: list[float] = []
        for code in CODES:
            positives = sum(code in codes for codes in gold.values())
            proposed = sum(code in codes for codes in predictions[arm].values())
            hits = sum(
                code in gold[key] and code in predictions[arm][key] for key in gold
            )
            fixed_scores.append(
                2 * hits / (positives + proposed) if positives + proposed else 0.0
            )
        summaries[arm]["macro_f1_fixed_11_zero_absent"] = sum(fixed_scores) / len(CODES)
        summaries[arm]["missed_true_code_instances"] = sum(
            len(gold[key] - predictions[arm][key]) for key in gold
        )
    comparisons: dict[str, dict[str, int]] = {}
    report = {
        "photos": len(gold),
        "arms": summaries,
        "scoring": "macro_f1 uses active gold-or-predicted codes; fixed11 counts absent codes as zero; local code score only, not private competition score",
        "comparisons": comparisons,
        "actual_generations": sum(not row["reused"] for row in all_records),
        "plan_sha256": sha256(run / "plan.json"),
        "sealed_sha256": sha256(run / "sealed.json"),
        "prediction_implementation_sha256": plan["implementation_sha256"],
        "scoring_implementation_sha256": sha256(Path(__file__)),
        "scoring_source_changed_since_prediction": sha256(Path(__file__))
        != plan["implementation_sha256"],
        "limitations": plan["interpretation"],
    }
    for left, right in plan.get(
        "comparison_pairs", [["fixed", "box"], ["native_full", "box"]]
    ):
        changes = sum(predictions[left][key] != predictions[right][key] for key in gold)
        report[f"{left}_vs_{right}_changed_code_sets"] = changes
        comparisons[f"{right}_minus_{left}"] = {
            "changed_code_sets": changes,
            "new_exact_matches": sum(
                predictions[right][key] == gold[key]
                and predictions[left][key] != gold[key]
                for key in gold
            ),
            "lost_exact_matches": sum(
                predictions[left][key] == gold[key]
                and predictions[right][key] != gold[key]
                for key in gold
            ),
            "gained_true_codes": sum(
                len((predictions[right][key] - predictions[left][key]) & gold[key])
                for key in gold
            ),
            "lost_true_codes": sum(
                len((predictions[left][key] - predictions[right][key]) & gold[key])
                for key in gold
            ),
            "new_false_codes": sum(
                len((predictions[right][key] - predictions[left][key]) - gold[key])
                for key in gold
            ),
            "removed_false_codes": sum(
                len((predictions[left][key] - predictions[right][key]) - gold[key])
                for key in gold
            ),
        }
    report["protocol"] = plan.get("protocol", "preregistered initial anchor")
    write_atomic(
        run / "summary.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Scored all public-format predictions: {run / 'summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "predict", "score"))
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="0 means all remaining; positive pauses without sealing",
    )
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    if args.max_calls < 0:
        parser.error("max-calls must be nonnegative")

    actions = {
        "prepare": prepare,
        "predict": lambda: predict(args.max_calls, args.run_dir.resolve()),
        "score": lambda: score(args.run_dir.resolve()),
    }
    actions[args.phase]()


if __name__ == "__main__":
    main()
