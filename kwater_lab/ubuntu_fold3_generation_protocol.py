"""Preregister and seal the official eight-code baseline on canonical Fold3.

Run: .venv/bin/python -m kwater_lab.ubuntu_fold3_generation prepare
Then: predict [--max-calls N], export. No evaluation gold is read by this module.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, TypedDict

from .fold0_official_resume import baseline_fingerprint
from .metrics import unique_keys
from .run_official_baseline import sha256
from .ubuntu_inner_verifier_eval import MANIFEST_SHA, PHOTO_COUNT, load_cohort
from .ubuntu_roi_generation import BASE, ROOT, load_entry, write_checked_submission

RUN: Final = ROOT / "data/ubuntu-fold3-baseline-20261127"
ARCHIVE_SHA: Final = "be91eb3a745dc6e1f305237b44842106707164f4174940439b246e4abb385aab"
MAX_PIXELS: Final = 384 * 32 * 32
MAX_TOKENS: Final = 900
PACKAGES: Final = (
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "bitsandbytes",
    "Pillow",
    "tokenizers",
    "accelerate",
    "safetensors",
    "numpy",
    "jsonschema",
)
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


@dataclass(frozen=True, slots=True)
class BaselineError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class Row(TypedDict):
    sample_id: str
    image_id: str
    image: str
    sha256: str
    group_id: str
    input_identity: str


class Protocol(TypedDict):
    official_files: dict[str, str]
    source: dict[str, str]
    packages: dict[str, str]
    prefix: str
    role: str


class Plan(TypedDict):
    schema: str
    protocol: Protocol
    protocol_sha256: str
    rows: list[Row]


def stable_hash[T](value: T) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def write_once[T](path: Path, payload: T) -> None:
    """Atomically install a complete artifact; reject differing resumptions."""
    if path.exists():
        if read_json(path) != payload:
            raise BaselineError(f"Refusing differing artifact: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(stream.name, path)
        except FileExistsError:
            if read_json(path) != payload:
                raise BaselineError("Concurrent differing artifact") from None


def source_files() -> list[Path]:
    """Include every local helper used by prediction and official export."""
    return [
        Path(__file__),
        Path(__file__).with_name("ubuntu_fold3_generation.py"),
        *(
            Path(__file__).with_name(name)
            for name in (
                "vlm_11code.py",
                "ubuntu_roi_generation.py",
                "fold0_official_resume.py",
                "run_official_baseline.py",
                "ubuntu_inner_verifier_eval.py",
                "metrics.py",
            )
        ),
        *sorted((ROOT / "data/local-kit/kit").glob("*.py")),
        *sorted((ROOT / "data/local-kit/kit").glob("*.json")),
    ]


def prepare(run: Path = RUN) -> None:
    """CPU-only provenance and canonical input checks, with no gold access."""
    from .vlm_11code import load_qwen_processor

    started = time.perf_counter()
    photos = load_cohort(ROOT)
    archive = ROOT / "data/packages/kwater-baseline.zip"
    if sha256(archive) != ARCHIVE_SHA:
        raise BaselineError("Official ZIP does not match the archived baseline")
    official = baseline_fingerprint(BASE, archive)
    entry = load_entry()
    processor = load_qwen_processor(BASE / "base_model", MAX_PIXELS)
    processor.tokenizer.padding_side = "left"
    prefix = processor.apply_chat_template(
        [
            {"role": "system", "content": (BASE / "system_prompt.txt").read_text()},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": entry.USER_TEXT},
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    protocol = {
        "canonical_manifest_sha256": MANIFEST_SHA,
        "official_archive_sha256": ARCHIVE_SHA,
        "official_files": official,
        "source": {str(p.relative_to(ROOT)): sha256(p) for p in source_files()},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "prefix": prefix,
        "processor": json.loads(processor.image_processor.to_json_string()),
        "generation_config": read_json(BASE / "base_model/generation_config.json"),
        "decoding": {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "seed": 42,
        },
        "image": {
            "min_pixels": 128 * 32 * 32,
            "max_pixels": MAX_PIXELS,
            "transform": "EXIF transpose then RGB; full image; no crop",
            "padding": True,
        },
        "batch_size": 1,
        "precision": "NF4 double quant BF16 compute, visual/lm_head excluded; sdpa; official adapter",
        "role": "Baseline diagnostic only; official adapter and retrieval examples may expose Fold3; not independent generalization",
        "archived_comparability": {
            "confirmed_same": [
                "official weights and adapter",
                "system and user prompt",
                "900 new tokens",
                "131072-393216 pixel budget",
                "full-image EXIF/RGB",
                "greedy",
                "seed42",
                "batch1",
            ],
            "different": [
                "NF4 instead of archived BF16",
                "Linux RTX3050 instead of Windows RTX3060",
                "runtime versions",
            ],
            "no_claim": "Not bitwise equivalence or competition-container compliance; no label-driven tuning",
        },
    }
    fingerprint = stable_hash(protocol)
    rows = []
    for photo in photos:
        rows.append(
            {
                **asdict(photo),
                "image_id": hashlib.sha256(
                    f"fold3-baseline:{photo.sample_id}".encode()
                ).hexdigest()[:32],
                "input_identity": stable_hash(
                    {"protocol": fingerprint, "photo": asdict(photo)}
                ),
            }
        )
    plan = {
        "schema": "fold3-official-fullfield-v1",
        "protocol": protocol,
        "protocol_sha256": fingerprint,
        "rows": rows,
    }
    write_once(run / "plan.json", plan)
    inputs = run / "inputs"
    (inputs / "images").mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = ROOT / "data/train" / row["image"]
        link = inputs / "images" / f"{row['image_id']}{source.suffix}"
        if link.exists() and sha256(link) != row["sha256"]:
            raise BaselineError("Checker input differs from canonical photo")
        if not link.exists():
            link.symlink_to(source)
    manifest = "".join(json.dumps({"image_id": r["image_id"]}) + "\n" for r in rows)
    manifest_path = inputs / "manifest.jsonl"
    if manifest_path.exists() and manifest_path.read_text() != manifest:
        raise BaselineError("Input manifest changed")
    manifest_path.write_text(manifest)
    print(
        json.dumps(
            {
                "prepared_photos": len(rows),
                "gold_opened": False,
                "cpu_prepare_seconds": time.perf_counter() - started,
                "run": str(run),
            }
        ),
        flush=True,
    )


def verify_plan(run: Path) -> Plan:
    """Check exact source, runtime, weights and images before cache consumption."""
    plan: Plan = read_json(run / "plan.json")
    protocol = plan["protocol"]
    if stable_hash(protocol) != plan["protocol_sha256"]:
        raise BaselineError("Protocol hash changed")
    photos = load_cohort(ROOT)
    expected = [
        {
            **asdict(p),
            "image_id": hashlib.sha256(
                f"fold3-baseline:{p.sample_id}".encode()
            ).hexdigest()[:32],
            "input_identity": stable_hash(
                {"protocol": plan["protocol_sha256"], "photo": asdict(p)}
            ),
        }
        for p in photos
    ]
    if plan["rows"] != expected:
        raise BaselineError("Plan is not the exact canonical Fold3 cohort")
    for relative, digest in protocol["official_files"].items():
        if sha256(BASE / relative) != digest:
            raise BaselineError("Official asset changed")
    for relative, digest in protocol["source"].items():
        if sha256(ROOT / relative) != digest:
            raise BaselineError("Prediction or export source changed")
    for name, version in protocol["packages"].items():
        if importlib.metadata.version(name) != version:
            raise BaselineError("Runtime package changed")
    return plan


def load_record(run: Path, row: Row):
    """Reject partial, mismatched or modified cached payloads."""
    path = run / "records" / f"{row['image_id']}.json"
    if not path.exists():
        return None
    record = read_json(path)
    if (
        record["input_identity"] != row["input_identity"]
        or record["sample_id"] != row["sample_id"]
        or record["image_id"] != row["image_id"]
        or stable_hash(record["payload"]) != record["payload_sha256"]
    ):
        raise BaselineError("Cached input identity or payload differs")
    return record


def seal(run: Path, plan: Plan) -> None:
    """Only the exact 201 complete per-image records can be sealed."""
    if len(plan["rows"]) != PHOTO_COUNT:
        raise BaselineError("Incomplete planned coverage")
    for row in plan["rows"]:
        if load_record(run, row) is None:
            raise BaselineError("Cannot seal missing prediction")
    write_once(
        run / "sealed.json",
        {
            "plan_sha256": sha256(run / "plan.json"),
            "records": {
                f"{r['image_id']}.json": sha256(
                    run / "records" / f"{r['image_id']}.json"
                )
                for r in plan["rows"]
            },
        },
    )


def verify_seal(run: Path) -> Plan:
    plan: Plan = read_json(run / "plan.json")
    frozen = read_json(run / "sealed.json")
    if frozen["plan_sha256"] != sha256(run / "plan.json"):
        raise BaselineError("Sealed plan changed")
    expected = {f"{r['image_id']}.json" for r in plan["rows"]}
    if len(expected) != PHOTO_COUNT or set(frozen["records"]) != expected:
        raise BaselineError("Sealed coverage differs")
    for name, digest in frozen["records"].items():
        if sha256(run / "records" / name) != digest:
            raise BaselineError("Sealed record changed")
    for row in plan["rows"]:
        load_record(run, row)
    return plan


def export(run: Path) -> None:
    """Use original parser/formatter/kit; preserve fallback attribution for mining."""
    plan = verify_plan(run)
    verify_seal(run)
    entry = load_entry()
    records = []
    for row in plan["rows"]:
        record = load_record(run, row)
        if record is None:
            raise BaselineError("Sealed record missing during export")
        records.append(record)
    valid = [
        {"sample_id": r["image_id"], "prediction": r["payload"]["prediction"]}
        for r in records
        if not r["payload"]["normal_fallback"]
    ]
    formatted = {}
    if valid:
        internal = run / "internal.jsonl"
        entry.format_submission(
            valid,
            entry.load_rows(BASE / "assets/laws.jsonl"),
            entry.load_rows(BASE / "assets/train.jsonl"),
            internal,
        )
        formatted = {r["sample_id"]: r["prediction"] for r in entry.load_rows(internal)}
    converted = [
        entry.convert(r["image_id"], formatted.get(r["image_id"])) for r in records
    ]
    submission = write_checked_submission(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in converted),
        run,
        run / "inputs",
    )
    candidate = [
        {
            "sample_id": r["sample_id"],
            "image_id": r["image_id"],
            "input_identity": r["input_identity"],
            "normal_fallback": r["payload"]["normal_fallback"],
            "parse_error": r["payload"]["parse_error"],
            "schema_errors": r["payload"]["schema_errors"],
            "codes": sorted(output["hazards"]),
            "prediction": output,
        }
        for r, output in zip(records, converted, strict=True)
    ]
    write_once(
        run / "candidate-source.json",
        {"schema": "fold3-baseline-candidates-v1", "rows": candidate},
    )
    write_once(
        run / "candidate-source.seal.json",
        {
            "plan_sha256": sha256(run / "plan.json"),
            "raw_seal_sha256": sha256(run / "sealed.json"),
            "candidate_source_sha256": sha256(run / "candidate-source.json"),
            "submission_sha256": sha256(submission),
        },
    )
    times = [r["payload"]["seconds"] for r in records]
    summary = {
        "photos": len(records),
        "gold_opened": False,
        "checker_valid_rows": len(records),
        "parse_failures": sum(r["payload"]["parse_error"] is not None for r in records),
        "schema_failures": sum(bool(r["payload"]["schema_errors"]) for r in records),
        "normal_fallbacks": sum(bool(r["normal_fallback"]) for r in candidate),
        "genuine_no_visible_hazard": sum(
            not r["codes"] and not r["normal_fallback"] for r in candidate
        ),
        "truncated_at_cap": sum(r["payload"]["truncated_at_cap"] for r in records),
        "generation_seconds": sum(times),
        "role": plan["protocol"]["role"],
        "candidate_source_sha256": sha256(run / "candidate-source.json"),
    }
    write_once(run / "aggregate.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
