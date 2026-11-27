"""Freeze CPU spatial ranks, then assess candidates on sealed official misses.

Run: .venv/bin/python -m kwater_lab.ubuntu_fold3_miss_oracle --help
This is an exploratory candidate oracle, never an automatic approval policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .metrics import CODES
from .spatial_oof_audit import FitConfig, paired_interval
from .spatial_probe import Row, digest
from .ubuntu_fold3_candidate_ranks import (
    ENCODER,
    FreezeConfig,
    OracleInputError,
    RankRow,
    freeze,
    write_json,
)

OFFICIAL8: Final = tuple(
    code
    for code in CODES
    if code not in ("drowning_inundation", "confined_space", "thermal_weather")
)


@dataclass(frozen=True, slots=True)
class BaselineRow:
    sample_id: str
    predicted: frozenset[str]
    valid: bool
    normal_fallback: bool
    parse_error: str | None


@dataclass(frozen=True, slots=True)
class Candidate:
    sample_id: str
    group_id: str
    predicted: frozenset[str]
    global_code: str | None
    regional_code: str | None
    valid: bool
    normal_fallback: bool
    full_output: bool


def choose_candidates(
    ranks: tuple[RankRow, ...],
    baseline: tuple[BaselineRow, ...],
    universe: tuple[str, ...],
) -> tuple[Candidate, ...]:
    """Select one unseen code without gold; abstain on invalid or full outputs."""
    index = {row.sample_id: row for row in baseline}
    if len(index) != len(baseline) or set(index) != {row.sample_id for row in ranks}:
        raise OracleInputError("baseline and frozen rank photo coverage differ")
    candidates = []
    for row in ranks:
        original = index[row.sample_id]
        if not original.predicted <= set(CODES):
            raise OracleInputError("baseline contains unknown code")
        for ordered in (row.global_rank, row.regional_rank):
            if len(ordered) != len(CODES) or set(ordered) != set(CODES):
                raise OracleInputError("frozen rank is not a full 11-code permutation")
        valid = (
            original.valid
            and not original.normal_fallback
            and original.parse_error is None
        )
        full = len(original.predicted) >= 3
        allowed = set(universe) - original.predicted if valid and not full else set()
        picks = [
            next((code for code in ordered if code in allowed), None)
            for ordered in (row.global_rank, row.regional_rank)
        ]
        candidates.append(
            Candidate(
                row.sample_id,
                row.group_id,
                original.predicted,
                picks[0],
                picks[1],
                valid,
                original.normal_fallback,
                full,
            )
        )
    return tuple(candidates)


def evaluate_candidates(
    candidates: tuple[Candidate, ...],
    gold: dict[str, frozenset[str]],
    universe: tuple[str, ...],
):
    """Count actual baseline false negatives, hypothetical oracle recovery, calls."""
    if set(gold) != {row.sample_id for row in candidates}:
        raise OracleInputError("gold and candidate coverage differ")
    valid = [row for row in candidates if row.valid]
    misses = [gold[row.sample_id] & set(universe) - row.predicted for row in valid]
    hits = np.asarray(
        [
            [row.global_code in missing, row.regional_code in missing]
            for row, missing in zip(valid, misses, strict=True)
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    denominator = sum(map(len, misses))
    counts = hits.sum(axis=0)
    outcomes = np.column_stack(
        (hits[:, 1] - hits[:, 0], [len(missing) for missing in misses])
    )
    rows = tuple(
        Row(row.sample_id, "", "", row.group_id, 3, frozenset()) for row in valid
    )
    interval = asdict(paired_interval(rows, outcomes, FitConfig())) if valid else None
    normal = [row for row in valid if not gold[row.sample_id]]
    per_code = {
        code: {
            "baseline_fn": sum(code in missing for missing in misses),
            "global_recovery": sum(
                row.global_code == code and code in missing
                for row, missing in zip(valid, misses, strict=True)
            ),
            "regional_recovery": sum(
                row.regional_code == code and code in missing
                for row, missing in zip(valid, misses, strict=True)
            ),
        }
        for code in universe
    }
    return {
        "photos": len(candidates),
        "valid_raw_photos": len(valid),
        "invalid_or_fallback_photos_excluded": len(candidates) - len(valid),
        "normal_fallback_photos": sum(row.normal_fallback for row in candidates),
        "full_output_photos": sum(row.full_output for row in valid),
        "full_output_baseline_fn": sum(
            len(missing)
            for row, missing in zip(valid, misses, strict=True)
            if row.full_output
        ),
        "baseline_fn_instances": denominator,
        "baseline_missed_photos": sum(bool(missing) for missing in misses),
        "eligible_missed_photos": sum(
            bool(missing) and row.global_code is not None
            for row, missing in zip(valid, misses, strict=True)
        ),
        "eligible_fn_instances": sum(
            len(missing)
            for row, missing in zip(valid, misses, strict=True)
            if row.global_code is not None
        ),
        "global_oracle_recovery": int(counts[0]),
        "regional_oracle_recovery": int(counts[1]),
        "global_recovery_fraction": int(counts[0]) / denominator
        if denominator
        else None,
        "regional_recovery_fraction": int(counts[1]) / denominator
        if denominator
        else None,
        "delta_recovery": int(counts[1] - counts[0]),
        "paired_group_interval_recovery_fraction": interval,
        "paired_hit_counts": {
            "both": int((hits[:, 0] & hits[:, 1]).sum()),
            "global_only": int(((hits[:, 0] == 1) & (hits[:, 1] == 0)).sum()),
            "regional_only": int(((hits[:, 1] == 1) & (hits[:, 0] == 0)).sum()),
            "neither": int(((hits[:, 0] == 0) & (hits[:, 1] == 0)).sum()),
        },
        "planned_extra_calls": {
            "global": sum(row.global_code is not None for row in valid),
            "regional": sum(row.regional_code is not None for row in valid),
        },
        "actual_verifier_calls": 0,
        "normal_photos": len(normal),
        "normal_proposals": {
            "global": sum(row.global_code is not None for row in normal),
            "regional": sum(row.regional_code is not None for row in normal),
        },
        "normal_struck_by_proposals": {
            "global": sum(row.global_code == "struck_by_lifting" for row in normal),
            "regional": sum(row.regional_code == "struck_by_lifting" for row in normal),
        },
        "per_code": per_code,
        "per_photo": [
            {
                "sample_id": row.sample_id,
                "valid_raw": row.valid,
                "global_candidate": row.global_code,
                "regional_candidate": row.regional_code,
                "global_calls": int(row.global_code is not None),
                "regional_calls": int(row.regional_code is not None),
                "baseline_fn": sorted(
                    gold[row.sample_id] & set(universe) - row.predicted
                )
                if row.valid
                else None,
                "global_recovers": row.global_code
                in (gold[row.sample_id] - row.predicted)
                if row.valid
                else None,
                "regional_recovers": row.regional_code
                in (gold[row.sample_id] - row.predicted)
                if row.valid
                else None,
            }
            for row in candidates
        ],
    }


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    ranks: Path
    baseline: Path
    manifest: Path
    out: Path
    cpu_reference: Path | None = None


def score(config: ScoreConfig) -> Path:
    """Seal label-free missing-code choices before reading Fold3 gold."""
    from .ubuntu_fold3_generation_protocol import verify_seal

    rank_seal = json.loads(config.ranks.with_suffix(".seal.json").read_text())
    if digest(config.ranks) != rank_seal["sha256"]:
        raise OracleInputError("frozen ranks changed")
    raw_ranks = json.loads(config.ranks.read_text())
    ranks = tuple(RankRow(**row) for row in raw_ranks["rows"])
    if len(ranks) != 201 or raw_ranks["validation_photos"] != 201:
        raise OracleInputError("production scoring requires exact Fold3 201 coverage")
    plan = verify_seal(config.baseline)
    source_path = config.baseline / "candidate-source.json"
    seal_path = config.baseline / "candidate-source.seal.json"
    source_seal = json.loads(seal_path.read_text())
    for key, path in (
        ("plan_sha256", config.baseline / "plan.json"),
        ("raw_seal_sha256", config.baseline / "sealed.json"),
        ("candidate_source_sha256", source_path),
        ("submission_sha256", config.baseline / "official-output/submission.jsonl"),
    ):
        if source_seal[key] != digest(path):
            raise OracleInputError(f"baseline seal differs: {key}")
    identities = {row.sample_id: (row.image_sha256, row.group_id) for row in ranks}
    if identities != {
        row["sample_id"]: (row["sha256"], row["group_id"]) for row in plan["rows"]
    }:
        raise OracleInputError(
            "baseline image/group identities differ from rank freeze"
        )
    source = json.loads(source_path.read_text())
    if source["schema"] != "fold3-baseline-candidates-v1":
        raise OracleInputError("unknown baseline candidate schema")
    baseline = tuple(
        BaselineRow(
            row["sample_id"],
            frozenset(row["codes"]),
            not row["schema_errors"],
            row["normal_fallback"],
            row["parse_error"],
        )
        for row in source["rows"]
    )
    policies = {
        name: choose_candidates(ranks, baseline, universe)
        for name, universe in (("all11", CODES), ("official8", OFFICIAL8))
    }
    selected_path = config.out / "candidates.json"
    write_json(
        selected_path,
        {
            "rank_sha256": digest(config.ranks),
            "baseline_seal_sha256": digest(seal_path),
            "policy": "highest score outside original output; abstain if invalid/fallback or >=3 hazards",
            "universes": {"all11": list(CODES), "official8": list(OFFICIAL8)},
            "rows": {
                name: [
                    {**asdict(row), "predicted": sorted(row.predicted)} for row in rows
                ]
                for name, rows in policies.items()
            },
        },
    )
    selected_seal = config.out / "candidates.seal.json"
    write_json(selected_seal, {"sha256": digest(selected_path)})
    # Evaluation labels first become available after both policies are immutable.
    if digest(config.manifest) != raw_ranks["manifest_sha256"]:
        raise OracleInputError("scoring manifest differs from frozen input")
    manifest = json.loads(config.manifest.read_text())
    gold = {
        row["sample_id"]: frozenset(row["codes"])
        for row in manifest
        if row["split"] == "dev" and row["dev_fold"] == 3
    }
    policy_reports = {
        name: evaluate_candidates(rows, gold, universe)
        for (name, rows), universe in zip(
            policies.items(), (CODES, OFFICIAL8), strict=True
        )
    }
    report = {
        "scope": "Exploratory CPU-refit candidate-only oracle on actual official first-pass misses; no deployment uplift or forced acceptance claim",
        "limitations": [
            "Fold3 previously explored; not an untouched confirmatory holdout",
            "Oracle accepts only true missing codes; practical verifier precision and recall unmeasured",
            "Intervals resample recorded incident groups with fixed fitted heads; grouping is not independently certified",
            "Official baseline uses collision_contact -> other and has no native drowning, confined-space, or thermal-weather code",
        ],
        "rank_sha256": digest(config.ranks),
        "baseline_seal_sha256": digest(seal_path),
        "candidate_seal_sha256": digest(selected_seal),
        "gold_manifest_sha256": digest(config.manifest),
        "candidate_fit": raw_ranks["fit"],
        "policies": policy_reports,
        "cpu_refit_top1": {
            "abnormal_photos": sum(bool(codes) for codes in gold.values()),
            "global_hits": sum(
                row.global_rank[0] in gold[row.sample_id] for row in ranks
            ),
            "regional_hits": sum(
                row.regional_rank[0] in gold[row.sample_id] for row in ranks
            ),
        },
    }
    comparison = None
    if config.cpu_reference is not None:
        reference = json.loads(config.cpu_reference.read_text())
        if (
            reference["fit_device"] != "cpu"
            or reference["inputs"]["manifest_sha256"] != raw_ranks["manifest_sha256"]
            or reference["inputs"]["encoder_sha256"] != raw_ranks["encoder_sha256"]
        ):
            raise OracleInputError("CPU comparison reference input identity differs")
        previous = next(row for row in reference["per_fold"] if row["fold"] == 3)
        eligible = set(previous["ranking_eligible_codes_from_fit_only"])
        matched_hits = {
            name: sum(
                next(code for code in ordered if code in eligible)
                in gold[row.sample_id]
                for row, ordered in ((row, getattr(row, field)) for row in ranks)
            )
            for name, field in (
                ("global_hits", "global_rank"),
                ("regional_hits", "regional_rank"),
            )
        }
        comparison = {
            "path": str(config.cpu_reference),
            "sha256": digest(config.cpu_reference),
            "reference_fit_positive_top1": {
                key: previous["observed"][key] for key in matched_hits
            },
            "new_cpu_refit_fit_positive_top1": matched_hits,
            "equal_top1_counts": all(
                value == previous["observed"][key]
                for key, value in matched_hits.items()
            ),
            "limitation": "Aggregate top1 comparison only; no original per-photo scores or weights persisted. CPU and GPU fits are distinct.",
        }
        report["known_cpu_comparison"] = comparison
    path = config.out / "report.json"
    write_json(path, report)
    aggregate = {
        key: value
        for key, value in report.items()
        if key not in ("policies", "known_cpu_comparison")
    }
    aggregate["policies"] = {
        name: {key: value for key, value in policy.items() if key != "per_photo"}
        for name, policy in policy_reports.items()
    }
    if comparison is not None:
        aggregate["known_cpu_comparison"] = {
            key: value for key, value in comparison.items() if key != "path"
        }
    write_json(config.out / "aggregate.json", aggregate)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/clean-split-v1/split_manifest_dev.json"),
    )
    freeze_parser.add_argument(
        "--summary", type=Path, default=Path("results/clean_split_v1_summary.json")
    )
    freeze_parser.add_argument(
        "--cache", type=Path, default=Path(".runtime/spatial-qwen3vl-cache")
    )
    freeze_parser.add_argument("--out", type=Path, required=True)
    freeze_parser.add_argument("--encoder", default=ENCODER)
    freeze_parser.add_argument("--steps", type=int, default=160)
    freeze_parser.add_argument("--threads", type=int, default=2)
    score_parser = commands.add_parser("score")
    score_parser.add_argument(
        "--ranks",
        type=Path,
        default=Path("data/ubuntu-fold3-miss-oracle-20261127/ranks.json"),
    )
    score_parser.add_argument(
        "--baseline", type=Path, default=Path("data/ubuntu-fold3-baseline-20261127")
    )
    score_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/clean-split-v1/split_manifest_dev.json"),
    )
    score_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/ubuntu-fold3-miss-oracle-20261127/scored"),
    )
    score_parser.add_argument(
        "--cpu-reference",
        type=Path,
        default=Path("data/ubuntu3050-runs/spatial-paired-oof-262144.json"),
    )
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(
            FreezeConfig(
                args.manifest,
                args.summary,
                args.cache,
                args.out,
                encoder=args.encoder,
                steps=args.steps,
                threads=args.threads,
            )
        )
    else:
        result = score(
            ScoreConfig(
                args.ranks, args.baseline, args.manifest, args.out, args.cpu_reference
            )
        )
    print(result)


if __name__ == "__main__":
    main()
