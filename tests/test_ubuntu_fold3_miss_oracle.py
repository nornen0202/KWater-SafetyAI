"""Protect real-miss denominators, output capacity, and pre-gold rank sealing."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from kwater_lab.metrics import CODES
from kwater_lab.spatial_probe import _atomic_npz, digest
from kwater_lab.ubuntu_fold3_candidate_ranks import (
    FreezeConfig,
    OracleInputError,
    RankRow,
    freeze,
    write_json,
)
from kwater_lab.ubuntu_fold3_miss_oracle import (
    OFFICIAL8,
    BaselineRow,
    ScoreConfig,
    choose_candidates,
    evaluate_candidates,
    score,
)


def ranking(sample_id: str, global_first: str, regional_first: str) -> RankRow:
    return RankRow(
        sample_id,
        sample_id,
        hashlib.sha256(sample_id.encode()).hexdigest(),
        tuple(float(len(CODES) - i) for i in range(len(CODES))),
        tuple(float(len(CODES) - i) for i in range(len(CODES))),
        (global_first, *(code for code in CODES if code != global_first)),
        (regional_first, *(code for code in CODES if code != regional_first)),
    )


def test_miss_oracle_respects_baseline_capacity_and_invalid_rows() -> None:
    ranks = tuple(ranking(str(i), "slip_trip", "struck_by_lifting") for i in range(5))
    baseline = (
        BaselineRow("0", frozenset({"slip_trip"}), True, False, None),
        BaselineRow("1", frozenset(), True, False, None),
        BaselineRow("2", frozenset(CODES[:3]), True, False, None),
        BaselineRow("3", frozenset(), False, True, "invalid JSON"),
        BaselineRow("4", frozenset(), True, False, None),
    )
    gold = {
        "0": frozenset({"slip_trip", "struck_by_lifting"}),
        "1": frozenset({"slip_trip", "other"}),
        "2": frozenset({"struck_by_lifting"}),
        "3": frozenset({"struck_by_lifting"}),
        "4": frozenset(),
    }

    candidates = choose_candidates(ranks, baseline, CODES)
    result = evaluate_candidates(candidates, gold, CODES)

    assert candidates[0].global_code == "fall_height"
    assert candidates[2].global_code is candidates[2].regional_code is None
    assert candidates[3].global_code is candidates[3].regional_code is None
    assert result["baseline_fn_instances"] == 4
    assert result["baseline_missed_photos"] == 3
    assert result["eligible_missed_photos"] == 2
    assert result["full_output_baseline_fn"] == 1
    assert result["global_oracle_recovery"] == result["regional_oracle_recovery"] == 1
    assert result["global_recovery_fraction"] == 0.25
    assert result["planned_extra_calls"] == {"global": 3, "regional": 3}
    assert result["normal_struck_by_proposals"] == {"global": 0, "regional": 1}
    assert result["paired_hit_counts"] == {
        "both": 0,
        "global_only": 1,
        "regional_only": 1,
        "neither": 2,
    }


def test_official_universe_changes_opportunity_without_hiding_all11_misses() -> None:
    ranks = (ranking("p", "drowning_inundation", "struck_by_lifting"),)
    baseline = (BaselineRow("p", frozenset(), True, False, None),)
    gold = {"p": frozenset({"drowning_inundation"})}

    all_result = evaluate_candidates(
        choose_candidates(ranks, baseline, CODES), gold, CODES
    )
    official = evaluate_candidates(
        choose_candidates(ranks, baseline, OFFICIAL8), gold, OFFICIAL8
    )

    assert all_result["global_oracle_recovery"] == 1
    assert all_result["baseline_fn_instances"] == 1
    assert official["baseline_fn_instances"] == 0
    assert official["global_recovery_fraction"] is None
    assert official["normal_photos"] == 0


def test_freeze_cpu_uses_only_training_labels_and_resumes_without_refit(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"
    rows = []
    for fold in (1, 2, 3):
        for index in range(2):
            sid = f"{fold}-{index}"
            sha = hashlib.sha256(sid.encode()).hexdigest()
            row: dict[str, str | int | list[str]] = {
                "sample_id": sid,
                "sha256": sha,
                "image": f"{sid}.png",
                "group_id": sid,
                "split": "dev",
                "dev_fold": fold,
            }
            if fold != 3:
                row["codes"] = [CODES[index]]
            rows.append(row)
            _atomic_npz(
                cache / f"{sha}-synthetic.npz",
                features=np.full((16, 2), index, dtype=np.float32),
                source_sha256=np.asarray(sha),
                encoder_sha256=np.asarray("synthetic"),
            )
    write_json(manifest, rows)
    summary = tmp_path / "summary.json"
    write_json(summary, {"dev_manifest_sha256": digest(manifest)})
    config = FreezeConfig(
        manifest, summary, cache, tmp_path / "ranks.json", "synthetic", 2
    )

    seal = freeze(config)
    first_hash = digest(config.out)
    resumed = freeze(config)

    frozen = json.loads(config.out.read_text())
    assert resumed == seal
    assert digest(config.out) == first_hash == json.loads(seal.read_text())["sha256"]
    assert frozen["fit_photos"] == 4
    assert {row["sample_id"] for row in frozen["rows"]} == {"3-0", "3-1"}
    assert frozen["train_positive_codes"] == list(CODES[:2])
    assert all(len(row["global_rank"]) == len(CODES) for row in frozen["rows"])


def scoring_fixture(root: Path) -> ScoreConfig:
    from kwater_lab.ubuntu_fold3_generation_protocol import Plan, Row, seal, stable_hash

    baseline = root / "baseline"
    manifest = root / "manifest.json"
    ranks_path = root / "ranks.json"
    ranks, source, gold = [], [], []
    planned: list[Row] = []
    for index in range(201):
        sid = f"private-photo-{index}"
        rank = ranking(sid, "fall_height", "slip_trip")
        ranks.append(asdict(rank))
        payload = {"normal_fallback": False, "parse_error": None, "schema_errors": []}
        identity = stable_hash({"sample_id": sid})
        planned.append(
            {
                "sample_id": sid,
                "image_id": sid,
                "sha256": rank.image_sha256,
                "image": f"{sid}.jpg",
                "group_id": sid,
                "input_identity": identity,
            }
        )
        write_json(
            baseline / "records" / f"{sid}.json",
            {
                "sample_id": sid,
                "image_id": sid,
                "input_identity": identity,
                "payload": payload,
                "payload_sha256": stable_hash(payload),
            },
        )
        source.append({"sample_id": sid, "codes": [], **payload})
        gold.append(
            {"sample_id": sid, "split": "dev", "dev_fold": 3, "codes": ["slip_trip"]}
        )
    write_json(manifest, gold)
    write_json(
        ranks_path,
        {
            "rows": ranks,
            "manifest_sha256": digest(manifest),
            "validation_photos": 201,
            "encoder_sha256": "synthetic",
            "fit": {"device": "cpu"},
        },
    )
    write_json(ranks_path.with_suffix(".seal.json"), {"sha256": digest(ranks_path)})
    plan: Plan = {
        "rows": planned,
        "schema": "fixture",
        "protocol_sha256": "fixture",
        "protocol": {
            "official_files": {},
            "source": {},
            "packages": {},
            "prefix": "",
            "role": "",
        },
    }
    write_json(baseline / "plan.json", plan)
    seal(baseline, plan)
    write_json(
        baseline / "candidate-source.json",
        {"schema": "fold3-baseline-candidates-v1", "rows": source},
    )
    write_json(baseline / "official-output/submission.jsonl", {"fixture": True})
    write_json(
        baseline / "candidate-source.seal.json",
        {
            "plan_sha256": digest(baseline / "plan.json"),
            "raw_seal_sha256": digest(baseline / "sealed.json"),
            "candidate_source_sha256": digest(baseline / "candidate-source.json"),
            "submission_sha256": digest(baseline / "official-output/submission.jsonl"),
        },
    )
    reference = root / "cpu-reference.json"
    write_json(
        reference,
        {
            "fit_device": "cpu",
            "inputs": {
                "manifest_sha256": digest(manifest),
                "encoder_sha256": "synthetic",
            },
            "per_fold": [
                {
                    "fold": 3,
                    "ranking_eligible_codes_from_fit_only": list(CODES),
                    "observed": {"global_hits": 0, "regional_hits": 201},
                }
            ],
        },
    )
    return ScoreConfig(ranks_path, baseline, manifest, root / "scored", reference)


def test_score_exact201_seals_choices_before_gold_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    config = scoring_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kwater_lab.ubuntu_fold3_miss_oracle",
            "score",
            "--ranks",
            str(config.ranks),
            "--baseline",
            str(config.baseline),
            "--manifest",
            str(config.manifest),
            "--out",
            str(config.out),
            "--cpu-reference",
            str(config.cpu_reference),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report_path = Path(completed.stdout.strip())

    report = json.loads(report_path.read_text())
    assert report["policies"]["all11"]["regional_oracle_recovery"] == 201
    assert report["policies"]["all11"]["global_oracle_recovery"] == 0
    assert report["policies"]["all11"]["planned_extra_calls"] == {
        "global": 201,
        "regional": 201,
    }
    assert report["candidate_seal_sha256"] == digest(
        config.out / "candidates.seal.json"
    )
    aggregate_text = (config.out / "aggregate.json").read_text()
    aggregate = json.loads(aggregate_text)
    assert "private-photo-" not in aggregate_text
    assert str(tmp_path) not in aggregate_text
    assert aggregate["policies"]["all11"] == {
        key: value
        for key, value in report["policies"]["all11"].items()
        if key != "per_photo"
    }
    assert aggregate["known_cpu_comparison"]["equal_top1_counts"]
    config.ranks.write_text("{}")
    with pytest.raises(OracleInputError, match="frozen ranks changed"):
        score(config)


def test_gold_failure_still_leaves_immutable_pregold_candidates(tmp_path: Path) -> None:
    config = scoring_fixture(tmp_path)
    config.manifest.write_text("changed after freeze")

    with pytest.raises(OracleInputError, match="scoring manifest differs"):
        score(config)

    assert (config.out / "candidates.seal.json").is_file()
    assert not (config.out / "report.json").exists()
