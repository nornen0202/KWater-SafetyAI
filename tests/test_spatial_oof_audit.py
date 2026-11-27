"""Paired metric, cluster sampling, boundary, and real CPU CLI contracts."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pytest

from kwater_lab.metrics import CODES
from kwater_lab.spatial_oof_audit import (
    AuditInputError,
    Dataset,
    FitConfig,
    Predictions,
    load_dataset,
    paired_interval,
    shuffle_training_labels,
    summarize,
)
from kwater_lab.spatial_probe import Row, _atomic_npz


def synthetic_rows(cache: Path) -> list[Row]:
    rows = []
    for fold in (1, 2, 3):
        for index in range(9):
            sample_id = f"synthetic-{fold}-{index}"
            codes = frozenset({CODES[index % 3]}) if index % 3 < 2 else frozenset()
            row = Row(
                sample_id,
                f"{sample_id}.png",
                hashlib.sha256(sample_id.encode()).hexdigest(),
                f"event-{fold}-{index // 3}",
                fold,
                codes,
            )
            features = np.zeros((16, 2), dtype=np.float32)
            if index % 3 < 2:
                features[:, index % 3] = 3
            _atomic_npz(
                cache / f"{row.sha256}-synthetic.npz",
                features=features,
                source_sha256=np.asarray(row.sha256),
                encoder_sha256=np.asarray("synthetic"),
            )
            rows.append(row)
    return rows


def test_paired_counts_distinguish_multilabel_code_switches_and_normals() -> None:
    label_sets = ({CODES[0]}, {CODES[1]}, {CODES[0]}, {CODES[0], CODES[1]}, set())
    rows = tuple(
        Row(str(i), "", str(i), str(i), 1, frozenset(codes))
        for i, codes in enumerate(label_sets)
    )
    labels = np.asarray([[code in row.codes for code in CODES] for row in rows])
    dataset = Dataset(rows, np.zeros((5, 16, 2), dtype=np.float32), labels)

    result = summarize(
        dataset,
        Predictions(np.asarray([0, 0, 0, 0, 0]), np.asarray([0, 1, 1, 1, 1])),
        FitConfig(bootstrap_resamples=2000),
    )

    assert (result.abnormal_photos, result.global_hits, result.regional_hits) == (
        4,
        3,
        3,
    )
    assert (
        result.contingency.both_hit,
        result.contingency.global_only,
        result.contingency.regional_only,
        result.contingency.neither_hit,
    ) == (2, 1, 1, 0)
    assert result.delta_regional_minus_global == 0.0
    assert (
        result.per_code[0].support,
        result.per_code[0].losses,
        result.per_code[1].gains,
    ) == (3, 2, 2)
    assert (
        result.normal_photos,
        result.normal_global_proposals[CODES[0]],
        result.normal_regional_proposals[CODES[1]],
    ) == (1, 1, 1)


@pytest.mark.parametrize("separate_folds", [False, True])
def test_cluster_interval_resamples_whole_events_with_photo_weighting(
    separate_folds: bool,
) -> None:
    rows = tuple(
        Row(
            str(i),
            "",
            str(i),
            "large" if i < 6 else "small",
            2 if separate_folds and i == 6 else 1,
            frozenset(),
        )
        for i in range(7)
    )
    outcomes = np.asarray([[1, 1]] * 6 + [[-1, 1]], dtype=np.int64)

    result = paired_interval(rows, outcomes, FitConfig(bootstrap_resamples=2000))

    expected_bounds = (5 / 7, 5 / 7) if separate_folds else (-1.0, 1.0)
    assert (
        result.lower,
        result.upper,
        result.event_groups,
        result.valid_resamples,
    ) == (*expected_bounds, 2, 2000)


def test_all_normal_rows_have_undefined_recall_and_no_bootstrap() -> None:
    rows = (Row("normal", "", "normal", "event", 1, frozenset()),)
    dataset = Dataset(
        rows,
        np.zeros((1, 16, 2), dtype=np.float32),
        np.zeros((1, len(CODES)), dtype=bool),
    )

    result = summarize(
        dataset, Predictions(np.asarray([0]), np.asarray([1])), FitConfig()
    )

    assert (
        result.global_recall_at_1,
        result.regional_recall_at_1,
        result.delta_regional_minus_global,
    ) == (None, None, None)
    assert (
        result.cluster_percentile_95_ci.valid_resamples,
        result.cluster_percentile_95_ci.lower,
    ) == (0, None)


def test_training_shuffle_preserves_label_vectors_and_original_gold() -> None:
    labels = np.eye(len(CODES), dtype=np.float32)
    original = labels.copy()

    shuffled, permutation = shuffle_training_labels(labels, 20261128)

    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, original))
    assert not np.array_equal(shuffled, original)
    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(shuffled, original[permutation])


@pytest.mark.parametrize("violation", ["event", "source", "fold", "code", "cache"])
def test_input_boundary_rejects_leakage_and_cache_mismatch(
    tmp_path: Path, violation: Literal["event", "source", "fold", "code", "cache"]
) -> None:
    rows = synthetic_rows(tmp_path)
    match violation:
        case "event":
            rows[9] = replace(rows[9], group_id=rows[0].group_id)
        case "source":
            rows[9] = replace(rows[9], sha256=rows[0].sha256)
        case "fold":
            rows[0] = replace(rows[0], fold=0)
        case "code":
            rows[0] = replace(rows[0], codes=frozenset({"invalid"}))
        case "cache":
            rows[0] = replace(rows[0], sha256="not-present")
        case unreachable:
            assert_never(unreachable)

    with pytest.raises(AuditInputError):
        load_dataset(rows, tmp_path, "synthetic")


def test_cpu_cli_writes_aggregate_oof_and_deterministic_control(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    rows = synthetic_rows(cache)
    manifest = tmp_path / "manifest.json"
    summary = tmp_path / "summary.json"
    report = tmp_path / "aggregate.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "sample_id": row.sample_id,
                    "image": row.image,
                    "sha256": row.sha256,
                    "group_id": row.group_id,
                    "dev_fold": row.fold,
                    "codes": sorted(row.codes),
                    "split": "dev",
                }
                for row in rows
            ]
        ),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {"dev_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "kwater_lab.spatial_oof_audit",
        "--manifest",
        str(manifest),
        "--summary",
        str(summary),
        "--cache-dir",
        str(cache),
        "--encoder-id",
        "synthetic",
        "--expected-rows",
        "27",
        "--steps",
        "40",
        "--report",
        str(report),
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )

    first = subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False
    )
    assert first.returncode == 0, first.stderr
    first_bytes = report.read_bytes()
    second = subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False
    )
    assert second.returncode == 0, second.stderr
    result = json.loads(report.read_text())

    assert report.read_bytes() == first_bytes
    assert (
        result["photos"],
        result["observed"]["abnormal_photos"],
        result["observed"]["normal_photos"],
    ) == (27, 18, 9)
    assert (
        result["observed"]["global_recall_at_1"]
        == result["observed"]["regional_recall_at_1"]
        == 1.0
    )
    shuffled = result["train_label_shuffle"]
    counts = shuffled["contingency"]
    assert shuffled["global_hits"] == counts["both_hit"] + counts["global_only"]
    assert shuffled["regional_hits"] == counts["both_hit"] + counts["regional_only"]
    assert sum(counts.values()) == 18
    assert result["observed"]["cluster_percentile_95_ci"]["valid_resamples"] == 2000
    assert result["normal_control"]["classifier_threshold"] is None
    assert result["normal_control"]["false_positive_rate"] is None
    assert all(
        fold["fit_photos"] == 18 and fold["validation_photos"] == 9
        for fold in result["per_fold"]
    )
    assert not any(
        row.sample_id in report.read_text() or row.group_id in report.read_text()
        for row in rows
    )
    (tmp_path / "cli.stdout").write_text(first.stdout + second.stdout, encoding="utf-8")
