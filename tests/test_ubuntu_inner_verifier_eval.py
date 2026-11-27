"""CPU contracts for paired prediction sealing, restart caches, and actual scoring CLI."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from kwater_lab.spatial_probe import digest
from kwater_lab.ubuntu_inner_verifier_eval import (
    ARMS,
    CODES,
    MANIFEST_SHA,
    MAX_PIXELS,
    PHOTO_COUNT,
    EvaluationError,
    Margin,
    Photo,
    load_cached_photo,
    load_cohort,
    parse_margins,
    score,
    seal_run,
    verify_run,
    write_json,
)


def prediction_fixture(root: Path) -> tuple[Path, Path, tuple[Photo, ...]]:
    run = root / "run"
    gold_path = root / "synthetic-gold.jsonl"
    photos = tuple(
        Photo(f"synthetic-{index}", f"{index}.png", f"{index:064x}", f"event-{index}")
        for index in range(PHOTO_COUNT)
    )
    gold = []
    for index, photo in enumerate(photos):
        code = (
            ("slip_trip", "fall_height", "chemical_exposure")[index]
            if index < 3
            else None
        )
        hazards = {code: {"risk_level": {"severity": {"score": 4}}}} if code else {}
        gold.append(
            {
                "sample_id": photo.sample_id,
                "target": {
                    "status": "hazard" if code else "no_visible_hazard",
                    "hazards": hazards,
                },
            }
        )
    gold_path.write_text("\n".join(json.dumps(row) for row in gold) + "\n")
    write_json(
        run / "identity.json",
        {
            "photos": [asdict(photo) for photo in photos],
            "codes": list(CODES),
            "gold_sha256": digest(gold_path),
            "manifest_sha256": MANIFEST_SHA,
            "max_pixels": MAX_PIXELS,
            "threshold": 0.0,
        },
    )
    for arm in ARMS:
        margins = []
        for index, photo in enumerate(photos):
            for code in CODES:
                accepted = (
                    arm == "control" and index == 0 and code == "slip_trip"
                ) or (
                    arm == "mined"
                    and (
                        (index == 1 and code == "fall_height")
                        or (index == 3 and code == "slip_trip")
                    )
                )
                margins.append(
                    asdict(
                        Margin(
                            photo.sample_id,
                            code,
                            1.0 if accepted else 0.0,
                            f"{index:064x}",
                        )
                    )
                )
        write_json(run / arm / "predictions.json", margins)
    return run, gold_path, photos


def test_cpu_score_cli_reports_candidate_and_all_code_denominators(
    tmp_path: Path,
) -> None:
    run, gold, photos = prediction_fixture(tmp_path)
    seal_run(run, photos)
    report = tmp_path / "aggregate.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kwater_lab.ubuntu_inner_verifier_eval",
            "score",
            "--run",
            str(run),
            "--gold",
            str(gold),
            "--report",
            str(report),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    aggregate = json.loads(report.read_text())
    control, mined = (aggregate["arms"][arm] for arm in ARMS)
    assert (control["tp"], control["fp"], control["fn"]) == (1, 0, 1)
    assert (mined["tp"], mined["fp"], mined["fn"]) == (1, 1, 1)
    assert aggregate["pooled_f1_delta_mined_minus_control"] == pytest.approx(-1 / 6)
    assert (mined["normal_photos"], mined["normal_false_alarm_photos"]) == (198, 1)
    assert mined["normal_false_alarm_rate"] == pytest.approx(1 / 198)
    assert mined["serious4_candidate_codes_primary"] == {
        "instances": 2,
        "hits": 1,
        "misses": 1,
        "recall": 0.5,
    }
    assert mined["serious4_all_codes_secondary"]["sev4_instances"] == 3
    assert mined["serious4_all_codes_secondary"]["sev4_instance_missed"] == 2
    assert aggregate["changed_photos"] == 3
    assert aggregate["changed_decisions"]["slip_trip"] == {
        "reject_to_accept": 1,
        "accept_to_reject": 1,
    }
    assert not any(photo.sample_id in report.read_text() for photo in photos)
    (tmp_path / "cli.stdout").write_text(result.stdout)


def test_unsealed_scoring_never_opens_gold(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as failure:
        score(
            tmp_path / "unsealed",
            tmp_path / "gold-must-not-open.jsonl",
            tmp_path / "report.json",
        )
    assert Path(failure.value.filename).name == "seal.json"


@pytest.mark.parametrize(
    "damage", ["missing_arm", "reordered", "input_mismatch", "nan"]
)
def test_seal_rejects_incomplete_or_unmatched_arm(tmp_path: Path, damage: str) -> None:
    run, _, photos = prediction_fixture(tmp_path)
    path = run / "mined/predictions.json"
    rows = json.loads(path.read_text())
    if damage == "missing_arm":
        path.unlink()
    elif damage == "reordered":
        write_json(path, rows[::-1])
    elif damage == "input_mismatch":
        rows[0]["input_sha256"] = "f" * 64
        write_json(path, rows)
    else:
        rows[0]["margin"] = float("nan")
        path.write_text(json.dumps(rows))

    with pytest.raises((EvaluationError, FileNotFoundError)):
        seal_run(run, photos)
    assert not (run / "seal.json").exists()


def test_sealed_prediction_edit_is_rejected(tmp_path: Path) -> None:
    run, _, photos = prediction_fixture(tmp_path)
    seal_run(run, photos)
    path = run / "control/predictions.json"
    rows = json.loads(path.read_text())
    rows[0]["margin"] = -1.0
    write_json(path, rows)

    with pytest.raises(EvaluationError, match="Sealed predictions changed"):
        verify_run(run)


def test_photo_cache_resumes_only_matching_finite_complete_protocol(
    tmp_path: Path,
) -> None:
    photo = Photo("synthetic", "a.png", "a" * 64, "event")
    rows = [
        Margin(photo.sample_id, code, index - 1.0, "b" * 64).to_record()
        for index, code in enumerate(CODES)
    ]
    path = tmp_path / "cache.json"
    assert load_cached_photo(path, "protocol", photo) is None
    write_json(path, {"cache_key": "protocol", "rows": rows})

    loaded = load_cached_photo(path, "protocol", photo)

    assert loaded is not None
    assert [row.margin for row in loaded] == [-1.0, 0.0, 1.0]
    with pytest.raises(EvaluationError, match="protocol identity"):
        load_cached_photo(path, "different-adapter", photo)
    with pytest.raises(EvaluationError, match="coverage"):
        parse_margins(rows[:-1], (photo,))


def test_noncanonical_manifest_is_rejected_before_cohort_use(tmp_path: Path) -> None:
    path = tmp_path / "data/clean-split-v1/split_manifest_dev.json"
    path.parent.mkdir(parents=True)
    path.write_text("[]")

    with pytest.raises(EvaluationError, match="manifest SHA-256"):
        load_cohort(tmp_path)
