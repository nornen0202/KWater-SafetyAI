"""Cache identity and complete sealing contracts for full-field generation."""

import json
from pathlib import Path

import pytest

from kwater_lab import ubuntu_fold3_generation as baseline
from kwater_lab import ubuntu_fold3_generation_protocol as protocol
from kwater_lab.ubuntu_fold3_generation_protocol import (
    BaselineError,
    load_record,
    seal,
    stable_hash,
    verify_seal,
    write_once,
)


def cohort_run(tmp_path: Path):
    rows = [
        {
            "sample_id": f"photo-{n}",
            "image_id": f"{n:032x}",
            "image": f"images/{n}.png",
            "sha256": "a" * 64,
            "group_id": f"event-{n}",
            "input_identity": f"{n:064x}",
        }
        for n in range(201)
    ]
    plan = {
        "schema": "fold3-official-fullfield-v1",
        "protocol": {},
        "protocol_sha256": stable_hash({}),
        "rows": rows,
    }
    write_once(tmp_path / "plan.json", plan)
    for row in rows:
        payload = {
            "text": "invalid JSON retained",
            "prediction": {},
            "parse_error": "invalid_json",
            "schema_errors": [],
            "normal_fallback": True,
        }
        write_once(
            tmp_path / "records" / f"{row['image_id']}.json",
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "input_identity": row["input_identity"],
                "payload": payload,
                "payload_sha256": stable_hash(payload),
            },
        )
    return plan


def test_same_cache_retains_raw_failure_without_rewriting(tmp_path: Path) -> None:
    plan = cohort_run(tmp_path)
    row = plan["rows"][0]
    path = tmp_path / "records" / f"{row['image_id']}.json"
    before = path.stat().st_mtime_ns

    record = load_record(tmp_path, row)
    write_once(path, record)

    assert record["payload"]["text"] == "invalid JSON retained"
    assert record["payload"]["normal_fallback"] is True
    assert path.stat().st_mtime_ns == before


def test_cache_refuses_different_input_identity(tmp_path: Path) -> None:
    plan = cohort_run(tmp_path)
    row = {**plan["rows"][0], "input_identity": "changed"}

    with pytest.raises(BaselineError, match="identity"):
        load_record(tmp_path, row)


def test_payload_tampering_is_detected_before_seal(tmp_path: Path) -> None:
    plan = cohort_run(tmp_path)
    path = tmp_path / "records" / f"{plan['rows'][0]['image_id']}.json"
    record = json.loads(path.read_text())
    record["payload"]["normal_fallback"] = False
    path.write_text(json.dumps(record))

    with pytest.raises(BaselineError, match="payload"):
        seal(tmp_path, plan)

    assert not (tmp_path / "sealed.json").exists()


def test_missing_last_photo_cannot_be_sealed(tmp_path: Path) -> None:
    plan = cohort_run(tmp_path)
    (tmp_path / "records" / f"{plan['rows'][-1]['image_id']}.json").unlink()

    with pytest.raises(BaselineError, match="missing"):
        seal(tmp_path, plan)

    assert not (tmp_path / "sealed.json").exists()


def test_full_seal_rejects_later_record_change(tmp_path: Path) -> None:
    plan = cohort_run(tmp_path)
    seal(tmp_path, plan)
    assert verify_seal(tmp_path) == plan
    path = tmp_path / "records" / f"{plan['rows'][0]['image_id']}.json"
    path.write_text(path.read_text() + " ")

    with pytest.raises(BaselineError, match="record changed"):
        verify_seal(tmp_path)


def test_write_once_preserves_conflicting_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_once(path, {"value": "first"})

    with pytest.raises(BaselineError, match="differing"):
        write_once(path, {"value": "replacement"})

    assert json.loads(path.read_text()) == {"value": "first"}


def test_official_parser_formatter_checker_preserve_failure_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cohort_run(tmp_path)
    plan["protocol"]["role"] = "Synthetic CPU contract check; no real predictions"
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    entry = baseline.load_entry()
    normal = {
        "scene_state": "no_visible_hazard",
        "hazards": [],
        "scene_note": "확인되는 위험 없음",
    }
    hazard = {
        "scene_state": "hazard",
        "scene_note": "미끄러짐 위험",
        "hazards": [
            {
                "code": "slip_trip",
                "evidence": "바닥의 장애물",
                "prevention": "장애물을 정리한다",
                "likelihood_score": 3,
                "severity_score": 2,
            }
        ],
    }
    for index, row in enumerate(plan["rows"]):
        raw = json.dumps(normal if index == 0 else hazard) if index < 2 else "not JSON"
        parsed, error = entry.parse_prediction(raw)
        errors = [] if error else entry.schema_errors(parsed)
        prediction = {} if error or errors else parsed
        payload = {
            "text": raw,
            "prediction": prediction,
            "parsed_prediction": parsed,
            "parse_error": error,
            "schema_errors": errors,
            "normal_fallback": bool(entry.validate_prediction(prediction)),
            "seconds": 0.01,
            "truncated_at_cap": False,
        }
        path = tmp_path / "records" / f"{row['image_id']}.json"
        path.unlink()
        write_once(
            path,
            {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "input_identity": row["input_identity"],
                "payload": payload,
                "payload_sha256": stable_hash(payload),
            },
        )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "manifest.jsonl").write_text(
        "".join(
            json.dumps({"image_id": row["image_id"]}) + "\n" for row in plan["rows"]
        )
    )
    seal(tmp_path, plan)
    monkeypatch.setattr(protocol, "verify_plan", lambda run: plan)

    protocol.export(tmp_path)

    summary = json.loads((tmp_path / "aggregate.json").read_text())
    source = json.loads((tmp_path / "candidate-source.json").read_text())["rows"]
    assert summary["checker_valid_rows"] == 201
    assert summary["normal_fallbacks"] == 199
    assert summary["genuine_no_visible_hazard"] == 1
    assert source[1]["codes"] == ["slip_trip"]
    assert (
        source[1]["prediction"]["hazards"]["slip_trip"]["risk_level"]["risk"]["score"]
        == 6
    )
    assert source[2]["normal_fallback"] is True
    assert "201/201" in (tmp_path / "checker.log").read_text()


def test_production_cli_rejects_public_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "baseline",
            "prepare",
            "--run",
            str(baseline.ROOT / "results/forbidden-baseline-test"),
        ],
    )

    with pytest.raises(SystemExit) as rejected:
        baseline.main()

    assert rejected.value.code == 2
    assert not (tmp_path / "plan.json").exists()


def test_stop_marker_retains_pending_photo_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = cohort_run(tmp_path)
    last = tmp_path / "records" / f"{plan['rows'][-1]['image_id']}.json"
    last.unlink()
    (tmp_path / ".runtime").mkdir()
    (tmp_path / ".runtime/WEEKEND_STOP").touch()
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    monkeypatch.setattr(baseline, "verify_plan", lambda run: plan)

    baseline.predict(tmp_path)

    assert not last.exists()
    assert not (tmp_path / "sealed.json").exists()
