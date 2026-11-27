"""CPU evidence for leakage boundaries, immutable threshold selection and resume."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from kwater_lab import ubuntu_calibration_protocol as protocol
from kwater_lab import ubuntu_calibration_stats as stats
from kwater_lab import ubuntu_fold_calibration as evaluation
from kwater_lab import ubuntu_inner_verifier_eval as verifier
from kwater_lab import vlm_11code
from kwater_lab.ubuntu_focused_eval import write_once
from kwater_lab.ubuntu_image_alignment import Labels


def stage_fixture(root: Path) -> tuple[Path, Path]:
    run = root / "data/run"
    gold = root / "gold.jsonl"
    cohorts = {}
    gold_records = []
    for fold in (2, 3):
        photos = tuple(
            verifier.Photo(
                f"fold{fold}-{i}",
                f"{fold}-{i}.jpg",
                f"{fold * 1000 + i:064x}",
                f"fold{fold}-group-{max(i - 99, 0)}",
            )
            for i in range(protocol.COUNTS[fold])
        )
        cohorts[str(fold)] = [asdict(photo) for photo in photos]
        rows = []
        for i, photo in enumerate(photos):
            code = verifier.CODES[i] if i < 3 else None
            hazards = {code: {"risk_level": {"severity": {"score": 4}}}} if code else {}
            gold_records.append(
                {"sample_id": photo.sample_id, "target": {"hazards": hazards}}
            )
            for candidate in verifier.CODES:
                margin = -1.0 if candidate == code else -3.0
                if fold == 3 and i == 3 and candidate == verifier.CODES[0]:
                    margin = -0.5
                rows.append(
                    verifier.Margin(
                        photo.sample_id, candidate, margin, "e" * 64
                    ).to_record()
                )
        write_once(run / f"fold{fold}/predictions.json", rows)
    gold.write_text("\n".join(json.dumps(record) for record in gold_records) + "\n")
    write_once(run / "protocol.json", protocol.PROTOCOL)
    write_once(
        run / "identity.json",
        {
            "schema": protocol.SCHEMA,
            "protocol_sha256": verifier.stable_hash(protocol.PROTOCOL),
            "manifest_sha256": verifier.MANIFEST_SHA,
            "gold_sha256": verifier.digest(gold),
            "cohorts": cohorts,
            "source": {
                "ubuntu_calibration_protocol.py": verifier.digest(
                    Path(protocol.__file__)
                )
            },
            "training_receipt_sha256": "a" * 64,
        },
    )
    protocol.seal_stage(run, 2)
    # Fold3's fixture payload is held outside the run until calibration has locked.
    (run / "fold3/predictions.json").rename(root / "fold3-pending.json")
    (run / "fold3").rmdir()
    return run, gold


def complete_fixture(root: Path) -> tuple[Path, Path]:
    run, gold = stage_fixture(root)
    protocol.fit_threshold(run, gold)
    write_once(
        run / "fold3/start.json",
        {"threshold_lock_sha256": verifier.digest(run / "threshold-lock.json")},
    )
    (root / "fold3-pending.json").rename(run / "fold3/predictions.json")
    protocol.seal_stage(run, 3)
    return run, gold


def test_cap_counts_photos_not_condition_alarms_and_ties_choose_higher_threshold() -> (
    None
):
    labels = Labels(
        (frozenset(verifier.CODES),) + (frozenset(),) * 10, (frozenset(),) * 11
    )
    rows = tuple(
        verifier.Margin(
            str(i), code, 3.0 if i == 0 else (2.0 if i == 1 else -1.0), "e" * 64
        )
        for i in range(11)
        for code in verifier.CODES
    )

    selected = stats.select_threshold(labels, rows)

    assert selected["threshold"] == 2.0
    assert selected["fixed_3code_macro_f1"] == 1.0
    assert selected["false_alarm_cap"] == 1
    assert selected["normal_false_alarm_photos"] == 0


def test_nonzero_cap_can_admit_three_code_alarms_on_one_normal_photo() -> None:
    labels = Labels(
        (frozenset(verifier.CODES),) + (frozenset(),) * 10, (frozenset(),) * 11
    )
    rows = tuple(
        verifier.Margin(str(i), code, 2.0 if i in (0, 1) else -1.0, "e" * 64)
        for i in range(11)
        for code in verifier.CODES
    )

    selected = stats.select_threshold(labels, rows)

    assert selected["threshold"] == -1.0
    assert selected["normal_false_alarm_photos"] == 1
    assert selected["fixed_3code_macro_f1"] == pytest.approx(2 / 3)


def test_zero_cap_for_small_normal_cohort_selects_finite_all_reject() -> None:
    labels = Labels((frozenset(verifier.CODES), frozenset()), (frozenset(),) * 2)
    rows = tuple(
        verifier.Margin(str(i), code, 1.0, "e" * 64)
        for i in range(2)
        for code in verifier.CODES
    )

    selected = stats.select_threshold(labels, rows)

    assert selected["threshold"] == selected["all_reject_threshold"] == 1.0
    assert selected["false_alarm_cap"] == 0
    assert selected["normal_false_alarm_photos"] == 0


def test_no_normals_cannot_claim_valid_calibration() -> None:
    labels = Labels((frozenset(verifier.CODES),), (frozenset(),))
    rows = tuple(verifier.Margin("one", code, 1.0, "e" * 64) for code in verifier.CODES)
    with pytest.raises(verifier.EvaluationError, match="at least one normal"):
        stats.select_threshold(labels, rows)


def test_exact_f1_and_alarm_ties_choose_higher_observed_threshold() -> None:
    code = verifier.CODES[0]
    labels = Labels(
        (
            frozenset({code}),
            frozenset({code}),
            frozenset({"chemical_exposure"}),
            frozenset({"chemical_exposure"}),
            frozenset(),
        ),
        (frozenset(),) * 5,
    )
    rows = tuple(
        verifier.Margin(
            str(i),
            candidate,
            (2.0 if i == 0 else 1.0) if i < 4 and candidate == code else -3.0,
            "e" * 64,
        )
        for i in range(5)
        for candidate in verifier.CODES
    )

    selected = stats.select_threshold(labels, rows)

    assert selected["threshold"] == 1.0
    assert selected["fixed_3code_macro_f1"] == pytest.approx(2 / 9)
    assert selected["normal_false_alarm_photos"] == 0


def test_incomplete_or_changed_prediction_seal_fails_before_gold(
    tmp_path: Path,
) -> None:
    run, _ = stage_fixture(tmp_path)
    path = run / "fold2/predictions.json"
    payload = json.loads(path.read_text())
    path.write_text(json.dumps(payload[:-1]))

    with pytest.raises(verifier.EvaluationError, match="Sealed predictions changed"):
        protocol.fit_threshold(run, tmp_path / "gold-must-not-open")

    assert not (run / "threshold-lock.json").exists()


def test_fold3_cannot_start_without_lock_or_read_gold(tmp_path: Path) -> None:
    run, _ = stage_fixture(tmp_path)
    paths = evaluation.Paths(
        tmp_path,
        tmp_path / "missing-training",
        tmp_path / "data",
        tmp_path / "report.json",
    )

    with pytest.raises(FileNotFoundError, match="threshold-lock"):
        evaluation.predict_fold(paths, run, 3)

    assert not (run / "fold3").exists()


def test_existing_fold3_work_blocks_calibration_before_gold(tmp_path: Path) -> None:
    run, _ = stage_fixture(tmp_path)
    (run / "fold3").mkdir()

    with pytest.raises(verifier.EvaluationError, match="Cannot fit after Fold3"):
        protocol.fit_threshold(run, tmp_path / "gold-must-not-open")


def test_locked_threshold_resumes_without_reopening_gold(tmp_path: Path) -> None:
    run, gold = stage_fixture(tmp_path)
    first = protocol.fit_threshold(run, gold)
    original = (run / "threshold-lock.json").read_bytes()

    resumed = protocol.fit_threshold(run, tmp_path / "gold-must-not-open")

    assert resumed == first
    assert first["selection"]["threshold"] == -3.0
    assert (run / "threshold-lock.json").read_bytes() == original


def test_complete_fold_resume_uses_seal_without_loading_model(tmp_path: Path) -> None:
    run, _ = complete_fixture(tmp_path)
    before = {path: path.read_bytes() for path in run.rglob("*.json")}
    paths = evaluation.Paths(
        tmp_path,
        tmp_path / "nonexistent-model",
        tmp_path / "data",
        tmp_path / "report.json",
    )

    evaluation.predict_fold(paths, run, 3)

    assert {path: path.read_bytes() for path in run.rglob("*.json")} == before


def test_unsealed_complete_photo_caches_resume_without_model_and_preserve_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = stage_fixture(tmp_path)
    identity = json.loads((run / "identity.json").read_text())
    prompt = tmp_path / "kwater_lab/system_prompt_11code.txt"
    prompt.parent.mkdir()
    prompt.write_text("synthetic prompt")
    identity.update(
        adapter={},
        base={},
        decision_ids=[2, 3],
        prompt_sha256=verifier.digest(prompt),
        processor_sha256=verifier.stable_hash("processor"),
    )
    (run / "identity.json").write_text(json.dumps(identity))
    rows = json.loads((run / "fold2/predictions.json").read_text())
    for index, photo in enumerate(protocol.photos_for(identity, 2)):
        key = verifier.stable_hash((verifier.stable_hash(identity), 2, asdict(photo)))
        block = rows[index * 3 : (index + 1) * 3]
        write_once(
            run / "fold2/cache" / f"{key}.json",
            {
                "cache_key": key,
                "rows": block,
                "rows_sha256": verifier.stable_hash(block),
            },
        )
    (run / "fold2/predictions.json").unlink()
    (run / "fold2/seal.json").unlink()

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if text.endswith("accept"):
                return [1, 2]
            if text.endswith("reject"):
                return [1, 3]
            return [1]

    processor = SimpleNamespace(
        tokenizer=Tokenizer(),
        image_processor=SimpleNamespace(to_json_string=lambda: "processor"),
    )
    monkeypatch.setattr(vlm_11code, "load_qwen_processor", lambda *_: processor)

    def forbid_model(*args, **kwargs):
        raise AssertionError("A complete cached cohort must not load a GPU model")

    monkeypatch.setattr(vlm_11code, "load_qwen_model", forbid_model)
    paths = evaluation.Paths(
        tmp_path, tmp_path / "missing", tmp_path / "data", tmp_path / "report.json"
    )

    evaluation.predict_fold(paths, run, 2)

    assert json.loads((run / "fold2/predictions.json").read_text()) == rows
    assert len(protocol.verify_stage(run, 2)) == 197


def test_changed_lock_is_rejected_before_fold3_gold(tmp_path: Path) -> None:
    run, _ = complete_fixture(tmp_path)
    path = run / "threshold-lock.json"
    lock = json.loads(path.read_text())
    lock["selection"]["threshold"] = -1.0
    path.write_text(json.dumps(lock))

    with pytest.raises(
        verifier.EvaluationError, match="Threshold lock selection checksum changed"
    ):
        protocol.load_labels(run, 3, tmp_path / "gold-must-not-open")


def test_synthetic_end_to_end_score_uses_actual_clusters_and_exports_no_photo_ids(
    tmp_path: Path,
) -> None:
    run, gold = complete_fixture(tmp_path)
    report_path = tmp_path / "aggregate.json"

    evaluation.score(run, gold, report_path)

    report = json.loads(report_path.read_text())
    fixed = report["arms"]["fixed_zero"]
    calibrated = report["arms"]["calibrated"]
    assert fixed["fixed_3code_macro_f1"] == 0.0
    assert calibrated["fixed_3code_macro_f1"] == pytest.approx(8 / 9)
    assert (calibrated["normal_photos"], calibrated["normal_false_alarm_photos"]) == (
        198,
        1,
    )
    assert fixed["severity4_candidate_complete_miss_photos"] == 3
    assert calibrated["severity4_candidate_complete_miss_photos"] == 0
    assert all(
        row["recovered_true_positives"] == 1
        for row in report["changes_by_code"].values()
    )
    assert report["paired_group_bootstrap"]["resamples"] == 2000
    assert report["paired_group_bootstrap"]["groups"] == 102
    assert report["largest_group_excluded_secondary"]["excluded_photos"] == 100
    assert report["secondary_threshold_free_auc"][verifier.CODES[0]] == pytest.approx(
        199 / 200
    )
    assert all(
        json.dumps(f"fold{fold}-{i}") not in report_path.read_text()
        for fold, count in protocol.COUNTS.items()
        for i in range(count)
    )


def test_crossfold_group_leak_is_rejected_before_image_or_gold_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "data/clean-split-v1/split_manifest_dev.json"
    manifest.parent.mkdir(parents=True)
    records = [
        {
            "split": "dev",
            "dev_fold": fold,
            "sample_id": f"{fold}-{i}",
            "image": f"{fold}-{i}.jpg",
            "sha256": f"{fold * 1000 + i:064x}",
            "group_id": "shared-event" if i == 0 else f"group-{fold}-{i}",
        }
        for fold, count in protocol.COUNTS.items()
        for i in range(count)
    ]
    manifest.write_text(json.dumps(records))
    monkeypatch.setattr(verifier, "MANIFEST_SHA", verifier.digest(manifest))

    with pytest.raises(
        verifier.EvaluationError, match="Protected fold overlap: group_id"
    ):
        protocol.load_cohorts(tmp_path)
