"""CPU-only sealed reuse, atomic cache publication and paired group statistics."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from test_ubuntu_inner_verifier_eval import prediction_fixture

from kwater_lab import ubuntu_focused_eval as focused
from kwater_lab import ubuntu_inner_verifier_eval as verifier
from kwater_lab import vlm_11code
from kwater_lab.ubuntu_image_alignment import Labels


def prepared_fixture(root: Path) -> tuple[Path, Path, Path, tuple[verifier.Photo, ...]]:
    source, gold, photos = prediction_fixture(root)
    identity = json.loads((source / "identity.json").read_text())
    identity["adapters"] = {
        "control": dict.fromkeys(focused.ADAPTER_FILES, "a" * 64),
        "mined": dict.fromkeys(focused.ADAPTER_FILES, "b" * 64),
    }
    verifier.write_json(source / "identity.json", identity)
    verifier.seal_run(source, photos)
    expected = json.loads(json.dumps(identity))
    expected["adapters"]["mined"] = dict.fromkeys(focused.ADAPTER_FILES, "c" * 64)
    run = focused.prepare_cache(source, root / "focused", expected)
    return source, run, gold, photos


def complete_fixture(root: Path) -> tuple[Path, Path, Path, tuple[verifier.Photo, ...]]:
    source, run, gold, photos = prepared_fixture(root)
    for arm in verifier.ARMS:
        verifier.write_json(
            run / arm / "predictions.json",
            json.loads((source / arm / "predictions.json").read_text()),
        )
    verifier.seal_run(run, photos)
    return source, run, gold, photos


def test_verified_mixed_margins_materialize_as_complete_exact_predictor_caches(
    tmp_path: Path,
) -> None:
    source, run, _, photos = prepared_fixture(tmp_path)
    expected = focused.read_identity(run / "identity.json")
    fingerprint = verifier.stable_hash(expected)
    _, _, arms = verifier.verify_run(source)

    resumed = focused.prepare_cache(source, run.parent, expected)

    loaded = []
    for photo in photos:
        key = verifier.stable_hash((fingerprint, "control", asdict(photo)))
        rows = verifier.load_cached_photo(
            run / "control/cache" / f"{key}.json", key, photo
        )
        assert rows is not None
        loaded.extend(rows)
    assert resumed == run
    assert tuple(loaded) == arms["control"]
    assert len(loaded) == 603
    assert not (run / "mined").exists()


def test_protocol_drift_is_rejected_before_materializing_any_control_cache(
    tmp_path: Path,
) -> None:
    source, run, _, _ = prepared_fixture(tmp_path)
    expected = focused.read_identity(run / "identity.json")
    expected["max_pixels"] = 999
    out = tmp_path / "drifted"

    with pytest.raises(verifier.EvaluationError, match="protocol drift"):
        focused.prepare_cache(source, out, expected)

    assert not out.exists()


def test_tampered_source_is_rejected_before_cache_reuse(tmp_path: Path) -> None:
    source, run, _, _ = prepared_fixture(tmp_path)
    expected = focused.read_identity(run / "identity.json")
    path = source / "control/predictions.json"
    rows = json.loads(path.read_text())
    rows[0]["margin"] = 44
    verifier.write_json(path, rows)

    with pytest.raises(verifier.EvaluationError, match="Sealed predictions changed"):
        focused.prepare_cache(source, tmp_path / "tampered", expected)

    assert not (tmp_path / "tampered").exists()


def test_atomic_writer_preserves_existing_file_and_publishes_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cache.json"
    focused.write_once(path, {"first": 1})
    with pytest.raises(verifier.EvaluationError, match="overwrite"):
        focused.write_once(path, {"second": 2})
    assert json.loads(path.read_text()) == {"first": 1}
    new_path = tmp_path / "interrupted.json"

    def interrupt_link(source: str, destination: Path) -> None:
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(focused.os, "link", interrupt_link)
    with pytest.raises(OSError, match="interrupted publication"):
        focused.write_once(new_path, {"complete": 42})

    assert not new_path.exists()
    assert not list(tmp_path.glob(".stage-*"))


def test_resealed_changed_control_is_rejected_before_gold(tmp_path: Path) -> None:
    source, run, _, photos = complete_fixture(tmp_path)
    for arm in verifier.ARMS:
        path = run / arm / "predictions.json"
        rows = json.loads(path.read_text())
        rows[0]["margin"] = 999
        verifier.write_json(path, rows)
    verifier.seal_run(run, photos)

    with pytest.raises(verifier.EvaluationError, match="Mixed control margins"):
        focused.score(source, run, tmp_path / "gold-must-not-open")


def test_score_cli_renames_arms_and_keeps_same_denominators(tmp_path: Path) -> None:
    source, run, gold, photos = complete_fixture(tmp_path)
    report = tmp_path / "aggregate.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kwater_lab.ubuntu_focused_eval",
            "score",
            "--source",
            str(source),
            "--run",
            str(run),
            "--gold",
            str(gold),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
    )

    assert result.returncode == 0, result.stderr
    aggregate = json.loads(report.read_text())
    mixed, candidate = (aggregate["arms"][key] for key in ("mixed", "focused"))
    assert mixed["fixed_3code_macro_f1"] == pytest.approx(1 / 3)
    assert candidate["fixed_3code_macro_f1"] == pytest.approx(1 / 3)
    assert (mixed["normal_photos"], candidate["normal_photos"]) == (198, 198)
    assert (
        mixed["normal_false_alarm_photos"],
        candidate["normal_false_alarm_photos"],
    ) == (0, 1)
    assert (
        mixed["severity4_candidate_photos"]
        == candidate["severity4_candidate_photos"]
        == 2
    )
    assert (
        mixed["severity4_candidate_complete_miss_photos"]
        == candidate["severity4_candidate_complete_miss_photos"]
        == 1
    )
    assert aggregate["paired_group_bootstrap"]["delta_focused_minus_mixed"] == 0
    assert all(photo.sample_id not in report.read_text() for photo in photos)


def test_group_bootstrap_preserves_cluster_multiplicity_and_paired_differences() -> (
    None
):
    photos = tuple(
        verifier.Photo(str(i), "x", "a" * 64, "large" if i < 94 else "small")
        for i in range(201)
    )
    gold = tuple(frozenset(verifier.CODES) for _ in photos)
    labels = Labels(gold, gold)
    mixed = [frozenset() for _ in photos]
    candidate = [
        frozenset(verifier.CODES) if i < 94 else frozenset() for i in range(201)
    ]

    result = focused.paired_bootstrap(photos, labels, (mixed, candidate))

    assert result["groups"] == 2
    assert result["resamples"] == 2000
    assert result["delta_focused_minus_mixed"] == pytest.approx(188 / 295)
    assert result["percentile_95"] == [0, 1]
    assert result == focused.paired_bootstrap(photos, labels, (mixed, candidate))


def test_largest_group_exclusion_is_secondary_and_untuned() -> None:
    photos = tuple(
        verifier.Photo(str(i), "x", "a" * 64, "large" if i < 94 else f"event-{i}")
        for i in range(201)
    )
    gold = tuple(frozenset(verifier.CODES) for _ in photos)
    labels = Labels(gold, gold)
    mixed = [frozenset() for _ in photos]
    candidate = [
        frozenset(verifier.CODES) if i < 94 else frozenset() for i in range(201)
    ]

    result = focused.largest_group_sensitivity(photos, labels, (mixed, candidate))

    assert result["retained_photos"] == 107
    assert result["excluded_photos"] == 94
    assert result["delta_focused_minus_mixed"] == 0


def test_existing_predictor_uses_rekeyed_mixed_caches_without_loading_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, gold, photos = prediction_fixture(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    control = tmp_path / "mixed/checkpoint-12"
    candidate = tmp_path / "focused/checkpoint-12"
    for path in (control, candidate):
        path.mkdir(parents=True)
        for name in focused.ADAPTER_FILES:
            (path / name).write_text(path.parent.name)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Synthetic system")
    paths = focused.Paths(
        source, tmp_path / "data/focused-eval", base, control, candidate, prompt
    )

    class Processor:
        @property
        def tokenizer(self):
            return self

        @property
        def image_processor(self):
            return self

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return (
                [1]
                if text == focused.PREFIX
                else [1, 2 if text.endswith("accept") else 3]
            )

        def to_json_string(self) -> str:
            return "{}"

    def forbidden_model_load(*args, **kwargs):
        pytest.fail("Complete reused caches must not load mixed or focused weights")

    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "load_cohort", lambda root: photos)
    monkeypatch.setattr(verifier, "GOLD_SHA", verifier.digest(gold))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "synthetic-cpu")
    monkeypatch.setattr(
        vlm_11code, "load_qwen_processor", lambda base, pixels: Processor()
    )
    monkeypatch.setattr(vlm_11code, "load_qwen_model", forbidden_model_load)
    expected = focused.runtime_identity(paths)
    original = json.loads(json.dumps(expected))
    original["adapters"]["mined"] = dict.fromkeys(focused.ADAPTER_FILES, "b" * 64)
    verifier.write_json(source / "identity.json", original)
    verifier.seal_run(source, photos)
    run = focused.prepare_cache(source, paths.out, expected)
    fingerprint = verifier.stable_hash(expected)
    _, _, source_arms = verifier.verify_run(source)
    for index, photo in enumerate(photos):
        key = verifier.stable_hash((fingerprint, "mined", asdict(photo)))
        rows = [
            row.to_record() for row in source_arms["mined"][index * 3 : (index + 1) * 3]
        ]
        focused.write_once(
            run / "mined/cache" / f"{key}.json",
            {
                "cache_key": key,
                "rows": rows,
                "rows_sha256": verifier.stable_hash(rows),
            },
        )

    actual = focused.predict(paths)

    assert actual == run
    _, actual_arms = focused.verify_comparison(source, actual)
    assert actual_arms["control"] == source_arms["control"]
    assert actual_arms["mined"] == source_arms["mined"]
