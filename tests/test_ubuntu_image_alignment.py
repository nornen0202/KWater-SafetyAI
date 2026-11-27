"""CPU contracts for constrained image reassignment and sealed-input gating."""

import json
from pathlib import Path

import pytest
from test_ubuntu_inner_verifier_eval import prediction_fixture

from kwater_lab import ubuntu_image_alignment as alignment
from kwater_lab.ubuntu_image_alignment import (
    CODES,
    EvaluationError,
    Labels,
    Margin,
    Photo,
    auc,
    build_plan,
    compare,
    score,
    validate_mapping,
)
from kwater_lab.ubuntu_inner_verifier_eval import seal_run, write_json


def photos_with_groups(groups: tuple[str, ...]) -> tuple[Photo, ...]:
    return tuple(
        Photo(str(i), f"{i}.png", f"{i:064x}", group) for i, group in enumerate(groups)
    )


@pytest.mark.parametrize(
    "groups", [("a", "b"), ("a", "a", "b", "b"), ("a", "a", "b", "c", "d")]
)
def test_every_mapping_preserves_all_donors_and_excludes_recipient_group(
    groups: tuple[str, ...],
) -> None:
    photos = photos_with_groups(groups)

    plan = build_plan(photos)

    assert len(plan["donor_indices"]) == 256
    assert plan == build_plan(photos)
    for mapping in plan["donor_indices"]:
        assert sorted(mapping) == list(range(len(groups)))
        assert all(groups[i] != groups[j] for i, j in enumerate(mapping))


@pytest.mark.parametrize("mapping", [[0, 1, 2, 3], [2, 2, 0, 1]])
def test_invalid_donor_mapping_is_rejected(mapping: list[int]) -> None:
    with pytest.raises(EvaluationError):
        validate_mapping(photos_with_groups(("a", "a", "b", "b")), mapping)


def test_majority_group_has_no_whole_cohort_bijection() -> None:
    with pytest.raises(EvaluationError, match="infeasible"):
        build_plan(photos_with_groups(("a", "a", "a", "b")))


def test_constant_probabilities_have_chance_auc_and_unchanged_operating_point() -> None:
    photos = photos_with_groups(tuple(str(i) for i in range(8)))
    labels = Labels(
        tuple(frozenset(CODES) if i < 4 else frozenset() for i in range(8)),
        tuple(frozenset() for _ in range(8)),
    )
    rows = tuple(
        Margin(photo.sample_id, code, 0.0, "a" * 64)
        for photo in photos
        for code in CODES
    )

    result = compare(labels, rows, build_plan(photos))

    assert result["correct_image"]["macro_auc"] == 0.5
    assert result["correct_image"]["approvals_per_code"] == dict.fromkeys(CODES, 0)
    assert result["donor_diagnostic_ranges"]["macro_auc"] == {
        "mean": 0.5,
        "minimum": 0.5,
        "maximum": 0.5,
    }
    assert result["correct_minus_donor"]["pooled_f1"]["mean"] == 0
    assert result["paired_margin_concordance_gold_positive"]["mean"] == 0.5


def test_strong_image_match_signal_loses_auc_f1_and_severe_recall_under_donors() -> (
    None
):
    photos = photos_with_groups(tuple(str(i) for i in range(20)))
    gold = tuple(frozenset(CODES) if i < 10 else frozenset() for i in range(20))
    labels = Labels(gold, gold)
    rows = tuple(
        Margin(photo.sample_id, code, 2.0 if i < 10 else -2.0, "a" * 64)
        for i, photo in enumerate(photos)
        for code in CODES
    )

    result = compare(labels, rows, build_plan(photos))

    assert result["correct_image"]["macro_auc"] == 1
    assert result["correct_image"]["pooled_f1"] == 1
    assert result["correct_image"]["severe_instances"] == 30
    assert result["correct_image"]["severe_misses"] == 0
    assert result["correct_minus_donor"]["macro_auc"]["mean"] > 0.4
    assert result["correct_minus_donor"]["pooled_f1"]["mean"] > 0.4
    assert result["donor_diagnostic_ranges"]["severe_misses"]["mean"] > 12
    assert result["donor_diagnostic_ranges"]["normal_false_alarm_photos"]["mean"] > 4
    assert result["marginal_approvals_preserved_all_replications"]


def test_auc_ties_and_absent_classes_use_explicit_denominators() -> None:
    assert auc([True, True, False, False], [1.0, 0.0, 0.0, -1.0]) == 0.875
    assert auc([True, True], [1.0, -1.0]) is None


@pytest.mark.parametrize("tamper", [False, True])
def test_unsealed_or_tampered_run_fails_before_opening_gold(
    tmp_path: Path, tamper: bool
) -> None:
    run, _, photos = prediction_fixture(tmp_path)
    if tamper:
        seal_run(run, photos)
        path = run / "mined/predictions.json"
        rows = json.loads(path.read_text())
        rows[0]["margin"] = 20.0
        write_json(path, rows)

    with pytest.raises((EvaluationError, FileNotFoundError)) as failure:
        score(run, tmp_path / "missing-plan", tmp_path / "gold-must-not-open")

    if tamper:
        assert "Sealed predictions changed" in str(failure.value)
    else:
        assert Path(failure.value.filename).name == "seal.json"


def test_sealed_score_returns_deidentified_aggregates_with_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, gold, photos = prediction_fixture(tmp_path)
    identity_path = run / "identity.json"
    identity = json.loads(identity_path.read_text())
    source = Path(alignment.__file__).parent
    identity.update(
        {
            "user_text": {code: alignment.verify_user_text(code) for code in CODES},
            "source": {
                name: alignment.digest(source / name)
                for name in ("ubuntu_inner_verifier_eval.py", "vlm_11code.py")
            },
            "prompt_sha256": alignment.digest(source / "system_prompt_11code.txt"),
        }
    )
    write_json(identity_path, identity)
    seal_run(run, photos)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, build_plan(photos))
    monkeypatch.setattr(alignment, "load_cohort", lambda root: photos)
    monkeypatch.setattr(alignment, "GOLD_SHA", alignment.digest(gold))

    result = score(run, plan_path, gold)

    assert result["arms"]["control"]["correct_image"]["pooled_f1"] == pytest.approx(
        2 / 3
    )
    assert result["arms"]["mined"]["correct_image"]["pooled_f1"] == 0.5
    assert result["source_sha256"]["seal.json"] == alignment.digest(run / "seal.json")
    assert all(photo.sample_id not in json.dumps(result) for photo in photos)
