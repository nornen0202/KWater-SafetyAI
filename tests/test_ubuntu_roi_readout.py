"""Protect paired-control accounting and the no-gold-before-seal boundary."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from kwater_lab.ubuntu_roi_readout import (
    ARMS,
    CODES,
    ROOT,
    ContractError,
    Photo,
    Prediction,
    caption_pairs,
    check_grid,
    classification_metrics,
    digest,
    paired_summary,
    stable_hash,
    verify_seal,
    write_json,
)


def test_fixed_denominator_preserves_absent_codes_and_true_normal_definition() -> None:
    gold = [
        frozenset({"slip_trip"}),
        frozenset({"slip_trip", "fall_height"}),
        frozenset({"other"}),
        frozenset(),
    ]
    predicted = [
        frozenset({"slip_trip", "fall_height"}),
        frozenset({"fall_height"}),
        frozenset({"slip_trip"}),
        frozenset(),
    ]

    result = classification_metrics(gold, predicted)

    assert result["fixed_3code_macro_f1"] == pytest.approx(7 / 18)
    assert result["active_code_macro_f1"] == pytest.approx(7 / 12)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 2, 1)
    assert result["precision"] == 0.5
    assert result["normal_photos"] == 1
    assert result["normal_false_alarm_photos"] == 0
    assert result["positive_photo_coverage"] == 0.75


def test_normal_false_alarms_count_photos_not_predicted_codes() -> None:
    result = classification_metrics(
        [frozenset(), frozenset()], [frozenset(CODES), frozenset()]
    )

    assert result["normal_false_alarm_photos"] == 1
    assert result["normal_false_alarm_rate"] == 0.5
    assert result["active_code_macro_f1"] is None
    assert result["fp"] == 3


def test_caption_donors_are_deranged_and_actual_qwen_token_lengths_match() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "data/baseline/base_model", local_files_only=True
    )
    captions = [
        "A crane lifts a load.",
        "A person walks.",
        "A red vehicle stands near a long concrete wall.",
        "There is a ladder beside the building.",
    ]

    pairs = caption_pairs(captions, tokenizer)

    assert sorted(donor for donor, _, _ in pairs) == [0, 1, 2, 3]
    for index, (donor, true, swapped) in enumerate(pairs):
        assert index != donor
        assert true != swapped
        assert len(tokenizer.encode(true, add_special_tokens=False)) == len(
            tokenizer.encode(swapped, add_special_tokens=False)
        )


@pytest.mark.parametrize(
    "grid,tokens",
    [
        ([[1, 24, 24]], 288),
        ([[1, 24, 24], [1, 32, 32]], 288),
        ([[1, 24, 24], [1, 24, 24]], 512),
    ],
)
def test_visual_parity_rejects_different_view_counts_geometry_or_budget(
    grid: list[list[int]], tokens: int
) -> None:
    with pytest.raises(ContractError, match="Unexpected visual budget"):
        check_grid(grid, tokens)


def test_six_group_sign_test_excludes_ties_from_binomial_denominator() -> None:
    result = paired_summary([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

    assert (result["positive"], result["negative"], result["ties"]) == (3, 0, 3)
    assert result["two_sided_sign_p"] == 0.25


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "nonfinite", "bytes"])
def test_seal_rejects_incomplete_duplicate_nonfinite_or_edited_predictions(
    tmp_path: Path, corruption: str
) -> None:
    photos = [
        Photo(f"photo-{i}", "unused.png", f"{i:064x}", f"group-{i // 4}")
        for i in range(24)
    ]
    records = [
        asdict(
            Prediction(
                p.sample_id,
                arm,
                code,
                1.0,
                "hash",
                [[1, 24, 24]] * 2,
                288,
                999,
                False,
                1.0,
            )
        )
        for p in photos
        for arm in ARMS
        for code in CODES
    ]
    identity = {"fixture": "sealed completeness contract"}
    if corruption == "duplicate":
        records[-1] = records[0]
    if corruption == "missing":
        records.pop()
    payload = {
        "identity_sha256": stable_hash(identity),
        "photos": [asdict(p) for p in photos],
        "predictions": records,
    }
    write_json(tmp_path / "identity.json", identity)
    write_json(tmp_path / "predictions.json", payload)
    if corruption == "nonfinite":
        records[0]["margin"] = float("nan")
        (tmp_path / "predictions.json").write_text(json.dumps(payload))
    write_json(
        tmp_path / "seal.json",
        {
            "predictions_sha256": digest(tmp_path / "predictions.json"),
            "identity_sha256": digest(tmp_path / "identity.json"),
        },
    )
    if corruption == "bytes":
        with (tmp_path / "predictions.json").open("a") as stream:
            stream.write(" ")

    with pytest.raises(ContractError):
        verify_seal(tmp_path)
