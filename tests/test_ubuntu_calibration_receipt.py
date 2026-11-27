"""Executed-step receipt validation rejects identity and optimizer drift."""

import math

import pytest

from kwater_lab.ubuntu_calibration_receipt import verify_step_records
from kwater_lab.ubuntu_focused_curriculum import Decision, FocusedError
from kwater_lab.ubuntu_focused_verifier import StepRecord


@pytest.mark.parametrize(
    "corruption",
    ["valid", "reordered_photo", "step", "count", "target", "lr", "nan", "truncated"],
)
def test_step_receipt_requires_complete_ordered_finite_control_updates(
    corruption: str,
) -> None:
    curriculum = tuple(
        Decision(
            str(index), f"{index}.jpg", str(index), 1, "slip_trip", True, "positive"
        )
        for index in range(96)
    )
    records: list[StepRecord] = []
    for step in range(12):
        factor = (
            step / 4 if step < 4 else 0.5 * (1 + math.cos(math.pi * (step - 4) / 59))
        )
        records.append(
            {
                "step": step + 1,
                "examples": 8,
                "supervised_tokens": 8,
                "sample_ids": [str(index) for index in range(step * 8, (step + 1) * 8)],
                "accept": 8,
                "reject": 0,
                "applied_lr": [0.00014 * factor, 0.000091 * factor],
                "loss": 0.6,
                "grad_norm": 0.4,
            }
        )
    if corruption == "reordered_photo":
        records[0]["sample_ids"] = [str(index) for index in reversed(range(8))]
    if corruption == "step":
        records[3]["step"] = 3
    if corruption == "count":
        records[5]["supervised_tokens"] = 7
    if corruption == "target":
        records[2]["accept"] = 7
    if corruption == "lr":
        records[8]["applied_lr"] = [0.00014, 0.000091]
    if corruption == "nan":
        records[1]["loss"] = math.nan
    if corruption == "truncated":
        records.pop()

    if corruption == "valid":
        verify_step_records(curriculum, records)
    else:
        with pytest.raises(FocusedError):
            verify_step_records(curriculum, records)
