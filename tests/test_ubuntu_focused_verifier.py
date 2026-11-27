"""CPU contracts for balanced supervision, causal loss, and interrupted step logs."""

import json
import math
from collections import Counter
from pathlib import Path

import pytest
import torch
from transformers import get_cosine_schedule_with_warmup

from kwater_lab import ubuntu_focused_curriculum as focused
from kwater_lab.spatial_probe import Row, digest
from kwater_lab.ubuntu_focused_curriculum import (
    CODES,
    CONTROL_HORIZON,
    CONTROL_WARMUP,
    LR,
    FocusedError,
    select_curriculum,
)
from kwater_lab.ubuntu_focused_verifier import decision_loss, restore_step_log


def cohort() -> list[Row]:
    result = []
    for category in (*CODES, "normal", "electrical_energy"):
        for index in range(40):
            name = f"{category}-{index}"
            result.append(
                Row(
                    name,
                    f"{name}.jpg",
                    name,
                    f"group-{name}",
                    index % 2 + 1,
                    frozenset() if category == "normal" else frozenset({category}),
                )
            )
    return result


def test_curriculum_balances_true_codes_with_distinct_photos_and_groups() -> None:
    rows = cohort()
    labels = {row.sample_id: row.codes for row in rows}

    actual = select_curriculum(rows)

    assert len(actual) == len({item.sample_id for item in actual}) == 96
    assert Counter((item.code, item.negative_kind) for item in actual) == {
        (code, kind): count
        for code in CODES
        for kind, count in (("positive", 16), ("normal", 8), ("other_hazard", 8))
    }
    assert all(
        item.accepted == (item.code in labels[item.sample_id]) for item in actual
    )
    assert all(
        bool(labels[item.sample_id]) == (item.negative_kind != "normal")
        for item in actual
    )
    assert len({item.group_id for item in actual}) == 96
    assert actual == select_curriculum(tuple(reversed(rows)))


def test_curriculum_refuses_filling_scarce_positives_by_photo_repetition() -> None:
    rows = [
        row
        for row in cohort()
        if "struck_by_lifting" not in row.codes
        or int(row.sample_id.rsplit("-", 1)[-1]) < 15
    ]

    with pytest.raises(FocusedError, match="distinct photographs"):
        select_curriculum(rows)


@pytest.mark.parametrize(
    "corruption", ["valid", "raw_code", "protected_group", "duplicate_raw"]
)
def test_training_boundary_checks_raw_codes_and_protected_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    rows = [
        {
            "sample_id": f"photo-{index}",
            "image": f"photo-{index}.jpg",
            "sha256": f"hash-{index}",
            "group_id": f"group-{index}",
            "dev_fold": index % 2 + 1,
            "split": "dev",
            "codes": ["slip_trip"],
        }
        for index in range(374)
    ]
    protected = {
        "sample_id": "protected",
        "image": "protected.jpg",
        "sha256": "protected-hash",
        "group_id": "group-0" if corruption == "protected_group" else "protected-group",
        "dev_fold": 3,
        "split": "dev",
    }
    manifest = tmp_path / focused.MANIFEST
    canonical = tmp_path / "data/clean-split-v1/split_manifest_dev.json"
    raw = tmp_path / "data/train/train.jsonl"
    for path in (manifest, canonical, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows))
    canonical.write_text(json.dumps([*rows, protected]))
    raw_rows = [
        {"sample_id": row["sample_id"], "target": {"hazards": {"slip_trip": {}}}}
        for row in rows
    ]
    if corruption == "raw_code":
        raw_rows[0] = {
            "sample_id": "photo-0",
            "target": {"hazards": {"fall_height": {}}},
        }
    if corruption == "duplicate_raw":
        raw_rows.append(raw_rows[0])
    raw.write_text("\n".join(json.dumps(row) for row in raw_rows))
    monkeypatch.setattr(focused, "MANIFEST_SHA256", digest(manifest))
    monkeypatch.setattr(focused, "MANIFEST_SHA", digest(canonical))

    if corruption == "valid":
        assert len(focused.load_training_rows(tmp_path)) == 374
    else:
        with pytest.raises(FocusedError, match="raw|Protected-fold"):
            focused.load_training_rows(tmp_path)


@pytest.mark.parametrize("target", [2, 4], ids=["accept", "reject"])
def test_decision_loss_only_backpropagates_the_last_conditioned_position(
    target: int,
) -> None:
    logits = torch.zeros((1, 7, 6), requires_grad=True)

    loss = decision_loss(logits, target)
    loss.backward()

    expected = torch.zeros_like(logits)
    expected[:, -1, :] = 1 / 6
    expected[:, -1, target] -= 1
    torch.testing.assert_close(logits.grad, expected)
    assert float(loss.detach()) == pytest.approx(math.log(6))


def test_focused_lr_preserves_control_first_twelve_updates() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=LR)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, CONTROL_WARMUP, CONTROL_HORIZON
    )

    actual = []
    for _ in range(12):
        actual.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    expected = [
        LR * step / 4
        if step < 4
        else LR * 0.5 * (1 + math.cos(math.pi * (step - 4) / 59))
        for step in range(12)
    ]
    assert actual == pytest.approx(expected)


def test_resume_log_discards_updates_after_last_durable_checkpoint(
    tmp_path: Path,
) -> None:
    (tmp_path / "steps.jsonl").write_text('{"step":1}\n{"step":2}\n{"step":3')

    restore_step_log(tmp_path, [{"step": 1}])

    assert [
        json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ] == [{"step": 1}]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["steps.jsonl"]
