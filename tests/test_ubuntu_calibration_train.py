"""CPU contracts for Fold1-only unique calibration supervision."""

from collections import Counter
from dataclasses import replace

import pytest

from kwater_lab.spatial_probe import Row
from kwater_lab.ubuntu_calibration_train import (
    CODES,
    FocusedError,
    select_curriculum,
)


def cohort() -> tuple[Row, ...]:
    result = []
    for category in (*CODES, "normal", "electrical_energy"):
        for index in range(32):
            name = f"{category}-{index}"
            result.append(
                Row(
                    name,
                    f"{name}.jpg",
                    name,
                    f"group-{name}",
                    1,
                    frozenset() if category == "normal" else frozenset({category}),
                )
            )
    return tuple(result)


def test_curriculum_uses_unique_photos_correct_labels_and_diverse_groups() -> None:
    rows = cohort()
    labels = {row.sample_id: row.codes for row in rows}

    actual = select_curriculum(rows)

    assert len(actual) == len({x.sample_id for x in actual}) == 96
    assert Counter((x.code, x.negative_kind) for x in actual) == {
        (code, kind): count
        for code in CODES
        for kind, count in (("positive", 12), ("normal", 10), ("other_hazard", 10))
    }
    assert all(x.accepted == (x.code in labels[x.sample_id]) for x in actual)
    assert all(
        bool(labels[x.sample_id]) == (x.negative_kind != "normal") for x in actual
    )
    assert len({x.group_id for x in actual}) == 96
    assert actual == select_curriculum(tuple(reversed(rows)))


@pytest.mark.parametrize("field", ["sample_id", "image", "sha256"])
def test_sampling_rejects_duplicate_photo_identity(field: str) -> None:
    rows = list(cohort())
    rows[1] = replace(rows[1], **{field: getattr(rows[0], field)})

    with pytest.raises(FocusedError, match=f"Duplicate Fold1 {field}"):
        select_curriculum(rows)


def test_sampling_rejects_nontraining_fold() -> None:
    rows = list(cohort())
    row = rows[0]
    rows[0] = Row(row.sample_id, row.image, row.sha256, row.group_id, 2, row.codes)

    with pytest.raises(FocusedError, match="Fold1 only"):
        select_curriculum(rows)


def test_sampling_fails_when_positive_quota_would_require_photo_reuse() -> None:
    rows = tuple(
        r
        for r in cohort()
        if "struck_by_lifting" not in r.codes
        or int(r.sample_id.rsplit("-", 1)[-1]) < 11
    )

    with pytest.raises(FocusedError, match="distinct photographs"):
        select_curriculum(rows)
