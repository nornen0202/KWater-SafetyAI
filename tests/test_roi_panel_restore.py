"""Restore the blinded packet from a tiny key without feature archives."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.prepare_vision_roi_panel import PanelError, RestoreContract, restore_from_key


def private_fixture(root: Path) -> tuple[Path, RestoreContract]:
    train = root / "train"
    train.mkdir()
    rows = []
    for index in range(1, 49):
        path = train / f"image-{index}.png"
        Image.new("RGB", (3, 2), (index, 100, 200)).save(path)
        rows.append(
            {
                "id": f"R{index:02}",
                "family": "slip_trip",
                "role": "SECRET_GOLD_ROLE",
                "source": {
                    "image": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sample_id": "SECRET_CANONICAL_ID",
                    "codes": ["SECRET_GOLD_CODE"],
                },
            }
        )
    key = root / "key.json"
    key.write_text(json.dumps(rows) + " \n", encoding="utf-8")
    return key, RestoreContract(train, hashlib.sha256(key.read_bytes()).hexdigest())


def test_rebuild_preserves_private_bytes_and_produces_only_blinded_png_labelme_csv(
    tmp_path: Path,
) -> None:
    key, contract = private_fixture(tmp_path)
    out = tmp_path / "restored"

    restore_from_key(key, out, contract)

    assert (out / "key-local.json").read_bytes() == key.read_bytes()
    review = out / "review"
    assert len(list(review.glob("*.png"))) == 48
    assert len(list(review.glob("*.json"))) == 48
    with (review / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 49
    assert rows[0] == [
        "id",
        "target_family",
        "object_visible",
        "counterpart_visible",
        "relation",
        "uncertainty_reason",
        "observation",
        "reviewer",
    ]
    for index, row in enumerate(rows[1:], 1):
        assert row == [f"R{index:02}", "slip_trip", "", "", "", "", "", ""]
        with Image.open(review / f"R{index:02}.png") as image:
            assert image.size == (3, 2)
            assert image.getpixel((0, 0)) == (index, 100, 200)
        label = json.loads((review / f"R{index:02}.json").read_text())
        assert label == {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": f"R{index:02}.png",
            "imageData": None,
            "imageHeight": 2,
            "imageWidth": 3,
        }
    assert all(b"SECRET_" not in path.read_bytes() for path in review.iterdir())


def test_wrong_private_key_digest_writes_no_review(tmp_path: Path) -> None:
    key, contract = private_fixture(tmp_path)
    key.write_bytes(key.read_bytes() + b" ")
    out = tmp_path / "restored"

    with pytest.raises(PanelError, match="Private key SHA-256"):
        restore_from_key(key, out, contract)

    assert not out.exists()


def test_existing_grading_is_preserved_byte_for_byte(tmp_path: Path) -> None:
    key, contract = private_fixture(tmp_path)
    out = tmp_path / "restored"
    out.mkdir()
    grading = out / "graded.csv"
    grading.write_bytes(b"reviewer,grade\nexpert,accepted\n")

    with pytest.raises(PanelError, match="Existing review packet preserved"):
        restore_from_key(key, out, contract)

    assert grading.read_bytes() == b"reviewer,grade\nexpert,accepted\n"
    assert sorted(p.name for p in out.iterdir()) == ["graded.csv"]


@pytest.mark.parametrize("corruption", ["escape", "duplicate_alias", "image_hash"])
def test_invalid_key_entries_fail_before_any_review_write(
    tmp_path: Path, corruption: str
) -> None:
    key, original = private_fixture(tmp_path)
    rows = json.loads(key.read_text())
    if corruption == "escape":
        outside = tmp_path / "outside.png"
        outside.write_bytes((original.train_root / "image-1.png").read_bytes())
        rows[0]["source"]["image"] = "../outside.png"
    if corruption == "duplicate_alias":
        rows[-1]["id"] = rows[0]["id"]
    if corruption == "image_hash":
        rows[0]["source"]["sha256"] = "0" * 64
    key.write_text(json.dumps(rows))
    contract = RestoreContract(
        original.train_root, hashlib.sha256(key.read_bytes()).hexdigest()
    )
    out = tmp_path / "restored"

    with pytest.raises(PanelError):
        restore_from_key(key, out, contract)

    assert not out.exists()
