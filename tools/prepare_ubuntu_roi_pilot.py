"""Reproduce the frozen 24-photo cohort without inspecting target fields.

Run: .venv/bin/python tools/prepare_ubuntu_roi_pilot.py --out /tmp/cohort.json
Windows exposure: --exposure data/prefix-pilot/exposure.json
Print optional official model setup: --model-download-commands
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
EXPECTED: Final = "9b2ba05e73682fa902b38c9cffe2c1b6533c07e4899811b411a7a3aa85f5e306"
SALT: Final = "kwater/ubuntu-roi-v1/20261127/"
SCHEME: Final = (
    "First four per historical prefix, ordered by "
    "SHA256(kwater/ubuntu-roi-v1/20261127/+sample_id); no risk labels read to select"
)
MODEL_COMMANDS: Final = """# Run from the project root; optional public model setup.
mkdir -p data/ubuntu-roi-20261127/models/blip
curl --fail --location --retry 3 --output data/ubuntu-roi-20261127/models/blip-tree.json 'https://huggingface.co/api/models/Salesforce/blip-image-captioning-base/tree/82a37760796d32b1411fe092ab5d4e227313294b?expand=true'
curl --fail --location --retry 3 --output data/ubuntu-roi-20261127/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
for name in config.json preprocessor_config.json pytorch_model.bin special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
  curl --fail --location --retry 3 --output "data/ubuntu-roi-20261127/models/blip/$name" "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/82a37760796d32b1411fe092ab5d4e227313294b/$name"
done
sha256sum --check <<'SHA256'
bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0  data/ubuntu-roi-20261127/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
d6638651a5526cc2ede56f2b5104d6851b0755816d220e5e046870430180c767  data/ubuntu-roi-20261127/models/blip/pytorch_model.bin
SHA256
"""


class CohortMismatch(ValueError):
    def __init__(self, observed: str) -> None:
        self.observed = observed
        super().__init__(
            f"Frozen cohort differs: observed {observed}, expected {EXPECTED}"
        )


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cohort_bytes(exposure: Path, train_root: Path) -> bytes:
    """Project ID/image metadata only, hash photos, then enforce the frozen output."""
    exposed = json.loads(exposure.read_text(encoding="utf-8"))["by_prefix"]
    image_by_id: dict[str, str] = {}
    train = train_root / "train.jsonl"
    with train.open(encoding="utf-8") as stream:
        for line in stream:
            metadata = json.loads(line)
            sample_id, image = metadata["sample_id"], metadata["image"]
            if sample_id in image_by_id:
                raise CohortMismatch("duplicate metadata ID")
            image_by_id[sample_id] = image
    rows = []
    for prefix in sorted(exposed):
        selected = sorted(
            exposed[prefix],
            key=lambda sample_id: hashlib.sha256(
                (SALT + sample_id).encode()
            ).hexdigest(),
        )[:4]
        for sample_id in selected:
            image = image_by_id[sample_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "image": image,
                    "sha256": digest(train_root / image),
                    "prefix": prefix,
                }
            )
    document = {
        "scheme": SCHEME,
        "exposure_sha256": digest(exposure),
        "train_source_sha256": digest(train),
        "rows": rows,
    }
    encoded = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode()
    observed = hashlib.sha256(encoded).hexdigest()
    if observed != EXPECTED:
        raise CohortMismatch(observed)
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", type=Path)
    parser.add_argument("--train-root", type=Path, default=ROOT / "data/train")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data/ubuntu-roi-20261127/cohort.json"
    )
    parser.add_argument("--model-download-commands", action="store_true")
    args = parser.parse_args()
    if args.model_download_commands:
        print(MODEL_COMMANDS, end="")
        return
    exposure = args.exposure
    if exposure is None:
        candidates = (
            ROOT / "data/ubuntu3050-runs/prefix-exposure.json",
            ROOT / "data/prefix-pilot/exposure.json",
        )
        exposure = next((path for path in candidates if path.is_file()), candidates[0])
    encoded = cohort_bytes(exposure, args.train_root)
    if args.out.exists() and args.out.read_bytes() != encoded:
        raise CohortMismatch("existing destination contains different bytes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(args.out)
    print(
        json.dumps(
            {"photos": 24, "prefixes": 6, "sha256": EXPECTED, "output": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
