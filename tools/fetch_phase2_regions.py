"""Fetch only the public CPU detector and segmentation graphs. No project image inputs."""

import requests
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from phase2_assets import acquire

SPECS = {
    "detector": (
        "onnx-community/grounding-dino-tiny-ONNX",
        "ff690b0a8050566c290287545bd059350f3e9096",
        [
            "onnx/model_int8.onnx",
            "preprocessor_config.json",
            "tokenizer.json",
            "config.json",
        ],
    ),
    "segmenter": (
        "onnx-community/sam2.1-hiera-tiny-ONNX",
        "814a066640debee5a91e70aa401fb8e17e030503",
        [
            "onnx/" + x + "_int8.onnx" + suffix
            for x in ("vision_encoder", "prompt_encoder_mask_decoder")
            for suffix in ("", "_data")
        ]
        + ["preprocessor_config.json", "config.json"],
    ),
}


if __name__ == "__main__":
    root = Path(sys.argv[1]) / "regions/models"
    receipt = root / "receipts.json"
    meta = json.loads(receipt.read_bytes()) if receipt.is_file() else {}
    chosen = SPECS
    if "--base" in sys.argv[2:]:
        observed = json.loads((root.parent / "engineering.json").read_bytes())[
            "frozen64"
        ]
        counts = observed["photo_counts_by_noun"]
        if not counts["person"] or any(n for k, n in counts.items() if k != "person"):
            raise ValueError("The preregistered detector escalation condition is unmet")
        chosen = {
            "detector_base": (
                "ZumaInnovations/zuma-ground-base-onnx",
                "0fb03e157f2beefc8dec89bc734721895465acb3",
                ["onnx/model.onnx", "vocab.txt", "README.md"],
            )
        }
    for key, (repo, rev, files) in chosen.items():
        reply = requests.get(
            f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true",
            timeout=30,
        )
        reply.raise_for_status()
        d = reply.json()
        assert d["sha"] == rev
        selected = [s for s in d["siblings"] if s["rfilename"] in files]
        assert len(selected) == len(files)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda item: acquire(item, root / key, repo, rev), selected))
        meta[key] = {
            "repo": repo,
            "revision": rev,
            "export_license": d.get("cardData", {}).get("license"),
            "files": selected,
        }
        receipt.write_text(json.dumps(meta, indent=2), encoding="utf8")
