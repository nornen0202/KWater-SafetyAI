"""Post-hoc preparation of matched mask/caption controls, reusing sealed box outputs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image

from . import ubuntu_roi_generation as anchor
from .ubuntu_roi_readout import caption_pairs
from .vlm_11code import load_qwen_processor


def prepare_followup() -> None:
    """Freeze the same twelve images and full-prompt caption parity without gold."""
    original = anchor.RUN
    run = anchor.WORK / "generation-followup"
    if (run / "plan.json").exists():
        raise anchor.AnchorError("Follow-up plan already exists; use its frozen run")
    anchor.verify_seal(original)
    parent = json.loads((original / "plan.json").read_text())
    for relative, digest in parent["official_file_sha256"].items():
        if anchor.sha256(anchor.BASE / relative) != digest:
            raise anchor.AnchorError(
                "Official model/prompt/formatter changed before reuse"
            )
    views = {
        row["sample_id"]: row
        for row in json.loads((anchor.WORK / "views/index.json").read_text())["rows"]
    }
    processor = load_qwen_processor(anchor.BASE / "base_model", 384 * 32 * 32)
    entry = anchor.load_entry()
    prompt = (anchor.BASE / "system_prompt.txt").read_text()
    pairs = caption_pairs(
        [views[row["sample_id"]]["caption"] for row in parent["rows"]],
        processor.tokenizer,
    )
    plan = deepcopy(parent)
    plan["arms"] = ["box", "mask", "sham", "caption", "swapped"]
    plan["comparison_pairs"] = [
        ["box", "mask"],
        ["sham", "mask"],
        ["box", "caption"],
        ["swapped", "caption"],
    ]
    plan["protocol"] = (
        "post-hoc controlled follow-up after initial anchor labels/metrics were observed; same twelve photos, no tuning of settings or selection"
    )
    plan["schema"] = "roi-official-generation-followup-v1"
    plan["implementation_sha256"] = anchor.sha256(Path(anchor.__file__))
    plan["preparation_implementation_sha256"] = anchor.sha256(Path(__file__))
    plan["parent_plan_sha256"] = anchor.sha256(original / "plan.json")
    plan["parent_sealed_sha256"] = anchor.sha256(original / "sealed.json")
    plan["caption_policy"] = (
        "minimum-length derangement using frozen BLIP captions; truncate both captions to matched tokenizer length; enforce full processed prompt parity; identical neutral auxiliary-description instruction"
    )
    identities = []
    for index, row in enumerate(plan["rows"]):
        view = views[row["sample_id"]]
        full, box = row["paths"]["box"]
        if [anchor.sha256(Path(path)) for path in [full, box]] != row["sha256"]["box"]:
            raise anchor.AnchorError("Sealed baseline image bytes changed")
        row["paths"] = {
            "box": [full, box],
            "mask": [full, view["mask"]],
            "sham": [full, view["sham"]],
            "caption": [full, box],
            "swapped": [full, box],
        }
        row["sha256"] = {
            arm: [anchor.sha256(Path(path)) for path in paths]
            for arm, paths in row["paths"].items()
        }
        donor, true, foreign = pairs[index]
        descriptions = {"caption": true, "swapped": foreign}
        row["user_text"] = {
            arm: entry.USER_TEXT
            + "\nAuxiliary description (may be inaccurate; verify in the image): "
            + text
            for arm, text in descriptions.items()
        }
        row["caption_donor_image_id"] = parent["rows"][donor]["image_id"]
        row["original_caption"] = view["caption"]
        row["caption_pair"] = descriptions
        row["processed_tokens"] = {}
        row["processed_grids"] = {}
        for arm in plan["arms"]:
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": row["user_text"].get(arm, entry.USER_TEXT),
                        },
                    ],
                },
            ]
            prefix = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images = []
            for path in row["paths"][arm]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            batch = processor(text=[prefix], images=images, return_tensors="pt")
            row["processed_tokens"][arm] = int(batch["input_ids"].shape[-1])
            row["processed_grids"][arm] = batch["image_grid_thw"].tolist()
            if arm == "box":
                identities.append(prefix)
        if row["processed_tokens"]["caption"] != row["processed_tokens"]["swapped"]:
            raise anchor.AnchorError("Full caption prompts differ in token length")
        if len({json.dumps(value) for value in row["processed_grids"].values()}) != 1:
            raise anchor.AnchorError("Follow-up visual grids are unmatched")
    (run / "records").mkdir(parents=True)
    (run / "cache").mkdir()
    (run / "inputs").symlink_to(original / "inputs", target_is_directory=True)
    anchor.write_atomic(
        run / "plan.json",
        json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    plan_sha = anchor.sha256(run / "plan.json")
    receipts = []
    for row, prefix in zip(plan["rows"], identities, strict=True):
        source = original / "records" / f"{row['image_id']}-box.json"
        record = json.loads(source.read_text())
        old_identity = hashlib.sha256(
            json.dumps(
                [plan["parent_plan_sha256"], prefix, row["sha256"]["box"]]
            ).encode()
        ).hexdigest()
        if (
            record["input_identity"] != old_identity
            or record["image_grid_thw"] != row["processed_grids"]["box"]
            or record["prompt_tokens"] != row["processed_tokens"]["box"]
        ):
            raise anchor.AnchorError(
                "Sealed box input identity differs from follow-up baseline"
            )
        identity = hashlib.sha256(
            json.dumps([plan_sha, prefix, row["sha256"]["box"]]).encode()
        ).hexdigest()
        reused = {
            **record,
            "input_identity": identity,
            "reused": True,
            "reused_from": str(source),
            "reused_source_sha256": anchor.sha256(source),
        }
        contents = json.dumps(reused, ensure_ascii=False) + "\n"
        anchor.write_atomic(run / "records" / source.name, contents)
        anchor.write_atomic(run / "cache" / f"{identity}.json", contents)
        receipts.append(
            {
                "image_id": row["image_id"],
                "source_sha256": anchor.sha256(source),
                "old_identity": old_identity,
                "new_identity": identity,
            }
        )
    anchor.write_atomic(
        run / "reuse-receipt.json", json.dumps(receipts, indent=2) + "\n"
    )
    print(
        f"Post-hoc follow-up frozen: 60 conditions, 12 sealed box reuses; {run}",
        flush=True,
    )


if __name__ == "__main__":
    prepare_followup()
