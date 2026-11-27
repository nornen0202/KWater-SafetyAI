"""Build offline image-only F32/BF16 weights and compare first-view features on one fixed GPU."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
import numpy as np

from kwater_lab.phase2_panel import (
    PHASE1,
    PROJECT,
    RUN,
    file_hash,
    registry,
    read,
    write,
)
from kwater_lab.phase2_paths import file_stem

ROOT = RUN / "agreement-heads/image-towers"
SOURCE = PHASE1 / "models"


def header(stream):
    size = struct.unpack("<Q", stream.read(8))[0]
    return json.loads(stream.read(size)), 8 + size


def describe():
    from collections import defaultdict

    table = {}
    for tower in ("sig", "dino"):
        with (SOURCE / tower / "model.safetensors").open("rb") as f:
            data, offset = header(f)
        groups = defaultdict(lambda: {"tensors": 0, "bytes": 0})
        for key, v in data.items():
            if key == "__metadata__":
                continue
            prefix = key.split(".", 1)[0]
            groups[prefix]["tensors"] += 1
            lo, hi = v["data_offsets"]
            groups[prefix]["bytes"] += hi - lo
        table[tower] = dict(
            prefixes=dict(groups),
            header_bytes=offset,
            preprocessor=read(SOURCE / tower / "preprocessor_config.json"),
        )
    table["reference"] = dict(
        global_view=0,
        linear_token=0,
        spatial_tokens=17,
        projection_seed_by_channel_width="404 + width",
        image_views_at_inference=1,
        revision={
            a["name"]: a["revision"]
            for a in read(PHASE1 / "inference/assets.json")["assets"]
            if a["kind"] == "model"
        },
        inference_training_text_embeddings=False,
        training_evidence_alignment="training only; 64-d hashed labels, no SigLIP text tower",
        eight_b_with_bf16_image_tensor_bytes=18999527992,
        package_headroom_before_runtime_and_adapters_bytes=1000472008,
        last_two_values_source="competition manager message f36e6d14746d; actual assembled archive measured separately",
    )
    write(ROOT / "header-audit.json", table)


def packed(source, dest, tower, precision):
    start = time.perf_counter()
    source_dir = source / tower
    dest = dest / tower
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "model.safetensors"
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copyfile(source_dir / name, dest / name)
    origin = source_dir / "model.safetensors"
    receipt = dest / "weights.json"
    if receipt.is_file() and path.is_file():
        meta = read(receipt)
        if meta["bytes"] != path.stat().st_size:
            raise ValueError("Image-tower file size changed")
        return
    with origin.open("rb") as f:
        state, offset = header(f)
        selected = sorted(
            (
                (k, v)
                for k, v in state.items()
                if k != "__metadata__"
                and (tower == "dino" or k.startswith("vision_model."))
            ),
            key=lambda item: item[1]["data_offsets"][0],
        )
        if len(selected) != {"sig": 448, "dino": 439}[tower]:
            raise ValueError("Unexpected image tensor count")
        end = 0
        h = {"__metadata__": {"format": "pt"}}
        for k, v in selected:
            if v["dtype"] != "F32":
                raise ValueError("Expected the original F32 release")
            lo, hi = v["data_offsets"]
            n = (hi - lo) // (2 if precision == "bf16" else 1)
            h[k] = dict(
                dtype="BF16" if precision == "bf16" else "F32",
                shape=v["shape"],
                data_offsets=[end, end + n],
            )
            end += n
        tmp = path.with_suffix(".writing")
        sha = hashlib.sha256()
        if tower == "dino" and precision == "fp32":
            if path.exists() and not os.path.samefile(path, origin):
                raise ValueError("Refusing to replace an unrelated image weights file")
            if not path.exists():
                os.link(origin, path)
            # The organizer retained this exact public file in the Phase1 local release.
            digest = file_hash(origin)
        else:
            payload = json.dumps(h, separators=(",", ":")).encode()
            payload += b" " * ((-len(payload)) % 8)
            with tmp.open("wb") as target:
                out = struct.pack("<Q", len(payload)) + payload
                target.write(out)
                sha.update(out)
                for _, v in selected:
                    lo, hi = v["data_offsets"]
                    f.seek(offset + lo)
                    left = hi - lo
                    while left:
                        data = f.read(min(4 * 1024 * 1024, left))
                        if not data:
                            raise IOError("Incomplete source tensor")
                        left -= len(data)
                        if precision == "bf16":
                            raw = np.frombuffer(data, dtype="<u4")
                            if np.any((raw & 0x7F800000) == 0x7F800000):
                                raise ValueError("Nonfinite published weight")
                            rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) >> 16
                            data = rounded.astype("<u2").tobytes()
                        target.write(data)
                        sha.update(data)
            os.replace(tmp, path)
            digest = sha.hexdigest()
        write(
            receipt,
            dict(
                tensors=len(selected),
                tensor_bytes=end,
                bytes=path.stat().st_size,
                sha256=digest,
                precision=precision,
                seconds=time.perf_counter() - start,
                source=str(origin),
            ),
        )


def build(precision):
    for tower in ("sig", "dino"):
        packed(SOURCE, ROOT / precision, tower, precision)


def compare(device):
    import gc
    import torch
    from PIL import Image
    from kwater_lab.phase01_vision import Vision
    from kwater_lab.phase01_head import SpatialHead, logits, decide
    from kwater_lab.metrics import CODES
    from kwater_lab.train_head import probabilities
    from kwater_lab.phase2_agreement_heads import ROOT as heads

    if not (heads / "shared-predictions.json").is_file():
        raise ValueError("The restored held-out fold heads must be checked first")
    rows = registry()
    panel = read(RUN / "plan.json")["panel"]
    reference = read(heads / "shared-predictions.json")
    report = {}
    for mode, folder, ids in (
        ("original", SOURCE, ["synthetic"]),
        ("fp32", ROOT / "fp32", ["synthetic", *panel]),
        ("bf16", ROOT / "bf16", ["synthetic", *panel]),
    ):
        out = ROOT / ("observed-" + mode)
        out.mkdir(parents=True, exist_ok=True)
        begun = time.perf_counter()
        vision = Vision(folder, device)
        load = time.perf_counter() - begun
        records = {}
        for sid in ids:
            start = time.perf_counter()
            if sid == "synthetic":
                yy, xx = np.indices((157, 263))
                image = Image.fromarray(
                    np.stack([xx, yy, (xx * 7 + yy * 3) % 251], axis=-1).astype(
                        np.uint8
                    )
                )
            else:
                with Image.open(PROJECT / "data/train" / rows[sid]["image"]) as im:
                    from PIL import ImageOps

                    image = ImageOps.exif_transpose(im).convert("RGB")
            tokens = vision.encode(image, multiscale=False)
            seconds = time.perf_counter() - start
            feat = out / (file_stem(sid) + ".npy")
            np.save(feat, tokens, allow_pickle=False)
            if sid == "synthetic":
                full = np.load(
                    ROOT / "observed-original" / "synthetic.npy", allow_pickle=False
                )
                records[sid] = dict(
                    max_abs_error=float(np.abs(full.astype(np.float32) - tokens).max()),
                    exact=bool(np.array_equal(full, tokens)),
                    seconds=seconds,
                )
                continue
            f = rows[sid]["dev_fold"]
            vector = tokens[:, 0].astype(np.float32)
            vector /= np.linalg.norm(vector, axis=1, keepdims=True)
            with np.load(heads / f"fold-{f}-linear.npz", allow_pickle=False) as head:
                code = CODES[int(probabilities(head, vector)[0].argmax())]
            net = SpatialHead(False)
            net.load_state_dict(
                torch.load(
                    heads / f"fold-{f}-spatial.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            z = logits(net.eval(), torch.from_numpy(tokens[None]))
            mask = decide(z, read(heads / f"fold-{f}-policy.json"))[0]
            passes = [c for c, b in zip(CODES, mask) if b]
            records[sid] = dict(
                candidate=code,
                spatial=passes,
                confirmed=code in passes,
                original_oof_top1_equal=code == reference[sid]["top1"],
                original_oof_set_equal=set(passes) == set(reference[sid]["spatial"]),
                seconds=seconds,
            )
        del vision
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        write(out / "rows.json", records)
        report[mode] = dict(
            source=str(folder),
            compute_dtype="FP16" if device == "cuda" else "FP32",
            load_seconds=load,
            synthetic=records["synthetic"],
        )
    a = read(ROOT / "observed-fp32/rows.json")
    b = read(ROOT / "observed-bf16/rows.json")
    report["paired_flips"] = {
        k: sum(a[s][k] != b[s][k] for s in panel)
        for k in ("candidate", "spatial", "confirmed")
    }
    report["panel"] = len(panel)
    report["text_model_invocations"] = 0
    write(ROOT / "comparison.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["describe", "fp32", "bf16", "compare"])
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = p.parse_args()
    if args.stage == "describe":
        describe()
    elif args.stage == "compare":
        compare(args.device)
    else:
        build(args.stage)
