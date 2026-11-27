"""Run Meta's three static-image paths on the frozen local RGB panel."""

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

from .phase2_panel import PROJECT, RUN, file_hash, read, registry, write
from .phase2_paths import file_stem
from .phase2_regions import ROOT, ROLES, selection
from .visual_probe import DEFAULT_CACHE, FILES
from tools.fetch_phase2_sam_source import REVISION


ARMS = {"sam3-image": "sam3", "sam3-frame": "sam3", "sam31-frame": "sam31"}
DEST = ROOT / "sam3-static"
THRESHOLD = 0.5


def to_array(value):
    if not hasattr(value, "detach"):
        return np.asarray(value)
    tensor = value.detach().cpu()
    if str(tensor.dtype) == "torch.bfloat16":
        tensor = tensor.float()
    return tensor.numpy()


def instances(output, image_arm, role, size, folder):
    masks = to_array(output["masks" if image_arm else "out_binary_masks"])
    if len(masks) == 0:
        return []
    scores = to_array(output["scores" if image_arm else "out_probs"]).reshape(-1)
    if len(scores) != len(masks):
        raise ValueError("Confidence and mask counts differ")
    records = []
    folder.mkdir(parents=True, exist_ok=True)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = np.asarray(mask, dtype=bool).reshape(size[1], size[0])
        pic = Image.fromarray(mask)
        box = pic.getbbox()
        if box is None:
            continue
        path = folder / f"{ROLES.index(role)}-{i}.png"
        pic.save(path)
        records.append(
            dict(
                role=role,
                score=float(score),
                area=int(mask.sum()),
                mask=str(path),
                xyxy=[v / size[j % 2] for j, v in enumerate(box)],
            )
        )
    return records


def detect(model, arm, path, pic, out):
    image_arm = arm == "sam3-image"
    records = []
    if image_arm:
        state = model.set_image(pic)
        for role in ROLES:
            model.reset_all_prompts(state)
            reply = model.set_text_prompt(prompt=role, state=state)
            records.extend(instances(reply, True, role, pic.size, out))
        return records
    request = model.handle_request
    sid = request({"type": "start_session", "resource_path": str(path)})["session_id"]
    try:
        for i, role in enumerate(ROLES):
            if i:
                # A new text concept has no earlier role, object IDs or masks attached.
                request({"type": "reset_session", "session_id": sid})
            reply = request(
                dict(
                    type="add_prompt",
                    session_id=sid,
                    frame_index=0,
                    text=role,
                    output_prob_thresh=THRESHOLD,
                )
            )
            if reply["frame_index"] != 0:
                raise ValueError("Unexpected frame index for a still photo")
            records.extend(instances(reply["outputs"], False, role, pic.size, out))
    finally:
        try:
            request({"type": "reset_session", "session_id": sid})
        finally:
            request({"type": "close_session", "session_id": sid})
    return records


def block_sockets(event, args):
    if event in {
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "socket.sendto",
    }:
        raise RuntimeError(f"Offline local-only SAM run blocked {event}")


def adapt_session(predictor):
    native = predictor.model.init_state
    params = inspect.signature(native).parameters
    if "offload_state_to_cpu" in params or any(
        p.kind == p.VAR_KEYWORD for p in params.values()
    ):
        return False

    def init_state(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu is not False:
            raise ValueError("The native static tracker supports GPU state only")
        return native(*args, **kwargs)

    predictor.model.init_state = init_state
    return True


def restore_rope_views(state):
    added = []
    for name, buf in list(state.items()):
        if (
            name.startswith("detector.")
            and name.endswith(".attn.freqs_cis")
            and buf.is_complex()
        ):
            for part in ("real", "imag"):
                dest = name + "_" + part
                if dest not in state:
                    state[dest] = getattr(buf, part)
                    added.append(dest)
    return added


def run(arm, wait_for_pid=None):
    key = ARMS[arm]
    source = RUN / "large-models/sam-source"
    weight = RUN / "large-models" / key
    assert read(source / "verified-source.json")["revision"] == REVISION
    sys.path.insert(0, str(source))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.addaudithook(block_sockets)
    start = time.perf_counter()
    import torch
    from sam3.model_builder import build_sam3_image_model, build_sam3_predictor
    from sam3.model.sam3_image_processor import Sam3Processor

    import_seconds = time.perf_counter() - start
    write(
        DEST / arm / "imports.json",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            seconds=import_seconds,
            code=REVISION,
            observed_at=time.time(),
        ),
    )
    while not (weight / "complete.json").exists():
        time.sleep(5)
    checkpoint = read(weight / "complete.json")
    path = weight / checkpoint["weight"]
    if not path.is_file() or path.stat().st_size != checkpoint["bytes"]:
        raise ValueError("Pinned local checkpoint is incomplete")
    if wait_for_pid is not None:
        import psutil

        try:
            active = psutil.Process(wait_for_pid)
            if "kwater_lab.phase2_official" not in active.cmdline():
                raise ValueError("Expected the official Qwen process at handoff")
            active.wait()
        except psutil.NoSuchProcess:
            pass
    start = time.perf_counter()

    native_load = torch.load
    using_mmap = False
    derived_rope = []
    key_check = {}
    session_compatibility = False
    skipped_unused_read = False
    structure_file = weight / "state-structure.json"
    structure = (
        read(structure_file) if key == "sam31" and structure_file.is_file() else {}
    )
    original_state_load = torch.nn.Module.load_state_dict

    def checked_state_load(module, weights, *args, **kwargs):
        integrated = (
            key == "sam31"
            and any(s.startswith("detector.") for s in weights)
            and hasattr(module, "detector")
            and hasattr(module, "tracker")
        )
        if integrated:
            kwargs["strict"] = True
        result = original_state_load(module, weights, *args, **kwargs)
        if integrated:
            key_check.update(
                supplied=len(weights),
                missing=list(result.missing_keys),
                unexpected=list(result.unexpected_keys),
            )
        return result

    def checkpoint_load(file, *args, **kwargs):
        nonlocal using_mmap, skipped_unused_read
        name = (
            file
            if isinstance(file, (str, os.PathLike))
            else getattr(file, "name", None)
        )
        if (
            name
            and Path(name).resolve() == path.resolve()
            and kwargs.get("map_location") == "cpu"
        ):
            if (
                key == "sam31"
                and not isinstance(file, (str, os.PathLike))
                and structure.get("checkpoint") == checkpoint
                and structure.get("tracker_unused_checkpoint", {}).get(
                    "matches_entire_checkpoint"
                )
            ):
                # The observed initial tracker rejected every key; the official full predictor loads them below.
                skipped_unused_read = True
                return {}
            with path.open("rb") as stream:
                if stream.read(4) == b"PK\x03\x04":
                    using_mmap = True
                    raw = native_load(str(path), *args, **dict(kwargs, mmap=True))
                    if key == "sam31" and isinstance(file, (str, os.PathLike)):
                        state = raw.get("model", raw)
                        derived_rope.extend(restore_rope_views(state))
                    return raw
        return native_load(file, *args, **kwargs)

    with (
        patch.object(torch, "load", checkpoint_load),
        patch.object(torch.nn.Module, "load_state_dict", checked_state_load),
    ):
        bpe = str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        if arm == "sam3-image":
            model = Sam3Processor(
                build_sam3_image_model(
                    checkpoint_path=str(path), bpe_path=bpe, load_from_HF=False
                ),
                confidence_threshold=THRESHOLD,
            )
        else:
            options = dict(
                checkpoint_path=str(path),
                bpe_path=bpe,
                compile=False,
                async_loading_frames=False,
            )
            context = nullcontext()
            if key == "sam3":
                # Upstream selects an unused rendezvous port even for one rank. No process group is created at world size one.
                from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

                context = patch.object(
                    Sam3VideoPredictorMultiGPU, "_find_free_port", return_value=0
                )
                options["gpus_to_use"] = [0]
            else:
                options["use_fa3"] = False
            with context:
                model = build_sam3_predictor(
                    version="sam3" if key == "sam3" else "sam3.1", **options
                )
            session_compatibility = adapt_session(model)
            if key == "sam31" and not key_check:
                raise ValueError(
                    "The final integrated SAM 3.1 model was not checked against its state dictionary"
                )
    torch.cuda.synchronize()
    write(
        DEST / arm / "runtime.json",
        dict(
            cold_import_and_load_seconds=import_seconds + time.perf_counter() - start,
            import_seconds=import_seconds,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            repo_code=REVISION,
            checkpoint=checkpoint,
            memory_mapped_checkpoint=using_mmap,
            skipped_unused_tracker_checkpoint_read=skipped_unused_read,
            derived_real_rope_buffers=derived_rope,
            final_state_dictionary=key_check,
            omitted_unsupported_default_false_state_offload=session_compatibility,
            confidence_threshold=THRESHOLD,
            native_internal_filters="upstream defaults",
            local_socket_audit=True,
            single_image_sessions=True,
            pixel_preprocessing="published native path at 1008",
        ),
    )
    rows, spec = registry(), read(ROOT / "design.json")
    ids = spec["review"]
    origin = {spec["donors"][s]: s for s in ids}
    public = DEFAULT_CACHE / "public-example.jpg"
    assert file_hash(public) == FILES[public.name]
    for s in dict.fromkeys(["public-example"] + ids + list(origin)):
        stem = file_stem(s)
        dest = ROOT / ("detect-" + arm) / (stem + ".json")
        if dest.is_file() and read(dest).get("status") == "ok":
            continue
        image_path = (
            public
            if s == "public-example"
            else PROJECT / "data/train" / rows[s]["image"]
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            with Image.open(image_path) as im:
                if im.getexif().get(274, 1) != 1:
                    raise ValueError(
                        "Native image and predictor require identical orientation"
                    )
                pic = im.convert("RGB")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                items = detect(model, arm, image_path, pic, DEST / arm / "masks" / stem)
            torch.cuda.synchronize()
            rec = dict(
                status="ok",
                detections=items,
                size=pic.size,
                rgb_sha256=hashlib.sha256(pic.tobytes()).hexdigest(),
            )
            if s != "public-example":
                selected = selection(rec, spec["target"][origin.get(s, s)])
                rec["selection"] = selected
                item = selected["person"] or selected["partner"]
                if item:
                    mask = ROOT / ("masks-" + arm) / (stem + ".png")
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    mask.write_bytes(Path(item["mask"]).read_bytes())
        except Exception as err:
            rec = dict(status="failed", error=repr(err))
        rec.update(
            sample_id=s,
            seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        write(dest, rec)
        print("SAM_ROLE_ROW", arm, s, rec["status"], rec["seconds"], flush=True)
        if s == "public-example" and rec["status"] != "ok":
            raise RuntimeError(f"Public static-image execution failed. See {dest}")
    panel = [read(ROOT / ("detect-" + arm) / (file_stem(s) + ".json")) for s in ids]
    write(
        DEST / arm / "engineering.json",
        dict(
            rows=len(panel),
            statuses=dict(Counter(r["status"] for r in panel)),
            role_image_counts={
                role: sum(
                    any(d["role"] == role for d in r.get("detections", []))
                    for r in panel
                )
                for role in ROLES
            },
            selected_person_and_target=sum(
                r.get("selection", {}).get("pair_visible", False) for r in panel
            ),
            seconds_p50_p95=np.percentile(
                [r["seconds"] for r in panel], [50, 95]
            ).tolist(),
            max_allocated_bytes=max(r["peak_allocated_bytes"] for r in panel),
            human_mask_truth=None,
            human_role_pair_recall=None,
            l4_measured=False,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=[*ARMS, "sam3-both", "sam-suite"])
    ap.add_argument("--wait-official-pid", type=int)
    args = ap.parse_args()
    if args.arm in ("sam3-both", "sam-suite"):
        import gc
        import torch

        outcomes = {}
        arms = (
            ("sam31-frame", "sam3-image", "sam3-frame")
            if args.arm == "sam-suite"
            else ("sam3-image", "sam3-frame")
        )
        for arm in arms:
            if (DEST / arm / "engineering.json").is_file():
                outcomes[arm] = "existing"
                continue
            try:
                run(arm, args.wait_official_pid)
                outcomes[arm] = "completed"
            except Exception as err:
                import traceback

                outcomes[arm] = dict(error=repr(err), traceback=traceback.format_exc())
            write(DEST / (args.arm + ".json"), outcomes)
            gc.collect()
            torch.cuda.empty_cache()
        if any(isinstance(v, dict) for v in outcomes.values()):
            raise SystemExit(1)
    else:
        run(args.arm, args.wait_official_pid)
