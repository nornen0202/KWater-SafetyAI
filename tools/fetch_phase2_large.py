"""Fetch pinned public weights for the large-model comparison; inspect gated models via official access."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import requests

from kwater_lab.phase2_io import RUN, read, write
from tools.phase2_assets import acquire


SPECS = {
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    "sam2-large": (
        "facebook/sam2.1-hiera-large",
        "665f8e2ad61cf5f53d65644ff27c8ee525124610",
    ),
    "sam2-large-onnx": (
        "onnx-community/sam2.1-hiera-large-ONNX",
        "3c23431f721e69cae82dbfd0c28fd692cc714021",
    ),
    "sam3": ("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7"),
    "sam31": ("facebook/sam3.1", "daa63191845a41281374e725f4c9e51c7a824460"),
    "9b-gguf": ("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5"),
    "27b-gguf": (
        "unsloth/Qwen3.8-27B-GGUF",
        "4af51b3542f775b0a8e4b310bfb3f17e01c306c9",
    ),
}


def fetch(key, mib_per_second=None, only=None, acquire_gated=False):
    repo, rev = SPECS[key]
    dest = RUN / "large-models" / key
    if (
        key in ("8b", "9b", "9b-gguf", "27b-gguf")
        and (RUN / "large-models/transfer-pause.flag").exists()
    ):
        policy_path = RUN / "large-models/transfer-policy.json"
        policy = read(policy_path) if policy_path.is_file() else {}
        if (
            key != policy.get("model")
            or not only
            or only != policy.get("file")
            or not mib_per_second
            or mib_per_second > policy.get("mib_s", 0)
        ):
            raise RuntimeError(
                "Transfer is outside the release manager's current network allowance"
            )
    endpoint = f"https://huggingface.co/{repo}"
    response = requests.get(
        f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true",
        timeout=30,
    )
    response.raise_for_status()
    details = response.json()
    assert details["sha"] == rev
    files = [
        f
        for f in details["siblings"]
        if "/" not in f["rfilename"]
        and not f["rfilename"].startswith(".")
        and f["rfilename"].endswith(
            (".safetensors", ".json", ".txt", ".jinja", ".md", "LICENSE")
        )
    ]
    if key == "sam2-large-onnx":
        files = [
            f
            for f in details["siblings"]
            if "_int8.onnx" in f["rfilename"]
            or f["rfilename"]
            in ("config.json", "preprocessor_config.json", "README.md")
        ]
    if key.endswith("-gguf"):
        language = (
            "Qwen3.5-9B-Q4_K_M.gguf"
            if key == "9b-gguf"
            else "Qwen3.8-27B-UD-Q4_K_M.gguf"
        )
        files = [
            f
            for f in details["siblings"]
            if f["rfilename"] in (language, "mmproj-BF16.gguf", "README.md", "LICENSE")
        ]
        assert sum(f["size"] for f in files if f["rfilename"].endswith(".gguf")) == (
            6602227488 if key == "9b-gguf" else 17495406176
        )
    info = dict(
        repo=repo,
        revision=rev,
        model_card=endpoint,
        license=details.get("cardData", {}).get("license"),
        gated=details.get("gated"),
        files=files,
    )
    if key in ("sam3", "sam31"):
        from datetime import datetime
        from kwater_lab.train_head import file_hash
        from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

        name = "sam3.pt" if key == "sam3" else "sam3.1_multiplex.pt"
        record = next(f for f in details["siblings"] if f["rfilename"] == name)
        info["files"] = [record]
        url = hf_hub_url(repo, name, revision=rev)
        try:
            meta = get_hf_file_metadata(url, token=True, timeout=15)
            info.update(
                configured_http_status=200,
                access_state="approved",
                weight_bytes=meta.size,
            )
        except Exception as err:
            response = getattr(err, "response", None)
            status = response.status_code if response is not None else None
            state = (
                "pending_review"
                if "awaiting a review" in str(err).lower()
                else "access_unavailable"
            )
            info.update(
                configured_http_status=status,
                access_state=state,
                error_type=type(err).__name__,
            )
        info.update(
            weight_name=name, checked_at=datetime.now().astimezone().isoformat()
        )
        write(dest / "access-latest.json", info)
        if not acquire_gated or info["configured_http_status"] != 200:
            return
        if (RUN / "large-models/transfer-pause.flag").exists():
            raise RuntimeError(
                "Model access approved; sequential weight transfer awaits release bandwidth"
            )
        snapshot_download(
            repo_id=repo,
            revision=rev,
            local_dir=str(dest),
            token=True,
            max_workers=1,
            allow_patterns=[
                name,
                "*.json",
                "*.txt",
                "*.jinja",
                "README.md",
                "LICENSE*",
            ],
        )
        path = dest / name
        if (
            path.stat().st_size != record["size"]
            or file_hash(path) != record["lfs"]["sha256"]
        ):
            raise ValueError("Official SAM checkpoint failed its pinned LFS digest")
        write(
            dest / "complete.json",
            dict(
                revision=rev,
                weight=name,
                bytes=record["size"],
                sha256=record["lfs"]["sha256"],
            ),
        )
        return
    write(dest / "source.json", info)
    if only:
        files = [f for f in files if f["rfilename"] == only]
        if len(files) != 1:
            raise ValueError("Requested source file was not found")
    with ThreadPoolExecutor(max_workers=1) as pool:
        limit = mib_per_second * 2**20 if mib_per_second is not None else None
        list(
            pool.map(
                lambda item: acquire(item, dest, repo, rev, bytes_per_second=limit),
                files,
            )
        )
    output = (
        (dest / "transfers" / (only + ".json")) if only else (dest / "complete.json")
    )
    write(
        output,
        dict(
            revision=rev, verified_files=len(files), bytes=sum(f["size"] for f in files)
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=SPECS)
    ap.add_argument("--mib-s", type=float)
    ap.add_argument("--only")
    ap.add_argument("--acquire-gated", action="store_true")
    args = ap.parse_args()
    if args.mib_s is not None and args.mib_s <= 0:
        ap.error("Transfer limit must be positive")
    fetch(args.model, args.mib_s, args.only, args.acquire_gated)
