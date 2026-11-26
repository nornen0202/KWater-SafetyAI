"""Build a ZIP from the pinned development-only snapshot and upstream Linux wheels."""

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile
from email.parser import BytesParser
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
SOURCES = [
    "metrics.py",
    "phase01_vision.py",
    "phase01_infer.py",
    "phase01_head.py",
    "submission2_output.py",
]
WEIGHTS = {
    "models/sig/model.safetensors": "9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2",
    "models/dino/model.safetensors": "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    "single-dev.pt": "bd83ca5c175e56adea5b4851009a7c71d909db3f66b4038fad8e7e3149637302",
    "policy.json": "895ac20771d4565baf363a34d89d74491c3048c2a1c870997696e9ea6dcfd95f",
}
WHEEL_VERSIONS = {
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "accelerate": "1.10.1",
    "huggingface-hub": "0.36.0",
}
SOURCES_BASE = {
    "torch": ("2.11.0+cu128", "BSD-3-Clause"),
    "numpy": ("2.2.6", "BSD-3-Clause"),
    "Pillow": ("11.3.0", "MIT-CMU"),
    "safetensors": ("0.8.0", "Apache-2.0"),
}


def dump(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf8",
        newline="\r\n",
    )


def build(run, out, extras):
    stage = out / "stage"
    dev = run / "phase3-development"
    paths = {}
    for name in [
        *WEIGHTS,
        "Apache-2.0.txt",
        *[
            f"models/{back}/{leaf}.json"
            for back in ("sig", "dino")
            for leaf in ("config", "preprocessor_config")
        ],
    ]:
        paths[name] = dev / name
    cfg = json.loads(paths["policy.json"].read_bytes())
    assert (
        cfg["head"] == "single-dev.pt"
        and not cfg["multiscale"]
        and cfg["fit_photos"] == cfg["calibration_photos"] == 920
    )
    assert cfg["policy"]["thresholds"] == [0.2] * 11
    for name in (
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        rel = "models/sig/" + name
        paths[rel] = extras / rel
    paths["main.py"] = REPO / "submission/phase01/main.py"
    paths["kwater_lab/__init__.py"] = dev / "kwater_lab/__init__.py"
    for name in SOURCES:
        paths["kwater_lab/" + name] = REPO / "kwater_lab" / name
    (stage / "licenses").mkdir(exist_ok=True, parents=True)
    assets = json.loads((dev / "assets.json").read_bytes())
    assets["assets"] = [a for a in assets["assets"] if a["kind"] != "library"]

    def library(name, version, license, use):
        assets["assets"].append(
            dict(
                kind="library",
                name=name,
                revision=version,
                license=license,
                use=use,
                source=f"https://pypi.org/project/{name}/",
            )
        )

    for name, (version, license) in SOURCES_BASE.items():
        library(name, version, license, "provided by pinned official runtime")
    lock = []
    seen = set()
    for wheel in sorted((extras / "wheelhouse").glob("*.whl")):
        with zipfile.ZipFile(wheel) as z:
            meta = BytesParser().parsebytes(
                z.read(
                    next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
                )
            )
            name = meta["Name"].lower().replace("_", "-")
            if name not in WHEEL_VERSIONS:
                continue
            assert meta["Version"] == WHEEL_VERSIONS[name] and name not in seen
            seen.add(name)
            library(name, meta["Version"], "Apache-2.0", "offline Linux wheel overlay")
            found = [
                n
                for n in z.namelist()
                if ".dist-info/" in n
                and ("license" in n.lower() or "notice" in n.lower())
                and not n.endswith("/")
            ]
            if not found:
                upstream = extras / "licenses" / name / "LICENSE"
                assert upstream.is_file(), (name, "missing upstream license")
                paths[f"licenses/{name}/LICENSE"] = upstream
            for n in found:
                dest = f"licenses/{name}/{n.split('.dist-info/', 1)[1]}"
                paths[dest] = stage / dest
                paths[dest].parent.mkdir(exist_ok=True, parents=True)
                paths[dest].write_bytes(z.read(n))
        h = hashlib.sha256(wheel.read_bytes()).hexdigest()
        lock.append(f"{name}=={meta['Version']} --hash=sha256:{h}")
        paths["wheelhouse/" + wheel.name] = wheel
    assert seen == set(WHEEL_VERSIONS)
    for name in ("assets.json", "requirements.lock"):
        paths[name] = stage / name
    dump(paths["assets.json"], assets)
    paths["requirements.lock"].write_text(
        "\n".join(sorted(lock)) + "\n", encoding="ascii", newline="\r\n"
    )
    img = Image.new("RGB", (640, 480), (184, 211, 234))
    draw = ImageDraw.Draw(img)
    draw.polygon(
        [(0, 330), (410, 180), (640, 325), (640, 480), (0, 480)], fill=(90, 118, 88)
    )
    draw.rectangle((138, 242, 402, 317), fill=(97, 107, 110))
    for x in range(140, 405, 31):
        draw.line((x, 242, x, 215), fill=(230, 181, 55), width=4)
    draw.line((140, 215, 402, 215), fill=(230, 181, 55), width=4)
    eid = "00000000000000000000000000000001"
    p = f"example/images/{eid}.jpg"
    paths[p] = stage / p
    paths[p].parent.mkdir(exist_ok=True, parents=True)
    img.save(paths[p], quality=87)
    p = "example/manifest.jsonl"
    paths[p] = stage / p
    paths[p].write_text(
        json.dumps({"image_id": eid}) + "\n", encoding="utf8", newline="\r\n"
    )
    archive = out / "kwater-phase1-dev920-linux-x86_64.zip"
    tmp = archive.with_suffix(".partial")
    members = {}
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for name, src in sorted(paths.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 26, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.file_size = src.stat().st_size
            h = hashlib.sha256()
            with (
                z.open(info, "w", force_zip64=info.file_size >= 2**31) as packed,
                src.open("rb") as f,
            ):
                while buf := f.read(4 * 1024 * 1024):
                    h.update(buf)
                    packed.write(buf)
            members[name] = {"bytes": info.file_size, "sha256": h.hexdigest()}
            if name in WEIGHTS:
                assert h.hexdigest() == WEIGHTS[name], name
    assert tmp.stat().st_size < 20_000_000_000
    tmp.replace(archive)
    with archive.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    size = archive.stat().st_size
    dump(
        out / "zip-manifest.json",
        dict(
            archive=archive.name,
            bytes=size,
            sha256=sha,
            GB=size / 1e9,
            GiB=size / 2**30,
            headroom_bytes=20_000_000_000 - size,
            members=members,
        ),
    )
    print(archive, size, sha, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for arg in ("run", "out", "extras"):
        ap.add_argument("--" + arg, required=True, type=Path)
    a = ap.parse_args()
    build(a.run, a.out, a.extras)
