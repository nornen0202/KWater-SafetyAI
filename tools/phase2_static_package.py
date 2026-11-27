"""Resolve Linux inference wheels against the organizer image and total actual payload bytes."""

import argparse
from email import message_from_bytes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import urllib.request
from zipfile import ZipFile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


def resolve(project, run):
    base_text = (project / "data/local-kit/README.txt").read_text(encoding="utf8")
    base = {
        canonicalize_name(k): v
        for k, v in re.findall(r"^    ([\w.-]+)==([\w.+-]+)\s*$", base_text, re.M)
    }
    assert base["torch"] == "2.11.0+cu128" and base["transformers"] == "4.52.4"
    candidates = [
        "transformers==5.5.4",
        "peft==0.19.1",
        "bitsandbytes==0.49.2",
        "accelerate==1.13.0",
    ]
    environment = dict(
        default_environment(),
        sys_platform="linux",
        platform_system="Linux",
        os_name="posix",
        platform_machine="x86_64",
        python_version="3.12",
        python_full_version="3.12.10",
        extra="",
    )
    platforms = [f"manylinux_2_{n}_x86_64" for n in range(39, 16, -1)] + [
        "manylinux2014_x86_64",
        "linux_x86_64",
    ]
    tags = list(cpython_tags((3, 12), platforms=platforms)) + list(
        compatible_tags((3, 12), "cp312", platforms=platforms)
    )
    ranks = {tag: i for i, tag in enumerate(tags)}
    folder = run / "linux-wheelhouse"
    folder.mkdir(exist_ok=True)
    pinned = {}
    env_used = {}
    while candidates:
        req = Requirement(candidates.pop(0))
        if req.marker and not req.marker.evaluate(environment):
            continue
        name = canonicalize_name(req.name)
        if name in pinned:
            if pinned[name]["version"] not in req.specifier:
                raise ValueError(f"Conflicting requirements: {req}")
            continue
        if name in base and base[name] in req.specifier:
            env_used[name] = base[name]
            continue
        version = importlib.metadata.version(name)
        if version not in req.specifier:
            raise ValueError(f"Training runtime pin does not satisfy {req}: {version}")
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/{version}/json", timeout=30
        ) as response:
            alternatives = json.load(response)["urls"]
        viable = []
        for a in alternatives:
            if not a["filename"].endswith(".whl"):
                continue
            wheel_tags = parse_wheel_filename(a["filename"])[-1]
            priority = min((ranks[t] for t in wheel_tags if t in ranks), default=None)
            if priority is not None:
                viable.append((priority, a["filename"], a))
        if not viable:
            raise ValueError(f"No CPython 3.12 Linux wheel for {req}")
        _, filename, a = min(viable)
        path = folder / filename
        if not path.exists():
            partial = path.with_suffix(".partial")
            with (
                urllib.request.urlopen(a["url"], timeout=180) as src,
                partial.open("wb") as dst,
            ):
                while block := src.read(2**20):
                    dst.write(block)
            partial.replace(path)
        with path.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        assert sha == a["digests"]["sha256"]
        pinned[name] = {
            "version": version,
            "wheel": filename,
            "bytes": path.stat().st_size,
            "sha256": sha,
        }
        with ZipFile(path) as z:
            meta = message_from_bytes(
                z.read(
                    next(x for x in z.namelist() if x.endswith(".dist-info/METADATA"))
                )
            )
            candidates.extend(meta.get_all("Requires-Dist", []))
    (folder.parent / "linux-requirements.lock").write_text(
        "".join(
            f"{k}=={v['version']} --hash=sha256:{v['sha256']}\n"
            for k, v in sorted(pinned.items())
        ),
        encoding="utf8",
    )
    total = sum(v["bytes"] for v in pinned.values())
    models = {}
    for key, folder in [
        ("3vl", project / "data/baseline/base_model"),
        ("35", run / "qwen35"),
    ]:
        files = [
            p
            for p in folder.iterdir()
            if p.is_file() and not p.name.endswith(".partial")
        ]
        sizes = {p.name: p.stat().st_size for p in files}
        adapter = run / ("lora-" + key) / "adapter"
        adapter_bytes = sum(p.stat().st_size for p in adapter.glob("*") if p.is_file())
        models[key] = {
            "source_files": sizes,
            "weights_and_configs_bytes": sum(sizes.values()),
            "adapter_bytes": adapter_bytes,
            "static_payload_bytes_before_wrapper_zip_headers": sum(sizes.values())
            + total
            + adapter_bytes,
        }
    (run / "static-package.json").write_text(
        json.dumps(
            {
                "linux_wheels": pinned,
                "organizer_packages_used": env_used,
                "wheel_bytes": total,
                "models": models,
                "linux_execution_measured": False,
            },
            indent=2,
        ),
        encoding="utf8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--project", type=Path, default=Path("G:/Projects/KWater-SafetyAI")
    )
    a = parser.parse_args()
    resolve(a.project, a.run)
