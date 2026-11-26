"""Stage the exact public wheels, SigLIP tokenizer and license used in the audited ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from email.parser import BytesParser

import requests
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def fetch(url, path, digest):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    h = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with temp.open("wb") as out:
            for block in response.iter_content(1024 * 1024):
                h.update(block)
                out.write(block)
    if h.hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--kit-readme", type=Path)
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[1] / "results/item39_zip_manifest.json"
    members = json.loads(reference.read_bytes())["members"]
    revision = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
    for name, info in members.items():
        leaf = Path(name).name
        if name.startswith("wheelhouse/"):
            project, version = leaf.split("-")[:2]
            result = requests.get(
                f"https://pypi.org/pypi/{project}/{version}/json", timeout=20
            )
            result.raise_for_status()
            item = next(f for f in result.json()["urls"] if f["filename"] == leaf)
            if item["digests"]["sha256"] != info["sha256"]:
                raise ValueError(f"Unexpected PyPI digest for {leaf}")
            url = item["url"]
        elif name.startswith("models/sig/") and "token" in leaf:
            url = f"https://huggingface.co/google/siglip2-so400m-patch14-384/resolve/{revision}/{leaf}"
        elif name == "licenses/tokenizers/LICENSE":
            url = "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.1/LICENSE"
        else:
            continue
        fetch(url, args.stage / name, info["sha256"])
        print(info["sha256"], name)
    if args.kit_readme:
        base = dict(
            (canonicalize_name(n), v)
            for n, v in re.findall(
                r"^\s*([\w.-]+)==([^\s]+)\s*$",
                args.kit_readme.read_text(encoding="utf8"),
                re.M,
            )
        )
        env = dict(
            default_environment(),
            python_version="3.12",
            python_full_version="3.12.3",
            implementation_name="cpython",
            implementation_version="3.12.3",
            platform_python_implementation="CPython",
            platform_system="Linux",
            sys_platform="linux",
            os_name="posix",
            platform_machine="x86_64",
            extra="",
        )
        metadata = {}
        for name in members:
            if name.startswith("wheelhouse/"):
                with zipfile.ZipFile(args.stage / name) as wheel:
                    meta = BytesParser().parsebytes(
                        wheel.read(
                            next(
                                n
                                for n in wheel.namelist()
                                if n.endswith(".dist-info/METADATA")
                            )
                        )
                    )
                    metadata[canonicalize_name(meta["Name"])] = meta
        versions = dict(base, **{n: m["Version"] for n, m in metadata.items()})
        active = []
        for project, meta in metadata.items():
            for raw in meta.get_all("Requires-Dist", []):
                req = Requirement(raw)
                if req.marker and not req.marker.evaluate(env):
                    continue
                name = canonicalize_name(req.name)
                version = versions.get(name)
                if not version or not req.specifier.contains(version):
                    raise ValueError(
                        f"{project} requires {req}, environment supplies {version}"
                    )
                active.append(
                    dict(
                        project=project,
                        requires=str(req),
                        version=version,
                        source="wheel" if name in metadata else "official base",
                    )
                )
        if args.comparison:
            args.comparison.write_text(
                json.dumps(active, indent=2) + "\n", encoding="utf8"
            )
        print(len(active), "applicable wheel dependency declarations satisfied")


if __name__ == "__main__":
    main()
