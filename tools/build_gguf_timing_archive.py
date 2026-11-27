"""Build the 27B timing ZIP from pinned weights, release binaries and the sixth submission."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

REVISION = "4af51b3542f775b0a8e4b310bfb3f17e01c306c9"
WEIGHTS = {
    "model.gguf": ("Qwen3.8-27B-UD-Q4_K_M.gguf", "1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529"),
    "mmproj.gguf": ("mmproj-BF16.gguf", "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53"),
}
BINARY_SHA = "f150cc31b798b334a11682812897e287c3e8041912df9936f95d52a2f9d5f7f5"
# Exact members of the accepted sixth ZIP that this timing package reuses.
REFERENCE_SHA = {
    "main.py": "b728c3be183cc3cfa4e49ee8bc38076ea7b572c8b91a11c159091eee3a9b1e98",
    "assets.json": "6d58ea41fe4d51155143c6509ee188da2f253583fe544ee8097fd9ec6a0b6448",
    "assets/cases_11code.jsonl": "b18b0d03534972b55876dbdff01fb7aaa772920257f11653023c8281c60ff7ad",
    "assets/laws.jsonl": "1408b69db444cf90b31f9bebb4a773a30fa4362739e50b22c95f3bbe0e75bd9d",
    "system_prompt_v3.txt": "13edaf96ccd032551ea0a1ebdc27934d958afd1259439f4b05773bdde1b2e07f",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_binary(name):
    return name in ("llama-server", "LICENSE", "libllama-server-impl.so", "libggml-cuda.so") or name.startswith((
        "libggml-cpu-", "libllama.so", "libllama-common.so", "libmtmd.so", "libggml.so", "libggml-base.so"))


def entry(name):
    info = zipfile.ZipInfo(name, (2026, 11, 28, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(weights, binaries, sixth, entrypoint, output):
    if digest(binaries) != BINARY_SHA:
        raise ValueError("Runtime bundle fingerprint differs")
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_suffix(".partial")
    with zipfile.ZipFile(partial, "w", allowZip64=True) as z, zipfile.ZipFile(sixth) as original:
        for name, checksum in REFERENCE_SHA.items():
            if hashlib.sha256(original.read(name)).hexdigest() != checksum:
                raise ValueError(f"Sixth submission reference differs: {name}")
        for dest, (name, checksum) in WEIGHTS.items():
            observed = hashlib.sha256()
            with (weights / name).open("rb") as src, z.open(entry(dest), "w", force_zip64=True) as sink:
                while chunk := src.read(4 << 20):
                    observed.update(chunk)
                    sink.write(chunk)
            if observed.hexdigest() != checksum:
                raise ValueError(f"Weight checksum mismatch: {name}")
        # Materialize shared-library SONAME aliases since the kit extracts regular files.
        members = {}
        with tarfile.open(binaries, "r|gz") as tar:
            for member in tar:
                name = PurePosixPath(member.name).name
                if selected_binary(name):
                    if member.issym():
                        members[name] = PurePosixPath(member.linkname).name
                    elif member.isfile():
                        members[name] = tar.extractfile(member).read()
        for name, value in sorted(members.items()):
            while isinstance(value, str):
                value = members[value]
            z.writestr(entry("runtime/" + name), value)
        for name in ("assets/cases_11code.jsonl", "assets/laws.jsonl", "system_prompt_v3.txt"):
            z.writestr(entry(name), original.read(name))
        z.writestr(entry("format_runtime.py"), original.read("main.py"))
        z.writestr(entry("main.py"), entrypoint.read_bytes())
        prior = json.loads(original.read("assets.json"))
        assets = [a for a in prior["assets"] if a["kind"] == "data"]
        assets.extend([
            {"kind": "model", "name": "Qwen3.8-27B GGUF UD-Q4_K_M and BF16 projector",
             "source": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF", "revision": REVISION,
             "license": "Apache-2.0", "use": "Offline multimodal timing probe, no domain adapter"},
            {"kind": "library", "name": "llama.cpp CUDA 12.8", "source": "https://github.com/ggml-org/llama.cpp",
             "revision": "b12721", "license": "MIT", "use": "Local Unix-socket multimodal inference"},
        ])
        z.writestr(entry("assets.json"), json.dumps({"schema_version": "1.0", "seed": 42, "assets": assets}))
    if partial.stat().st_size > 20_000_000_000:
        raise ValueError("Timing archive exceeds the project upload limit")
    partial.replace(output)
    print(json.dumps({"bytes": output.stat().st_size, "sha256": digest(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "binaries", "sixth", "entrypoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))
