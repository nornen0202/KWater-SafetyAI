"""Measure incremental, stored Linux ZIP payloads for the local engineering trial."""

import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

from kwater_lab.phase2_panel import PROJECT, RUN, file_hash, read, write


def build():
    root = RUN / "regions"
    out = root / "package-increments"
    out.mkdir(exist_ok=True)
    previous = PROJECT / ".runtime/submission2-zipcheck-final"
    license_file = previous / "licenses/siglip2-APACHE-2.0.txt"
    specs = read(root / "models/receipts.json")
    groups = {}
    for key, spec in specs.items():
        groups[key] = [
            (root / "models" / key / f["rfilename"], f"models/{key}/" + f["rfilename"])
            for f in spec["files"]
        ]
        groups[key].append((license_file, f"licenses/{key}-Apache-2.0.txt"))
    depth = Path("G:/CodexData/kwater-rd-20261125/depth-anything-v2-small")
    groups["depth"] = [
        (depth / f, "models/depth/" + f)
        for f in ("model.onnx", "preprocessor_config.json")
    ] + [(license_file, "licenses/depth-Apache-2.0.txt")]
    files = [
        previous / "requirements.lock",
        *sorted((previous / "wheelhouse").glob("*.whl")),
        *sorted((previous / "licenses").glob("onnxruntime*")),
        *sorted((previous / "licenses").glob("flatbuffers*")),
    ]
    groups["cpu_runtime"] = [(f, f.relative_to(previous).as_posix()) for f in files]
    report = {}
    for key, items in groups.items():
        p = out / (key + ".zip")
        with ZipFile(p, "w", ZIP_STORED) as z:
            for path, name in items:
                z.write(path, name)
            if key in specs:
                z.writestr(
                    "assets/" + key + ".json", json.dumps(specs[key], sort_keys=True)
                )
        with ZipFile(p) as z:
            if z.testzip():
                raise ValueError("Archive integrity check failed")
            members = {i.filename: i.file_size for i in z.infolist()}
        report[key] = {
            "bytes": p.stat().st_size,
            "sha256": file_hash(p),
            "contents": members,
        }
        print("PACKED", key, report[key]["bytes"], flush=True)
    write(
        root / "package.json",
        {
            "separate_zip_increments": report,
            "compression": "ZIP_STORED",
            "scope": "Each standalone measured ZIP includes its own headers. Shared runtime counted once; local experiment runner and Qwen payload separate.",
            "local_provider": "onnxruntime 1.26.0 CPUExecutionProvider; no network calls during model inference",
            "l4_cold_seconds": None,
        },
    )


if __name__ == "__main__":
    build()
