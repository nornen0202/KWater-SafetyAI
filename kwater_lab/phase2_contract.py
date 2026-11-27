"""Run the packaged second entry on the frozen development images and check both kit receipts."""

import json
import os
import subprocess
import sys
import time
from zipfile import ZipFile

from .phase2_panel import PROJECT, RUN, file_hash, order, read, registry, write


def check(inp, out):
    checker = PROJECT / "data/local-kit/kit/check_output.py"
    py = PROJECT / ".runtime/baseline-venv/Scripts/python.exe"
    result = subprocess.run(
        [str(py), "-X", "utf8", str(checker), str(inp), str(out)],
        capture_output=True,
        encoding="utf8",
        check=True,
    )
    return result.stdout


def run():
    package = PROJECT / "data/submissions/kwater-fusion-20261125.zip"
    expected = read(PROJECT / "results/public_submission_002.json")["sha256"]
    assert file_hash(package) == expected
    folder = PROJECT / ".runtime/submission2-zipcheck-final"
    with ZipFile(package) as z:
        import hashlib

        for f in z.infolist():
            if not f.is_dir():
                with z.open(f) as stream:
                    assert hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest() == file_hash(folder / f.filename)
    rows = registry()
    ids = read(RUN / "plan.json")["panel"]
    inp, out = RUN / "fusion-input", RUN / "fusion-panel"
    (inp / "images").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sid in ids:
        anon = order("kwater/scale", sid)[:32]
        dest = inp / "images" / (anon + ".jpg")
        src = PROJECT / "data/train" / rows[sid]["image"]
        if not dest.exists():
            os.link(src, dest)
        assert file_hash(dest) == rows[sid]["sha256"]
        manifest.append(json.dumps({"image_id": anon}))
    (inp / "manifest.jsonl").write_text("\n".join(manifest) + "\n", encoding="utf8")
    start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(folder / "main.py"),
            "--input-dir",
            str(inp),
            "--output-dir",
            str(out),
        ],
        check=True,
    )
    elapsed = time.perf_counter() - start
    kit = PROJECT / "data/local-kit/sample-input"
    receipt = {
        "submitted2_zip_sha256": expected,
        "panel_seconds": elapsed,
        "panel_checker": check(inp, out),
        "fusion_kit": check(kit, RUN / "fusion-kit"),
        "original_qwen_kit_cache": check(
            kit, PROJECT / "data/official-baseline-10/output"
        ),
        "fusion_kit_sha256": file_hash(RUN / "fusion-kit/submission.jsonl"),
        "original_qwen_kit_sha256": file_hash(
            PROJECT / "data/official-baseline-10/output/submission.jsonl"
        ),
    }
    prior = read(PROJECT / "results/public_submission_001.json")
    first = PROJECT / "data/packages" / prior["submitted_archive"]
    assert file_hash(first) == prior["sha256"]
    receipt["submitted1_zip_sha256"] = prior["sha256"]
    receipt["submitted1_bytes"], receipt["submitted2_bytes"] = (
        first.stat().st_size,
        package.stat().st_size,
    )
    write(RUN / "contracts.json", receipt)


if __name__ == "__main__":
    run()
