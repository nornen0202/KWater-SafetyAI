"""Expose only the named rehearsal inputs on localhost to a local QEMU guest."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

ap = argparse.ArgumentParser()
for arg in ("run", "kit"):
    ap.add_argument("--" + arg, required=True, type=Path)
ap.add_argument("--port", type=int, default=18439)
a = ap.parse_args()
runtime = a.run / "vm/runtime"
manifest = runtime / "manifest.json"
m = json.loads(manifest.read_bytes())
files = {
    "/umoci": a.run / "vm" / "umoci.linux.amd64",
    "/manifest.json": manifest,
    "/submission.zip": a.run / "kwater-phase1-dev920-linux-x86_64.zip",
    "/benchmark.py": Path(__file__).with_name("item39_linux_benchmark.py"),
    "/prepare.sh": Path(__file__).with_name("item39_vm_prepare.sh"),
}
for position in ("config", "layers"):
    entries = m[position] if position == "layers" else [m[position]]
    for item in entries:
        digest = item["digest"][7:]
        files["/blobs/" + digest] = runtime / digest
for name in (
    "check_zip.py",
    "zipcheck.py",
    "check_output.py",
    "prediction.schema.json",
    "sandbox_exec.py",
):
    files["/kit/" + name] = a.kit / name


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        return str(files.get(path, runtime / "no-such-resource"))

    def list_directory(self, path):
        self.send_error(404)
        return None


ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
