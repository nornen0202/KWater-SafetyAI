import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

from kwater_lab.offline_archive_audit import entries, inspect_bundle, log_summary


class OfflineArchiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.kit = Path("data/local-kit/kit").resolve()
        root = Path(".runtime/temp")
        root.mkdir(parents=True, exist_ok=True)
        scratch = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)

    def bundle(self, wheelname="test_fixture-1.0-py3-none-any.whl", extra=None):
        payload = b"fixture wheel bytes; not an installable wheel"
        members = {"main.py": b"print('fixture')\n", "assets.json": json.dumps({
            "schema_version": "1.0", "seed": 42, "assets": [],
        }).encode(), "requirements.lock": f"test-fixture==1.0 --hash=sha256:{hashlib.sha256(payload).hexdigest()}\n".encode(),
                    f"wheelhouse/{wheelname}": payload, **(extra or {})}
        path = self.scratch / "fixture.zip"
        with ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return path

    def test_actual_kit_accepts_tiny_layout_and_dry_audit_identifies_scope(self):
        source = self.scratch / "protected-original.jsonl"
        source.write_bytes(b"restricted training fixture")
        path = self.bundle(extra={"runtime.log": b"long log fixture", "assets/renamed.jsonl": source.read_bytes()})
        check = subprocess.run([sys.executable, "-X", "utf8", str(self.kit / "check_zip.py"), str(path),
                                str(self.scratch / "extract")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
        with ZipFile(path) as archive:
            result = inspect_bundle(entries(archive), archive.open, self.kit, self.scratch, {}, [source])
        self.assertEqual(result["restricted_or_unneeded_members"], ["runtime.log"])
        self.assertEqual(result["restricted_source_exact_content_matches"], ["assets/renamed.jsonl"])
        self.assertEqual(len(result["offline_wheels"]), 1)
        self.assertFalse(result["offline_install_executed"])

    def test_offline_tag_and_traversal_rejections(self):
        path = self.bundle(wheelname="test_fixture-1.0-cp312-cp312-win_amd64.whl")
        with ZipFile(path) as archive, self.assertRaisesRegex(ValueError, "Linux x86-64"):
            inspect_bundle(entries(archive), archive.open, self.kit, self.scratch, {})
        raw = io.BytesIO()
        with ZipFile(raw, "w") as archive:
            archive.writestr("../outside.txt", "forbidden")
        with ZipFile(raw) as archive, self.assertRaisesRegex(ValueError, "Unsafe"):
            entries(archive)
        bad = self.scratch / "traversal.zip"
        bad.write_bytes(raw.getvalue())
        rejected = subprocess.run([sys.executable, "-X", "utf8", str(self.kit / "check_zip.py"), str(bad),
                                   str(self.scratch / "bad-extract")], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(rejected.returncode, 10, rejected.stdout + rejected.stderr)
        self.assertFalse((self.scratch / "outside.txt").exists())

    def test_log_counters_handle_chunk_boundary_without_copying_content(self):
        path = self.scratch / "fixture.log"
        path.write_bytes(b"x" * 66_000 + b"\ny\nlast")
        self.assertEqual(log_summary(path), {"bytes": 66_007, "lines": 3, "longest_line_bytes": 66_001,
                                            "lines_above_4096_bytes": 1})
