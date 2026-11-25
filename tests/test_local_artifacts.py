import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kwater_lab.local_artifacts import inventory


class LocalInventoryTests(unittest.TestCase):
    def test_metadata_only_and_append_checks_known_sizes(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            data = root / "data"
            data.mkdir()
            (root / "results").mkdir()
            old = data / "private.jsonl"
            old.write_bytes(b"private row")
            (data / "pending.zip.partial").write_bytes(b"in progress")
            (data / "pending.zip.assembling").write_bytes(b"unfinished")
            status_dir = data / "external/aihub/507"
            status_dir.mkdir(parents=True)
            (status_dir / "transfer-status.json").write_text("{}")
            original = inventory(root)
            self.assertEqual(original["artifacts"], [{"path": "data/private.jsonl", "bytes": 11,
                "sha256": hashlib.sha256(b"private row").hexdigest()}])
            (data / "later.txt").write_bytes(b"another private row")
            self.assertEqual(inventory(root, original), inventory(root))
            old.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "changed size"):
                inventory(root, original)
            old.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                inventory(root, original)

    def test_refresh_rehashes_a_same_size_revised_design(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data/prompt-pilot"
            folder.mkdir(parents=True)
            design = folder / "design.json"
            design.write_bytes(b'{"arm":"a"}')
            original = inventory(root)
            design.write_bytes(b'{"arm":"b"}')
            revised = inventory(root, original, refresh_derived=True)
            self.assertNotEqual(original["artifacts"][0]["sha256"], revised["artifacts"][0]["sha256"])
            self.assertEqual(revised, inventory(root))

    def test_new_aihub_packages_use_the_completed_download_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            packages = root / "data/external/aihub/507/packages"
            packages.mkdir(parents=True)
            old = inventory(root)
            archive = packages / "2-example.zip"
            archive.write_bytes(b"example")
            self.assertEqual(inventory(root, old), old)
            receipt = {"name": "example.zip", "dataset": "507", "filekey": "2", "bytes": 7, "zip_members": 1,
                       "sha256": hashlib.sha256(b"example").hexdigest()}
            archive.with_suffix(".zip.json").write_text(json.dumps(receipt))
            self.assertEqual(inventory(root, old), inventory(root))
            receipt["bytes"] = 99
            archive.with_suffix(".zip.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "receipt differ"):
                inventory(root, old)

    def test_new_floodnet_archive_receipt_and_full_verification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data/external/floodnet"
            folder.mkdir(parents=True)
            old = inventory(root)
            archive = folder / "FloodNet.zip"
            archive.write_bytes(b"public example")
            receipt = {"source_url": "https://www.dropbox.com/scl/fo/example", "bytes": archive.stat().st_size,
                       "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            local = archive.with_suffix(".zip.json")
            local.write_text(json.dumps(receipt))
            self.assertEqual(inventory(root, old), inventory(root))
            receipt["source_url"] = "https://www.dropbox.com.example.org/scl/fo/example"
            local.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "acquisition receipt"):
                inventory(root, old)
            receipt["source_url"] = "https://www.dropbox.com/scl/fo/example"
            receipt["bytes"] += 1
            local.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "acquisition receipt"):
                inventory(root, old)
            # A later archive update must be hashed from disk even if the receipt still exists.
            newer = local.stat().st_mtime_ns + 2_000_000_000
            os.utime(archive, ns=(newer, newer))
            self.assertEqual(inventory(root, old), inventory(root))
