import hashlib
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
