import random
from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from requests.exceptions import ChunkedEncodingError

from tools import fetch_aihub
from tools.fetch_aihub import OVERLAP, receive


class Response:
    def __init__(self, body, content_range=None, interrupt=False):
        self.body = body
        self.status_code = 502  # The provider reports 502 even for usable content.
        self.headers = {"Content-Range": content_range} if content_range else {}
        self.interrupt = interrupt

    def iter_content(self, chunk_size):
        for n in range(0, len(self.body), 2201):
            yield self.body[n:n + 2201]
        if self.interrupt:
            raise ChunkedEncodingError("interrupted")


class AihubRangeTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.partial = Path(directory.name) / "sample.tar.partial"
        output = BytesIO()
        with ZipFile(output, "w") as z:
            z.writestr("test.bin", random.Random(42).randbytes(4 * OVERLAP))
        self.full = output.getvalue()
        self.offset = 2 * OVERLAP
        self.partial.write_bytes(self.full[:self.offset])

    def test_range_checks_local_overlap_even_with_nonstandard_server_status(self):
        start = self.offset - OVERLAP
        for header in (None, f"bytes {start}-{len(self.full)-1}/{len(self.full)}"):
            with self.subTest(header=header):
                self.partial.write_bytes(self.full[:self.offset])
                target, more = receive(Response(self.full[start:], header), self.partial, self.offset)
                self.assertEqual(target.read_bytes(), self.full)
                self.assertFalse(more)

    def test_replayed_start_cannot_destroy_a_longer_previous_partial(self):
        for count in (self.offset // 2, self.offset + 20000):
            with self.subTest(count=count):
                response = Response(self.full[:count], interrupt=True)
                with self.assertRaises(ChunkedEncodingError):
                    receive(response, self.partial, self.offset)
                self.assertEqual(self.partial.read_bytes(), self.full[:max(count, self.offset)])

    def test_mismatched_or_unrelated_response_cannot_be_appended(self):
        start = self.offset - OVERLAP
        corrupted = bytes([self.full[start] ^ 1]) + self.full[start + 1:]
        for response in (Response(b"denied"), Response(corrupted, f"bytes {start}-{len(self.full)-1}/{len(self.full)}")):
            with self.assertRaises((ValueError, RuntimeError)):
                receive(response, self.partial, self.offset)
            self.assertEqual(self.partial.read_bytes(), self.full[:self.offset])

    def test_bounded_range_requests_more_bytes(self):
        start, end = self.offset - OVERLAP, self.offset + 4000
        response = Response(self.full[start:end], f"bytes {start}-{end-1}/{len(self.full)}")
        target, more = receive(response, self.partial, self.offset)
        self.assertTrue(more)
        self.assertEqual(target.read_bytes(), self.full[:end])

    def test_fetch_commits_complete_package_and_reuses_its_receipt(self):
        root = self.partial.parent
        packages = root / "71793/packages"
        packages.mkdir(parents=True)
        target = packages / "77.tar.partial"
        target.write_bytes(self.full[:self.offset])
        row = {"filekey": "77", "name": "TS.zip"}
        start = self.offset - OVERLAP
        body = Response(self.full[start:], f"bytes {start}-{len(self.full)-1}/{len(self.full)}")
        with patch.object(fetch_aihub, "ROOT", root), patch.object(fetch_aihub, "provider_response", return_value=nullcontext(body)) as provider:
            fetch_aihub.fetch("71793", row, "test-only")
            fetch_aihub.fetch("71793", row, "test-only")
            provider.assert_called_once()
        final = packages / "77-TS.zip"
        receipt = json.loads((packages / "77-TS.zip.json").read_text(encoding="utf-8"))
        self.assertEqual(final.read_bytes(), self.full)
        self.assertEqual(receipt["sha256"], sha256(self.full).hexdigest())
        self.assertFalse(target.exists())
