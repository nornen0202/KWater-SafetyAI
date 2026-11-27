"""Run the organizer's unmodified ZIP extractor and manifest check and hash the submitted bytes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile


class DigestReader:
    """File-like archive accepted by zipfile and the organizer's path size check.

    Forward extraction supplies almost all bytes in order. ZIP metadata can be
    read out of order. Hash each byte in physical order exactly once, finishing
    outstanding bytes from disk if a reader skipped them.
    """

    def __init__(self, path, handle):
        self.path = path
        self.handle = handle
        self.digest = hashlib.sha256()
        self.count = 0

    def stat(self):
        return self.path.stat()

    def tell(self):
        return self.handle.tell()

    def seek(self, offset, whence=0):
        return self.handle.seek(offset, whence)

    def seekable(self):
        return True

    def read(self, size=-1):
        at = self.tell()
        if self.count < at <= self.count + 65536:
            self.seek(self.count)
            gap = self.handle.read(at - self.count)
            self.digest.update(gap)
            self.count += len(gap)
        block = self.handle.read(size)
        if at <= self.count < at + len(block):
            self.digest.update(memoryview(block)[self.count - at:])
            self.count = at + len(block)
        return block

    def finish(self):
        self.seek(self.count)
        while self.read(8 << 20):
            pass
        if self.count != self.stat().st_size:
            raise ValueError("Incomplete archive hash")
        return self.digest.hexdigest()


def inventory(path):
    with ZipFile(path) as archive:
        return {entry.filename: (entry.file_size, entry.CRC) for entry in archive.infolist()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "archive", "kit", "extract_to", "receipt"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    sys.path.insert(0, str(args.kit))
    import zipcheck

    before, after = inventory(args.baseline), inventory(args.archive)
    choice = set(after) - set(before)
    modules = {"article_scope.py", "risk_text.py", "risk_observation.py", "redundant_hazards.py", "assets/redundancy.json"}
    if not choice or not choice <= modules:
        raise ValueError(f"Unexpected file additions {choice}")
    changed = {"main.py"}
    if "redundant_hazards.py" in choice:
        if "assets/redundancy.json" not in choice:
            raise ValueError("Missing frozen text case configuration")
        changed.add("assets.json")
    if set(before) - set(after) or {k for k in before if before[k] != after[k]} != changed:
        raise ValueError("Unexpected changes to existing assets, code, weights or dependencies")
    size = args.archive.stat().st_size
    if size > 20_000_000_000:
        raise ValueError("Archive over competition limit")
    start = time.perf_counter()
    with args.archive.open("rb") as handle:
        reader = DigestReader(args.archive, handle)
        zipcheck.extract(reader, args.extract_to)
        extracted_hash_bytes = reader.count
        sha = reader.finish()
    seed, lock = zipcheck.manifest(args.extract_to)
    if (seed, lock) != ("42", True):
        raise ValueError("Organizer manifest check did not return expected seed/lock")
    record = {"path":str(args.archive.resolve()),"bytes":size,"sha256":sha,
              "member_count":len(after),"original_member_count":len(before),
              "unchanged_member_count":len(before)-len(changed),"changed_original_members":sorted(changed),
              "added_members":sorted(choice),"unchanged_comparison":"uncompressed length and CRC32 of every original member; organizer's extractor validated each decompressed member",
              "nominal_byte_limit":20_000_000_000,"official_extractor":"zipcheck.extract",
              "extractor_sha256":hashlib.sha256(Path(zipcheck.__file__).read_bytes()).hexdigest(),
              "official_manifest":{"seed":seed,"lock":lock},"archive_bytes_hashed_during_extract":extracted_hash_bytes,
              "elapsed_seconds_extraction_and_hash":round(time.perf_counter()-start,3),
              "safe_extracted_at":str(args.extract_to),"inventory":{k:{"bytes":v[0],"crc32":f"{v[1]:08x}"} for k,v in after.items()}}
    args.receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n","utf-8")
    print(sha, size, "seed=42 lock=1", str(args.archive), flush=True)


if __name__ == "__main__":
    main()
