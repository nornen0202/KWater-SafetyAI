"""Verify downloaded archives on disk and emit the public-source ledger when complete."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import time
from zipfile import BadZipFile, ZipFile

from external_inventory import DATA, FLOODNET_SOURCE, ROOT, read, snapshot, write
from fetch_aihub import SETS, transfer_lock


BUFFER = 2 * 1024**2
PUBLIC = {
    "kalis/sample.zip": "https://www.data.go.kr/data/15142200/fileData.do",
    "floodnet/FloodNet.zip": FLOODNET_SOURCE,
    "kosha-confined/confined-space-2022.pdf": "https://www.moel.go.kr/local/boryeong/info/dataroom/view.do?bbs_seq=20230501649",
    "kosha-sif/sif-20260401.xlsx": "https://www.moel.go.kr/policy/policydata/view.do?bbs_seq=20260400196",
}
HEAT_FILES = ("20260501390", "20260501391", "20260501392", "20260501393", "20260501394")


class DigestReader:
    """Hash contiguous bytes once while zipfile seeks and checks entry CRCs."""

    def __init__(self, stream, length):
        self.stream = stream
        self.length = length
        self.hashed = 0
        self.digest = hashlib.sha256()
        self.enabled = False

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()

    def seekable(self):
        return True

    def include_gap(self, end):
        self.stream.seek(self.hashed)
        while self.hashed < end:
            block = self.stream.read(min(BUFFER, end - self.hashed))
            if not block:
                raise EOFError("Archive ended during readback")
            self.digest.update(block)
            self.hashed += len(block)

    def read(self, amount=-1):
        start = self.tell()
        if self.enabled and start > self.hashed:
            self.include_gap(start)
        block = self.stream.read(amount)
        stop = start + len(block)
        if self.enabled and stop > self.hashed:
            self.digest.update(memoryview(block)[self.hashed - start:])
            self.hashed = stop
        return block

    def finish(self):
        self.include_gap(self.length)
        return self.digest.hexdigest()


def check_bytes(path, size):
    with path.open("rb") as stream:
        if path.suffix.lower() not in {".zip", ".xlsx"}:
            if stream.read(5) != b"%PDF-":
                raise ValueError("Unexpected document signature")
            stream.seek(0)
            return hashlib.file_digest(stream, "sha256").hexdigest(), None
        reader = DigestReader(stream, size)
        with ZipFile(reader) as archive:
            entries = sorted(archive.infolist(), key=lambda i: i.header_offset)
            if not entries:
                raise BadZipFile("Empty archive")
            # Do not hash the central-directory seek until all preceding file bytes have been seen.
            reader.enabled = True
            for entry in entries:
                with archive.open(entry) as body:
                    while body.read(BUFFER):
                        pass
        return reader.finish(), len(entries)


def expected_files():
    items = {}
    for sid in SETS:
        source = "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=" + sid
        catalog = read(DATA / "aihub" / sid / "catalog.json")
        for row in catalog:
            name, key = row["name"], str(row["filekey"])
            if not key.isdecimal() or Path(name).name != name or not name.lower().endswith(".zip"):
                raise ValueError("Unexpected provider archive name or key")
            rel = f"aihub/{sid}/packages/{key}-{name}"
            if rel in items:
                raise ValueError("Duplicate provider entry")
            items[rel] = {"dataset": sid, "filekey": key, "filename": name, "source": source}
        if sid in {"71407", "71793"}:
            rel = f"aihub/{sid}/sample.zip"
            items[rel] = {"dataset": sid, "filename": "sample.zip", "source": source}
    other = dict(PUBLIC)
    for key in HEAT_FILES:
        other[f"moel-heat/{key}.pdf"] = "https://www.moel.go.kr/policy/policydata/view.do?bbs_seq=20260500439"
    for rel, url in other.items():
        path = Path(rel)
        items[rel] = {"dataset": path.parts[0], "filename": path.name, "source": url}
    return items


def verify(rel, meta):
    path = DATA / rel
    receipt = read(path.with_name(path.name + ".json"))
    stat = path.stat()
    digest = receipt["sha256"]
    if (not re.fullmatch(r"[0-9a-f]{64}", digest) or stat.st_size != receipt["bytes"]
            or ("filekey" in meta and (str(receipt["filekey"]), receipt["dataset"], receipt["name"])
                != (meta["filekey"], meta["dataset"], meta["filename"]))):
        raise ValueError("Acquisition receipt does not match file")
    stamp = {"mtime_ns": stat.st_mtime_ns, "bytes": stat.st_size, "sha256": digest, "format": 1}
    proof_path = path.with_name(path.name + ".integrity.json")
    proof = read(proof_path) if proof_path.exists() else {}
    if proof.get("stamp") != stamp:
        actual, members = check_bytes(path, stat.st_size)
        if actual != digest or ("zip_members" in receipt and receipt["zip_members"] != members):
            raise ValueError("Checksum or member count differs from acquisition receipt")
        stable = path.stat()
        if (stable.st_size, stable.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError("File changed during verification")
        proof = {"stamp": stamp, "members_crc_checked": members}
        write(proof_path, proof)
    return {**meta, "bytes": stat.st_size, "sha256": digest, "members_crc_checked": proof["members_crc_checked"]}


def ready_files(items):
    ready = set()
    for parent in sorted({(DATA / rel).parent for rel in items}):
        for receipt in parent.glob("*.json"):
            rel = receipt.relative_to(DATA).as_posix()[:-5]
            if rel in items:
                ready.add(rel)
    return ready


def unresolved(ready, items):
    acquired = {(items[rel]["dataset"], items[rel]["filekey"]) for rel in ready if "filekey" in items[rel]}
    failures = []
    for sid in SETS:
        status = DATA / "aihub" / sid / "transfer-status.json"
        if status.exists():
            failures.extend((sid, key) for key in read(status).get("failed", {}) if (sid, key) not in acquired)
    return failures


def inspect_files(watch):
    items = expected_files()
    verified = {}
    last_failures = ()
    while len(verified) < len(items):
        ready = ready_files(items)
        failures = tuple(sorted(unresolved(ready, items)))
        if failures and failures != last_failures:
            print(f"Waiting for repair of {len(failures)} provider packages; continuing other checks", flush=True)
        last_failures = failures
        for rel in sorted(ready - verified.keys(), key=lambda name: (DATA / name).stat().st_size):
            try:
                verified[rel] = verify(rel, items[rel])
            except (OSError, EOFError, ValueError, RuntimeError, BadZipFile) as error:
                raise RuntimeError(f"Verification failed for {rel} ({type(error).__name__})") from None
            if len(verified) % 25 == 0:
                print(f"Checked {len(verified)}/{len(items)} sources", flush=True)
        if len(verified) == len(items):
            break
        if not watch:
            print(f"Checked {len(verified)}/{len(items)} sources; waiting for acquisition receipts", flush=True)
            return False
        time.sleep(60)
    # Confirm files and their receipts still agree; unchanged files reuse their readback proof.
    records = [verify(rel, items[rel]) for rel in sorted(items)]
    counts = Counter(row["dataset"] for row in records if "filekey" in row)
    inventory = snapshot()
    for sid, count in counts.items():
        row = inventory["aihub"][sid]
        if row["published_files"] != count or row["completed_files"] != count:
            raise RuntimeError(f"Inventory differs from verified ledger for {sid}")
    catalogs = {}
    for sid in SETS:
        file = DATA / "aihub" / sid / "catalog.json"
        catalogs[sid] = {"files": counts[sid], "catalog_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                         "source": f"https://api.aihub.or.kr/info/{sid}.do"}
    final = {"observed_utc": datetime.now(timezone.utc).isoformat(), "aihub_packages": catalogs,
             "verified_files": len(records), "verified_bytes": sum(row["bytes"] for row in records),
             "verification": "Local SHA-256 readback against acquisition receipt. Every ZIP/XLSX entry read through CRC validation. PDF signature and SHA-256 checked.",
             "files": records}
    write(ROOT / "results/external_inventory.json", inventory)
    write(ROOT / "results/external_file_ledger.json", final)
    print(f"Complete: {len(records)} files, {final['verified_bytes']} bytes, {sum(counts.values())} catalog ZIPs", flush=True)
    return True


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--watch", action="store_true", help="Continue verifying while the existing downloader runs.")
    args = cli.parse_args()
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00100000)  # Windows background processing mode
    with transfer_lock(DATA / "integrity.lock"):
        raise SystemExit(0 if inspect_files(args.watch) else 2)
