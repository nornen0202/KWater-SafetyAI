"""Retry retired provider files, then take over missing files when acquisition stops.

Run alongside the existing transfer. The API key is accepted at a masked local
prompt. Only source IDs, received byte counts and response types are reported.
"""

import argparse
from contextlib import ExitStack
import errno
import getpass
import sys
import tarfile
import time
from zipfile import BadZipFile

import requests

from external_inventory import read
from fetch_aihub import OVERLAP, ROOT, SETS, ZIP_HEADERS, fetch, provider_response, save, transfer_lock


def size(path):
    if not path.is_file():
        return 0
    with path.open("rb") as file:
        return file.seek(0, 2)


def pending(*, include_unattempted=False):
    failed, complete = [], True
    for sid in SETS:
        directory = ROOT / sid
        rows = read(directory / "catalog.json")
        parcels = directory / "packages"
        received = {p.name.split("-", 1)[0] for p in parcels.glob("*.zip.json")}
        if not {r["filekey"] for r in rows}.issubset(received):
            complete = False
        status = directory / "transfer-status.json"
        try:
            retired = set(read(status).get("failed", {}))
        except (PermissionError, FileNotFoundError):
            retired = set()
        failed.extend((sid, row, parcels / (row["filekey"] + ".tar.partial"))
                      for row in rows if row["filekey"] not in received
                      and (include_unattempted or row["filekey"] in retired))
    return sorted(failed, key=lambda item: item[1]["estimate"]), complete


def check_resume(sid, row, path, key):
    existing = size(path)
    if existing <= OVERLAP:
        return {"provider_range": "short or absent local prefix", "initial_bytes": existing}
    start = existing - OVERLAP
    with path.open("rb") as local:
        local.seek(start)
        expected = local.read(OVERLAP)
    with provider_response(sid, row["filekey"], key, f"{start}-{existing - 1}") as reply:
        prefix = bytearray()
        for block in reply.iter_content(OVERLAP):
            prefix.extend(block)
            if len(prefix) >= OVERLAP:
                break
        head = prefix[:OVERLAP]
        equals = head == expected
        if equals:
            kind = "matched saved bytes"
        elif head.startswith((*ZIP_HEADERS, b"\x1f\x8b")) or head[257:262] == b"ustar":
            kind = "server replayed the archive start"
        else:
            raise ValueError("Provider probe did not match the archive start or a saved range")
        return {"provider_range": kind, "http": reply.status_code,
                "content_range": reply.headers.get("Content-Range"), "initial_bytes": existing}


def run(key, ownership):
    report = ROOT / "repair-status.json"
    state = read(report) if report.exists() else {}
    retry_at = {}
    successor = False
    exceptions = (OSError, EOFError, requests.RequestException, ValueError, RuntimeError, tarfile.TarError, BadZipFile)
    while True:
        if not successor:
            try:
                ownership.enter_context(transfer_lock())
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
            else:
                successor = True
                print("Original transfer has released its lock; checking all remaining catalog keys", flush=True)
        files, complete = pending(include_unattempted=successor)
        if complete:
            print("All catalog files have acquisition receipts", flush=True)
            return
        now = time.monotonic()
        item = next((v for v in files if retry_at.get((v[0], v[1]["filekey"]), 0) <= now), None)
        if item is None:
            time.sleep(60)
            continue
        sid, row, path = item
        pair = sid, row["filekey"]
        tag = "/".join(pair)
        previous = state.get(tag, {})
        entry = {**previous, "attempts": previous.get("attempts", 0) + 1,
                 "state": "receiving", "initial_bytes": size(path)}
        state[tag] = entry
        save(report, state)
        try:
            if "probe" not in entry and size(path) > OVERLAP:
                entry["probe"] = check_resume(sid, row, path, key)
                save(report, state)
                print(sid, row["filekey"], entry["probe"]["provider_range"], flush=True)
            result = fetch(sid, row, key)
            receipt_path = path.parent / (row["filekey"] + "-" + row["name"] + ".json")
            receipt = read(receipt_path)
            if result and result["sha256"] != receipt["sha256"]:
                raise ValueError("Recovered receipt differs")
            entry.update(state="acquired", bytes=receipt["bytes"], sha256=receipt["sha256"])
            entry.pop("last_error", None)
            print("Recovered", sid, row["filekey"], receipt["bytes"], "bytes", flush=True)
        except exceptions as error:
            entry.update(state="waiting", last_error=type(error).__name__, retained_bytes=size(path))
            delay = 60 if entry["retained_bytes"] > entry["initial_bytes"] else min(1800, 60 * 2**min(entry["attempts"], 5))
            retry_at[pair] = time.monotonic() + delay
            print("Retry pending", sid, row["filekey"], type(error).__name__, "retained", entry["retained_bytes"], flush=True)
        save(report, state)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pipe", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    with transfer_lock(ROOT / "repair.lock"), ExitStack() as ownership:
        secret = sys.stdin.readline().strip() if args.pipe else getpass.getpass("AI-Hub API key (hidden): ").strip()
        if not secret:
            raise SystemExit("Missing provider API key")
        run(secret, ownership)
