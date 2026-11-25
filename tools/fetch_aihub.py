"""Fetch the three approved datasets from AI-Hub. A secret is required only by its issuer.

python tools/fetch_aihub.py --launch
The launcher asks for the key without echo. The background child inherits it on a private
pipe. Downloads, retries, and resumable status are in data/external/aihub on G:.
"""

import argparse
import concurrent.futures
from contextlib import contextmanager
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from urllib.parse import urljoin, urlsplit

import requests


ROOT = Path(__file__).resolve().parents[1] / "data/external/aihub"
SETS = ("71407", "71793", "507")
PATTERN = re.compile(r"[^├└─│\n]*?[├└]─([^\r\n/|]+?\.zip)\s*\|\s*([\d.,]+)\s*(B|KB|MB|GB|TB)\s*\|\s*(\d+)")
UNITS = {name: 1024**i for i, name in enumerate(("B", "KB", "MB", "GB", "TB"))}
ZIP_HEADERS = (b"PK\x03\x04", b"PK\x07\x08", b"PK\x05\x06")


def save(path, obj):
    part = path.with_name(path.name + ".tmp")
    part.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    part.replace(path)


@contextmanager
def transfer_lock(path=None):
    """Release the OS lock even if a worker is interrupted or the PC restarts."""
    with (path or ROOT/"transfer.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def catalog(sid):
    folder = ROOT / sid
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://api.aihub.or.kr/info/{sid}.do"
    reply = requests.get(url, timeout=(12, 40))
    tree = reply.content.decode("utf-8-sig")
    rows = PATTERN.findall(tree)
    if not rows or len(rows) != len([line for line in tree.splitlines() if re.search(r"\|\s*\d+\s*$", line)]):
        raise ValueError(f"Unrecognized catalog for {sid}, HTTP {reply.status_code}")
    (folder / "file-tree.txt").write_text(tree, encoding="utf-8")
    records = [{"name": n, "estimate": round(float(size.replace(",", "")) * UNITS[unit]), "filekey": k}
               for n, size, unit, k in rows]
    if len({r["filekey"] for r in records}) != len(records):
        raise ValueError("Duplicate file key")
    save(folder / "catalog.json", records)
    return records


def unpack(archive, final):
    output = final.with_suffix(".zip.assembling")
    checksum = hashlib.sha256()
    with output.open("wb") as dst:
        def copy(src):
            while chunk := src.read(4 * 1024**2):
                checksum.update(chunk)
                dst.write(chunk)
        with archive.open("rb") as probe:
            zip_header = probe.read(4) in ZIP_HEADERS
        if zip_header:
            with archive.open("rb") as stream:
                copy(stream)
        else:
            with tarfile.open(archive) as tar:
                parts = []
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = PurePosixPath(member.name).name
                    found = re.fullmatch(r"(.+\.zip)(?:\.part0*(\d+))?", name, re.IGNORECASE)
                    if not found:
                        raise ValueError("Unexpected file in provider TAR")
                    parts.append((int(found[2] or 0), found[1], member))
                parts.sort(key=lambda p:p[0])
                numbers = [p[0] for p in parts]
                offsets, position = [], 0
                for _, _, member in parts:
                    offsets.append(position)
                    position += member.size
                contiguous = (numbers == offsets or numbers == list(range(len(numbers)))
                              or numbers == list(range(1,len(numbers)+1)))
                if not parts or not contiguous or len({p[1] for p in parts}) != 1:
                    raise ValueError("Incomplete or ambiguous ZIP part set")
                for _, _, member in parts:
                    with tar.extractfile(member) as stream:
                        copy(stream)
    with zipfile.ZipFile(output) as z:
        members = len(z.infolist())
        if not members:
            raise ValueError("Empty provider ZIP")
    output.replace(final)
    return {"bytes": final.stat().st_size, "sha256": checksum.hexdigest(), "zip_members": members}


def fetch(sid, row, secret):
    out = ROOT / sid / "packages"
    out.mkdir(exist_ok=True)
    key = row["filekey"]
    if Path(row["name"]).name != row["name"]:
        raise ValueError("Invalid name in catalog")
    dest = out / (key + "-" + row["name"])
    receipt = dest.with_suffix(".zip.json")
    if dest.is_file() and receipt.is_file() and json.loads(receipt.read_text(encoding="utf-8"))["bytes"] == dest.stat().st_size:
        with dest.open("rb") as f:
            if f.read(4) in ZIP_HEADERS:
                return
        fixed = unpack(dest, dest)
        save(receipt, {**row, **fixed, "dataset":sid, "source":f"https://api.aihub.or.kr/down/0.6/{sid}.do?fileSn={key}"})
        return
    tar = out / (key + ".tar.partial")
    url = f"https://api.aihub.or.kr/down/0.6/{sid}.do"
    offset = tar.stat().st_size if tar.exists() else 0
    if offset >= 1024:
        with tar.open("rb") as previous:
            previous.seek(-1024,os.SEEK_END)
            has_footer = previous.read(1024) == bytes(1024)
        if has_footer:
            try:
                metrics = unpack(tar,dest)
            except (tarfile.TarError,zipfile.BadZipFile,ValueError,EOFError):
                pass
            else:
                save(receipt,{**row,**metrics,"dataset":sid,"source":url+"?fileSn="+key})
                tar.unlink()
                return
    headers = {"apikey": secret}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    route = url + "?fileSn=" + key
    for _ in range(5):
        parsed = urlsplit(route)
        name = parsed.hostname or ""
        if parsed.scheme != "https" or not (name == "aihub.or.kr" or name.endswith(".aihub.or.kr")):
            raise RuntimeError(f"Unrecognized provider download host {parsed.scheme}://{name}")
        safe_headers = headers if name == "api.aihub.or.kr" else {k:v for k,v in headers.items() if k != "apikey"}
        reply = requests.get(route, headers=safe_headers, stream=True, allow_redirects=False, timeout=(20,150))
        if not reply.is_redirect:
            break
        route = urljoin(route, reply.headers["Location"])
        reply.close()
    else:
        raise RuntimeError("Too many download redirects")
    with reply:
        # Some official endpoints set HTTP 502 even when serving a well-formed catalog or TAR.
        continuation = offset and reply.status_code == 206 and reply.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        pieces = reply.iter_content(1024**2)
        first = next(pieces, b"")
        if not continuation and not (first.startswith((*ZIP_HEADERS, b"\x1f\x8b")) or first[257:262] == b"ustar"):
            raise RuntimeError(f"Provider did not send a ZIP/TAR (HTTP {reply.status_code}, {len(first)} initial bytes)")
        with tar.open("ab" if continuation else "wb") as f:
            f.write(first)
            for block in pieces:
                f.write(block)
    metrics = unpack(tar, dest)
    save(receipt, {**row, **metrics, "dataset": sid, "source": url + "?fileSn=" + key})
    tar.unlink()
    return metrics


def run_dataset(sid, secret):
    rows = catalog(sid)
    by_size = sorted(rows, key=lambda r: (r["estimate"], r["filekey"]))
    first = [r for r in by_size if r["name"].startswith(("VS", "VL"))]
    priority = []
    for prefix in ("VS","TS"):
        raw = next((r for r in by_size if r["name"].startswith(prefix)), None)
        if raw:
            label = next((r for r in rows if r["name"] == prefix[0]+"L"+raw["name"][2:]),None)
            if label:
                priority.append(label)
            priority.append(raw)
    completed = {p.name.split("-",1)[0] for p in (ROOT / sid / "packages").glob("*.zip.json")}
    priority = [r for r in rows if r["filekey"] in completed] + priority
    order = list({r["filekey"]:r for r in [*priority, *first, *by_size]}.values())
    done, errors = 0, {}
    for row in order:
        for attempt in range(3):
            try:
                fetch(sid,row,secret)
                done += 1
                break
            except (requests.RequestException, OSError, ValueError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as e:
                if attempt == 2:
                    errors[row["filekey"]] = type(e).__name__ if isinstance(e, requests.RequestException) else str(e)[:220]
                else:
                    time.sleep(2**attempt)
        save(ROOT / sid / "transfer-status.json", {"total":len(rows),"complete":done,"failed":errors,
                                                  "last_filekey":row["filekey"],"time_unix":time.time()})
        if errors and done == 0:
            raise RuntimeError(f"Provider did not supply first verified package for {sid}: {errors}")
    return sid, done, len(errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch",action="store_true")
    p.add_argument("--pipe",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    key = sys.stdin.readline().strip() if args.pipe else getpass.getpass("AI-Hub API key (hidden): ").strip()
    if not key:
        raise ValueError("An AI-Hub API key is required")
    if args.launch:
        with (ROOT/"transfer.log").open("a",encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--pipe"],
                                    stdin=subprocess.PIPE,stdout=log,stderr=log, text=True,
                                    creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            proc.stdin.write(key+"\n")
            proc.stdin.close()
            print("Started local transfer process",proc.pid,flush=True)
        return
    with transfer_lock(), concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_dataset,sid,key) for sid in SETS]
        failed = False
        for future in concurrent.futures.as_completed(futures):
            try:
                result=future.result()
                print(result,flush=True)
                failed |= result[2] != 0
            except Exception as exc:
                print(type(exc).__name__, str(exc)[:300],flush=True)
                failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
