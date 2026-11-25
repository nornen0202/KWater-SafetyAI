"""Download the public experiment inputs into the ignored data/external directory."""

import argparse
import hashlib
import json
from pathlib import Path
import os
import re

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/external"
FLOODNET = ("https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc"
            "?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1")
HEAT = "https://www.moel.go.kr/policy/policydata/view.do?bbs_seq=20260500439"


def download(session, url, path, signature, expected_size=None, params=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix + ".json")
    if path.is_file() and receipt.is_file():
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if path.stat().st_size == record["bytes"]:
            return record
        raise ValueError(f"Size differs from receipt: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, params=params, headers=headers, timeout=(20, 100), stream=True) as reply:
        reply.raise_for_status()
        append = offset and reply.status_code == 206
        if append and not reply.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Invalid continuation range")
        offset = offset if append else 0
        expected = expected_size
        if expected is None and reply.headers.get("Content-Length") and not reply.headers.get("Content-Encoding"):
            expected = offset + int(reply.headers["Content-Length"])
        chunks = reply.iter_content(2**20)
        first = next(chunks, b"")
        if not first or (not offset and not first.startswith(signature)):
            raise ValueError(f"Unexpected content type for {path.name}: {reply.headers.get('Content-Type')}")
        with partial.open("ab" if append else "wb") as output:
            output.write(first)
            for chunk in chunks:
                output.write(chunk)
    size = partial.stat().st_size
    if expected is not None and expected != size:
        raise ValueError(f"Incomplete download, {size}/{expected} bytes")
    with partial.open("rb") as f:
        if not f.read(len(signature)).startswith(signature):
            raise ValueError("Wrong file signature")
        f.seek(0)
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    partial.replace(path)
    record = {"source_url": url, "bytes": size, "sha256": digest}
    temp = receipt.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, receipt)
    print(f"{path.relative_to(ROOT)} {size} bytes {digest}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("floodnet", "kalis", "heat", "confined-guide", "sif-workbook", "aihub-docs"))
    args = parser.parse_args()
    with requests.Session() as session:
        if args.source == "floodnet":
            download(session, FLOODNET, ROOT / "floodnet/FloodNet.zip", b"PK", 13057835433)
        elif args.source == "kalis":
            origin = "https://www.data.go.kr"
            page = origin + "/data/15142200/fileData.do"
            response = session.get(page, timeout=25)
            response.raise_for_status()
            button = BeautifulSoup(response.content, "html.parser").select_one('[onclick*="fn_fileDataDown"]')
            dataset, detail, attachment, serial, _ = re.findall(r"'([^']*)'", button["onclick"])
            session.headers.update({"Referer": page})
            spec = session.post(origin + "/tcs/dss/selectFileDataDownload.do", data={
                "publicDataPk": dataset, "publicDataDetailPk": detail, "atchFileId": attachment,
                "fileDetailSn": serial, "publicDataTyCode": "PR0051"}, timeout=25)
            spec.raise_for_status()
            info = spec.json()
            if not info.get("status"):
                raise ValueError("Provider did not grant download")
            ids = {"atchFileId": info["atchFileId"], "fileDetailSn": info["fileDetailSn"]}
            limit = session.post(origin + "/cmm/cmm/check-limit.json", data=ids, timeout=25)
            limit.raise_for_status()
            if limit.json().get("needCaptcha") is not False:
                raise ValueError("Download requires manual confirmation on the provider's page")
            download(session, origin + "/cmm/cmm/fileDownload.do", ROOT / "kalis/sample.zip", b"PK",
                     params={**ids, "dataNm": info["dataSetFileDetailInfo"]["dataNm"]})
        elif args.source == "heat":
            response = session.get(HEAT, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            links = sorted({urljoin(HEAT, node["href"]) for node in soup.select('a[href*="downloadFile.do"]')
                            if "file_ext=pdf" in node["href"]})
            if not links:
                raise ValueError("No attachment links")
            for url in links:
                file_id = re.search(r"file_seq=(\d+)", url)[1]
                download(session, url, ROOT / "moel-heat" / (file_id + ".pdf"), b"%PDF")
        elif args.source == "confined-guide":
            url = ("https://www.moel.go.kr/local/boryeong/common/downloadFile.do?"
                   "file_seq=20230502606&bbs_seq=20230501649&bbs_id=LOCAL5")
            download(session, url, ROOT / "kosha-confined/confined-space-2022.pdf", b"%PDF")
        elif args.source == "sif-workbook":
            url = ("https://www.moel.go.kr/common/downloadFile.do?"
                   "file_seq=20260400329&bbs_seq=20260400196&bbs_id=29&file_ext=xlsx")
            download(session, url, ROOT / "kosha-sif/sif-20260401.xlsx", b"PK")
        else:
            for sid in ("71407", "71793", "507"):
                page = "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=" + sid
                response = session.get(page, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                for node in soup.select('a[href*="dataMnlFileSn"], a[href*="cnstcPrcuseFileSn"]'):
                    url = urljoin(page, node["href"])
                    filename = "manual" if "dataMnl" in url else "guide"
                    try:
                        download(session, url, ROOT / "aihub" / sid / (filename + ".pdf"), b"%PDF")
                    except (ValueError, requests.RequestException) as error:
                        print(f"{sid} {filename}: {error}", flush=True)


if __name__ == "__main__":
    main()
