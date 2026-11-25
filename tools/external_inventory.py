"""Summarize external dataset versions and fetch progress without publishing source records."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, doc):
    part = path.with_name(path.name+".tmp")
    part.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    part.replace(path)


def snapshot(hash_samples=False):
    out = {"observed_utc":datetime.now(timezone.utc).isoformat(),"aihub":{},"references":[]}
    for sid in ("71407","71793","507"):
        folder=DATA/"aihub"/sid
        catalog=read(folder/"catalog.json")
        files = [read(p) for p in sorted((folder/"packages").glob("*.zip.json"))]
        status_path=folder/"transfer-status.json"
        status=read(status_path) if status_path.is_file() else {}
        record = {"source":f"https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn={sid}",
                  "approved":True,"published_files":len(catalog),"provider_estimated_bytes":sum(r["estimate"] for r in catalog),
                  "completed_files":len(files),"completed_bytes":sum(r["bytes"] for r in files),
                  "transfer_errors":len(status.get("failed",{}))}
        sample = folder/"sample.zip"
        receipt = folder/"sample.zip.json"
        if sample.is_file() and hash_samples and not receipt.is_file():
            with sample.open("rb") as stream:
                sha=hashlib.file_digest(stream,"sha256").hexdigest()
            write(receipt,{"bytes":sample.stat().st_size,"sha256":sha,"source_page":record["source"]})
        if sample.is_file():
            record["sample"] = read(receipt) if receipt.is_file() else {"bytes":sample.stat().st_size}
        out["aihub"][sid]=record
    for loc in ("kalis/sample.zip","floodnet/FloodNet.zip","kosha-confined/confined-space-2022.pdf",
                "kosha-sif/sif-20260401.xlsx"):
        p=DATA/loc
        receipt=p.with_name(p.name+".json")
        out[loc]=read(receipt) if receipt.exists() else {"verified_receipt":False}
    for receipt in sorted((DATA/"moel-heat").glob("*.pdf.json")):
        out["references"].append(read(receipt))
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hash-samples",action="store_true")
    p.add_argument("--save",action="store_true",help="Write a timestamped aggregate snapshot to results")
    args=p.parse_args()
    doc=snapshot(args.hash_samples)
    if args.save:
        write(ROOT/"results/external_inventory.json",doc)
    for sid,row in doc["aihub"].items():
        print(sid,row["completed_files"],"/",row["published_files"],"verified,",row["completed_bytes"],"bytes")
