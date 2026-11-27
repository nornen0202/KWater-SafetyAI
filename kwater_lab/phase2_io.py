"""Phase 2 run paths and atomic JSON receipts without importing numerical or model libraries."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

PROJECT = Path(os.environ.get("KW_PHASE2_PROJECT", "G:/Projects/KWater-SafetyAI"))
PHASE1 = PROJECT / "data/phase01-runs/phase01-20261126-0513-2155e8fa"
RUN = Path(
    os.environ.get("KW_PHASE2_RUN", PROJECT / "data/phase2-runs/p38-20261126-49bb82ce")
)


def _retry_sharing(operation):
    for attempt in range(6):
        try:
            return operation()
        except PermissionError as err:
            if attempt == 5 or getattr(err, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(0.05 * 2**attempt)


def read(p):
    return json.loads(_retry_sharing(p.read_bytes))


def write(p, data):
    payload = (
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    p.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=p.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        _retry_sharing(lambda: os.replace(temp, p))
    finally:
        temp.unlink(missing_ok=True)
