"""Verify collected official-kit evidence without reopening the ZIP or any development image."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def rows(path):
    with path.open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(run, counts):
    manifest = json.loads((run / "zip-manifest.json").read_bytes())
    checkers = json.loads((run / "official-checker-hashes.json").read_bytes())
    log = (run / "linux-session.log").read_text(encoding="utf8")
    for filename, digest in checkers.items():
        if not re.search(rf"^{digest}  [^\n]*/kit/{re.escape(filename)}$", log, re.M):
            raise ValueError(f"Missing Linux hash for {filename}")
    if not re.search(rf"^{manifest['sha256']} +[^\n]*submission\.zip$", log, re.M):
        raise ValueError("Missing Linux verification of the submission archive")
    seen_ids, seen_hashes = set(), set()
    cohorts = {}
    for count in counts:
        path = run / "linux" / f"n{count}"
        timing = json.loads((path / "timing.json").read_bytes())
        host = json.loads((path / "host-timing.json").read_bytes())
        inputs = rows(path / "synthetic-provenance.csv")
        photos = rows(path / "photos.csv")
        outputs = [
            json.loads(line)
            for line in (path / "submission.jsonl").read_bytes().splitlines()
        ]
        ids = [row["image_id"] for row in inputs]
        hashes = {row["jpeg_sha256"] for row in inputs}
        if (
            len(ids) != count
            or len(set(ids)) != count
            or len(hashes) != count
            or any(re.fullmatch(r"[0-9a-f]{64}", h) is None for h in hashes)
            or seen_ids.intersection(ids)
            or seen_hashes.intersection(hashes)
            or ids != [r["image_id"] for r in photos]
            or ids != [r["image_id"] for r in outputs]
        ):
            raise ValueError(
                f"Input/output identity or cross-run uniqueness failure in n{count}"
            )
        expected = [
            hashlib.sha256(
                f"kwater39-synthetic-{20261126 + count}-{i}".encode()
            ).hexdigest()[:32]
            for i in range(count)
        ]
        if ids != expected:
            raise ValueError(
                f"Synthetic manifest does not match fixed seed for n{count}"
            )
        seen_ids.update(ids)
        seen_hashes.update(hashes)
        checker_text = (path / "output-check.log").read_text(encoding="utf8")
        if f"유효 결과 {count}/{count}" not in checker_text:
            raise ValueError(
                f"Official output checker did not validate all {count} rows"
            )
        if timing["seccomp_ipv4_socket_errno"] != 13 or timing["uid"] != 10001:
            raise ValueError("Guest identity/network isolation evidence disagrees")
        if abs(timing["all_photo_s"] - sum(float(p["total_s"]) for p in photos)) > 1e-5:
            raise ValueError("Per-photo times disagree with the guest aggregate")
        environment = timing["post_run_environment"]
        cohorts[str(count)] = {
            "guest": timing,
            "host": host,
            "official_valid_rows": count,
            "input_jpeg_bytes": sum(int(i["jpeg_bytes"]) for i in inputs),
            "distinct_dimensions": sorted(
                {(int(i["width"]), int(i["height"])) for i in inputs}
            ),
            "gpu_reported_as_l4": environment.get("gpu") == "NVIDIA L4",
        }
    return {
        "zip_sha256_verified_on_linux": manifest["sha256"],
        "official_kit_sha256": checkers,
        "unique_synthetic_jpegs": len(seen_hashes),
        "cohorts": cohorts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--counts", type=int, choices=(3, 200, 300), nargs="+", default=[3, 200, 300]
    )
    args = parser.parse_args()
    evidence = verify(args.run, args.counts)
    args.output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
    )
    print(
        f"Verified {evidence['unique_synthetic_jpegs']} unique synthetic inputs.",
        flush=True,
    )
