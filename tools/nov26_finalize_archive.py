"""Complete a staged Nov 26 submission, keeping the organizer's GPU inference unchanged."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[1]
MODULES = REPO / "submission/nov26"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected baseline source around {old}")
    return text.replace(old, new)


def overlay(source, scoped, risk, observed=False, redundancy_config=None):
    entry = source.read("main.py").decode("utf-8")
    files = {}
    if scoped:
        files["article_scope.py"] = (MODULES / "article_scope.py").read_bytes()
        entry = replace_once(entry, "from submission import format_submission",
                             "import article_scope\nfrom submission import format_submission")
        entry = replace_once(entry, "    if valid:\n", "    laws = load_rows(ROOT / 'assets/laws.jsonl') if valid else []\n"
                             "    active = article_scope.active_articles(laws)\n    if valid:\n")
        entry = replace_once(entry, 'format_submission(valid, load_rows(ROOT / "assets" / "laws.jsonl"),',
                             "format_submission(valid, laws,")
        entry = replace_once(entry, "json.dumps(convert(i, final.get(i)),",
                             "json.dumps(article_scope.apply(convert(i, final.get(i)), active),")
    if risk or observed:
        name = "risk_observation" if observed else "risk_text"
        payload = (MODULES / (name + ".py")).read_bytes()
        if observed and hashlib.sha256(payload).hexdigest() != "3d56e5a93cd47de91ab1e6584517c82d31ae749bb4dfcbb2e5ad7a7a293eb0fb":
            raise ValueError("Observation-only candidate differs from its frozen source")
        files[name + ".py"] = payload
        entry = replace_once(entry, "from submission import format_submission",
                             f"from {name} import reconcile\nfrom submission import format_submission")
        entry = replace_once(entry, "    final = {}",
                             "    for row in valid:\n        reconcile(row['prediction'])\n    final = {}")
    assets = source.read("assets.json")
    if redundancy_config:
        code = (MODULES / "redundant_hazards.py").read_bytes()
        config = redundancy_config.read_bytes()
        if (hashlib.sha256(code).hexdigest() != "dbf850cb4a7282cb734c2c5fd57f9e9e2d5c02e6a43947997f48423cbeb695c2"
                or hashlib.sha256(config).hexdigest() != "990a6fb318194da1a8fd7fa424a8447d75daece9044b8c47930fba9c31c8cb27"):
            raise ValueError("Redundancy policy differs from the prospective freeze")
        files["redundant_hazards.py"] = code
        files["assets/redundancy.json"] = config
        entry = replace_once(entry, "from submission import format_submission",
                             "from redundant_hazards import RedundancyCheck\nfrom submission import format_submission")
        entry = replace_once(entry, "    final = {}",
                             "    if any(len(r['prediction']['hazards']) > 1 for r in valid):\n"
                             "        check = RedundancyCheck.from_file(ROOT / 'assets/redundancy.json')\n"
                             "        for row in valid:\n            check.reconcile(row['prediction'])\n    final = {}")
        metadata = json.loads(assets)
        metadata["assets"].append({
            "kind": "data", "name": "중복 위험 확인용 한국어 관찰·예방조치 사례",
            "source": "주최 측 배포 assets/train.jsonl 중 개발 11개 그룹의 위험 항목 310개",
            "revision": "20261126-173014", "license": "대회 참가 목적 이용 허용",
            "use": "동시에 생성된 위험 항목 간 관찰·예방조치 중복 확인",
        })
        assets = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf8")
    files["main.py"] = entry.encode("utf8")
    files["assets.json"] = assets
    return files


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("baseline", type=Path)
    cli.add_argument("staging", type=Path)
    cli.add_argument("archive", type=Path)
    cli.add_argument("--scoped", action="store_true")
    variants = cli.add_mutually_exclusive_group()
    variants.add_argument("--risk", action="store_true")
    variants.add_argument("--observed", action="store_true")
    variants.add_argument("--redundancy-config", type=Path)
    args = cli.parse_args()
    if not (args.scoped or args.risk or args.observed or args.redundancy_config) or args.archive.exists():
        cli.error("Select a change and a new release filename.")
    with ZipFile(args.baseline) as source:
        files = overlay(source, args.scoped, args.risk, args.observed, args.redundancy_config)
        expected = set(source.namelist()) - {"main.py", "assets.json"}
    with ZipFile(args.staging, "a", compression=ZIP_DEFLATED) as dest:
        if set(dest.namelist()) != expected or len(dest.namelist()) != len(expected):
            raise ValueError("Staging archive does not contain exactly the reserved baseline entries")
        for name, payload in files.items():
            dest.writestr(name, payload)
    if args.staging.stat().st_size > 20_000_000_000:
        raise ValueError("Archive exceeds competition limit")
    args.staging.rename(args.archive)


if __name__ == "__main__":
    main()
