"""Compare the original training conversion with an 11-code candidate and the official checker."""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .metrics import CODES, decode_target
from .prototype import assemble


def training_answer(row):
    """Represent all target codes in the content schema consumed by prototype.assemble."""
    target = row["target"]
    return {
        "scene_observation": target.get("scene_evidence", ""),
        "hazards": [{"code": code, "evidence": item["evidence"], "prevention": item["prevention"],
                     "likelihood": item["risk_level"]["likelihood"]["score"],
                     "severity": item["risk_level"]["severity"]["score"]}
                    for code, item in target["hazards"].items()],
    }


def run(labels, checker, interpreter):
    script = Path("data/baseline/scripts/prepare_dataset.py")
    spec = importlib.util.spec_from_file_location("official_converter", script)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    inverse = {v: k for k, v in legacy.TO_MODEL.items()}
    original, preserved, candidate = Counter(), Counter(), Counter()
    total = omitted = changed_text = 0
    exemplars = {}
    for raw in labels.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        target = row["target"]
        _, truth = decode_target(target, local_ids=True)
        original.update(truth)
        old = legacy.convert(row)
        if old is None:
            omitted += 1
        else:
            preserved.update(inverse[h["code"]] for h in old["target"]["hazards"])
        content = training_answer(row)
        output = assemble(sha256(row["sample_id"].encode()).hexdigest()[:32], content)
        _, actual = decode_target(output)
        if actual != truth or output["status"] != target["status"]:
            raise ValueError("Native code round trip changed labels")
        candidate.update(actual)
        for code in actual:
            got, expected = output["hazards"][code], target["hazards"][code]
            if got["evidence"] != expected["evidence"] or got["risk_level"] != expected["risk_level"]:
                raise ValueError("Changed evidence or risk")
            changed_text += got["prevention"] != expected["prevention"]
        for item in content["hazards"]:
            exemplars.setdefault(item["code"], item)
        total += 1
    if set(exemplars) != set(CODES):
        raise ValueError("Incomplete 11-code contract")
    cases = [[c] for c in CODES] + [[], list(CODES[5:8]), [CODES[0], CODES[-1]]]
    with tempfile.TemporaryDirectory(dir="data/external") as folder:
        tmp = Path(folder)
        manifests, results = [], []
        for idx, codes in enumerate(cases):
            name = f"{idx:032x}"
            manifests.append({"image_id": name})
            result = assemble(name, {"scene_observation": "형식 검사용 안전 장면 설명",
                                     "hazards": [exemplars[code] for code in codes]})
            if decode_target(result)[1] != set(codes):
                raise ValueError("Lost combination")
            results.append(result)
        out = tmp / "out"
        out.mkdir()
        for path, rows in ((tmp / "manifest.jsonl", manifests), (out / "submission.jsonl", results)):
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        output = subprocess.run([interpreter, "-X", "utf8", str(checker), str(tmp), str(out)],
                                capture_output=True, check=True, encoding="utf-8").stdout
        if f"유효 결과 {len(cases)}/{len(cases)}" not in output:
            raise ValueError("Official check did not accept every row")
    return {"rows": total, "original": dict(original), "legacy": {c: preserved[c] for c in CODES},
            "candidate": dict(candidate), "legacy_missing_rows": omitted,
            "candidate_missing_rows": 0, "official_checker_valid": len(cases), "official_checker_total": len(cases),
            "prototype_prevention_overrides": changed_text,
            "scope": "Structural round trips only. The prototype replaces electrical prevention and empties references."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker-python", default=sys.executable)
    p = parser.parse_args()
    report = run(Path("data/train/train.jsonl"), Path("data/local-kit/kit/check_output.py"), p.checker_python)
    Path("results/external_contract.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Rows", report["rows"], "officially valid", report["official_checker_valid"], "legacy lost", report["legacy_missing_rows"])
