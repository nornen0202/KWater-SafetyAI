"""Score the stock 27B deployment and the four frozen fine-tuned Fold 4 arms."""
import argparse
import json
from pathlib import Path
import re
import sys

from .audit import digest
from .fold0_production_compare import image_id
from .fold4_crosssplit_audit import DEV, LABELS, LOCAL, V3_EVAL, _score_panels, write
from .law_reranker_11code import read_selected_cases
from .metrics import decode_target, read_index
from .offline_archive_audit import run_output_checker
from .train_head import save_atomic


def score(recovered, output):
    receipt = json.loads((recovered / "input-receipt.json").read_bytes())
    source = Path("results/seoul-l4-roadmap-experiments-20261128.json")
    timing = json.loads(source.read_bytes())["gguf_27b_official"]
    if timing["panels"]["input200"]["result"] != "OK" or timing["sha256"] != receipt["archive_sha256"]:
        raise ValueError("Quality comparison requires the same archive that passed the 200-photo gate")

    manifest = [row for row in json.loads(DEV.read_bytes()) if row["split"] == "dev" and row["dev_fold"] == 4]
    canonical = {image_id(row["sample_id"]): row["sample_id"] for row in manifest}
    ids = [json.loads(line)["image_id"] for line in (recovered / "manifest.jsonl").read_bytes().splitlines()]
    raw = recovered / "output/submission.jsonl"
    predictions = [json.loads(line) for line in raw.read_bytes().splitlines()]
    if (len(canonical) != 175 or list(canonical) != ids or [row.get("image_id") for row in predictions] != ids
            or digest(recovered / "manifest.jsonl") != receipt["manifest_sha256"]):
        raise ValueError("Run IDs, order or frozen input fingerprint differ")
    kit_log = (recovered / "kit.log").read_text("utf-8")
    checker = run_output_checker(Path("data/local-kit/kit"), recovered, raw.parent, sys.executable)
    generation = re.findall(r"GGUF_ROWS 175/175 parsed=(\d+) schema_valid=(\d+) tokens=(\d+) prompt_tokens=(\d+) capped=(\d+) seconds=([\d.]+)", kit_log)
    clock = re.findall(r"종료 코드 (\d+), 실행 (\d+)초 \+ 설치 (\d+)초 = (\d+)초 / 상한 (\d+)초", kit_log)
    if (re.findall(r"^RESULT (\w+)", kit_log, re.M) != ["OK"] or len(clock) != 1 or len(generation) != 1
            or not checker["all_rows_valid"] or checker["valid_rows"] != 175):
        raise ValueError("Official run or recovered output check is incomplete")

    mapped = recovered / "canonical-predictions.jsonl"
    for row in predictions:
        decode_target(row)
        row["image_id"] = canonical[row["image_id"]]
    data = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in predictions)
    save_atomic(mapped, lambda out: out.write(data.encode("utf-8")))
    eight = Path("data/vlm-scale-seoul-8b-fold4-20261128")
    arms = {"4b_epoch1": V3_EVAL / "v3_epoch1_11code_standalone.jsonl",
            "8b_epoch1": eight / "eval1/predictions.jsonl",
            "4b_epoch2": V3_EVAL / "v3_11code_standalone.jsonl",
            "8b_epoch2": eight / "eval2/predictions.jsonl", "stock_27b_q4": mapped}
    baseline = json.loads(Path("results/paired-qwen-4b-8b-fold4-20261128.json").read_bytes())
    if not all(digest(arms[name]) == sha for name, sha in baseline["prediction_sha256"].items()):
        raise ValueError("Stored 4B/8B comparison predictions have changed")
    result = _score_panels(LOCAL, arms, None, "frozen_4b_8b_and_stock_27b_fold4")
    for panel in ("primary_175", "sensitivity_107"):
        for name in baseline["prediction_sha256"]:
            if result["panels"][panel]["arm_metrics"][name] != baseline["panels"][panel]["arm_metrics"][name]:
                raise ValueError("Original baseline metrics were not reproduced")
    excluded = {r["fold4_group_id"] for r in json.loads((LOCAL / "fold4-group-exclusions-local.json").read_bytes())["confirmed_cross_group_links"]}
    severe = {r["sample_id"]: {k for k, h in r["target"]["hazards"].items() if h["risk_level"]["severity"]["score"] == 4}
              for r in read_selected_cases(LABELS, set(canonical.values()))}
    decode = lambda row: decode_target(row, local_ids=True)
    tuned = read_index(arms["8b_epoch1"], decode, limit_size=True)
    stock = read_index(mapped, decode, limit_size=True)
    for panel in ("primary_175", "sensitivity_107"):
        rows = [r for r in manifest if panel == "primary_175" or r["group_id"] not in excluded]
        severe_ids = [r["sample_id"] for r in rows if severe[r["sample_id"]]]
        normal_ids = [r["sample_id"] for r in rows if not r["codes"]]
        result["panels"][panel]["stock_27b_vs_tuned_8b_epoch1_disagreements"] = {
            "severe_detected_only_by_27b": sum(bool(stock[i] & severe[i]) and not (tuned[i] & severe[i]) for i in severe_ids),
            "severe_detected_only_by_8b": sum(bool(tuned[i] & severe[i]) and not (stock[i] & severe[i]) for i in severe_ids),
            "normal_false_alarm_only_by_27b": sum(bool(stock[i]) and not tuned[i] for i in normal_ids),
            "normal_false_alarm_only_by_8b": sum(bool(tuned[i]) and not stock[i] for i in normal_ids),
        }
    last = generation[0]
    result.update(scope="Stock Q4 27B deployment versus domain-trained BF16 4B/8B candidates. Training, image preprocessing and runtime differ. Group-based development comparison.",
                  input_receipt=receipt, archive_sha256=timing["sha256"], output_sha256=digest(raw),
                  official_kit_log_sha256=digest(recovered / "kit.log"), official_checker=checker,
                  kit_clock=dict(zip(("return_code", "execute", "install", "total", "limit"), map(int, clock[0]))),
                  generation=dict(zip(("parsed", "raw_schema_valid", "output_tokens", "prompt_tokens", "capped"),
                                      map(int, last[:5]))), model_started_to_final_row_seconds=float(last[-1]))
    write(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.recovered, args.output)
