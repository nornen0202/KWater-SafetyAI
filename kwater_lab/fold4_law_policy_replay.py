"""Replay two frozen Fold 4 generators against two top-1 law policies."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .fold4_code_diagnosis import DEV, GOLD, MASK, fold4
from .fold4_native_archive_replay import load_packaged_entrypoint, raw_index, sha256_file
from .law_reranker_11code import read_selected_cases
from .metrics import read_index

DATA = Path("data/local-eval-predictions/fold4-v3-benchmark")
ARCHIVE = Path("data/submissions/kwater-safeground-v3-epoch1-vlmonly-20261127.zip")
WITNESS = Path("results/fold4_v3_epoch1_standalone_benchmark.json")
ZIP_WITNESS = Path("results/fold4-v3-epoch1-archive-check.json")
POLICIES = ("published", "native", "action")


def records(path):
    return read_index(path, lambda row: (row["image_id"], row), limit_size=True)


def first(hazard):
    return next((ref["id"] for ref in hazard.get("references", ()) if ref["kind"] == "law"), None)


def prediction_refs(queries, finalized, native, action):
    return {
        (sid, code): {
            "published": first(finalized[sid]["hazards"][code]),
            "native": first({"references": native.select_osh_references(code, *query, max_refs=1)}),
            "action": first({"references": action.select_osh_references(code, *query, max_refs=1)}),
        }
        for (sid, code), query in queries.items()
    }


class TraceRetriever:
    """Capture the exact query passed by the archived formatter."""

    def __init__(self, action):
        self.action = action
        self.queries = {}

    def select_osh_references(self, code, evidence, prevention, scene_note, max_refs):
        self.queries[code] = (evidence, prevention, scene_note)
        return self.action.select_osh_references(code, evidence, prevention, scene_note, max_refs=max_refs)


def paired(keys, gold, before, after):
    counts = Counter(dict(gained=0, lost=0, both_hit=0, neither_hit=0, changed_first=0))
    for key in keys:
        left, right = before[key], after[key]
        hit_left, hit_right = left in gold[key], right in gold[key]
        counts[{(False, True): "gained", (True, False): "lost", (True, True): "both_hit",
                (False, False): "neither_hit"}[hit_left, hit_right]] += 1
        counts["changed_first"] += left != right
    return dict(counts)


def summarize(panel, gold, outputs):
    eligible = {arm: {key for key in gold if key[0] in panel and key in out} for arm, out in outputs.items()}
    common = eligible["eight"] & eligible["v3"]
    selectors = {(arm, p): {k: row[p] for k, row in out.items()}
                 for arm, out in outputs.items() for p in POLICIES}
    pairs = [(arm, lhs, arm, rhs) for arm in outputs
             for lhs, rhs in (("native", "action"), ("published", "native"), ("published", "action"))]
    pairs += [("eight", p, "v3", p) for p in POLICIES]
    def scores(keys, arm):
        return {p: sum(selectors[arm, p][key] in gold[key] for key in keys) for p in POLICIES}

    v3 = outputs["v3"]
    all_v3 = {key for key in v3 if key[0] in panel}
    return {
        "photos": len(panel), "common_annotated_law_hazards": len(common),
        "common_hits": {arm: scores(common, arm) for arm in outputs},
        "common_paired_comparisons": {
            f"{a}_{p}_to_{b}_{q}": paired(common, gold, selectors[a, p], selectors[b, q])
            for a, p, b, q in pairs
        },
        "eligible_annotated_law_hazards": {
            arm: {"denominator": len(keys), "hits": scores(keys, arm)} for arm, keys in eligible.items()
        },
        "v3_native_to_action_full_eligible": paired(eligible["v3"], gold, selectors["v3", "native"], selectors["v3", "action"]),
        "v3_all_emitted_hazards": len(all_v3),
        "v3_native_to_action_changed_first_all_emitted": sum(v3[key]["native"] != v3[key]["action"] for key in all_v3),
        "v3_published_to_action_changed_first_all_emitted": sum(v3[key]["published"] != v3[key]["action"] for key in all_v3),
    }


def run(archive):
    panels, _, _ = fold4()
    ids = set(panels["primary_175"])
    evidence = json.loads(WITNESS.read_bytes())["provenance_sha256"]
    zip_evidence = json.loads(ZIP_WITNESS.read_bytes())
    raw8_path = DATA / "raw8_generations_cache.json"
    raw3_path = DATA / "v3_epoch1_raw_175.jsonl"
    final8_path = DATA / "clean_8code_baseline.jsonl"
    final3_path = DATA / "v3_epoch1_11code_standalone.jsonl"
    for path, key in ((raw3_path, "raw_175_jsonl_sha256"), (final8_path, "clean_8code_local175_jsonl_sha256"),
                      (final3_path, "formatted_local175_jsonl_sha256")):
        if sha256_file(path) != evidence[key]:
            raise ValueError("Frozen Fold 4 input differs from the benchmark witness")
    raw8_rows = json.loads(raw8_path.read_bytes())["raw8_outs"]
    raw8 = {r["sample_id"]: r["raw_text"] for r in raw8_rows}
    raw3, _ = raw_index(raw3_path)
    final8, final3 = records(final8_path), records(final3_path)
    if len(raw8_rows) != 175 or any(index.keys() != ids for index in (raw8, raw3, final8, final3)):
        raise ValueError("Expected all exact Fold 4 input IDs")

    with ZipFile(archive) as bundle, TemporaryDirectory(dir="data/local-reports") as tmp:
        main = bundle.read("main.py")
        if hashlib.sha256(main).hexdigest() != zip_evidence["archive_main_sha256"]:
            raise ValueError("Archive entrypoint differs from the independently checked submission")
        path = Path(tmp) / "main.py"
        path.write_bytes(main)
        runtime = load_packaged_entrypoint(path)
        law_bytes = bundle.read("assets/laws.jsonl")
        case_bytes = bundle.read("assets/cases_11code.jsonl")
    laws = [json.loads(line) for line in law_bytes.splitlines() if line]
    cases = [json.loads(line) for line in case_bytes.splitlines() if line]
    if len(cases) != zip_evidence["train_only_cases"] or len(cases) != 658:
        raise ValueError("Unexpected number of bundled train-only compact cases")
    native = runtime.Native11LawRetriever(laws, cases)
    action = runtime.ActionSupportedLawRetriever(laws, cases)
    q8, q3 = {}, {}
    for sid in panels["primary_175"]:
        original8, _ = runtime.parse_json_loose(raw8[sid])
        for code, hazard in final8[sid]["hazards"].items():
            q8[sid, code] = hazard["evidence"], hazard["prevention"], original8["scene_note"]
        original3, _ = runtime.parse_json_loose(raw3[sid])
        tracer = TraceRetriever(action)
        formatted = runtime.format_v3_record(sid, original3, tracer)
        if formatted != final3[sid] or tracer.queries.keys() != final3[sid]["hazards"].keys():
            raise ValueError("Archived v3 formatter differs from the frozen output")
        q3.update({(sid, code): query for code, query in tracer.queries.items()})

    gold = {}
    for row in read_selected_cases(GOLD, ids):
        for code, hazard in row["target"]["hazards"].items():
            refs = {ref["id"] for ref in hazard.get("references", ()) if ref["kind"] == "law"}
            if refs:
                gold[row["sample_id"], code] = refs
    outputs = {"eight": prediction_refs(q8, final8, native, action),
               "v3": prediction_refs(q3, final3, native, action)}
    out = {name: summarize(panel, gold, outputs) for name, panel in panels.items()}
    for name, denominator, hits in (("primary_175", 71, (60, 54)), ("sensitivity_107", 50, (40, 35))):
        report = out[name]
        if (report["common_annotated_law_hazards"] != denominator
                or tuple(report["common_hits"][arm]["published"] for arm in ("eight", "v3")) != hits
                or report["v3_published_to_action_changed_first_all_emitted"] != 0):
            raise ValueError("Frozen official baselines or v3 parity do not reproduce")
    return {
        "scope": "Read-only CPU replay of existing Fold 4 text. Hit means first article matches any provided gold law ID.",
        "cases": {"training_photos": 791, "compact_cases": len(cases)},
        "fixed_query_text": {"eight": "Official formatted evidence/prevention; raw legacy scene note.",
                             "v3": "Exact query traced from archived formatter, including existing sanitization and action repair."},
        "published": {"eight": "Official eight-code formatter run with Fold 4 train-only cases.",
                      "v3": "Actual submitted archive formatter, verified against frozen Fold 4 output."},
        "input_sha256": {str(p): sha256_file(p) for p in (DEV, MASK, GOLD, raw8_path, raw3_path, final8_path, final3_path)},
        "archive_entrypoint_sha256": hashlib.sha256(main).hexdigest(),
        "law_corpus_sha256": hashlib.sha256(law_bytes).hexdigest(),
        "compact_cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "panels": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
