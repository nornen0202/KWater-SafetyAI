"""Run the archived standalone v3 on frozen external sites and trace both citations."""

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from zipfile import ZipFile

from .citation_review_panel import ROOT, TRIAL, freeze, read
from .fold4_citation_alignment import scope_veto
from .fold4_law_policy_replay import ARCHIVE, TraceRetriever
from .prevention_law_guard import build_active_records
from .qualitative_gate_pilot import digest


STAGE = Path("data/submissions/safeground_v3_epoch1_vlm_only_dir")
WAVE = ROOT / "challenge"


def match_controls(observed):
    from .citation_new_sites import rank
    positives = sorted((r for r in observed if r["current"] != r["candidate"]), key=lambda r: rank(r["sid"]+r["code"]))
    changed_groups = {r["group"] for r in positives}
    controls = sorted((r for r in observed if r["current"] == r["candidate"] and r["group"] not in changed_groups),
                      key=lambda r: rank("control/"+r["sid"]+r["code"]))
    used, chosen, unmatched = set(), [], []
    for change in positives:
        if change["group"] in used:
            continue
        control = next((r for r in controls if r["code"] == change["code"] and r["group"] not in used), None)
        if control is None:
            unmatched.append(change)
            continue
        used.update((change["group"], control["group"]))
        chosen.extend([dict(change, kind="change"), dict(control, kind="control")])
    counts = dict(actual_changed_rows=len(positives), actual_changed_sites=len(changed_groups),
                  matched_pairs=len(chosen)//2, unmatched_changed_rows=len(unmatched),
                  matched_codes=dict(Counter(r["code"] for r in chosen if r["kind"] == "change")))
    for change in unmatched:
        if change["group"] not in used:
            chosen.append(dict(change, kind="unmatched_change"))
            used.add(change["group"])
    return chosen, counts


def attest():
    """Bind the staged files on the inference worker to the locally archived submission."""
    from .fold4_native_archive_replay import sha256_file
    hashes = {}
    with ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or not (name in ("main.py","pipeline_config.json","system_prompt_v3.txt")
                    or name.startswith(("base_model/","adapter/","assets/"))):
                continue
            with archive.open(info) as file:
                hashes[name] = hashlib.file_digest(file,"sha256").hexdigest()
            if sha256_file(STAGE/name) != hashes[name]:
                raise ValueError(f"Staged file differs from original archive: {name}")
    freeze(Path("results/citation-generator-files.json"),hashes)


def main():
    from .fold4_native_archive_replay import sha256_file
    if digest(Path("kwater_lab/fold4_citation_alignment.py")) != read(TRIAL)["policy_script_sha256"]:
        raise ValueError("Policy changed after development")
    directory = ROOT / "prospective-input"
    pool_receipt = read(Path("results/citation-prospective-site-inputs.json"))
    if digest(directory / "selection.json") != pool_receipt["local_manifest_sha256"]:
        raise ValueError("Recruitment manifest changed")
    selected = {r["sample_id"]: r for r in read(directory / "selection.json")}
    rows = []
    for sid, row in selected.items():
        if digest(directory / row["image"]) != row["sha256"]:
            raise ValueError("External frame changed")
        rows.append(dict(image_id=sid, image=row["image"]))
    (directory/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    witness = read(Path("results/fold4-law-policy-replay-20261127.json"))
    for name, field in (("main.py","archive_entrypoint_sha256"), ("assets/laws.jsonl","law_corpus_sha256"),
                        ("assets/cases_11code.jsonl","compact_cases_sha256")):
        if digest(STAGE / name) != witness[field]:
            raise ValueError(f"Staged archive differs: {name}")
    for name, expected in read(Path("results/citation-generator-files.json")).items():
        if sha256_file(STAGE / name) != expected:
            raise ValueError(f"Staged asset differs: {name}")
    input_bytes = "".join(json.dumps(dict(image_id=s),separators=(",",":"))+"\n" for s in selected).encode()
    manifest = directory / "manifest.jsonl"
    if manifest.exists() and manifest.read_bytes() != input_bytes:
        raise ValueError("Unexpected previous input manifest")
    if not manifest.exists():
        manifest.write_bytes(input_bytes)
    WAVE.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("citation_archived_standalone", STAGE / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    law_rows = list(map(json.loads,(STAGE / "assets/laws.jsonl").read_bytes().splitlines()))
    cases = list(map(json.loads,(STAGE / "assets/cases_11code.jsonl").read_bytes().splitlines()))
    native = module.Native11LawRetriever(law_rows,cases)
    active = build_active_records(law_rows)
    actual = module.format_v3_record
    observed = []
    def capture(sid, parsed, retriever):
        tracer = TraceRetriever(retriever)
        record = actual(sid, parsed, tracer)
        for code, query in tracer.queries.items():
            original = record["hazards"][code]["references"][0]["id"]
            neighbour = native.select_osh_references(code,*query,max_refs=1)[0]["id"]
            candidate = scope_veto(code, query, neighbour, original, active)
            image = selected[sid]
            observed.append(dict(sid=sid, group=image["site"], source="71407", photo=image["image"],
                                 image_sha256=image["sha256"], code=code, evidence=query[0], prevention=query[1],
                                 current=original, candidate=candidate))
        return record
    module.format_v3_record = capture
    os.environ["KWATER_SEED"] = "42"
    os.environ["KW_BATCH"] = "4"
    sys.argv = [str(STAGE/"main.py"), "--input-dir", str(directory), "--output-dir", str(WAVE)]
    module.main()
    output = WAVE/"submission.jsonl"
    results = [json.loads(l) for l in output.read_bytes().splitlines()]
    if [r["image_id"] for r in results] != list(selected) or len(observed) != sum(len(r["hazards"]) for r in results):
        raise ValueError("Output order or hazard trace mismatch")
    matched, counts = match_controls(observed)
    counts.update(photos=len(selected), sites=pool_receipt["sites"], emitted_hazards=len(observed),
                  policy_sha256=read(TRIAL)["policy_script_sha256"], submission_sha256=digest(output),
                  trace_sha256=freeze(WAVE/"traced-private.json",observed),
                  matched_sha256=freeze(WAVE/"matched-private.json",matched))
    freeze(Path("results/citation-new-site-changes.json"), counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["attest","run"])
    if parser.parse_args().action == "attest":
        attest()
    else:
        main()
