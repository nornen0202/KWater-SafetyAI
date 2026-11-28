"""Read a completed Fold 2 first-epoch run, verify formatter parity, screen its frozen 55."""

from pathlib import Path
from .citation_review_panel import DEV, ROOT, TRIAL, freeze, read
from .citation_site_generation import STAGE, match_controls
from .fold4_citation_alignment import scope_veto
from .fold4_law_policy_replay import TraceRetriever, records
from .fold4_native_archive_replay import load_packaged_entrypoint
from .law_reranker_11code import read_selected_cases
from .prevention_law_guard import build_active_records
from .qualitative_gate_pilot import digest
import json


def main():
    out = Path("data/v3-11code-fold2-4b-epoch1")
    receipt = read(out/"epoch1-bf16-inference-receipt.json")
    final = out/"epoch1-bf16-predictions.jsonl"
    cache = out/"epoch1-bf16-raw-cache.json"
    if (receipt["prediction_sha256"] != digest(final) or receipt["rows"] != 197
            or digest(Path("kwater_lab/fold4_citation_alignment.py")) != read(TRIAL)["policy_script_sha256"]):
        raise ValueError("Fold 2 predictions or citation policy differ from witness")
    mask = read(Path("data/sub78-fold2-prospective-readiness-20261128/private-prospective-mask-and-recipe.json"))
    if digest(DEV) != mask["manifest_sha256"]:
        raise ValueError("Manifest changed since the prospective freeze")
    population = {r["sample_id"]:r for r in read(DEV) if r["split"]=="dev"}
    train = [sid for sid,r in population.items() if r["dev_fold"]!=2]
    outer = set(population)-set(train)
    keep = set(mask["prospective_mask_retained_ids"])
    if len(train)!=769 or len(outer)!=197 or len(keep)!=55 or not keep <= outer or set(train)!=set(mask["matched_outer_training_ids_both_models"]):
        raise ValueError("Fold 2 populations differ")
    corpus = {r["sample_id"]:r for r in read_selected_cases(Path("data/train/train.jsonl"),set(train))}
    ordered = [corpus[s] for s in train]
    runtime = load_packaged_entrypoint(STAGE/"main.py")
    laws = list(map(json.loads, Path("data/baseline/assets/laws.jsonl").read_bytes().splitlines()))
    active = build_active_records(laws)
    native, published = runtime.Native11LawRetriever(laws,ordered), runtime.ActionSupportedLawRetriever(laws,ordered)
    predictions = records(final)
    raw = read(cache)["outputs"]
    if predictions.keys()!=outer or raw.keys()!=outer:
        raise ValueError("Missing original generations")
    rows = []
    for sid in sorted(outer):
        tracer = TraceRetriever(published)
        actual = runtime.format_v3_record(sid,raw[sid]["parsed"],tracer)
        if actual != predictions[sid]:
            raise ValueError("Archive replay differs from the Fold 2 train-only formatter")
        for code,q in tracer.queries.items():
            before = actual["hazards"][code]["references"][0]["id"]
            base = native.select_osh_references(code,*q,max_refs=1)[0]["id"]
            new = scope_veto(code,q,base,before,active)
            photo = population[sid]
            rows.append(dict(sid=sid,group=photo["group_id"],photo=photo["image"],image_sha256=photo["sha256"],
                             code=code,evidence=q[0],prevention=q[1],current=before,candidate=new))
    retained = [r for r in rows if r["sid"] in keep]
    cards, counts = match_controls(retained)
    dest = ROOT/"fold2"
    dest.mkdir(exist_ok=True)
    counts.update(outer_photos=197, retained_photos=55, train_only_photos=769, retained_hazards=len(retained),
                  outer_changes=sum(r["current"]!=r["candidate"] for r in rows),prediction_sha256=digest(final),
                  raw_cache_sha256=digest(cache), matched_sha256=freeze(dest/"matched-private.json",cards))
    freeze(Path("results/citation-fold2-changes.json"),counts)


if __name__=="__main__":
    main()
