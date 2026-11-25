"""Attach one active candidate article per generated hazard using the local official corpus.

python -m kwater_lab.attach_law --input submission.jsonl --input-dir sample-input \
    --output cited/submission.jsonl --report cited-report.json

An opaque-ID JSONL on its own uses only the law corpus. A manifest/image directory
permits an example library after excluding exact/near visual matches. A known
research source map (JSONL: image_id,sample_id) additionally excludes the same pilot
prefix. Exclusions apply across the whole batch. Query labels never enter ranking.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .audit import inspect_photo
from .law_examples import select_articles, training_examples
from .law_retrieval import digest, official_ranker_type, official_ranking
from .metrics import decode_target, read_index, same_ids, unique_keys
from .train_head import save_atomic


# Hashes recorded during the official-file and six-prefix pilot audits.
PINNED = {
    "labels": "ab427fed4615b32627b76001e27124e99ae9202c855ba4170916ebe4fdc4de55",
    "exposure": "3e049cac60ec3db236ad3a98bcf44785ccf24621eadd1d16286f72855f0643e4",
    "laws": "1408b69db444cf90b31f9bebb4a773a30fa4362739e50b22c95f3bbe0e75bd9d",
    "rag": "4fdd5046e93cf0689b40fdf209b9f8a5344a5dbe5a6ac43641c14a755043e640",
    "audit": "9ea46cd904b181405052304c4f439cf2e4252a44b8c094c218c989f63e92d502",
}


def choose_examples(exposure, inventory, source_ids, visual_fingerprints):
    pilot = set(exposure["exposed_ids"])
    if source_ids is None and visual_fingerprints is None:
        return [], "law_corpus_only_unmapped_JSONL"
    excluded = set()
    fingerprints = list(visual_fingerprints or ())
    for sid in source_ids or ():
        if sid not in inventory:
            raise ValueError(f"Unknown training source sample ID: {sid}")
        excluded.add(sid)
        fingerprints.append(inventory[sid])
        for members in exposure["by_prefix"].values():
            if sid in members:
                excluded.update(members)
    rgb_hashes = {x["pixels_sha256"] for x in fingerprints}
    dhashes = {int(x["dhash64"], 16) for x in fingerprints}
    for sid in pilot:
        record = inventory[sid]
        h = int(record["dhash64"], 16)
        if record["pixels_sha256"] in rgb_hashes or any((h ^ other).bit_count() <= 4 for other in dhashes):
            excluded.add(sid)
    return sorted(pilot - excluded), "pilot_with_batch_self_and_near_exclusions"


def prediction_with_article(row, retriever, articles):
    image_id, codes = decode_target(row)
    updated = deepcopy(row)
    choices = []
    for code, item in updated["hazards"].items():
        previous = item.get("references", [])
        preserved = [r for r in previous if r["kind"] != "law"]
        chosen = None
        status = "no_slot_after_nonlaw_references"
        if len(preserved) < 3:
            ranking = official_ranking(retriever, item["evidence"], item["prevention"], code)
            chosen = next((ref for ref in ranking if ref in articles), None)
            status = "active_article_candidate" if chosen else "no_active_candidate"
        refs = [*preserved, {"kind": "law", "id": chosen}] if chosen else preserved
        if previous != refs:
            item["references"] = refs
        choices.append({"code": code, "status": status, "article_id": chosen,
                        "article_title": articles[chosen]["title"] if chosen else None})
    if decode_target(updated) != (image_id, codes):
        raise ValueError("Postprocessing changed image identity or hazard codes")
    return updated, choices


def attach(args):
    paths = {name: getattr(args, name) for name in PINNED}
    source_path = args.input or args.replay
    inputs = [source_path, *paths.values()]
    if args.source_map:
        inputs.append(args.source_map)
    if args.input_dir:
        inputs.append(args.input_dir / "manifest.jsonl")
    if (args.output.resolve() == args.report.resolve() or
            {args.output.resolve(), args.report.resolve()} & {p.resolve() for p in inputs}):
        raise ValueError("Output and report must be distinct from all inputs and one another")
    if args.output.name == "submission.jsonl" and args.output.parent.resolve() == args.report.parent.resolve():
        raise ValueError("Keep the diagnostic report outside the submission output directory")
    if args.input_dir and any(p.resolve().is_relative_to(args.input_dir.resolve()) for p in (args.output, args.report)):
        raise ValueError("Write results outside the read-only image input directory")
    if args.replay and args.source_map:
        raise ValueError("Saved prevention replay already has a source identity")
    source_hash = digest(source_path)
    hashes = {name: digest(path) for name, path in paths.items()}
    if hashes != PINNED:
        raise ValueError("Source checksum mismatch: " + ", ".join(k for k in PINNED if hashes[k] != PINNED[k]))
    exposure = json.loads(paths["exposure"].read_text(encoding="utf8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf8"))
    if (len(set(exposure["exposed_ids"])) != 170 or len(exposure["exposed_ids"]) != 170 or
            hashes["labels"] != exposure["labels_sha256"] or hashes["labels"] != audit["labels_sha256"]):
        raise ValueError("Wrong reviewed pilot or labels")
    source_ids = None
    if args.replay:
        saved = json.loads(source_path.read_text(encoding="utf8"), object_pairs_hook=unique_keys)
        row = saved["after_row"]
        key, _ = decode_target(row)
        if not saved["image_sha256"].startswith(key):
            raise ValueError("Public replay image ID and image checksum differ")
        predictions = {key: row}
        source_ids = [saved["sample_id"]]
    else:
        predictions = read_index(source_path, lambda row: (decode_target(row)[0], row), limit_size=True)
        if args.source_map:
            mapped = read_index(args.source_map, lambda row: (row["image_id"], row["sample_id"]))
            same_ids(predictions, mapped, "Research source map")
            source_ids = list(mapped.values())
    fingerprints = None
    if args.input_dir:
        root = args.input_dir.resolve()
        manifest = read_index(root / "manifest.jsonl", lambda row: (row["image_id"], None))
        same_ids(predictions, manifest, "Inference manifest")
        fingerprints = [inspect_photo(root, f"images/{key}.jpg") for key in manifest]
    ids, mode = choose_examples(exposure, audit["inventory"], source_ids, fingerprints)
    with paths["laws"].open(encoding="utf8") as stream:
        laws = [json.loads(line) for line in stream]
    articles = select_articles(laws)
    labels = {}
    allowed = set(ids)
    if allowed:
        with paths["labels"].open(encoding="utf8") as stream:
            for item in map(json.loads, stream):
                sid = item["sample_id"]
                if sid in allowed:
                    labels[sid] = item["target"]["hazards"]
        if set(labels) != allowed:
            raise ValueError("Pilot labels incomplete")
    library, _ = training_examples(ids, labels, articles)
    retriever = official_ranker_type(paths["rag"])(laws, train=library)
    choices = []
    lines = []
    for row in predictions.values():
        result, record = prediction_with_article(row, retriever, articles)
        line = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")
        if len(line) > 2**20:
            raise ValueError(f"Output row exceeds 1 MiB: {row['image_id']}")
        decode_target(json.loads(line))
        choices.append({"image_id": row["image_id"], "candidates": record})
        lines.append(line)
    save_atomic(args.output, lambda f: f.write(b"\n".join(lines) + b"\n"))
    report = {"purpose": "candidate law retrieval from generated code, evidence and prevention",
              "photo_count": len(predictions), "valid_rows": len(lines), "new_inference_requests": 0,
              "input_sha256": source_hash, "output_sha256": digest(args.output), "source_sha256": hashes,
              "library_mode": mode, "pilot_photos": 170, "example_photos": len(ids),
              "excluded_pilot_photos": 170-len(ids), "active_articles": len(articles),
              "ranking_inputs": ["predicted_code", "generated_evidence", "generated_prevention"],
              "candidate_articles": choices}
    if args.source_map:
        report["research_source_map_sha256"] = digest(args.source_map)
    if args.input_dir:
        report["input_manifest_sha256"] = digest(args.input_dir / "manifest.jsonl")
    save_atomic(args.report, lambda f: f.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf8")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="valid submission.jsonl")
    source.add_argument("--replay", type=Path, help="archived scoped prevention replay with after_row")
    parser.add_argument("--input-dir", type=Path, help="official manifest.jsonl and images directory")
    parser.add_argument("--source-map", type=Path, help="known research image_id,sample_id mapping in JSONL")
    for name in ("output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    defaults = {"laws": "data/prefix-pilot/laws.jsonl", "rag": "data/prefix-pilot/rag.py",
                "labels": "data/train/train.jsonl", "exposure": "data/prefix-pilot/exposure.json",
                "audit": "results/train_audit_full.json"}
    for key, path in defaults.items():
        parser.add_argument("--" + key, type=Path, default=Path(path))
    args = parser.parse_args()
    result = attach(args)
    print(f"{result['valid_rows']} valid rows, {result['example_photos']}/170 pilot examples")


if __name__ == "__main__":
    main()
