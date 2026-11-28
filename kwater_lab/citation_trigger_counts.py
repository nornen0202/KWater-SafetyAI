"""Count which emitted hazards can reach the locked citation correction."""
from collections import Counter
import json
from pathlib import Path

from .citation_review_panel import ROOT, TARGET, freeze, read
from .fold4_citation_alignment import scope_veto
from .prevention_law_guard import build_active_records
from .qualitative_gate_pilot import digest


def count(path, active, keep=None):
    eligible = Counter()
    offending = Counter()
    possible = Counter()
    photos = hazards = 0
    for row in map(json.loads, path.read_bytes().splitlines()):
        if keep is not None and row["image_id"] not in keep:
            continue
        photos += 1
        hazards += len(row["hazards"])
        for code, prediction in row["hazards"].items():
            if code not in TARGET:
                continue
            eligible[code] += 1
            current = next((r["id"] for r in prediction.get("references", ()) if r["kind"] == "law"), None)
            if current != ("OSH-45" if code == "slip_trip" else "OSH-319"):
                continue
            offending[code] += 1
            # Grant the required retrieval result and drop only negative scene predicates.
            # This is an upper bound on changes after the full policy sees the raw scene.
            query = prediction["evidence"], prediction["prevention"], ""
            if scope_veto(code, query, TARGET[code], current, active) != current:
                possible[code] += 1
    return dict(photos=photos, emitted_hazards=hazards, eligible_code=dict(eligible),
                eligible_code_and_source_article=dict(offending), possible_changes_upper_bound=dict(possible),
                output_sha256=digest(path))


def main():
    laws = map(json.loads, Path("data/baseline/assets/laws.jsonl").read_bytes().splitlines())
    active = build_active_records(list(laws))
    mask = read(Path("data/sub78-fold2-prospective-readiness-20261128/private-prospective-mask-and-recipe.json"))
    fold2 = Path("data/v3-11code-fold2-4b-epoch1/epoch1-bf16-predictions.jsonl")
    counts = dict(fold2_197=count(fold2, active),
                  fold2_55=count(fold2, active, set(mask["prospective_mask_retained_ids"])))
    for wave in ("challenge", "reserve-run"):
        path = ROOT / wave / "submission.jsonl"
        if path.exists():
            counts[wave] = count(path, active)
    if "challenge" not in counts or ("reserve-run" not in counts and
                                     read(Path("results/citation-new-site-changes.json"))["matched_pairs"] < 2):
        raise ValueError("Field generation is incomplete")
    freeze(Path("results/citation-trigger-counts.json"), counts)


if __name__ == "__main__":
    main()
