"""Freeze the replacement Fold 4 panel and screen real outputs for the fixed veto."""

from collections import Counter
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path

from .fold0_production_compare import image_id
from .fold4_citation_alignment import scope_veto
from .fold4_law_policy_replay import DEV, DATA
from .prevention_law_guard import build_active_records
from .qualitative_gate_pilot import digest, frozen_v3

ROOT = Path("data/fold4-citation-review-v2")
OLD = Path("data/fold4-qualitative-blind-20261127")
OUT = Path("results/fold4-citation-review-panel-v2.json")
OLD_RECEIPT = Path("results/fold4-qualitative-panel-20261127.json")
TRIAL = Path("results/fold4-citation-alignment-20261128.json")
SEED = "kwater-s1-panel-replacement-v2"
TARGET = {"slip_trip": "OSH-3", "fire_explosion": "OSH-241", "caught_in_machine": "OSH-87"}


def read(path):
    return json.loads(path.read_bytes())


def freeze(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace frozen artifact: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    return digest(path)


def select_panel(panels, reviewed, severe):
    original = read(OLD / "frozen-panel-local.json")
    witness = read(OLD_RECEIPT)["sha256"]
    for file, tag in (("frozen-panel-local.json", "local_manifest"), ("sealed-assignment-local.json", "sealed_assignment")):
        if digest(OLD / file) != witness[tag]:
            raise ValueError("Historical panel or assignment changed")
    groups = panels["primary_175"]
    banned = {groups[sid] for sid in reviewed}
    removed = [r for r in original if r["group_id"] in banned]
    if len(removed) != 4:
        raise ValueError("Unexpected historical group overlap")
    old_groups = {r["group_id"] for r in original}
    stratum = lambda r: ("safe" if not r["codes"] else
                         "severity4" if r["sample_id"] in severe else "other_hazard")
    prior = set(read(Path("data/fold0-ai-blind-review-20261127/freeze.json"))["selected_anonymous_ids"])
    options = [r for r in read(DEV) if r["sample_id"] in panels["sensitivity_107"]
               and r["group_id"] not in old_groups | banned and image_id(r["sample_id"]) not in prior]
    slots = [r["review_id"] for r in removed]
    best, winners = None, None
    retained = [r for r in original if r["review_id"] not in slots]
    old_sources = Counter(r["source"] for r in original)
    stable = {r["sample_id"]: sha256(f"{SEED}/{r['sample_id']}".encode()).hexdigest() for r in options}
    # Each original slot retains its stratum if feasible, then preserve source distribution.
    for picks in permutations(options, len(slots)):
        if len({r["group_id"] for r in picks}) != len(slots):
            continue
        src = Counter(r["source"] for r in retained) + Counter(r["family"] for r in picks)
        rank = (sum(stratum(r) != old["gold_stratum"] for r, old in zip(picks, removed)),
                -len(src), sum(abs(src[s]-old_sources[s]) for s in src.keys() | old_sources.keys()),
                tuple(stable[r["sample_id"]] for r in picks))
        if best is None or rank < best:
            best, winners = rank, picks
    if winners is None:
        raise ValueError("No valid four distinct new groups")
    replace = {slot: dict(review_id=slot, sample_id=r["sample_id"], group_id=r["group_id"],
                          image=r["image"], image_sha256=r["sha256"], source=r["family"], gold_stratum=stratum(r))
               for slot, r in zip(slots, winners)}
    new = [replace.get(r["review_id"], r) for r in original]
    if len({r["group_id"] for r in new}) != 24 or any(r["group_id"] in banned for r in new):
        raise ValueError("Group leakage")
    for r in new:
        if digest(Path("data/train") / r["image"]) != r["image_sha256"]:
            raise ValueError("Panel image hash mismatch")
    ROOT.mkdir(exist_ok=True)
    signature = freeze(ROOT / "panel.json", new)
    return new, banned, dict(photos=len(new), distinct_groups=24, retained=20, replaced=4,
                             old_review_groups=len(banned), overlap_review_groups=0, replacement_pool=len(options),
                             strata=dict(Counter(r["gold_stratum"] for r in new)),
                             sources=dict(Counter(r["source"] for r in new)), panel_sha256=signature,
                             historical_panel_sha256=witness["local_manifest"],
                             unchanged_three_arm_slot_assignment_sha256=witness["sealed_assignment"])


def screen(formatted_path, rows, active, banned):
    """Dropping the unavailable scene and granting the required native law gives an upper bound."""
    predictions = [json.loads(line) for line in formatted_path.read_bytes().splitlines()]
    index = {r["sample_id"]: r for r in rows}
    index.update({image_id(r["sample_id"]): r for r in rows})
    possible = []
    eligible = set()
    hazard_rows = 0
    for prediction in predictions:
        row = index[prediction["image_id"]]
        if row["group_id"] in banned:
            continue
        eligible.add(row["sample_id"])
        for code, h in prediction["hazards"].items():
            hazard_rows += 1
            laws = [ref["id"] for ref in h.get("references", []) if ref["kind"] == "law"]
            if code in TARGET and laws:
                law = scope_veto(code, (h["evidence"], h["prevention"], ""), TARGET[code], laws[0], active)
                if law != laws[0]:
                    possible.append((row["sample_id"], code))
    return dict(photos=len(predictions), outside_review_photos=len(eligible), outside_review_hazards=hazard_rows,
                possible_changes_upper_bound=len(possible), output_sha256=digest(formatted_path)), possible


def run(panels, gold, base, queries, native, laws):
    from .fold4_code_diagnosis import fold4
    if digest(Path("kwater_lab/fold4_citation_alignment.py")) != read(TRIAL)["policy_script_sha256"]:
        raise ValueError("S1 policy differs from its locked version")
    reviewed = {s for (s,c), r in base.items() if r["native"] in gold.get((s,c), ()) and r["published"] not in gold[s,c]}
    _, _, severe = fold4()
    new, banned, panel = select_panel(panels, reviewed, {s for s, codes in severe.items() if codes})
    paths = dict(fold4_epoch1=DATA / "v3_epoch1_11code_standalone.jsonl",
                 fold4_epoch2=DATA / "v3_11code_standalone.jsonl",
                 sixth=Path("data/local-reports/seoul-sixth-200-20261128/output/submission.jsonl"),
                 seventh=Path("data/local-reports/seoul-seventh-200-20261128/seven/output/submission.jsonl"),
                 eighth=Path("data/local-reports/seoul-eighth-200-20261128/output/submission.jsonl"))
    active = build_active_records(laws)
    rows = read(DEV)
    screens = {name: screen(path,rows,active,banned)[0] for name,path in paths.items()}
    members = {r["sample_id"] for r in new}
    changed = [(s,c) for (s,c), r in base.items() if s in members
               and scope_veto(c,queries[s,c],r["native"],r["published"],active) != r["published"]]
    return dict(panel=panel, locked_policy_sha256=read(TRIAL)["policy_script_sha256"], screens=screens,
                exact_epoch1_changes_on_panel=len(changed))


if __name__ == "__main__":
    result, _ = frozen_v3(run)
    freeze(OUT, result)
