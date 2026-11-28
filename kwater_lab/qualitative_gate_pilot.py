"""CPU-only development experiments on frozen legal queries and training annotations."""

from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from .fold4_law_policy_replay import ARCHIVE, DATA, DEV, GOLD, MASK, TraceRetriever, fold4, prediction_refs, records
from .fold4_native_archive_replay import load_packaged_entrypoint, raw_index
from .law_reranker_11code import extract_terms, read_selected_cases
from .prevention_law_guard import action_support, build_active_records, check_unsafe_action_order


OUT = Path("results/qualitative-gate-pilot-20261128.json")
LOCAL = Path("data/local-reports/qualitative-gate-pilot-20261128")
GENERIC = {"OSH-20", "OSH-32"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def weighted_query(parts, weights):
    terms = Counter()
    for text, weight in zip(parts, weights):
        terms.update({term: count * weight for term, count in extract_terms(text).items()})
    return terms


def retrieve_weighted(native, code, query):
    with ExitStack() as stack:
        for index, weights in ((native.clause_index, (.7, 1.3, .42)),
                               (native.case_indices.get(code), (1.3, .7, .42))):
            if index is not None:
                method, terms = index.query, weighted_query(query, weights)
                stack.enter_context(patch.object(index, "query", lambda _, f=method, q=terms: f(q)))
        return [ref["id"] for ref in native.select_osh_references(code, *query, max_refs=1)]


def analyze(panels, gold, baseline, queries, native, laws):
    active = build_active_records(laws)
    observations = {}
    for key in sorted(gold.keys() & queries.keys()):
        before, after = (baseline[key][k] for k in ("native", "action"))
        q = queries[key]
        protected = before if after in GENERIC and before and before not in GENERIC else after
        support = action_support(q[1], active, evidence=q[0], scene_note=q[2])
        candidate = [r["id"] for r in native.select_osh_references(key[1], *q, max_refs=2)]
        adaptive = [before] + [law for law in candidate[1:] if law in support and law != before]
        direct_second = [before] + ([after] if after != before and after in support else [])
        observations[key] = {"native": [before], "action": [after], "protected": [protected],
                             "adaptive": adaptive, "direct_second": direct_second,
                             "weighted": retrieve_weighted(native, key[1], q)}
    result = {}
    for name, panel in panels.items():
        keys = [k for k in observations if k[0] in panel]
        changes = Counter()
        scores = {}
        for mode in next(iter(observations.values())):
            hits = emitted = correct = two = 0
            for k in keys:
                refs = [r for r in observations[k][mode] if r]
                hits += bool(refs and refs[0] in gold[k])
                emitted += len(refs)
                correct += len(set(refs) & gold[k])
                two += len(refs) == 2
            target = sum(len(gold[k]) for k in keys)
            scores[mode] = dict(top1=hits, two_laws=two, correct_refs=correct, output_refs=emitted,
                                gold_refs=target, micro_set_f1=2 * correct / (emitted + target))
        for k in keys:
            before, after = (observations[k][m][0] for m in ("native", "action"))
            if before in gold[k] and after not in gold[k]:
                changes[f"{before} -> {after}"] += 1
        result[name] = dict(eligible=len(keys), policies=scores, lost_transitions=dict(changes))
    return result


def training_packet():
    manifest = json.loads(DEV.read_bytes())
    mask0 = Path("data/fold0-crosssplit-visual-review/fold0-group-exclusions-confirmed.json")
    excluded = {x["train_group_id"] for path in (mask0, MASK) for x in
                json.loads(path.read_bytes())["confirmed_cross_group_links"]}
    train = {r["sample_id"]: r for r in manifest if r["split"] == "dev" and r["dev_fold"] in (1, 2, 3)
             and not r["exposed_prior"] and r["group_id"] not in excluded}
    uncertain, review, annotated = [], [], []
    pairs, by_status = Counter(), Counter()
    laws = [json.loads(line) for line in Path("data/prefix-pilot/laws.jsonl").read_bytes().splitlines() if line]
    active = build_active_records(laws)
    for row in read_selected_cases(GOLD, train):
        sid, target = row["sample_id"], row["target"]
        hazards = target["hazards"]
        note = row["notes"].get("uncertainty", "").strip()
        base = dict(sample_id=sid, group=train[sid]["group_id"], image=train[sid]["image"])
        if note:
            uncertain.append(dict(**base, status=target["status"], note=note, codes=sorted(hazards)))
            by_status[target["status"]] += 1
        pairs.update(" + ".join(pair) for pair in combinations(sorted(hazards), 2))
        for code, h in hazards.items():
            refs = sorted({r["id"] for r in h.get("references", ()) if r["kind"] == "law"})
            if not refs:
                continue
            order = check_unsafe_action_order(h["prevention"]).status
            support = action_support(h["prevention"], active, evidence=h["evidence"])
            entry = dict(**base, code=code, prevention=h["prevention"], evidence=h["evidence"], laws=refs,
                         matched=sorted(set(refs) & support.keys()), safety_order=order)
            annotated.append(entry)
            if order == "unsafe_order" or not entry["matched"]:
                review.append(entry)
    # Fixed development controls, kept disjoint by group from the S2 editing queue.
    # The judge sees A/B and image; gold/source decisions stay in a separate key.
    used, controls, key = set(), [], []
    ordered = sorted(annotated, key=lambda r: sha256((r["sample_id"] + r["code"]).encode()).digest())
    for r in ordered:
        if r["group"] in used or not r["matched"]:
            continue
        donor = next((s for s in ordered if s["group"] not in used | {r["group"]}
                      and s["code"] != r["code"] and not set(s["laws"]) & set(r["laws"])
                      and s["evidence"] != r["evidence"]), None)
        if not donor:
            continue
        used.update((r["group"], donor["group"]))
        for axis in ("evidence", "law"):
            good = r["evidence"] if axis == "evidence" else r["matched"][0]
            bad = donor["evidence"] if axis == "evidence" else donor["laws"][0]
            cid = f"{len(controls) + 1:03d}"
            left = sha256(("side/" + cid).encode()).digest()[0] % 2 == 0
            controls.append(dict(id=cid, image=r["image"], axis=axis, code=r["code"],
                                 prevention=r["prevention"], A=good if left else bad, B=bad if left else good))
            key.append(dict(id=cid, expected="A" if left else "B", target=r["sample_id"], donor=donor["sample_id"]))
        if len(controls) >= 24:
            break
    edits = [r for r in review if r["group"] not in used]
    challenge, challenge_key = [], []
    used_donors = set()
    source = {(r["sample_id"], r["code"]): r for r in ordered}
    for item, answer in zip(controls, key):
        if item["axis"] != "law":
            continue
        target = source[answer["target"], item["code"]]
        donor = next((s for s in ordered if s["code"] == target["code"]
                      and s["group"] not in used | used_donors
                      and not set(s["laws"]) & set(target["laws"]) and s["evidence"] != target["evidence"]), None)
        if donor is None:
            continue
        used_donors.add(donor["group"])
        for axis in ("law", "evidence"):
            field = "laws" if axis == "law" else "evidence"
            qid = f"H{len(challenge) + 1:02d}"
            left = sha256(qid.encode()).digest()[0] % 2 == 0
            challenge.append(dict(id=qid, image=target["image"], axis=axis, code=target["code"],
                                  prevention=target["prevention"], A=target[field] if left else donor[field],
                                  B=donor[field] if left else target[field]))
            challenge_key.append(dict(id=qid, expected="A" if left else "B", target=target["sample_id"],
                                      donor=donor["sample_id"]))
    mirror, mirror_key = [], []
    for item, answer in zip(controls + challenge, key + challenge_key):
        for flip in (False, True):
            qid = sha256(f"mirror/{item['id']}/{flip}".encode()).hexdigest()[:12]
            mirror.append({**item, "id": qid, "A": item["B"] if flip else item["A"],
                           "B": item["A"] if flip else item["B"]})
            expected = {"A": "B", "B": "A"}[answer["expected"]] if flip else answer["expected"]
            mirror_key.append(dict(id=qid, pair=item["id"], swapped=flip, expected=expected))
    save(LOCAL / "s4-swapped-development.json", sorted(mirror, key=lambda r: r["id"]))
    save(LOCAL / "s4-swapped-key.json", mirror_key)
    save(LOCAL / "s4-hard-control-provenance.json", challenge_key)
    # Reserve hard donors from aligned-target development as well.
    edits = [r for r in edits if r["group"] not in used_donors]
    for name, data in (("s2-review", edits), ("s3-uncertainty", uncertain), ("s4-controls", controls),
                       ("s4-control-key", key)):
        save(LOCAL / f"{name}.json", data)
    return dict(training_photos=len(train), groups=len({r["group_id"] for r in train.values()}),
                split_hash=digest(DEV), labels_hash=digest(GOLD), law_hash=digest(Path("data/prefix-pilot/laws.jsonl")),
                uncertainty=dict(by_status), cooccurrence=dict(pairs.most_common(8)),
                law_annotated=len(annotated), no_rule_or_unsafe=len(review), s2_review_after_control_exclusion=len(edits),
                s4_control_items=len(controls), s4_distinct_groups=len(used), s4_same_code_items=len(challenge),
                s4_additional_donor_groups=len(used_donors), s4_blinded_presentations=len(mirror), s4_scored=0)


def frozen_v3(analyzer=analyze):
    panels, _, _ = fold4()
    prior = json.loads(Path("results/fold4-law-policy-replay-20261127.json").read_bytes())
    raw = DATA / "v3_epoch1_raw_175.jsonl"
    final = DATA / "v3_epoch1_11code_standalone.jsonl"
    hashes = {p.as_posix(): digest(p) for p in (raw, final, DEV, GOLD, MASK)}
    for p, actual in hashes.items():
        if actual != {Path(k.replace(chr(92), "/")).as_posix(): v for k, v in prior["input_sha256"].items()}[p]:
            raise ValueError(f"Frozen input changed: {p}")
    with ZipFile(ARCHIVE) as bundle, TemporaryDirectory(dir=LOCAL) as tmp:
        path = Path(tmp) / "main.py"
        path.write_bytes(bundle.read("main.py"))
        if digest(path) != prior["archive_entrypoint_sha256"]:
            raise ValueError("Formatter witness mismatch")
        runtime = load_packaged_entrypoint(path)
        def asset(name, field):
            content = bundle.read(name)
            if sha256(content).hexdigest() != prior[field]:
                raise ValueError(name)
            return [json.loads(line) for line in content.splitlines() if line]
        laws = asset("assets/laws.jsonl", "law_corpus_sha256")
        cases = asset("assets/cases_11code.jsonl", "compact_cases_sha256")
    native = runtime.Native11LawRetriever(laws, cases)
    action = runtime.ActionSupportedLawRetriever(laws, cases)
    raw_rows, _ = raw_index(raw)
    outputs = records(final)
    ids = set(panels["primary_175"])
    if ids != raw_rows.keys() or ids != outputs.keys():
        raise ValueError("Incomplete fixed v3 generations")
    queries, gold = {}, {}
    for sid in sorted(ids):
        tracer = TraceRetriever(action)
        parsed, _ = runtime.parse_json_loose(raw_rows[sid])
        if runtime.format_v3_record(sid, parsed, tracer) != outputs[sid]:
            raise ValueError("Formatter output differs")
        queries.update({(sid, code): q for code, q in tracer.queries.items()})
    for row in read_selected_cases(GOLD, ids):
        for code, h in row["target"]["hazards"].items():
            refs = {r["id"] for r in h.get("references", []) if r["kind"] == "law"}
            if refs:
                gold[row["sample_id"], code] = refs
    baseline = prediction_refs(queries, outputs, native, action)
    return analyzer(panels, gold, baseline, queries, native, laws), hashes


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    packet = training_packet()
    panels, hashes = frozen_v3()
    save(OUT, dict(panels=panels, fixed_inputs=hashes, training=packet))


if __name__ == "__main__":
    main()
