"""First-citation replay over the archived epoch-one v3 output. No model generation."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

from .fold0_law_top1_replay import paired_ci
from .fold4_law_policy_replay import DATA, paired, records
from .prevention_law_guard import action_support, build_active_records
from .qualitative_gate_pilot import digest, frozen_v3, save


OUT = Path("results/fold4-citation-alignment-20261128.json")
BLIND = Path("data/local-reports/issue2-20261127")
SCORES = ("published", "native", "aligned", "scope_veto")
FLOOR_OBJECTS = r"바닥|발판|돌|자재|부재|잔재|파편|판자|판재|철근|장애물|전선|흙|유리"
PATH_CONTEXT = r"통로|통행|이동.{0,3}(?:경로|동선)|출입구"
DENIED = re.compile(r"하지\s*(?:않|말)|하지\s*못|필요.{0,3}없|생략|불필요|금지")


def positive_clause(text, pattern):
    clauses = re.split(r"[.!?,;\n。]", text)
    return any(re.search(pattern, clause) and not DENIED.search(clause) for clause in clauses)


def floor_cleanup(evidence, prevention, active):
    return (("OSH-3", "article:3:p:1") in active and re.search(FLOOR_OBJECTS, evidence)
            and positive_clause(prevention, rf"(?:{FLOOR_OBJECTS}).{{0,22}}(?:정리|제거|옮기|치우|치워|수거)"))


def choose(code, query, native, published, active):
    """Prefer case retrieval on a tie. Grant a switch only with a scoped action."""
    if native == published or not native or not published:
        return published
    evidence, prevention, scene = query
    support = set(action_support(prevention, active, evidence=evidence, scene_note=scene))
    # Electrical isolation during electrical work has a more specific scope
    # than stopping a machine or removing fuel before hot work.
    if code != "electrical_energy" or not re.search(r"충전|단자|배선|전선|전기|분전|접속", evidence):
        support.discard("OSH-319")
    if code == "slip_trip":
        if floor_cleanup(evidence, prevention, active):
            support.add("OSH-3")
        if (("OSH-22", "article:22:p:1") in active and re.search(PATH_CONTEXT, evidence)
                and positive_clause(prevention, rf"(?:{PATH_CONTEXT}).{{0,16}}(?:확보|설치|유지|구획)")):
            support.add("OSH-22")
    return published if published in support and native not in support else native


def scope_veto(code, query, native, published, active):
    """Override only a roof/electrical-work citation with a supported alternative."""
    evidence, prevention, _ = query
    context = " ".join(query)
    if published == "OSH-45" and not re.search("지붕|채광창|슬레이트", context):
        if native == "OSH-3" and code == "slip_trip" and floor_cleanup(evidence, prevention, active):
            return native
    if published != "OSH-319" or code == "electrical_energy":
        return published
    if re.search(r"충전\s*부|활선|노출.{0,12}(?:단자|도체)|단자.{0,12}노출|전기\s*(?:정비|수리|점검|작업)", context):
        return published
    if (code == "fire_explosion" and native == "OSH-241" and ("OSH-241", "article:241:p:2:i:3") in active
            and re.search("불티|불꽃|용접|용단", evidence)
            and positive_clause(prevention, r"가연(?:물|성\s*물질).{0,15}(?:이격|제거|차폐|격리)")):
        return native
    if (code == "caught_in_machine" and native == "OSH-87" and ("OSH-87", "article:87:p:1") in active
            and re.search("회전부|회전체|회전축|가동부|기어", evidence)
            and positive_clause(prevention, r"(?:방호문|문|덮개|울)(?:을|를)?.{0,10}(?:닫|설치|고정)")):
        return native
    return published


def semantic_check(all_refs):
    witness = json.loads(Path("results/fold4-law-support-blind-review-20261128.json").read_bytes())
    files = {name: BLIND / path for name, path in (
        ("blinded_prompt", "law-support-blind-local.md"),
        ("ratings_before_unblinding", "law-support-frozen-ratings-local.json"),
        ("sealed_assignment", "law-support-sealed-key-local.json"))}
    if {key: digest(path) for key, path in files.items()} != witness["frozen_sha256"]:
        raise ValueError("The earlier blinded ratings or their key changed")
    ratings = json.loads(files["ratings_before_unblinding"].read_bytes())["ratings"]
    keyed = json.loads(files["sealed_assignment"].read_bytes())["rows"]
    if len(keyed) != 8 or ratings.keys() != keyed.keys():
        raise ValueError("Incomplete eight-case semantic control")
    levels = {mode: [0, 0, 0] for mode in SCORES}
    delta = {mode: Counter(dict(better=0, same=0, worse=0)) for mode in ("aligned", "scope_veto")}
    for token, row in keyed.items():
        refs = all_refs[row["sample_id"], row["hazard_code"]]
        lookup = {}
        for side, score in zip(("left", "right"), ratings[token][:2]):
            law = row[side]["law"]
            policy = row[side]["policy"]
            if refs["published" if policy == "action" else "native"] != law:
                raise ValueError("Historical review does not match this model output")
            lookup[law] = score
        for mode in SCORES:
            levels[mode][lookup[refs[mode]]] += 1
        for mode, counts in delta.items():
            change = lookup[refs[mode]] - lookup[refs["published"]]
            counts["better" if change > 0 else "worse" if change < 0 else "same"] += 1
    return dict(n=len(keyed), groups=2, ratings_unrelated_partial_direct=levels,
                versus_published=delta, frozen_sha256=witness["frozen_sha256"])


def projection(rows):
    result = deepcopy(rows)
    for row in result.values():
        for hazard in row["hazards"].values():
            for ref in hazard["references"]:
                if ref["kind"] == "law":
                    ref["id"] = "<first-law>"
    return sha256(json.dumps(result, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate(panels, gold, base, queries, native_index, laws):
    active = build_active_records(laws)
    originals = records(DATA / "v3_epoch1_11code_standalone.jsonl")
    rewritten = {name: deepcopy(originals) for name in ("aligned", "scope_veto")}
    refs = {}
    losses = []
    for key, entry in sorted(base.items()):
        sid, code = key
        previous, published = entry["native"], entry["published"]
        if published != entry["action"]:
            raise ValueError("Published and action baselines differ")
        chosen = choose(code, queries[key], previous, published, active)
        narrow = scope_veto(code, queries[key], previous, published, active)
        refs[key] = dict(native=previous, published=published, aligned=chosen, scope_veto=narrow)
        for mode, output in rewritten.items():
            emitted = [r for r in output[sid]["hazards"][code]["references"] if r["kind"] == "law"]
            if len(emitted) != 1 or emitted[0]["id"] != published or refs[key][mode] not in native_index.osh_to_idx:
                raise ValueError("Expected one existing law and one valid replacement")
            emitted[0]["id"] = refs[key][mode]
        if previous in gold.get(key, ()) and published not in gold[key]:
            losses.append(dict(case=len(losses)+1, code=code, native=previous, published=published,
                               aligned=chosen, scope_veto=narrow, gold=sorted(gold[key]),
                               in_107=sid in panels["sensitivity_107"]))
    if any(projection(originals) != projection(output) for output in rewritten.values()) or len(losses) != 9:
        raise ValueError("Changed a non-citation field or failed to reproduce the nine regressions")
    results = {}
    by_mode = {mode: {k: row[mode] for k, row in refs.items()} for mode in SCORES}
    for name, panel in panels.items():
        emitted = sorted(k for k in refs if k[0] in panel)
        keys = [k for k in emitted if k in gold]
        total_gold = sum(len(gold[k]) for k in keys)
        metrics = {}
        for mode in SCORES:
            hits = sum(refs[k][mode] in gold[k] for k in keys)
            metrics[mode] = dict(hits=hits, top1_rate=hits/len(keys),
                                 set_micro_f1=2*hits/(len(keys)+total_gold))
        comparisons = {}
        for left, right in (("published", "aligned"), ("native", "aligned"), ("published", "scope_veto"),
                            ("native", "scope_veto"), ("native", "published")):
            differences = defaultdict(list)
            for k in keys:
                differences[panel[k[0]]].append(int(refs[k][right] in gold[k])-int(refs[k][left] in gold[k]))
            comparisons[f"{left}_to_{right}"] = dict(
                **paired(keys, gold, by_mode[left], by_mode[right]),
                changed_all_emitted=sum(refs[k][left] != refs[k][right] for k in emitted),
                eligible_group_bootstrap_delta95=paired_ci(differences))
        results[name] = dict(photos=len(panel), groups=len(set(panel.values())), emitted=len(emitted),
                             eligible=len(keys), eligible_groups=len({panel[k[0]] for k in keys}),
                             annotated_laws=total_gold, metrics=metrics, paired=comparisons)
    panel_path = Path("data/fold4-qualitative-blind-20261127/frozen-panel-local.json")
    panel_photos = {r["sample_id"] for r in json.loads(panel_path.read_bytes())}
    reviewed_ids = {sid for (sid, code), row in refs.items()
                    if row["native"] in gold.get((sid, code), ()) and row["published"] not in gold[sid, code]}
    changed_ids = {sid for (sid, _), row in refs.items() if row["scope_veto"] != row["published"]}
    groups = panels["primary_175"]
    reviewed_groups = {groups[s] for s in reviewed_ids}
    exposure = dict(reviewed_photos=len(reviewed_ids), reviewed_groups=len(reviewed_groups),
                    sealed_panel_photos=len(panel_photos), reviewed_sealed_photos=len(panel_photos & reviewed_ids),
                    sealed_photos_in_reviewed_groups=sum(groups[s] in reviewed_groups for s in panel_photos),
                    narrow_change_groups=len({groups[s] for s in changed_ids}), panel_sha256=digest(panel_path))
    return dict(panels=results, reviewed_nine=losses, exposure=exposure, semantic_control=semantic_check(refs),
                noncitation_projection_sha256=projection(originals))


if __name__ == "__main__":
    result, hashes = frozen_v3(evaluate)
    result["frozen_inputs"] = hashes
    result["policy_script_sha256"] = digest(Path(__file__))
    save(OUT, result)
