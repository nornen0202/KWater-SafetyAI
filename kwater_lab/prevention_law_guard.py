"""Conservative checks against the bundled occupational safety rules."""

from dataclasses import dataclass
import re
import unicodedata

from .law_reranker_11code import article_to_osh_id


@dataclass(frozen=True)
class SupportRule:
    name: str
    law: str
    record: str
    action: str
    scope: str = ""


RULES = (
    SupportRule("dust_spray", "OSH-4-2", "article:4-2",
                r"살수(?:를)?\s*(?:하|한|실시|작업)|물(?:을)?\s*(?:뿌리|뿌린|살포)", r"분진|먼지|비산"),
    SupportRule("helmet", "OSH-32", "article:32:p:1:i:1",
                r"안전모.{0,14}(?:착용|지급)", r"추락|낙하|떨어|비래|인양"),
    SupportRule("harness_two_metres", "OSH-32", "article:32:p:1:i:2",
                r"안전대.{0,14}(?:착용|지급)", r"(?=.*(?:높이|깊이|추락))"
                r"(?=.*(?<![\d.])(?:[2-9]|[1-9]\d+)(?:\.\d+)?\s*(?:미터|m)(?=에서|의|[\s.,)]|$)"
                r"(?!\s*(?:미만|이하|보다\s*낮|안팎|정도)))"),
    SupportRule("insulating_ppe", "OSH-32", "article:32:p:1:i:6",
                r"절연(?:용\s*보호구|\s*장갑).{0,14}(?:착용|지급)", r"감전|충전부|전기"),
    SupportRule("hold_loading_dust_mask", "OSH-32", "article:32:p:1:i:8",
                r"방진마스크.{0,14}(?:착용|지급)", r"(?=.*선창)(?=.*하역)(?=.*분진)"),
    SupportRule("platform_edge_guardrail", "OSH-43", "article:43:p:1",
                r"(?:안전\s*난간|고정식\s*난간|고정\s*난간).{0,15}(?:설치|세우|세운)",
                r"(?:작업발판|작업대|통로).{0,22}(?:끝|가장자리|난간.{0,10}(?:없|보이지))|개구부|추락\s*위험.{0,12}단부"),
    SupportRule("structural_assessment", "OSH-52", "article:52",
                r"(?:구조\s*검토|안전성\s*평가|안전\s*진단).{0,14}(?:실시|시행|수행|하(?:고|도록|여|게|라|는|며)|한다)",
                r"(?=.*(?:구축물|구조물|비계|거푸집|동바리|건물|외벽))(?=.*(?:붕괴|침하|균열|비틀림|내력|하중|재사용|구조부.{0,15}변경))"),
    SupportRule("restricted_area", "OSH-20", "article:20",
                r"출입(?:을|이|의)?\s*(?:금지|제한|차단|통제)(?:하|한)|(?:울타리|펜스).{0,14}(?:설치|세우|세운)",
                r"추락\s*(?:위험|우려)|개구부|단부|해체\s*작업|화재\s*위험|폭발\s*위험|굴착기.{0,20}선회|인양전자석"),
    SupportRule("construction_vehicle_exclusion", "OSH-200", "article:200:p:1",
                r"유도자.{0,14}(?:배치|두)|(?:출입|접근)(?:을|이|의)?\s*(?:금지|제한|차단|통제)(?:하|한)",
                r"(?:굴착기|굴삭기|백호|불도저|로더|차량계\s*건설기계).{0,45}(?:운전|작업|이동|선회|접촉|충돌)"),
    SupportRule("hot_work_spark_screen", "OSH-241", "article:241:p:2:i:4",
                r"(?:불연\s*덮개|비산방지\s*덮개|용접방화포).{0,15}(?:설치|씌우|놓|둔|두)|불티.{0,16}(?:구간|범위).{0,10}가리",
                r"(?=.*(?:용접|용단|절단점|화기작업))(?=.*(?:가연성|목재|나무))"),
    SupportRule("live_part_insulation", "OSH-301", "article:301:p:1:i:2",
                r"(?:절연\s*덮개|방호망).{0,15}(?:설치|씌우|부착)", r"충전\s*(?:부|부분)|노출.{0,15}(?:분전반|배전반)"),
    SupportRule("live_part_enclosure", "OSH-301", "article:301:p:1:i:1",
                r"(?:폐쇄형\s*외함|외함).{0,15}(?:설치|밀폐|닫)",
                r"충전\s*(?:부|부분)|노출.{0,15}(?:분전반|배전반)"),
    SupportRule("live_part_wrapping", "OSH-301", "article:301:p:1:i:3",
                r"(?:충전부|충전부분).{0,16}절연물.{0,10}(?:감싸|덮)",
                r"충전\s*(?:부|부분)|노출.{0,15}(?:분전반|배전반)"),
    SupportRule("live_panel_door", "OSH-301", "article:301:p:3",
                r"(?:분전반|제어반)(?:의)?\s*문.{0,15}고정", r"감전|충전부|전기"),
    SupportRule("leakage_breaker", "OSH-304", "article:304:p:1",
                r"누전\s*차단기.{0,15}(?:설치|접속)",
                r"습윤\s*장소|임시\s*배선.{0,35}(?:이동형|휴대형)|(?:이동형|휴대형).{0,35}(?:임시\s*배선|철판|철골|150\s*볼트)|수분.{0,20}저압"),
    SupportRule("electrical_deenergization", "OSH-319", "article:319:p:1",
                r"(?:전원|전로|전기\s*공급)(?:을|를)?(?:\s*먼저)?\s*(?:차단|분리|격리)",
                r"(?=.*(?:작업|점검|정비|수리|보수))(?=.*(?:충전\s*(?:부|부분)|접속부|단자|전기함|분전함))"
                r"(?=.*(?:노출|열려|드러나|벗겨|파손))"),
)

_CLAUSE = re.compile(r"[.。!?;\n,]+")
_DENIAL = re.compile(r"(?:착용|지급|설치|배치|실시|검토|차단|고정)(?:는|을|를)?\s*(?:하지\s*않|하지\s*말|안\s*하|생략|불필요|할\s*필요(?:가)?\s*없)")
_CONTACT = re.compile(r"(?:충전\s*(?:부|부분)|접속부|(?:노출\s*)?전극|(?:노출|벗겨진)\s*(?:전선|도체)).{0,36}?(?:직접\s*)?(?:접촉|만지|만진|손을\s*대|수리|정비|감싸|감싼|절연(?=하|한)|테이프.{0,6}감)")
_CUT = r"(?:전원|전로|전기\s*공급)(?:을|를|의)?(?:\s*먼저)?\s*(?:차단|격리)"
_CUT_BYPASS = re.compile(_CUT + r"(?:을|를)?\s*(?:없이|하지\s*않고|하지\s*않은\s*채|전(?:에|\s*먼저))")
_CONTACT_PROHIBITION = re.compile(r"^(?:도록\s*해서는\s*안|하지\s*않|하지\s*말|면\s*안|지\s*않|지\s*말|을\s*(?:금지|방지|막)|을\s*하지\s*않|(?:이|가|을)\s*없|위험(?:이|을)?\s*(?:없|방지|예방))")
_EXPLICIT_FIRST = re.compile(_CUT + r"(?:한\s*(?:뒤|후)|하고\s*(?:나서|난\s*뒤)|\s*후)")
_FIRE_POSITION = re.compile(r"(?:불티|불꽃|스파크)(?:가|이|의)?(?P<direction>.{0,34}?)(?:방향|경로|하류|낙하지점)(?P<location>.{0,30}?)(?:작업자|근로자|사람)(?:를|을|가|이)?\s*(?:배치|세우|두|대기시키|작업시키|서(?=도록|게|지))")
_AWAY = re.compile(r"반대|피해|벗어나|이격|바깥|밖|없는|닿지\s*않|차폐된")


@dataclass(frozen=True)
class OrderCheck:
    status: str
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RerankDecision:
    top1: str | None
    support_rules: tuple[str, ...]


def _text(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").lower()


def check_unsafe_action_order(prevention: str) -> OrderCheck:
    """Flag explicit instructions and separate prohibitions from missing ordering evidence."""
    text = _text(prevention)
    unsafe, prevention_only, unclear = [], [], []
    for contact in _CONTACT.finditer(text):
        suffix = text[contact.end():].lstrip()
        if _CONTACT_PROHIBITION.match(suffix) or re.match(r"^(?:하지\s*않|피하)", suffix):
            prevention_only.append("electrical_contact_prohibited")
            continue
        before = _CLAUSE.split(text[:contact.start()])[-1]
        after = _CLAUSE.split(suffix)[0]
        if _CUT_BYPASS.search(before) or re.search(r"^(?:한|하고)\s*(?:난\s*)?(?:뒤|후).{0,25}" + _CUT, after):
            unsafe.append("energized_contact_before_isolation")
        elif _EXPLICIT_FIRST.search(text[:contact.start()]):
            prevention_only.append("electrical_isolation_stated_first")
        else:
            unclear.append("electrical_sequence_unspecified")
    for fire in _FIRE_POSITION.finditer(text):
        suffix = text[fire.end():].lstrip()
        if _AWAY.search(fire['direction'] + fire['location']) or _CONTACT_PROHIBITION.match(suffix):
            prevention_only.append("spark_path_avoidance_stated")
        elif re.search(r"날|향하|튀|진행|받|흐르|낙하|도달", fire['direction']) or "불티 방향" in fire[0]:
            unsafe.append("worker_placed_in_spark_path")
        else:
            unclear.append("spark_exposure_unspecified")
    for state, findings in (("unsafe_order", unsafe), ("order_unclear", unclear), ("explicit_prevention", prevention_only)):
        if findings:
            return OrderCheck(state, tuple(dict.fromkeys(findings)))
    return OrderCheck("not_applicable")


def build_active_records(laws):
    """Return the article and exact bundled record ID tails accepted by the guard."""
    return {(article_to_osh_id(row['article']), row['id'].partition(':')[2]) for row in laws
            if row.get('status') == 'active' and row.get('id') and row.get('article') and row.get('text')}


def action_support(prevention, active_records, *, evidence="", scene_note=""):
    action = _text(prevention)
    context = " ".join(map(_text, (prevention, evidence, scene_note)))
    if check_unsafe_action_order(action).status == "unsafe_order":
        return {}
    supported = {}
    clauses = [clause for clause in _CLAUSE.split(action) if not _DENIAL.search(clause)]
    for rule in RULES:
        if (rule.law, rule.record) not in active_records or (rule.scope and not re.search(rule.scope, context)):
            continue
        if any(re.search(rule.action, clause) for clause in clauses):
            supported.setdefault(rule.law, []).append(rule.name)
    return {law: tuple(rules) for law, rules in supported.items()}


def rerank_top1_by_action_support(prevention, ranked_candidates, active_records, *, evidence="", scene_note=""):
    """Keep the original first candidate unless a top-three alternative directly matches."""
    candidates = tuple(ranked_candidates[:3])
    if not candidates:
        return RerankDecision(None, ())
    supports = action_support(prevention, active_records, evidence=evidence, scene_note=scene_note)
    for candidate in candidates:
        if candidate in supports:
            return RerankDecision(candidate, supports[candidate])
    return RerankDecision(candidates[0], ())


def replay(output, local_audit):
    """Replay the two frozen generations without changing the production payloads."""
    from collections import Counter, defaultdict
    import json
    from pathlib import Path

    from .fold0_law_top1_replay import DEV, GOLD, LAWS, MASK, load_model_outputs, paired_ci, sha
    from .fold0_production_compare import fold_rows, image_id
    from .law_reranker_11code import Native11LawRetriever, read_selected_cases
    from .production_law_replay import load_legacy, read_json, read_rows, refs
    from .train_head import save_atomic

    if not local_audit.resolve().is_relative_to(Path("data").resolve()):
        raise ValueError("Per-occurrence audit rows must stay below data/")
    development = read_json(DEV)
    photos = fold_rows(DEV)
    groups = {row["sample_id"]: row["group_id"] for row in photos}
    ids = set(groups)
    train_ids = {row["sample_id"] for row in development if row["split"] == "dev" and row["dev_fold"] != 0}
    train = list(read_selected_cases(GOLD, train_ids))
    if len(train_ids) != 750 or {row["sample_id"] for row in train} != train_ids or ids & train_ids:
        raise ValueError("Expected the same 750 training photos and 216 unseen fold members")
    law_rows = read_rows(LAWS)
    active = build_active_records(law_rows)
    if any((rule.law, rule.record) not in active for rule in RULES):
        raise ValueError("An exact active statutory record referenced by the guard is missing")
    dust_record = next(row for row in law_rows if row["status"] == "active" and row["id"].endswith(":article:4-2"))
    dust_fixture = "분진이 심하게 흩날리는 작업장에 물을 뿌린다."
    if not (all(fragment in dust_record["text"] for fragment in ("분진이 심하게 흩날리는", "물을 뿌리"))
            and action_support(dust_fixture, active).get("OSH-4-2") == ("dust_spray",)):
        raise ValueError("The bundled dust provision and explicit action fixture no longer agree")
    native = Native11LawRetriever(law_rows, train)
    _, code_map = load_legacy(Path("data/baseline"))
    models, generation_hashes = load_model_outputs(photos)
    gold = {r["sample_id"]: r["target"]["hazards"] for r in read_selected_cases(GOLD, ids)}
    if gold.keys() != ids:
        raise ValueError("Missing the complete Fold 0 matched gold population")
    blind_path = Path("data/fold0-ai-blind-review-20261127/freeze.json")
    frozen_blind = read_json(blind_path)
    inspected = {sid for sid in ids if image_id(sid) in frozen_blind["selected_anonymous_ids"]}
    mask = read_json(MASK)
    if len(inspected) != 12 or frozen_blind["canonical_manifest_sha256"] != sha(DEV) or mask["canonical_manifest_sha256"] != sha(DEV):
        raise ValueError("The previously inspected cohort or exclusion mask differs")
    excluded_groups = {link["fold0_group_id"] for link in mask["confirmed_cross_group_links"]}
    cohorts = {
        "all_fold0": ids,
        "previously_inspected_12": inspected,
        "remaining_204": ids - inspected,
        "confirmed_cross_group_exclusion": {sid for sid in ids if groups[sid] not in excluded_groups},
    }
    audit = []
    for name, (formatted, raw) in models.items():
        for sid, final in formatted.items():
            if not final["hazards"]:
                continue
            raw_obj = raw[sid]
            mapped = {}
            for hazard in raw_obj["hazards"]:
                code = code_map.get(hazard["code"]) if name == "official" else hazard["code"]
                mapped.setdefault(code, hazard)
            for code, saved in final["hazards"].items():
                hazard = mapped[code]
                prevention = hazard["prevention"]
                context = {"evidence": hazard["evidence"], "scene_note": raw_obj.get("scene_note", "")}
                refs3 = native.select_osh_references(code, prevention=prevention, max_refs=3, ranked_candidates=True, **context)
                candidates = [ref["id"] for ref in refs3]
                original = candidates[0] if candidates else None
                predicted = rerank_top1_by_action_support(prevention, candidates, active, **context)
                supported = action_support(prevention, active, **context)
                check = check_unsafe_action_order(prevention)
                audit.append({
                    "model": name, "sample_id": sid, "code": code, "prevention": prevention, **context,
                    "cohorts": [label for label, members in cohorts.items() if sid in members],
                    "saved_top1": next(iter(refs(saved)), None), "candidate_laws": candidates,
                    "baseline_top1": original, "guard_top1": predicted.top1, "support_rules": predicted.support_rules,
                    "supported_laws_before_candidate_restriction": list(supported),
                    "directly_supported_in_candidates": [law for law in candidates if law in supported],
                    "matched_gold_laws": refs(gold[sid].get(code, {})),
                    "baseline_direct_support": original in supported, "guard_direct_support": predicted.top1 in supported,
                    "order": check.status, "order_findings": check.findings,
                })

    def aggregate(model, cohort):
        rows = [row for row in audit if row["model"] == model and row["sample_id"] in cohort]
        annotated = [row for row in rows if row["matched_gold_laws"]]
        counts = Counter()
        paired = defaultdict(list)
        for row in annotated:
            prior = row["baseline_top1"] in row["matched_gold_laws"]
            after = row["guard_top1"] in row["matched_gold_laws"]
            counts.update(saved=row["saved_top1"] in row["matched_gold_laws"], baseline=prior, guard=after,
                          gain=after and not prior, loss=prior and not after)
            paired[groups[row["sample_id"]]].append(int(after) - int(prior))
        changes = [row for row in rows if row["baseline_top1"] != row["guard_top1"]]
        return {
            "photos": len(cohort), "groups": len({groups[sid] for sid in cohort}), "queries": len(rows),
            "candidate_list_empty": sum(not row["candidate_laws"] for row in rows), "top1_changed": len(changes),
            "action_rule_trigger_queries": sum(bool(row["supported_laws_before_candidate_restriction"]) for row in rows),
            "action_rule_triggers_in_top3": sum(bool(row["directly_supported_in_candidates"]) for row in rows),
            "changed_by_support_rule": dict(Counter(rule for row in changes for rule in row["support_rules"])),
            "direct_support_rule_proxy": {"baseline": sum(row["baseline_direct_support"] for row in rows),
                                            "guard": sum(row["guard_direct_support"] for row in rows)},
            "order_status_counts": dict(Counter(row["order"] for row in rows)),
            "order_findings": dict(Counter(f for row in rows for f in row["order_findings"])),
            "matched_annotated_queries": len(annotated), "annotated_groups": len(paired),
            "annotated_top1_hits": {k: counts[k] for k in ("saved", "baseline", "guard", "gain", "loss")},
            "paired_group_bootstrap_guard_minus_native_hit_rate_ci95": paired_ci(paired),
        }

    result = {
        "scope": "CPU-only audit of fixed raw official and v2 sentences; annotations used only after prediction for a top-1 proxy.",
        "training_pool_photos": len(train_ids), "candidate_pool_top_k": 3,
        "sealed_holdout_used": False, "previously_inspected_cohort_is_development_only": True,
        "official_outputs_used_for_generation_training": "Official comparator LoRA had development exposure; replay changes only references.",
        "direct_support_measurement": "Rule-trigger counts, no fresh blinded visual rating or official qualitative score.",
        "bundled_clause_fixture_verified": {"record_id": dust_record["id"], "rule": "dust_spray"},
        "input_sha256": generation_hashes | {path.as_posix(): sha(path) for path in (DEV, GOLD, LAWS, MASK, blind_path)},
        "bundled_legal_record_rules": [{"rule": r.name, "law": r.law, "record": r.record} for r in RULES],
        "panels": {model: {label: aggregate(model, cohort) for label, cohort in cohorts.items()} for model in models},
    }
    save_atomic(local_audit, lambda f: f.write(json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8")))
    save_atomic(output, lambda f: f.write((json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/prevention_law_guard_eval.json"))
    parser.add_argument("--local-audit", type=Path, default=Path("data/prevention-law-guard/fold0-local.json"))
    args = parser.parse_args()
    replay(args.output, args.local_audit)
