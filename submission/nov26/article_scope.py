"""Check selected high-frequency citations against the generated observation and action.

Article scopes refer to the organizer's assets/laws.jsonl (산업안전보건기준에 관한 규칙).
Only use a replacement that has an active article record in that file.
"""

import re


def active_articles(records):
    return {
        "OSH-" + match[1] + ("-" + match[2] if match[2] else "")
        for row in records
        if row.get("status") == "active" and row.get("record_type") == "article"
        if (match := re.fullmatch(r"제(\d+)조(?:의(\d+))?", row["article"]))
    }


def supported_reference(hazard, original, active):
    evidence = hazard["evidence"]
    action = hazard["prevention"]

    def cites(article):
        return [{"kind": "law", "id": article}] if article in active else []

    # A demolition exclusion radius is more directly covered by art. 384(1).
    if (re.search("철거|파쇄|해체", evidence)
            and re.search("건물|벽체|외벽|구조물|구축물", evidence)
            and re.search("철거|파쇄|해체", action)
            and re.search("출입.{0,8}(제한|통제|막|금지)|접근.{0,8}(제한|통제|막|금지)", action)):
        return cites("OSH-384")

    # Falling material above a person and an exclusion zone are art. 14(2).
    if (re.search("머리 위|바로 위|상부|머리 위쪽|천장", evidence)
            and re.search("물체|자재|돌|흙|파편|판재|부재|낙하물", evidence)
            and re.search("떨어|낙하|비래", action)
            and re.search("출입|접근|방지망|방호선반|보호망", action)):
        return cites("OSH-14")

    # These observed mismatches identify a different physical mechanism.
    # Keep unreviewed article families on the organizer's retrieval path.
    ids = {r["id"] for r in original}
    footing = re.search("개구부|개방부|가장자리|단부|난간|낭떠러지|추락|끝단|단차", evidence)
    if "OSH-43" in ids and not footing and re.search("기울|무너|침하|붕괴|토사면", evidence):
        return []
    if ("OSH-14" in ids and re.search("바닥|지면", evidence)
            and re.search("치우|정리", action)
            and not re.search("낙하|인양|비래|머리 위|상부|천장|매달|떨어|들려", evidence + action)):
        return []
    if ("OSH-52" in ids and re.search("바닥|지면", evidence)
            and re.search("적치|적재|치우|정리", action)
            and not re.search("붕괴|침하|균열|벽체|무너|기울|구조|변형", evidence + action)):
        return []
    return original


def apply(record, active):
    for hazard in record["hazards"].values():
        refs = supported_reference(hazard, hazard.get("references", []), active)
        if refs:
            hazard["references"] = refs
        else:
            hazard.pop("references", None)
    return record
