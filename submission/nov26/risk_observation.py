"""Conservatively relabel a generated item using its observation alone."""

import re


STRUCTURE = re.compile("비계|가설대|동바리|거푸집|지지대|지주|벽체|철골|가새|교각|기둥|구조물")
SLOPE = re.compile("토사면|사면|흙막이|옹벽|비탈면|절개면")
DEFORMATION = re.compile("기울(?:고|어|었)|처(?:져|졌|지고)|휘(?:어|었)|꺾(?:여|였)|휜|붕괴|무너|변형|침하|균열|갈라|함몰|벌어(?:져|졌|진)")
SUBJECT = "(?:" + STRUCTURE.pattern + "|" + SLOPE.pattern + "|(?:철골|금속|가설|지지)\\s*부재)"
PART = "(?:일부|한쪽|중앙부|중앙|상단|하단|측면|표면|하부|기초)"
DIRECTION = "(?:(?:크게|심하게|비스듬히|비정상적으로|아래쪽으로|아래로|한쪽으로|안쪽으로|바깥쪽으로|옆으로)\\s*)*"
DIRECT = re.compile(SUBJECT + "(?:의)?\\s*(?:" + PART + ")?(?:이|가|은|는|에)?\\s*" + DIRECTION + "(?:" + DEFORMATION.pattern + ")")
DESCRIBED = re.compile("(?:기울어진|처진|휘어진|휜|꺾인|갈라진|무너진|변형된|침하한|함몰된)\\s*" + SUBJECT)
PART_OF_GROUP = re.compile(SUBJECT + "(?:가|들이|가 서로)\\s*(?:겹쳐|이어져|맞물려)\\s*있고[,]?\\s*일부(?:가|는)\\s*" + DIRECTION + "(?:" + DEFORMATION.pattern + ")")
UNCERTAIN = re.compile(
    "않|없|아니|아닌|아님|못하|못했|보이지|관찰되지|확인되지|뚜렷하지|불분명|불확실|판단하기|단정하기|"
    "기울지|처지지|무너지지|갈라지지|가능성만|징후만|여부|만약|가정|있는지|의심|듯"
)
WALKING = re.compile(
    "걸려|걸릴|걸리|걸림|넘어|미끄|보행|통행|통로|발밑|발끝|발 옆|발 디딜|발 디딤|이동로|"
    "동선|이동 공간|이동 구간|출입로|물기|고인 물|물이 고|케이블|전선|호스|바닥|지면"
)
FALLING_PERSON = re.compile(
    "추락|개구부|개방부|난간|가장자리|단부|끝단|단차|사다리|고소작업|안전대|안전띠|"
    "작업발판의 끝|발판 끝|발판 사이|발 디딜|높이차|아래로 열린|아래로 뚫|수직 낙차"
)
OTHER_CAUSE = re.compile("낙하|비래|떨어|인양|매달|머리 위|머리 위쪽|후크|슬링|안전망|방지망")


def structure_observed(evidence):
    # A negative or ambiguous clause is a reason to defer. This preserves
    # missing protection as originally classified, including an unguarded edge.
    if UNCERTAIN.search(evidence) or OTHER_CAUSE.search(evidence):
        return False
    if WALKING.search(evidence) or FALLING_PERSON.search(evidence):
        return False
    return any(pattern.search(evidence) for pattern in (DIRECT, DESCRIBED, PART_OF_GROUP))


def reconcile(prediction):
    occupied = {h["code"] for h in prediction["hazards"]}
    if "collision_contact" in occupied:
        return
    for hazard in prediction["hazards"]:
        if hazard["code"] in {"slip_trip", "fall_height"} and structure_observed(hazard["evidence"]):
            # The official formatter maps collision_contact to canonical other.
            hazard["code"] = "collision_contact"
            return
