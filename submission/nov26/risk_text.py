"""Resolve two specific disagreements between a generated type and its own Korean description."""

import re


def supported_mechanism(evidence, action):
    structure = re.search("비계|가새|동바리|거푸집|지지대|벽체|철골|기둥|천장|구조물", evidence)
    deformation = re.search("기울|비스듬.{0,12}돌출|처져|처지고|휘어|휘어진|균열|무너|변형", evidence)
    stabilization = re.search("안정화|안정성|지지 상태|보강|구조.{0,5}점검|안전.{0,6}해체", action)
    footing = re.search("바닥|지면|보행면|발 디딜|통로.{0,8}(막|가로|좁)|미끄러|걸려 넘어|걸림|개구부|개방부|난간|가장자리|추락", evidence)
    elevated = re.search("머리 위|바로 위|공간 위|상부|머리 위쪽|천장|지붕|처마|매달", evidence)
    material = re.search("물체|자재|돌[이과을]|흙|파편|판재|부재|낙하물|패널|조각|기와", evidence)
    exclusion = re.search("출입|접근|방지망|방호선반|보호망", action)
    lower_zone = re.search("아래쪽|하부|아래 공간|아래 출입|부재의 아래", action)
    falling = re.search("떨어|낙하|비래", evidence + " " + action)
    hanging = re.search("늘어|처져|느슨|이탈", evidence) and re.search("고정|제거|보수", action)
    slope = re.search("토사면|사면|흙막이|옹벽|비탈면|절개면", evidence)
    collapse = re.search("붕괴|균열|갈라|무너|침하|벌어|유실|함몰", evidence)
    slope_control = re.search("보강|지반.{0,5}(점검|확인)|사면.{0,5}(점검|확인)|안정성", action)
    if slope and collapse and slope_control and not footing:
        return "collision_contact"
    if (elevated and material and lower_zone and exclusion and not slope
            and not re.search("구조 상태|지반 안정|구조 안전", action)
            and (falling or hanging)):
        return "struck_crushed"
    if structure and deformation and stabilization and not (footing or falling or lower_zone):
        # The official training conversion uses collision_contact for canonical
        # 'other', including structural instability. The output formatter maps back.
        return "collision_contact"
    return None


def reconcile(prediction):
    occupied = {h["code"] for h in prediction["hazards"]}
    for hazard in prediction["hazards"]:
        if hazard["code"] not in {"slip_trip", "fall_height", "struck_crushed", "collision_contact"}:
            continue
        mechanism = supported_mechanism(hazard["evidence"], hazard["prevention"])
        if mechanism and mechanism not in occupied:
            occupied.add(mechanism)
            hazard["code"] = mechanism
