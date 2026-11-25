"""Format code-only image predictions into the official offline v1.0 schema.

Risk pairs are code-level starting estimates. The classifier supplies no
bounding boxes, observations, or measured exposure levels.
"""

import re


# evidence, preventive action, applicable organizer-law article, likelihood, severity
ITEMS = {
    "fall_height": (
        "높이차에 따른 추락 위험이 후보로 판독됐다. 사람이 설 위치와 단부·개구부 사이의 거리, 발판과 가장자리 방호 상태는 추가 확인이 필요하다.",
        "높이차가 있는 작업면과 접근로를 확인할 때까지 접근을 제한한다. 노출된 단부에는 난간을, 개구부에는 충분한 강도의 고정 덮개나 난간을 설치한 뒤 작업을 재개한다.",
        "OSH-43", 3, 4),
    "slip_trip": (
        "같은 높이에서 발이 걸리거나 미끄러질 위험이 후보로 판독됐다. 통행면의 장애물·오염·단차와 실제 보행 경로가 겹치는지 확인해야 한다.",
        "보행 구간의 자재와 케이블은 통로 밖에 정리하고, 젖거나 오염된 바닥은 통행을 잠시 제한해 닦는다. 보행로 확보와 바닥 상태를 확인한 후 통행을 재개한다.",
        "OSH-3", 3, 2),
    "caught_in_machine": (
        "기계의 움직이는 부분에 끼이거나 감길 가능성이 후보로 판독됐다. 손·의복의 접근 범위와 회전부·구동부 방호 상태를 확인해야 한다.",
        "회전부와 구동부에 손이 닿지 않도록 방호 덮개와 울을 점검한다. 이상이 있으면 설비를 정지하고 담당자가 에너지를 차단한 후 방호 상태를 복구한다.",
        "OSH-87", 2, 3),
    "struck_by_lifting": (
        "물체가 떨어지거나 날아오거나 인양 중 사람과 접촉할 위험이 후보로 판독됐다. 물체의 지지·결속 상태와 하부 또는 이동 경로의 인원 위치를 확인해야 한다.",
        "물체 하부와 예상 이동 범위를 출입금지 구역으로 정한다. 담당자가 물체의 지지·결속을 확인하고, 낙하·비래가 예상되는 통로에는 방호설비를 마련한 뒤 작업한다.",
        "OSH-14", 3, 4),
    "electrical_energy": (
        "전기에너지와 접촉할 위험이 후보로 판독됐다. 작업자가 손을 뻗는 범위와 전기설비의 절연·충전부 덮개 상태는 현장 확인이 필요하다.",
        "전기설비에 임의로 접촉하지 않도록 접근을 통제한다. 전기 작업 담당자가 적절한 정전 절차와 무전압 확인을 거쳐 충전부 덮개나 절연 방호를 점검·복구한다.",
        "OSH-301", 2, 4),
    "drowning_inundation": (
        "물에 빠지거나 침수될 위험이 후보로 판독됐다. 작업 위치와 수면·유입 경로, 대피로의 관계 및 수위 변화는 현장에서 확인해야 한다.",
        "수변·침수 우려 구역 출입을 통제하고 대피로와 수위 확인 담당자를 정한다. 수중 추락 위험이 있는 장소에는 구명구와 구조 장비를 접근 가능한 위치에 비치한다.",
        "OSH-47", 2, 4),
    "fire_explosion": (
        "화재·폭발 또는 압력 파열 위험이 후보로 판독됐다. 점화원과 가연물, 압력설비가 작업 범위에 함께 있는지는 확인이 필요하다.",
        "작업 책임자가 점화원·가연물·잔류 압력을 점검할 때까지 관련 작업을 중지한다. 화기 작업이 필요하면 가연물을 격리하고 소화설비와 화재 예방조치를 갖춘 뒤 시작한다.",
        "OSH-241", 3, 4),
    "confined_space": (
        "밀폐공간 출입 위험이 후보로 판독됐다. 출입구·환기 경로를 확인해야 하며 산소와 유해가스 농도는 사진으로 측정할 수 없다.",
        "출입을 통제하고 담당자가 외부에서 산소와 유해가스 농도를 측정한다. 작업 전과 작업 중 적절히 환기하고 감시인·구조 절차를 마련한 후 출입을 허용한다.",
        "OSH-620", 2, 4),
    "chemical_exposure": (
        "유해물질의 누출·접촉 또는 흡입 가능성이 후보로 판독됐다. 물질 종류와 누출 여부, 사람이 노출될 수 있는 경로는 현장 확인이 필요하다.",
        "확인되지 않은 물질에 접근하거나 직접 만지지 않도록 구역을 통제한다. 담당자가 물질안전보건자료에 맞는 보호구와 누출 대응 절차를 확인하고 취급설비의 밀폐·누출 방지 상태를 점검한다.",
        "OSH-299", 2, 3),
    "thermal_weather": (
        "온도 또는 기상 조건에 따른 노출 위험이 후보로 판독됐다. 실제 기온·체감 조건·풍속과 작업자의 노출 시간은 측정·확인이 필요하다.",
        "현장 온열·한랭 조건을 확인해 작업 시간과 휴식을 조정한다. 강풍·폭우 등 악천후로 작업 안전을 확보하기 어렵다면 해당 작업을 중지하고 안전한 곳으로 대피한다.",
        "OSH-37", 2, 3),
    "other": (
        "정해진 유형으로 원인을 특정하기 어려운 위험 후보가 판독됐다. 작업 위치와 주변 구조물·자재 상태를 함께 확인해 노출 범위를 정해야 한다.",
        "담당자가 위험원과 접근 범위를 확인할 때까지 주변 통행을 제한한다. 구축물의 손상이나 변형 징후를 확인하면 안전성을 평가하고 필요한 보강을 마친 뒤 작업을 재개한다.",
        "OSH-52", 3, 4),
}

LIKELIHOOD = ("", "최하", "하", "중", "상", "최상")
SEVERITY = ("", "소", "중", "대", "최대")
BANDS = ((3, "매우낮음"), (6, "낮음"), (8, "보통"), (12, "약간높음"),
         (15, "높음"), (20, "매우높음"))


def make_row(image_id, codes):
    """Return one independent row, retaining confidence order and up to three unique codes."""
    if not isinstance(image_id, str) or not re.fullmatch("[a-f0-9]{32}", image_id):
        raise ValueError("Expected a manifest image ID")
    if isinstance(codes, str):
        raise ValueError("Expected a sequence of code strings")
    hazards = {}
    for code in dict.fromkeys(codes):
        if code not in ITEMS:
            raise ValueError("Unknown risk code")
        if len(hazards) == 3:
            break
        evidence, prevention, law, likelihood, severity = ITEMS[code]
        score = likelihood * severity
        grade = next(name for top, name in BANDS if score <= top)
        hazards[code] = {"evidence": evidence, "prevention": prevention,
                         "references": [{"kind": "law", "id": law}],
                         "risk_level": {"likelihood": {"grade": LIKELIHOOD[likelihood], "score": likelihood},
                                        "severity": {"grade": SEVERITY[severity], "score": severity},
                                        "risk": {"grade": grade, "score": score}}}
    row = {"schema_version": "1.0", "image_id": image_id,
           "status": "hazard" if hazards else "no_visible_hazard", "hazards": hazards}
    if not hazards:
        row["scene_evidence"] = "사진 전체에서 지정 위험 유형을 지지하는 단서가 분류 기준에 이르지 않았다. 가려진 영역과 사진 밖의 작업 상태는 현장 점검이 필요하다."
    return row
