# [긴급 작업 정렬 지시문] Antigravity(주 GPU 실행) ↔ Codex GPT(CPU 검증·정성 보강) 역할 분담

- **발효 시각**: 2026-11-27 11:12 KST
- **목표**: 금일(11/27) 자정 전 **Fold 4 공정 미노출 대조 검증 완료** 및 **전량(`1,045장`) 본학습 기반 5차 제출 ZIP 완성**

---

## 1. 핵심 정렬 결정 (즉시 적용)

1. **Windows 로컬 GPU(`RTX 3060 12GB`) 전담권 일원화 (Antigravity 전담)**
   - 단일 12GB GPU에서 복수 에이전트가 동시에 모델을 적재하면 VRAM 스필오버 및 드라이버 리셋(`Event 153`)이 발생합니다.
   - 따라서 지금부터 **Windows 호스트의 모든 CUDA/GPU 학습·추론·패키징 실행은 Antigravity 세션이 단독 수행**합니다.
   - 기존 6장(`3쌍`) 규모 `paired_verifier_ab` GPU 작업은 표본 부족(미학습 코드 67%) 및 승격 보류 대상이므로 종료합니다.
   - **Codex 세션은 Windows에서 절대 GPU 프로세스를 띄우지 말고 반드시 `$env:CUDA_VISIBLE_DEVICES = '-1'`(CPU 전용)로만 작업**해 주십시오.

2. **파일 충돌 방지 경계 (File Ownership)**
   - **Antigravity 전용 수정 경로 (Codex 수정 금지)**:
     - `kwater_lab/system_prompt_v3.txt`, `kwater_lab/v3_turnaround.py`, `kwater_lab/package_submission.py`
     - `reports/live-turnaround-execution-report.md`
     - `data/v3-fold4-*`, `data/v3-full1045-*`, `data/submissions/safeground_v5*`
   - **Codex 전용 작업 경로**:
     - `kwater_lab/fold4_crosssplit_audit.py` → `results/fold4-crosssplit-audit.json`, `docs/fold4-crosssplit-visual-audit-20261127.md`
     - `kwater_lab/prevention_law_guard.py` + `tests/test_prevention_law_guard.py` → `results/prevention_law_guard_eval.json`

---

## 2. Codex 세션에 요청하는 3가지 고부가가치 CPU 전담 과업

Antigravity가 GPU에서 **① Fold 4 공정 대조(신규 학습 8종 790장 vs 개선 11종 `v3` + 내측 OOF 하드네거티브 & 1-토큰 로그확률 마진 검증기)** 및 **② 전체 `1,045장` 본학습·패키징**을 수행하는 동안, Codex는 강점인 아래 **CPU 정밀 감사 및 정성 30점 방어 모듈**을 완성해 주십시오:

### 과업 A: `Fold 4`(`175장` 평가 vs `791장` 훈련) 교차 현장·근접 중복 정밀 감사 (`CPU 전용`)
- Fold 0에서 수행했던 것과 동일한 기준(고정 `SigLIP2+DINOv2` 코사인 상위 후보 + `SIFT/ORB + 호모그래피 RANSAC` 기하 검증 + 시각 대조)으로 **Fold 4(175장·39그룹)와 훈련 측(Fold 0~3, 791장) 사이의 실제 동일 현장 교차 쌍 유무**를 전수 점검해 `results/fold4-crosssplit-audit.json` 및 `docs/fold4-crosssplit-visual-audit-20261127.md`에 기록해 주십시오. (ID 원문은 `data/`에만 보관하고 `results/`에는 집계·해시만 기록)

### 과업 B: 정성 30점(`prevention` 12점 + `references` 6점) 직접 지지율 향상용 `prevention_law_guard.py` 구현 (`CPU 전용`)
- 앞서 Codex 블라인드 12장 감사에서 발견한 두 가지 정성 감점 요인을 CPU 결정론 모듈(`kwater_lab/prevention_law_guard.py`)과 단위 테스트(`tests/test_prevention_law_guard.py`)로 구현해 주십시오:
  1. **조치-조문 행위 정합성 재정렬(`rerank_top1_by_action_support`)**: 상위 3개 후보 조문(`OSH-*`) 중 모델이 생성한 `prevention`의 핵심 행위·대상(예: `보호구/방진마스크 착용 → OSH-32`, `살수/물 뿌리기 → OSH-4-2`, `비계/거푸집/동바리 구조 보강 → OSH-52`, `출입금지/표지 설치 → OSH-20`, `장비 유도자/접근통제 → OSH-200`, `전원 차단/충전부 절연 → OSH-301·304`)과 직접 부합하는 조문 1개를 1순위로 선택하는 함수 작성 및 기존 216장 생성문 재생 평가.
  2. **위험 작업순서 오류 감지(`check_unsafe_action_order`)**: 감전 위험에서 전원 차단 없이 충전부 접촉 조치를 권고하거나 화기 불티 방향에 작업자를 두는 등 안전상 역순 문장을 탐지하는 검사 함수 작성.

### 과업 C: 오프라인 제출 규격(`wheelhouse`·`seccomp`·`assets.json`) 독립 검증 준비
- `G:` 드라이브 분리나 AI-Hub 357GB 복사는 수행하지 마십시오(외부·합성 사진 미사용 확정).
- Antigravity가 `data/submissions/`에 5차 제출 후보 ZIP을 생성하면 즉시 교차 검산할 수 있도록 CPU 기반 아카이브·스키마·해시 감사 스크립트와 기존 단위 테스트(`unittest discover -s tests`) 통과 상태를 유지해 주십시오.
