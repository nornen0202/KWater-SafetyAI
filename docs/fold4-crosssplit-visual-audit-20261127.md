# Fold 4 교차 현장과 근접 중복 감사

**동일 장면으로 확인한 교차 링크 3개가 기존 Fold 4 검증 그룹 3개에 걸쳐 있었다.** 주평가는 기존 175장·39그룹으로 유지한다. 별도 민감도에서는 해당 그룹 전체 68장(38·29·1장)을 양쪽 조건에서 같은 마스크로 제외하면 **107장·36그룹**이 남는다. 68장은 영향을 받는 추정 그룹의 전체 크기다. 직접 확인한 것은 세 사진쌍의 장면 대응이다. 모델 출력, 위험 라벨과 성능은 감사에 사용하지 않았다. [CPU 집계](../results/fold4-crosssplit-audit.json)

## 전쌍 점수와 시각 대조

Fold 4 175장과 기존 훈련 쪽 Fold 0~3의 791장·153그룹 사이 **138,425쌍**을 계산했다. 기존 Fold 0 감사의 개발 목록 SHA와 고정 SigLIP2·DINOv2 2,176차원 캐시 SHA를 대조한 뒤, 첫 뷰에서 각 인코더 코사인을 구해 평균했다. 최대 코사인은 0.9712이고, 0.80/0.85/0.90/0.95 이상은 각각 151/27/5/2쌍이다. 같은 접두사에서 숫자 차이가 1~2인 인접 ID 교차는 41쌍이다. ID 인접은 촬영 순서나 현장 일치의 증거로 간주하지 않았다.

각 그룹쌍의 가장 높은 후보를 골라 전체 상위 16쌍, 추가 교차 출처 6쌍, 추가 인접 ID 6쌍을 선정했다. 28쌍에서 SIFT/ORB 상호 기술자 매칭과 호모그래피 RANSAC(오차 3픽셀)을 계산했다. 기존 기하 기준의 inlier 16개 및 비율 0.18을 모두 넘은 쌍은 양 방식 모두 0개였다. 최대 inlier는 SIFT 11, ORB 15였다. 시점과 축척이 크게 달라 자동 문턱을 넘지 않은 사례를 놓치지 않도록, 점수·라벨 없는 접촉 시트 14장을 육안으로 직접 대조했다.

| 시각 판정 | 쌍 |
|---|---:|
| 동일 가시 장면 확인 | 3 |
| 같은 현장일 가능성이 있으나 확인 근거 부족 | 9 |
| 가시 장면의 구조 또는 설비 배치가 다름 | 16 |

확인한 유형은 같은 고층 외벽의 창과 중앙 마스트 앵커 배열, 같은 제조 라인의 기둥 표지판과 설비 배치, 같은 도로 레일 절단의 작업 위치·받침·호스 대응이다. 모두 시점이나 구도가 다른 사진이다. 앞의 두 쌍은 같은 외벽 또는 설비 구간을 보여 주며, 세 번째는 같은 작업 장면의 광각과 근접 구도다. 촬영 시각의 일치는 확인하지 못했다. 최고 코사인 0.9712의 동바리 후보는 반복 격자에서 같은 위치를 확인할 독특한 단서가 부족해 보류했다. 보류와 다른 가시 장면 판정은 실제 출처 독립성의 증명이 아니다.

전수 계산 범위는 코사인 비교다. 기하와 육안 검토는 28개 우선 후보에 적용했으므로 검증·훈련의 모든 실제 현장 교차를 확인했다고 볼 수 없다. 확인된 그룹 마스크는 같은 자료에서의 민감도 분석이며 신규 분할이나 학습 경로를 바꾸지 않았다. 원본·접촉 시트·ID·판정은 Git에서 제외한 로컬 `data/`에 둔다. 이 결과는 언제 처음 평가에 노출됐는지나 모델 성능을 판정하지 않는다.

## 재현

프로젝트 루트의 기존 NumPy/Pillow/OpenCV 로컬 환경에서 실행한다. `scan`은 새로운 로컬 작업 폴더에서 기존 그룹·캐시로 재현한다. `finalize`는 전체 로컬 판정표와 매핑 해시가 일치해야 집계와 마스크를 쓴다. 코드 실행에는 torch가 필요하지 않다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe -m kwater_lab.fold4_crosssplit_audit scan --local data/fold4-crosssplit-review-reproduction
& C:/Users/JY/anaconda3/python.exe -m kwater_lab.fold4_crosssplit_audit finalize
```

## 실제 예측 도착 후 107장 재집계

독립 CPU 경로는 원래 175개 사진 ID가 각 조건에 중복 없이 모두 존재하는지 먼저 검사하고, 동결 마스크를 그 뒤에 각 조건에 똑같이 적용한다. 기존 Fold 0 민감도 지표 및 그룹 쌍체 bootstrap을 재사용한다. 기본 입력은 현재 v3 평가 코드가 계획한 8코드·11코드 단독 JSONL 경로다. 실제 공식 LoRA Fold 4 출력도 로컬 canonical ID로 대응된 파일이 준비된 경우에만 선택 인수 `--official`로 더한다. 이전 Fold 0 공식 결과를 Fold 4에 대입할 수 없다.

```powershell
& C:/Users/JY/anaconda3/python.exe -m kwater_lab.fold4_crosssplit_audit score --score-output results/fold4-fixed-mask-sensitivity.json
```

이 감사 시점에는 175장씩의 두 실제 출력 파일이 준비되지 않아 성능 수치는 쓰지 않았다. 로컬 검토의 기하 배열은 검증→훈련 순서로 계산했고 접촉 시트 방향은 별도로 가렸다.

## 3개 모델의 175장·107장 쌍체 채점

세 파일이 모두 준비된 뒤 아래 명령을 실행한다. 각 파일의 스키마와 원래 175개 로컬 ID를 전부 검사하고, 예측 순서가 달라도 같은 ID끼리 비교한다. 107장 패널은 사전 고정한 3개 검증 그룹을 세 모델에 동일하게 제외한다. 이 채점은 위험 코드, 정상 사진의 오경보, 심각도 4 사진과 코드의 미탐, 위험 코드 추가·제거율을 집계한다. 세 비교 모두 2,000회 그룹 단위 쌍체 부트스트랩을 사용하며, 차이의 방향은 두 번째 모델에서 첫 번째 모델을 뺀 값이다. 공식 평가 도구의 스키마 검사는 별도로 수행한다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& C:/Users/JY/anaconda3/envs/env_aias_test/python.exe -m kwater_lab.fold4_crosssplit_audit score-three
```

107장 중 최대 그룹 49장을 추가 제외한 58장·35그룹에서는 탐색적 ΔF1과 모델별로 활성 코드에서 빠진 코드도 함께 기록한다. 기본 집계 출력은 `results/fold4-paired-code-evaluation.json`이다.

## 별도 v3 1에폭 체크포인트의 2군 채점

v3 1에폭 단독 평가에서 생성한 175행, 독립 검산된 8종 175행과 각 생성 영수증을 연결한다. `score-epoch1`은 두 파일의 원래 ID를 확인한 뒤 175장과 107장을 채점한다. 이 결과는 2에폭 단독·하이브리드 비교와 다른 파일에 저장된다.

```powershell
python -m kwater_lab.fold4_crosssplit_audit score-epoch1 --legacy8 $Baseline175 --epoch1 $V3Epoch1
python -m kwater_lab.fold4_official_output_check --legacy8 $Baseline175 --epoch1 $V3Epoch1 --workspace data/local-eval-predictions/fold4-epoch1-official-check --out results/fold4-epoch1-official-output-check.json
```

기본 정량 결과는 `results/fold4-v3-epoch1-paired-code-evaluation.json`이다. 원생성·최종 파일과 체크포인트·프롬프트·생성 설정·시간의 출처를 결과와 함께 기록한다.
