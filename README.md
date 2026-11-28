# K-water 현장 안전 AI 연구 기록

**4B·8B를 791장, 같은 138개 학습 업데이트로 비교했다.** 첫 에폭에서 8B의 macro-F1은 175장 0.63415, 교차 현장을 제외한 107장 0.62091로 4B보다 각각 0.09214·0.05838 높았다. 107장 중대 위험 완전 미탐은 18→20장으로 늘었다. 27B 4비트 수정 ZIP은 17,801,904,939바이트로, 같은 200장을 26분 31초에 처리했다. 6차 단독은 18분 59초, 7차 전량 하이브리드는 24분 57초였다. [서울 L4 실험 원장](reports/seoul-l4-roadmap-experiments-20261128.md) · [두 에폭의 그룹별 불확실성 그림](reports/figures/qwen-scale-fold4-20261128.png)

**11월 28일 연구 로드맵.** 4B·8B는 같은 11종 사진·학습 예산으로 비교하고, Qwen3.8-27B 4비트는 배포 조건을 통과했다. 107장 품질 검사에서는 F1 0.51522, 정상 오경보 19/27, 중대 미탐 12/39로 직접 제출을 보류했다. 어댑터 분리와 오경보 제거는 서로 다른 실험으로 검증한다. [카본 리뷰 재검토와 마스터 로드맵](reports/comprehensive-audit-and-vlm-roadmap-20261128.md) · [모델별 비교 조건](reports/comprehensive-audit-and-vlm-roadmap-20261128.md#paired-models) · [비전 태스크 모델(SAM·YOLO·Anything) 감사 및 3단계 게이트](reports/vision-task-models-audit-and-roadmap-20261128.md) · [이슈 #2 심층 감사 및 정성 30점 보완 설계](reports/issue-2-deep-review-and-roadmap-alignment-20261128.md)

**서울 L4에서 같은 4B 학습 8스텝은 로컬 652.78초, L4 251.27초였다.** 11월 28일 4B·8B 학습 비교와 실제 ZIP 리허설, 공개 27B의 Fold 4 추가 검사까지 총 추정 9,273원(예비비 적용 10,566원)이었다. 사용한 VM·디스크를 모두 삭제했다. 계획 당시의 누적 연구 예산은 약 14,300원, 독립 재학습·Private 리허설까지 확장하면 약 28,800원이었다. 환율 1,500원·한국 VAT·Spot 학습을 반영한 견적이다. [서울 실측 원장](reports/fold4-v3-seoul-l4-benchmark-20261128.md) · [누적 비용](reports/comprehensive-audit-and-vlm-roadmap-20261128.md#l4-budget) · [상금별 손익분기](reports/comprehensive-audit-and-vlm-roadmap-20261128.md#prize-roi)

**11월 28일 7·8차 제출 전략.** 12:33 KST에 27/133위, 53.65점으로 당시 20위까지 2.24점, 10위까지 6.54점이었다. 7차는 서울에서 200장을 통과한 전량 4B 하이브리드로 Public 접수 승인, 8차 전량 첫 에폭 단독은 실물 검사 후 별도 결정이다. 8B 초벌+4B 검증기의 Fold 4 이득과 20 GB 용량 한계도 검산했다. [요약·그림·실행 현황](reports/submission-7-8-decision-and-l4-value-20261128.md) · [직전 개발 평가](reports/seventh-submission-decision-20261128.md)

**11월 27일 제출 결과.** 21:22에 접수한 새 8종 모델은 Public **47.35**, 22:22에 접수한 11종 v3 첫 에폭 단독 모델은 **53.65**를 기록했다. 기존 42.04에서 **11.61점 상승**했다. 23:16 KST 기준 **100팀 중 12위**로 10위까지 1.55점 차이였다. 이 PC에서 `data/local-reports/issue2-20261127/visual-review.html`을 열면 실제 현장 사진 54장을 볼 수 있다. 6장은 같은 사진에 두 제출이 쓴 근거·조치·출처를 비교하고, 48장은 전체 사진과 확대 영역을 함께 살펴볼 수 있다. [이슈 #2 연구 로드맵](reports/issue-2-transfer-experiments-20261127.md) · [Fold 4 실험과 제출 판단](reports/fold4-fifth-submission-decision.md) · [5차 영수증](results/public_submission_005.json) · [6차 영수증](results/public_submission_006.json)

[Ubuntu 26.04 보조 PC 복원 및 3050 실험 인계](docs/ubuntu-3050-handoff.md)

**11월 27일 인계.** 00:05 KST 공개 순위는 도니 30/52위(42.04점), 10위 48.49점이었다. 같은 사진에서 기준 모델과 새 11종 모델을 비교하는 실험, 과거 보류 표본의 노출 이력, 전문가 후보의 적중·오탐 및 인용 재생 결과는 [11월 27일 실험 인계 및 제출 판단 근거](reports/turnaround-handoff-20261127.md)에 있다.

[시각화 연구 보고서](reports/competition-research-2026-11-25.md) · [PDF 보고서](reports/competition-research-2026-11-25.pdf) · [CLIP·SigLIP 2 실측과 생성 VLM 도입 검토](docs/vlm-model-selection.md) · [후속 실험 설계](docs/improvement-experiments.md) · [외부 데이터 전후 비교](reports/external-first-experiment.md) · [F1 0.25 돌파 심층 분석 및 혁신 전략](docs/f1-breakthrough-strategy-2026-11-26.md)

제5회 K-water AI 경진대회 현장 사진 위험성평가 부문 연구. 2026년 11월 25일 Codex에 등록한 로컬 프로젝트는 `G:\Projects\KWater-SafetyAI`이다. 새 Codex 대화에서는 이 폴더를 열면 코드·실험 기록·참가자 자료를 이어서 사용할 수 있다. 기존 공개 실험과 첨부 계획서를 옮겼고 공식 배포 ZIP 세 개를 확보했다. 11월 25일 20:42(KST) 공식 Qwen3-VL-4B+LoRA와 23:57(KST) CLIP 224 + SigLIP 2 384 결합 모델을 각각 접수했다. 11월 26일 13시 공식 Public 점수는 **1차 42.04, 2차 16.12**, 팀 순위는 **11/18위**다. 최고점 자동 선택은 1차다. [1차 접수](results/public_submission_001.json) · [2차 접수](results/public_submission_002.json)

## 참가자 데이터와 연구 결과

같은 170장의 내측 F1 보정 결과는 CLIP 224 **0.223**, SigLIP 2 224 **0.209**, SigLIP 2 384 **0.239**였다. 고해상도 SigLIP 2와 CLIP의 분류 점수를 절반씩 섞으면 **0.247**이었다. CLIP 대비 무위험 오경보는 31→26/66, 정답 위험을 하나도 맞히지 못한 사진은 47→42/104, 추락 적중은 12→13/34건이었다. 2차 배포 임계값 0.20으로 동일 탐색 분할의 예측을 재평가한 F1은 **0.2508**, 미관측 오경보는 **30/66장**이다. 새 사건에서도 확인할 예정이다. 생성기 교체도 시험했다. 공개 사진 한 장에 대한 한국어 질문 두 개에서 Qwen3.5-4B는 매번 정답 0/2건, 원본 Qwen3-VL-4B는 매번 1/2건을 찾았다. 첫 제출은 공식 생성기, 두 번째는 결합 분류기와 코드별 문구다. [비교 그림과 연구 보고서](reports/competition-research-2026-11-25.md) · [모델별 평가와 선택 근거](docs/vlm-model-selection.md)

| 공식 패키지 | 원본 바이트 | 프로젝트에서의 용도 |
|---|---:|---|
| `kwater-local-kit.zip` | 2,713,157 | 입력 10장, 제출 형식 및 출력 검사 |
| `kwater-train.zip` | 1,802,107,979 | 학습 라벨, 사진 및 중복 감사 |
| `kwater-baseline.zip` | 9,032,392,642 | 공식 Qwen 가중치·LoRA, 추론 코드 및 법령 |

세 ZIP 모두 N: 다운로드본과 G: 보관본의 SHA-256이 일치하고, 압축 항목별 CRC 검사를 통과했다. [패키지별 검증 기록](results/package_manifest.json)

원본 다운로드 폴더는 `N:\User Manual\Downloads`이며 G 프로젝트의 보관 위치는 `data/packages`다. 학습셋은 1,045장 전부 라벨 형식을 통과했다. 위험 730장, 위험 미관측 315장이고, 186장은 복수 위험을 포함한다. 걸림은 317장, 추락은 224장, 밀폐공간·익수/침수·온도/기상은 각 1장이다. 제공 모델의 코드 매핑은 8종으로, 후자의 3종을 출력할 수 없다. 코드별 F1의 단순평균을 쓰는 대회이므로 이 세 유형을 독립적으로 판별하는 것이 모델 설계 과제다. 한 장으로는 해당 유형의 재현율을 신뢰성 있게 추정하기 어렵다. [라벨 집계](results/train_audit_labels.json) · [학습 ZIP 해시와 CRC](results/train_audit_package.json)

학습 라벨의 법령 인용 1,086건은 모두 동봉 자료에 실재하는 70개 조문으로 연결됐다. 출처 검색 실험은 이미 탐색용으로 지정한 170장에서 법령이 붙은 위험 147건을 사용했다. 검색어에서 명시적 조문 번호를 지웠고, 정답의 한국어 근거·조치와 제공 법령만 사용했다. 인용 조문이 상위 3개 안에 든 경우는 같은 질의에서 어절 BM25 17/147건, 글자 2개 단위 BM25 30/147건이었다. 예방조치만 넣은 후자는 31/147건이었다. 실제 모델이 생성한 설명·조치의 출처 적합성은 별도 평가 대상이다. [인용 ID 감사](results/train_audit_references.json) · [검색 방법별 결과와 해시](results/law_retrieval_pilot.json)

같은 147건에서 학습 사례를 활용한 효과도 비교했다. 한 번에 사진 ID 접두사 한 묶음을 빼고 나머지 다섯 묶음에서만 사례를 가져왔다. 정답 위험 코드의 최빈 조문을 고르면 118건(80.3%), 정답 코드·근거·조치를 공식 검색기에 넣고 다른 묶음의 사례를 참조하면 124건(84.4%)에서 첫 후보가 정답 인용에 포함됐다. 사례 없이 공식 검색기를 쓰면 15건(10.2%)이었다. 동봉된 전체 사례집은 사용하지 않았다. 완성 모델에서는 위험 코드·근거·조치 자체가 예측값이므로, 이 결과는 올바른 판독이 주어졌을 때 기대할 수 있는 검색 동작을 보여준다. [6개 묶음별 검색 결과](results/law_examples_pilot.json)

연구 추론기의 실제 저장 출력에도 법령 검색을 연결했다. 공개 사진 S16C159에서 걸림에는 정답 조문 `OSH-3`이 붙었고, 잘못 예측한 감전에도 실재하는 `OSH-301`이 붙었다. 놓친 추락은 관련 조문을 검색할 기회 자체가 없었다. 공식 출력 검사는 1/1 유효였다. 해당 조문이 실재하는지와 사진에서 조치를 뒷받침하는지는 별도로 평가할 필요가 있다. 실제 생성 출력의 조문 검토 (로컬 `results/public_law_replay_review.json`)

전체 사진 중 동일 화소는 없었고, 64비트 dHash의 서로 다른 비트가 4개 이하인 쌍도 없었다. 12개 이하로 넓히면 47쌍이다. 대기열에는 1,045장 모두의 최근접 후보를 추가했다. 후보를 확인하기 전까지 독립 사건 그룹은 미확정이다. 탐색 170장, 공개 예시, 키트 입력과 일치하거나 유사한 사진을 합쳐 182장을 최종 보류 제외 목록에 기록했다. [사진 감사](results/train_audit_summary.json) · 검토할 사진 쌍 (로컬 `data/group-review/dhash_pairs.csv`) · 보류 제외 목록 (로컬 `data/holdout-exclusions.json`)

### 위험 분류와 오경보

탐색 사진은 위험 104장, 위험 미관측 66장이다. 고정된 CLIP 영상 특징에 선형 분류기를 학습하고, 매번 ID 접두사 한 묶음을 빼서 그 묶음의 예측을 평가했다. 보정이 필요한 정책은 남은 묶음 안에서만 임계값을 선택했다. 여섯 번의 예측을 합친 결과다. [탐색 기준선](results/prefix_pilot.json) · [오경보 제약 실험](results/prefix_pilot_fpr20.json)

| 임계값 선택 방식 | Macro F1 | 미관측 사진에 위험을 한 가지라도 예측 | 위험 사진에서 위험을 하나도 예측하지 못함 |
|---|---:|---:|---:|
| 0.5 고정 | 0.195 | 13/66 | 37/104 |
| 내부 Macro F1 최대화 | 0.223 | 31/66 | 19/104 |
| 내부 오경보율 20% 이하에서 Macro F1 최대화 | 0.189 | 9/66 | 41/104 |

20%는 첫 결과를 보고 도입한 연구용 목표다. 접두사 묶음 둘에서는 외부 오경보율이 20%를 넘었다. 탐색 사진에는 희귀 3종 양성이 없으므로 별도로 기존 CLIP 텍스트 12개의 순위를 조사했다. 밀폐공간이 170장 중 72장에서 상위 3개에 들어갔다. 희귀 후보를 단순히 상위 3개에서 가져오면 잘못된 밀폐공간 후보가 많다. 우선 시각 근거를 재확인할 조건을 검증하고, 희귀 유형의 재현율 판단은 양성 검증 자료를 확보한 뒤 진행한다. [희귀 후보별 결과](results/rare_candidate_pilot.json)

같은 탐색 자료에서 유사 사진 검색과 선형 분류기를 반씩 결합하면 내측 F1 보정 조건의 Macro F1이 0.223에서 0.244로 올랐다. 위험 사진 중 정답 코드를 하나도 맞히지 못한 사진은 47장에서 37장으로 줄고, 무위험 오경보는 31장에서 37장으로 늘었다. 내측 오경보를 20%로 제한하면 분류기와 결합의 F1은 각각 0.189와 0.197이었다. 비대칭 손실은 내측 F1 보정에서 0.240을 기록했으나 무위험 사진 66장 중 53장에 경보를 냈다. [검색·가중치 편집 집계](results/weight_pilot.json) · [손실함수 비교](results/loss_pilot.json)

### 공식 모델 10장 실행

배포된 **Qwen3-VL-4B와 LoRA**를 Windows RTX 3060 12GB에서 실행했다. ZIP과 대조한 코드·프롬프트 11개가 모두 일치했다. 배치 1에서 모델 원출력이 **10/10** 파싱됐고, 실패를 위험 미관측으로 바꾼 경우는 **0건**이었다. 공식 출력 검사도 **10/10** 통과했다. 예측은 위험 2장과 위험 미관측 8장이었으며, 생성된 법령 참조 3개는 모두 동봉 자료의 실재 조문이었다. 키트 10장은 출력 품질 점수를 계산하는 데이터가 아니다. [공식 모델 실행 기록](results/official_baseline_10.json) · [키트 템플릿 검사 기록](results/official_kit_smoke.json)

모델 로그의 10장 생성은 **90.0초**, 첫 Windows 기동과 가중치 적재를 포함한 공식 프로그램 실행은 **2,035.4초(33분 55초)**였다. GPU 전체 최대 점유는 11,831MiB였다. 기존 Windows용 PyTorch 2.5.1/CUDA 12.4와 G:의 별도 라이브러리 경로를 사용했고 Hugging Face 오프라인 모드를 켰다. L4 Linux 컨테이너의 설치·실행 제한 충족 여부는 이 수치에서 추정할 수 없다. [재현 명령과 환경 차이](docs/official-baseline-windows.md)

## 첫 실험에서 놓친 위험

공식 예시 S16C159에는 덮개 없는 개구부와 바닥 통행을 방해하는 케이블이 있다. 공개 정답은 높이차 추락과 같은 높이 걸림 두 가지다. 모델 입력에는 사진과 공통 위험 코드 설명을 넣었다. 사진 번호, 공식 캡션, 장면 메모, 정답 코드는 전달하지 않았다. [운영진 공개 예시 및 이용 조건](https://hackathon.imaginationgroup.co.kr/kwater/#core-data)

| 조건 | 실제 출력 또는 순위 | 관찰 |
|---|---|---|
| Qwen3-VL-4B-Instruct Q4_K_M, 직접 판독 | 감전, 걸림, 밀폐공간 | 걸림을 맞혔다. 모델이 바닥의 구멍을 묘사하고도 추락을 놓쳤다. |
| 동일 모델, 위치·대상·위험 상태를 먼저 확인하도록 지시 | 감전, 걸림, 밀폐공간 | 코드가 한 개도 바뀌지 않았다. 케이블이 있다는 관찰에서 전기 위험까지 추론했다. |
| CLIP ViT-B/32, 중앙 224픽셀 크롭 | 추락 10위, 걸림 6위 | 위험 11종과 위험 미관측 설명 1종을 비교한 코사인 유사도 순위다. |
| 동일 모델, 화면 전체를 224픽셀에 맞춰 여백 추가 | 추락 8위, 걸림 9위 | 두 정답 모두 상위 세 후보 밖에 있었다. |

**연구 판단**은 CLIP의 제로샷 상위 3개로 Qwen의 후보를 제한하지 않는 것이다. Qwen은 대상의 존재를 어느 정도 읽었지만 사고 유형으로 연결하는 단계에서 틀렸다. 다음 독립 비교는 추락↔밀폐공간, 걸림↔감전 오분류와 위험 미관측 사진 오탐에 초점을 맞춘다. 공개 예시에서 발견한 오류와 개선 가설은 모두 탐색 결과로 취급한다.

Qwen은 공개 기반 모델을 양자화한 것으로 대회가 제공하는 학습된 LoRA가 포함되지 않았다. 두 조건은 한 번 기동한 RTX 3060 12GB 및 llama.cpp Vulkan에서 차례로 실행했다. 첫 서비스 기동 208.442초, 요청 25.167초와 9.180초였다. 순서와 캐시 효과가 섞여 있으므로 두 시간의 차이로 모델을 선택할 수 없다. 비공개 200·300장 처리시간은 L4 환경에서 별도로 확인해야 한다. Qwen 실행 기록 (로컬 `results/visual_probe.json`) · CLIP 순위와 버전 (로컬 `results/clip_probe.json`)

### 후속 실험에서 바뀐 것

공개 정답과 최초 오류를 본 후 11개 코드에 각각 관측 기준을 한 문장씩 더했다. 같은 조건에서 감전·걸림·밀폐공간이 다시 나왔다. 이 장면에서는 프롬프트에 경계를 설명하는 것만으로 추락 미탐을 고치지 못했다. CLIP은 새 프로세스에서 재실행해 두 전처리의 순위와 코사인 값이 각각 일치하는 것도 확인했다. 코드 정의 보강 실험 (로컬 `results/taxonomy_followup.json`)

별도 실험에서는 한국어 근거와 조치, 가능성·중대성을 함께 생성하고 위험도 곱과 등급을 코드로 확정했다. 모델 준비 후 한 장에 7.52초가 걸렸다. 결과는 걸림·감전 2개였고 두 항목의 위험도 형식은 모두 유효했다. 감전 오탐에 붙인 자유생성 조치에는 작업자가 그대로 수행하면 위험한 지시가 있었다. 법령 파일 확보 전이어서 출처는 빈 목록으로 뒀다. 최초 프로토타입 실행 원본 (로컬 `results/prototype.json`)

연구용 추론기에서는 감전으로 판정된 경우에 한해 작업·접근 제한과 자격 있는 전기 담당자의 현장 확인을 요청하는 문구를 사용하도록 수정했다. 저장된 원 응답을 재생해 감전 예방조치 한 필드만 바뀌었고, 수정 전후 출력이 모두 로컬 스키마 검사를 통과했음을 확인했다. 감전 오탐, 추락 미탐과 다른 코드의 자유생성 조치는 여전히 검토 대상이다. 수정 전후 검산 (로컬 `results/prevention_replay.json`)

## 수상 목표에 맞춘 다음 판정

Public 상위 20팀이 Private로 진출하며 합산 상위 10팀은 현장 활용성 40점, 방법론 30점, 재현성 30점의 심사를 받는다. 두 리더보드 모두 코드별 F1 평균이 70점이며, 근거·조치·출처가 각각 12점·12점·6점이다. 작업 우선순위는 아래 결과를 얻는 순서로 정했다. [공식 평가 기준](https://hackathon.imaginationgroup.co.kr/kwater/#evaluation) · [공유 대화](https://chatgpt.com/s/t_6b06928236d88191a253a6ee7c61f91b)

1. **운영 기준선 확보.** 공식 4B+LoRA의 모델 생성 성공 수와 키트 출력 검사 통과 수를 각각 확인한다. 공식 코드는 생성·파싱 실패를 위험 미관측 출력으로 바꿀 수 있다. 배포 어댑터는 제공 학습 데이터를 이미 보았으므로 사후 분할 점수는 실행 진단에만 쓴다. 독립 비교에서는 LoRA를 새로 훈련하고 법령 검색용 예시도 학습 그룹으로 제한한다.
2. **독립 비교.** 1,045장의 원본과 라벨을 검사하고 사례 그룹을 확정한다. 학습 그룹에서만 초기화한 LoRA를 학습한다. 고정 시각 특징에 붙인 선형 분류기, 코드 경계를 보강한 프롬프트를 각각 기존 조건과 비교한다. 개발셋에서 무위험 사진 오탐률, 중대 위험 미탐 수, 코드별 TP·FP·FN을 함께 보고 새로 맞힌 사례와 새로 틀린 사례를 보존한다.
3. **설명과 출처.** 개발셋의 동일 코드에 대해 근거·조치·제공 법령의 연결을 익명 순서로 검토한다. 법령 ID의 존재와 실제 예방조치에 대한 조문 적합성을 따로 기록한다. 작은 대상이 반복해서 누락될 때 확대 실험을 연다.
4. **제출 후보 선정.** 사례 단위 개발 검증과 공식 키트 실행을 통과한 후보의 보류셋을 한 번 평가한다. Public 200장 30분, Private 300장 60분에 설치와 기동이 포함된다. 오프라인 실행, 한 개 JSONL, 코드·가중치·출처 명세·해시가 맞는 후보만 접수 대상으로 삼는다. [공식 제출 안내](https://hackathon.imaginationgroup.co.kr/kwater/#submission)

## 같은 PC에서 재현하기

아래는 이 폴더에서 실행하는 PowerShell 명령이다. 기본 Python 3.14.3과 로컬 NumPy 2.4.4, Pillow 12.3.0, ONNX Runtime 1.26.0, tokenizers 0.23.1을 사용했다. 운영 서버는 별도의 Linux/Python 3.12 환경이다. 실험용 대형 모델과 공식 사진은 G:에 저장한다. 제공 데이터는 대회 참가 목적 로컬 사용과 종료 후 삭제 조건이 적용된다.

```powershell
$labData = 'G:/CodexData/kwater-rd-20261125'
$clipModel = "$labData/clip-vit-base-patch32"
python -m unittest discover -s tests -v
python -m kwater_lab.visual_probe
python -m kwater_lab.clip_probe rank --model $clipModel --image "$labData/visual-probe/public-example.jpg" --output "$labData/clip-repeat.json"
python -m kwater_lab.prototype
python -m kwater_lab.prevention_replay
```

새 Windows Vulkan 환경에서는 `python -m kwater_lab.visual_probe --prepare`로 실험용 공개 이미지·모델·실행기를 받아 추론한다. CLIP은 `python -m kwater_lab.clip_probe fetch --model $clipModel`로 고정 리비전을 받는다. 다운로드가 끝나면 두 모델 모두 로컬에서 추론한다.

로컬 manifest 입력은 `python -m kwater_lab.prototype --input-dir 입력폴더 --output-dir 출력폴더`로 실행한다. 출력은 UTF-8 `submission.jsonl` 한 파일이다. 이 실험기는 해당 PC의 루프백 모델 서버를 사용한다. 공식 채점용 Linux 패키징과 L4 시간 검증은 별도로 진행해야 한다.

생성 출력에 법령을 연결하려면 `python -m kwater_lab.attach_law --input 출력폴더/submission.jsonl --input-dir 입력폴더 --output 후보폴더/submission.jsonl --report 후보폴더/law_report.json`을 실행한다. 입력 사진을 대조해 일치·유사 탐색 사례를 제외하고, 익명 ID의 JSONL만 주어지면 법령 본문에서 검색한다. 후보 파일은 원본과 다른 위치를 지정한다. 공식 Qwen+LoRA의 별도 재현 명령과 환경 정보는 [Windows 실행 문서](docs/official-baseline-windows.md)에 있다.

### 참가자용 학습 데이터 연결

프로젝트 루트에서 다음 명령을 실행한다. 원본 아카이브의 `images/`와 `train.jsonl`은 `data/train` 아래에 있다.

```powershell
$dataset = 'data/train'
$labels = "$dataset/train.jsonl"
python -m kwater_lab.audit $labels --image-root $dataset --output results/train_audit_full.json
python -m kwater_lab.law_retrieval --laws data/baseline/assets/laws.jsonl --baseline-rag data/baseline/scripts/rag.py --labels $labels --exposure data/prefix-pilot/exposure.json --output results/law_retrieval_pilot.json
python -m kwater_lab.law_examples --laws data/baseline/assets/laws.jsonl --rag data/baseline/scripts/rag.py --labels $labels --exposure data/prefix-pilot/exposure.json --audit results/train_audit_full.json --output results/law_examples_pilot.json
```

배포본에 독립 사건 ID가 없어 `sample_id`의 접두부는 탐색용으로만 쓴다. 원본 사진과 유사 후보를 검토해 `sample_id,group_id` 열을 가진 CSV를 만든다. 그룹을 가로지르는 유사 사진 후보는 `left,right,decision` CSV에 `same` 또는 `separate`를 기록한다. 아래 명령은 검토된 그룹 파일이 있을 때 실행한다. 탐색·공개 예시에 노출된 사진과 같은 그룹은 최종 보류에서 제외한다. 첫 감사 출력과 검토 후 출력은 별도 파일로 저장한다.

```powershell
python -m kwater_lab.audit $labels --image-root $dataset --group-map data/reviewed-groups.csv --review-pairs data/reviewed-pairs.csv --exclude-holdout-ids data/holdout-exclusions.json --output results/train_audit_reviewed.json
python -m kwater_lab.clip_probe extract --model $clipModel --audit results/train_audit_reviewed.json --labels $labels --image-root $dataset --output data/features.npz
python -m kwater_lab.train_head --audit results/train_audit_reviewed.json --features data/features.npz --model data/head.npz --report results/head-dev.json
```

특징 추출은 학습·개발 이미지만 열고, 선형 분류기는 학습 특징으로 11개 위험을 적합한다. 학습 빈도만 쓰는 기준선, 고정 임계값 0.5, 개발셋으로 고른 공통 임계값을 구분한다. 같은 개발셋에서 고른 임계값의 점수는 보류 성능과 별도로 기록한다. 두 추론 결과를 비교할 때는 `python -m kwater_lab.metrics --gold dev.jsonl --pred a.jsonl --compare b.jsonl --groups groups.jsonl`을 사용한다. 채점기는 ID 누락·형식 오류가 있으면 비교를 중단하고, 유효한 행에서 정량 70점과 사례 단위 재표집 구간을 계산한다.

## 현재 검증 범위

첫 제출은 배포 모델의 대회 채점 기준선을 얻기 위해 진행했다. 이후 후보는 그 기준선과 같은 공식 조건에서 비교한다. [공식 자료 목록](https://hackathon.imaginationgroup.co.kr/kwater/#core-data)

개발용 통제 이미지 24장을 21개 그룹으로 묶어 학습 16장·개발 4장·보류 4장으로 분할했다. 임시 보류 이미지 파일 4개를 제거한 후에도 중앙 크롭과 전체 화면의 특징 추출, 분류기 학습, 해시 확인이 모두 완료됐다. 이 검사는 코드 연결과 보류셋 격리를 확인한다. 공개 사진 1장은 해당 통제 실험에 들어가지 않았다.

코딩 도구는 실험 코드, 프롬프트, 채점기, 테스트 및 결과 정리에 사용했다. 공개 예시의 정답은 운영진이 제공했다. 기반 모델, 양자화 리비전, 추론기, 프롬프트와 원본 해시는 실행 기록에서 확인할 수 있다. [CLIP 모델 카드](https://huggingface.co/Xenova/clip-vit-base-patch32) · [Qwen 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)

## Git 관리와 후속 실험 재현

11월 25일 23:57에 접수된 [두 번째 제출 ZIP](https://github.com/nornen0202/KWater-SafetyAI/releases/download/submission-20261125-02/kwater-fusion-20261125.zip)을 팀 단독 비공개 GitHub 릴리스에서 받는다. 두 공개 모델 가중치, 학습한 분류 헤드, 직접 작성한 코드와 Linux 런타임 휠은 각각 원래 파일 단위로 Git에도 보관한다. 참가자 전용 사진·라벨과 운영진 배포 LoRA, AI-Hub 원본은 국내 G:에서 관리한다. [공식 규칙](https://hackathon.imaginationgroup.co.kr/kwater/#rules) · [제공 자료 조건](https://hackathon.imaginationgroup.co.kr/kwater/#core-data) · [최종 ZIP의 SHA-256과 로컬 키트 확인](results/public_submission_002.json)

두 번째 제출의 실행 진입점·헤드·환경 잠금·라이선스를 [제출 소스](submission/fusion/)에 보존했다. 체크아웃한 프로젝트 루트에서 `python tools/build_fusion_submission.py`로 같은 ZIP을 복원하고 접수본과 바이트 단위 SHA-256을 비교한다. 훈련된 헤드의 11개 출력 중 학습 양성이 없던 익수·침수, 밀폐공간, 화학물질 노출, 온도·기상 네 개의 확률은 이번 배포판에서 0이다. [헤드와 공통 임계값 검증](results/submission2_head.json)

비공개 원격 저장소 `git@github.com:nornen0202/KWater-SafetyAI.git`의 `main`에서 연구 코드·문서·도표·집계 결과를 관리한다. 대회가 GitHub를 포함한 외부 저장소로의 제공 데이터 업로드를 제한하므로, 참가자가 직접 받은 ZIP과 해제 자료, 이미지 특징·행별 예측은 G: 프로젝트에 둔다. 기존 공개 모델 실험의 다운로드 캐시는 G:의 별도 경로를 유지한다. [대회 이용 조건](https://hackathon.imaginationgroup.co.kr/kwater/#core-data) · [프로젝트 로컬 파일 원장](results/local_artifacts.json) · [초기 모델·런타임 의존성](results/public_probe_dependencies.json)

새 체크아웃에서는 공식 ZIP 세 개를 참가 계정으로 다운로드한 뒤 `results/package_manifest.json`의 SHA-256과 대조한다. ZIP을 `data/packages`에 보관하고 각각 `data/train`, `data/local-kit`, `data/baseline`에 압축을 푼다. 아래 명령은 프로젝트 루트에서 실행한다. CLIP 실험은 NumPy·Pillow·ONNX Runtime·tokenizers가 필요하다. 공개 모델 리비전과 플랫폼별 환경은 위 실행 기록을 따른다.

```powershell
$clipModel = 'G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32'
python -m kwater_lab.clip_probe fetch --model $clipModel
python -m kwater_lab.audit data/train/train.jsonl --image-root data/train --output results/train_audit_full.json
python -m kwater_lab.prefix_pilot --audit results/train_audit_full.json --labels data/train/train.jsonl --image-root data/train --model $clipModel --feature-cache data/prefix-pilot/features.npz --exposure data/prefix-pilot/exposure.json --output data/prefix-pilot/diagnostic_fpr20.json --summary results/prefix_pilot_fpr20.json
python -m kwater_lab.weight_pilot
python -m kwater_lab.loss_pilot
python -m kwater_lab.encoder_pilot --prepare --clip $clipModel
python -m kwater_lab.encoder_pilot --prepare --resolution 384 --clip $clipModel
python -m kwater_lab.encoder_fusion
python -m kwater_lab.fusion_timing --clip $clipModel
python -m kwater_lab.adapter_spectrum
python -m unittest discover -s tests -q
```

새 환경에서 수치 파일을 재생성하면 라이브러리·직렬화 방법에 따라 파일 해시가 달라질 수 있다. 새 실행에서 만든 `prefix_pilot_fpr20.json`을 `weight_pilot --reference`로 지정하면 같은 특징·같은 대조군에서 비교한다. 실제 Qwen 배율 실험은 [Windows GPU 환경](docs/official-baseline-windows.md)에서 `$runner -m kwater_lab.adapter_scale_pilot`을 실행한다(PowerShell 호출 연산자는 `&`). 실험은 같은 모델을 메모리에 한 번 적재하고 원본 가중치 파일을 유지한다.

연구 보고서의 그림은 `python tools/plot_research.py`로, PDF는 `python tools/render_research.py`로 다시 만든다. 그림에는 Matplotlib와 한글 글꼴이, PDF 출력에는 markdown-it-py와 Chromium 계열 브라우저가 필요하다. 다른 운영체제에서는 `--font`와 `--browser`로 경로를 지정할 수 있다.

로컬 파일 원장은 `python -m kwater_lab.local_artifacts`로 완료된 파일의 전체 해시를 작성한다. 전송 중인 파일과 내려받기 상태 로그는 제외한다. `--append-new`는 기존 자료의 존재·크기를 확인하고 새 파일을 추가한다. 새 AI-Hub 정식 ZIP은 수신 중 계산한 SHA-256과 완성된 ZIP의 항목 수·크기·파일명을 기록한 영수증을 사용한다. 이미 기록한 프롬프트·생성 결과가 바뀌면 `--refresh-derived`로 크기가 같은 수정까지 다시 해시한다. 전체 바이트를 다시 대조할 때는 `--check`로 실행한다. 대회 종료 후 자료 처리에는 주최 측 안내를 따른다.

현재 항목 38에서 실행한 [동결 패널·후보 헤드·외부 상태쌍 대조](docs/phase2-item38-measured-20261126.md)와 [대형 Qwen·SAM 실행 원장](docs/phase2-large-model-execution-20261126.md)은 로컬 실측을 기록한다.
