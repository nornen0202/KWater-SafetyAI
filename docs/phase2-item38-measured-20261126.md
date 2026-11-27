# 항목 38 현장 위험 판독 실험 기록

실행 ID는 `p38-20261126-49bb82ce`다. 이 문서의 수치는 국내 로컬에서 참가자 사진으로 얻은 값이다. Public 42.04와 16.12는 이전 실제 제출 기록이다. 당일 실전 제출은 관리자가 담당했다. 관리자가 18:42 접수한 A와 19:51 접수한 B2는 모두 Public 42.04로 기존 배포와 동점이었다. 20:46 팀 순위는 36팀 중 21위, 3위와 점수 차는 8.00이었다. [공식 이력 확인 시점](../data/phase2-runs/p38-20261126-49bb82ce/manager-submission-facts.json)

## 판정 범위와 비용

개발 등록부 920장·29그룹에서 이전 노출 사진을 제외하고 64장·23그룹을 먼저 고정했다. 이 중 16장·15그룹을 재판독 대상으로 미리 고정했다. 주최측 LoRA와 사례집이 기존 훈련 자료를 사용했으므로 이 모델에서 얻은 수치는 동작 진단이다. 별도 새 어댑터의 첫 비교는 외측 0번과 그룹이 겹치지 않는 대회 사진 32장, 외측 0번 평가 사진 11장을 쓴다. 개발 풀에 양성이 없는 기상 위험을 포함해 출력 계약은 11종으로 둔다. 봉인 125장은 평가하지 않았다. [등록부와 패널](../data/phase2-runs/p38-20261126-49bb82ce/plan.json) · [새 어댑터 분할](../data/phase2-runs/p38-20261126-49bb82ce/sft-split.json)

## 11월 26일 22:06 원판독 복구와 첫 신규 적합 순서

동결 64장 중 28장의 배포 Qwen 원문을 저장했다. 다음 사진의 생성에서 CUDA 메모리가 고갈됐고, 같은 설정으로 재개한 프로세스는 2개 샤드 중 첫 번째 적재 후 두 번째 파일을 열 때 Windows 페이징 파일 부족 오류 1455로 종료 단계에 들어갔다. 당시 해당 프로세스가 점유한 전용 메모리는 15.7 GB였고 RTX 3060 가용 메모리는 약 0.2 GiB였다. 두 오류의 발생 지점은 각각 사진 생성과 가중치 파일 열기였다. [기존 원판독 로그](../data/phase2-runs/p38-20261126-49bb82ce/official-oom-resume.err.log)

원본 BF16 샤드 2개의 텐서 713개에 대해 형태·자료형·바이트 범위가 파일 크기와 맞는지 확인했다. 6개 표본은 독립적인 파일 읽기와 바이트가 일치했다. 복구 실행은 파일 전체 매핑 없이 각 텐서를 읽고 CUDA 할당기 사전 예약을 생략한다. 마지막 비전 블록 8개는 CPU에 보관하고 순전파 때 CUDA에서 계산한다. 사용한 기반·어댑터·배율 1.0·SDPA·seed 42·최대 시각 토큰 384·출력 토큰 900은 같다. 남은 원문 저장과 원판독 결과 일치는 아직 검증 중이다. [실파일 검증](../data/phase2-runs/p38-20261126-49bb82ce/official-loader-byte-check.json) · [복구 원판독 로그](../data/phase2-runs/p38-20261126-49bb82ce/official-stream-1455.out.log)

22:06 실제 대기 큐의 첫 학습 작업을 Qwen3-VL-4B 새 QLoRA로 지정했다. 64장의 원문이 모두 저장되면, 기반 모델에서 시작해 그룹이 분리된 32장으로 누적 2장씩 16회 갱신하고 미학습 11장을 비교한다. 첫 배치의 두 loss, 전후 어댑터 파일별 지문·텐서별 변화량, 저장한 어댑터를 다시 불러와 같은 미학습 사진에서 얻은 생성문을 각각 기록한다. 그다음 고정 16장 재판독과 공식 예시 키트에서 저장된 결과를 재사용한다. 22:06 기준 신규 학습의 실제 갱신 수는 0회였다. [실행 순서와 프로세스](../data/phase2-runs/p38-20261126-49bb82ce/first-fit-live-queue.json)

## 공식 제출물 재현

기존 공개 사진 1장·한국어 질문 2개에서 무적응 Qwen3-VL-4B는 각 질문에서 위험 2개 중 1개를, Qwen3.5-4B Q4는 0개를 회수했다. 양쪽 JSON은 2/2 유효했다. 기존 6장은 이미 본 사진의 프롬프트 비교, 18장은 같은 사진에 배포 어댑터 배율만 바꾼 비교였다. 18장 F1은 배율 1.0에서 0.5244, 0.75에서 0.4875였다. 이번에는 새 LoRA의 32장 적응 데이터와 평가 11장을 그룹으로 분리했고 배포 어댑터 진단용 64장을 사전 고정했다. [1장 원생성 결과](../results/vlm_candidate_public.json) · [6장 및 선호자료의 노출 이력](preference-pilot.md) · [18장 배율 실측](../reports/official-lora-scale-pilot.md)

두 번째 제출의 실제 CLIP와 SigLIP2-384 결합 모델 및 고정 문구를 로컬에서 실행했다. 주최측 예시 10장 출력 검사를 통과했고 동결 패널 64장 전체가 유효했다. 64장 Macro F1은 0.23059, 무위험 오경보는 6/15, 중대성 4 완전 미탐은 15/26이었다. [64장 지표](../data/phase2-runs/p38-20261126-49bb82ce/fusion64-metrics.json) · [원출력](../data/phase2-runs/p38-20261126-49bb82ce/fusion-panel/submission.jsonl)

## 기존 원문의 코드 집합과 설명 2×2

기노출 28장·23그룹에서 배포 Qwen과 제출물 2의 코드 집합에 같은 고정 문구를 붙인 뒤, 각 사진의 저장 Qwen 원문에 같은 코드로 생성된 근거·예방문이 있는 항목만 실제 원문으로 바꿨다. 일치하지 않는 항목은 고정 문구를 유지하고 폴백으로 계수했다. 무위험 설명은 Qwen도 해당 사진을 무위험으로 판독한 경우에만 재사용했다. 사진이나 코드가 다른 문장은 재사용하지 않았다. 이 28장 중 9장이 지정코드 조건부 재판독 16장과 겹친다. 총 CPU 경과는 18.16초였다. [네 조건의 사진 원장](../data/phase2-runs/p38-20261126-49bb82ce/cached-writing-28/rows.json)

| 코드 집합 | 원문 재사용 위험 항목 | 원문 재사용 무위험 행 | 고정 / 원문 재사용의 고정 11종 F1 | 외측 훈련 사례의 1위 조문·코드 동시 일치, 고정 / 재사용 |
|---|---:|---:|---:|---:|
| 배포 Qwen | 20/20 | 10/10 | 0.28941 / 0.28941 | 12/20 / 11/20 |
| 제출물 2 | 14/33 | 7/10 | 0.15043 / 0.15043 | 11/33 / 10/33 |

두 코드 집합 모두 정답 조문이 존재하는 예측 항목은 15개였다. 본문과 사례가 합의할 때만 인용하면 Qwen 코드·고정 문구에서 10건 인용 중 5건, 원문 재사용에서 3건 인용 중 3건의 코드·조문 ID가 일치했다. 제출물 2 코드에서 이 수치는 각각 21건 중 5건과 15건 중 3건이었다. 원생성 28행의 파싱 실패는 0행이었다. 사실 지지·예방의 적절성·인용 조문의 법적 지지에 대한 사람 평정은 미측정이다. 재현 명령은 고정한 소스 경로에서 `python -B -m kwater_lab.phase2_cached_writing`이다. [집계·폴백·분모](../data/phase2-runs/p38-20261126-49bb82ce/cached-writing-28/metrics.json) · [출처별 원장](../data/phase2-runs/p38-20261126-49bb82ce/cached-writing-28/source-ledger.json)

## 동결 제출 후보 두 개의 CPU 재생

릴리스 담당이 동결한 `scope` 및 `mechanism` ZIP의 중앙 디렉터리에서 46개 공통 항목의 크기와 CRC가 원본과 일치했다. 각 ZIP의 `main.py`와 추가 모듈만 읽어 동결 규칙을 확인했다. 같은 Qwen 원출력을 순정과 두 제출 후보로 재생하고, 순정 CPU 결과가 GPU 판독 당시 저장한 순정 최종 행과 같은지도 사진마다 검증한다. 후보 A는 최종 조문을 점검하고 후보 B는 생성문에 따라 내부 위험 코드를 보정한 뒤 주최측 사례 검색을 다시 실행한다. 아래의 무힌트·실후보 재판독과는 별개 실험이다. [ZIP 내 소스 지문](../data/phase2-runs/p38-20261126-49bb82ce/release-replay/sources.json) · [표본별 누적 검산](../data/phase2-runs/p38-20261126-49bb82ce/release-replay/progress.json)

이전 18장·6그룹에서 B의 Macro F1은 순정 0.5244에서 0.6473으로 변했다. 세 사진에서 `other`를 회수하고 `slip_trip` 오탐 2개와 `fall_height` 오탐 1개를 제거했다. 중대성 4 완전 미탐은 7/12에서 4/12, 무위험 오경보는 모두 0/4였다. 인양물 정답 3개는 양쪽 모두 미회수였다. 신규 패널에서 처음 보존된 6장은 모두 이전 18장과 다른 그룹이었다. 6장에서는 두 후보 모두 조문과 코드가 순정과 일치했다. [노출 18장 원장](../data/phase2-runs/p38-20261126-49bb82ce/release-replay/prototype18) · [신규 판독 원장](../data/phase2-runs/p38-20261126-49bb82ce/release-replay/fresh64)

공식 배율 1.0의 로컬 첫 모델 로딩은 3,837.71초였다. 첫 사진은 이전 18장의 재현용 센티널이며 판독·포맷 2,439.86초, PyTorch 최대 할당 9,721 MiB였다. 뒤이은 신규 6장의 판독·포맷은 각각 192.72, 38.50, 20.55, 17.65, 20.35, 21.87초였다. 그 다음 사진은 NTFS가 원본 ID의 콜론을 받아들이지 않아 원문을 쓰는 단계에서 실패했다. 당시 완성한 포맷 파일을 보존하고, 사진 ID를 UTF-8 퍼센트 인코딩한 경로로 재개했다. 모든 후속 판정은 개발 OOF의 920개 ID 전용 등록부와 정답 파일을 사용한다. [재개 및 복구한 행 지문](../data/phase2-runs/p38-20261126-49bb82ce/official-restart.json) · [공식 실행 기록](../data/phase2-runs/p38-20261126-49bb82ce/official.log)

## 캐시 기반 동등 예산 후보 비교

이전 18장은 프로토타입 재현용이다. 대형 헤드의 종전 결과에서 Qwen이 놓친 11개 중 9개가 40개 신규 후보에 포함됐고, 신규 오탐은 31개였다. 사진당 1개를 제한해 정답을 알고 선택해도 회수 상한은 7개였다. 이번 비교는 외측 훈련 그룹에서 같은 선형 헤드를 적합하고 사진당 기존 코드 밖 1개를 제안했다. 각 후보가 실제 정답인지는 사후 집계했다. 정답 후보 수는 최종 Qwen 채택 수와 구분한다. [18장 종전 원장 재집계](../data/phase2-runs/p38-20261126-49bb82ce/anchor18.json) · [새 캐시 비교](../data/phase2-runs/p38-20261126-49bb82ce/cache-comparison.json)

| 특징 | 미탐 정답 후보 | 새 오탐 후보 |
|---|---:|---:|
| CLIP 224 | 4 | 14 |
| SigLIP2 Base 224 | 4 | 14 |
| Phase 1 SigLIP2 전체 뷰·투영 64차원 | 5 | 13 |
| Phase 1 DINOv2 전체 뷰·투영 64차원 | 6 | 12 |
| Phase 1 두 특징 결합 128차원 | 5 | 13 |

### 개발 920장, 29그룹의 외측 예측

모든 사진에 대해 동일한 표준화·선형 헤드·최대 1개 제안을 적용했다. 외측 5분할마다 다른 그룹으로 나눈 내측 3분할의 무위험 오경보 허용률 20%에 맞춰 임계값을 정했다. 외측의 실제 오경보 수와 호출 후보 수는 모델마다 달랐다. 아래 중대 미탐은 해당 사진의 중대성 4 코드가 단독 제안에 하나도 없는 경우다. 기존 Qwen과 합친 수치는 쌍체 재판독에서 따로 산출한다. [외측 예측 원장](../data/phase2-runs/p38-20261126-49bb82ce/nested-predictions.json) · [그룹 신뢰구간과 점수](../data/phase2-runs/p38-20261126-49bb82ce/nested-metrics.json)

| 특징 | Macro F1 | 고정 11종 F1 | 제안 횟수 | 무위험 오경보 / 308 | 중대성 4 완전 미탐 / 270 |
|---|---:|---:|---:|---:|---:|
| CLIP 224 | 0.1523 | 0.1384 | 366 | 60 | 209 |
| SigLIP2 Base 224 | 0.1534 | 0.1394 | 360 | 64 | 209 |
| 대형 SigLIP2 단독 | 0.2004 | 0.1822 | 348 | 50 | 187 |
| DINOv2 단독 | 0.2089 | 0.1899 | 348 | 49 | 200 |
| 대형 SigLIP2 + DINOv2 | 0.2495 | 0.2268 | 371 | 52 | 162 |

경량 SigLIP2 대비 DINOv2 단독의 쌍체 ΔF1 95% 구간은 +0.0137∼+0.1093, 두 특징 결합은 +0.0528∼+0.1595였다. 외측 훈련의 양성 그룹 수가 1 이하인 코드를 판정 단계에서 억제해도 이 표의 예측은 바뀌지 않았다. 원래 Phase 1의 학습 손실에 들어간 로그 사전확률은 별개 실험으로 다룬다. [대형 SigLIP2 제작사](https://huggingface.co/google/siglip2-so400m-patch14-384/tree/e8e487298228002f3d8a82e0cd5c8ea9c567f57f) · [DINOv2 제작사](https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c)

### 사전확률을 제거한 실제 손실 재적합

920장·29그룹의 원래 공간 헤드에서 0.5배 로그 사전확률을 제거해 외측 5분할과 각 내측 3분할로 20회 적합했다. 클래스별 임계값 정책은 기존 GPU 학습에서 Macro F1 0.2900, CPU 재적합에서 0.3125였다. 무위험 오경보는 284/308에서 151/308로, 중대성 4 완전 미탐은 111/270에서 100/270으로 줄었다. 29그룹 쌍체 ΔF1은 +0.0225이고 95% 구간은 +0.0029∼+0.0686이다. 고정 임계값 정책에서는 F1이 0.2658에서 0.2297로 내려갔다. 직접적인 손실 효과는 동일 CPU에서 0.5로 다시 적합한 대조로 확인한다. Qwen의 최종 채택·기각은 별도 진단이다. [분할별 재적합 결과](../data/phase2-runs/p38-20261126-49bb82ce/prior-refit/metrics.json) · [사진별 예측](../data/phase2-runs/p38-20261126-49bb82ce/prior-refit/predictions.json)

## 현장 상태쌍 출처

AI Hub 71407에서 국내 보관 훈련 TS/TL 대조 16개 자료묶음의 주석 12,800개를 점검했다. 기존 진단 현장·영상 배제 후 관측 상태를 확인할 수 있는 1,923개가 7개 영상·2개 현장에 남았다. 보호된 개구부와 방호 결함의 쌍을 만들 수 있던 현장은 한 곳이었다. 4개 서로 다른 영상에서 2쌍을 선택했다. 학습 조건당 동일 사진 수와 16회 갱신 예산으로 실제 상태, 동일 사진 내 쌍별 대응을 뒤집은 라벨, 대회 재표집을 비교한다. 기존 진단은 다른 현장·영상의 8쌍이다. 현장 한 곳의 상태 구분을 11종 위험의 훈련 정답으로 바꾸지 않았다. [제외·선택 원장](../data/phase2-runs/p38-20261126-49bb82ce/external/audit.json) · [대회 신고 초안](phase2-71407-declaration-draft.md)

AI-Hub 507 로프 작업에서는 공급자 훈련 5,743장과 검증 707장의 JPEG·JSON 식별자가 모두 일치했다. 양쪽의 원본 영상 17개와 현장 3개가 전부 겹쳤다. 영상마다 각 분할에서 1장씩 실제 JPEG 34장을 복호화해 주석 해상도와 확인했다. 작업자 상자는 훈련 5,419장·검증 668장, 로프 상자는 5,410장·665장에 있었지만, 안전고리 미착용 객체 및 미체결 상황은 양쪽 모두 0장이었다. 이 로프 장면의 객체 상자로 고리 체결 여부를 감독할 수 없다. [분할별 실사](../data/phase2-runs/p38-20261126-49bb82ce/external/507-rope/inventory.json) · [JPEG 표본 지문](../data/phase2-runs/p38-20261126-49bb82ce/external/507-rope/jpeg-ledger.json)

별도의 507 공통작업 검증 원본과 주석 18,598쌍에서 공급자 미체결 상황 UA-01은 4,413장·19영상·3현장이었다. 4,041장에 안전고리 미착용 객체 WO-06 상자가 있고 같은 장면의 별도 작업자 WO-01 상자는 0장이었다. 영상별 1장씩 19개 JPEG를 CRC·복호화·해상도로 검산했다. 원본 ZIP 16,233,979,149 B의 다운로드 SHA-256 기록이 있으며, 전체 멤버의 후속 CRC 검사는 이 시점에 미확인이다. 공통작업 공급자 훈련 원본 2개 ZIP은 로컬에 없었다. 체결 양성 상태와 사진에서의 연결점 가시성은 미확인이다. 507 학습 사용은 0장이다. 재현 명령은 `python -B -m kwater_lab.phase2_external_507 rope`와 `python -B -m kwater_lab.phase2_external_507 common-hook`이다. [공통작업 실사](../data/phase2-runs/p38-20261126-49bb82ce/external/507-common-hook/inventory.json) · [실제 JPEG와 상자](../data/phase2-runs/p38-20261126-49bb82ce/external/507-common-hook/jpeg-ledger.json) · [공급자 유형표](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=507)

## 깊이와 객체 영역의 첫 로컬 출력

고정한 첫 사진의 Grounding DINO Tiny INT8 CPU 실행은 6.66초였다. 작업자·하중·고리·난간·개구부·단부 여섯 명사 중 기준 0.35를 넘긴 박스가 없어 원경으로 복귀했다. 외측 훈련 분할에서 미리 연결한 교환 사진에서는 7.02초에 사람 박스 4개가 나왔다. 가장 높은 점수의 사람 박스 2,414픽셀과 같은 박스로 생성한 SAM 2.1 Hiera Tiny 마스크 1,534픽셀의 교집합은 1,474픽셀, 두 영역 간 IoU는 0.596이었다. SAM은 9.21초였다. 이 수치에는 수작업 정답 마스크가 없다. 170장의 기존 상대 깊이 캐시 지문을 확인하고 이 두 사진의 상대 깊이 지도를 각각 5.64초, 3.37초에 얻었다. [첫 실제 검출·분할 대조](../data/phase2-runs/p38-20261126-49bb82ce/regions/first-local.json) · [동결 배정과 기존 깊이 캐시 지문](../data/phase2-runs/p38-20261126-49bb82ce/regions/design.json)

실제 마스크·사각형·교환 마스크를 사용할 때 동일한 수검자 박스와 RGB 전체 화면을 보존한다. 거리·높이·안전 여부를 박스 겹침이나 깊이 점수로 확정하지 않는다. 사후 중대 미탐 47장은 평가 표본에 사용하지 않았다. [Grounding DINO Tiny 제작사](https://huggingface.co/IDEA-Research/grounding-dino-tiny) · [SAM 2.1 제작사](https://huggingface.co/facebook/sam2.1-hiera-tiny) · [로컬 INT8 내보내기 파일과 revision](../data/phase2-runs/p38-20261126-49bb82ce/regions/models/receipts.json)

64장의 Tiny 검출은 387.05초, 중앙값 5.09초였다. 20장에 사람 박스가 있었고, 44장은 여섯 종류 모두 미검출이었다. 64장 전체에 하중·고리·난간·개구부·단부 박스가 하나도 없었다. 16장 재판독 패널에서는 2장에 사람 마스크를 생성했고 14장은 모든 조건에서 동일한 768×384 원경 복제로 돌아갔다. 교환 사진에도 마스크가 있어 면적을 맞출 수 있던 수검 사진은 1장이었다. 상대 깊이 지도 16장은 49.85초였다. [패널별 처리 수·실패·시간·보조 화면 검산](../data/phase2-runs/p38-20261126-49bb82ce/regions/engineering.json)

로컬 공학 실험용 독립 ZIP 실측은 Tiny 검출 204,550,857바이트, SAM 61,986,003바이트, 깊이 99,073,086바이트, 공유 Linux CPU 런타임 18,550,785바이트다. ZIP_STORED를 사용했다. 각 모듈은 별도 헤더를 포함하고 Qwen 가중치와 생성 코드는 별개다. [파일별 크기·해시](../data/phase2-runs/p38-20261126-49bb82ce/regions/package.json)

개별 사진에 관한 관찰과 기록은 국내 복원본의 원문에 보존했다.

## 모델과 실행 조건

8B·9B 전용 LoRA, 9B·27B 저비트 설치, SAM 2.1 Large 및 SAM 3의 접근·실측·개발 분할은 별도 원장에 기록했다. [대형 모델의 첫 실제 분할과 실행 기록](phase2-large-model-execution-20261126.md)

Qwen3.5-4B의 제작사 공개 원가중치는 9.320 GB다. 9B는 19.306 GB, 27B는 55.563 GB이며 Flash-Next는 약 360 GB와 별도 라이선스를 사용한다. 27B Q4 GGUF와 비전 프로젝터의 공개 파일은 16,564,259,744 B와 931,146,432 B다. 용량만으로 L4의 오프라인 커널·시간 제한을 확인할 수 없어 첫 적응 비교는 4B로 지정했다. Qwen 단독 기준과 같은 사진·헤드·제안 수의 캐시 비교를 우선했다. PE-Core-B16-224의 공식 .pt는 1,790,786,632 B, Qwen3-VL-Embedding-2B의 공식 safetensors는 4,255,140,312 B다. 둘 다 Apache 2.0이다. 캐시로 완료한 제안기 대조와 현재 공식 판독을 우선해 이 두 모델의 국내 사진 처리 시간과 효용은 미측정이다. [Meta 원본](https://huggingface.co/facebook/PE-Core-B16-224/tree/a16450b46fef32363459920c2685a1b4ef13dcd9) · [Qwen 검색 모델 원본](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/tree/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda). [고정 revision 및 파일 명세](../results/phase2_public_model_sources.json) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [ms-swift 지원표](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md)

## 재현

작업 브랜치는 `nornen0202/kwater-phase2-model-comparison`이다. 대회 사진·아카이브·원시 출력은 G:의 원래 프로젝트에 있다. 아래 명령은 해당 브랜치의 루트에서 실행한다. 중간 결과와 장비·예산은 실행 원장에서 확인할 수 있다.

```powershell
./tools/run_phase2_study.ps1 -Stage Freeze
./tools/run_phase2_study.ps1 -Stage Cache
./tools/run_phase2_study.ps1 -Stage Contracts
./tools/run_phase2_study.ps1 -Stage Heads
./tools/run_phase2_study.ps1 -Stage Nested
./tools/run_phase2_study.ps1 -Stage Prior
./tools/run_phase2_study.ps1 -Stage External
$trial = 'G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce'
python -B tools/fetch_phase2_regions.py $trial
python -B -m kwater_lab.phase2_regions detect --limit 16
python -B -m kwater_lab.phase2_regions segment --limit 16
python -B -m kwater_lab.phase2_regions depth --limit 16
python -B -m kwater_lab.phase2_regions detect --limit 64 --recipients-only
python -B -m kwater_lab.phase2_region_review prepare
python -B -m kwater_lab.phase2_region_review engineering
python -B -m tools.phase2_region_package
./tools/run_phase2_study.ps1 -Stage Official
python -B -m kwater_lab.phase2_release --watch
./tools/setup_phase2_training.ps1
./tools/run_phase2_study.ps1 -Stage Train3VL
./tools/run_phase2_study.ps1 -Stage Train35
python -B -m kwater_lab.phase2_shared 3vl
python -B -m kwater_lab.phase2_shared 35
./tools/run_phase2_study.ps1 -Stage Static
./tools/run_phase2_study.ps1 -Stage Score
python -B -m kwater_lab.phase2_region_review score
```

같은 CPU에서 양쪽 사전확률 조건을 각각 20회 재적합해 개발 920장·29그룹을 비교했다. 클래스별 정책의 macro F1은 prior 0에서 0.02253 증가했다(쌍체 95% 구간 0.00291–0.06861). 고정 임계값에서는 0.03613 감소했다(−0.04726–−0.02455). 이 결과는 후보 헤드의 독립 그룹 대조에 해당한다. [같은 장치의 쌍체 원장](../data/phase2-runs/p38-20261126-49bb82ce/prior-refit-cpu-reference/cpu-paired.json) · [대형 모델 실측](phase2-large-model-execution-20261126.md)

## 같은 230회 제안 예산

개발 920장의 기존 외측 그룹 분리 점수에서 각 fold의 점수만으로 상위 사진을 정했다. 45·48·45·44·48장에 각 1개 코드만 제안해 총 230회를 맞췄다. 아래는 독립 그룹 제안 원장의 단독 성능이다. Qwen이 기각하거나 이미 낸 코드는 후속 쌍체 판독에서 집계한다.

| 인코더 | 정답 제안 | 오탐 제안 | macro F1 | 무위험 오경보 / 308 | 중대성 4 완전미탐 / 270 |
|---|---:|---:|---:|---:|---:|
| CLIP B/32 | 149 | 81 | 0.1159 | 32 | 226 |
| 경량 SigLIP2 Base 224 | 145 | 85 | 0.1292 | 32 | 225 |
| 대형 SigLIP2 단독 | 162 | 68 | 0.1634 | 26 | 209 |
| DINOv2 단독 | 165 | 65 | 0.1831 | 23 | 217 |
| 대형 SigLIP2 + DINOv2 | 172 | 58 | 0.1958 | 20 | 193 |

[예산·그룹 쌍체 구간](../data/phase2-runs/p38-20261126-49bb82ce/matched-budget-metrics.json) · [사진별 후보·순위·제안 여부](../data/phase2-runs/p38-20261126-49bb82ce/matched-budget-predictions.json)


앞의 230개 제안을 외측 fold 안의 순위 백분율로 정렬했다. 오탐을 40개로 맞춘 사후 진단에서 정답 제안은 CLIP 89개 · 경량 SigLIP2 91개 · 대형 SigLIP2 113개 · DINOv2 126개 · 대형 결합 136개였다. 10개 오탐에서는 대형 SigLIP2가 70개, DINOv2가 59개, 결합이 39개로 순서가 달랐다. 이 정렬 중단 지점에 개발 정답을 사용했으므로 실제 배포 임계값을 고르는 데 사용하지 않는다. [10·20·40개 오탐의 전체 원장](../data/phase2-runs/p38-20261126-49bb82ce/matched-budget-metrics.json)

## 제안기 단독 호출 층과 공식 원판독의 복구

같은 230회 예산에서 DINOv2와 결합 모델이 같은 코드를 제안한 143장 중 115장이 정답이었다. 서로 다른 코드를 제안한 10장에서는 양쪽 각 7장, 두 코드 모두 정답인 사진은 4장이었다. DINOv2만 호출한 77장 중 43장, 결합만 호출한 77장 중 50장이 정답이었다. 둘 다 호출하지 않은 613장에도 위험 정답이 있는 사진이 341장이었다. [그룹별 제안과 실제 재판독 교차 원장](../data/phase2-runs/p38-20261126-49bb82ce/proposer-strata-metrics.json)

동결 64장에는 DINOv2만 호출한 6장과 결합만 호출한 4장이 있다. 두 층의 10장·8그룹을 빠짐없이 고정했다. 표본 선택 입력은 패널 ID와 외측 fold의 제안 호출 여부다. 층별 코드 빈도를 동일하게 유지하면서 실제와 교환 후보가 다른 사진 수를 최대로 했다. 이 제약에서 DINOv2 2장, 결합 2장은 같은 코드가 된다. 같은 원판독·사진·후보·프롬프트·144 토큰의 판독을 재사용한다. 원출력에 코드가 3개면 제안만 남긴다. 결과를 층별로 보고하며 16장 주 비교와 합산하지 않는다. [고정 사진·그룹·양팔 프롬프트](../data/phase2-runs/p38-20261126-49bb82ce/exclusive-review/design.json)

개별 사진에 관한 관찰과 기록은 국내 복원본의 원문에 보존했다.

교환 제공자 10장은 모두 수신자 외측 훈련 그룹에서 적합한 같은 헤드의 11종 1위 제안을 실제 재계산해 확인했다. 사진은 모두 고유하고 원래 64장과 기존 18장을 제공자 풀에서 제외했다. [교환 사진·외측 fold·학습 목록 지문](../data/phase2-runs/p38-20261126-49bb82ce/exclusive-review/training-donors.json) · [두 구간의 호출 및 쌍체 지표](../data/phase2-runs/p38-20261126-49bb82ce/exclusive-review/metrics.json)

같은 28장의 제출물 2(CLIP+SigLIP 고정 문구)는 Macro F1 0.2758(11종 고정 0.1504), 무위험 오경보 2/5, 중대성 4 완전미탐 5/11이었다. 제출물 2로 교체하면 16장의 코드가 달라져 정답 5개를 새로 얻고 기존 정답 4개를 잃었으며 오탐 14개가 추가되고 2개가 제거됐다. 새로 얻은 정답은 추락 1·기타 2·미끄러짐 2개였다. 공식 Qwen 28장 모두 원문 파싱이 유효했고 사진당 처리 시간 p50은 21.21초, p95는 144.60초였다. 이 값은 국내 PC의 사진별 판독·포맷 시간으로 프로세스 시작 및 모델 로딩 시간은 따로 기록했다. [두 배포물의 같은 사진 쌍체 원장](../data/phase2-runs/p38-20261126-49bb82ce/deployed-paired-rows.json) · [비용과 누적 표본 수](../data/phase2-runs/p38-20261126-49bb82ce/review-progress.json)

아래 명령은 항목 38 소스 사본에서 실행한다. 교환 제공자 분류 헤드의 재적합에는 기존 개발 사진 임베딩만 사용한다.

```powershell
$run = 'G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce'
$source = Get-Content "$run/source-current.json" -Raw | ConvertFrom-Json
Set-Location (Join-Path $run $source.directory)
python -B -m kwater_lab.phase2_exclusive_review freeze
python -B -m kwater_lab.phase2_exclusive_review donors
python -B -m kwater_lab.phase2_exclusive_review replay
python -B -m kwater_lab.phase2_lineage
python -B -m kwater_lab.phase2_agreement
python -B -m kwater_lab.phase2_agreement --legacy
python -B -m kwater_lab.phase2_verifier prepare
```

[실행 소스와 SHA-256](../data/phase2-runs/p38-20261126-49bb82ce/source-current.json)

## 정식 제출과 개발 사진의 분류 진단

인양·양중물 충돌만 보면 개발 원판독 첫 28장에 정답은 1장·1그룹, Qwen 예측은 0장으로 TP 0·FP 0·FN 1이었다. 중대성 4 인양 정답은 0장이어서 이 표본에서는 해당 유형의 중대 정답 회수율을 계산할 수 없다. 관리자가 따로 확인한 학습측 679장에서 중대 인양 32건 중 결합 헤드의 1·2·3위에 나온 건수는 각각 5·10·12건이었다. 인양이 상위 3위에 등장한 335장에는 실제 인양 위험 41장, 무위험 사진 133장이 포함됐다. 합의가 없을 때 2~3위까지 자동으로 추가하는 규칙은 기존 28장에서 정답 증분 0개·새 오탐 2개였고 과거 18장에서도 무위험 경보가 2/4에서 3/4로 늘어 후속 선택에서 제외했다. [현 원판독·재판독의 인양 분모](../data/phase2-runs/p38-20261126-49bb82ce/review-progress.json) · [관리자 학습측 순위 대조](../data/phase2-runs/p38-20261126-49bb82ce/manager-severe-rank-controls-train679.json) · [기노출 사진에 적용한 상위 3위 대조](../data/phase2-runs/p38-20261126-49bb82ce/manager-top3-spatial-replay-exposed.json)

11월 26일 관리자가 접수한 A 범위 수정본과 B2 원본 인용본은 모두 Public 42.04로 기존 배포 모델과 동점이었다. 접수 시각은 각각 18:42와 19:51이며, 20:46 확인 순위는 36팀 중 21위였다. 아래 개발 사진의 후보 추가와 새 LoRA 학습은 별개 실험이다. [관리자 접수·점수 기록](../data/phase2-runs/p38-20261126-49bb82ce/manager-submission-facts.json)

기존 28장에는 Qwen 미탐 코드가 16개 있었다. 관리자 집계에서 고정 호출의 DINO 또는 결합 1위가 이 중 3개를 포함했다. 호출을 모두 열면 정답 6개와 오탐 20개가 더 생겼고 나머지 미탐 7개는 두 제안기의 1위에 없었다. 다음 표는 각 사진에서 **결합 1위 하나**만 다루고, 원판독에 없고 빈 슬롯이 있을 때 prior=0 유형별 정책의 판독과 일치하는지 확인한 별도 진단이다. 고정 230회 호출 게이트는 적용하지 않았다.

| 기존 28장에 적용한 규칙 | 추가 정답 | 추가 오탐 | 중대성 4 완전미탐 | 무위험 경보 | 고정 11종 Macro F1 |
|---|---:|---:|---:|---:|---:|
| 배포 Qwen 원판독 | 0 | 0 | 5/11 | 1/5 | 0.28941 |
| 결합 1위 추가 | 6 | 11 | 1/11 | 5/5 | 0.33963 |
| 결합 1위와 유형별 정책이 일치할 때 추가 | 6 | 2 | 1/11 | 1/5 | 0.36149 |

사전등록 지문과 두 예측 캐시의 SHA-256을 검사한 독립 CPU 재생에서 위 수치를 확인했다. 남은 36장은 사진 단위로 구분했고 기존 28장과 22개 그룹이 겹친다. 원판독 저장 시점마다 두 집합의 사진 수, 실제 공유 그룹, 유형별 TP·FP·FN과 개별 코드 이동을 따로 기록한다. 후보 코드에 대한 사진 근거·예방문과 사람 평정은 이 표의 측정 대상에서 제외했다. [사전등록](../data/phase2-runs/p38-20261126-49bb82ce/manager-agreement-prereg.json) · [독립 재계산 지표](../data/phase2-runs/p38-20261126-49bb82ce/agreement-independent/metrics.json) · [사진별 원장](../data/phase2-runs/p38-20261126-49bb82ce/agreement-independent/rows.json)

기존 탐색에 사용한 18장·6그룹에서도 같은 합의 규칙을 다시 적용했다. 이 18장은 64장과 사진·그룹이 겹치지 않는다. 공식 배율 1.0에서 정답 5개를 회복하고 오탐 2개가 생겼다. 고정 11종 Macro F1은 0.19068에서 0.21435, 중대 완전미탐은 7/12에서 3/12로 바뀌었다. 무위험 사진의 경보는 0/4에서 2/4로 늘고 사진 단위 정확 일치는 9장에서 7장으로 줄었다. 배율 0.75에서도 같은 두 무위험 사진에 경보가 생겼다. 이 결과로 후보를 바로 추가하는 정책의 승격을 보류했다. 원래 64장의 합의 발동 후보 전부에 대해 사진 근거를 재판독하고 기각·불명·실패를 추적한다. [배율별 독립 재생 지표](../data/phase2-runs/p38-20261126-49bb82ce/agreement-independent/legacy-metrics.json) · [원래 18장의 선택·노출 이력](../data/phase2-runs/p38-20261126-49bb82ce/manager-agreement-legacy18.json)

## 학습 분할과 평가 사진의 중복

국내 공식 학습 사진으로 후보를 수락하거나 기각하는 검증기 훈련 자료도 만들었다. 170장 중 과거 18장과 프롬프트 탐색 6장을 제외한 146장에서 고정한 16쌍이다. 각 쌍은 동일 그룹·동일 후보 코드의 실제 위험 사진과 명시적 무위험 사진으로 구성했다. 후보 7종, 6그룹, 사진 32장이며 64장과 사진 해시·그룹 중복이 없다. 입력은 원본 사진과 후보 코드이고, 공식 관찰 근거는 감독용 출력에만 들어간다. 수락 16건의 예방문과 기각 16건의 빈 예방 필드를 검증했다. 이는 훈련 자료 측정으로, 검증기 실제 학습과 무적응·적응 간 오수락 비교는 별도로 기록한다. [훈련 자료와 SHA-256](../data/phase2-runs/p38-20261126-49bb82ce/verifier-16pairs/audit.json)

| 적합 사진 | 해당 모델 평가 사진 | 동일 사진·그룹 | 학습 양성 유형 | 학습 양성 사진이 없는 유형 |
|---|---:|---|---:|---|
| 32장 | 11장 | 0장·0그룹 | 10/11 | 온열·한랭·기상 |
| 170장 | 64장 | 0장·0그룹 | 7/11 | 익수·침수, 밀폐공간, 화학물질, 온열·한랭·기상 |
| 727장 | 175장 | 0장·0그룹 | 9/11 | 밀폐공간, 온열·한랭·기상 |

32장으로 적합한 모델에서는 64장 중 38장이 학습 그룹 13개에 속한다. 외측 175장에는 같은 학습 사진이 12장이고, 학습 그룹 4개에 속한 사진은 173장이다. 727장으로 적합하면 64장 중 48장·17그룹이 겹친다. 따라서 각 외측 모델은 원기반에서 새 어댑터를 생성하며, 적용 전 부모 어댑터가 있는지와 누적 학습 ID·그룹을 저장한다. 위 표는 분할 실사 결과다. 실제 모델별 학습 완료 여부는 optimizer·재로딩 원장에서 확인한다. [9가지 분할 교차표와 11종별 양성 사진·그룹 수](../data/phase2-runs/p38-20261126-49bb82ce/development-ancestry-audit.json)

원문 보관 경로: `data/local-reports/ubuntu-handoff-originals/docs/phase2-item38-measured-20261126.md`
