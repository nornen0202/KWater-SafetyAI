# 서울 L4 후속 실험 원장

2026년 11월 28일. 로컬 GPU는 제미나이의 1,045장 전량 학습에 배정하고 서울 L4에서 모델 크기와 제출 패키지를 시험했다. 이번 실행의 승인 예산은 15,000원이다. [마스터 로드맵](comprehensive-audit-and-vlm-roadmap-20261128.md) · [집계 JSON](../results/seoul-l4-roadmap-experiments-20261128.json)

## 같은 32장으로 4B와 8B 학습

| 서울 L4에서 측정한 항목 | Qwen3-VL-4B-Instruct | Qwen3-VL-8B-Instruct |
|---|---:|---:|
| 고유 훈련 사진 / 현장 그룹 | 32 / 32 | 32 / 32 |
| 옵티마이저 업데이트 | 16 | 16 |
| 감독 토큰 / 결정 가중 토큰 | 4,110 / 623 | 4,110 / 623 |
| 16개 스텝 시간 합계 | 64.29초 | 102.47초 |
| 스텝 로그의 최대 GPU 할당 메모리 | 4.74 GiB | 8.76 GiB |
| 학습 가능한 매개변수 | 33,980,416 | 44,793,856 |
| 새 프로세스에서 어댑터 재로드 후 원생성 스키마 통과 | 2/2 | 2/2 |

두 모델을 각각 원본 가중치에서 NF4 QLoRA로 학습했다. 검증 Fold 4와 확인된 훈련측 교차 그룹, 이전 노출 표시를 제외하고 위험 22장·정상 10장을 골랐다. 재로드용 위험 1장·정상 1장은 별도 그룹에서 선정했다. 두 팔 모두 시드 42, 동일 순서와 프롬프트, 393,216화소, LoRA 순위 16·alpha 32, 학습률 0.00014, 두 예제마다 한 업데이트를 사용했다. 이미지·언어부 어댑터에서 모두 기울기를 확인했고, 모든 스텝의 감독·결정 토큰 수가 같았다. 모델 크기를 키우면 이 조건에서 업데이트 시간이 **1.59배**였다. 2장 검사는 저장·재로드·생성 경로를 확인하며, 정확도 우위를 판단할 표본은 Fold 4의 175장과 교차 그룹을 제외한 107장이다. [파일럿 구현](../kwater_lab/scale_training_pilot.py) · [분할별 비교 설계](comprehensive-audit-and-vlm-roadmap-20261128.md#paired-models)

## 재현과 예산

파일럿은 프로젝트 루트에서 `python -B -m kwater_lab.scale_training_pilot prepare --panel PANEL`로 훈련·재로드 명단을 만든다. 각 모델마다 별도 출력 폴더로 `train`과 `reload`를 별도 프로세스에서 실행하며 각각 `--panel PANEL --base BASE --out OUTPUT`을 넘긴다. 사진별 결과와 어댑터는 비공개 로컬 산출물로 보관하고 집계에는 모델 리비전과 구성·어댑터·프롬프트 해시를 남겼다. [원본 4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) · [원본 8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

원화 추정에는 계획 환율 1,500원/달러, 한국 VAT 10%, VM 전체 단가와 부팅 디스크·외부 IPv4·반출량을 사용한다. 팀 전용 서울 VM에서만 대회 자료를 처리하고 실험 후 디스크까지 회수한다. 공정한 실험의 우선순위는 제출 ZIP의 실제 시간, 동일 조건 학습 결과, 다음 후보의 제출 가능성 순으로 판단한다. [요금 가정](comprehensive-audit-and-vlm-roadmap-20261128.md#l4-budget) · [주최 측 클라우드 이용 FAQ](https://hackathon.imaginationgroup.co.kr/kwater/api/report)
