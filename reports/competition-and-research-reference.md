# K-water 현장 안전 AI에 적용할 종료 대회와 관련 연구

2026년 11월 25일 조사 기준. 사진에서 보이는 사실을 먼저 기록하고, 그 사실로 위험 유형을 판정한 뒤 예방조치와 법령을 연결하는 접근을 중심으로 정리했다. 각 표의 순위와 수치는 해당 대회·논문의 결과이며, 마지막 열은 K-water에서 검증할 제안이다.

## 1. 종료 대회에서 무엇을 가져올 것인가

원문은 중복 주소를 제거한 후보 URL 500여 개에서 종료 기록·상위권 성적·공개 방법론을 대조해 국내 5개, 해외 10개 부문을 선정했다. 해외 표에서는 AI City 2024·2025의 같은 트랙을 한 행에 묶었다.

### 국내 5개 부문

| 대회·트랙 | 상위 팀의 방법과 결과 | K-water 적용과 출처 |
|---|---|---|
| **한솔데코 시즌3 건설공사 사고 예방 및 대응책 생성, 2025** | AIDA가 1위 결과물을 공개했습니다. 2위 `안전에진심으로`는 대응책의 현장 적용 가능성을 지속적으로 확인했습니다. 3위 `5월 3일 결혼`은 안전문서의 표·그림 정보를 활용하는 Vision RAG와 VLM을 설명했습니다. | 사진에서 확인한 상태에 맞는 조치를 생성하고, 문장 유사도와 현장 실행 가능성을 각각 평가한다. [1위 결과물](https://dacon.io/en/competitions/official/236455/codeshare/12265) · [1위 인터뷰](https://dacon.io/en/competitions/open/236455/talkboard/413613) · [2위](https://dacon.io/competitions/official/236455/talkboard/413604) · [3위](https://dacon.io/forum/413611) |
| **Superb AI × B.D.A.I 비전 AI 해커톤, 2026** | 1등 **SEEN**은 작업자와 보호구를 YOLO로 탐지하고, 위험구역과의 관계로 우선순위를 계산한 뒤 VLM·LLM으로 상황과 조치를 설명했습니다. 주최 측 인터뷰에 나온 mAP50 0.843은 보호구 탐지 지표입니다. | 작업자·물체의 위치와 상태를 확인한 뒤 위험을 판정한다. 전선이 보이면 통행 방해와 전기적 손상의 근거를 구분한다. [주최 측 우승작 기술 인터뷰](https://blog-ko.superb-ai.com/vision-ai-hackathon-industrial-safety/) |
| **월간 데이콘 이미지 기반 질의응답, 2023** | Private 1위 **arist0n**은 BEiT-3 Large, 2위 **ATL**은 LLaVA, 3위 공개 풀이는 BLIP 다수결을 사용했습니다. | 11개 코드 분류기와 설명 생성 모델이 서로 다르게 판정한 사진에서 각각의 정확도를 비교한다. [1위 코드](https://dacon.io/en/competitions/official/236118/codeshare/8671) · [2위 팀 인터뷰](https://dacon.io/en/competitions/official/236118/talkboard/410420) · [상위 풀이 모음](https://dacon.io/competitions/official/236118/codeshare) |
| **반도체 소자 이상 탐지 AI 경진대회, 2024** | Private 1위 공개 풀이인 **태형이**의 PatchCore 앙상블은 여러 모델의 점수를 결합했습니다. 팀은 수상 후 재현 과정에서 설치 문제와 Ubuntu 성능 차이도 직접 보고했습니다. | 정상 참조 사진과의 거리로 오경보를 분석하고, 현장 변화와 Linux L4 실행 환경에서 결과가 유지되는지 확인한다. [1위 코드와 재현 기록](https://dacon.io/en/competitions/official/236224/codeshare/9868) |
| **금융 AI Challenge 금융 AI 모델 경쟁, 2025** | 카카오뱅크 **LostCow**는 Public 1위, Private 2위, 최종 우수상을 기록했습니다. 한국어 도메인 적응, BM25 문서 검색, 양자화와 추론 최적화를 함께 적용했습니다. | 한국어 조문의 대상·행위를 보존하는 BM25를 기준으로 의미 검색과 재정렬의 추가 이득을 측정한다. [대회 리더보드](https://dacon.io/competitions/official/236527/leaderboard) · [수상팀 기술 보고](https://tech.kakaobank.com/posts/2602-financial-security-ai-challenge-review/) |

### 해외 10개 부문

| 대회·트랙 | 상위 팀의 방법과 결과 | K-water 적용과 출처 |
|---|---|---|
| **AI City 2024·2025 교통안전 설명, 각 Track 2** | 2024년 **AliOpenTrek / CityLLaVA** 1위는 짧은 질의와 순차 질문을 사용했습니다. 2025년 **CHTTLIOT / TrafficInternVL** 1위는 보행자·차량 상호작용을 위한 구조화된 미세조정과 코드를 공개했습니다. | 전체 사진 한 장에서 사람·물체·위치·상호작용을 짧은 질문으로 나누고 시각 특징을 재사용한다. [2024 결과](https://www.aicitychallenge.org/2024-challenge-winners/) · [CityLLaVA 논문](https://arxiv.org/abs/2405.03194) · [2025 결과](https://www.aicitychallenge.org/2025-challenge-winners/) · [TrafficInternVL 코드](https://github.com/q104769424/2025AICITY_Track2_TrafficInternVL) |
| **AI City 2026 교통안전 이해·설명, Track 2** | **Latent Painter–UTE** 1위, **UIT–Kitchen** 2위입니다. 1위 팀은 장면에 관한 질문을 구조화된 사실로 확정하고 Qwen3-VL이 이를 받아 설명하게 했습니다. 공식 S2는 **60.0853**이었습니다. | 관찰 사실을 먼저 확정한 뒤 설명을 생성한다. 설명이 위험을 새로 만들어내는 경우와 이미 판정한 위험을 빠뜨리는 경우를 따로 센다. [공식 순위](https://www.aicitychallenge.org/2026-challenge-winners/) · [우승 논문](https://arxiv.org/abs/2609.18562) |
| **ECCV W-CODA 2024 희귀 도로상황 이해** | **llmforad** 1위, **OpenDriver** 2위, **NexusAD** 3위입니다. 평가는 전체 장면 이해, 특정 대상 이해, 상황별 대응 제안으로 구성됐고, 우승팀은 2단계 VLM을 사용했습니다. | 장면 관찰, 대상별 위험 판정, 대응 제안을 나눠 희귀 장면에서 어느 단계가 실패하는지 확인한다. [공식 순위·팀별 보고서](https://coda-dataset.github.io/w-coda2024/track1/) |
| **ICCV 2COOOL 2025 미지 위험 사고보고서 생성** | **Yokoi·Sasaki·Yamaguchi** 팀은 공개 리더보드 29팀 중 2위였습니다. 관찰과 세부 사건 판독을 나누고, 보고서 품질은 작성 방식을 가린 A/B 비교로도 평가했습니다. | 같은 사진의 두 한국어 답변을 눈가림 비교해 근거의 정확성과 조치의 실행 가능성을 각각 평가한다. [경진대회](https://www.kaggle.com/competitions/2coool) · [2위 팀 논문](https://arxiv.org/abs/2510.12190) |
| **AI City 2024 헬멧 위반 검출, Track 5** | **UIT**가 Co-DETR와 소수 클래스 보강으로 우승했습니다. | 희귀 사례를 보강할 때 사람과 보호구의 대응을 보존하고, 가림 때문에 착용 여부가 불확실한 사진에서 오경보를 센다. [공식 순위](https://www.aicitychallenge.org/2024-challenge-winners/) · [우승팀 논문](https://openaccess.thecvf.com/content/CVPR2024W/AICity/html/Vo_Robust_Motorcycle_Helmet_Detection_in_Real-World_Scenarios_Using_Co-DETR_and_CVPRW_2024_paper.html) |
| **WACV MaCVi 2023 해상 인명·물체 검출** | 시드니대 **Maritime-VSA** 1위, Fraunhofer **DetectoRS** 2위, **YOLOv7-Sea** 3위였습니다. 주최 측은 수영자와 구명장비가 작고 외형 변화가 커 어려웠다고 분석했습니다. | 물·사람·구명장비 검출에 물가 접근 가능성과 방호 상태를 결합한다. 물체만 보고 익수 경보를 낸 사례를 점검한다. [주최 측 결과·클래스별 분석](https://arxiv.org/html/2211.13508v2) |
| **LVIS 2021 장기꼬리 인스턴스 분할** | **FuXi-Fresher** 1위는 데이터 표집과 손실함수 양쪽에서 클래스 불균형을 다뤘습니다. | 희귀 위험과 함께 등장한 흔한 위험도 과대표집되는지 확인하고, 동시 출현을 반영한 표집을 단순 반복과 비교한다. [우승팀 기술 보고](https://arxiv.org/abs/2111.02668) |
| **CVPR VAND 2023 산업 이상 탐지** | **APRIL-GAN**은 제로샷 1위, 퓨샷 4위였습니다. CLIP 이미지·텍스트 특징을 비교하고 정상 참조 사진이 있는 조건에서는 특징 저장소를 사용했습니다. | 위험 설명과 정상 대조 설명을 함께 제시하고, 정상 참조 사진을 추가했을 때 실제 정상 사진의 오경보를 측정한다. [주최 측 과제](https://sites.google.com/view/vand-cvpr23/challenge) · [팀 기술 보고](https://arxiv.org/abs/2305.17382) |
| **COLIEE 2025 법령 조문 검색, Task 3** | **JNLP** 1위 F2 **0.8365**, **CAPTAIN** 차상위 **0.8301**이었습니다. 두 팀 모두 후보 조문 검색 후 관련성을 다시 판단했습니다. | 후보 조문을 찾은 뒤 대상·행위·조건이 제안 조치를 뒷받침하는지 다시 평가한다. [공식 순위](https://coliee.org/COLIEE2025/results/task3) · [주최 측 방법 비교](https://pmc.ncbi.nlm.nih.gov/articles/PMC13095985/) |

## 2. 대회 밖에서 찾은 논문과 공개 프로젝트

| 연구·프로젝트 | 연구가 다룬 문제와 결과 | K-water 적용과 출처 |
|---|---|---|
| **수력공사 현장 위험 인식 연구, 2026** | 수력공사 사진·텍스트에 Qwen2.5-VL을 LoRA와 지시 학습으로 적응시켰습니다. 저자들은 위험 판독에서 설명문이 사진보다 더 많은 핵심 정보를 제공하는 경우를 관찰했습니다. | 사진과 고정 질문만 제공하는 조건을 기본으로 삼고, 사고 경위나 정답 설명을 제공하는 조건의 성능은 별도로 기록한다. [원 논문](https://www.sciopen.com/article/10.26599/JIC.2026.9180117) |
| **건설 위험 식별·시정조치 생성 연구, 2026** | 현장 이미지 3,237장, 불안전 행동·상태 21종에 Qwen3-VL-8B를 LoRA 학습해 자체 평가에서 유형 평균 F1 **0.861**을 보고했습니다. 비슷한 위험 유형과 복잡한 장면은 여전히 어려웠습니다. | 위험별 조치를 연결해 학습하되, K-water의 11개 코드와 위험 미관측 사진을 포함한 조건에서 다시 측정한다. [원 논문](https://www.mdpi.com/2075-5309/16/22/4498) |
| **RCASR, 건설 사진과 안전규정 연결, 2026** | 48개 현장 346장을 현장 단위로 나눴습니다. 사진당 후보 3개에서 근거가 확인된 위반 재현율은 **0.434 → 0.625**, 사진당 오경보도 **1.28 → 1.56**으로 올랐습니다. 후보를 평균 약 2개로 줄인 조건에서는 오경보가 **1.00**으로 줄었지만 개구부 유형은 더 취약했습니다. | 경보 개수를 바꿀 때 위험 재현율과 사진당 오경보를 함께 보고, 개구부·추락의 미탐을 별도로 추적한다. [원 논문 및 후보 수별 오경보 표](https://www.mdpi.com/2075-5309/16/22/4502) |
| **SafeBuild-Bench / GEMS, KDD 2026** | 전문가 검토 이미지 3천여 장에서 3,314개 평가 항목을 구성했습니다. 현장·시점 변화와 중복을 고려하고, 모델이 혼동하면서 기존 표본과도 다른 사례를 선정했습니다. | 모델 간 불일치, 정상 사진 오경보, 두 번째 위험 누락을 우선 검수하되 비슷한 장면은 사건별로 묶는다. [논문](https://arxiv.org/abs/2608.00068) · [공개 코드](https://github.com/safebuild/gems) |
| **SafeSceneReason 및 ConstructionSite 10k** | SafeSceneReason은 객체·관계·안전규칙을 연결하고 사고조사 자료의 근거 범위를 표시했습니다. ConstructionSite 10k는 사진 1만 장에서 설명, 위반 여부 질의응답, 설명에 대응하는 시각 대상 찾기를 함께 다룹니다. | 관찰 대상·위치 관계·위험 근거·조치를 연결해 기록하고, 위험 코드는 맞았지만 설명이 틀린 사례를 찾는다. [SafeSceneReason 논문](https://arxiv.org/abs/2608.09230) · [프로젝트](https://github.com/KangKanng/safescenereason) · [ConstructionSite 10k](https://doi.org/10.1017/dce.2026.10044) |
| **국내 비계 점검 연구와 추락 위험 언어추론 연구** | 국내 연구는 비계 사진의 점검 결과를 KOSHA 규정과 연결했습니다. 별도 추락 연구는 개구부·비계·작업자 행동을 분리해 설명하고 그 설명이 위험 판단을 지지하는지 검증했습니다. | 추락 위험에서는 개구부·가장자리·난간·덮개·접근 가능성을 각각 확인하고 판정 근거를 해당 규정에 연결한다. [국내 KOSHA 연계 연구](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12489337) · [추락 위험 연구](https://www.mdpi.com/2075-5309/16/20/4131) |
| **iSafetyBench 및 InspecSafe-V1** | iSafetyBench는 산업 영상 1,100개에 정상 행동 98종과 위험 행동 67종을 담았습니다. InspecSafe는 터널·전력시설 등에서 RGB 이외에 가스·온도·습도 등 센서도 수집했습니다. | 사진에서 관찰한 출입구와 센서로 측정해야 하는 산소농도를 구분해, 보이지 않는 상태를 단정하는 오류를 검수한다. [iSafetyBench 프로젝트](https://isafetybench.github.io/) · [InspecSafe 논문](https://arxiv.org/abs/2601.21173) |
| **KoBLEX, ALCE, POPE, SugarCrepe** | KoBLEX는 한국어 법률 질문과 뒷받침 조문을 함께 제공하고, ALCE는 생성문의 인용 품질을 평가합니다. POPE와 SugarCrepe는 사진에 없는 물체나 틀린 관계를 모델이 받아들이는 문제를 진단합니다. | 법령은 조문 번호 일치와 조치에 대한 지지를 분리해 채점한다. 사진 설명은 위치 관계 하나만 바꾼 오답으로 검증한다. [KoBLEX](https://arxiv.org/abs/2509.01324) · [ALCE](https://arxiv.org/abs/2305.14627) · [POPE](https://arxiv.org/abs/2305.10355) · [SugarCrepe](https://arxiv.org/abs/2306.14610) |
