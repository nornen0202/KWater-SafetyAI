# RTX 3060, 대회 L4와 외부 노트북 활용

2026년 11월 25일 기준. 이 PC에서 GPU 모델과 VRAM은 `nvidia-smi`, CPU 및 RAM은 Windows CIM으로 확인했다. 채점기와 클라우드 성능 수치는 실제 측정과 계획값을 구분해서 기록한다.

| 항목 | 이 PC | 대회 제출기 |
|---|---|---|
| GPU | GeForce RTX 3060 데스크톱, 12,288MiB, Ampere | NVIDIA L4 24GB 1장 단독, Ada |
| CPU / RAM | i5-12400F, 물리 6코어, RAM 15.85GiB | CPU 6개, RAM 56GiB |
| OS / 실행 | Windows 개발 환경, 모델·캐시 G: | Ubuntu 24.04, Python 3.12, CUDA 12.8, PyTorch 2.11 |
| 용도 | 데이터 품질 확인, 특징 추출, 소형·양자화 모델과 어댑터의 반복 학습 | 제출 ZIP에서 인터넷 없이 설치·로딩·추론·결과 저장 |

L4는 VRAM이 약 두 배다. 가중치 외에도 사진의 시각 토큰, KV 캐시, 배치, 학습 활성값이 메모리를 쓴다. 예를 들어 70억 파라미터의 16비트 가중치만 약 14GB라서 3060은 양자화나 일부 CPU 배치가 필요하다. 4비트 가중치의 단순 산술은 약 3.5GB이고 실제 실행에는 양자화 메타데이터와 작업공간이 추가된다. 12GB PC에서는 작은 배치, 기울기 누적, 체크포인팅, LoRA/QLoRA로 후보를 줄이고 24GB 이상에서는 해상도·배치·모델 크기 후보를 넓힐 수 있다. 학습에 더 큰 GPU를 써도 모든 참가자의 최종 추론 예산은 같다. 사진당 생성 길이와 모델 로딩 시간에 따라 속도 차이가 달라진다. [NVIDIA RTX 3060 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/) · [NVIDIA L4 사양](https://www.nvidia.com/en-us/data-center/l4/)

공식 배포 Qwen3-VL-4B와 LoRA의 기존 3060 실측은 배치 1, 10장, 최초 모델 시작을 포함해 2,035.436초(33분 55초)였다. 시스템 전체 GPU 사용량은 시작 전 1,968MiB에서 최대 11,831MiB에 이르렀다. Windows의 소프트웨어 버전과 1회 로딩 비용이 포함된 수치라 L4 200장의 총시간은 별도로 재야 한다. [로컬 원본 실행 집계](../results/official_baseline_10.json)

## 채점 시간을 연구 예산으로 환산

Public 200장·30분은 설치·로딩·저장까지 평균 9초/장이다. Private 300장·60분은 12초/장이다. 고정비를 120초로 가정하면 각각 8.4초와 11.6초가 남는다. L4 리허설은 Public 쪽을 우선 맞춘다. 동일한 ZIP으로 1회 cold start와 200장 전체를 실행하고 설치·최초 로딩·이미지 전처리·생성·저장을 각각 잰다. 생성 토큰 수, P50/P95, 전체 시간, 최대 GPU/RAM, 실패 행도 기록한다. 100장 결과를 두 배로 환산한 값은 예측값으로 표시한다. 20GB 한도와 외부 네트워크 차단 조건은 모델 가중치·wheel을 제출물 안에서 읽는 방식으로 재현한다.

## 계정별로 사용할 수 있는 자원

| 서비스 | 확인된 제공 조건 | 이 프로젝트에서 맡길 일 |
|---|---|---|
| Google AI Pro의 Colab | 적격 국가 유료 구독에 월 200 Colab 컴퓨팅 단위(CCU). 한국어·영어 도움말의 해당 혜택 국가 목록에 대한민국이 없다. 만 18세 이상, 가족 요금제는 관리자, 체험 구독 제외. | 한국 결제 계정의 기본 예산은 0 CCU. 실제 계정 잔액에 표시된 경우 공개 데이터 고해상도 특징과 어댑터 후보를 비교 |
| 무료 Colab | 무료 GPU는 할당 상황과 사용 한도의 영향을 받는다. 무료 런타임은 가용성과 사용 패턴에 따라 최대 약 12시간 | 작은 독립 단위로 쪼갠 공개 데이터 전처리, 노트북 재현 |
| Kaggle Notebooks | 계정의 세션 시작 화면에 표시되는 GPU와 잔여 할당량에 따라 이용. 종류와 주간 한도는 바뀔 수 있다. | 공개 데이터만 포함한 비공개 노트북으로 seed 반복. GPU가 2장이어도 개별 GPU VRAM은 분리됨 |
| AWS SageMaker Studio Lab | 신규 고객 접수 종료. 기존 계정은 가용한 GPU를 하루 최대 4시간 사용 가능 | 기존 사용자에게만 짧은 재현 실험 |
| Google Cloud 일반 VM | Compute Engine 상시 무료 범위는 GPU·TPU 제외. Free Trial 계정에는 GPU 추가 제한 | 무료 학습 시간 산정에서 제외 |

Colab은 브라우저에서 Python을 실행하는 서비스다. Gemini 앱이나 NotebookLM의 콘텐츠 처리 한도와 별도로 Colab의 잔액을 사용한다. Google AI Pro에서 **200시간**을 보장한다는 뜻은 아니다. 예를 들어 런타임이 시간당 5 CCU를 표시한다면 200 CCU로 40시간 상당이라는 산술이 성립한다. 로그인된 Chrome에서 공개 예제의 ‘추가 연결 옵션 → 리소스 보기’를 열어 현재 사용 가능한 단위가 없고 런타임이 연결되지 않았음을 확인했다. 이 계정의 현재 잔액은 0 CCU다. Gemini 결제 계정과의 일치 여부는 확인되지 않았다. [한국 Google One 혜택](https://support.google.com/googleone/answer/14534406?hl=ko) · [Google의 2026년 9월 발표](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) · [Colab 자원·실행 FAQ](https://research.google.com/colaboratory/faq.html)

Kaggle에 공개되지 않은 대회 사진과 AI-Hub 원본을 올리지 않는다. Kaggle은 노트북 설정에서 해당 계정의 실제 가속기 및 잔여 할당량을 확인한 뒤 사용한다. 기존 Studio Lab 계정이 없다면 Colab과 Kaggle로 계획한다. [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) · [Kaggle의 변동 할당량 안내](https://www.kaggle.com/discussions/product-feedback/173129) · [AWS Studio Lab 이용 조건](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html) · [Google Cloud 무료 범위](https://docs.cloud.google.com/free/docs/free-cloud-features)

## 실험 배치

1. 국내 PC에서 AI-Hub 원본과 대회 자료의 사건 중복을 확인한다. 확정된 분할에서 11종 F1, 무위험 오경보, 희귀 위험 재현율을 함께 기록한다. 국내 AI 학습용으로 동의한 AI-Hub 원본과 그 특징은 이 경로에 둔다. [AI-Hub 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)
2. Colab에는 허용 범위를 확인한 FloodNet 공개 데이터 또는 여기서 추출한 특징만 사용한다. `tools/floodnet_cloud_pilot.ipynb`는 동봉한 `public-data/floodnet-pilot/pilot-features.npz`의 원래 학습·시험 구분을 유지해 중심점 모델과 100스텝 선형 분류기를 실행하고 환경·시간·최대 GPU 사용량을 저장한다. [FloodNet 배포처와 CDLA 라이선스](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
3. 기본 계획은 로컬과 무료 Colab·Kaggle을 쓴다. 계정에 실제 200 CCU가 표시될 경우에 한해 20을 환경 확인, 100을 GPU가 필요한 공개 자료 실험, 60을 유망 후보의 다른 시드·해상도 재검증, 20을 L4를 배정받았을 때의 공개 입력 추론 점검에 남길 수 있다. 가속기 소모율과 단계별 개선을 보고 다음 단계 사용량을 정한다. 세션 중단에 대비해 공개 실험의 체크포인트와 결과를 저장한다.
4. 공개 데이터에서 얻은 후보 가중치를 G:로 가져와 국내에서 공식 자료와 결합해 비교한다. 대회 제출 후보는 오프라인 환경과 실제 L4에서 다시 잰다. 외부 데이터 사용 내역은 대회 공고의 신고 조건에 맞춰 추적한다.
