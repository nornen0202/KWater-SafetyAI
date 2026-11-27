# 대형 Qwen과 분할기의 실행 원장

실행 ID는 `p38-20261126-49bb82ce`다. 대형 비교에 쓰는 대회 학습·평가 입력은 국내 G:의 개발 920개 허용 명단으로 제한한다. 운영사는 대회 규칙·개별 라이선스에 맞으면 기반 모델을 선택할 수 있다고 명시한다. 로컬 검사 키트는 채점 시 외부 인터넷과 localhost가 차단된다고 명시한다. 대회 사진을 외부 추론 서비스로 보내지 않는다. [운영사 규정](https://hackathon.imaginationgroup.co.kr/kwater/api/report) · [공식 로컬 키트](../data/local-kit/README.txt) · [분할과 제외 기록](../data/phase2-runs/p38-20261126-49bb82ce/plan.json)

| 모델 | 가중치 또는 변환본 | 로컬 실행 단계 |
|---|---|---|
| Qwen3-VL-8B-Instruct | Apache 2.0, BF16 17,534,339,512 B | 공식 원가중치 부분 수신. RTX 3060에서 BF16 소스로 NF4를 만든 후 이 기반 전용 LoRA를 적합한다. |
| Qwen3.5-9B | Apache 2.0, BF16 19,306,310,880 B | 공식 원가중치 부분 수신. 같은 그룹·예산으로 무적응과 전용 LoRA를 비교한다. |
| Qwen3.5-9B Q4_K_M | 언어 모델 5,680,522,464 B + BF16 비전 프로젝터 921,705,024 B | 저비트 파일만 쓰는 독립 설치·실행 경로. 합계 6,602,227,488 B. |
| Qwen3.8-27B UD-Q4_K_M | 언어 모델 16,564,259,744 B + BF16 비전 프로젝터 931,146,432 B | 합계 17,495,406,176 B. 십진 20 GB 한도에서 실행기·설정·어댑터·ZIP 헤더를 넣기 전 남은 용량은 2,504,593,824 B. |
| SAM 2.1 Hiera Large | Apache 2.0, 공식 safetensors 897,897,416 B. 별도 INT8 정지사진 인코더·디코더 | 국내 CPU에서 공식 공개 예시와 Grounding DINO Base 고정 박스의 첫 마스크 저장. |
| SAM 3 / SAM 3.1 | Meta 전용 약관. SAM 3 safetensors 3,439,938,512 B, SAM 3.1 multiplex 3,502,755,717 B | 18:37:56과 18:38:03 표준 HF 인증에서 3.0·3.1 모두 HTTP 200. 3.1 원본부터 공식 206으로 순차 전송 중. 두 버전의 단일 프레임 경로를 준비했다. |

Qwen3-VL-8B의 제작사 FP8은 10,590,299,512 B다. RTX 3060의 학습 실험은 BF16 원본으로 NF4를 구성할 때의 커널과 가중치 사용을 별도로 측정한다. 8B·9B에는 기존 4B 어댑터를 적용하지 않는다. [8B 원본](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) · [8B FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8/tree/9cdc6310a8cb770ce18efaf4e9935334512aee45) · [9B 원본](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) · [9B 변환 제작사](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5) · [27B 변환 제작사](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9)

## 같은 박스에서의 첫 분할

개별 사진에 관한 관찰과 기록은 국내 복원본의 원문에 보존했다.

개별 사진에 관한 관찰과 기록은 국내 복원본의 원문에 보존했다.

SAM 3과 3.1 직접 검출에는 사람과 위험 물체를 나타내는 같은 영문 명사 6개를 사용한다. 11월 26일 18:21 표준 HF 계정에서 두 파일이 심사 대기였고 18:38까지 모두 접근 승인을 받았다. 3.1 원본의 공식 전송 응답은 HTTP 206이었다. 첫 105,906,176 B를 보존한 채 해당 오프셋부터 같은 리비전의 후속 Range 요청을 다시 검증했다. Meta의 공개 구현에서 3.0 이미지 전용 경로와 같은 3.0 체크포인트의 단일 프레임 predictor, 3.1 multiplex 단일 프레임 predictor를 대조한다. 사진별로 새 세션을 열고 프레임 0만 처리한 뒤 세션을 닫는다. 공식 Python 경로에 필요한 3.0 원본 `sam3.pt`는 3,450,062,241 B이고, 두 3.0 조건이 이 파일을 공유한다. 3.1은 `sam3.1_multiplex.pt` 3,502,755,717 B를 쓴다. 모델 고유의 내부 필터와 공통 출력 임계값을 기록한다. 역할별 검출·쌍 검출과 위험 코드·설명 평가는 별도로 원장에 남긴다. 가중치 수신 후 실제 사진 처리와 시간을 측정한다. 공식 환경 하한은 Python 3.12, PyTorch 2.7, CUDA 12.6이다. [Meta 단일 사진 벤치](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/scripts/bench_compile.py) · [두 버전의 모델 빌더](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model_builder.py) · [SAM 3 공식 파일](https://huggingface.co/facebook/sam3/tree/3c879f39826c281e95690f02c7821c4de09afae7) · [SAM 3.1 공식 파일](https://huggingface.co/facebook/sam3.1/tree/daa63191845a41281374e725f4c9e51c7a824460)

16장 전량 인코딩과 선별된 6회 디코딩을 다시 실행했다. Tiny 61.77초(사진당 p50 2.97초, p95 7.34초), Large 167.30초(p50 8.90초, p95 16.77초)였다. 같은 박스 6장만 호출한 첫 실행은 Tiny 23.32초, Large 123.09초였다. 각 모델이 다시 생성한 6개 마스크는 앞선 파일과 SHA-256이 일치했다. 측정 시점별 캐시·I/O가 달라 차이 전체를 호출 정책의 인과 효과로 환산할 수 없다. [16장 호출별 시간](../data/phase2-runs/p38-20261126-49bb82ce/regions/base-sam-size/always-encoder-metrics.json) · [조건부 마스크·시간](../data/phase2-runs/p38-20261126-49bb82ce/regions/base-sam-size/engineering.json)

## 적응과 배포 검증의 분모

첫 32장·16회 업데이트는 메모리, 최적화 및 재로딩을 확인하는 단계다. 확대 시 동결 64장·23그룹과 겹치지 않는 개발 사진 전량은 170장·6그룹이다. 1회 순회·2장 누적의 85회 업데이트로 유망한 대형 기반 1개와 같은 세대 4B를 새로 적합한다. 사진마다 감독하는 토큰 수도 맞춘다. 고정 64장 중 첫 평가 그룹 밖의 53장 결과를 별도 집계한다. 다음 확인은 외측 1번을 지정했다. 나머지 네 fold의 727장·23그룹으로 학습하고 외측 1번의 193장 가운데 과거 노출·패널 사진을 제외한 175장·5그룹에서 평가한다. 대상 모델이 해당 그룹을 학습한 예측은 이 확인에 포함하지 않는다. [170장 분할](../data/phase2-runs/p38-20261126-49bb82ce/large-models/expansion-split.json) · [외측 1번 분할](../data/phase2-runs/p38-20261126-49bb82ce/large-models/expansion-outer1.json)

정식 L4 24 GB의 오프라인 cold 200장·300장 실행은 미측정이다. 국내 Windows 장치의 모델 로딩·마스크 시간은 위의 실제 관측값이다. Linux CPython 3.12·주최측 설치 목록에 대해 고정한 4B 런타임 추가 휠은 실제 다운로드 파일 합계 78,241,543 B다. Qwen3-VL-4B와 Qwen3.5-4B의 가중치·설정·추가 휠 합계는 각각 8,965,532,756 B와 9,421,147,442 B다. 이 산출 시점에는 새 어댑터와 포장 헤더가 들어 있지 않다. [Linux 휠·패키지 크기와 해시](../data/phase2-runs/p38-20261126-49bb82ce/static-package.json) · [대형 모델 용량 감사](large-vlm-sam3-fit-audit-20261126.md)

외측 1의 모델 계열 선택에 쓸 3개 내측 분할은 외측 훈련 727장 안에서만 구성했다. 각 내측 모델은 32장·16회 업데이트로 기술적 안정성을 확인하고 미사용 그룹 11장을 평가한다. 세 내측 평가의 그룹 수는 각각 7·7·9다. 외측 1의 학습·검증 그룹 중복은 0이며, 64장 평가에서 외측 1의 결과를 다시 찾아 설정을 조정하지 않는다. [분할 검증](../data/phase2-runs/p38-20261126-49bb82ce/large-models/verified-expansion.json) · [내측 분할 재현 소스](../data/phase2-runs/p38-20261126-49bb82ce/source-86c1ba9e5ece/kwater_lab/phase2_scaleup.py)

G: 원본에서 실행할 때는 아래의 연구 소스 경로에서 명령을 실행한다. 소스 사본은 Git 커밋의 Python 모듈·도구와 해당 시험 파일을 담고 있다. [소스 사본의 해시](../data/phase2-runs/p38-20261126-49bb82ce/source-86c1ba9e5ece.json)

```powershell
Set-Location G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce/source-86c1ba9e5ece
python -B -m kwater_lab.phase2_scaleup verify
python -B -m kwater_lab.phase2_matched_budget
```

## 같은 면적의 교환 마스크

16장 중 Grounding DINO Base가 비교용 박스를 만든 6장에서 SAM2.1 Tiny와 Large 마스크를 얻었다. 최초의 미리 정한 사진 교환은 수신 사진과 제공 사진에 유효 마스크가 함께 있는 2장만 비교 가능했다. 추가 대조는 원래 훈련측 제공자 16장 안에서 수신 fold 밖의 고유 제공자를 정해 6장 모두를 확보했다. 사진·모델별 실제 마스크 픽셀 수는 Tiny 497~61,037, Large 525~54,542였고, 직사각형·교환 마스크 픽셀 수를 각 실제 마스크에 맞췄다. 두 순열의 원장은 따로 보존했다. 위험 회수와 오탐은 Qwen의 실제 수락·기각 뒤에 계산한다. [사진별 세 조건 픽셀 수](../data/phase2-runs/p38-20261126-49bb82ce/regions/base-sam-size/matched-swap-design.json)

## 저비트 언어 가중치와 시각 프로젝터의 실파일 헤더

이미 받은 파일의 GGUF v3 헤더와 텐서 설명표를 읽었다. 9B 언어 파일은 텐서 427개·임베딩 폭 4,096, 27B는 866개·폭 5,120이었고 두 파일 모두 아키텍처 식별자가 `qwen35`였다. 각 BF16 시각 파일에는 텐서 334개가 있고 `clip` 아키텍처·`qwen3vl_merger` 프로젝터·이미지 크기 768을 명시한다. 언어 파일의 헤더와 설명표는 각각 10,968,259 B와 10,996,621 B다. 완성 파일의 전체 SHA-256, 실제 사진 출력, Linux/L4 실행은 별도 검증 항목이다. [4개 로컬 파일의 헤더 지문](../data/phase2-runs/p38-20261126-49bb82ce/large-models/gguf-headers.json) · [동일 빌드의 GGUF 형식](https://github.com/ggml-org/llama.cpp/blob/b12721/ggml/include/gguf.h)

3.1 공식 PT 파일의 249,782 B 상태 사전 메타데이터에는 검출기 텐서 1,166개·추적기 457개가 있다. 채점용 추론은 Meta의 통합 predictor 최종 적재 뒤 실행한다. 트래커 임시 조립 단계는 이름 접두사가 달라 1,623개 키 전부를 사용하지 않았고, 제작사 빌더가 다음 단계에서 같은 파일을 통합 모델에 다시 읽는 것을 확인했다. [체크포인트 구성과 로딩 단계](../data/phase2-runs/p38-20261126-49bb82ce/large-models/sam31/state-structure.json) · [제작사 통합 모델 경로](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model_builder.py)

## 첫 SAM 3.1 공개 사진 실행

공식 체크포인트를 RTX 3060에 실제 적재했다. 당시 호스트 경합을 포함한 import·적재 합계는 4,884.69초였고 이 중 import는 2,761.20초였다. 공개 사진의 첫 세션 호출은 49.32초, CUDA 최대 할당 3,926,040,576 B·예약 4,037,017,600 B에서 실패했다. 공통 predictor가 기본값 `offload_state_to_cpu=False`를 전달하지만 3.1의 `Sam3MultiplexTrackingWithInteractivity.init_state`에는 해당 인자가 없었다. 이 실행에서 유효 마스크와 개발 사진 처리 수는 모두 0이다. [첫 실패 원문과 런타임](../data/phase2-runs/p38-20261126-49bb82ce/regions/sam3-static/sam31-frame/attempts/original)

통합 모델은 32개 복소 회전 위치 인코딩 텐서를 읽었다. 제작사 기본 실수 연산 모드가 등록하는 64개 실수·허수 뷰는 저장된 키에 없어 경고가 남았다. 후속 실행기는 공식 구현의 `freqs_cis.real`과 `freqs_cis.imag`를 그대로 사용해 실수 연산 경로를 유지하고 통합 상태 사전의 키 일치를 검사한다. 세션 호출에서는 대상 메서드가 받지 않는 기본 false 옵션만 생략한다. 가중치 메모리 매핑과 키가 하나도 일치하지 않았던 임시 추적기 읽기의 생략은 후속 적재 조건이다. 이 변경의 실제 마스크는 첫 실패 원장과 별도 측정한다. [공통 세션 API](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model/sam3_base_predictor.py) · [회전 인코딩 버퍼 생성](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model/vitdet.py)

첫 접수 ZIP에서 베이스 두 파일과 어댑터 한 파일은 모두 STORED였고 합계는 9,007,914,792 B였다. 같은 방식으로 각 생성기를 교체하고 SAM 3.1을 더하면 8B BF16 21.037 GB, 9B BF16 22.809 GB, 27B GGUF·프로젝터 20.998 GB다. 8B FP8과 9B GGUF·프로젝터는 각각 14.093 GB와 10.105 GB다. 이 합계에는 새 어댑터·런타임·ZIP 헤더가 들어 있지 않다. 단독 모델의 실측과 배포 예산을 함께 판단하며 결합 후보 ZIP은 만들지 않았다. [실제 ZIP 항목·대형 체크포인트별 바이트](../data/phase2-runs/p38-20261126-49bb82ce/large-models/stored-package-budgets.json)

## 합의 제안에 쓰는 이미지 가중치와 캐시

두 분류 헤드는 EXIF 방향을 적용한 원본 RGB의 한 뷰를 공유한다. Phase1 인코더는 긴 변을 보존하며 평균색으로 정사각형을 채우고 SigLIP2 So400m은 384픽셀, DINOv2-L은 224픽셀로 처리한다. 각 이미지 타워의 전역 풀링 토큰과 4×4 공간 평균 토큰을 고정 행렬로 64차원까지 투영한 뒤 정규화한다. 타워별 벡터를 이으면 원경 한 장에 17×128 특징이 나온다. 선형 결합 헤드는 첫 전역 토큰만 사용하고, 사전확률 항을 제거한 공간 헤드는 17개 토큰을 사용한다. 학습에 쓰인 보조 근거 벡터는 현장 문장의 해시 특징이다. 추론에서 참가자 문장과 SigLIP 텍스트 타워를 호출하는 단계는 없다. 두 원가중치의 라이선스는 Apache 2.0이다. [Google 게시 리비전](https://huggingface.co/google/siglip2-so400m-patch14-384/tree/e8e487298228002f3d8a82e0cd5c8ea9c567f57f) · [Meta 게시 리비전](https://huggingface.co/facebook/dinov2-large/tree/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c)

로컬 파일 헤더에서 SigLIP 이미지 가중치는 FP32 텐서 448개·1,712,902,400 B, 텍스트 가중치는 438개·2,831,131,584 B였고 DINO 이미지 가중치는 439개·1,217,474,560 B였다. 관리자가 실사한 8B BF16과 두 BF16 이미지 타워의 텐서 소계는 18,999,527,992 B다. 20 GB까지 남는 1,000,472,008 B에 어댑터·헤드·런타임·실행 코드·ZIP 구조가 들어가는지는 별도 포장 검증 대상이다. [헤더와 실제 전처리 설정](../data/phase2-runs/p38-20261126-49bb82ce/agreement-heads/image-towers/header-audit.json)

개발 920장의 특징을 모은 뒤 외측 훈련 그룹에서 5개 선형 헤드를 다시 적합했다. 헤드별 정규화 중심·스케일·가중치·편향·사전확률을 저장하고 독립적으로 불러와 계산한 11종 확률은 기존 원장과 모두 일치했다. fold별 1위 일치는 179/179, 193/193, 179/179, 177/177, 192/192장이었다. 캐시 수집을 제외한 적합·저장·재로딩·평가 합계는 7.84초였다. [5개 헤드 지문·훈련 명단·수치 오차](../data/phase2-runs/p38-20261126-49bb82ce/agreement-heads/linear-progress.json)

SAM 3.0 공식 파일 3,450,062,241 B의 SHA-256이 제작사 값과 일치했다. 상태 사전은 검출기 1,156개·추적기 309개 텐서를 포함했다. 검증 직후 Qwen3-VL-8B 두 번째 샤드는 보존된 3,090,939,904 B에서 후속 바이트를 받기 시작했다. [SAM 3.0 파일 검증 및 구성](../data/phase2-runs/p38-20261126-49bb82ce/large-models/sam3/state-structure.json)

원문 보관 경로: `data/local-reports/ubuntu-handoff-originals/docs/phase2-large-model-execution-20261126.md`
