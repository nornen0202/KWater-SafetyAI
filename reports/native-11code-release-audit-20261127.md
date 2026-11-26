# 11코드 후보 제작 및 공식 도구 검증

## 공식 실행과 패키지

공식 1차 제출본은 대회 제공 ZIP 원본이다. 보관 ZIP은 9,032,392,642바이트, 47개 항목이며, 공식 추출 과정에서 CRC가 검증됐다. 원래 실행한 사진 10장의 출력에 공식 `check_output.py`를 다시 적용해 10/10을 확인했다. 기존 Windows 실행에서 모델 시작을 포함해 2,035.436초, 생성 구간 90초, 시스템 최대 GPU 메모리 11,831MiB가 기록됐다. 이 시간은 L4 제출 컨테이너의 측정값으로 사용할 수 없다. [원본 ZIP 검증](../results/official_baseline_archive.json) · [원본 모델 실측](../results/official_baseline_10.json) · [실제 1차 접수·점수 기록](../results/public_submission_001.json)

후보의 `main.py`는 평가 코드와 동일한 11코드 포맷터, 후보별 엄격한 시각 검증, 7뷰 전처리와 특징 추출, 공간 전문가 헤드, 절 단위 법령 검색기를 사용한다. 최초 생성본은 문자열 이스케이프 때문에 컴파일에 실패했다. 수정한 생성 진입점을 CPU에서 컴파일하고 import했으며, 11종 예제와 파싱 실패 예제 한 건으로 만든 출력은 공식 검사에서 12/12였다. 이 회귀 검사는 이미지 추론을 실행하지 않는다. [패키지 생성기](../kwater_lab/package_submission.py) · [회귀 검사](../tests/test_release_pipeline.py)

CPU에서 기반 모델 shard 2개의 713개 키가 공식 모델 인덱스와 일치하고, 오프라인 wheel 5개가 `requirements.lock` 해시와 일치하며, 전문가 head 15개가 엄격한 상태 로드에 통과했다. 전체 `unittest discover -s tests -v`는 73개 통과했다. 재실행에 필요한 ONNX Runtime 1.20.1은 테스트용 G: overlay로만 설치했다.

```powershell
$python = 'C:/Users/JY/anaconda3/envs/env_aias_test/python.exe'
$env:UV_CACHE_DIR = (Join-Path $PWD '.runtime/uv-cache')
$env:TEMP = (Join-Path $PWD '.runtime/temp')
$env:TMP = $env:TEMP
uv pip install --python $python --target .runtime/release-qa-deps --no-deps onnxruntime==1.20.1
$env:PYTHONPATH = (Join-Path $PWD '.runtime/release-qa-deps')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONUTF8 = '1'
& $python -m unittest discover -s tests -v
```

회귀 검사에 넣은 테스트 문장은 자체 작성 예제다. 원격에 보존하는 법령 집계는 사진 수, 사건 수, 인용·위험 집계와 비율만 포함한다. 사진, 행별 식별자, 원문, 모델 출력, 시험 결과 JSONL, 눈가림 키는 해당 집계와 소스에 넣지 않았다.

패키지 생성기는 완료된 학습 요약, SHA-256이 맞는 LoRA, 언어 모델과 일반·DeepStack 시각 merger 가중치, 기반 모델 shard 키, wheel 해시를 확인한다. 접수용 앙상블에는 개발셋 전체에서 학습한 `val_fold=-1` LoRA가 필요하다. 검증용 fold0 가중치는 `--vlm-only --stage-only`로만 이 감사에서 사용한다. 넣는 법령 사례는 어댑터 학습 manifest의 사진으로 제한한다. 79장 보호 패널은 새 사례집이나 이번 추가 법령 평가에 사용하지 않았다. 28장 mixed 기존 캐시는 이미 생성된 출력에 대한 역사적 기록이며, 최종 실험 집계에는 개발 집합 밖 7장을 제외했다. [패키지 생성기](../kwater_lab/package_submission.py) · [캐시 재생](../kwater_lab/production_law_replay.py)

제작기는 기존 폴더나 ZIP을 교체하지 않는다. ZIP을 만들 때 게시 전 실제 바이트 수를 검사한다. 후보 예산을 safetensors 헤더의 원소 수로 계산한 결과, 기반 모델·wheel·조문 8,907,305,057바이트에 FP16 SigLIP 시각 tower 본체 약 856,451,200바이트, FP16 DINOv2 약 608,737,280바이트, 전문가 head 38,931,754바이트가 추가된다. 이 소계는 약 10.41GB이며 어댑터·신고·코드·사례집·ZIP 오버헤드는 실제 ZIP에서 다시 확인할 항목이다. 공식 상한은 ZIP 20GB, L4 단일 GPU 24GB, RAM 56GiB, 임시 공간 32GiB, 설치를 포함한 Public 1,800초 및 Private 3,600초다. 체크 키트 Python의 ZIP 구조 검사와 대회 공지의 20GB 상한을 모두 적용한다. [공식 로컬 키트 README](../data/local-kit/README.txt)

2026-11-27 00:27 KST에는 완료된 fold0 가중치가 없어 실제 후보 staging 명령이 출력 폴더를 만들기 전에 안전하게 종료했다. L4 호환 모델 실행, 오프라인 설치, seccomp 및 시간·메모리 상한은 후보 ZIP에서 검증이 필요하다. Windows 호스트에 Docker CLI와 설치된 WSL 배포판이 없다. G:의 Item 39에는 QEMU Ubuntu guest와 OCI manifest가 요구하는 blob 15개가 있고, 15개 모두 바이트 수가 manifest와 일치했다. 현재 실행 프로세스는 없고 과거 로그에는 준비 중단까지 기록되어 있으므로 공식 컨테이너 실행 여부는 확인되지 않았다.

## 법령 재생

원본 공식 LoRA가 이미 생성한 원문을 재사용했다. Prototype 18장과 fresh 개발 21장에 대해, 두 재생기는 조회 사례에서 평가 사진의 사건 그룹을 모두 제외한 동일한 172장 개발 사진과 위험별 인용 1개 상한을 적용했다. 비교 기준인 원본 production 인용은 원래 대회 사례집을 사용했다. [정량 집계](../results/native_11code_production_law_replay.json)

| 평가 범위 | 원본 production 인용 F1 | 동일 훈련 사례 legacy top-1 | 동일 훈련 사례 native top-1 | 정답코드 일치 위험의 첫 인용 적중 `원본 → native` | 인용 선택으로 복구 불가한 위험코드 누락 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prototype 18 | 0.4348 | 0.3913 | 0.3913 | 10 → 9 / 12 | 11 |
| Fresh 개발 21 | 0.4444 | 0.4889 | 0.4889 | 10 → 11 / 11 | 14 |

정답 텍스트와 코드가 모두 주어지는 별도 오라클에서는 legacy→native F1이 prototype에서 0.8077→0.8462, fresh 개발에서 0.6182→0.6545였다. 반면 전체 개발셋의 별도 5-fold 정답 텍스트 평가는 legacy top-1 0.7752, native top-1/2 0.7719였고, 재실행 결과가 보관 집계와 바이트 단위로 일치했다. 공식 LoRA 생성문의 동등 조건에서 법령 교체로 얻은 개선은 확인되지 않았다. [5-fold 오라클](../results/law_reranker_11code_oof.json)

```powershell
python -m kwater_lab.production_law_replay --out results/native_11code_production_law_replay.json
```

## Fold0 대응 평가와 눈가림 정성 검토

Fold0 216장 입력은 개발 manifest에서 하드링크로 준비하고 216장 모두 ID 매핑과 이미지 SHA-256을 대조했다. 원본 재개 실행기는 입력 이미지 SHA와 공식 ZIP 원본 코드 및 가중치 CRC·SHA를 검사한 다음, 원본 `main.py`의 모델·파서·포맷터로 생성한다. 원문은 현지 `data/` 아래에서 매 배치 원자 저장된다. 후보 평가도 베이스·LoRA·설정·프롬프트·사진 식별자가 일치하는 캐시에서 재개한다. 원본과 후보 양쪽 출력에 216장이 모두 있어야 paired 비교가 실행된다. 숫자로 비교하는 원본 LoRA는 전체 공식 학습셋에 이미 노출된 비교 기준이다. Zero-shot은 별도 기준이다. Fold0 전문가 기본 정책은 외측 fold 라벨을 사용한 모델의 OOF 보정을 이어받지 않도록 고정 0.5로 지정했다. [원본 재개 실행기](../kwater_lab/fold0_official_resume.py) · [후보 실행기](../kwater_lab/hybrid_pipeline.py) · [대응 비교기](../kwater_lab/fold0_production_compare.py)

대응 파일이 모두 준비되면 사전 등록한 서로 다른 사건 36장 안에서 A/B 순서를 무작위 배정한다. 검토표와 정답 키는 서로 다른 파일로 `data/`에 저장된다. 사람이 근거의 가시성, 조치의 구체성, 인용의 지지를 검토하며 미실시 점수는 빈칸으로 남는다. 빈 필드, 동일 문장 반복, 활성 조문 존재, 정답 코드·인용 일치는 자동 대리 지표로 별도 집계한다. 공식 정성 30점은 산출하지 않았다. [눈가림 검토 생성기](../kwater_lab/fold0_qualitative_review.py)
