# 기존 4B 유지와 Linux 병렬 실행 준비

로컬 Qwen3-VL-4B 학습은 계속 실행한다. 외부 반출 승인을 확보하면, 별도 CUDA GPU에서 가장 먼저 돌릴 일은 공식 1차 LoRA의 10장 점검과 동일 Fold0 216장 생성이다. 공식 기준 모델의 216장 생성은 새 LoRA 체크포인트를 기다릴 필요가 없다. 두 결과가 모두 나오면 대응 성능 비교와 정성 검토표 구성은 기존 로컬 실행기로 이어진다. 무료 런타임에 T4만 배정되면 공식 비교는 진행할 수 없다. 원본 로더에는 단일 CUDA GPU의 BF16 요건이 있고, 동일 비교에는 실제 BF16을 지원하는 L4·A100급 GPU가 필요하다. GPU 배정, 외부 전송, 권한 및 비용은 이 검증에서 실행하지 않았다.

## 입력과 저장 공간

| 항목 | 현재 로컬에서 확인한 크기 또는 범위 |
| --- | ---: |
| 원본 공식 ZIP | 9,032,392,642바이트 |
| 공식 ZIP 압축 해제 파일 총합 | 9,056,261,772바이트 |
| 위 압축 해제본 중 기반 4B 모델 디렉터리 | 8,887,291,213바이트 |
| 공식 로컬 검사 키트 ZIP | 2,713,157바이트 |
| 키트 10장과 manifest | 2,935,530바이트 |
| Fold0 216장과 manifest | 337,543,417바이트 |
| 수정판 Fold0 학습용 이미지 750장 | 1,280,371,988바이트 |

ZIP을 유지하며 해제하려면 데이터만 약 18.1GB이고 216장까지 올리면 약 18.4GB다. Python 환경, 캐시, 실행 결과도 별도의 쓰기 가능 공간을 사용한다. 공식 비교는 원본 추론 코드·프롬프트·LoRA와 동봉 사례집을 모두 사용한다. 공개 기반 모델 가중치만으로는 원래 42.04점의 공식 1차 LoRA를 재현할 수 없다. 키트 사진도 공식 참가 자료다. 참가 자료와 파생 예측의 외부 사용 근거가 확보된 경로에서만 아래 절차를 사용한다.

```powershell
python -m tools.plan_linux_fold0 --verify-inputs
```

이 명령은 사진을 복제하지 않고 로컬 크기와 분할을 조사한다. 현재 216개의 익명 ID·순서와 파일 SHA-256이 개발 manifest에 맞는 것을 확인했고, 수정판의 750개 학습 사진과 Fold0 평가 사진 사이에 공유 사건 그룹이 없음을 확인했다.

## 승인된 Linux 환경에서의 공식 기준 모델 실행

저장소 루트를 작업 위치로 사용한다. 경로는 `data/packages/kwater-baseline.zip`, 해제 대상 `data/baseline/`, 공식 키트 `data/local-kit/`, 준비된 동일 Fold0 입력 `data/fold0-production-input/` 그대로 사용한다. 평가 라벨이나 전체 학습 이미지가 공식 모델 추론에 추가로 필요하지 않다. 기준 모델에 이미 들어 있는 사례집은 원본 ZIP 구성의 일부다. Colab에서 실행하는 경우에도 노트북 코드는 아래 CLI를 호출하는 용도면 충분하다. 이 실행은 Linux GPU 성능과 파서 검증이며 공식 L4 채점 컨테이너 통과와 별도로 기록한다.

Python 3.12 및 CUDA에 맞춰 함께 설치된 PyTorch·torchvision을 확인한다. NumPy, Pillow, safetensors, jsonschema도 있어야 한다. 공식 아카이브의 오프라인 wheel 5개는 Linux용 `requirements.lock` 해시로 검증된다. Colab에 기본으로 들어 있는 패키지의 버전은 시작 시점에 확인하고, GPU에 맞는 PyTorch를 유지한다.

```bash
mkdir -p data/local-kit data/baseline
printf '%s  %s\n' be91eb3a745dc6e1f305237b44842106707164f4174940439b246e4abb385aab data/packages/kwater-baseline.zip | sha256sum -c -
unzip -q data/packages/kwater-local-kit.zip -d data/local-kit
python data/local-kit/kit/check_zip.py data/packages/kwater-baseline.zip data/baseline
python -m pip install --no-index --find-links data/baseline/wheelhouse --require-hashes --no-deps -r data/baseline/requirements.lock
python -m pip check
nvidia-smi
python - <<'PY'
import sys
import torch, torchvision, numpy, PIL, safetensors, jsonschema
from transformers import Qwen3VLForConditionalGeneration
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.cuda.device_count() == 1, 'Expected one CUDA GPU'
assert torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported(), 'Need native CUDA BF16'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY

mkdir -p results .runtime/hf
export HF_HOME="$PWD/.runtime/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1 PYTHONHASHSEED=42
python -B -u -m kwater_lab.fold0_official_resume --input-dir data/local-kit/sample-input \
  --batch-size 1 --run-dir data/linux-official-smoke10 --out results/linux_official_smoke10.json
```

10장 결과에서 모델 파싱 수, 공식 checker 10/10, 계산 시간, GPU 종류를 확인한 뒤 216장을 시작한다. 시간을 비교할 때는 집계 JSON의 `current_invocation_*_seconds`에서 출처 검증, 모델 로드, 생성 구간을 따로 본다. 10장의 생성 구간을 216장으로 선형 환산한 값은 실행 계획용 추정치다. 배치 크기는 우선 1로 유지한다. 동일 GPU 종류와 런타임에서 중단된 경우 동일 경로·명령으로 재개한다. 원본 코드, 모델 파일, GPU 종류, 패키지 버전, 입력 manifest·이미지 SHA가 바뀌면 기존 캐시를 이어 쓰지 않는다. 10장과 216장의 기록도 각각 분리한다. 휘발성 런타임에서는 승인된 내구성 있는 작업 공간을 별도로 확보할 수 있는 경우에만 재개를 기대할 수 있다.

```bash
python -B -u -m kwater_lab.fold0_official_resume --input-dir data/fold0-production-input \
  --batch-size 1 --run-dir data/linux-official-fold0 --out results/linux_official_fold0.json
```

공식 216/216 출력과 바이트 해시가 확인된 뒤 기존 Fold0 대응 비교와 눈가림 검토에 사용한다. `CANDIDATE_PREDICTIONS`는 로컬 학습 모델의 실제 포맷 출력 `ft_vlm_11code.jsonl`로, `CANDIDATE_SUMMARY`는 그 출력의 해시와 생성 설정을 담은 실행 요약으로 지정한다. 운영체제를 옮길 때 줄바꿈을 변환하면 원래 출력 해시와 일치하지 않는다. 판단 본문과 행별 출력은 승인된 로컬 `data/`에서 관리한다. ZIP과 배치 원문은 원격 저장소에 커밋하지 않는다.

```bash
python -m kwater_lab.fold0_production_compare compare \
  --baseline data/linux-official-fold0/output/submission.jsonl --candidate "$CANDIDATE_PREDICTIONS" \
  --baseline-summary results/linux_official_fold0.json --candidate-summary "$CANDIDATE_SUMMARY" \
  --output results/linux_official_fold0_paired.json
python -m kwater_lab.fold0_qualitative_review \
  --baseline data/linux-official-fold0/output/submission.jsonl --candidate "$CANDIDATE_PREDICTIONS" \
  --local-output data/linux-fold0-blind-review --summary results/linux_official_fold0_qualitative_proxies.json
```

## 수정 hard-negative 대안

공식 대조와 병렬로 수정판 학습까지 실행하려면 별도 GPU가 필요하다. 같은 Fold0 성능을 비교할 목적이라면 수정된 코드·프롬프트의 커밋을 고정하고, 기존 개발 manifest에서 Fold0을 뺀 750장으로 `--val-fold 0`을 사용한다. 개발 manifest는 그대로 유지하고 `data/train/train.jsonl`은 학습용 750개 행만 준비하며 이미지 상대 경로를 보존한다. Fold0 평가 라벨과 내부 보류 자료 전부는 학습 입력에서 제외한다. 다른 폴드를 검증 대상으로 학습한 모델은 Fold0 표준 대조에 합류시키지 않는다. 같은 기반 모델과 공식 VLM 의존성에 추가로 CUDA bitsandbytes가 필요하다. 학습 로그와 최종 요약에서 기존 로컬 run과 수정판을 따로 식별한다.

수정판 커리큘럼을 로컬 CPU에서 재생한 결과 기본 750개, 소수코드 증강 109개, 수용 예제 35개, 혼동 위험 거부 18개, 무위험 거부 17개로 929개였다. 거부 예제 35개 모두 후보 코드가 해당 사진의 라벨에 없었다. 이 검사로 모델을 학습하거나 GPU 생성 속도를 측정하지 않았다.

```bash
python -B -u -m kwater_lab.vlm_11code --mode preflight --val-fold 0 \
  --out-dir data/vlm-11code-fold0-corrected-linux
python -B -u -m kwater_lab.vlm_11code --mode train --val-fold 0 \
  --out-dir data/vlm-11code-fold0-corrected-linux --epochs 2 --grad-accum 16 \
  --checkpoint-steps 5 --stop-after-steps 5
```

이후 현재 체크포인트 `checkpoint-5`를 `--resume-from data/vlm-11code-fold0-corrected-linux/checkpoint-5`로 명시하고 학습 인수 및 절대 작업 경로를 유지한다. 체크포인트는 PEFT 가중치, 최적화기·스케줄러, CPU/CUDA RNG와 진행 위치를 포함한다. 기존 Windows 레거시 훈련의 중간 진행은 이 방법으로 이어갈 수 없다. 새 수정판의 5개 스텝 이후 재개를 실제 Linux CUDA 장비에서 먼저 확인하고, 같은 GPU 종류·런타임을 유지한다. 생산 공개 점수 개선은 학습 완료 후에도 Fold0 대응 평가로 별도 확인한다.
