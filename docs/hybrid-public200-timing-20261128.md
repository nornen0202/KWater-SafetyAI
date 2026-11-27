# 하이브리드 ZIP 200장 오프라인 시간 검사

## CPU 사전 준비

프로젝트 루트에서 기존 개발 manifest의 사진 200장을 고정한다. 그룹마다 한 장씩 먼저 뽑고 고정 해시 순서로 채운다. 원본을 SHA-256으로 확인하고 같은 디스크의 로컬 `data/`에 하드링크를 만든다. 이 사진들은 전체 훈련에 쓰일 수 있는 개발 사진이다. 성능이나 일반화 점수 대신 처리 시간과 출력 형식을 검사한다.

```bash
python -B -m kwater_lab.hybrid_public200_timing prepare \
  --input-dir data/local-reports/hybrid-public200-timing-input \
  --out data/local-reports/hybrid-public200-panel-witness.json
```

전체 학습과 해당 어댑터의 마진 보정이 완료된 뒤 제작한다. 직접 번들을 만들어 정의한 실제 종료 어댑터의 SHA와 설정을 보존한다.

```bash
python -B -m kwater_lab.package_submission \
  --adapter-dir data/v3-11code-full1045/adapter \
  --specialist-ckpt data/v3-full1045-specialist/specialist_5fold_1045_ensemble.pt \
  --prompt-path kwater_lab/system_prompt_v3.txt \
  --batch-size 4 --verify-batch-size 2 --verify-max-new-tokens 280 --max-candidates-per-photo 1 \
  --dest-dir data/submissions/full1045-hybrid-stage \
  --zip-path data/submissions/kwater-full1045-hybrid.zip
python -B -m kwater_lab.offline_archive_audit --archive data/submissions/kwater-full1045-hybrid.zip \
  --out data/local-reports/full1045-hybrid-archive-audit.json
```

기존 6차 ZIP은 69/138스텝 Fold 4 단독 모델이다. 이 하이브리드의 실행 증거로 사용하지 않는다.

## 200장 실측

Linux x86-64, Docker/NVIDIA Container Toolkit 및 공식 키트의 고정 이미지가 있는 호스트에서 실행한다. 먼저 GPU 모델, VRAM, 드라이버와 로컬 Docker 이미지 ID를 원장에 기록한다. 공식 키트가 ZIP에서 실행한 `main.py`를 사용한다. 설치는 네트워크 차단과 해시 고정 wheelhouse에서 진행하고, 실행에는 seccomp를 적용한다.

```bash
set -o pipefail
unset KWATER_IMAGE KWATER_NO_GPU KWATER_RAM KWATER_CPUS KWATER_SCRATCH
ZIP=data/submissions/kwater-full1045-hybrid.zip
ARCHIVE_SHA256="$(sha256sum "$ZIP" | cut -d' ' -f1)"
{
  printf 'ARCHIVE_SHA256 %s\n' "$ARCHIVE_SHA256"
  KWATER_TIME_LIMIT=1800 bash data/local-kit/run_local.sh "$ZIP" \
    data/local-reports/hybrid-public200-timing-input
} 2>&1 | tee data/local-reports/full1045-hybrid-public200-kit.log
```

키트가 출력한 `작업 폴더`의 `output`을 지정한다. 출력 폴더에는 공식 제출 JSONL만 둔다. 키트 종료 코드가 0인 경우에도 마지막 상태, 200/200 유효 행, 출력 ID·SHA를 확인한다.

```bash
python -B -m kwater_lab.hybrid_public200_timing summarize \
  --input-dir data/local-reports/hybrid-public200-timing-input \
  --output-dir /path/from-kit/output \
  --panel-witness data/local-reports/hybrid-public200-panel-witness.json \
  --log data/local-reports/full1045-hybrid-public200-kit.log \
  --hardware "$GPU_AND_DRIVER" --archive-sha256 "$ARCHIVE_SHA256" \
  --out data/local-reports/full1045-hybrid-public200-timing.json
```

Public 시간에는 오프라인 설치와 `main.py` 실행이 포함된다. 로그의 specialist·모델 적재·1차 생성·마진·2차 검증·포맷 및 저장 시간을 함께 본다. 이 런타임에는 훈련측 마진 보정을 다시 적합하는 단계가 없다. Windows 3060의 기존 175장 결과나 Ubuntu 3050 결과를 L4로 환산하지 않는다. L4 실기가 없으면 측정한 장치에 한해 결과를 기록한다.

## Windows 네이티브 진단

GPU 담당자가 Windows에서 같은 SHA-256의 최종 ZIP을 새 로컬 폴더에 풀어 실행하는 경로다. 공식 Python ZIP 검사기, 실제 200개 입력 원본 SHA와 출력 checker를 재사용한다. ZIP 추출 시간과 `main.py` 프로세스 전체 벽시계를 나눠 기록하고, 단계별 시간과 실패·시간 초과를 로컬에 보존한다.

```powershell
$ZIP = 'data/submissions/kwater-full1045-hybrid.zip'
$SHA = (Get-FileHash -LiteralPath $ZIP -Algorithm SHA256).Hash.ToLowerInvariant()
& C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe -X utf8 -B -m kwater_lab.hybrid_public200_timing native `
  --archive $ZIP --archive-sha256 $SHA `
  --input-dir data/local-reports/hybrid-public200-timing-input `
  --panel-witness data/local-reports/hybrid-public200-panel-witness.json `
  --workspace data/local-reports/full1045-hybrid-windows-run1 `
  --out data/local-reports/full1045-hybrid-windows-timing.json
```

이미 별도 인계 원장에서 확인한 SHA가 있으면 그대로 사용해 선행 9GB 해시 읽기를 생략한다. `native`가 실행 직전에 실제 ZIP을 해시한다. Windows의 기존 venv를 사용하므로 빈 Linux 컨테이너에서의 설치 시간, seccomp, L4 벽시계는 이 기록으로 판정할 수 없다.

## 전량 전문가·마진의 범위

전량 전문가 체크포인트는 1,045장 그룹 단위 5-fold의 각 홀드아웃에서 3개 seed 판독을 평균해 OOF 점수를 만든다. 온도, 코드 문턱과 장면 기준은 전체 1,045개 OOF 행에서 한 번 적합한다. 제작기는 같은 행에서 계산해 체크포인트에 저장한 코드별 정밀도를 읽는다. 이 수치는 별도의 내측 폴드에서 적합하고 외측에서 재검증한 정밀도와 구별한다. 추론에는 각 사진마다 전체 15개 head를 사용한다. [전량 전문가 생성](../kwater_lab/v3_turnaround.py) · [제작기의 전문가·마진 로드](../kwater_lab/package_submission.py)

마진 보정에서 감독과 같은 정확한 사진·코드 쌍을 쓰지 않았더라도 사진 또는 사건 그룹과 1차 판독 감독의 중복은 별도로 확인해야 한다. 정규 Fold 4에서는 동일 보정 쌍 중복이 0건이었고, 검증기 학습과 사진 3장·그룹 2개가 겹쳤다. 전량 학습의 checkpoint-30 설정 지문으로 훈련 예제 1,345개를 재구성했다. 마진 보정 251쌍은 170사진·40그룹이며, 사진 170장 모두 1차 판독 모델의 감독에 쓰였다. 검증기 감독과 사진 5장·그룹 4개가 겹쳤고 동일한 사진·위험 코드 쌍은 0건이었다. 전량 어댑터와 전문가의 Fold 4 성능을 독립 검증으로 해석할 수 없다.
