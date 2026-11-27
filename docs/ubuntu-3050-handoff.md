# Ubuntu 26.04 보조 PC 인계

2026-11-27. Windows는 기존 RTX 3060 실험을 이어간다. Ubuntu는 별도 체크아웃과 결과 폴더를 사용한다. 최신 비교 결과, 데이터 노출 이력과 제출 판단은 [실험 인계](../reports/turnaround-handoff-20261127.md)를 기준으로 한다. 공식 Public 최고점은 42.04였다. 원본 데이터는 대회 종료 후 삭제 조건을 따른다.

## 외장 G:로 전체 작업 복원

Windows의 학습·Drive 복사 등 G:를 읽거나 쓰는 작업이 끝난 뒤 안전하게 분리한다. Ubuntu에서 가능하면 읽기 전용으로 마운트하고 ext4 등 Linux 작업 디스크에 복원한다. AI-Hub 원본만 약 357 GB다. 먼저 아래 두 명령으로 실제 필요한 공간과 여유 공간을 비교한다. 외장 디스크가 다른 이름으로 마운트되면 첫 줄을 바꾼다.

```bash
src="/media/$USER/디스크이름/Projects/KWater-SafetyAI"
du -sh "$src/data" "$src/.git"
df -h "$HOME"
bash "$src/tools/restore_ubuntu_from_g.sh" "$src" "$HOME/KWater-SafetyAI"
cd "$HOME/KWater-SafetyAI"
git branch -r
```

우선 공식 데이터와 모델로 실험을 시작할 때는 복원 명령 끝에 `--core`를 추가한다. 이 모드는 AI-Hub 357 GB와 과거 중간 체크포인트를 건너뛴다. 나중에 옵션을 빼고 같은 명령으로 나머지를 복사할 수 있다.

복원기는 외장 원본에서 Git 이력과 로컬 실험 브랜치를 복제하고, Git에서 제외한 데이터·개별 예측·중간 체크포인트·제출 ZIP·과거 보고서를 상대 경로 그대로 복사한다. 같은 경로와 같은 원본 커밋이면 중단 후 재실행할 수 있다. 새 복원 디렉터리를 사용한다. 기존 Linux 작업을 다른 스냅샷으로 덮어쓰지 않는다. 외장 PC의 브랜치는 `windows/`, GitHub는 `origin`이다. Windows 전용 `.runtime`의 기존 실행 로그·패키지·도구는 외장 원본에도 남는다. 필요한 파일만 열어 보고 Windows 가상환경은 Linux에서 새로 설치한다.

## Drive와 원본의 구분

GitHub의 `ubuntu-handoff/kwater-phase2-model-comparison`과 `ubuntu-handoff/kwater-nov26-public-releases`는 과거 개발 브랜치의 코드 스냅샷이다. 개별 사진과 연결된 기록은 외장 원본의 `windows/nornen0202/...`에 있다. GitHub의 `main`은 현재 개발 버전이다.

지정 계정은 `jdragon2289@gmail.com`이다. 내 드라이브의 `KWater-Ubuntu-20261127-Research`에는 자체 학습 가중치·재개 체크포인트와 합성 관찰 자료를 프로젝트 상대 경로로 둔다. 기존 `KWater-SafetyAI`에는 공개 인코더와 FloodNet 등 앞서 전송한 자료가 있다. 전체 개발 자료는 외장 G: 복원 경로를 사용한다. 공식 사진·라벨·배포 가중치는 주최 측 참가 페이지에서도 받을 수 있다. AI-Hub 71407·71793·507은 국내 승인 환경에서 이용한다. [대회 데이터 규정](https://hackathon.imaginationgroup.co.kr/kwater/api/report) · [AI-Hub 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105)

## Linux 실행 확인

기존 실행의 Python은 3.12.8이었다. 아래는 Python 3.12가 설치된 Ubuntu에서 시작하는 절차다. 시스템 기본 Python 버전이 다르면 별도 3.12 환경을 먼저 준비한다. CUDA 12.4용 PyTorch 휠에 맞는 NVIDIA 드라이버를 사용하는 예다. [PyTorch 공식 버전별 설치표](https://pytorch.org/get-started/previous-versions/)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.57.1 tokenizers==0.22.1 accelerate==1.10.1 peft==0.17.1 huggingface-hub==0.36.0 bitsandbytes==0.50.2 numpy==1.26.4 Pillow==11.0.0 scipy==1.15.1 scikit-learn==1.6.1 jsonschema==4.23.0 safetensors==0.8.0
python -m pip check
```

이 버전은 Windows 실험의 패키지 메타데이터에서 추출했다. 모델별 ONNX 등 추가 의존성은 기존 실험 문서를 따른다. 예전 CLIP·Depth 파일럿은 외장 G:의 `CodexData/kwater-rd-20261125/`에 프로젝트 밖의 모델 캐시를 사용한다. Linux에서는 해당 실험의 `--model` 또는 `--clip` 옵션으로 마운트 경로를 지정한다. 파일럿에 Windows 고정 경로만 있으면 해당 상수를 Linux 복원 경로로 바꾼 작업 브랜치에서 실행한다. Windows 전용 실행 기록까지 보관하려면 별도 보관 디렉터리로 `rsync -rltH --safe-links --partial "$src/.runtime/" "$HOME/kwater-windows-runtime/"`을 실행한다.

실제 GPU 모델명·전용 VRAM을 먼저 읽는다. OEM 3050의 메모리나 실행 속도는 이 PC에서 측정해야 한다. 기존 3060의 QLoRA 학습 최대 할당은 6.65 GiB였으며, 모델 로딩·예약 메모리는 더 클 수 있다. Python 3.12 가상환경, CUDA용 PyTorch·torchvision 및 실험별 의존성을 준비한다. 공식 오프라인 휠과 해시 검증 순서는 [Linux 실행 문서](linux-fold0-parallel-readiness-20261127.md)를 따른다.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
python -m unittest discover -s tests -q
```

우선 CPU 지표 재생과 법령 평가를 수행할 수 있다. GPU 실험은 batch 1로 단일 사진 또는 preflight에서 실제 메모리를 확인한다. 새 결과는 예컨대 `data/ubuntu3050-runs/`에 둔다. Windows 학습과 같은 파일에 동시에 쓰지 않는다. Fold 4 검증 전에는 평가 사진을 훈련이나 임계값 선택에 포함하지 않는다. 개발 전체 적합 옵션 `val_fold=-1`은 현재 manifest에서 966장을 선택한다. 모델·GPU·정밀도·화소·입력 명단과 기준 커밋을 매번 기록한다.
