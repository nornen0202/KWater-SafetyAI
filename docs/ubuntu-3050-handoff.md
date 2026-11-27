# Ubuntu 26.04 보조 PC 인계

## Google Drive에서 별도 디스크로 전체 복원

Windows와 다른 장소에 있는 Ubuntu PC에서 지정한 프로젝트 경로로 내려받는다. 원본은 `jdragon2289@gmail.com`의 [비공개 복원 폴더](https://drive.google.com/drive/folders/1oca7JBCxPG8hiM80rRr9TumB9uK7JnDk) 안 `full-project-v1`이다. `READY.json`은 모든 아카이브 조각과 목록의 서버 측 크기·MD5가 일치한 후에 생성된다. 이 파일이 보일 때 전체 복원을 시작한다. 상위 폴더의 이전 가중치 백업은 별도로 사용할 수 있다.

1. Ubuntu에서 Python 3.12 이상과 rclone을 준비한다. `rclone config`로 유형이 `drive`인 원격을 만들고 이름을 `kwater`로 지정한다. Google 로그인 화면에서 위 이메일 주소를 확인한다. 다운로드 전용 권한으로 충분하다.
2. 작업 디스크에 쓸 수 있는 상위 폴더를 준비한다. 아래의 `/mnt/work`를 실제 경로로 바꾼다. 복원기는 프로젝트와 같은 위치에 이름이 `.dependencies`로 끝나는 형제 디렉터리를 만든다. 모든 자료의 압축 해제 용량과 임시 조각 1 GiB 이상의 여유가 필요하다. `manifest.json`에 그룹별 실제 파일 용량이 기록된다.
3. 아래 명령으로 복원한다. 중단되면 같은 명령과 같은 목적지로 재개한다. 완료한 그룹은 건너뛰고 중단된 그룹부터 다시 받는다. 복원 중에는 목적지에서 다른 작업을 실행하지 않는다.

```bash
set -e
parent=/mnt/work
dst="$parent/KWater-SafetyAI"
mkdir -p "$parent"
folder=1oca7JBCxPG8hiM80rRr9TumB9uK7JnDk
rclone copyto --drive-root-folder-id "$folder" kwater:full-project-v1/restore_ubuntu_from_drive.py "$parent/restore_ubuntu_from_drive.py"
python3 "$parent/restore_ubuntu_from_drive.py" --remote kwater --dest "$dst"
cd "$dst"
git status --short --branch
git fsck --connectivity-only --no-reflogs
```

각 조각은 내려받은 뒤 SHA-256을 검사한다. Git 이력과 인계 시점의 미커밋 파일, 공식 ZIP·사진·라벨, AI-Hub 원본, 실험 산출물과 Windows 실행 기록을 프로젝트 상대 경로로 복원한다. 프로젝트 밖의 전용 모델·캐시 디렉터리 `G:/CodexData/kwater-rd-20261125`는 `"$dst.dependencies"`에 있다. Windows 가상환경은 기록으로 보관하며 실제 학습은 아래 Linux 환경을 사용한다. 과거 로그·설정에 기록된 Windows 절대경로는 당시 값이다. 지원되는 명령은 프로젝트 루트에서 `--root "$PWD"`를 주고 외부 모델 경로는 `"$dst.dependencies"` 아래의 실제 파일을 지정한다. NTFS 정션 등 재분석 지점은 따라가지 않으며 목록은 manifest의 `skipped_reparse_points`에 있다. Windows에서 학습하며 저장한 여러 파일은 각 파일을 읽은 시점 기준이므로 진행 중 실험은 완료된 체크포인트에서 재개한다.

## Linux 실행 확인


기존 실행의 Python은 3.12.8이었다. 아래는 Python 3.12가 설치된 Ubuntu에서 시작하는 절차다. 시스템 기본 Python 버전이 다르면 별도 3.12 환경을 먼저 준비한다. CUDA 12.4용 PyTorch 휠에 맞는 NVIDIA 드라이버를 사용하는 예다. [PyTorch 공식 버전별 설치표](https://pytorch.org/get-started/previous-versions/)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.57.1 tokenizers==0.22.1 accelerate==1.10.1 peft==0.17.1 huggingface-hub==0.36.0 bitsandbytes==0.50.2 numpy==1.26.4 Pillow==11.0.0 scipy==1.15.1 scikit-learn==1.6.1 jsonschema==4.23.0 safetensors==0.8.0
python -m pip check
```

이 버전은 Windows 실험의 패키지 메타데이터에서 추출했다. 모델별 ONNX 등 추가 의존성은 기존 실험 문서를 따른다. 예전 CLIP·Depth 파일럿은 복원한 `$dst.dependencies/`에 프로젝트 밖의 모델 캐시를 사용한다. Linux에서는 해당 실험의 `--model` 또는 `--clip` 옵션으로 복원 경로를 지정한다. 파일럿에 Windows 고정 경로만 있으면 해당 상수를 Linux 복원 경로로 바꾼 작업 브랜치에서 실행한다.

실제 GPU 모델명·전용 VRAM을 먼저 읽는다. OEM 3050의 메모리나 실행 속도는 이 PC에서 측정해야 한다. 기존 3060의 QLoRA 학습 최대 할당은 6.65 GiB였으며, 모델 로딩·예약 메모리는 더 클 수 있다. Python 3.12 가상환경, CUDA용 PyTorch·torchvision 및 실험별 의존성을 준비한다. 공식 오프라인 휠과 해시 검증 순서는 [Linux 실행 문서](linux-fold0-parallel-readiness-20261127.md)를 따른다.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
python -m unittest discover -s tests -q
```

우선 CPU 지표 재생과 법령 평가를 수행할 수 있다. GPU 실험은 batch 1로 단일 사진 또는 preflight에서 실제 메모리를 확인한다. 새 결과는 예컨대 `data/ubuntu3050-runs/`에 둔다. Windows 학습과 같은 파일에 동시에 쓰지 않는다. Fold 4 검증 전에는 평가 사진을 훈련이나 임계값 선택에 포함하지 않는다. 개발 전체 적합 옵션 `val_fold=-1`은 현재 manifest에서 966장을 선택한다. 모델·GPU·정밀도·화소·입력 명단과 기준 커밋을 매번 기록한다.
