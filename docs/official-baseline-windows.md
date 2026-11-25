# 공식 베이스라인의 Windows 재현

이 실험은 제공된 Qwen3-VL-4B 본체, LoRA, 프롬프트와 법령 색인을 로컬 NVIDIA GPU에서 실행한다. 대회 ZIP을 푼 `data/baseline`과 키트를 푼 `data/local-kit`이 필요하다. 모델을 다시 내려받을 필요는 없다. 아래 명령은 프로젝트 루트에서 실행한다.

## Python 환경

Windows x64 Python 3.12와 NVIDIA GPU가 필요하다. 전용 환경 및 대용량 캐시를 G: 프로젝트에 둔다. 아래 PyTorch와 torchvision은 운영진 README의 학습 환경에 기재된 CUDA 12.6 조합이다. 운영 서버의 Linux용 추가 배포물 가운데 Windows에서 실행할 수 없는 tokenizers 휠은 같은 버전의 Windows 휠을 사용한다.

```powershell
Set-Location -LiteralPath 'G:/Projects/KWater-SafetyAI'
$env:UV_CACHE_DIR = 'G:/Projects/KWater-SafetyAI/.runtime/uv-cache'
$nativePython = (py -3.12 -c 'import sys; print(sys.executable)').Trim()
uv venv --python $nativePython .runtime/baseline-venv
$runner = '.runtime/baseline-venv/Scripts/python.exe'
uv pip install --python $runner torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $runner -r data/baseline/requirements.txt tokenizers==0.22.1 jsonschema
```

기존 환경으로 반복 실행할 때는 첫 명령으로 프로젝트 위치를 지정하고 `$runner`만 선택한다. 새 환경 생성은 처음 한 번만 필요하다.

### 이 PC의 CUDA 환경 재사용

기존 `env_aias_test`의 Python 3.12.8, CUDA 12.4용 PyTorch 2.5.1과 torchvision 0.20.1을 사용할 수 있다. 이 경로에서는 기반 환경의 패키지를 수정하지 않고 별도 G: 폴더의 패키지를 해석 경로 앞에 둔다. 프로젝트에 저장된 경로 목록은 Windows용 Pillow 11.3.0·safetensors 0.8.0과 주최측 지정 라이브러리 다섯 개를 가리킨다. 기반 환경의 NumPy 1.26.4는 배포 README에 기재된 2.2.6과 다르다.

```powershell
Set-Location -LiteralPath 'G:/Projects/KWater-SafetyAI'
$runtimeBase = 'C:/Users/JY/anaconda3/envs/env_aias_test'
$runner = "$runtimeBase/python.exe"
$env:PYTHONPATH = [IO.File]::ReadAllText((Resolve-Path '.runtime/legacy-overlay-path.txt'))
$env:PATH = "$runtimeBase/Library/bin;$runtimeBase/DLLs;$runtimeBase;$env:PATH"
$env:PYTHONUTF8 = '1'
```

환경을 새로 구성해서 경로 목록이 없을 때는 같은 버전의 Windows 휠을 한 폴더에 설치할 수 있다. 명령은 현재 PowerShell 프로세스에서만 환경변수를 설정한다.

```powershell
$env:UV_CACHE_DIR = 'G:/Projects/KWater-SafetyAI/.runtime/uv-cache'
$overlay = 'G:/Projects/KWater-SafetyAI/.runtime/official-overrides'
uv pip install --python $runner --target $overlay --no-deps transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 huggingface-hub==0.36.0 tokenizers==0.22.1 Pillow==11.3.0 safetensors==0.8.0
$env:PYTHONPATH = $overlay
```

## 예시 10장 실행

```powershell
& $runner -m kwater_lab.run_official_baseline --batch-size 1 --time-limit 5400 --run-name official-baseline-10-repeat
```

실행기는 공식 `main.py --input-dir … --output-dir …` 명령을 자식 프로세스로 띄우고 제공 ZIP의 파일을 그대로 사용한다. `--batch-size`는 대회 코드가 지원하는 `KW_BATCH` 환경변수다. 공식 기본값은 16이고, 12GB GPU에서 첫 검증에는 1을 사용한다. 5,400초는 이 Windows PC의 느린 콜드 로딩까지 관찰하기 위한 로컬 제한이다. Public 서버의 공식 제한은 1,800초다. `--run-name`에 새 이름을 주면 이전 실험 결과가 보존된다.

결과 JSONL과 원 실행 로그는 `data/<run-name>/output/submission.jsonl`, `data/<run-name>/model.log`에 생긴다. `results/<run_name>.json`에는 실행 시간, 환경, SHA-256, 배치별 생성 진척, 공식 검사 결과, 출력 코드와 법령 ID의 집계가 남는다. PDF나 외부 서비스에 입력 사진이나 예측 본문을 전송할 필요가 없다.

## 결과 해석

공식 모델 로그의 `valid model outputs`는 모델의 원 생성문이 내부 형식 검사를 통과한 수다. 내부 형식 오류가 나면 공식 래퍼가 해당 사진을 `no_visible_hazard`로 채울 수 있으므로 내부 유효 수를 함께 확인한다. 키트의 `check_output.py`는 직렬화가 끝난 예측을 검사한다. 키트 예시에는 대회 정답이나 리더보드 점수가 포함되어 있지 않다.

Windows 실행기는 Hugging Face·Transformers의 오프라인 환경변수를 설정하고 로컬 경로에서 모델을 읽는다. 모델 로딩부터 생성·저장까지의 벽시계 시간은 `elapsed_seconds_including_native_model_startup`으로 남기며, Python 환경 설치와 공식 출력 검사 시간은 제외한다. GPU 메모리는 기기 전체 사용량을 주기적으로 샘플링한 수치다.

공식 Linux 실행 이미지에서 네트워크 차단, seccomp, 메모리 제한과 설치 시간을 포함해 검사하려면 WSL2 또는 Linux x86-64와 Docker, NVIDIA Container Toolkit이 있는 환경에서 `data/local-kit/README.txt`의 `run_local.sh`를 따른다.
