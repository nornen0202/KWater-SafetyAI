# LoRA 배율과 전체 사진 프롬프트 비교 재현

프로젝트 루트는 `G:/Projects/KWater-SafetyAI`다. 대회 학습 사진과 정답, 공식 모델, 검사 키트, 기존 탐색 170장의 노출 목록과 특징 캐시는 이 루트 아래 `data/`에서 읽는다. 모델 원문과 사진별 결과도 이 폴더에서 보관한다. Git의 집계에는 사진 또는 생성문이 들어가지 않는다.

## 18장 LoRA 배율

탐색에 사용한 6개 사진 ID 접두사에서 접두사당 3장을 고른다. 선정에는 정답을 쓰지 않고 고정 문자열을 더한 사진 ID의 SHA-256 정렬순서를 쓴다. 두 조건은 같은 사진 순서, 원본 가중치, 공식 LoRA, 원본 프롬프트를 사용한다. 기본 배율과 배율 0.75를 한 차례 모델 적재로 비교한다. PEFT의 기본 어댑터가 활성화된 각 층의 기존 scaling에 이 값을 곱하고 조건을 마치면 복구한다.

Windows에서 사용한 CUDA 환경은 [공식 베이스라인 재현](official-baseline-windows.md)에 있다. 다음 명령을 PowerShell에서 실행한다.

```powershell
Set-Location -LiteralPath 'G:/Projects/KWater-SafetyAI'
$runtimeBase = 'C:/Users/JY/anaconda3/envs/env_aias_test'
$runner = "$runtimeBase/python.exe"
$env:PYTHONPATH = [IO.File]::ReadAllText((Resolve-Path '.runtime/legacy-overlay-path.txt'))
$env:PATH = "$runtimeBase/Library/bin;$runtimeBase/DLLs;$runtimeBase;$env:PATH"
$env:PYTHONUTF8 = '1'
& $runner -X faulthandler -u -m kwater_lab.adapter_scale_pilot --weight-reader pread-cpu --save-progress
```

실험 코드는 입력과 공식 진입점의 해시를 대조하고, 생성 시드 42, 사진당 배치 1, SDPA, 공식 최대 화소 설정 `384 × 32 × 32`을 쓴다. 옵션 `pread-cpu`는 safetensors 0.8의 위치 지정 읽기를 써서 체크포인트 텐서를 CPU로 읽은 뒤 Transformers가 원래의 CUDA 장치에 올리게 한다. `--weight-reader default`는 원래의 파일 읽기 경로를 쓴다. `--save-progress`를 지정하면 사진별 내부 검증 결과를 `data/adapter-scale-pilot/progress-<배율>/`에 즉시 저장한다. 각 조건이 끝나면 원래 함수가 반환한 모든 행을 `generated-scale-1.0.json` 또는 `generated-scale-0.75.json`에 쓰고 사진별 복사본과 대조한다. 내부 형식에 맞지 않은 결과는 빈 위험 집합으로 집계하고 폴백 건수를 병기한다.

아래 명령은 새 PowerShell에서 실행한다. 첫 번째 Python에는 NumPy가, 검사기로 지정하는 Python에는 jsonschema가 필요하다. 저장한 결과를 공식 래퍼·법령 목록으로 변환하고, 변환 전후 위험 집합과 입력 목록 전체에 대한 공식 형식을 검사한다. 이 재생은 검색용 학습 사례를 비워 둔다. 결과는 `results/adapter_scale_pilot.json`의 `format_replay`에 추가된다.

```powershell
Set-Location -LiteralPath 'G:/Projects/KWater-SafetyAI'
$env:PYTHONUTF8 = '1'
python -m kwater_lab.adapter_scale_pilot --verify --checker-python .runtime/baseline-venv/Scripts/python.exe
```

## 6장과 다섯 프롬프트

위의 CUDA 작업이 끝난 뒤 같은 GPU에서 실행한다. 이 실험에는 공개 Qwen3-VL-4B-Instruct Q4_K_M과 F16 비전 프로젝터를 사용한다. `kwater_lab.visual_probe`가 지정한 고정 리비전·SHA-256, llama.cpp b12721 Vulkan 실행 파일, 기존 CLIP·SigLIP2 특징과 임계값을 사용한다. 캐시 기본 위치는 `G:/CodexData/kwater-rd-20261125/visual-probe`다.

```powershell
Set-Location -LiteralPath 'G:/Projects/KWater-SafetyAI'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
python -m kwater_lab.prompt_pilot
python -u -m kwater_lab.prompt_pilot --run
python -m kwater_lab.prompt_replay --checker-python .runtime/baseline-venv/Scripts/python.exe
```

6개 접두사에서 두 인코더의 위험 후보가 다른 사진을 우선해 1장씩 고른다. 같은 사진에 모든 조건을 적용한다. `swapped_11`은 원래 후보를 고정하고 다음 사진으로 회전시켜 잘못 짝지은 후보의 영향을 본다. 이 조건의 정답 대조와 사람 판정은 실제 모델에 들어간 사진을 기준으로 한다.

| 조건 | 위험 코드 | 판독 지시 |
|---|---|---|
| direct_8 | 8개 | 위험을 먼저 쓰고 장면을 요약한다. 익수·침수, 밀폐공간, 온열·기상은 후보에서 제외한다. |
| direct_11 | 11개 | 위험을 먼저 쓰고 장면을 요약한다. |
| facts_11 | 11개 | 대상·위치·방호 상태를 먼저 쓰고 위험을 판정한다. |
| verify_11 | 11개 | 장면 관찰 후 두 인코더 후보의 합집합을 사진과 대조한다. |
| swapped_11 | 11개 | 동일한 후보 검증 지시와 다른 사진의 후보를 조합한다. |

원본 전체 사진을 로컬 루프백 서버로 보내며 온도 0, 시드 42, 조건당 최대 640토큰이다. 모든 사진·조건에 같은 생성 한도를 적용한다. 대용량 가중치는 `--load-mode none`으로 읽고 서버 준비에는 최대 1,800초를 허용한다. 서버 실행 전에 가중치 해시도 검증한다. 각 응답은 `data/prompt-pilot/<설계 서명 앞 12자리>/`에 즉시 저장된다. 중단 뒤 같은 설계로 재실행하면 완료된 응답을 재사용한다.

코드 점수는 다섯 조건 모두에서 변환 가능한 사진의 교집합에서 계산한다. 조건별 전체 생성 수와 변환 가능한 수를 함께 읽어야 한다. 공식 재생은 각 조건의 6개 ID를 모두 manifest에 남기고 변환이 불가능한 응답은 출력 행에서 생략한다. 각 조건의 공식 유효 행 수, 종료 코드와 제출 파일 해시는 `results/prompt_format_replay.json`에 기록한다. 프롬프트 비교의 조치와 법령은 프로토타입 직렬화를 따르며, 전기 위험의 조치는 지정 문구로 대체하고 법령 인용은 빈 목록으로 둔다.

## 산출물 해석

위험 코드 Macro-F1은 정답이나 예측에 한 번 이상 등장한 코드의 F1 평균이다. 접두사는 독립 사건 식별자로 검증되지 않았다. 이 사진들은 탐색 과정에 노출됐고, 배포 LoRA도 대회 학습 자료를 사용했다. 두 실험의 수치는 해당 표본에서만 읽을 수 있다. 공식 형식 검사는 사진별 근거의 타당성, 예방 조치의 실효성, 법적 적절성을 평가하지 않는다.

프롬프트 실험의 `blind-review.json`, `blind-key.json`, `blind-pairs.json`은 로컬에 생성된다. 블라인드 판정자는 사진과 원 생성문을 대조해 관측 근거와 조치 실행 가능성을 따로 판단한다. 원 생성문에는 직렬화 시 적용되는 전기 위험 조치 대체가 반영되지 않는다. 독립 사건 성능은 사건 식별자를 확인하고 학습·후보 생성부터 분리한 평가에서 별도로 측정해야 한다.
