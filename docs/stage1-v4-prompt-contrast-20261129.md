# Fold 2 · 같은 8B 어댑터의 v3/v4 프롬프트 비교

**같은 Fold 2 8B 첫 에폭 어댑터를 고정하고 추론 프롬프트만 바꾼다.** 기본 대조는 회수한 혼합 v3 어댑터의 67/134스텝 체크포인트다. Pure-v3 어댑터가 회수되면 두 추론의 어댑터 경로를 함께 교체해 별도 대조를 실행한다. 학습 커리큘럼, 적재 정밀도, 화소 상한은 각각의 쌍 안에서 고정한다. 기존 추론·채점 경로를 재사용하며 이 준비 단계의 신규 모델 적재·생성은 0건이다. [회수 혼합 8B](../results/seoul-fold2-matched-8b-epoch1-20261129.json) · [Pure-v3 준비](stage1-pure-v3-overlay-20261128.md) · [CPU 검산 집계](../results/stage1-v4-prompt-contrast-preflight-20261129.json)

8B BF16의 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, 393,216픽셀, 출력 상한 620토큰, 같은 디코딩·법령 포매터를 사용한다. 처음 44장은 4장씩 11배치, 뒤 153장은 2장씩 76배치에 마지막 1장이다. 실제 배치 축소가 발생하면 동일 조건 대조에서 실패로 처리한다. 55장은 197장에서 이미 고정한 민감도 부분집합이며 별도 생성 비용이 없다. 이 부분집합에는 과거 탐색 노출 이력이 있다. [공용 추론 계약](../kwater_lab/stage1_pure_inspect.py) · [Fold 2 민감도 원장](../results/sub78-fold2-prospective-readiness-20261128.json)

## 서울 BF16 추론 전용 경로

로컬 NF4의 고정 55장에서는 v4의 활성 Macro-F1이 `+.11656` 높았고 쌍체 95% 구간은 `[+.04077, +.24172]`였다. 중대성 4 완전 미탐은 그대로였고, `slip_trip`을 다른 코드와 함께 낸 오탐은 2건에서 4건으로 늘었다. 서울 BF16의 197장에서는 같은 혼합 어댑터로 두 팔을 한 실행환경에서 새로 판독한다. [로컬 NF4 실측](../results/stage1-local-nf4-prompt-v4-55-20261129.json)

로컬 준비기는 원래 `eval.tar`의 334,417,920바이트와 SHA-256을 대조하고, 197개 사진의 실제 바이트와 ID·경로·지문만 담긴 명세 한 개를 검사한다. 전송할 소스는 고정 목록의 자체 Python 모듈 18개, v3/v4 프롬프트, 해시만 담은 실행 계획으로 이뤄진 **368,640바이트 `source.tar`**다. 어댑터는 회수한 혼합 체크포인트의 `adapter_config.json`과 `adapter_model.safetensors` 두 파일만 사용한다. 훈련 TAR, 평가 정답과 민감도 마스크, 과거 예측, 법령 원문과 사례집은 G:에 남긴다. [전송 멤버와 지문](../results/stage1-prompt-remote-preflight-20261129.json) · [추론·로컬 채점 코드](../kwater_lab/stage1_prompt_remote.py)

```powershell
$Python = 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe'
$Work = 'data/stage1-prompt-remote-20261129'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $Python -B -u -m kwater_lab.stage1_prompt_remote prepare --work $Work
if ($LASTEXITCODE) { throw 'Eval-only preparation failed' }
$Proof = Get-Content -Raw results/stage1-prompt-remote-preflight-20261129.json | ConvertFrom-Json
```

다음 전송·실행 명령은 별도 승인을 받은 서울 인스턴스에서 쓴다. `$SshHost`에는 승인된 SSH 별칭, `$RemoteTrial`에는 새 전용 디렉터리의 절대 경로를 지정한다. 같은 평가 TAR가 승인된 인스턴스에 있으면 원장 SHA-256을 확인해 재사용할 수 있다.

```powershell
ssh $SshHost "umask 077 && mkdir -m 0700 '$RemoteTrial' '$RemoteTrial/adapter'"
if ($LASTEXITCODE) { throw 'Remote trial setup failed' }
$Adapter = $Proof.adapter_upload.local_directory
scp $Proof.eval_tar.path $Proof.source_tar.path "${SshHost}:${RemoteTrial}/"
if ($LASTEXITCODE) { throw 'Eval-only archive upload failed' }
scp "$Adapter/adapter_config.json" "$Adapter/adapter_model.safetensors" "${SshHost}:${RemoteTrial}/adapter/"
if ($LASTEXITCODE) { throw 'Adapter upload failed' }
```

아래의 `EVAL_SHA`, `SOURCE_SHA`, `ADAPTER_CONFIG_SHA`, `ADAPTER_SHA`는 로컬 준비 영수증의 지문이다. `BASE8`은 같은 고정 리비전의 BF16 파일이 준비된 평가 디렉터리 밖 절대 경로이며 `PYTHON`은 앞선 서울 실측에 쓴 CUDA 환경의 인터프리터다. 런처는 데이터 트리를 사진과 명세로 제한하고 원격 기반 가중치·프로세서 파일과 어댑터를 해시한다. 모델과 프로세서를 각각 한 번 적재한 뒤 v3·v4를 이어서 실행한다. 실제 적재된 FP32 LoRA 520개, 렌더링된 지시문, 두 팔의 런타임, 각 원생성 구간 지문을 영수증에 기록한다.

```bash
set -euo pipefail
umask 077
cd "$REMOTE_TRIAL"
printf '%s  %s\n' "$EVAL_SHA" eval.tar "$SOURCE_SHA" source.tar \
  "$ADAPTER_CONFIG_SHA" adapter/adapter_config.json "$ADAPTER_SHA" adapter/adapter_model.safetensors \
  | sha256sum --check
tar -xf source.tar
tar -xf eval.tar
unset PYTHONPATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
flock -n infer.lock "$PYTHON" -B -u -m kwater_lab.stage1_prompt_remote infer --base "$BASE8" --returned returned
```

G:로 회수할 파일은 `returned/run-receipt.json`과 `returned/{v3,v4}/raw-{0-44,44-197}.json` 네 개다. 동일 `--work`의 `returned/`에 각 하위 경로 그대로 저장한다. 로컬 브리지는 원생성 바이트를 보존하며 같은 학습 측 769장의 법령 사례로 양쪽을 포매팅한다. 이때 공용 영수증 형식을 생성하고 기존 `stage1_fold2_quality` 검사와 쌍체 197장·55장 채점을 호출한다.

```powershell
$Returned = Join-Path $Work 'returned'
foreach ($Tag in ('v3', 'v4')) {
    $Dest = New-Item -ItemType Directory -Path (Join-Path $Returned $Tag) -Force
    scp "${SshHost}:${RemoteTrial}/returned/$Tag/raw-0-44.json" `
        "${SshHost}:${RemoteTrial}/returned/$Tag/raw-44-197.json" $Dest.FullName
    if ($LASTEXITCODE) { throw 'Raw cache download failed' }
}
scp "${SshHost}:${RemoteTrial}/returned/run-receipt.json" $Returned
if ($LASTEXITCODE) { throw 'Run receipt download failed' }
& $Python -B -m kwater_lab.stage1_prompt_remote score --work $Work --returned $Returned
if ($LASTEXITCODE) { throw 'Paired local scoring failed' }
$V3 = Join-Path $Work 'scored/v3'
$V4 = Join-Path $Work 'scored/v4'
```

로컬 검산에서 소스 TAR만 펼친 빈 디렉터리의 진입점을 확인했다. 합성 원생성 두 팔로 G:의 769개 법령 사례, 포매터, 정본 197장·부분집합 55장 공용 검사를 통과했고 원생성 바이트가 그대로 유지됐다. 이 준비에서는 서울로 전송하거나 GPU 판독·학습한 건이 0이다. [격리 준비와 브리지 검산](../results/stage1-prompt-remote-preflight-20261129.json)

## 실행과 채점

승인된 동일 실행 환경에서 두 팔을 연속 실행한다. 기존 수신 혼합 어댑터를 쓰는 명령이다. Pure-v3에는 `$Adapter`를 회수한 `data/fold2-8b-pure-v3/fit-pure/epoch-1`로 바꾸고 `$Work`·`--result`에 별도 이름을 쓴다. 각 실행의 원생성 캐시·예측·영수증은 `data/`에 둔다.

```powershell
$Python = 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe'
$Base = 'data/phase2-runs/p38-20261126-49bb82ce/large-models/8b'
$Adapter = 'data/fold2-8b-matched-epoch1/fit/epoch-1'
$Work = 'data/stage1-pure-inspect/fold2/mixed-prompt-ab-20261129'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

& $Python -B -m kwater_lab.stage1_pure_inspect infer --scope fold2 `
  --base-model $Base --adapter $Adapter --out $Work --max-pixels 393216 --prompt v3
if ($LASTEXITCODE) { throw 'v3 generation failed' }
& $Python -B -m kwater_lab.stage1_pure_inspect infer --scope fold2 `
  --base-model $Base --adapter $Adapter --out $Work --max-pixels 393216 --prompt v4
if ($LASTEXITCODE) { throw 'v4 generation failed' }

$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath "$Adapter/adapter_model.safetensors").Hash.ToLower().Substring(0, 16)
$V3 = Join-Path $Work "train-v3-infer-v3-px393216-$Hash"
$V4 = Join-Path $Work "train-v3-infer-v4-px393216-$Hash"
& $Python -B -m kwater_lab.stage1_fold2_quality --factor prompt `
  --control "$V3/predictions.jsonl" --candidate "$V4/predictions.jsonl" `
  --control-receipt "$V3/receipt.json" --candidate-receipt "$V4/receipt.json" `
  --result results/stage1-fold2-8b-mixed-inference-v3-v4-20261129.json
if ($LASTEXITCODE) { throw 'paired scoring failed' }
```

회수한 역사적 v3의 두 배치 원생성·사진·프롬프트와 67스텝 어댑터를 검산했다. 훈련 TAR의 추론 엔진·포매터·v3 프롬프트는 현재 파일과 바이트 단위로 같았지만, 역사적 추론 영수증에는 공용 실행환경 지문과 모델·어댑터·법령·추론 소스를 함께 묶은 `controls`가 없었다. 이 대조는 v3와 v4를 같은 새 실행환경에서 생성한다. [기존 영수증 검사](../kwater_lab/fold2_8b_paired_score.py) · [공용 채점 계약](../kwater_lab/stage1_fold2_quality.py)

CPU 검산에서는 현재 기반 샤드 4개가 검증을 마친 8차 스테이지 파일과 같은 파일임을 확인하고, 그 실물 원장의 SHA-256을 고정 배포처 LFS 지문과 대조했다. 같은 USB 디스크의 전체 샤드 재독은 생략했다. 어댑터 지문은 회수 파일에서 새로 계산했고, 평가 사진은 197장 각각을 읽어 고정 명세와 대조했다. 각 프롬프트가 사용할 법령 원문과 학습측 사진 769장의 사례 원천은 같은 순서·지문으로 준비했다. 기존 v3 원생성 197장의 재포매팅 결과는 회수 예측과 바이트 단위로 같았다. 같은 원생성 양쪽을 대입한 채점 자체 검산에서 두 평가군의 ΔF1은 0이었고, 실제 프로세서로 확인한 이미지 슬롯 1개 기준 입력은 v3 1,148토큰과 v4 1,200토큰이었다. [8차 실물 원장](../results/fold4-8b-epoch1-research-package-20261128.json) · [프롬프트 사전검산](../results/stage1-v4-prompt-contrast-preflight-20261129.json)

## 판정 지표와 고정 분모

| 평가군 | 사진 / 현장 그룹 | 복수위험 사진 / 정답 코드 | 정상 | 중대성 4 | `slip_trip` 비정답 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fold 2 전체 | 197 / 38 | 38 / 80 | 67 | 59 | 136 |
| 민감도 부분집합 | 55 / 34 | 15 / 31 | 12 | 15 | 37 |

- **독립 가시 위험의 정답 회수**는 공식 코드가 둘 이상인 사진에서 맞힌 코드 수와 마이크로 재현율, 코드 집합 완전일치를 본다. 각 코드 적중·완전일치의 v4 획득과 소실을 같은 사진에서 짝지어 센다. 공식 정답은 코드 집합으로 평가한다.
- **정상과 중대성 4**는 정상에서 위험을 하나라도 낸 오경보와 심각도 4 정답을 하나도 맞히지 못한 사진 수를 센다. 현행 정상 오경보 참고선 18%에서는 허용 사진 수가 전체 최대 12장, 민감도 최대 2장이다. 위험별 TP·FN·FP와 정상·중대성 4의 쌍체 획득·소실도 함께 본다.
- **점수 불확실성**에는 기존 현장 묶음 쌍체 bootstrap 2,000회, seed 42를 쓴다. 팔별 활성 코드 macro-F1과 고정 공통 8코드 ΔF1·95% 구간을 함께 읽는다. 부분집합은 민감도 대조다.
- **`slip_trip` 오탐**은 공용 집계의 전체 `slip_trip` 음성 FP와, 아래의 다른 코드 동시 출력 FP를 함께 본다. 동시 출력은 부수 코드의 재현 가능한 대리지표다. 배경 건설자재만 보고 추가했는지 확인하려면 신규 오탐의 사진 원경과 모델 근거를 팔 이름을 가린 채 검토하고, 통로·발밑 장애가 보이는 경우, 배경 자재만 보이는 경우, 판단을 유보한 경우를 따로 집계한다. 코드 적중 여부와 ROI 좌표만으로 배경 여부를 판단할 수 없다. [공용 지표와 쌍체 판정](../kwater_lab/stage1_fold2_quality.py) · [기본 분모와 지표](../kwater_lab/fold2_8b_paired_score.py)

회수한 역사적 혼합 v3의 전체 `slip_trip` 오탐은 197장 평가의 음성 136장 중 4건이다. 그중 다른 코드를 함께 낸 오탐은 3건이고, 세 건 모두 함께 낸 코드 중 공식 정답이 있었다. 55장 민감도에서는 음성 37장 중 전체 오탐이 0건이다. [회수 혼합 실행 집계](../results/fold2-matched-4b-8b-epoch1-20261128.json)

위 명령으로 두 결과가 생긴 뒤, 아래 명령은 **다른 코드와 함께 나온 `slip_trip` 오탐**을 197장과 고정 55장에서 짝지어 집계한다. `gains`는 v4에서 새로 발생한 오탐 수, `losses`는 v4에서 사라진 오탐 수다. 다른 출력 코드가 실제 정답과 겹친 경우의 오탐도 함께 센다.

```powershell
$env:KWATER_PROMPT_AB_CONTROL = $V3
$env:KWATER_PROMPT_AB_CANDIDATE = $V4
@'
import os
from pathlib import Path
from kwater_lab.fold2_8b_baseline import write_json
from kwater_lab.fold2_v3_epoch1 import MASK, split
from kwater_lab.fold4_same_verifier import digest, read
from kwater_lab.stage1_fold2_quality import READINESS, audit_changed_factor, validate_arm
from kwater_lab.stage1_pure_inspect import PROMPTS
from kwater_lab.sub78_public200_compare import match_changes

_, outer = split()
arms = []
for name in ('KWATER_PROMPT_AB_CONTROL', 'KWATER_PROMPT_AB_CANDIDATE'):
    folder = Path(os.environ[name]).resolve()
    if not folder.is_relative_to(Path('data').resolve()):
        raise ValueError('Input must stay under data/')
    arms.append(validate_arm(folder/'predictions.jsonl', folder/'receipt.json', outer))
(before, a_receipt), (after, b_receipt) = arms
audit_changed_factor(a_receipt, b_receipt, 'prompt')
if [r['controls']['prompt_sha256'] for r in (a_receipt,b_receipt)] != [digest(PROMPTS[p]) for p in ('v3','v4')]:
    raise ValueError('Expected v3 then v4')
if digest(MASK) != read(READINESS)['sensitivity_mask']['recipe_sha256']:
    raise ValueError('Frozen sensitivity mask changed')
gold = {row['sample_id']: set(row['codes']) for row in outer}
selected = set(read(MASK)['prospective_mask_retained_ids'])
panels = {}
for name, ids in (('primary_197', gold.keys()), ('sensitivity_55', selected)):
    negatives = {sid for sid in ids if 'slip_trip' not in gold[sid]}
    a, b = ({sid for sid in negatives if 'slip_trip' in pred[sid] and len(pred[sid]) > 1}
            for pred in (before, after))
    a_hit = {sid for sid in a if gold[sid] & before[sid]}
    b_hit = {sid for sid in b if gold[sid] & after[sid]}
    panels[name] = {'slip_trip_absent': len(negatives), 'co_emitted_fp_before': len(a),
                    'co_emitted_fp_after': len(b), 'co_emitted_fp': match_changes(a,b,len(negatives)),
                    'co_emitted_with_correct_code_fp': match_changes(a_hit,b_hit,len(negatives))}
write_json(Path('results/stage1-fold2-v4-co-emitted-slip-trip.json'), {'panels':panels})
'@ | & $Python -B -
```
