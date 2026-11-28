# Pure-Inspect v3 서울 입력 재사용

다음 Fold 2 대조는 기존 8B 혼합 v3 훈련 TAR의 769장·주석·v3 프롬프트와 실행기를 재사용한다. **1,136,640바이트** 오버레이에 Pure 커리큘럼, 전용 학습 구성, 입력 지문만 담았다. 기존 검증 예제 156개를 같은 사진의 판독 타깃으로 바꾸고, 나머지 910개와 셔플 뒤 사진·증강 순서를 유지했다. 1,066예제와 67/134스텝 학습 일정도 같다. 이 준비에서 새 학습과 판독은 각 0건이다. [로컬 입력 검산](../results/stage1-pure-overlay-preflight-20261128.json) · [원래 학습 TAR 검산](../results/fold2-8b-matched-preflight-20261128.json)

## 회수 결과

11월 29일 서울 실행에서 Pure-Inspect v3는 혼합 v3보다 정상 오경보를 197장에서 두 건, 고정 55장에서 한 건 줄였다. 심각도 4 완전 미탐이 각 네 건과 한 건 늘어 승격을 보류한다. [사진 순서·학습 조건·쌍체 평가 영수증](../results/stage1-pure-vs-mixed-fold2-20261128.json)

| 같은 평가 사진 | 활성 Macro-F1 혼합 → Pure | 차이 [95% 구간] | 공통 8종 F1 | 정상 오경보 | 심각도 4 완전 미탐 |
|---|---:|---:|---:|---:|---:|
| 정본 197장 | .52644 → .50269 | −.02374 [−.07383, +.02820] | .59224 → .56553 | 5 → 3 / 67 | 25 → 29 / 59 |
| 고정 55장 | .50555 → .50342 | −.00213 [−.05587, +.07457] | .56875 → .56634 | 2 → 1 / 12 | 6 → 7 / 15 |

정본에서 Pure만 심각도 4 정답을 맞힌 사진은 한 장이고 혼합만 맞힌 사진은 다섯 장이었다. 위험 두 개 이상인 38장에서 맞힌 코드는 43→36/80이다. 추락 정답은 17→13/37, 인양은 1/12로 같았고 끼임은 3→2/8이었다. 고정 55장에서는 양쪽 모두 인양 정답 3장을 놓쳤다. 두 모델의 평가에는 기존 Fold 2 노출과 현장 제약이 그대로 적용된다. [코드별 전환](../results/stage1-pure-vs-mixed-fold2-20261128.json) · [Fold 2 표본의 범위](../reports/fold2-8b-matched-baseline-20261128.md)

학습 67단계의 예제 수와 학습률은 혼합 실행과 일치했다. 검증 답변을 판독 답변으로 바꾸면서 감독 토큰은 134,372→145,562개가 됐다. Pure 첫 에폭은 3,388.0초였다. 평가 197/197행이 JSON과 내부 스키마를 통과했고 토큰 상한 도달은 0건이었다. 같은 배치의 생성 누적 시간은 혼합 1,297.2초, Pure 1,213.4초다. 설치·모델 적재와 실제 제출 ZIP 검사는 별도 측정으로 남아 있다.

## 재현

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
& 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe' -B -m kwater_lab.stage1_pure_overlay prepare
```

별도 승인을 받은 실행에서 SSH로 기존 `train.tar`와 `pure-v3-overlay.tar`를 보낸다. 기존 파일이 승인된 실행 머신에 남아 있을 때는 그 파일의 SHA-256을 확인해 재사용한다. 새 디스크라면 원본 TAR를 한 번 전송한다. 아래 세 SHA에 각각 로컬 원장의 학습·오버레이·평가 지문을 넣는다.

```bash
umask 077
printf '%s  %s\n' "$TRAIN_SHA256" "$TRAIN_TAR" | sha256sum --check
printf '%s  %s\n' "$OVERLAY_SHA256" "$OVERLAY_TAR" | sha256sum --check
mkdir -m 0700 "$TRIAL_DIR"
cd "$TRIAL_DIR"
tar -xf "$TRAIN_TAR"
tar -xf "$OVERLAY_TAR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
flock -n fit-pure.lock "$PYTHON" -B -u -m kwater_lab.fold2_8b_baseline train \
  --base "$BASE8" --out fit-pure

# 첫 에폭을 마친 뒤 검증용 사진 TAR를 보낸다.
test -s fit-pure/epoch-1/training_state.pt
printf '%s  %s\n' "$EVAL_SHA256" "$EVAL_TAR" | sha256sum --check
tar -xf "$EVAL_TAR"
flock -n eval-pure.lock "$PYTHON" -B -u -m kwater_lab.fold2_8b_baseline infer \
  --base "$BASE8" --fit fit-pure --out eval-pure
```

`$TRIAL_DIR`은 빈 독립 디렉터리로 시작한다. 체크포인트 재개는 같은 디렉터리와 같은 `train` 명령을 쓴다. 실행기는 2·10·20·30·40·50·60번과 첫 에폭 종료에서 만든 체크포인트 중 가장 최근 완료본을 확인하며, Pure 실험 이름과 커리큘럼 지문이 일치해야 재개한다. 원래 혼합 실행의 체크포인트는 이 조건에서 통과할 수 없다. 평가 TAR에는 197장의 사진과 ID·경로·지문만 있다. 평가 주석·고정 마스크와 기존 모델 출력은 G:에서 채점에 사용한다.

회수 대상은 `fit-pure/epoch-1/`의 어댑터·상태, `fit-pure/steps.jsonl`, `fit-pure/history.json`, `eval-pure/`의 원생성 캐시·영수증이다. 아래 명령은 이미 회수하고 채점한 8B 혼합 v3 예측 파일과 영수증을 대조한 뒤 Pure 출력만 새로 포매팅한다. 학습 예제 수·학습률과 평가 입력·프롬프트·실제 배치를 양쪽에서 확인한다. 197장과 고정 55장의 활성 코드별 F1, 공통 8종, 현장 묶음 쌍체 재표집, 정상·중대·복수 위험 지표를 집계한다.

```powershell
& 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe' -B -m kwater_lab.stage1_pure_overlay score `
  --pure-fit data/fold2-8b-pure-v3/fit-pure `
  --pure-eval data/fold2-8b-pure-v3/eval-pure
```

이어서 실제 첫 에폭 어댑터의 오프라인 패키징이 필요하면 Stage 1 조립기로 좌표·훈련 지문을 다시 확인한다.

```powershell
& 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe' -B -m kwater_lab.prepare_stage1_epoch1_package `
  --source data/fold2-8b-pure-v3/fit-pure --candidate data/submissions/fold2-pure-v3-input `
  --scope fold2 --curriculum pure-inspect --prompt-version v3
```
