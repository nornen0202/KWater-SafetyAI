# Fold 4 v3 한 에폭 후보 ZIP 대조

원래 2에폭 계획의 첫 에폭 체크포인트, 175장의 실제 원생성, 원래 Fold 4 ID로 포맷한 175행, ZIP에 사용한 프롬프트를 각각 고정한다. 학습 원래 스케줄, 실제 업데이트 수, 어댑터·프롬프트·manifest·원생성의 SHA-256과 생성 BF16 설정을 런너의 원장과 대조한다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -B -m kwater_lab.fold4_native_archive_replay --archive $CandidateZip --raw $Fold4Raw --reference $Fold4Formatted --adapter $Epoch1AdapterWeights --prompt $FrozenPrompt --benchmark-witness $GenerationWitness --source-observations $LocalSourceObservations --workspace data/local-reports/v3-epoch1-official-check --out results/fold4-v3-epoch1-archive-check.json
```

실제 ZIP의 진입점을 읽어 175개 원생성의 파싱, 법령 인용, 행 포맷을 재실행한다. 모델 및 사진 디코딩 경계만 로컬 재생으로 대체한다. 별도 생성 산출물과는 결정적으로 바꾼 `image_id`를 제외한 모든 행 필드를 비교한다. 공식 결과 검사 175/175와 행 차이 0을 함께 통과 조건으로 사용한다. 실제 실행 원장의 SHA-256과 BF16 조건을 대조한다. 어댑터 바이트와 프롬프트, 단독 분기, 배치 4, 화소 상한 393,216, 생성 상한 620을 확인한다. 법령 사례는 Fold 4 외 개발 791장에 해당하는 명단으로 다시 만들고 ZIP의 사례를 전부 대조한다. 휠 SHA 및 공식 명세 점검을 포함한다.

로컬 원장과 파일은 `data/`에 둔다. 이 검사는 GPU를 호출하지 않는다. 동일한 ZIP의 L4 제한시간 및 Linux seccomp 실행은 별도로 측정한다.

## 실제 ZIP과 측정 결과

2026년 11월 27일 22:17 KST에 후보 ZIP의 실제 진입점에서 175장의 원생성을 재생했다. 독립 포맷과 내용이 모두 일치했고 공식 검사 175/175를 통과했다. 실 GPU로 생성한 공식 키트 10장도 10/10이었고, 훈련측 법령 사례 658개와 오프라인 wheel 5개를 확인했다. 원래 2에폭의 69/138스텝 체크포인트이며, [ZIP·원생성·소스 지문 집계](../results/fold4-v3-epoch1-archive-check.json)와 [로컬 아티팩트 원장](../results/local_artifacts.json)에 기록했다.

| 범위 | Macro-F1 8종 → v3 | 정상 오경보 8종 → v3 | 중대성 4 완전 미탐 8종 → v3 |
|---|---:|---:|---:|
| 전체 175장·39그룹 | 0.52162 → 0.54201 | 10 → 10 / 49 | 32 → 29 / 60 |
| 고정 마스크 107장·36그룹 | 0.54289 → 0.56252 | 6 → 6 / 27 | 20 → 18 / 39 |

그룹 쌍체 95% 구간은 전체 −0.06745~+0.10178, 민감도 −0.09888~+0.06793으로 두 범위 모두 0을 포함했다. F1 이득은 약 0.02이며 사전 승격 기준 +0.03에 미달했다. 정상 오경보율도 두 범위에서 20%를 넘었다. [쌍체 평가 집계](../results/fold4-v3-epoch1-paired-code-evaluation.json)

Windows RTX 3060의 BF16 배치 4에서 적재는 403.87초, Fold 4 원생성 175장은 1,060.7초였다. 출력 토큰 상한 도달과 배치 감소는 0건이었다. 공식 Linux L4의 전체 설치·추론 시간과 seccomp 실실행은 미측정이다.
