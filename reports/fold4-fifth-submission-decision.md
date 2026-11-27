# Fold 4 실행 상태와 5차 제출 판단

2026년 11월 27일 17:08 KST 확인 기준, **5차 제출을 보류한다.** Fold 4 8종 대조군 학습은 완료됐다. v3 학습은 8/154스텝 이후 프로세스가 종료됐고, 독립 비교에 필요한 세 예측 JSONL과 5차 후보 ZIP은 확인되지 않았다.

## 확인한 산출물

| 조건 | 상태 |
|---|---|
| 새 8종 대조군 | 2에폭·100/100스텝 완료. 개발 Fold 0–3의 원본 791장 중 8종 변환을 통과한 790장으로 학습했다. 훈련 요약과 일치한 최종 어댑터 파일의 SHA-256은 `00386e3407ef361738341ab38091f43b9f7191021b73532a2b06c023c04c4bf4`다. |
| v3 11종 | 최종 스텝 로그는 16:57:40 KST의 8/154였다. 확인된 마지막 재개 지점은 `checkpoint-2`이며 어댑터와 훈련 상태 파일이 있다. 17:07 이후 해당 훈련·평가 프로세스와 최종 훈련 요약을 확인하지 못했다. 17:05:04 KST에 NVIDIA 드라이버 System 이벤트 153이 기록됐다. 이 이벤트와 프로세스 종료의 인과관계와 종료 코드는 미확인이다. [실행 로그와 재개 지점](fold4-live-training-20261127.md) |
| 세 모델의 Fold 4 예측 | `clean_8code_baseline.jsonl`, `v3_11code_standalone.jsonl`, `v3_hybrid_margin_verified.jsonl` 모두 미생성. 예측 지문과 공식 검사 통과 행 수는 미측정이다. |
| 5차 ZIP | `data/submissions/`에서 새 Fold 4 v3 후보를 확인하지 못했다. ZIP 무결성·해시와 Linux 오프라인 실행·소요 시간은 미측정이다. |

## 고정 검증 패널

| 패널 | 사진 | 그룹 | 정답 정상 사진 | 심각도 4 정답 사진 | 심각도 4 정답 코드 건수 |
|---|---:|---:|---:|---:|---:|
| 주평가 | 175 | 39 | 49 | 60 | 61 |
| 동일 그룹 제외 민감도 | 107 | 36 | 27 | 39 | 39 |

107장 패널은 시각적으로 확인된 세 교차 현장의 검증 그룹을 동일하게 제외한다. 잔존 최대 그룹 49장을 추가로 빼는 58장·35그룹 비교는 탐색 분석이다. 세 모델의 패널별 활성 Macro-F1·정상 오경보·중대 위험 완전 미탐, 두 패널의 세 쌍씩 ΔF1과 그룹 부트스트랩은 모두 **미측정**이다. [교차 현장 감사](../docs/fold4-crosssplit-visual-audit-20261127.md)

## 복구 후 검증 순서

1. 훈련 담당자가 기존 실행 상태와 복구 여부를 확인한다. 동일 실험의 완료 어댑터와 세 모델의 원래 175개 사진 출력이 확보되면 각 출력의 해시를 기록한다. 공식 검사에는 세 파일에 동일한 32자리 익명 ID 매핑을 적용하고, 각 조건의 유효 행 175/175와 오류 0건을 확인한다. [Fold 4 공식 검사 CLI](../docs/fold4-official-output-check-20261127.md)
2. 아래 CPU 채점 명령으로 `results/fold4-paired-code-evaluation.json`을 생성한다. 원래 175개 ID를 조건별로 검증한 뒤 동일 마스크를 적용한다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& C:/Users/JY/anaconda3/envs/env_aias_test/python.exe -m kwater_lab.fold4_crosssplit_audit score-three
```

3. 두 패널에서 새 8종 대비 ΔF1 ≥+0.03, 정상 오경보 ≤20%(각각 9/49, 5/27 이하), 심각도 4 완전 미탐 비증가를 대조한다. 코드 변경률 10%는 변화 규모의 참고값이다. 이전 계획의 ΔF1 ≥+0.05·변경률 ≥25%·무효 0건 기준은 별도로 기록한다. 패키지가 도착하면 ZIP의 예측 경로가 평가와 일치하는지 검증한다. [최신 Fold 4 판단 기준](safeground-v3-review-parallel-roadmap-20261127.md) · [이전 제출 관문](../docs/top10-turnaround-audit-and-plan-2026-11-26.md) · [ZIP 검산](../docs/fifth-archive-offline-audit-20261127.md)
