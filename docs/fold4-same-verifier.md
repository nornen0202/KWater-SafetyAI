# Fold 4에서 초벌만 바꾼 검증기 비교

6차에 사용한 1에폭 초벌과 정규 2에폭 초벌을 고정한다. 두 조건에 동일한 정규 2에폭 검증 어댑터·시스템 프롬프트·BF16·이미지 크기 제한 393216·배치 2·검증 생성 상한 280을 적용한다. 내부 검증으로 이미 고정한 beta 0.6, 위험 초벌 tau −0.9, 무위험 초벌 tau 1.489를 사용한다.

CPU에서 재현한 제안은 1에폭 54개, 2에폭 66개이고 합집합은 사진·코드 72쌍이다. 원 2에폭은 검증 행별 마진·응답을 보존하지 않아 다시 측정한다. 2에폭의 배치 순서를 재현한 뒤 새 1에폭 제안만 추가로 읽는다. 같은 사진과 코드의 마진과 생성 응답은 공유한다. 초벌이 비어 있는지에 따라 통과 기준은 각 조건에 별도로 적용한다.

## CPU 계획 확인

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python -B -m kwater_lab.fold4_same_verifier plan
```

## GPU 소유자 실행

Windows의 별도 GPU 셸에서 정규 2에폭 평가의 Python과 로컬 베이스 모델 환경을 그대로 사용한다. 위 CPU 확인에서 사용한 `CUDA_VISIBLE_DEVICES=-1`은 GPU 실행 환경에 전달하지 않는다. 기존 GPU 작업을 완료하고 실행 가능 상태에서 아래를 예약한다.

```powershell
pwsh -NoProfile -File tools/run_fold4_same_verifier.ps1 -Python "C:/path/to/original/python.exe"
```

표준출력·오류와 원자적으로 갱신하는 마진·생성 캐시는 모두 `data/local-eval-predictions/fold4-same-verifier/`에 남는다. 같은 입력·런타임으로 다시 실행하면 완료한 배치는 유지한다. 원래 2에폭 하이브리드의 175행 객체 및 통과·수용 건수가 일치해야 두 후보의 JSONL과 완료표를 기록한다. 불일치 시 오류 로그를 남겨 비교를 중단한다.

로컬 출력 두 파일은 `epoch1_fixed_epoch2_verifier.jsonl`, `epoch2_fixed_epoch2_verifier.jsonl`이다. 완료표는 `completed.json`이다. 개발 표본 175장과 고정 민감도 107장 채점은 별도로 수행한다. GPU 실행과 대회 제출은 별개다.
