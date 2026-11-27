# Fold 4 공식 출력 형식 검사

세 예측 JSONL이 준비된 뒤 저장소 루트에서 실행한다. Python 3.11 이상과 공식 키트가 필요하며, 검사기를 실행하는 Python에는 `jsonschema`가 설치되어 있어야 한다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
python -m kwater_lab.fold4_official_output_check --workspace data/local-eval-predictions/fold4-official-check-run1 --out results/fold4-official-output-check.json
```

기본 입력은 `data/local-eval-predictions/fold4-v3-benchmark/`의 `clean_8code_baseline.jsonl`, `v3_11code_standalone.jsonl`, `v3_hybrid_margin_verified.jsonl`이다. 다른 위치는 `--predictions`, 별도 검사기 환경은 `--checker-python`으로 지정한다. 세 파일이 각각 개발 manifest의 Fold 4 원본 ID 175개를 중복 없이 포함해야 한다. 각 행에서 `image_id`만 같은 결정적 32자리 16진수로 치환한다. 키 매핑, 공통 입력 manifest와 변환한 제출 행은 새 로컬 `--workspace`에 저장된다. 파일 재실행에는 새 작업 폴더와 집계 파일 경로를 사용한다.

공식 `check_output.py`의 종료 코드가 0이고 유효 행이 175/175일 때 해당 조건을 통과로 기록한다. 세 조건 중 하나라도 실패하면 집계를 쓴 뒤 종료 코드 1을 반환한다. 집계에는 입력·치환본·검사기·스키마 SHA-256과 유효 행 수를 기록한다. 세 조건 공식 출력 형식 검사 후 동일 107장 마스크를 적용해 별도 점수 집계를 실행한다. ZIP 검사와 배포 Linux 실실행은 각각 별도 단계다.
