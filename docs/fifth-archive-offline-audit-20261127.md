# 5차 제출 아카이브의 오프라인 검산 준비

점검 시점에 `data/submissions/`에는 5차 후보 ZIP이 없었다. CPU 검산은 대회가 제공한 기존 원본 ZIP과 이미 만들어 둔 v2 Fold0 평가 디렉터리에서 실행했다. 5차 후보의 결과나 접수 적합성은 미측정이다. [CPU 검사 도구](../kwater_lab/offline_archive_audit.py) · [검사 증거 집계](../results/fifth_archive_cpu_readiness.json)

| 확인 항목 | 공식 기존 ZIP | v2 평가용 디렉터리 |
| --- | ---: | ---: |
| ZIP 바이트 | 9,032,392,642 | 디렉터리 |
| 파일 수 / 논리 크기 | 47 / 9,056,261,772 B | 28 / 9,043,642,968 B |
| 공식 자산 신고 검사 | 시드 42, 잠금 파일 있음 | 시드 42, 잠금 파일 있음 |
| Linux x86-64 Python 3.12 태그·SHA 일치 wheel | 5 | 5 |
| 기존 출력 공식 스키마 검사 | 10/10 | 동봉 실제 출력 없음 |

v2 평가 디렉터리는 `assets.json`에 기재된 어댑터 SHA-256도 대조했다. 둘 다 원본 공식 학습 JSONL·학습 ZIP과 크기가 같은 항목을 바이트로 대조했으며 원본 일치 항목을 발견하지 않았다. 공식 기존 ZIP이 동봉한 자체 파생 사례집 `assets/train.jsonl`은 파일명 검토 대상으로 남겼다. v2 디렉터리에 저장물 이름 검사에서 발견된 사진·원문·로그 후보는 없었다. 공식 기존 ZIP에는 두 `subprocess` 호출이 정적 검사에 잡혔다. 실제 실행 여부와 네트워크 차단 적합성은 정적 호출만으로 판정할 수 없다. 기존 모델 실행 로그는 547바이트, 최대 행 339바이트였다.

독립 검사는 ZIP 중앙 디렉터리에서 위험 경로, 중복·경로 충돌, 심볼릭 링크, 암호화 및 크기를 확인한다. 공식 `zipcheck.manifest`로 `assets.json`·잠금 파일만 임시 점검하므로 가중치를 통째로 풀지 않는다. 대회 공지의 ZIP 20GB 상한과 키트의 해제 크기 64GiB 상한을 적용한다. ZIP 내부 모든 바이트의 CRC는 `--stream-crc`를 따로 지정했을 때만 읽는다. 원본 혼입 검사는 알려진 원본의 동일 바이트와 경로·확장자 단서를 확인한다. 변형 또는 재인코딩된 파일의 동일성까지 판단하지 않는다. 후보가 나오면 공개 기록에는 집계만 남기고 내부 경로를 담은 검산 JSON은 로컬 `data/`에 보존한다.

공식 Linux 실행 키트는 `--network none`, 읽기 전용 루트, 6 CPU·56GiB RAM·32GiB scratch와 seccomp를 적용하며, 설치를 포함해 Public 1,800초 또는 Private 3,600초를 검사한다. 이번 Windows CPU 검사에서 실제 Linux 설치, GPU 추론, 네트워크·seccomp 제한 및 시간 상한은 실증하지 않았다. 작은 합성 ZIP을 대상으로 공식 `check_zip.py`의 정상 통과와 위험 경로 거부를 별도로 확인했다. 대형 ZIP은 압축을 풀거나 복제하지 않았다.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python -B -m kwater_lab.offline_archive_audit --archive $CandidateZip `
  --out data/local-reports/fifth-archive-cpu.json
```

이미 같은 후보의 결과가 있는 경우에만 `--input-dir`, `--existing-output`, `--checker-python`을 함께 사용한다. 마지막 인수는 `jsonschema`가 설치된 현재 로컬 Python을 가리킨다. 기존 실행 로그가 있으면 `--runtime-log`로 원문을 보존한 채 전체 바이트 및 긴 줄 수를 검사한다. 배포 승인에 필요한 Linux 실실행은 같은 후보의 ZIP으로 따로 확인한다. 메모리 가용 약 1.17GiB에서 v3 GPU 작업이 진행 중이어서 이 체크아웃의 전체 PyTorch 시험은 보류했다. 예전에 통과한 84개 전체 시험과 이번 아카이브 도구의 표준 라이브러리 회귀는 다른 실행이다.
