# 다른 PC에서 연구 이어가기

Google Drive의 내 드라이브에서 KWater-SafetyAI를 동기화하고 파일 다운로드가 끝난 뒤 PowerShell에서 이 폴더를 연다. 작업 체크아웃은 Drive 밖에 둔다.

```powershell
$backup = (Get-Location).Path
$checkout = 'G:\Projects\KWater-SafetyAI'
$cache = 'G:\CodexData\kwater-rd-20261125'
$archive = Get-ChildItem -LiteralPath $backup -Filter 'research-*.bundle' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
git clone -b main $archive.FullName $checkout
git -C $checkout remote set-url origin git@github.com:nornen0202/KWater-SafetyAI.git
git -C $checkout pull --ff-only
Copy-Item -LiteralPath (Join-Path $backup 'data') -Destination $checkout -Recurse -Force
New-Item -ItemType Directory -Force -Path $cache | Out-Null
Copy-Item -Path (Join-Path $backup 'shared-cache\*') -Destination $cache -Recurse -Force
```

이미 체크아웃이 있으면 clone을 건너뛴다. 오프라인에서는 pull을 생략한다. 프로젝트 README의 환경 설정과 실험별 실행 명령을 따른다. 파일 검증은 PowerShell에서 다음과 같이 실행한다.

```powershell
$records = @((Get-Content .\SHA256.json -Raw | ConvertFrom-Json).files)
$records += @(Get-Content .\SHA256-cache.json -Raw | ConvertFrom-Json)
$records += @(Get-Content .\SHA256-reference.json -Raw | ConvertFrom-Json)
foreach ($item in $records) {
    $hash = (Get-FileHash -LiteralPath $item.path -Algorithm SHA256 -ErrorAction Stop).Hash
    if ($hash -ne $item.sha256) { throw "검증 실패 $($item.path)" }
}
```

## 국내에서 별도로 복원하는 자료

- 대회 참가 계정으로 https://hackathon.imaginationgroup.co.kr/kwater/ 에서 kwater-train.zip, kwater-local-kit.zip, kwater-baseline.zip을 받아 data/packages에 둔다. results/package_manifest.json으로 확인하고 README에 따라 각각 data/train, data/local-kit, data/baseline에 푼다.
- AI-Hub 승인 계정에서 자료 71407, 71793, 507을 국내 PC에 직접 받는다. 승인 자료의 원본과 라벨, 대회 사진·정답·개별 예측·특징 캐시는 국내 G:에서 관리한다. 허용된 국내 작업 PC로 원본을 옮길 때 results/local_artifacts.json의 상대 경로와 SHA-256을 사용한다. 원장은 작성 시점의 완료 파일을 기록한다.
- FloodNet은 BinaLab의 FloodNet-Supervised v1.0이며 public-data/floodnet-pilot에 출처와 CDLA-Permissive-1.0 고지가 있다. 국토안전관리원 자료의 원본 출처는 https://www.data.go.kr/data/15142200/fileData.do 이다. 고용노동부·안전보건공단 참고 문서는 제공자 원본 그대로 보관한다. 자료별 출처는 reports/external-first-experiment.md와 동봉 JSON에 있다.
- Git에는 실험 코드·보고서·집계가 들어 있다. 두 번째 제출본은 data/submissions에 있으며 제출 당시의 SHA-256은 22e32c10f56c4dec50a9365ee59c16b79439fa28787132d6f2fcfa500070eaa0이다.
