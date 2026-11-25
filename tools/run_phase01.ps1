[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Run,
    [string]$Project = 'G:/Projects/KWater-SafetyAI',
    [string]$Runtime = 'C:/Users/JY/anaconda3/envs/env_aias_test',
    [string]$ReviewedGroups,
    [ValidateRange(1,500)][int]$Epochs = 14
)
$ErrorActionPreference = 'Stop'
$src = Split-Path -Parent $PSScriptRoot
$runPath = [IO.Path]::GetFullPath($Run)
$python = Join-Path $Runtime 'python.exe'
$checker = Join-Path $Project '.runtime/baseline-venv/Scripts/python.exe'
[IO.Directory]::CreateDirectory($runPath) | Out-Null
$lock = [IO.File]::Open((Join-Path $runPath 'run.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
Push-Location -LiteralPath $src
try {
    $env:PYTHONUTF8='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:OMP_NUM_THREADS='2'
    $env:MKL_NUM_THREADS='2'
    $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
    if (-not (Test-Path -LiteralPath (Join-Path $runPath 'models/sig/model.safetensors')) -or
        -not (Test-Path -LiteralPath (Join-Path $runPath 'models/dino/model.safetensors'))) {
        python -B -u tools/fetch_phase01_backbones.py (Join-Path $runPath 'models')
        if ($LASTEXITCODE) { throw "Backbone acquisition failed ($LASTEXITCODE)" }
    }
    $env:PATH="$Runtime/Library/bin;$Runtime/DLLs;$Runtime;$env:PATH"
    $env:PYTHONPATH=[IO.File]::ReadAllText((Join-Path $Project '.runtime/legacy-overlay-path.txt'))
    $env:HF_HOME=Join-Path $runPath 'hf'
    $env:CUDA_CACHE_PATH=Join-Path $runPath 'cuda-cache'
    $env:HF_HUB_OFFLINE='1'
    if (-not (Test-Path -LiteralPath (Join-Path $runPath 'data-summary.json'))) {
        $extra = @()
        if ($ReviewedGroups) { $extra=@('--reviewed',$ReviewedGroups) }
        & $python -B -u -m kwater_lab.phase01_data --project $Project --out $runPath @extra
        if ($LASTEXITCODE) { throw "Data audit failed ($LASTEXITCODE)" }
    } else {
        $audit = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $runPath 'data-summary.json') | ConvertFrom-Json
        $reviewHash = if ($ReviewedGroups) { (Get-FileHash -LiteralPath $ReviewedGroups).Hash } else { $null }
        if ($audit.reviewed_sha256 -ne $reviewHash) { throw 'Reviewed groups changed. Choose a new run directory.' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $runPath 'completed.json'))) {
        & $python -B -u -m kwater_lab.phase01_experiment --project $Project --run $runPath --checker-python $checker --epochs $Epochs
        if ($LASTEXITCODE) { throw "Experiment failed ($LASTEXITCODE)" }
    } else {
        $complete = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $runPath 'completed.json') | ConvertFrom-Json
        if ($complete.epochs -ne $Epochs) { throw 'Epoch count changed. Choose a new run directory.' }
    }
    & $python -B -u -m kwater_lab.phase01_verify --project $Project --run $runPath --checker-python $checker
    if ($LASTEXITCODE) { throw "OOF verification failed ($LASTEXITCODE)" }
} finally {
    Pop-Location
    $lock.Dispose()
}
