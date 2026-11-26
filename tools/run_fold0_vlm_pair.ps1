param(
    [int]$TrainingPid = 0,
    [string]$RunName = 'fold0-qlora-118',
    [string]$VlmPython = 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe',
    [string]$BaselineRuntime = 'C:/Users/JY/anaconda3/envs/env_aias_test',
    [string]$TrainDir = 'data/vlm-11code-fold0',
    [switch]$SkipZeroShot
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
if ($RunName -cnotmatch '^[a-z0-9][a-z0-9-]+$') {
    throw 'RunName must use lowercase alphanumerics and hyphens.'
}
$gpuLock = [System.Threading.Mutex]::new($false, 'Local\KWater-SafetyAI-Fold0-VLM')
try {
    $acquired = $gpuLock.WaitOne(0)
}
catch [System.Threading.AbandonedMutexException] {
    $acquired = $true
}
if (-not $acquired) { throw 'The fold-0 VLM runner is already active.' }

$manifest = 'data/clean-split-v1/split_manifest_dev.json'
$baselineRun = "$RunName-official"
$baselineRunDir = "data/$baselineRun"
$baselineInputDir = 'data/fold0-production-input'
$baselinePredictions = "$baselineRunDir/output/submission.jsonl"
$baselineReport = "results/$($baselineRun.Replace('-', '_')).json"
$adapterDir = Join-Path $TrainDir 'adapter'
$firstHistory = Join-Path $TrainDir 'history.json'
$firstEpochAudit = Join-Path $TrainDir 'epoch-1-adapter-audit.json'

if ($TrainingPid) {
    $trainer = Get-CimInstance Win32_Process -Filter "ProcessId=$TrainingPid"
    if ($trainer -and $trainer.CommandLine -notmatch '(?:-m\s+kwater_lab\.vlm_11code)\s+--mode\s+train') {
        throw 'TrainingPid belongs to a different process.'
    }
    if ($trainer) {
        Write-Output "Waiting for QLoRA training process $TrainingPid"
        while ((Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) -and -not (Test-Path -LiteralPath $firstEpochAudit)) {
            if (Test-Path -LiteralPath $firstHistory) {
                $firstCheckpoint = Join-Path $TrainDir 'epoch-1'
                $env:OPENBLAS_NUM_THREADS = '1'
                $env:OMP_NUM_THREADS = '1'
                $audit = & $VlmPython -B tools/check_qwen3vl_adapter.py --adapter-dir $firstCheckpoint
                if ($LASTEXITCODE -ne 0) { throw 'First-epoch adapter validation failed.' }
                $audit | Set-Content -LiteralPath $firstEpochAudit -Encoding Ascii
                Write-Output "Validated first-epoch adapter with CPU: $firstEpochAudit"
                break
            }
            Start-Sleep -Seconds 30
        }
        if (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
            Wait-Process -Id $TrainingPid -ErrorAction SilentlyContinue
        }
    }
}

$training = Get-Content -LiteralPath (Join-Path $TrainDir 'training_summary.json') -Raw | ConvertFrom-Json
$expectedSteps = [Math]::Ceiling($training.curriculum_examples / $training.grad_accum) * $training.epochs
$adapterHash = (Get-FileHash -LiteralPath (Join-Path $adapterDir 'adapter_model.safetensors') -Algorithm SHA256).Hash
if ($training.val_fold -ne 0 -or $training.optimizer_steps -ne $expectedSteps -or $adapterHash -ne $training.adapter_safetensors_sha256) {
    throw 'Expected a completed fold-0 training summary with its matching adapter.'
}

$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$finalAudit = & $VlmPython -B tools/check_qwen3vl_adapter.py --adapter-dir $adapterDir
if ($LASTEXITCODE -ne 0) { throw 'Final adapter validation failed.' }
$finalAudit | Set-Content -LiteralPath (Join-Path $TrainDir 'final-adapter-audit.json') -Encoding Ascii

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:CUDA_CACHE_PATH = Join-Path (Get-Location).Path '.runtime/cuda-cache'
$env:HF_HOME = Join-Path (Get-Location).Path '.runtime/hf'

function Invoke-Fold0Vlm([string]$Precision, [switch]$CompareZeroShot) {
    $evaluation = "$RunName-$Precision"
    $predictionDir = "data/local-eval-predictions/fold0/$evaluation"
    $stage = if ($CompareZeroShot) { 'zeroshot_ablation' } else { 'vlm' }
    $metricsPath = "results/$($evaluation.Replace('-', '_'))_$stage.json"
    $attempt = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $TrainDir "$evaluation-$stage-$attempt.stdout.log"
    $stderr = Join-Path $TrainDir "$evaluation-$stage-$attempt.stderr.log"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        $arguments = @('-B', '-u', '-m', 'kwater_lab.hybrid_pipeline', '--split', 'fold0', '--manifest', $manifest, '--skip-verifier', '--adapter-dir', ('"' + $adapterDir + '"'), '--out', $metricsPath, '--prediction-dir', $predictionDir, '--batch-size', '1')
        if (-not $CompareZeroShot) { $arguments += '--skip-zeroshot' }
        if ($Precision -eq 'bf16') { $arguments += '--bf16' }
        Write-Host "Starting $Precision $stage evaluation. See $stdout and $stderr."
        $worker = Start-Process -FilePath $VlmPython -ArgumentList $arguments -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
        if ($worker.ExitCode -ne 0) {
            if (-not $CompareZeroShot -and $Precision -eq 'bf16' -and (Select-String -LiteralPath $stderr -Pattern '^torch(?:\.cuda)?\.OutOfMemoryError:|CUDA (?:error: )?out of memory' -Quiet)) {
                Write-Host 'BF16 CUDA OOM. Continuing with the separate NF4 evaluation.'
                return $null
            }
            throw "Fine-tuned $Precision fold-0 inference failed. See $stderr."
        }
    }
    return @{
        Precision = $Precision
        Metrics = $metricsPath
        Prediction = "$predictionDir/ft_vlm_11code.jsonl"
        Pair = "results/$($evaluation.Replace('-', '_'))_paired.json"
        Qualitative = "results/$($evaluation.Replace('-', '_'))_qualitative_proxies.json"
        ReviewRoot = "$predictionDir/blind-review"
    }
}

$bf16Oom = Join-Path $TrainDir "$RunName-bf16-oom.txt"
$oomAlreadyObserved = (Test-Path -LiteralPath $bf16Oom) -and ((Get-Content -LiteralPath $bf16Oom -Raw).Trim() -eq $adapterHash)
$evaluation = $null
if (-not $oomAlreadyObserved) {
    $evaluation = Invoke-Fold0Vlm -Precision 'bf16'
    if (-not $evaluation) { $adapterHash | Set-Content -LiteralPath $bf16Oom -Encoding Ascii }
}
if (-not $evaluation) { $evaluation = Invoke-Fold0Vlm -Precision 'nf4' }
$ftReport = Get-Content -LiteralPath $evaluation.Metrics -Raw | ConvertFrom-Json
$manifestHash = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash
$expectedWeightMode = if ($evaluation.Precision -eq 'bf16') { 'BF16 base' } else { 'NF4 language / BF16 vision' }
if ($ftReport.adapter_sha256 -ne $adapterHash -or $ftReport.manifest_sha256 -ne $manifestHash -or $ftReport.qlora_inference_weight_mode -ne $expectedWeightMode) {
    throw 'Fine-tuned metrics use a different adapter, fold manifest, or precision.'
}

if (-not (Test-Path -LiteralPath $baselineReport)) {
    $priorPythonPath = $env:PYTHONPATH
    $priorPath = $env:PATH
    try {
        $env:PYTHONPATH = (Get-Content -LiteralPath '.runtime/legacy-overlay-path.txt' -Raw).Trim()
        $env:PATH = "$BaselineRuntime/Library/bin;$BaselineRuntime/DLLs;$BaselineRuntime;$priorPath"
        & "$BaselineRuntime/python.exe" -B -u -m kwater_lab.fold0_official_resume --input-dir $baselineInputDir --run-dir $baselineRunDir --batch-size 1 --out $baselineReport
        if ($LASTEXITCODE -ne 0) { throw 'Official baseline inference failed.' }
    }
    finally {
        $env:PYTHONPATH = $priorPythonPath
        $env:PATH = $priorPath
    }
}
$official = Get-Content -LiteralPath $baselineReport -Raw | ConvertFrom-Json
$officialInputHash = (Get-FileHash -LiteralPath "$baselineInputDir/manifest.jsonl" -Algorithm SHA256).Hash
$officialOutputHash = (Get-FileHash -LiteralPath $baselinePredictions -Algorithm SHA256).Hash
if ($official.official_checker_valid_rows -ne $ftReport.photos -or $official.source_manifest_sha256 -ne $officialInputHash -or $official.output_sha256 -ne $officialOutputHash) {
    throw 'Expected complete official predictions validated by the official checker.'
}

& $VlmPython -B -u -m kwater_lab.fold0_production_compare --manifest $manifest compare --baseline $baselinePredictions --candidate $evaluation.Prediction --output $evaluation.Pair
if ($LASTEXITCODE -ne 0) { throw 'Paired comparison failed.' }
Write-Output "Paired fold-0 results saved to $($evaluation.Pair)"

if (-not (Test-Path -LiteralPath $evaluation.Qualitative)) {
    $review = $evaluation.ReviewRoot
    if (Test-Path -LiteralPath $review) {
        $review += '-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    }
    & $VlmPython -B -u -m kwater_lab.fold0_qualitative_review --manifest $manifest --baseline $baselinePredictions --candidate $evaluation.Prediction --local-output $review --summary $evaluation.Qualitative
    if ($LASTEXITCODE -ne 0) { throw 'Blinded review preparation failed.' }
    Write-Output "Blinded 36-photo review prepared in $review"
}

if (-not $SkipZeroShot) {
    $ablation = Invoke-Fold0Vlm -Precision $evaluation.Precision -CompareZeroShot
    Write-Output "Same-precision zero-shot ablation saved to $($ablation.Metrics)"
}
$gpuLock.ReleaseMutex()
$gpuLock.Dispose()
