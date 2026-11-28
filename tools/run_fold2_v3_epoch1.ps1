param(
    [string] $Python = 'C:/Users/JY/.antigravity_kwater_venv/Scripts/python.exe',
    [string] $Resume,
    [switch] $RetryBf16OOM
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path
$run = Join-Path $root 'data/v3-11code-fold2-4b-epoch1'
New-Item -ItemType Directory -Path $run -Force | Out-Null
$gpu = [System.Threading.Mutex]::new($false, 'Local\KWater-SafetyAI-Fold0-VLM')
$locked = $false
try {
    try { $locked = $gpu.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $locked = $true }
    if (-not $locked) { throw 'The local K-water GPU mutex is occupied.' }
    $env:PYTHONUTF8 = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:KWATER_LOCAL_BASE_MODEL = Join-Path $root 'data/baseline/base_model'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    foreach ($stage in @('train', 'infer', 'score')) {
        $cli = @('-B', '-u', '-m', 'kwater_lab.fold2_v3_epoch1', $stage)
        if ($stage -eq 'infer' -and $RetryBf16OOM) {
            $cli = @('-B', '-u', '-m', 'kwater_lab.fold2_bf16_oom_recovery')
        }
        if ($stage -eq 'train' -and $Resume) { $cli += @('--resume', ('"{0}"' -f $Resume)) }
        if ($stage -eq 'infer' -and (Test-Path -LiteralPath (Join-Path $run 'epoch1-bf16-inference-receipt.json'))) { continue }
        if ($stage -eq 'score') { $env:CUDA_VISIBLE_DEVICES = '-1' }
        $p = Start-Process -FilePath $Python -ArgumentList $cli -WorkingDirectory $root -WindowStyle Hidden -PassThru `
             -RedirectStandardOutput (Join-Path $run "$stamp-$stage.stdout.log") `
             -RedirectStandardError (Join-Path $run "$stamp-$stage.stderr.log")
        Write-Output "$stage PID $($p.Id)"
        $p.WaitForExit()
        $p.Refresh()
        if ($null -ne $p.ExitCode -and $p.ExitCode -ne 0) { throw "$stage failed with process exit $($p.ExitCode)." }
        if ($stage -eq 'train' -and -not (Test-Path -LiteralPath (Join-Path $run 'epoch-1/training_state.pt'))) { throw 'Missing epoch-1 checkpoint.' }
        if ($stage -eq 'infer' -and -not (Test-Path -LiteralPath (Join-Path $run 'epoch1-bf16-inference-receipt.json'))) { throw 'Missing inference receipt.' }
        if ($stage -eq 'score' -and -not (Test-Path -LiteralPath 'results/fold2-v3-4b-epoch1-20261128.json')) { throw 'Missing aggregate score.' }
    }
} finally {
    if ($locked) { $gpu.ReleaseMutex() }
    $gpu.Dispose()
}
