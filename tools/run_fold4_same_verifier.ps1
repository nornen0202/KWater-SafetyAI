param(
    [Parameter(Mandatory = $true)]
    [string] $Python
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$run = Join-Path $root 'data/local-eval-predictions/fold4-same-verifier'
$gpuLock = [System.Threading.Mutex]::new($false, 'Local\KWater-SafetyAI-Fold0-VLM')
$held = $false
try {
    try { $held = $gpuLock.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $held = $true }
    if (-not $held) { throw 'The local VLM GPU mutex is occupied. Requeue after the current owner releases it.' }
    $env:PYTHONUTF8 = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $started = Get-Date
    $process = Start-Process -FilePath $pythonPath -ArgumentList @('-B', '-u', '-m', 'kwater_lab.fold4_same_verifier', 'verify') `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $run "verify-$stamp.stdout.log") `
        -RedirectStandardError (Join-Path $run "verify-$stamp.stderr.log")
    $process.WaitForExit()
    $process.Refresh()
    $receipt = Join-Path $run 'completed.json'
    if (($null -ne $process.ExitCode -and $process.ExitCode -ne 0) -or -not (Test-Path -LiteralPath $receipt)) {
        throw "Verification process did not complete. See verify-$stamp logs."
    }
    if ((Get-Item -LiteralPath $receipt).LastWriteTime -lt $started) {
        throw "Verification did not publish a fresh completion receipt. See verify-$stamp logs."
    }
}
finally {
    if ($held) { $gpuLock.ReleaseMutex() }
    $gpuLock.Dispose()
}
