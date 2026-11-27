[CmdletBinding()]
param([string]$Run='G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce',[int]$Existing8B=57660,[double]$MiBPerSecond=0)
$ErrorActionPreference='Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $env:TEMP=Join-Path $Run 'temp'
    $env:TMP=$env:TEMP
    $env:PYTHONUTF8='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:OMP_NUM_THREADS='1'
    $initial=Get-CimInstance Win32_Process -Filter "ProcessId=$Existing8B"
    if($initial -and $initial.CommandLine -match '-m tools.fetch_phase2_large 8b$'){
        Wait-Process -Id $Existing8B -ErrorAction SilentlyContinue
    }
    foreach($name in '8b','9b-gguf','9b','27b-gguf') {
        if(Test-Path -LiteralPath (Join-Path $Run 'large-models/transfer-pause.flag')){break}
        if(Test-Path -LiteralPath (Join-Path $Run "large-models/$name/complete.json")){continue}
        $extra=@()
        if($MiBPerSecond -gt 0){$extra=@('--mib-s',$MiBPerSecond)}
        & python -B -X utf8 -u -m tools.fetch_phase2_large $name @extra *> (Join-Path $Run "fetch-$name-resumed.log")
        if($LASTEXITCODE){throw "Public weight transfer failed for $name, partials retained"}
    }
} finally {Pop-Location}
