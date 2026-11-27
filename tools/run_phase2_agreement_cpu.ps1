[CmdletBinding()]
param(
    [string]$Run='G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce',
    [string]$Legacy='C:/Users/JY/anaconda3/envs/env_aias_test'
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $env:KW_PHASE2_RUN=$Run
    $env:CUDA_VISIBLE_DEVICES='-1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:OMP_NUM_THREADS='2'
    $env:PYTHONUTF8='1'
    $env:HF_HUB_OFFLINE='1'
    $env:PATH="$Legacy/Library/bin;$Legacy/DLLs;$Legacy;$env:PATH"
    $env:TEMP=Join-Path $Run 'temp'
    $env:TMP=$env:TEMP
    while(-not (Test-Path "$Run/agreement-heads/linear-predictions.json")){Start-Sleep -Seconds 30}
    $ready=0
    while($ready -lt 2){
        $memory=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
        $drive=Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk | Where-Object Name -match 'G:'
        if($memory.AvailableMBytes -ge 1400 -and $drive.CurrentDiskQueueLength -le 6){$ready++}else{$ready=0}
        if($ready -lt 2){Start-Sleep -Seconds 45}
    }
    if(-not (Test-Path "$Run/agreement-heads/shared-predictions.json")){
        & "$Legacy/python.exe" -B -u -m kwater_lab.phase2_agreement_heads spatial
        if($LASTEXITCODE){throw "Spatial refit exited with $LASTEXITCODE"}
    }
    foreach($v in @('fp32','bf16')){
        & python -B -u -m tools.phase2_image_towers $v
        if($LASTEXITCODE){throw "Image weight conversion exited with $LASTEXITCODE"}
    }
} finally {Pop-Location}
