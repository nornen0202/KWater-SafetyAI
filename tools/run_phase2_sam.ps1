[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('sam31-frame','sam3-frame','sam3-image','sam3-both','sam-suite')][string]$Arm,
    [int]$OfficialPid,
    [string]$Run='G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce',
    [string]$Legacy='C:/Users/JY/anaconda3/envs/env_aias_test'
)
$ErrorActionPreference='Stop'
$src=Split-Path -Parent $PSScriptRoot
$env:PATH="$Legacy/Library/bin;$Legacy/DLLs;$Legacy;$env:PATH"
$env:PYTHONPATH="$Run/sam-extra;$Run/sam-kernel;$src"
$env:KW_PHASE2_RUN=$Run
$env:PYTHONNOUSERSITE='1'
$env:HF_HUB_OFFLINE='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:TRITON_CACHE_DIR="$Run/sam-triton-cache"
$bundledCuda="$Run/sam-kernel/triton/backends/nvidia"
if((Test-Path -LiteralPath "$bundledCuda/bin/ptxas.exe") -and
   (Test-Path -LiteralPath "$bundledCuda/include/cuda.h") -and
   (Test-Path -LiteralPath "$bundledCuda/lib/x64/cuda.lib")){
    $env:CUDA_PATH=$bundledCuda
    $env:TRITON_PTXAS_PATH="$bundledCuda/bin/ptxas.exe"
}
$env:TORCHINDUCTOR_CACHE_DIR="$Run/sam-inductor-cache"
$env:CUDA_CACHE_PATH="$Run/cuda-cache"
$env:TEMP="$Run/temp"
$env:TMP=$env:TEMP
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
Set-Location -LiteralPath $src
while (!(Test-Path -LiteralPath "$Run/large-models/sam-wheels/runtime.json") -or !(Test-Path -LiteralPath "$Run/sam-extra/timm/__init__.py")) {
    Start-Sleep -Seconds 4
}
$runnerArgs=@('-B','-X','utf8','-u','-m','kwater_lab.phase2_sam3',$Arm)
if($OfficialPid){$runnerArgs+=@('--wait-official-pid',"$OfficialPid")}
& "$Legacy/python.exe" @runnerArgs
exit $LASTEXITCODE
