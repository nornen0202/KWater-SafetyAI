[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('Freeze','Cache','Contracts','Heads','Nested','Prior','External','Official','Train3VL','Train35','TrainBoth','Queue','Static','Score')][string]$Stage,
    [string]$Project='G:/Projects/KWater-SafetyAI',
    [string]$Run='G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce',
    [string]$Legacy='C:/Users/JY/anaconda3/envs/env_aias_test',
    [string]$TrainingPython,
    [int]$OfficialPid,
    [int]$SamWrapperPid,
    [int]$PreviousQueuePid,
    [int]$DeferQueueUntilSamPid
)
$ErrorActionPreference='Stop'
$src=Split-Path -Parent $PSScriptRoot
$names=@('PATH','PYTHONPATH','PYTHONUTF8','KW_PHASE2_RUN','KW_PHASE2_PROJECT','HF_HOME','HF_HUB_OFFLINE',
         'HF_HUB_DISABLE_TELEMETRY','CUDA_CACHE_PATH','TORCH_HOME','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
         'MKL_NUM_THREADS','TEMP','TMP','TOKENIZERS_PARALLELISM','PYTHONNOUSERSITE','CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF')
$old=@{}
foreach($n in $names){$old[$n]=[Environment]::GetEnvironmentVariable($n)}
Push-Location -LiteralPath $src
try{
    $env:KW_PHASE2_RUN=$Run
    $env:KW_PHASE2_PROJECT=$Project
    $env:HF_HOME=Join-Path $Run 'hf'
    $env:TORCH_HOME=Join-Path $Run 'torch-cache'
    $env:CUDA_CACHE_PATH=Join-Path $Run 'cuda-cache'
    $env:HF_HUB_OFFLINE='1'
    $env:HF_HUB_DISABLE_TELEMETRY='1'
    $env:PYTHONUTF8='1'
    $env:PYTHONNOUSERSITE='1'
    $env:OPENBLAS_NUM_THREADS='1'
    $env:OMP_NUM_THREADS='2'
    $env:MKL_NUM_THREADS='2'
    $env:TOKENIZERS_PARALLELISM='false'
    $env:TEMP=Join-Path $Run 'temp'
    $env:TMP=$env:TEMP
    [IO.Directory]::CreateDirectory($env:TEMP) | Out-Null
    $modern=Join-Path $Run 'venv/Scripts/python.exe'
    $bridge=Join-Path $Run 'bridge/Scripts/python.exe'
    $bridgeProof=Join-Path $Run 'bridge-import.log'
    if($TrainingPython){$modern=$TrainingPython}
    elseif((Test-Path -LiteralPath $bridgeProof) -and (Select-String -LiteralPath $bridgeProof -SimpleMatch 'both_architectures_available' -Quiet)){$modern=$bridge}
    if($modern -eq $bridge -and $Stage -in 'Train3VL','Train35','TrainBoth','Queue','Static'){
        $env:PATH="$Legacy/Library/bin;$Legacy/DLLs;$Legacy;$env:PATH"
    }
    function Invoke-Study([string]$Python, [string[]]$Arguments){
        & $Python -B -u @Arguments
        if($LASTEXITCODE){throw "$Stage exited with $LASTEXITCODE"}
    }
    switch($Stage){
        'Freeze' {
            Invoke-Study 'python' @('-m','kwater_lab.phase2_panel','project-development')
            Invoke-Study 'python' @('-m','kwater_lab.phase2_panel','freeze')
            Invoke-Study 'python' @('-m','kwater_lab.phase2_panel','sft-split')
        }
        'Cache' {Invoke-Study 'python' @('-m','kwater_lab.phase2_panel','cache')}
        'Contracts' {
            Invoke-Study 'python' @("$Project/.runtime/submission2-zipcheck-final/main.py",'--input-dir',"$Project/data/local-kit/sample-input",'--output-dir',"$Run/fusion-kit")
            Invoke-Study 'python' @('-m','kwater_lab.phase2_contract')
        }
        'Heads' {
            Invoke-Study 'python' @('-m','kwater_lab.phase2_panel','dev')
            Invoke-Study 'python' @('-m','kwater_lab.phase2_donors')
        }
        'Nested' {Invoke-Study 'python' @('-m','kwater_lab.phase2_nested')}
        'Prior' {
            $env:PATH="$Legacy/Library/bin;$Legacy/DLLs;$Legacy;$env:PATH"
            $env:CUDA_VISIBLE_DEVICES='-1'
            Invoke-Study "$Legacy/python.exe" @('-m','kwater_lab.phase2_prior')
        }
        'External' {Invoke-Study 'python' @('-m','kwater_lab.phase2_external')}
        'Official' {
            $env:PATH="$Legacy/Library/bin;$Legacy/DLLs;$Legacy;$env:PATH"
            $env:PYTHONPATH=[IO.File]::ReadAllText("$Project/.runtime/legacy-overlay-path.txt").Trim()
            $env:PYTORCH_CUDA_ALLOC_CONF='garbage_collection_threshold:0.80,max_split_size_mb:128'
            if($OfficialPid){
                $oldProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$OfficialPid"
                if($oldProcess){
                    if($oldProcess.CommandLine -notmatch 'kwater_lab\.phase2_official'){throw 'Expected the previous official runner'}
                    Wait-Process -Id $OfficialPid -ErrorAction SilentlyContinue
                }
            }
            $officialArgs=@('-m','kwater_lab.phase2_official')
            if($SamWrapperPid){$officialArgs+=@('--wait-sam-wrapper',"$SamWrapperPid")}
            Invoke-Study "$Legacy/python.exe" $officialArgs
        }
        'Train3VL' {Invoke-Study $modern @('-m','kwater_lab.phase2_lora','3vl')}
        'Train35' {Invoke-Study $modern @('-m','kwater_lab.phase2_lora','35','--series','--external')}
        'TrainBoth' {Invoke-Study $modern @('-m','kwater_lab.phase2_lora','both','--series','--external')}
        'Queue' {
            if(-not $OfficialPid){throw 'Pass the PID of the official generation process'}
            if($DeferQueueUntilSamPid){
                $reserved=Get-CimInstance Win32_Process -Filter "ProcessId=$DeferQueueUntilSamPid"
                if($reserved){
                    if($reserved.CommandLine -notmatch 'run_phase2_sam\.ps1'){throw 'Expected the current SAM wrapper'}
                    Wait-Process -Id $DeferQueueUntilSamPid -ErrorAction SilentlyContinue
                }
            }
            if($PreviousQueuePid){
                $previous=Get-CimInstance Win32_Process -Filter "ProcessId=$PreviousQueuePid"
                if($previous){
                    if($previous.CommandLine -notmatch 'tools\.phase2_gpu_sequence'){throw 'Expected the superseded training queue'}
                    Wait-Process -Id $PreviousQueuePid -ErrorAction SilentlyContinue
                }
            }
            $queueArgs=@('-m','tools.phase2_gpu_sequence','--official-pid',"$OfficialPid")
            if($SamWrapperPid){$queueArgs+=@('--sam-wrapper-pid',"$SamWrapperPid")}
            Invoke-Study $modern $queueArgs
        }
        'Static' {Invoke-Study $modern @('tools/phase2_static_package.py',$Run,'--project',$Project)}
        'Score' {Invoke-Study 'python' @('-m','kwater_lab.phase2_score')}
    }
}finally{
    Pop-Location
    foreach($n in $names){[Environment]::SetEnvironmentVariable($n,$old[$n])}
}
