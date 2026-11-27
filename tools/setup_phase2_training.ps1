[CmdletBinding()]
param(
    [string]$Run='G:/Projects/KWater-SafetyAI/data/phase2-runs/p38-20261126-49bb82ce',
    [string]$Legacy='C:/Users/JY/anaconda3/envs/env_aias_test'
)
$ErrorActionPreference='Stop'
$bridge=Join-Path $Run 'bridge'
$py=Join-Path $bridge 'Scripts/python.exe'
$names=@('PATH','TEMP','TMP','UV_CACHE_DIR','UV_HTTP_TIMEOUT','PYTHONNOUSERSITE','PYTHONUTF8')
$old=@{}
foreach($n in $names){$old[$n]=[Environment]::GetEnvironmentVariable($n)}
try {
    $env:PATH="$Legacy/Library/bin;$Legacy/DLLs;$Legacy;$env:PATH"
    $env:TEMP=Join-Path $Run 'temp'
    $env:TMP=$env:TEMP
    $env:UV_CACHE_DIR=Join-Path $Run 'uv-cache'
    $env:UV_HTTP_TIMEOUT='3600'
    $env:PYTHONUTF8='1'
    $env:PYTHONNOUSERSITE='1'
    [IO.Directory]::CreateDirectory($env:TEMP) | Out-Null
    if(-not (Test-Path -LiteralPath $py)){
        & "$Legacy/python.exe" -B -m venv --without-pip --system-site-packages $bridge
        if($LASTEXITCODE){throw "venv creation failed with $LASTEXITCODE"}
    }
    $pins=@('transformers==5.5.4','peft==0.19.1','bitsandbytes==0.49.2','accelerate==1.13.0',
            'huggingface-hub==1.33.0','tokenizers==0.22.2','safetensors==0.8.0')
    & uv pip install --no-deps --python $py @pins
    if($LASTEXITCODE){throw "pinned libraries failed with $LASTEXITCODE"}
    & uv pip install --python $py 'typer==0.20.0' 'regex==2026.11.25' 'hf-xet==1.8.0' 'click==8.5.1'
    if($LASTEXITCODE){throw "declared runtime dependencies failed with $LASTEXITCODE"}
    & $py -B -u -c "import torch,torchvision,transformers,peft,bitsandbytes;print(torch.__version__,torchvision.__version__,transformers.__version__,peft.__version__,bitsandbytes.__version__,torch.cuda.get_device_name());from transformers import Qwen3_5ForConditionalGeneration,Qwen3VLForConditionalGeneration;print('both_architectures_available')" *> (Join-Path $Run 'bridge-import.log')
    if($LASTEXITCODE){throw "architecture imports failed with $LASTEXITCODE"}
} finally {
    foreach($n in $names){[Environment]::SetEnvironmentVariable($n,$old[$n])}
}
