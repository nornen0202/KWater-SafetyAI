[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$VmRoot,
    [ValidateRange(2048,8192)][int]$MemoryMiB=3584,
    [string]$RuntimeBin,
    [string]$LogDir,
    [string]$CpuModel='qemu64,vendor=GenuineIntel,+ssse3,+sse4.1,+sse4.2,+popcnt,+xsave,+avx,+avx2,+fma,+f16c,+bmi1,+bmi2,-svm'
)
$ErrorActionPreference='Stop'
$vm=(Resolve-Path -LiteralPath $VmRoot).Path
$bin=Join-Path $vm 'qemu'
$runtime=if($RuntimeBin){(Resolve-Path -LiteralPath $RuntimeBin).Path}else{$bin}
$logs=if($LogDir){(Resolve-Path -LiteralPath $LogDir).Path}else{$vm}
$guestDisk=Join-Path $vm 'guest.qcow2'
$base=Join-Path $vm 'noble.img'
if(-not (Test-Path -LiteralPath $guestDisk)){
    & "$bin/qemu-img.exe" create -f qcow2 -F qcow2 -b $base $guestDisk 90G
    if($LASTEXITCODE){throw 'Could not create the guest overlay'}
}
if (Get-NetTCPConnection -LocalPort 22439 -State Listen -ErrorAction SilentlyContinue){throw 'Guest SSH port is in use'}
# Paths used for -drive and -serial below are quoted for Windows process argument parsing.
$argsList=@('-name','kwater39','-L',"`"$bin/share`"",'-machine','pc','-accel','whpx','-cpu',$CpuModel,'-smp','2','-m',"$MemoryMiB",'-vga','none',
    '-drive',"`"file=$guestDisk,format=qcow2,if=virtio`"",'-drive',"`"file=$vm/seed.iso,media=cdrom,readonly=on`"",
    '-netdev','user,id=n0,hostfwd=tcp:127.0.0.1:22439-:22','-device','virtio-net-pci,netdev=n0',
    '-display','none','-serial',"`"file:$logs/serial.log`"",'-monitor','none','-no-reboot')
$proc=Start-Process -WindowStyle Hidden -PassThru -FilePath "$runtime/qemu-system-x86_64.exe" -ArgumentList $argsList -WorkingDirectory $vm -RedirectStandardOutput "$logs/qemu.stdout" -RedirectStandardError "$logs/qemu.stderr"
$proc.Id | Set-Content -LiteralPath "$vm/qemu.pid"
Write-Output "QEMU pid $($proc.Id), 2 vCPU, $MemoryMiB MiB"
