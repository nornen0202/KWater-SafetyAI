#!/usr/bin/env bash
# One supervised local GPU job; continue research until explicitly stopped.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
unit=kwater-weekend-gpu
mode=${1:-}
if [[ -z $mode ]]; then
    echo "Usage: $0 check | run MODULE [ARGS...] | stop" >&2
    exit 64
fi
shift
if [[ $mode == stop ]]; then
    exec systemctl --user stop "$unit.service"
fi
if [[ $(findmnt -n -o UUID -T "$root") != 459502bd-ca06-4a5a-a9b1-1b528704102d ]]; then
    echo "Project disk UUID differs" >&2
    exit 66
fi
if (( $(df -B1 --output=avail . | tail -1) < 21474836480 )); then
    echo "Less than 20 GiB free on project disk" >&2
    exit 75
fi
systemctl --user show-environment >/dev/null
mkdir -p .runtime
launch=(systemd-run --user --collect --working-directory="$root" --property=KillMode=control-group
        --property=TimeoutStopSec=10s --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1
        --setenv=PYTHONUNBUFFERED=1 --setenv=OMP_NUM_THREADS=2)
locked=(/usr/bin/flock --nonblock --conflict-exit-code 75 "$root/.runtime/ubuntu-weekend-gpu.lock")
if [[ $mode == check ]]; then
    exec "${launch[@]}" --unit="$unit-check-$$" --wait --pipe --quiet --property=RuntimeMaxSec=60s \
        "${locked[@]}" /usr/bin/systemd-inhibit --what=sleep --mode=block --why='K-water GPU preflight' \
        "$root/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.rand(8, device="cuda").numel())'
fi
if [[ $mode != run || $# -lt 1 ]]; then exit 64; fi
case "$1" in
    kwater_lab.spatial_probe|kwater_lab.spatial_oof_audit|kwater_lab.ubuntu_position_control|kwater_lab.ubuntu_law_control|tools.ubuntu_margin_probe) ;;
    *) echo "Review the experiment entry point before adding it to the launcher" >&2; exit 64 ;;
esac
seconds=${KWATER_JOB_TIMEOUT_SECONDS:-21600}
if [[ ! $seconds =~ ^[1-9][0-9]{1,5}$ ]] || (( seconds < 60 )); then
    echo "KWATER_JOB_TIMEOUT_SECONDS must be an integer from 60 to 999999" >&2
    exit 64
fi
if [[ -e .runtime/WEEKEND_STOP ]]; then echo "Weekend stopped locally" >&2; exit 69; fi
if [[ $(git branch --show-current) != ubuntu/weekend-20261127 ]]; then
    echo "Expected the auxiliary-PC weekend branch" >&2
    exit 69
fi
exec 8>.runtime/ubuntu-weekend-launch.lock
flock --nonblock --conflict-exit-code 75 8 || exit 75
if systemctl --user is-active --quiet "$unit.service"; then echo "GPU job already active" >&2; exit 75; fi
read -r gpu_free gpu_temp < <(nvidia-smi --id=0 --query-gpu=memory.free,temperature.gpu --format=csv,noheader,nounits | tr -d ',')
if (( gpu_free < 1400 || gpu_temp >= 82 )); then echo "GPU memory/temperature preflight deferred" >&2; exit 75; fi
# Expansion intentionally occurs inside the supervised service, at its actual start time.
# shellcheck disable=SC2016
exec "${launch[@]}" --unit="$unit" --wait --pipe --property="RuntimeMaxSec=$((seconds + 15))s" \
    "${locked[@]}" /usr/bin/systemd-inhibit --what=sleep --mode=block --why='K-water weekend experiment' \
    /usr/bin/bash -c '
        root=$1; seconds=$2; shift 2
        [[ ! -e "$root/.runtime/WEEKEND_STOP" ]] || exit 69
        [[ $(git -C "$root" branch --show-current) == ubuntu/weekend-20261127 ]] || exit 69
        exec /usr/bin/timeout --signal=TERM --kill-after=10s "${seconds}s" \
            "$root/.venv/bin/python" -B -u -m "$@"
    ' weekend-job "$root" "$seconds" "$@"
