#!/usr/bin/env bash
# CLI boundary checks: no real systemd, GPU, network, or repository mutations.
set -euo pipefail
source_root=${1:-$(cd -- "$(dirname -- "$0")/.." && pwd -P)}
sandbox=$(mktemp -d)
trap 'rm -rf -- "$sandbox"' EXIT
mkdir -p "$sandbox"/{tools,bin,.runtime,.venv/bin,reports}
cp "$source_root"/tools/ubuntu_weekend_{gpu,publish,first}.sh "$sandbox/tools/"
export TEST_ROOT=$sandbox PATH="$sandbox/bin:$PATH"
export TEST_NOW=1795978800 TEST_BRANCH=ubuntu/weekend-20261127
cat > "$sandbox/bin/host-fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "${0##*/}" in
    date)
        if [[ ${1:-} == +%s ]]; then echo "$TEST_NOW"; else /usr/bin/date "$@"; fi ;;
    findmnt) echo 459502bd-ca06-4a5a-a9b1-1b528704102d ;;
    df) printf 'Avail\n99999999999\n' ;;
    nvidia-smi) echo '8000, 40' ;;
    systemctl)
        case "$2" in
            is-active) exit 3 ;;
            stop) touch "$TEST_ROOT/stopped" ;;
        esac ;;
    git)
        [[ $1 != -C ]] || shift 2
        case "$1" in
            branch) echo "$TEST_BRANCH" ;;
            diff|add) : ;;
            push) touch "$TEST_ROOT/published" ;;
            rev-parse|ls-remote) echo 1234567890abcdef ;;
            *) exit 97 ;;
        esac ;;
    systemd-run)
        while [[ $1 != /usr/bin/flock ]]; do
            case "$1" in --property=RuntimeMaxSec=*) echo "${1#*=}" > "$TEST_ROOT/runtime" ;; esac
            shift
        done
        lock=$5
        shift 9
        [[ $1 == /usr/bin/bash ]] || exit 96
        if [[ ${TEST_STOP_AT_START:-0} == 1 ]]; then touch "$TEST_ROOT/.runtime/WEEKEND_STOP"; fi
        exec /usr/bin/flock --nonblock --conflict-exit-code 75 "$lock" "$@" ;;
esac
SH
chmod +x "$sandbox/bin/host-fake"
for name in date findmnt df nvidia-smi systemctl git systemd-run; do
    ln -s host-fake "$sandbox/bin/$name"
done
cat > "$sandbox/.venv/bin/python" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$TEST_ROOT/executed"
SH
cat > "$sandbox/tools/ubuntu_weekend_split.sh" <<'SH'
#!/usr/bin/env bash
touch "$TEST_ROOT/split-reached"
exit 75
SH
chmod +x "$sandbox/.venv/bin/python" "$sandbox/tools/ubuntu_weekend_split.sh"
printf 'Reviewed aggregate only.\n' > "$sandbox/reports/ubuntu-test.md"

expect_status() {
    local expected=$1 actual=0
    shift
    "$@" > "$sandbox/output" 2>&1 || actual=$?
    if (( actual != expected )); then
        cat "$sandbox/output" >&2
        echo "FAIL expected=$expected actual=$actual command=$*" >&2
        exit 1
    fi
}

for TEST_NOW in 1795748400 1796036400; do
    # Friday before the old start and Monday after the old cutoff.
    export TEST_NOW
    expect_status 0 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" run kwater_lab.spatial_probe
    test -s "$sandbox/executed"
    [[ $(cat "$sandbox/runtime") == RuntimeMaxSec=21615s ]]
    rm "$sandbox/executed"
    expect_status 0 bash "$sandbox/tools/ubuntu_weekend_publish.sh" test reports/ubuntu-test.md
    test -f "$sandbox/published"
    rm "$sandbox/published"
    expect_status 75 bash "$sandbox/tools/ubuntu_weekend_first.sh"
    test -f "$sandbox/split-reached"
    rm "$sandbox/split-reached"
    echo "PASS uninterrupted startup and publication at epoch=$TEST_NOW"
done

expect_status 0 env KWATER_JOB_TIMEOUT_SECONDS=90 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" \
    run kwater_lab.spatial_oof_audit --report 'path with spaces.json'
[[ $(cat "$sandbox/runtime") == RuntimeMaxSec=105s ]]
[[ $(tail -1 "$sandbox/executed") == 'path with spaces.json' ]]
rm "$sandbox/executed"
echo 'PASS reviewed audit entry point and explicit job budget'

for budget in 0 59 060 -1 abc 1000000 '1+90'; do
    expect_status 64 env KWATER_JOB_TIMEOUT_SECONDS="$budget" bash "$sandbox/tools/ubuntu_weekend_gpu.sh" \
        run kwater_lab.spatial_probe
    test ! -e "$sandbox/executed"
done
expect_status 64 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" run unreviewed.module
test ! -e "$sandbox/executed"
echo 'PASS invalid budgets and unreviewed modules rejected'

touch "$sandbox/.runtime/WEEKEND_STOP"
for script in gpu publish first; do
    case $script in
        gpu) args=(run kwater_lab.spatial_probe) ;;
        publish) args=(test reports/ubuntu-test.md) ;;
        first) args=() ;;
    esac
    expect_status 69 bash "$sandbox/tools/ubuntu_weekend_$script.sh" "${args[@]}"
done
expect_status 0 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" stop
test -f "$sandbox/stopped"
test ! -e "$sandbox/executed"
test ! -e "$sandbox/published"
test ! -e "$sandbox/split-reached"
rm "$sandbox/.runtime/WEEKEND_STOP"
echo 'PASS stop flag blocks work and manual stop stays available'

export TEST_BRANCH=main
expect_status 69 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" run kwater_lab.spatial_probe
expect_status 69 bash "$sandbox/tools/ubuntu_weekend_publish.sh" test reports/ubuntu-test.md
expect_status 69 bash "$sandbox/tools/ubuntu_weekend_first.sh"
export TEST_BRANCH=ubuntu/weekend-20261127
test ! -e "$sandbox/executed"
echo 'PASS branch guards retained'

expect_status 69 env TEST_STOP_AT_START=1 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" run kwater_lab.spatial_probe
test ! -e "$sandbox/executed"
rm "$sandbox/.runtime/WEEKEND_STOP"
echo 'PASS stop flag rechecked at supervised job start'

for lock in ubuntu-weekend-launch ubuntu-weekend-gpu; do
    exec 7>"$sandbox/.runtime/$lock.lock"
    flock 7
    expect_status 75 bash "$sandbox/tools/ubuntu_weekend_gpu.sh" run kwater_lab.spatial_probe
    test ! -e "$sandbox/executed"
    flock -u 7
    exec 7>&-
done
echo 'PASS launch and GPU locks prevent duplicate execution'
