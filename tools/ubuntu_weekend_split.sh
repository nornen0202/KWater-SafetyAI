#!/usr/bin/env bash
# Accept only the byte-identical, published development split.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
expected=3864848fe26b951ec070710499d2694078b17c76ffd2aaf38d312b99ad3b2f60
name=split_manifest_dev.json
target=data/clean-split-v1/$name
receipt() { printf '%s  %s\n' "$expected" "$1" | sha256sum --status -c; }
mkdir -p .runtime "$(dirname -- "$target")"
exec 9>.runtime/weekend-split.lock
flock --nonblock --conflict-exit-code 75 9 || exit 75
if [[ -e $target ]]; then
    receipt "$target" || { echo 'Local split differs from the published hash' >&2; exit 65; }
    echo "Verified $target"
    exit 0
fi
temp=$(mktemp data/clean-split-v1/.incoming-XXXXXX)
trap 'rm -f -- "$temp"' EXIT
restored=../KWater-SafetyAI-restore-20261127/data/clean-split-v1/$name
if [[ $# == 2 && $1 == --from ]]; then
    cp -- "$2" "$temp"
elif [[ $# == 0 && -f $restored ]]; then
    cp -- "$restored" "$temp"
elif [[ $# == 0 ]]; then
    config_dir="$HOME/.local/share/kwater-safetyai"
    timeout 60 "$config_dir/tools/rclone" copyto --config "$config_dir/rclone.conf" \
        --drive-root-folder-id 1oca7JBCxPG8hiM80rRr9TumB9uK7JnDk --retries 1 --low-level-retries 1 \
        "kwater:ubuntu-weekend-20261127/$name" "$temp"
else
    echo "Usage: $0 [--from LOCAL_MANIFEST]" >&2
    exit 64
fi
receipt "$temp" || { echo 'Incoming split differs from the published hash' >&2; exit 65; }
mv -T -- "$temp" "$target"
echo "Verified $target"
