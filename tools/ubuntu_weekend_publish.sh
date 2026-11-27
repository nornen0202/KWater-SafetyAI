#!/usr/bin/env bash
# Publish only reviewed, explicit paths from the auxiliary checkout.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
branch=ubuntu/weekend-20261127
if [[ $(git branch --show-current) != "$branch" || -e .runtime/WEEKEND_STOP ]]; then exit 69; fi
now=$(date +%s)
if (( now < $(date -d '2026-11-27T20:00:00+09:00' +%s) || now >= $(date -d '2026-11-29T23:59:00+09:00' +%s) )); then
    echo "Outside weekend publication window" >&2
    exit 69
fi
if (( $# < 2 )); then echo "Usage: $0 'commit message' reviewed-path [...]" >&2; exit 64; fi
message=$1
shift
exec 9>.runtime/ubuntu-weekend-publish.lock
flock --nonblock --conflict-exit-code 75 9 || exit 75
if ! git diff --cached --quiet; then echo "Review the existing staged changes first" >&2; exit 65; fi
for path in "$@"; do
    case "$path" in
        reports/ubuntu*.md|results/ubuntu*.json|kwater_lab/*.py|tests/*.py|tools/*.py|tools/*.sh) ;;
        *) echo "Path is outside published source and aggregate results: $path" >&2; exit 65 ;;
    esac
    if [[ ! -f $path || -L $path || $(dirname -- "$path") == */* || $(stat -c %s -- "$path") -gt 1048576 ]]; then
        echo "Expected a small, regular, top-level file in its directory: $path" >&2
        exit 65
    fi
done
git add -- "$@"
git diff --cached --check
if ! git diff --cached --quiet; then git commit -m "$message" -- "$@"; fi
git push origin "HEAD:refs/heads/$branch"
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote --exit-code origin "refs/heads/$branch" | cut -f1)
[[ $local_sha == "$remote_sha" ]] || { echo "Remote tip differs; inspect before retrying" >&2; exit 75; }
echo "$branch $local_sha"
