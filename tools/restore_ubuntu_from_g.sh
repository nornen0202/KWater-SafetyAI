#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )); then
  echo "Usage: bash tools/restore_ubuntu_from_g.sh /media/USER/DISK/Projects/KWater-SafetyAI /path/to/new-or-resumed-checkout" >&2
  exit 2
fi
src=$(realpath -- "$1")
dst=$(realpath -m -- "$2")
case "$dst/" in "$src/"*) echo "Choose a destination outside the source." >&2; exit 2;; esac
case "$src/" in "$dst/"*) echo "Choose a destination separate from the source." >&2; exit 2;; esac
test -f "$src/data/train/train.jsonl"
command -v rsync >/dev/null
source_git=(git -c "safe.directory=$src" -C "$src")
commit=$("${source_git[@]}" rev-parse refs/heads/main)
if [[ ! -e "$dst" ]]; then
  git -c "safe.directory=$src" clone --no-local --origin windows --branch main "$src" "$dst"
  git -C "$dst" remote add origin git@github.com:nornen0202/KWater-SafetyAI.git
fi
test "$(git -C "$dst" remote get-url windows)" = "$src"
test "$(git -C "$dst" rev-parse HEAD)" = "$commit" || { echo "Source main changed. Choose a new destination." >&2; exit 1; }
# Files maintained outside Git include original data, previous releases, logs and checkpoints.
# Windows-only environments and caches can be bulky. Preserve them on the source disk.
"${source_git[@]}" ls-files --others --ignored --exclude-standard -z -- . ':(exclude).runtime' ':(exclude,glob)**/__pycache__/**' ':(exclude,glob)**/*.pyc' |
  rsync -rltH --safe-links --partial --from0 --files-from=- --info=progress2 "$src/" "$dst/"
cd "$dst"
sha256sum -c <<'SUMS'
be91eb3a745dc6e1f305237b44842106707164f4174940439b246e4abb385aab  data/packages/kwater-baseline.zip
28acc73fab6a7605361866f87825bd00740b1aacd9bfe35996bdec542dbc37c2  data/packages/kwater-local-kit.zip
2dc849828eb6fc6e4d380aced9757655275f40aa50ab13f4723f6868216dfb77  data/packages/kwater-train.zip
SUMS
printf 'Restored %s into %s\n' "$commit" "$dst"
