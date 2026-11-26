#!/bin/bash
# Optional prep-only bind for the two large future OCI layers on a fragmented guest disk.
set -euo pipefail
cd /var/lib/kwater39/oci/blobs/sha256
mount_root=/run/kwater39-oci-stream
layers=(4614b301ea4206e46e0a8db954fcb25bb0a89da5d116f75fc3e5820ff0f416b0
        3cfcd3ae4193f80850f7320dea05b76c4145e14239fceebaccdf804b216c390f)
if [[ ${1:-} == cleanup ]]; then
    while kill -0 "$2" 2>/dev/null; do sleep 30; done
    for f in "${layers[@]}"; do
        if mountpoint -q "$f"; then umount "$f"; fi
    done
    systemctl stop kwater39-oci-read.service
    exit
fi
pid=$(pgrep -x umoci)
for f in "${layers[@]}"; do
    python3 - "$mount_root/mnt/$f" "$f" <<'PY'
import hashlib
import os
import sys
import time

measurements = []
for path in sys.argv[1:]:
    with open(path, "rb") as file:
        length = os.fstat(file.fileno()).st_size
        offset = 2_200_000_021 if length > 3_000_000_000 else 121_234_567
        start = time.monotonic()
        data = os.pread(file.fileno(), 8192, offset)
        evidence = (length, len(data), hashlib.sha256(data).hexdigest())
        print(path, offset, evidence, round(time.monotonic() - start, 4), flush=True)
        measurements.append(evidence)
if measurements[0] != measurements[1] or measurements[0][1] != 8192:
    raise ValueError("Range evidence disagrees")
PY
done
if grep -q LOCAL_FALLBACK "$mount_root/stream.log"; then exit 42; fi
systemd-run --unit=kwater39-oci-read-cleanup --collect /bin/bash "$0" cleanup "$pid"
for f in "${layers[@]}"; do
    if ls -l "/proc/$pid/fd" | grep -q "$f"; then continue; fi
    mount --bind "$mount_root/mnt/$f" "$PWD/$f"
    mount -o remount,bind,ro "$PWD/$f"
    findmnt -n -o TARGET,FSTYPE,OPTIONS --target "$PWD/$f"
done
date -u
