#!/usr/bin/env bash
# Ubuntu 24.04 local guest, with only a loopback-published host file service.
set -euo pipefail
umask 022
host=http://10.0.2.2:18439
work=/var/lib/kwater39
mkdir -p "$work/oci/blobs/sha256"
cd "$work"
fetch_checked() {
  python3 - "$1" "$2" "$3" <<'PY'
import hashlib, os, pathlib, sys, time, urllib.request
url, dest, expected = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
part = dest.with_name(dest.name + ".part")
for attempt in range(4):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=180) as r:
            size = int(r.headers["Content-Length"])
        h = hashlib.sha256()
        with part.open("a+b") as out:
            start = out.tell()
            if start > size:
                raise ValueError("partial file is larger than source")
            out.seek(0)
            while buf := out.read(4 * 1024 * 1024):
                h.update(buf)
            if start < size:
                headers = {"Range": f"bytes={start}-"} if start else {}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as response:
                    if start and (response.status != 206 or response.headers["Content-Range"] != f"bytes {start}-{size-1}/{size}"):
                        raise ValueError("unexpected range response")
                    while buf := response.read(4 * 1024 * 1024):
                        out.write(buf)
                        h.update(buf)
            out.flush()
            os.fsync(out.fileno())
            if out.tell() != size:
                raise OSError("incomplete transfer")
        if h.hexdigest() != expected:
            raise ValueError("unexpected transfer SHA-256")
        part.replace(dest)
        print(h.hexdigest(), dest, flush=True)
        break
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and part.exists():
            part.replace(part.with_name(part.name + ".invalid-" + str(time.time_ns())))
        if attempt == 3:
            raise
        time.sleep(5)
PY
}
if ! command -v umoci >/dev/null; then
  curl -fsSL "$host/umoci" -o umoci
  echo 'b51c267ec394499e42c6fde47f240b7b7dba57ea49df0b5acd304378b82a3b71  umoci' | sha256sum -c -
  install -m 0755 umoci /usr/local/bin/umoci
fi
if ! swapon --noheadings | read -r _; then
  if [[ ! -f "$work/swap" ]]; then
    fallocate -l 5G "$work/swap"
    chmod 600 "$work/swap"
    mkswap "$work/swap"
  fi
  swapon "$work/swap"
fi
curl -fsSL "$host/manifest.json" -o manifest.json
digest=2e924d941c41876e9eb1de79a1af7eaf796aacb1210fb86f6ff3345cc7c58d64
echo "$digest  manifest.json" | sha256sum -c -
cp manifest.json "oci/blobs/sha256/$digest"
python3 - <<'PY'
import json,pathlib,hashlib
p=pathlib.Path('manifest.json'); r=p.read_bytes(); m=json.loads(r)
entry=dict(mediaType=m['mediaType'],digest='sha256:'+hashlib.sha256(r).hexdigest(),size=len(r),annotations={'org.opencontainers.image.ref.name':'runtime'})
pathlib.Path('oci/index.json').write_text(json.dumps(dict(schemaVersion=2,manifests=[entry])))
pathlib.Path('oci/oci-layout').write_text('{"imageLayoutVersion":"1.0.0"}')
pathlib.Path('blob-ids').write_text('\n'.join(o['digest'][7:] for o in [m['config'],*m['layers']])+'\n')
PY
while read -r blob; do
  dest="oci/blobs/sha256/$blob"
  if [[ ! -f "$dest" ]]; then
    fetch_checked "$host/blobs/$blob" "$dest" "$blob"
  fi
done <blob-ids
if [[ ! -d container/rootfs ]]; then
  mkdir -p container
  unpack_dir=$(mktemp -d "$work/container/unpack.XXXXXX")
  umoci raw unpack --image oci:runtime "$unpack_dir/rootfs"
  mv "$unpack_dir/rootfs" "$work/container/rootfs"
  rmdir "$unpack_dir"
fi
root="$work/container/rootfs"
mkdir -p "$root/rehearsal/kit" "$root/work"
chown 10001:10001 "$root/work"
for file in check_zip.py zipcheck.py check_output.py prediction.schema.json sandbox_exec.py; do
  curl -fsSL "$host/kit/$file" -o "$root/rehearsal/kit/$file"
done
curl -fsSL "$host/benchmark.py" -o "$root/rehearsal/benchmark.py"
sha256sum "$root/rehearsal/kit/"* "$root/rehearsal/benchmark.py"
if [[ ! -f "$root/rehearsal/submission.zip" ]]; then
  fetch_checked "$host/submission.zip" "$root/rehearsal/submission.zip" 1006bb157da7df77b5bdd8858b0c7198e2348b973153ac33dba849992f640d6f | tee archive-guest.sha256
else
  sha256sum "$root/rehearsal/submission.zip" > archive-guest.sha256
fi
grep -q '^1006bb157da7df77b5bdd8858b0c7198e2348b973153ac33dba849992f640d6f ' archive-guest.sha256
cat archive-guest.sha256
uname -a
lscpu
free -h
mkdir -p "$root/proc" "$root/dev" "$root/sys"
mountpoint -q "$root/proc" || mount -t proc proc "$root/proc"
mountpoint -q "$root/dev" || mount --rbind /dev "$root/dev"
mountpoint -q "$root/sys" || mount --rbind /sys "$root/sys"
for n in "${@:-3}"; do
  job="/work/n$n"
  extra=()
  if [[ "$n" != 3 && -f "$root/work/n3/timing.json" ]]; then
    mkdir -p "$root/rehearsal/verified-code"
    if ! mountpoint -q "$root/rehearsal/verified-code"; then
      mount --bind "$root/work/n3/code" "$root/rehearsal/verified-code"
      mount -o remount,bind,ro "$root/rehearsal/verified-code"
    fi
    extra=(--verified-code /rehearsal/verified-code)
  fi
  sync
  echo 3 > /proc/sys/vm/drop_caches
  unshare -n -- chroot --userspec=10001:10001 "$root" /usr/bin/env -i \
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8 HOME=/work PIP_DISABLE_PIP_VERSION_CHECK=1 \
    python /rehearsal/benchmark.py --kit /rehearsal/kit --zip /rehearsal/submission.zip \
    --n "$n" --seed "$((20261126+n))" --out "$job" "${extra[@]}"
  export_dir="/home/runner/item39/n$n"
  install -d -m 0755 "$export_dir"
  cp "$root$job"/{timing.json,photos.csv,synthetic-provenance.csv,main.log,zip-check.log,install.log,output-check.log} "$export_dir/"
  cp "$root$job/output/submission.jsonl" "$export_dir/"
  echo "ARTIFACTS_READY $n"
done
