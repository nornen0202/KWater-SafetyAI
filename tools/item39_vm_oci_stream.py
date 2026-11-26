"""Serve immutable, previously verified OCI layers to the local unpacker without another disk copy."""

import argparse
import errno
import http.client
import itertools
import json
import os
from pathlib import Path
import stat
import urllib.request

from mfusepy import FUSE, FuseOSError, Operations


class Layers(Operations):
    def __init__(self, work, digests):
        manifest = json.loads((work / "manifest.json").read_bytes())
        self.sizes = {
            x["digest"][7:]: x["size"]
            for x in manifest["layers"]
            if x["digest"][7:] in digests
        }
        if set(self.sizes) != set(digests):
            raise ValueError("Unexpected layer outside pinned OCI manifest")
        self.original = {}
        for digest, size in self.sizes.items():
            fd = os.open(work / "oci/blobs/sha256" / digest, os.O_RDONLY)
            if os.fstat(fd).st_size != size:
                raise ValueError("Missing verified, local fallback layer")
            self.original[digest] = fd
        self.handles = {}
        self.identifiers = itertools.count(1)

    def getattr(self, path, fh=None):
        if path == "/":
            return dict(st_mode=stat.S_IFDIR | 0o555, st_nlink=2)
        if path[1:] not in self.sizes:
            raise FuseOSError(errno.ENOENT)
        return dict(
            st_mode=stat.S_IFREG | 0o444, st_nlink=1, st_size=self.sizes[path[1:]]
        )

    def readdir(self, path, fh):
        return [".", "..", *self.sizes]

    def open(self, path, flags):
        if flags & os.O_ACCMODE != os.O_RDONLY:
            raise FuseOSError(errno.EROFS)
        self.getattr(path)
        handle = next(self.identifiers)
        self.handles[handle] = dict(stream=None, next_offset=None, local=False, count=0)
        return handle

    def read(self, path, size, offset, fh):
        digest = path[1:]
        length = self.sizes[digest]
        size = min(size, length - offset)
        if size <= 0:
            return b""
        state = self.handles[fh]
        if not state["local"]:
            try:
                if state["next_offset"] != offset:
                    if state["stream"]:
                        state["stream"].close()
                    req = urllib.request.Request(
                        "http://10.0.2.2:18439/blobs/" + digest,
                        headers={"Range": f"bytes={offset}-"},
                    )
                    stream = urllib.request.urlopen(req, timeout=120)
                    state["stream"] = stream
                    if (
                        stream.status != 206
                        or stream.headers["Content-Range"]
                        != f"bytes {offset}-{length - 1}/{length}"
                    ):
                        raise OSError("Unexpected range from local file service")
                data = state["stream"].read(size)
                if len(data) != size:
                    raise OSError("Short response from local file service")
                state["next_offset"] = offset + size
                state["count"] += len(data)
                return data
            except (OSError, http.client.HTTPException) as exc:
                if state["stream"]:
                    state["stream"].close()
                state["local"] = True
                print("LOCAL_FALLBACK", digest, offset, type(exc).__name__, flush=True)
        return os.pread(self.original[digest], size, offset)

    def release(self, path, fh):
        state = self.handles.pop(fh)
        if state["stream"]:
            state["stream"].close()
        print("CLOSED", path[1:], state["count"], "streamed bytes", flush=True)
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work", type=Path)
    ap.add_argument("mountpoint")
    ap.add_argument("digests", nargs="+")
    args = ap.parse_args()
    FUSE(
        Layers(args.work, args.digests),
        args.mountpoint,
        foreground=True,
        nothreads=True,
        ro=True,
    )
