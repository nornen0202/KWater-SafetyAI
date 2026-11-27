#!/usr/bin/env python3
"""Restore the private K-water snapshot to a new or interrupted Linux checkout."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

FOLDER = "1oca7JBCxPG8hiM80rRr9TumB9uK7JnDk"
SNAPSHOT = "full-project-v1"


def download(remote, name, target):
    subprocess.run(["rclone", "copyto", "--drive-root-folder-id", FOLDER,
                    "--retries", "6", "--low-level-retries", "12",
                    remote + ":" + SNAPSHOT + "/" + name, str(target)], check=True)


class VerifiedParts(io.RawIOBase):
    def __init__(self, parts, fetch, temp):
        self.parts = iter(parts)
        self.fetch = fetch
        self.temp = temp
        self.file = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.file is None:
                part = next(self.parts, None)
                if part is None:
                    return 0
                name = part["name"]
                if PurePosixPath(name).name != name:
                    raise ValueError(name)
                self.fetch(name, self.temp)
                if self.temp.stat().st_size != part["bytes"]:
                    raise ValueError("Size mismatch: " + name)
                self.file = self.temp.open("rb")
                if hashlib.file_digest(self.file, "sha256").hexdigest() != part["sha256"]:
                    raise ValueError("SHA-256 mismatch: " + name)
                self.file.seek(0)
            n = self.file.readinto(buffer)
            if n:
                return n
            self.file.close()
            self.file = None
            self.temp.unlink()

    def close(self):
        if self.file:
            self.file.close()
        super().close()


def restore(args):
    project = args.dest.expanduser().resolve()
    deps = project.with_name(project.name + ".dependencies")
    with tempfile.TemporaryDirectory(dir=project.parent) as tempdir:
        tmp = Path(tempdir)
        fetch = lambda name, path: download(args.remote, name, path)
        fetch("READY.json", tmp / "ready")
        fetch("manifest.json", tmp / "manifest")
        raw = (tmp / "manifest").read_bytes()
        manifest = json.loads(raw)
        identity = hashlib.sha256(raw).hexdigest()
        if json.loads((tmp / "ready").read_bytes())["manifest_sha256"] != identity:
            raise ValueError("Drive verification receipt does not match the manifest.")
        if manifest["format"] != 1:
            raise ValueError("Unsupported snapshot")
        marker = project / ".kwater-drive-restore.json"
        if marker.is_file():
            state = json.loads(marker.read_text())
            if state["snapshot"] != identity:
                raise ValueError("Choose a new destination for this snapshot.")
        else:
            if any(p.exists() and (not p.is_dir() or any(p.iterdir())) for p in (project, deps)):
                raise ValueError("Restore into a new or empty project and dependency directory.")
            project.mkdir(exist_ok=True)
            state = {"snapshot": identity, "completed": []}
            marker.write_text(json.dumps(state))
        for i, group in enumerate(manifest["archives"]):
            if i in state["completed"]:
                continue
            print(f'[{i+1}/{len(manifest["archives"])}] {group["group"]}', flush=True)
            with VerifiedParts(group["parts"], fetch, tmp / "part") as stream:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for entry in archive:
                        path = PurePosixPath(entry.name)
                        if (not path.parts or path.parts[0] not in {"project", "dependencies"}
                                or ".." in path.parts or not (entry.isfile() or entry.isdir())):
                            raise ValueError(entry.name)
                        root = project if path.parts[0] == "project" else deps
                        root.mkdir(exist_ok=True)
                        entry.name = str(PurePosixPath(*path.parts[1:]))
                        archive.extract(entry, root, filter="data")
                # tar can stop at the end marker before it reads final padding.
                while stream.read(1024 * 1024):
                    pass
            state["completed"].append(i)
            pending = marker.with_suffix(".tmp")
            pending.write_text(json.dumps(state))
            os.replace(pending, marker)
        print(f"Verified and restored {project}\nDependencies: {deps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="rclone Google Drive remote name without colon")
    parser.add_argument("--dest", type=Path, required=True)
    restore(parser.parse_args())

