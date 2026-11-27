"""Read local Qwen safetensors one tensor at a time on a low-commit Windows host."""

import json
import math
from pathlib import Path
import struct


class TensorSlice:
    def __init__(self, file, tensor):
        self.file = file
        self.tensor = tensor

    def get_shape(self):
        return self.tensor["shape"]

    def get_dtype(self):
        return self.tensor["dtype"]

    def __getitem__(self, key):
        import torch

        if key is not Ellipsis:
            raise ValueError(
                "The registered non-sharded model loader requests complete tensors"
            )
        data = self.file.read_bytes_for(self.tensor)
        dtype = getattr(torch, self.file.types[self.get_dtype()][0])
        if not data:
            return torch.empty(self.get_shape(), dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).reshape(self.get_shape())


class TensorFile:
    types = {
        "BF16": ("bfloat16", 2),
        "F16": ("float16", 2),
        "F32": ("float32", 4),
        "U8": ("uint8", 1),
        "I8": ("int8", 1),
        "I32": ("int32", 4),
        "I64": ("int64", 8),
    }

    def __init__(self, filename, framework, device="cpu"):
        if framework != "pt":
            raise ValueError("Expected PyTorch safetensors")
        self.filename = Path(filename)
        self.stream = self.filename.open("rb", buffering=0)
        try:
            size = struct.unpack("<Q", self.stream.read(8))[0]
            if not 2 <= size <= 100_000_000:
                raise ValueError("Invalid safetensors header size")
            self.tensors = json.loads(self.stream.read(size))
            self.payload = 8 + size
            self.meta = self.tensors.pop("__metadata__", None)
            end = 0
            for tensor in sorted(
                self.tensors.values(), key=lambda t: t["data_offsets"][0]
            ):
                lo, hi = tensor["data_offsets"]
                if (
                    tensor["dtype"] not in self.types
                    or lo != end
                    or hi - lo
                    != self.types[tensor["dtype"]][1] * math.prod(tensor["shape"])
                ):
                    raise ValueError("Unexpected tensor shape, dtype, or offsets")
                end = hi
            if self.payload + end != self.filename.stat().st_size:
                raise ValueError(
                    "The safetensors payload size does not match its header"
                )
        except BaseException:
            self.stream.close()
            raise

    def metadata(self):
        return self.meta

    def keys(self):
        return sorted(self.tensors)

    def get_slice(self, name):
        return TensorSlice(self, self.tensors[name])

    def get_tensor(self, name):
        return self.get_slice(name)[...]

    def read_bytes_for(self, tensor):
        lo, hi = tensor["data_offsets"]
        self.stream.seek(self.payload + lo)
        data = bytearray(hi - lo)
        view = memoryview(data)
        while view:
            n = self.stream.readinto(view[: 8 * 1024 * 1024])
            if not n:
                raise EOFError(f"Truncated tensor in {self.filename.name}")
            view = view[n:]
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stream.close()


class BF16TensorFile(TensorFile):
    """The official unquantized 3-VL checkpoint has exactly one on-disk dtype."""

    types = {"BF16": ("bfloat16", 2)}


def verify_checkpoint_headers():
    """Check each mapped dtype against the native safetensors reader on a small file."""
    import hashlib
    from importlib.metadata import version
    from safetensors import deserialize

    from .phase2_io import PROJECT, RUN, write

    receipt = {}
    for model, folder in (
        ("3vl", PROJECT / "data/baseline/base_model"),
        ("35", RUN / "qwen35"),
    ):
        for path in sorted(folder.glob("*.safetensors")):
            with TensorFile(path, "pt") as tensors:
                populations = {}
                for key, tensor in tensors.tensors.items():
                    populations.setdefault(tensor["dtype"], []).append(key)
                samples = []
                for dtype, names in populations.items():
                    ordered = sorted(
                        names, key=lambda k: math.prod(tensors.tensors[k]["shape"])
                    )
                    for key in ordered[:3]:
                        tensor = tensors.tensors[key]
                        lo, hi = tensor["data_offsets"]
                        with path.open("rb") as original:
                            original.seek(tensors.payload + lo)
                            payload = original.read(hi - lo)
                        header = json.dumps(
                            {key: dict(tensor, data_offsets=[0, len(payload)])}
                        ).encode()
                        header += b" " * (-len(header) % 8)
                        native_key, native_tensor = deserialize(
                            struct.pack("<Q", len(header)) + header + payload
                        )[0]
                        actual = tensors.read_bytes_for(tensor)
                        if (
                            native_key != key
                            or native_tensor["dtype"] != dtype
                            or list(native_tensor["shape"]) != tensor["shape"]
                            or bytes(native_tensor["data"]) != actual
                        ):
                            raise ValueError(
                                "Native reader and streaming reader differ"
                            )
                        samples.append(
                            dict(
                                key=key,
                                dtype=dtype,
                                bytes=len(actual),
                                sha256=hashlib.sha256(actual).hexdigest(),
                            )
                        )
                receipt[f"{model}/{path.name}"] = dict(
                    bytes=path.stat().st_size,
                    by_dtype={
                        dtype: len(names) for dtype, names in populations.items()
                    },
                    native_comparisons=samples,
                )
    write(
        RUN / "weight-reader-compatibility.json",
        dict(
            safetensors=version("safetensors"),
            files=receipt,
            scope="Headers and sampled tensor bytes",
        ),
    )


if __name__ == "__main__":
    verify_checkpoint_headers()
