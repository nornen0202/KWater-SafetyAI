"""Read the official SAM state dictionary structure without allocating checkpoint tensors."""

import argparse
import ast
from collections import Counter, OrderedDict
from io import BytesIO
from math import prod
import pickle
from zipfile import ZipFile

from kwater_lab.phase2_panel import RUN, read, write


def tensor(storage, offset, size, stride, *metadata):
    return storage[0], tuple(size)


class StructureReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return tensor
        if module == "torch" and name in {
            "ComplexFloatStorage",
            "FloatStorage",
            "HalfStorage",
            "LongStorage",
            "BFloat16Storage",
            "IntStorage",
            "BoolStorage",
            "ByteStorage",
            "DoubleStorage",
        }:
            return name
        raise ValueError(f"Unexpected checkpoint metadata type {module}.{name}")

    def persistent_load(self, pid):
        if len(pid) != 5 or pid[0] != "storage" or not isinstance(pid[1], str):
            raise ValueError("Unexpected storage metadata")
        return pid[1], pid[4]


def inspect(key):
    root = RUN / "large-models" / key
    rec = read(root / "complete.json")
    checkpoint = root / rec["weight"]
    with ZipFile(checkpoint) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        data = archive.read(metadata)
        state = StructureReader(BytesIO(data)).load()
    state = state.get("model", state)
    dtypes, groups, size = Counter(), Counter(), Counter()
    for name, (dtype, shape) in state.items():
        dtypes[dtype] += 1
        groups[name.split(".", 1)[0]] += 1
        size[dtype] += prod(shape)
    result = dict(
        checkpoint=rec,
        metadata_bytes=len(data),
        tensors=len(state),
        tensor_types=dict(dtypes),
        elements_by_type=dict(size),
        top_prefixes=dict(groups),
    )
    if key == "sam31":
        log = RUN / "sam31-frame.out.log"
        if log.is_file():
            lines = log.read_text(encoding="utf8", errors="replace").splitlines()
            for field, prefix in (
                ("tracker_unmatched_targets", "Missing keys: "),
                ("tracker_unused_checkpoint", "Unexpected keys: "),
            ):
                found = next((s for s in lines if s.startswith(prefix)), None)
                if found:
                    keys = ast.literal_eval(found[len(prefix) :])
                    result[field] = dict(
                        count=len(keys),
                        examples=keys[:3],
                        matches_entire_checkpoint=set(keys) == set(state),
                    )
            result["integrated_load_reports"] = [
                s[:500]
                for s in lines
                if s.startswith(("Missing keys (", "Unexpected keys ("))
            ]
            complete = (RUN / "regions/sam3-static/sam31-frame/runtime.json").is_file()
            result["integrated_builder_returned"] = complete
            result["integrated_exact_key_match"] = (
                (not result["integrated_load_reports"]) if complete else None
            )
            if complete:
                runtime = read(RUN / "regions/sam3-static/sam31-frame/runtime.json")
                verified = runtime.get("final_state_dictionary") or {}
                if verified.get("supplied"):
                    result["strict_integrated_state"] = verified
                    result["integrated_exact_key_match"] = not (
                        verified["missing"] or verified["unexpected"]
                    )
    write(root / "state-structure.json", result)
    print(key, len(state), dict(dtypes), dict(groups), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", choices=["sam31", "sam3"])
    inspect(p.parse_args().model)
