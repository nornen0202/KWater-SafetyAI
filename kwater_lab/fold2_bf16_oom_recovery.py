"""Resume Fold 2 BF16 using its existing cache and a smaller effective batch."""
import json
from unittest.mock import patch

from .fold2_v3_epoch1 import DEFAULT_RUN, infer
from .fold4_same_verifier import read


def adapt_generate(generate, oom_error, maximum_batch):
    def guarded_generate(*args, **kwargs):
        if kwargs["input_ids"].shape[0] > maximum_batch:
            raise oom_error("Requested lower effective batch after CUDA RuntimeError OOM")
        try:
            return generate(*args, **kwargs)
        except RuntimeError as error:
            if isinstance(error, oom_error) or "cuda error: out of memory" not in str(error).lower():
                raise
            raise oom_error("Recovered non-typed CUDA out-of-memory error") from error

    return guarded_generate


def main():
    import torch

    from . import vlm_11code as vlm

    cache = read(DEFAULT_RUN / "epoch1-bf16-raw-cache.json")
    print("BF16_RECOVERY " + json.dumps({"retained_rows": len(cache["outputs"]), "resume_batch_cap": 2,
                                        "prior_effective_batch": cache["effective_batch_size"]}), flush=True)
    original_load = vlm.load_qwen_4bit_or_bf16

    def load(*args, **kwargs):
        model, processor = original_load(*args, **kwargs)
        model.generate = adapt_generate(model.generate, torch.OutOfMemoryError, maximum_batch=2)
        return model, processor

    with patch.object(vlm, "load_qwen_4bit_or_bf16", load):
        infer(DEFAULT_RUN.resolve())


if __name__ == "__main__":
    main()
