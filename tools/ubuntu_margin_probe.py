"""Read-only GPU log-probability probe for a frozen verifier candidate order.

The probe never opens training labels or generates a verifier response.  It scores
the two one-token continuations after the identical verifier chat prompt plus the
fixed JSON prefix ``{\"decision\":\"``.  Inputs and results belong under ignored
``data/``; a diagnostic proposal is deliberately not an evaluation cohort.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Final

from PIL import Image, ImageOps


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DECISION_PREFIX: Final = '{"decision":"'
FORBIDDEN_LABEL_KEYS: Final = frozenset({"label", "labels", "target", "truth", "gold"})
FROZEN_CANDIDATE_COUNT: Final = 81
PRECISIONS: Final = frozenset({"bf16", "nf4"})


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ordered, label-free photo/code verifier request."""

    sample_id: str
    image: str
    code: str
    image_sha256: str


@dataclass(frozen=True, slots=True)
class Proposal:
    """Parsed P3 candidate list, whose order is part of its identity."""

    schema_version: int
    mode: str
    candidates: tuple[Candidate, ...]


def sha256_file(path: Path) -> str:
    """Return a content digest without interpreting the file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: list[dict[str, str]]) -> str:
    """Hash ordered JSON with a stable representation."""
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def contains_label_key(value: object) -> bool:
    """Reject accidental label-bearing proposal JSON before model loading."""
    if isinstance(value, dict):
        return any(key.casefold() in FORBIDDEN_LABEL_KEYS or contains_label_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_label_key(item) for item in value)
    return False


def parse_proposal(path: Path, diagnostic_allowed: bool) -> Proposal:
    """Parse the intentionally narrow frozen-proposal schema."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or contains_label_key(raw):
        raise ValueError("Proposal must be a label-free JSON object")
    if set(raw) != {"schema_version", "mode", "candidates"}:
        raise ValueError("Proposal keys must be schema_version, mode, candidates only")
    if raw["schema_version"] != 1 or raw["mode"] not in {"frozen_p3", "diagnostic_synthetic"}:
        raise ValueError("Unsupported proposal schema_version or mode")
    if raw["mode"] == "diagnostic_synthetic" and not diagnostic_allowed:
        raise ValueError("Diagnostic proposals require --allow-diagnostic")
    if raw["mode"] == "frozen_p3" and len(raw["candidates"]) != FROZEN_CANDIDATE_COUNT:
        raise ValueError("A frozen_p3 proposal must contain exactly 81 candidates")
    if not isinstance(raw["candidates"], list) or not raw["candidates"]:
        raise ValueError("Proposal candidates must be a nonempty list")
    candidates: list[Candidate] = []
    for item in raw["candidates"]:
        if not isinstance(item, dict) or set(item) != {"sample_id", "image", "code", "image_sha256"}:
            raise ValueError("Each candidate must have sample_id, image, code, image_sha256 only")
        if not all(isinstance(item[key], str) and item[key] for key in item):
            raise ValueError("Candidate values must be nonempty strings")
        image = Path(item["image"])
        if image.is_absolute() or ".." in image.parts:
            raise ValueError("Candidate image must be a relative non-traversing path")
        candidates.append(Candidate(**item))
    keys = [(candidate.sample_id, candidate.code) for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate sample_id/code pairs must be unique")
    return Proposal(schema_version=1, mode=raw["mode"], candidates=tuple(candidates))


def verify_images(proposal: Proposal, image_root: Path) -> None:
    """Require exactly the frozen photo bytes without reading labels."""
    root = image_root.resolve()
    for candidate in proposal.candidates:
        image_path = (root / candidate.image).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Missing or escaping candidate image: {candidate.image}")
        if sha256_file(image_path) != candidate.image_sha256:
            raise ValueError(f"Candidate image hash differs: {candidate.image}")


def require_output_under_data(path: Path) -> None:
    """Keep raw P3 inputs and outputs out of versioned result directories."""
    data_root = (Path.cwd() / "data").resolve()
    if not path.resolve().is_relative_to(data_root):
        raise ValueError("--out must be under the ignored data/ directory")


def model_identity(base_model: Path, adapter: Path, prompt: Path, precision: str) -> dict[str, object]:
    """Fingerprint every condition that could change a measured margin."""
    import torch

    return {
        "base_model_config_sha256": sha256_file(base_model / "config.json"),
        "base_weight_sha256": {part.name: sha256_file(part) for part in sorted(base_model.glob("*.safetensors"))},
        "processor_config_sha256": sha256_file(base_model / "preprocessor_config.json"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "prompt_sha256": sha256_file(prompt),
        "precision": precision,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
    }


def run_probe(proposal: Proposal, args: argparse.Namespace) -> dict[str, object]:
    """Load the requested local adapter once and score ordered decision margins."""
    import torch
    from kwater_lab.vlm_11code import CANDIDATE_MAX_PIXELS, load_qwen_model, load_qwen_processor, verify_user_text

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU-only probe")
    base_model, adapter, prompt = args.base_model, args.adapter, args.prompt
    if args.precision not in PRECISIONS:
        raise ValueError("precision must be bf16 or nf4")
    identity = model_identity(base_model, adapter, prompt, args.precision)
    if proposal.mode == "frozen_p3":
        if identity["adapter_model_sha256"] != args.expected_adapter_sha256:
            raise ValueError("Adapter differs from the frozen sender receipt")
        if identity["prompt_sha256"] != "107142013bc95376c9044fc5025079f1e68dd56b4d95b20b1550deeafe4712fb":
            raise ValueError("Prompt differs from the original v2 replay")
    load_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    processor = load_qwen_processor(base_model, CANDIDATE_MAX_PIXELS)
    accept_ids = processor.tokenizer.encode("accept", add_special_tokens=False)
    reject_ids = processor.tokenizer.encode("reject", add_special_tokens=False)
    if len(accept_ids) != 1 or len(reject_ids) != 1:
        raise RuntimeError("accept/reject must each be exactly one tokenizer token")
    system_prompt = prompt.read_text(encoding="utf-8")
    model = load_qwen_model(base_model, use_4bit=args.precision == "nf4", adapter_dir=adapter)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    rows: list[dict[str, str | float]] = []
    started = time.perf_counter()
    selected = proposal.candidates[: args.max_candidates]
    with torch.inference_mode():
        for candidate in selected:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verify_user_text(candidate.code)}]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + DECISION_PREFIX
            with Image.open(args.image_root / candidate.image) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            accept_logprob = float(log_probs[accept_ids[0]].item())
            reject_logprob = float(log_probs[reject_ids[0]].item())
            rows.append({"sample_id": candidate.sample_id, "image": candidate.image, "image_sha256": candidate.image_sha256,
                         "code": candidate.code, "accept_logprob": accept_logprob, "reject_logprob": reject_logprob,
                         "accept_minus_reject_logprob": (logits[accept_ids[0]] - logits[reject_ids[0]]).item()})
            del inputs, logits, log_probs
    elapsed = time.perf_counter() - started
    peak_bytes = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    ordered = [asdict(candidate) for candidate in selected]
    return {
        "purpose": "diagnostic GPU token-margin measurement only; no effect or accuracy claim",
        "proposal_mode": proposal.mode,
        "candidate_count": len(rows),
        "frozen_candidate_count_required_for_p3": FROZEN_CANDIDATE_COUNT,
        "ordered_candidate_sha256": canonical_sha256(ordered),
        "proposal_file_sha256": sha256_file(args.proposal),
        "decision_prefix": DECISION_PREFIX,
        "decision_token_ids": {"accept": accept_ids[0], "reject": reject_ids[0]},
        "identity": identity,
        "processor_max_pixels": CANDIDATE_MAX_PIXELS,
        "processor_and_model_load_seconds": round(load_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "peak_cuda_allocated_bytes": peak_bytes,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("data/baseline/base_model"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-proposal-sha256")
    parser.add_argument("--prompt", type=Path, default=Path("kwater_lab/system_prompt_11code.txt"))
    parser.add_argument("--precision", choices=sorted(PRECISIONS), required=True)
    parser.add_argument("--max-candidates", type=int, default=FROZEN_CANDIDATE_COUNT)
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be positive")
    require_output_under_data(args.out)
    proposal = parse_proposal(args.proposal, args.allow_diagnostic)
    if proposal.mode == "frozen_p3":
        if not args.expected_adapter_sha256 or sha256_file(args.proposal) != args.expected_proposal_sha256:
            raise ValueError("Frozen P3 requires a matching proposal SHA and the sender's v2 adapter SHA")
        args.adapter = args.adapter or Path("data/vlm-11code-fold0-v2/adapter")
    else:
        args.adapter = args.adapter or Path("data/baseline/adapter")
    verify_images(proposal, args.image_root)
    if args.validate_only:
        print(json.dumps({"proposal_mode": proposal.mode, "candidates": len(proposal.candidates),
                          "proposal_file_sha256": sha256_file(args.proposal)}, sort_keys=True))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = run_probe(proposal, args)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.out.parent, delete=False) as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    Path(stream.name).replace(args.out)


if __name__ == "__main__":
    main()
