"""CPU-only Fold1+2 feature fitting and immutable, label-blind Fold3 ranks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch

from .metrics import CODES
from .spatial_probe import Row, _fit_scores, digest, read_cached
from .ubuntu_focused_eval import write_once

ENCODER: Final = "c260ed2ccf2837df7e8f6a1b2f56aab0df1fbb2e4c6848d23bf3b54b404ff7e8"


@dataclass(frozen=True, slots=True)
class OracleInputError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class FreezeConfig:
    manifest: Path
    summary: Path
    cache: Path
    out: Path
    encoder: str = ENCODER
    steps: int = 160
    seed: int = 20261127
    threads: int = 2


@dataclass(frozen=True, slots=True)
class RankRow:
    sample_id: str
    group_id: str
    image_sha256: str
    global_scores: tuple[float, ...]
    regional_scores: tuple[float, ...]
    global_rank: tuple[str, ...]
    regional_rank: tuple[str, ...]


def write_json[T](path: Path, payload: T) -> None:
    """Create a durable JSON artifact without replacing an earlier freeze."""
    write_once(path, payload)


def freeze(config: FreezeConfig) -> Path:
    """Fit Fold1+2 only; never construct or inspect Fold3 target code sets."""
    if config.out.exists():
        seal_path = config.out.with_suffix(".seal.json")
        if not seal_path.exists():
            raise OracleInputError(
                "unsealed existing ranks preserved; inspect and seal manually, never refit"
            )
        seal = json.loads(seal_path.read_text())
        existing = json.loads(config.out.read_text())
        expected = (
            digest(config.manifest),
            config.encoder,
            config.steps,
            config.seed + 3,
        )
        actual = (
            existing["manifest_sha256"],
            existing["encoder_sha256"],
            existing["fit"]["steps"],
            existing["fit"]["seed"],
        )
        if digest(config.out) != seal["sha256"] or actual != expected:
            raise OracleInputError(
                "existing rank seal or requested configuration differs; original preserved"
            )
        return seal_path
    summary = json.loads(config.summary.read_text(encoding="utf-8"))
    if digest(config.manifest) != summary["dev_manifest_sha256"]:
        raise OracleInputError("development manifest hash mismatch")
    raw = json.loads(config.manifest.read_text(encoding="utf-8"))
    rows = []
    for item in raw:
        if item["split"] != "dev":
            raise OracleInputError("non-development row in manifest")
        fold = int(item["dev_fold"])
        if fold in (1, 2, 3):
            codes = frozenset(item["codes"]) if fold in (1, 2) else frozenset()
            if not codes <= set(CODES):
                raise OracleInputError("unknown training code")
            rows.append(
                Row(
                    item["sample_id"],
                    item["image"],
                    item["sha256"],
                    item["group_id"],
                    fold,
                    codes,
                )
            )
    train = [row for row in rows if row.fold in (1, 2)]
    valid = [row for row in rows if row.fold == 3]
    if not train or not valid or len({row.sample_id for row in rows}) != len(rows):
        raise OracleInputError("missing folds or duplicate sample IDs")
    if {row.group_id for row in train} & {row.group_id for row in valid}:
        raise OracleInputError("incident group crosses training and Fold3")
    features = []
    for row in train + valid:
        grid = read_cached(row, config.cache, config.encoder)
        if grid is None:
            raise OracleInputError(f"missing or invalid cache for {row.sample_id}")
        features.append(grid)
    stacked = np.stack(features)
    targets = np.asarray(
        [[code in row.codes for code in CODES] for row in train], dtype=np.float32
    )
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    scores = [
        _fit_scores(
            stacked[: len(train)],
            targets,
            stacked[len(train) :],
            regional=regional,
            top_k=3,
            seed=config.seed + 3,
            steps=config.steps,
            device_name="cpu",
        )
        for regional in (False, True)
    ]
    rank_rows = [
        RankRow(
            row.sample_id,
            row.group_id,
            row.sha256,
            tuple(map(float, scores[0][index])),
            tuple(map(float, scores[1][index])),
            tuple(
                CODES[column] for column in np.argsort(-scores[0][index], kind="stable")
            ),
            tuple(
                CODES[column] for column in np.argsort(-scores[1][index], kind="stable")
            ),
        )
        for index, row in enumerate(valid)
    ]
    write_json(
        config.out,
        {
            "kind": "CPU refit; not reused GPU scores",
            "created_utc": datetime.now(UTC).isoformat(),
            "manifest_sha256": digest(config.manifest),
            "summary_sha256": digest(config.summary),
            "encoder_sha256": config.encoder,
            "max_pixels": 262144,
            "fit_folds": [1, 2],
            "validation_fold": 3,
            "fit_photos": len(train),
            "validation_photos": len(valid),
            "fit_groups": len({row.group_id for row in train}),
            "validation_groups": len({row.group_id for row in valid}),
            "fit": {
                "device": "cpu",
                "seed": config.seed + 3,
                "steps": config.steps,
                "optimizer": "AdamW",
                "lr": 0.03,
                "weight_decay": 0.02,
                "initialization": "zero weight and bias",
                "threads": config.threads,
                "torch_version": torch.__version__,
                "parameters_each": (stacked.shape[2] + 1) * len(CODES),
            },
            "code_order": list(CODES),
            "tie_break": "stable canonical code order",
            "train_positive_codes": [
                code for column, code in enumerate(CODES) if targets[:, column].any()
            ],
            "validation_labels_accessed": False,
            "baseline_accessed": False,
            "source_sha256": {
                name: digest(Path(__file__).with_name(name))
                for name in ("spatial_probe.py", "ubuntu_fold3_candidate_ranks.py")
            },
            "rows": [asdict(row) for row in rank_rows],
        },
    )
    seal = config.out.with_suffix(".seal.json")
    write_json(
        seal,
        {
            "path": str(config.out),
            "sha256": digest(config.out),
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )
    return seal
