from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    output_dir: Path
    latent_key: str
    counts_key: str
    sample_key: str
    sample_times: dict[str, float]
    checkpoint_name: str = "best.pt"
    rollout_normalization_checkpoint: Path | None = None
    device: str = "cuda:0"
    epochs: int = 100
    patience: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    factor_dim: int = 256
    n_blocks: int = 4
    dropout: float = 0.1
    skip_weight: float = 0.2
    library_weight: float = 0.5
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 2025
    max_blocks_per_split: int | None = None
    overwrite: bool = False

    def validate(self) -> None:
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        if self.rollout_normalization_checkpoint is not None:
            if not self.rollout_normalization_checkpoint.is_file():
                raise FileNotFoundError(self.rollout_normalization_checkpoint)
        if Path(self.checkpoint_name).name != self.checkpoint_name or not self.checkpoint_name.endswith(".pt"):
            raise ValueError("checkpoint_name must be a .pt filename without directories")
        if len(self.sample_times) < 2:
            raise ValueError("sample_times must contain at least two samples")
        if any(not isinstance(sample, str) or not sample for sample in self.sample_times):
            raise ValueError("sample_times keys must be non-empty strings")
        positive = {
            "epochs": self.epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "hidden_dim": self.hidden_dim,
            "factor_dim": self.factor_dim,
            "n_blocks": self.n_blocks,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0 or self.library_weight < 0 or self.skip_weight < 0:
            raise ValueError(
                "weight_decay, library_weight, and skip_weight must be non-negative"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.validation_fraction <= 0 or self.test_fraction <= 0:
            raise ValueError("validation/test fractions must be positive")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation/test fractions must sum to less than one")
        if self.max_blocks_per_split is not None and self.max_blocks_per_split <= 0:
            raise ValueError("max_blocks_per_split must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_path"] = str(self.data_path)
        values["output_dir"] = str(self.output_dir)
        if self.rollout_normalization_checkpoint is not None:
            values["rollout_normalization_checkpoint"] = str(
                self.rollout_normalization_checkpoint
            )
        return values

    def with_overrides(self, **overrides: Any) -> "TrainConfig":
        selected = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **selected)

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        base = config_path.parent

        def resolve_path(value: str | None) -> Path | None:
            if value is None:
                return None
            candidate = Path(value).expanduser()
            return candidate if candidate.is_absolute() else (base / candidate).resolve()

        payload["data_path"] = resolve_path(payload["data_path"])
        payload["output_dir"] = resolve_path(payload["output_dir"])
        payload["rollout_normalization_checkpoint"] = resolve_path(
            payload.get("rollout_normalization_checkpoint")
        )
        payload["sample_times"] = {
            str(key): float(value) for key, value in payload["sample_times"].items()
        }
        config = cls(**payload)
        config.validate()
        return config
