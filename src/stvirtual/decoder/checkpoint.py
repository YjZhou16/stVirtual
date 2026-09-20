from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .model import CountAwareDecoder


@dataclass(frozen=True)
class RolloutNormalization:
    mean: np.ndarray
    std: np.ndarray
    source: str


def load_rollout_normalization(path: Path) -> RolloutNormalization:
    """Safely read Stage-1 f_mu/f_std used to normalize rollout latents."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        normalization = payload["global_norm"]
        mean = normalization["f_mu"].detach().cpu().numpy().astype(np.float32).reshape(-1)
        std = normalization["f_std"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError(f"checkpoint has no valid global_norm f_mu/f_std: {checkpoint}") from error
    if mean.shape != std.shape or mean.size == 0:
        raise ValueError("rollout normalization mean/std dimensions do not match")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("rollout normalization mean/std must be finite with positive std")
    return RolloutNormalization(mean=mean, std=std, source=str(checkpoint))


def save_decoder_checkpoint(path: Path, payload: dict) -> Path:
    """Atomically save a weights-only-compatible decoder payload."""
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    return checkpoint


def load_decoder_checkpoint(
    path: Path, device: torch.device
) -> tuple[CountAwareDecoder, dict]:
    """Load only tensors and primitive metadata; arbitrary pickle is disabled."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported decoder checkpoint schema in {checkpoint}")
    if payload.get("time_conditioning") is not False:
        raise ValueError(
            "time-conditioned or unspecified decoder checkpoints are unsupported; "
            "retrain with time_conditioning=false"
        )
    try:
        model = CountAwareDecoder(**payload["model_config"])
        model.load_state_dict(payload["model_state"], strict=True)
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError(f"invalid decoder checkpoint: {checkpoint}") from error
    model.to(device).eval()
    return model, payload
