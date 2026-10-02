"""Canonical AnnData contracts and dataset adapters."""

from .contracts import (
    DEFAULT_LATENT_KEY,
    LEGACY_LATENT_KEY,
    attach_simulation_contract,
    canonicalize_celltypes,
    read_latent,
    save_simulation_frames,
)

__all__ = [
    "DEFAULT_LATENT_KEY",
    "LEGACY_LATENT_KEY",
    "attach_simulation_contract",
    "canonicalize_celltypes",
    "read_latent",
    "save_simulation_frames",
]
