from __future__ import annotations

from collections import Counter

import numpy as np

from .checkpoint import load_rollout_normalization
from .config import TrainConfig
from .data import H5ADBatchReader, build_sample_blocks


def inspect_training_config(config: TrainConfig) -> dict:
    """Read representative batches and report the exact training/inference scales."""

    config.validate()
    reader = H5ADBatchReader(
        config.data_path, latent_key=config.latent_key, counts_key=config.counts_key
    )
    blocks = build_sample_blocks(
        reader.obs,
        sample_key=config.sample_key,
        sample_times=config.sample_times,
        batch_size=config.batch_size,
    )
    block_counts = Counter(block.sample for block in blocks)
    sample_cells = {
        sample: int((reader.obs[config.sample_key].astype(str) == sample).sum())
        for sample in config.sample_times
    }
    representative_libraries: dict[str, dict[str, float]] = {}
    for sample in config.sample_times:
        block = next(block for block in blocks if block.sample == sample)
        latent, counts = reader.read_rows(block.start, block.stop)
        libraries = np.asarray(counts.sum(axis=1)).reshape(-1)
        representative_libraries[sample] = {
            "min": float(libraries.min()),
            "median": float(np.median(libraries)),
            "max": float(libraries.max()),
        }
        if latent.shape[1] != reader.latent_dim or counts.shape[1] != reader.n_genes:
            raise AssertionError("reader returned inconsistent tensor dimensions")

    normalization_summary = None
    if config.rollout_normalization_checkpoint is not None:
        normalization = load_rollout_normalization(
            config.rollout_normalization_checkpoint
        )
        if normalization.mean.size != reader.latent_dim:
            raise ValueError("Stage-1 normalization does not match latent dimension")
        normalization_summary = {
            "source": normalization.source,
            "dimension": int(normalization.mean.size),
            "std_min": float(normalization.std.min()),
            "std_max": float(normalization.std.max()),
        }

    return {
        "data_path": str(config.data_path),
        "latent_key": config.latent_key,
        "counts_key": config.counts_key,
        "sample_key": config.sample_key,
        "sample_times": config.sample_times,
        "sample_cells": sample_cells,
        "streamed_blocks": dict(block_counts),
        "n_cells_total": reader.n_cells,
        "latent_dim": reader.latent_dim,
        "n_genes": reader.n_genes,
        "representative_raw_library_size": representative_libraries,
        "raw_count_validation": "finite_nonnegative_integer",
        "nb_target": "untransformed_raw_counts",
        "library_target": "log1p(raw_counts.sum(axis=1))",
        "prediction": "mu=rho*L (negative_binomial_expected_counts)",
        "prediction_is_sampled_counts": False,
        "rollout_normalization": normalization_summary,
    }

