from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .checkpoint import load_rollout_normalization, save_decoder_checkpoint
from .config import TrainConfig
from .data import (
    H5ADBatchReader,
    RowBlock,
    build_sample_blocks,
    split_sample_blocks,
)
from .model import CountAwareDecoder, reconstruction_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_blocks_per_sample(
    blocks: Sequence[RowBlock], limit: int | None
) -> list[RowBlock]:
    if limit is None:
        return list(blocks)
    selected: list[RowBlock] = []
    for sample in sorted({block.sample for block in blocks}):
        selected.extend([block for block in blocks if block.sample == sample][:limit])
    return selected


def compute_training_statistics(
    reader: H5ADBatchReader, blocks: Sequence[RowBlock]
) -> dict[str, torch.Tensor | int | float]:
    latent_sum = np.zeros(reader.latent_dim, dtype=np.float64)
    latent_square_sum = np.zeros(reader.latent_dim, dtype=np.float64)
    gene_sums = np.zeros(reader.n_genes, dtype=np.float64)
    log_library_sum = 0.0
    log_library_square_sum = 0.0
    n_cells = 0
    library_min = math.inf
    library_max = -math.inf
    for _block, latent, counts, _times in reader.iter_blocks(blocks):
        library = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.float64)
        log_library = np.log1p(library)
        latent_sum += latent.sum(axis=0, dtype=np.float64)
        latent_square_sum += np.square(latent, dtype=np.float64).sum(axis=0)
        gene_sums += np.asarray(counts.sum(axis=0)).reshape(-1)
        log_library_sum += float(log_library.sum())
        log_library_square_sum += float(np.square(log_library).sum())
        library_min = min(library_min, float(library.min()))
        library_max = max(library_max, float(library.max()))
        n_cells += len(latent)
    if n_cells < 2 or gene_sums.sum() <= 0:
        raise ValueError("training split has insufficient cells or raw counts")
    latent_mean = latent_sum / n_cells
    latent_variance = np.maximum(latent_square_sum / n_cells - latent_mean**2, 1e-8)
    log_library_mean = log_library_sum / n_cells
    log_library_variance = max(
        log_library_square_sum / n_cells - log_library_mean**2, 1e-8
    )
    return {
        "latent_mean": torch.tensor(latent_mean, dtype=torch.float32),
        "latent_std": torch.tensor(np.sqrt(latent_variance), dtype=torch.float32),
        "log_library_mean": torch.tensor(log_library_mean, dtype=torch.float32),
        "log_library_std": torch.tensor(
            math.sqrt(log_library_variance), dtype=torch.float32
        ),
        "gene_sums": torch.tensor(gene_sums, dtype=torch.float32),
        "n_cells": n_cells,
        "raw_library_min": library_min,
        "raw_library_max": library_max,
    }


def _run_blocks(
    model: CountAwareDecoder,
    reader: H5ADBatchReader,
    blocks: Sequence[RowBlock],
    device: torch.device,
    *,
    library_weight: float,
    optimizer: torch.optim.Optimizer | None,
    use_bfloat16: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: defaultdict[str, float] = defaultdict(float)
    n_cells = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for _block, latent, sparse_counts, times in reader.iter_blocks(blocks):
            latent_tensor = torch.from_numpy(latent).to(device, non_blocking=True)
            counts_tensor = torch.from_numpy(
                sparse_counts.toarray().astype(np.float32, copy=False)
            ).to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bfloat16,
            ):
                output = model(latent_tensor)
                losses = reconstruction_loss(
                    output, counts_tensor, library_weight=library_weight
                )
            if not torch.isfinite(losses.total):
                raise FloatingPointError("decoder loss became non-finite")
            if training:
                losses.total.backward()
                clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_cells = len(latent)
            n_cells += batch_cells
            totals["total"] += float(losses.total.detach()) * batch_cells
            totals["nb_nll"] += float(losses.nb_nll.detach()) * batch_cells
            totals["library"] += float(losses.library.detach()) * batch_cells
    return {name: value / n_cells for name, value in totals.items()}


def _checkpoint_payload(
    model: CountAwareDecoder,
    config: TrainConfig,
    reader: H5ADBatchReader,
    statistics: dict[str, torch.Tensor | int | float],
    rollout_normalization,
    *,
    epoch: int,
    validation_loss: float,
) -> dict:
    normalization_payload = None
    if rollout_normalization is not None:
        normalization_payload = {
            "mean": torch.from_numpy(rollout_normalization.mean.copy()),
            "std": torch.from_numpy(rollout_normalization.std.copy()),
            "source": rollout_normalization.source,
        }
    return {
        "schema_version": 1,
        "time_conditioning": False,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_config": model.architecture_config(),
        "gene_names": reader.gene_names.astype(str).tolist(),
        "sample_key": config.sample_key,
        "sample_times": dict(config.sample_times),
        "data_path": str(config.data_path),
        "latent_key": config.latent_key,
        "counts_key": config.counts_key,
        "target_scale": "integer_raw_counts",
        "prediction_scale": "negative_binomial_expected_counts",
        "rollout_normalization": normalization_payload,
        "statistics": {
            key: value
            for key, value in statistics.items()
            if isinstance(value, torch.Tensor)
        },
        "epoch": int(epoch),
        "best_validation_loss": float(validation_loss),
    }


def train_decoder(config: TrainConfig) -> Path:
    config.validate()
    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "checkpoints" / config.checkpoint_name
    artifact_stem = Path(config.checkpoint_name).stem
    if checkpoint_path.is_file() and not config.overwrite:
        return checkpoint_path
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({config.device}) but unavailable")

    set_seed(config.seed)
    device = torch.device(config.device)
    reader = H5ADBatchReader(
        config.data_path, latent_key=config.latent_key, counts_key=config.counts_key
    )
    all_blocks = build_sample_blocks(
        reader.obs,
        sample_key=config.sample_key,
        sample_times=config.sample_times,
        batch_size=config.batch_size,
    )
    splits = split_sample_blocks(
        all_blocks,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    splits = {
        name: limit_blocks_per_sample(blocks, config.max_blocks_per_split)
        for name, blocks in splits.items()
    }
    statistics = compute_training_statistics(reader, splits["train"])

    rollout_normalization = None
    if config.rollout_normalization_checkpoint is not None:
        rollout_normalization = load_rollout_normalization(
            config.rollout_normalization_checkpoint
        )
        if rollout_normalization.mean.size != reader.latent_dim:
            raise ValueError(
                "Stage-1 f_mu/f_std dimensions do not match the training latent"
            )

    model = CountAwareDecoder(
        reader.latent_dim,
        reader.n_genes,
        hidden_dim=config.hidden_dim,
        factor_dim=config.factor_dim,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
        skip_weight=config.skip_weight,
    ).to(device)
    model.set_input_statistics(
        statistics["latent_mean"],
        statistics["latent_std"],
        statistics["log_library_mean"],
        statistics["log_library_std"],
    )
    model.initialize_gene_bias(statistics["gene_sums"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"config_{artifact_stem}.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    stale_epochs = 0
    progress = tqdm(
        range(1, config.epochs + 1),
        desc="Train decoder",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in progress:
        rng = np.random.default_rng(config.seed + epoch)
        train_blocks = [
            splits["train"][int(index)]
            for index in rng.permutation(len(splits["train"]))
        ]
        train_metrics = _run_blocks(
            model,
            reader,
            train_blocks,
            device,
            library_weight=config.library_weight,
            optimizer=optimizer,
            use_bfloat16=use_bfloat16,
        )
        validation_metrics = _run_blocks(
            model,
            reader,
            splits["validation"],
            device,
            library_weight=config.library_weight,
            optimizer=None,
            use_bfloat16=use_bfloat16,
        )
        scheduler.step(validation_metrics["total"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(row)
        progress.set_postfix(
            train=f"{train_metrics['total']:.6f}",
            val=f"{validation_metrics['total']:.6f}",
            best=f"{min(best_validation, validation_metrics['total']):.6f}",
        )
        if validation_metrics["total"] < best_validation - 1e-6:
            best_validation = validation_metrics["total"]
            stale_epochs = 0
            save_decoder_checkpoint(
                checkpoint_path,
                _checkpoint_payload(
                    model,
                    config,
                    reader,
                    statistics,
                    rollout_normalization,
                    epoch=epoch,
                    validation_loss=best_validation,
                ),
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    pd.DataFrame(history).to_csv(
        output_dir / f"training_history_{artifact_stem}.csv", index=False
    )
    if not checkpoint_path.is_file():
        raise RuntimeError("training completed without a finite decoder checkpoint")
    from .checkpoint import load_decoder_checkpoint

    best_model, _payload = load_decoder_checkpoint(checkpoint_path, device)
    test_metrics = _run_blocks(
        best_model,
        reader,
        splits["test"],
        device,
        library_weight=config.library_weight,
        optimizer=None,
        use_bfloat16=use_bfloat16,
    )
    scale_audit = {
        "data_path": str(config.data_path),
        "latent_key": config.latent_key,
        "counts_key": config.counts_key,
        "sample_key": config.sample_key,
        "sample_times": config.sample_times,
        "n_cells": reader.n_cells,
        "latent_dim": reader.latent_dim,
        "n_genes": reader.n_genes,
        "training_cells": statistics["n_cells"],
        "raw_library_min": statistics["raw_library_min"],
        "raw_library_max": statistics["raw_library_max"],
        "target_scale": "integer_raw_counts",
        "library_target": "log1p(raw_counts.sum(axis=1))",
        "prediction_scale": "negative_binomial_expected_counts",
        "nb_uses_log1p": False,
        "expected_counts_are_samples": False,
        "rollout_normalization_source": (
            rollout_normalization.source if rollout_normalization is not None else None
        ),
        "test_loss": test_metrics,
    }
    (output_dir / f"scale_audit_{artifact_stem}.json").write_text(
        json.dumps(scale_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path
