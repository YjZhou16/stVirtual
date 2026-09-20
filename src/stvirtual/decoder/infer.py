from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .checkpoint import load_decoder_checkpoint, load_rollout_normalization
from .model import denormalize_latent


def normalize_latent_key(key: str) -> str:
    value = str(key).strip().strip("/")
    if value == "X" or "/" in value:
        return value
    return f"obsm/{value}"


def normalize_output_key(key: str) -> str:
    value = str(key).strip().strip("/")
    if value == "X":
        return value
    if "/" not in value:
        value = f"layers/{value}"
    if not value.startswith("layers/") or value.count("/") != 1:
        raise ValueError("output key must be X, a layer name, or layers/<name>")
    return value


def _prepare_output(
    input_path: Path,
    partial_path: Path,
    gene_names: list[str],
    checkpoint_path: Path,
    output_key: str,
) -> int:
    source = ad.read_h5ad(input_path, backed="r")
    try:
        n_cells = source.n_obs
        obs = source.obs.copy()
        obs["predicted_library_size"] = np.zeros(n_cells, dtype=np.float32)
        output = ad.AnnData(
            X=sp.csr_matrix((n_cells, len(gene_names)), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
        )
        for key in source.obsm.keys():
            output.obsm[key] = np.asarray(source.obsm[key])
        output.uns.update(dict(source.uns))
        output.uns["decoder_checkpoint"] = str(checkpoint_path)
        output.uns["expression_key"] = output_key
        output.uns["expression_scale"] = "negative_binomial_expected_counts"
        output.uns["expression_is_sampled_counts"] = False
        output.uns["nb_mean_symbol"] = "mu"
        output.write_h5ad(partial_path)
    finally:
        source.file.close()
    return n_cells


def _normalization_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray] | None:
    stored = payload.get("rollout_normalization")
    if stored is None:
        return None
    try:
        mean = stored["mean"].detach().cpu().numpy().astype(np.float32)
        std = stored["std"].detach().cpu().numpy().astype(np.float32)
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("decoder checkpoint has invalid rollout normalization") from error
    return mean.reshape(-1), std.reshape(-1)


def decode_h5ad(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    latent_key: str,
    latent_is_normalized: bool,
    output_key: str = "layers/reconstructed_expression",
    normalization_checkpoint: Path | None = None,
    device_name: str = "cuda:0",
    batch_size: int = 256,
    overwrite: bool = False,
) -> Path:
    """Stream decoder NB means (mu), not sampled counts, into an H5AD file."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass overwrite=True: {output_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({device_name}) but unavailable")

    latent_path = normalize_latent_key(latent_key)
    output_matrix_path = normalize_output_key(output_key)
    device = torch.device(device_name)
    model, payload = load_decoder_checkpoint(checkpoint_path, device)
    gene_names = list(map(str, payload["gene_names"]))
    if len(gene_names) != model.n_genes:
        raise ValueError("checkpoint gene names do not match decoder output dimension")

    normalization = _normalization_from_payload(payload)
    if normalization_checkpoint is not None:
        external = load_rollout_normalization(normalization_checkpoint)
        normalization = (external.mean, external.std)
    if latent_is_normalized and normalization is None:
        raise ValueError(
            "normalized rollout latent requires f_mu/f_std in decoder or external checkpoint"
        )

    with h5py.File(input_path, "r") as source_handle:
        if latent_path not in source_handle:
            raise KeyError(f"input H5AD has no latent matrix {latent_path!r}")
        latent_dataset = source_handle[latent_path]
        if not isinstance(latent_dataset, h5py.Dataset) or latent_dataset.ndim != 2:
            raise TypeError("inference latent must be a dense two-dimensional array")
        n_cells, latent_dim = map(int, latent_dataset.shape)
    if latent_dim != model.latent_dim:
        raise ValueError(
            f"inference latent dimension {latent_dim} != decoder dimension {model.latent_dim}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.h5ad")
    if partial_path.exists():
        partial_path.unlink()
    prepared_cells = _prepare_output(
        input_path,
        partial_path,
        gene_names,
        checkpoint_path,
        output_matrix_path,
    )
    if prepared_cells != n_cells:
        raise ValueError("input latent rows do not match input obs rows")

    try:
        with h5py.File(input_path, "r") as source_handle, h5py.File(
            partial_path, "r+"
        ) as output_handle:
            if output_matrix_path == "X":
                del output_handle["X"]
                parent = output_handle
                name = "X"
            else:
                parent_name, name = output_matrix_path.split("/", 1)
                parent = output_handle.require_group(parent_name)
                if name in parent:
                    del parent[name]
            expression = parent.create_dataset(
                name,
                shape=(n_cells, model.n_genes),
                dtype=np.float32,
                chunks=(min(batch_size, max(n_cells, 1)), min(1024, model.n_genes)),
                compression="lzf",
            )
            expression.attrs["encoding-type"] = "array"
            expression.attrs["encoding-version"] = "0.2.0"
            library_dataset = output_handle["obs/predicted_library_size"]
            latent_dataset = source_handle[latent_path]
            model.eval()
            with torch.inference_mode():
                for start in range(0, n_cells, batch_size):
                    stop = min(start + batch_size, n_cells)
                    latent = np.asarray(latent_dataset[start:stop], dtype=np.float32)
                    if not np.all(np.isfinite(latent)):
                        raise ValueError("inference latent contains non-finite values")
                    if latent_is_normalized:
                        latent = denormalize_latent(latent, *normalization)
                    latent_tensor = torch.from_numpy(latent).to(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
                    ):
                        decoded = model(latent_tensor)
                    expected = decoded.expected_counts.float().cpu().numpy()
                    library = decoded.library_size.float().cpu().numpy()
                    if not np.all(np.isfinite(expected)) or np.any(expected < 0):
                        raise ValueError("decoder produced invalid NB expected counts")
                    np.testing.assert_allclose(
                        expected.sum(axis=1), library, rtol=2e-4, atol=1e-3
                    )
                    expression[start:stop] = expected
                    library_dataset[start:stop] = library
            output_handle.flush()
        os.replace(partial_path, output_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise
    return output_path
