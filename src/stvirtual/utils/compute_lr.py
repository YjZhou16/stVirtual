"""GPU-friendly ligand-receptor potential utilities used by Stage-2 models."""

from __future__ import annotations

from typing import Mapping

import torch


def knn_self_chunked(
    coords: torch.Tensor,
    k: int = 16,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute self-KNN in chunks without allocating a full distance matrix."""

    if coords.ndim != 2:
        raise ValueError("coords must have shape (n_cells, spatial_dim)")
    if k <= 0 or chunk_size <= 0:
        raise ValueError("k and chunk_size must be positive")
    cell_count = int(coords.shape[0])
    neighbor_count = min(int(k), max(cell_count - 1, 0))
    distances = torch.empty(
        (cell_count, neighbor_count), device=coords.device, dtype=coords.dtype
    )
    indices = torch.empty(
        (cell_count, neighbor_count), device=coords.device, dtype=torch.long
    )
    if neighbor_count == 0:
        return distances, indices

    with torch.no_grad():
        for start in range(0, cell_count, chunk_size):
            end = min(start + chunk_size, cell_count)
            pairwise = torch.cdist(coords[start:end], coords)
            values, neighbors = torch.topk(
                pairwise, neighbor_count + 1, largest=False
            )
            distances[start:end] = values[:, 1:]
            indices[start:end] = neighbors[:, 1:]
    return distances, indices


def compute_lr_potential_gpu(
    expr_union_t: torch.Tensor,
    coords_t: torch.Tensor,
    lr_cfg: Mapping[str, torch.Tensor],
    k: int = 16,
    use_cpm: bool = True,
    lib_power: float = 0.5,
    lib_clip_q: tuple[float, float] | None = (5.0, 95.0),
    hill: bool = True,
    combine_mode: str = "zdiff",
    pair_chunk: int = 64,
    knn_chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute per-cell ligand-receptor potential from decoded expression."""

    if expr_union_t.ndim != 2 or coords_t.ndim != 2:
        raise ValueError("expression and coordinates must both be two-dimensional")
    if expr_union_t.shape[0] != coords_t.shape[0]:
        raise ValueError("expression and coordinates must contain the same cells")
    cell_count = int(expr_union_t.shape[0])
    if cell_count < 2:
        return torch.zeros(cell_count, device=expr_union_t.device, dtype=torch.float32)

    distances, neighbors = knn_self_chunked(
        coords_t, k=k, chunk_size=knn_chunk_size
    )
    sigma = torch.clamp(distances[:, -1], min=1e-6)
    spatial_kernel = torch.exp(
        -(distances**2) / (2.0 * sigma[:, None] ** 2)
    )

    if use_cpm:
        library_size = expr_union_t.sum(dim=1).to(torch.float32)
        if lib_clip_q is not None:
            low, high = torch.quantile(
                library_size,
                torch.tensor(
                    [lib_clip_q[0] / 100.0, lib_clip_q[1] / 100.0],
                    device=expr_union_t.device,
                ),
            )
            low = torch.maximum(low, torch.tensor(1.0, device=expr_union_t.device))
            high = torch.maximum(high, low + 1.0)
            library_size = torch.clamp(library_size, min=low, max=high)
        scale = (1e4 / library_size).pow(float(lib_power))
    else:
        scale = torch.ones(cell_count, device=expr_union_t.device)

    ligand = expr_union_t[:, lr_cfg["lig_idx"]] * scale.view(-1, 1)
    receptor = expr_union_t[:, lr_cfg["rec_idx"]] * scale.view(-1, 1)
    positive = torch.zeros(cell_count, device=expr_union_t.device)
    negative = torch.zeros(cell_count, device=expr_union_t.device)

    for start in range(0, ligand.shape[1], pair_chunk):
        end = min(start + pair_chunk, ligand.shape[1])
        ligand_neighbor = ligand[:, start:end][neighbors]
        receptor_cell = receptor[:, start:end].unsqueeze(1)
        if hill:
            exponent = lr_cfg["nH"][start:end].view(1, 1, -1)
            ligand_power = torch.clamp(ligand_neighbor, min=0.0).pow(exponent)
            receptor_power = torch.clamp(receptor_cell, min=0.0).pow(exponent)
            interaction = (
                ligand_power
                / (lr_cfg["KL"][start:end].view(1, 1, -1).pow(exponent) + ligand_power + 1e-8)
            ) * (
                receptor_power
                / (lr_cfg["KR"][start:end].view(1, 1, -1).pow(exponent) + receptor_power + 1e-8)
            )
        else:
            raw = ligand_neighbor * receptor_cell
            denominator = torch.clamp(torch.quantile(raw.reshape(-1), 0.99), min=1e-6)
            interaction = torch.clamp(raw / denominator, min=0.0, max=1.0)
        positive += (
            (interaction * lr_cfg["w_pos"][start:end].view(1, 1, -1)).sum(dim=2)
            * spatial_kernel
        ).sum(dim=1)
        negative += (
            (interaction * lr_cfg["w_neg"][start:end].view(1, 1, -1)).sum(dim=2)
            * spatial_kernel
        ).sum(dim=1)

    if combine_mode in {"sum", "diff"}:
        potential = positive - negative
    elif combine_mode == "zdiff":
        pos_z = (positive - positive.mean()) / (positive.std(correction=0) + 1e-6)
        neg_z = (negative - negative.mean()) / (negative.std(correction=0) + 1e-6)
        potential = pos_z - neg_z
    else:
        raise ValueError(f"unsupported combine_mode: {combine_mode!r}")
    return potential.to(torch.float32)


def compute_lr_potential_per_sample(
    adata,
    lr_pairs,
    *,
    sample_key: str = "sample",
    coords_key: str = "spatial_aligned",
    layer: str | None = "counts",
    k: int = 16,
    use_cpm: bool = True,
    hill: bool = True,
    pair_chunk: int = 64,
    device: str | torch.device | None = None,
    output_key: str = "LR_potential",
    z_output_key: str = "LR_potential_z",
):
    """Compute LR potential independently within each sample.

    The function updates ``adata.obs`` in place and returns ``adata`` for
    convenient notebook chaining. Coordinates must be stored in ``adata.obsm``;
    expression is read from ``layer`` when present, otherwise from ``adata.X``.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    if sample_key not in adata.obs:
        raise KeyError(f"adata.obs does not contain {sample_key!r}")
    if coords_key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain {coords_key!r}")
    required = {"ligand", "receptor"}
    if not required <= set(lr_pairs.columns):
        raise ValueError("lr_pairs must contain ligand and receptor columns")

    compute_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    expression = adata.layers[layer] if layer and layer in adata.layers else adata.X
    coordinates = np.asarray(adata.obsm[coords_key], dtype=np.float32)
    samples = adata.obs[sample_key].astype(str).to_numpy()
    gene_to_index = {str(gene): index for index, gene in enumerate(adata.var_names)}

    pairs = lr_pairs.copy()
    pairs = pairs[
        pairs["ligand"].astype(str).isin(gene_to_index)
        & pairs["receptor"].astype(str).isin(gene_to_index)
    ].reset_index(drop=True)
    if pairs.empty:
        raise ValueError("no ligand-receptor pairs match adata.var_names")

    ligand_indices = [gene_to_index[str(gene)] for gene in pairs["ligand"]]
    receptor_indices = [gene_to_index[str(gene)] for gene in pairs["receptor"]]
    def numeric_series(column: str, default: float):
        source = pairs[column] if column in pairs else pd.Series(default, index=pairs.index)
        return pd.to_numeric(source, errors="coerce").fillna(default)

    if "sign" in pairs:
        sign_source = pairs["sign"].replace({"+": 1.0, "-": -1.0})
        sign = pd.to_numeric(sign_source, errors="coerce").fillna(1.0)
    else:
        sign = pd.Series(1.0, index=pairs.index)
    weight = numeric_series("weight", 1.0)

    def values(column: str, default: float) -> torch.Tensor:
        series = numeric_series(column, default)
        return torch.as_tensor(
            series.fillna(default).to_numpy(np.float32), device=compute_device
        )

    config = {
        "lig_idx": torch.as_tensor(ligand_indices, dtype=torch.long, device=compute_device),
        "rec_idx": torch.as_tensor(receptor_indices, dtype=torch.long, device=compute_device),
        "KL": values("KL", 5.0),
        "KR": values("KR", 5.0),
        "nH": values("hill_n", 1.0),
        "w_pos": torch.as_tensor(
            np.clip((weight * sign).to_numpy(np.float32), 0.0, None),
            device=compute_device,
        ),
        "w_neg": torch.as_tensor(
            np.clip(-(weight * sign).to_numpy(np.float32), 0.0, None),
            device=compute_device,
        ),
    }

    potential = np.zeros(adata.n_obs, dtype=np.float32)
    potential_z = np.zeros(adata.n_obs, dtype=np.float32)
    for sample in pd.unique(samples):
        indices = np.flatnonzero(samples == sample)
        sample_expression = expression[indices]
        if sparse.issparse(sample_expression):
            sample_expression = sample_expression.toarray()
        expression_tensor = torch.as_tensor(
            np.asarray(sample_expression, dtype=np.float32), device=compute_device
        )
        coordinate_tensor = torch.as_tensor(
            coordinates[indices], dtype=torch.float32, device=compute_device
        )
        sample_potential = compute_lr_potential_gpu(
            expression_tensor,
            coordinate_tensor,
            config,
            k=k,
            use_cpm=use_cpm,
            hill=hill,
            pair_chunk=pair_chunk,
        ).detach().cpu().numpy()
        potential[indices] = sample_potential
        standard_deviation = float(sample_potential.std())
        if standard_deviation > 1e-6:
            potential_z[indices] = (
                sample_potential - float(sample_potential.mean())
            ) / standard_deviation

    adata.obs[output_key] = potential
    adata.obs[z_output_key] = potential_z
    return adata
