"""Canonical data-access contracts for public stVirtual interfaces."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LATENT_KEY = "X_scanVI"
LEGACY_LATENT_KEY = "X_scVI"


def _as_valid_latent(value: Any, *, n_obs: int, key: str) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    try:
        latent = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"latent representation {key!r} must be numeric") from error
    if latent.ndim != 2 or latent.shape[0] != n_obs or latent.shape[1] == 0:
        raise ValueError(
            f"latent representation {key!r} must have shape (n_obs, latent_dim)"
        )
    if not np.all(np.isfinite(latent)):
        raise ValueError(f"latent representation {key!r} contains NaN or Inf")
    return latent


def read_latent(adata: Any, *, latent_key: str = DEFAULT_LATENT_KEY) -> np.ndarray:
    """Read a numeric latent matrix from ``adata.obsm``.

    ``X_scanVI`` is recommended and used by default. When that default is
    requested but absent, the legacy ``X_scVI`` key is migrated with a warning.
    Any explicitly selected custom key is strict and never falls back to another
    key, an AnnData layer, or ``adata.X``.
    """

    if not isinstance(latent_key, str) or not latent_key:
        raise ValueError("latent_key must be a non-empty string")
    if latent_key in adata.obsm:
        return _as_valid_latent(adata.obsm[latent_key], n_obs=adata.n_obs, key=latent_key)
    if latent_key != DEFAULT_LATENT_KEY:
        raise KeyError(f"requested latent key {latent_key!r} was not found in adata.obsm")
    if LEGACY_LATENT_KEY not in adata.obsm:
        raise KeyError(
            f"recommended latent key {DEFAULT_LATENT_KEY!r} was not found in adata.obsm"
        )

    warnings.warn(
        f"adata.obsm[{LEGACY_LATENT_KEY!r}] is deprecated; migrating it to "
        f"adata.obsm[{DEFAULT_LATENT_KEY!r}]",
        UserWarning,
        stacklevel=2,
    )
    latent = _as_valid_latent(
        adata.obsm[LEGACY_LATENT_KEY], n_obs=adata.n_obs, key=LEGACY_LATENT_KEY
    )
    adata.obsm[DEFAULT_LATENT_KEY] = latent.copy()
    return latent


def canonicalize_celltypes(
    adata: Any,
    *,
    source_key: str,
    celltype_key: str = "celltype",
    celltype_id_key: str = "celltype_id",
) -> pd.DataFrame:
    """Create canonical cell-type columns and a deterministic mapping table."""

    if source_key not in adata.obs:
        raise KeyError(f"cell type source key {source_key!r} was not found in adata.obs")
    source = adata.obs[source_key]
    if source.isna().any():
        raise ValueError(f"cell type source key {source_key!r} contains missing values")
    labels = source.astype(str)
    categories = sorted(labels.unique().tolist())
    if not categories:
        raise ValueError("cell type mapping cannot be empty")
    label_to_id = {label: index for index, label in enumerate(categories)}
    adata.obs[celltype_key] = labels.to_numpy()
    adata.obs[celltype_id_key] = labels.map(label_to_id).to_numpy(dtype=np.int64)
    return pd.DataFrame(
        {
            "celltype_id": np.arange(len(categories), dtype=np.int64),
            "celltype": categories,
            "source_celltype": categories,
        }
    )


def attach_rollout_contract(
    adata: Any,
    *,
    latent: Any,
    celltypes: Sequence[str],
    mapping_path: str | Path,
    celltype_ids: Sequence[int] | None = None,
    write_mapping: bool = True,
) -> pd.DataFrame:
    """Attach canonical public rollout fields and write their mapping table."""

    latent_array = _as_valid_latent(latent, n_obs=adata.n_obs, key="X_latent")
    labels = pd.Series(celltypes, dtype="string")
    if len(labels) != adata.n_obs:
        raise ValueError("celltypes length must equal adata.n_obs")
    if labels.isna().any():
        raise ValueError("rollout celltypes contain missing values")
    if celltype_ids is None:
        categories = sorted(labels.astype(str).unique().tolist())
        label_to_id = {label: index for index, label in enumerate(categories)}
        ids = labels.astype(str).map(label_to_id).to_numpy(dtype=np.int64)
    else:
        ids = np.asarray(celltype_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != adata.n_obs or np.any(ids < 0):
            raise ValueError("celltype_ids must be non-negative and match adata.n_obs")
        pairs = pd.DataFrame({"celltype_id": ids, "celltype": labels.astype(str)})
        if pairs.groupby("celltype_id")["celltype"].nunique().max() != 1:
            raise ValueError("each celltype_id must map to exactly one celltype")
        mapping = pairs.drop_duplicates().sort_values("celltype_id").reset_index(drop=True)
        if mapping["celltype"].duplicated().any():
            raise ValueError("each celltype must map to exactly one celltype_id")

    adata.obsm["X_latent"] = latent_array
    adata.obs["celltype"] = labels.astype(str).to_numpy()
    adata.obs["celltype_id"] = ids
    if celltype_ids is None:
        mapping = pd.DataFrame(
            {
                "celltype_id": np.arange(len(categories), dtype=np.int64),
                "celltype": categories,
            }
        )
    mapping["source_celltype"] = mapping["celltype"]
    if write_mapping:
        destination = Path(mapping_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(destination, index=False)
    return mapping


def _time_tag(value: float) -> str:
    return f"{float(value):.4f}".replace(".", "p")


def save_rollout_frames(
    rollout: dict[str, Any],
    *,
    celltype_names: Sequence[str],
    output_dir: str | Path,
    prefix: str = "rollout",
) -> list[Path]:
    """Save model rollout frames with the canonical public AnnData contract."""

    import anndata as ad

    required = ("coords", "latent", "layers", "t")
    missing = [key for key in required if key not in rollout]
    if missing:
        raise KeyError(f"rollout is missing required fields: {', '.join(missing)}")
    frame_count = len(rollout["coords"])
    if any(len(rollout[key]) != frame_count for key in required):
        raise ValueError("rollout frame fields must have equal lengths")
    names = [str(name) for name in celltype_names]
    if not names:
        raise ValueError("celltype_names cannot be empty")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mapping_path = destination / "celltype_mapping.csv"
    pd.DataFrame(
        {
            "celltype_id": np.arange(len(names), dtype=np.int64),
            "celltype": names,
            "source_celltype": names,
        }
    ).to_csv(mapping_path, index=False)
    paths: list[Path] = []
    for frame in range(frame_count):
        coords = np.asarray(rollout["coords"][frame], dtype=np.float32)
        latent = np.asarray(rollout["latent"][frame], dtype=np.float32)
        ids = np.asarray(rollout["layers"][frame], dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(names)):
            raise ValueError("rollout layers contain an invalid celltype_id")
        frame_adata = ad.AnnData(X=latent.copy())
        frame_adata.obsm["spatial"] = coords
        attach_rollout_contract(
            frame_adata,
            latent=latent,
            celltypes=[names[index] for index in ids],
            celltype_ids=ids,
            mapping_path=mapping_path,
            write_mapping=False,
        )
        frame_adata.obs["rollout_frame"] = frame
        frame_adata.obs["rollout_time"] = float(rollout["t"][frame])
        path = destination / (
            f"{prefix}_f{frame:04d}_t{_time_tag(rollout['t'][frame])}.h5ad"
        )
        frame_adata.write_h5ad(path, compression="gzip")
        paths.append(path)
    return paths
