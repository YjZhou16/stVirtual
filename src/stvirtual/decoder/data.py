from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class RowBlock:
    sample: str
    time: float
    start: int
    stop: int

    @property
    def n_rows(self) -> int:
        return self.stop - self.start


def normalize_h5ad_key(key: str, container: str) -> str:
    value = str(key).strip().strip("/")
    if not value:
        raise ValueError(f"{container} key cannot be empty")
    return value if "/" in value else f"{container}/{value}"


def matrix_shape(obj: h5py.Dataset | h5py.Group) -> tuple[int, int]:
    if isinstance(obj, h5py.Dataset):
        if obj.ndim != 2:
            raise ValueError("expression matrix must be two-dimensional")
        return tuple(map(int, obj.shape))
    shape = obj.attrs.get("shape")
    if shape is None or len(shape) != 2:
        raise ValueError("sparse expression matrix is missing a two-dimensional shape")
    return tuple(map(int, shape))


def read_csr_rows(group: h5py.Group, start: int, stop: int) -> sp.csr_matrix:
    shape = matrix_shape(group)
    if str(group.attrs.get("encoding-type", "")) != "csr_matrix":
        raise TypeError("only dense and CSR H5AD count matrices are supported")
    if not 0 <= start <= stop <= shape[0]:
        raise IndexError(f"invalid row range [{start}, {stop}) for shape {shape}")
    pointers = np.asarray(group["indptr"][start : stop + 1], dtype=np.int64)
    data_start, data_stop = int(pointers[0]), int(pointers[-1])
    data = np.asarray(group["data"][data_start:data_stop], dtype=np.float32)
    indices = np.asarray(group["indices"][data_start:data_stop], dtype=np.int32)
    return sp.csr_matrix(
        (data, indices, pointers - data_start),
        shape=(stop - start, shape[1]),
        dtype=np.float32,
    )


def validate_raw_count_array(counts: np.ndarray | sp.spmatrix) -> None:
    values = counts.data if sp.issparse(counts) else np.asarray(counts)
    values = np.asarray(values)
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("raw counts must be finite and non-negative")
    if not np.all(values == np.floor(values)):
        raise ValueError("count matrix must contain integer raw counts, not log1p counts")


class H5ADBatchReader:
    """Stream contiguous rows from configurable latent and raw-count H5AD keys."""

    def __init__(self, path: Path, *, latent_key: str, counts_key: str):
        self.path = Path(path)
        self.latent_path = normalize_h5ad_key(latent_key, "obsm")
        self.counts_path = normalize_h5ad_key(counts_key, "layers")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as handle:
            if self.latent_path not in handle:
                raise KeyError(f"missing latent matrix {self.latent_path!r}")
            if self.counts_path not in handle:
                raise KeyError(f"missing raw-count matrix {self.counts_path!r}")
            latent = handle[self.latent_path]
            counts = handle[self.counts_path]
            if not isinstance(latent, h5py.Dataset) or latent.ndim != 2:
                raise TypeError("latent matrix must be a dense two-dimensional H5AD array")
            self.n_cells, self.latent_dim = map(int, latent.shape)
            count_cells, self.n_genes = matrix_shape(counts)
            if count_cells != self.n_cells:
                raise ValueError("latent and raw-count matrices have different cell counts")
            self.obs = ad.io.read_elem(handle["obs"])
            var = ad.io.read_elem(handle["var"])
            self.gene_names = var.index.astype(str).to_numpy()
        if len(self.obs) != self.n_cells or len(self.gene_names) != self.n_genes:
            raise ValueError("H5AD obs/var dimensions do not match latent/count matrices")

    def read_rows(self, start: int, stop: int) -> tuple[np.ndarray, sp.csr_matrix]:
        if not 0 <= start <= stop <= self.n_cells:
            raise IndexError(f"invalid row range [{start}, {stop})")
        with h5py.File(self.path, "r") as handle:
            latent = np.asarray(self._read_dense_rows(handle[self.latent_path], start, stop))
            count_obj = handle[self.counts_path]
            if isinstance(count_obj, h5py.Dataset):
                dense = np.asarray(count_obj[start:stop], dtype=np.float32)
                counts = sp.csr_matrix(dense)
            else:
                counts = read_csr_rows(count_obj, start, stop)
        latent = latent.astype(np.float32, copy=False)
        if not np.all(np.isfinite(latent)):
            raise ValueError("latent matrix contains non-finite values")
        validate_raw_count_array(counts)
        return latent, counts

    @staticmethod
    def _read_dense_rows(dataset: h5py.Dataset, start: int, stop: int) -> np.ndarray:
        return np.asarray(dataset[start:stop], dtype=np.float32)

    def iter_blocks(
        self, blocks: Sequence[RowBlock]
    ) -> Iterator[tuple[RowBlock, np.ndarray, sp.csr_matrix, np.ndarray]]:
        for block in blocks:
            latent, counts = self.read_rows(block.start, block.stop)
            times = np.full(block.n_rows, block.time, dtype=np.float32)
            yield block, latent, counts, times


def _contiguous_runs(rows: np.ndarray) -> Iterator[tuple[int, int]]:
    if rows.size == 0:
        return
    starts = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1]
    stops = np.r_[starts[1:], rows.size]
    for left, right in zip(starts, stops):
        yield int(rows[left]), int(rows[right - 1] + 1)


def build_sample_blocks(
    obs: pd.DataFrame,
    *,
    sample_key: str,
    sample_times: Mapping[str, float],
    batch_size: int,
) -> list[RowBlock]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sample_key not in obs:
        raise KeyError(f"sample key {sample_key!r} is absent from obs")
    if len(sample_times) < 2:
        raise ValueError("at least two sample-to-time mappings are required")
    values = obs[sample_key].astype(str).to_numpy()
    blocks: list[RowBlock] = []
    for sample, time in sample_times.items():
        if not np.isfinite(float(time)):
            raise ValueError(f"time for sample {sample!r} is not finite")
        rows = np.flatnonzero(values == str(sample))
        if rows.size == 0:
            raise ValueError(f"no cells found for {sample_key}={sample!r}")
        if rows.size < 3:
            raise ValueError(
                f"sample {sample!r} needs at least three cells for train, validation, and test splits"
            )
        effective_batch_size = min(batch_size, max(1, (int(rows.size) + 2) // 3))
        for run_start, run_stop in _contiguous_runs(rows):
            for start in range(run_start, run_stop, effective_batch_size):
                blocks.append(
                    RowBlock(
                        str(sample),
                        float(time),
                        start,
                        min(start + effective_batch_size, run_stop),
                    )
                )
    return blocks


def split_sample_blocks(
    blocks: Sequence[RowBlock],
    *,
    seed: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[RowBlock]]:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    rng = np.random.default_rng(seed)
    splits: dict[str, list[RowBlock]] = {"train": [], "validation": [], "test": []}
    samples = sorted({block.sample for block in blocks})
    for sample in samples:
        sample_blocks = [block for block in blocks if block.sample == sample]
        if len(sample_blocks) < 3:
            raise ValueError(f"sample {sample!r} needs at least three streamed blocks")
        order = rng.permutation(len(sample_blocks))
        n_test = max(1, int(round(len(order) * test_fraction)))
        n_validation = max(1, int(round(len(order) * validation_fraction)))
        if n_test + n_validation >= len(order):
            n_test = n_validation = 1
        assignments = {
            "test": order[:n_test],
            "validation": order[n_test : n_test + n_validation],
            "train": order[n_test + n_validation :],
        }
        for split, positions in assignments.items():
            splits[split].extend(sample_blocks[int(position)] for position in positions)
    for split, selected in splits.items():
        if not selected:
            raise ValueError(f"{split} split is empty")
        splits[split] = [selected[int(i)] for i in rng.permutation(len(selected))]
    return splits

