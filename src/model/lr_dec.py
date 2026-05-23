import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import trange
import scipy.sparse as sp


# -----------------------------
# Small utils
# -----------------------------
def set_seed(seed: int = 2025) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dense_from_layer_slice(adata, layer: str, gene_names: List[str]) -> np.ndarray:
    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers")
    if len(gene_names) == 0:
        return np.zeros((adata.n_obs, 0), dtype=np.float32)

    X = adata.layers[layer]
    gene_idx = adata.var_names.get_indexer(gene_names)
    if np.any(gene_idx < 0):
        missing = [g for g, i in zip(gene_names, gene_idx) if i < 0]
        raise KeyError(f"Genes not found in var_names: {missing[:10]} (showing up to 10)")

    if sp.issparse(X):
        Y = X[:, gene_idx].toarray()
    else:
        Y = np.asarray(X)[:, gene_idx]
    return Y.astype(np.float32, copy=False)


def _make_adjacent_pairs(sample_ids: Sequence[str]) -> List[Tuple[str, str]]:
    s = list(map(str, sample_ids))
    return [(s[i], s[i + 1]) for i in range(len(s) - 1)]


# -----------------------------
# Dataset
# -----------------------------
class ZCountDataset(Dataset):
    def __init__(self, Z: np.ndarray, Y: np.ndarray):
        assert Z.shape[0] == Y.shape[0]
        self.Z = torch.from_numpy(Z).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return self.Z[idx], self.Y[idx]


# -----------------------------
# Decoder: single head, regress raw counts
# -----------------------------
class NoisyCountDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h=(256, 256),
                 dropout=0.2, noise_in=0.0, noise_h=0.0):
        super().__init__()
        self.noise_in = float(noise_in)
        self.noise_h  = float(noise_h)
        self.p_drop   = float(dropout)

        layers = []
        dims = [in_dim] + list(h)
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
            ]
        self.backbone = nn.ModuleList(layers)
        self.head = nn.Linear(dims[-1], out_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _jitter(x, scale: float, enabled: bool):
        if scale <= 0 or (not enabled):
            return x
        s = x.detach().std(dim=0, keepdim=True).clamp_(min=1e-6)
        return x + torch.randn_like(x) * (scale * s)

    def forward(self, z, *, sample: bool = False):
        enabled = sample or self.training
        h = self._jitter(z, self.noise_in, enabled)

        it = iter(self.backbone)
        for lin, ln in zip(it, it):
            h = lin(h)
            h = ln(h)
            h = F.silu(h)
            h = self._jitter(h, self.noise_h, enabled)
            h = F.dropout(h, p=self.p_drop, training=enabled)

        return self.head(h)


@dataclass
class DecoderTrainResult:
    pair_id: str
    n_cells: int
    n_genes: int
    best_val: float
    decoder_ckpt_path: str
    decoder_stat_path: str


# -----------------------------
# ONLY: pairs training function
# -----------------------------
def train_count_decoder_pairs(
    *,
    adata, model, genes: Iterable[str], sample_ids: Sequence[str],                      # e.g. ["T167","T168","T169","T170","T171"]
    sample_key: str = "sample", layer_counts: str = "counts",
    out_dir: str = "/path/to/decoder_output",  sample_prefix: str = "decoder_counts_pair",
    skip_if_exists: bool = False,
    pair_ids: Optional[Sequence[Tuple[str, str]]] = None,  # optional: [("T167","T168"), ...]
    # train params
    test_size: float = 0.2, seed: int = 2025, batch_size: int = 2048,
    epochs: int = 100, patience: int = 10,
    lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.2, noise_in: float = 0.0,
    noise_h: float = 0.0, device: Optional[str] = None,
) -> List[DecoderTrainResult]:

    set_seed(seed)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # genes present
    genes = sorted(set(map(str, genes)))
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        raise ValueError("None of the given genes exist in adata.var_names")

    if pair_ids is None:
        pair_ids = _make_adjacent_pairs(sample_ids)

    svals = adata.obs[sample_key].astype(str).values
    results: List[DecoderTrainResult] = []

    for a, b in pair_ids:
        a = str(a); b = str(b)
        pair_name = f"{a}_{b}"

        ckpt_path = out_dir_p / f"{sample_prefix}_{pair_name}.pt"
        stat_path = out_dir_p / f"{sample_prefix}_{pair_name}.json"
        if skip_if_exists and ckpt_path.exists() and stat_path.exists():
            continue

        mask = (svals == a) | (svals == b)
        adata_pair = adata[mask].copy()
        if adata_pair.n_obs == 0:
            continue

        # X: latent Z
        Z = model.get_latent_representation(adata=adata_pair).astype(np.float32, copy=False)
        # Y: raw counts (selected genes)
        Y = _dense_from_layer_slice(adata_pair, layer_counts, genes)  # float32

        idx_all = np.arange(Z.shape[0], dtype=np.int64)
        idx_tr, idx_va = train_test_split(
            idx_all, test_size=float(test_size), random_state=int(seed), shuffle=True
        )

        Z_tr, Z_va = Z[idx_tr], Z[idx_va]
        Y_tr, Y_va = Y[idx_tr], Y[idx_va]

        dl_tr = DataLoader(ZCountDataset(Z_tr, Y_tr), batch_size=int(batch_size), shuffle=True)
        dl_va = DataLoader(ZCountDataset(Z_va, Y_va), batch_size=int(batch_size), shuffle=False)

        dec = NoisyCountDecoder(
            in_dim=Z.shape[1],
            out_dim=Y.shape[1],
            h=(256, 256),
            dropout=float(dropout),
            noise_in=float(noise_in),
            noise_h=float(noise_h),
        ).to(device)

        opt = torch.optim.AdamW(dec.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

        # MSE on raw counts; reduction="sum" then we normalize by batch*genes -> mean MSE per element
        mse = nn.MSELoss(reduction="sum")

        best_va = float("inf")
        no_imp = 0
        n_genes = int(Y.shape[1])

        pbar = trange(1, int(epochs) + 1, ncols=90, dynamic_ncols=False)
        for ep in pbar:
            dec.train()
            tr_sum, n_tr = 0.0, 0

            for zb, yb in dl_tr:
                zb = zb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                pred = dec(zb)
                loss_sum = mse(pred, yb)
                (loss_sum / (zb.size(0) * n_genes)).backward()
                nn.utils.clip_grad_norm_(dec.parameters(), 5.0)
                opt.step()

                tr_sum += float(loss_sum.item())
                n_tr += int(zb.size(0))

            tr_loss = tr_sum / max(n_tr * n_genes, 1)

            dec.eval()
            va_sum, n_va = 0.0, 0
            with torch.no_grad():
                for zb, yb in dl_va:
                    zb = zb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = dec(zb)
                    va_sum += float(mse(pred, yb).item())
                    n_va += int(zb.size(0))

            va_loss = va_sum / max(n_va * n_genes, 1)
            sched.step(va_loss)

            pbar.set_description(f"{pair_name} ep {ep:03d}/{int(epochs)}")
            pbar.set_postfix_str(f"tr={tr_loss:.4f} va={va_loss:.4f}")

            if np.isfinite(va_loss) and (va_loss + 1e-8 < best_va):
                best_va = float(va_loss)
                no_imp = 0

                torch.save(dec.state_dict(), str(ckpt_path))
                with open(stat_path, "w") as f:
                    json.dump(
                        {
                            "genes": genes,
                            "in_dim": int(Z.shape[1]),
                            "out_dim": int(Y.shape[1]),
                            "pair_id": pair_name,
                            "layer_counts": layer_counts,
                        },
                        f,
                    )
            else:
                no_imp += 1
                if no_imp >= int(patience):
                    break

        results.append(
            DecoderTrainResult(
                pair_id=pair_name,
                n_cells=int(adata_pair.n_obs),
                n_genes=int(n_genes),
                best_val=float(best_va),
                decoder_ckpt_path=str(ckpt_path),
                decoder_stat_path=str(stat_path),
            )
        )

    return results