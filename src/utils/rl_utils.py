import gc, glob, os, re, json, inspect
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from matplotlib.path import Path as MplPath
from tqdm import trange
from geomloss import SamplesLoss
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from model.lr_dec import NoisyCountDecoder
from utils.compute_lr import compute_lr_potential_gpu 

# -------------------- small utils --------------------
def _round_to_base(v: float, base: int) -> int: return int(np.ceil(v / base) * base)

def _to_dense_np(X: Any) -> np.ndarray:
    try:
        import scipy.sparse as sp
        if sp.issparse(X): return X.A
    except Exception:
        pass
    return np.asarray(X)

# -------------------- shells --------------------
def _read_shell_csv(path: str) -> np.ndarray:
    dfb = pd.read_csv(path)
    if "x_norm" in dfb.columns and "y_norm" in dfb.columns:
        x = dfb["x_norm"].to_numpy(np.float32); y = dfb["y_norm"].to_numpy(np.float32)
    elif "x" in dfb.columns and "y" in dfb.columns:
        x = dfb["x"].to_numpy(np.float32); y = dfb["y"].to_numpy(np.float32)
    else:
        raise ValueError(f"Unknown shell csv columns in {path}: {list(dfb.columns)}")
    return np.stack([x, y], axis=1)

def load_shells_from_dir(bound_dir: str, *, include_loops: bool = False):
    if include_loops:
        paths = glob.glob(os.path.join(bound_dir, "bound_z*_loop*.csv")) + glob.glob(os.path.join(bound_dir, "bound_z*.csv"))
    else:
        paths = glob.glob(os.path.join(bound_dir, "bound_z*.csv"))
        paths = [p for p in paths if "_loop" not in os.path.basename(p)]
    if len(paths) == 0: raise FileNotFoundError(f"No bound csv found in {bound_dir}")

    def parse_key(p):
        name = os.path.basename(p)
        m = re.search(r"bound_z(\d+)", name); z = int(m.group(1)) if m else 10**9
        m2 = re.search(r"_loop(\d+)", name); loop = int(m2.group(1)) if m2 else -1
        return (z, loop, name)

    paths.sort(key=parse_key)
    if not include_loops: return [_read_shell_csv(p) for p in paths]
    out = {}
    for p in paths:
        name = os.path.basename(p)
        m = re.search(r"bound_z(\d+)", name); z = int(m.group(1))
        out.setdefault(z, []).append(_read_shell_csv(p))
    return out

# -------------------- xy norm from ckpt --------------------
def get_xy_norm_from_ckpt(ckpt_obj: dict) -> Tuple[np.ndarray, np.ndarray]:
    norm = ckpt_obj.get("global_norm", None)
    if norm is None: raise KeyError("Checkpoint has no 'global_norm'")
    def _to_np(x: torch.Tensor) -> np.ndarray: return x.detach().cpu().numpy().astype(np.float32)
    xy_mu = _to_np(norm["xy_mu"]).reshape(1, -1)
    xy_s = _to_np(norm["xy_s"])
    if xy_s.ndim == 0: xy_s = np.array([[xy_s]], dtype=np.float32)
    elif xy_s.ndim == 1: xy_s = xy_s.reshape(1, -1)
    return xy_mu, xy_s

def normalize_xy_from_obs(ad: sc.AnnData, *, x_key: str, y_key: str, xy_mu: np.ndarray, xy_s: np.ndarray) -> np.ndarray:
    x = ad.obs[x_key].to_numpy(dtype=np.float32); y = ad.obs[y_key].to_numpy(dtype=np.float32)
    xy = np.stack([x, y], axis=1).astype(np.float32)
    return (xy - xy_mu) / (xy_s + 1e-12)

# -------------------- latent --------------------
def get_latent_tensor(
    adata_sub: sc.AnnData, *, device: torch.device,
    model_dir: Optional[str] = None, model_type: str = "scanvi",
    fallback_obsm: str = "X_scVI", fallback_layer: Optional[str] = None,
    fallback_X: bool = True, cache: Optional[dict] = None,
    load_ref_adata: Optional[sc.AnnData] = None,
) -> Tuple[torch.Tensor, dict]:
    cache = {} if cache is None else cache
    print("LOADING latent model from:", model_dir)

    if model_dir is not None and Path(model_dir).exists():
        key = (model_type.lower(), str(Path(model_dir).resolve()))
        if key not in cache:
            mt = model_type.lower()

            try:
                if mt == "scanvi":
                    m = scvi.model.SCANVI.load(model_dir)
                elif mt == "scvi":
                    m = scvi.model.SCVI.load(model_dir)
                else:
                    raise ValueError(f"Unknown model_type={model_type}")

                try:
                    print(f"[latent] loaded {mt}, summary_stats.n_labels={getattr(m, 'summary_stats', {}).get('n_labels', 'NA')}")
                    if mt == "scanvi" and hasattr(m, "module") and hasattr(m.module, "y_prior"):
                        print(f"[latent] scanvi head dim = {tuple(m.module.y_prior.shape)}")
                except Exception:
                    pass

            except Exception as e_no_adata:
                print(f"[warn] load(model_dir) failed without adata: {e_no_adata}")

                if load_ref_adata is None:
                    raise RuntimeError(
                        "Model load without adata failed. "
                        "Please either save model with save_anndata=True, "
                        "or pass load_ref_adata=training_schema_adata (full adata used in setup_anndata)."
                    ) from e_no_adata

                if mt == "scanvi":
                    m = scvi.model.SCANVI.load(model_dir, adata=load_ref_adata)
                elif mt == "scvi":
                    m = scvi.model.SCVI.load(model_dir, adata=load_ref_adata)
                else:
                    raise ValueError(f"Unknown model_type={model_type}")

                print(f"[latent] loaded {mt} with load_ref_adata (training schema)")

            cache[key] = m

        m = cache[key]

        Z_np = m.get_latent_representation(adata=adata_sub)
        return torch.tensor(Z_np, dtype=torch.float32, device=device), cache

    if fallback_obsm is not None and fallback_obsm in adata_sub.obsm:
        Z_np = np.asarray(adata_sub.obsm[fallback_obsm], dtype=np.float32)
        return torch.tensor(Z_np, dtype=torch.float32, device=device), cache

    if fallback_layer is not None and fallback_layer in adata_sub.layers:
        Z_np = _to_dense_np(adata_sub.layers[fallback_layer]).astype(np.float32)
        return torch.tensor(Z_np, dtype=torch.float32, device=device), cache

    if fallback_X:
        Z_np = _to_dense_np(adata_sub.X).astype(np.float32)
        return torch.tensor(Z_np, dtype=torch.float32, device=device), cache

    raise KeyError("Cannot get latent: model_dir missing and fallback_obsm/layer not found.")

def genes_union_from_lr_pairs(lr_pairs_df: pd.DataFrame, var_names: pd.Index) -> List[str]:
    lig = lr_pairs_df["ligand"].astype(str).tolist()
    rec = lr_pairs_df["receptor"].astype(str).tolist()
    union = sorted(set(lig).union(set(rec)))
    return [g for g in union if g in var_names]

def dense_from_layer_by_genes(adata, layer: str, genes: List[str]) -> np.ndarray:
    if len(genes) == 0: return np.zeros((adata.n_obs, 0), dtype=np.float32)
    if layer not in adata.layers: raise KeyError(f"Layer '{layer}' not in adata.layers")
    X = adata.layers[layer]
    idx = adata.var_names.get_indexer(genes)
    if np.any(idx < 0):
        missing = [g for g, i in zip(genes, idx) if i < 0]
        raise KeyError(f"Missing genes in var_names: {missing[:10]}")
    Y = _to_dense_np(X)[:, idx]
    return np.asarray(Y, dtype=np.float32)

# -------------------- grid cache --------------------
def auto_choose_grid_size(coords_norm: np.ndarray, shell_xy: np.ndarray, *, pts_per_cell: float = 1.0, base: int = 4, max_hw: int = 512, min_hw: int = 32, iters: int = 3) -> Tuple[int, int, dict]:
    poly = MplPath(shell_xy); in_shell = poly.contains_points(coords_norm); n_in = int(in_shell.sum())
    xmin, ymin = shell_xy.min(axis=0); xmax, ymax = shell_xy.max(axis=0)
    bw, bh = float(xmax - xmin), float(ymax - ymin); bw = max(bw, 1e-6); bh = max(bh, 1e-6)
    ar = bh / bw
    n_valid_desired = max(1, int(np.ceil(n_in / max(pts_per_cell, 1e-6))))
    W = np.sqrt(n_valid_desired / max(ar, 1e-6)); H = W * ar
    H = _round_to_base(H, base); W = _round_to_base(W, base)
    max_hw_aligned = (max_hw // base) * base; min_hw_aligned = max(base, (min_hw // base) * base)
    def clip_hw(h, w):
        h = int(np.clip(h, min_hw_aligned, max_hw_aligned)); w = int(np.clip(w, min_hw_aligned, max_hw_aligned))
        return h, w
    H, W = clip_hw(H, W)
    for _ in range(iters):
        xs = np.linspace(xmin, xmax, W, endpoint=False) + (bw / W) * 0.5
        ys = np.linspace(ymin, ymax, H, endpoint=False) + (bh / H) * 0.5
        XX, YY = np.meshgrid(xs, ys, indexing="xy"); centers = np.stack([XX.ravel(), YY.ravel()], axis=1)
        n_valid = int(poly.contains_points(centers).sum()); n_valid = max(n_valid, 1)
        scale = np.sqrt(n_valid_desired / n_valid)
        H2 = _round_to_base(H * scale, base); W2 = _round_to_base(W * scale, base)
        H2, W2 = clip_hw(H2, W2)
        if (H2 == H) and (W2 == W): break
        H, W = H2, W2
    xs = np.linspace(xmin, xmax, W, endpoint=False) + (bw / W) * 0.5
    ys = np.linspace(ymin, ymax, H, endpoint=False) + (bh / H) * 0.5
    XX, YY = np.meshgrid(xs, ys, indexing="xy"); centers = np.stack([XX.ravel(), YY.ravel()], axis=1)
    n_valid = int(poly.contains_points(centers).sum()); avg_pts = n_in / max(n_valid, 1)
    info = {"n_in": n_in, "n_valid_cells": n_valid, "avg_pts_per_valid_cell": float(avg_pts), "desired_pts_per_cell": float(pts_per_cell)}
    return int(H), int(W), info

@torch.no_grad()
def estimate_shell_need_cap_from_coords(
    coords: torch.Tensor, grid_item_dev, H: int, W: int,
    *, q: float = 0.90, min_cap: int = 1, max_cap: int = 8,
    ignore_zeros: bool = True
):
    x0, y0, dx, dy, in_shell = grid_item_dev
    dev, dtype = coords.device, coords.dtype

    gx = torch.floor((coords[:, 0] - x0) / dx).long()
    gy = torch.floor((coords[:, 1] - y0) / dy).long()
    inb = (gx >= 0) & (gx < int(W)) & (gy >= 0) & (gy < int(H))
    if not bool(inb.any()):
        return int(min_cap), {"reason": "no points in bbox"}

    gx = gx[inb]; gy = gy[inb]
    in_sh = in_shell[gy, gx].bool()
    if not bool(in_sh.any()):
        return int(min_cap), {"reason": "no points in shell"}

    gx = gx[in_sh]; gy = gy[in_sh]
    cell = (gy * int(W) + gx).long()
    counts = torch.bincount(cell, minlength=int(H) * int(W)).float()

    shell_mask = in_shell.reshape(-1).bool()
    counts_shell = counts[shell_mask]
    if ignore_zeros:
        counts_shell = counts_shell[counts_shell > 0]

    if counts_shell.numel() == 0:
        return int(min_cap), {"reason": "all-zero shell counts"}

    # quantile -> cap
    try:
        qv = torch.quantile(counts_shell, float(q)).item()
    except Exception:
        # fallback
        k = max(1, int(round(float(q) * (counts_shell.numel() - 1))) + 1)
        qv = counts_shell.kthvalue(k).values.item()

    cap = int(np.clip(int(np.ceil(qv)), int(min_cap), int(max_cap)))
    info = {
        "q": float(q),
        "quantile_value": float(qv),
        "mean": float(counts_shell.mean().item()),
        "max": float(counts_shell.max().item()),
        "n_cells_used": int(counts_shell.numel()),
    }
    return cap, info

def auto_shell_need_cap_for_stage(
    coords_src: torch.Tensor, coords_tgt: torch.Tensor,
    grid_cache_dev: list, H: int, W: int, T: int,
    *, q: float = 0.90, min_cap: int = 1, max_cap: int = 8
):
    cap0, info0 = estimate_shell_need_cap_from_coords(
        coords_src, grid_cache_dev[0], H, W, q=q, min_cap=min_cap, max_cap=max_cap
    )
    capT, infoT = estimate_shell_need_cap_from_coords(
        coords_tgt, grid_cache_dev[int(T)], H, W, q=q, min_cap=min_cap, max_cap=max_cap
    )
    cap = max(cap0, capT)
    return cap, {"src": info0, "tgt": infoT, "cap0": cap0, "capT": capT, "cap": cap}


def build_shell_grid_cache(shells_norm: List[np.ndarray], H: int, W: int, margin: float):
    cache = [None] * len(shells_norm)
    for k, shell_xy in enumerate(shells_norm):
        x_min, x_max = shell_xy[:, 0].min(), shell_xy[:, 0].max()
        y_min, y_max = shell_xy[:, 1].min(), shell_xy[:, 1].max()
        dx = (x_max - x_min) * (1.0 + 2 * margin) / W
        dy = (y_max - y_min) * (1.0 + 2 * margin) / H
        x0 = x_min - margin * (x_max - x_min)
        y0 = y_min - margin * (y_max - y_min)
        x_centers = x0 + (np.arange(W) + 0.5) * dx
        y_centers = y0 + (np.arange(H) + 0.5) * dy
        Xc, Yc = np.meshgrid(x_centers, y_centers)
        pts = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
        poly = MplPath(shell_xy)
        in_shell = poly.contains_points(pts).reshape(H, W)
        cache[k] = (float(x0), float(y0), float(dx), float(dy), in_shell.astype(bool))
    return cache

# -------------------- map sampling / gather --------------------
@torch.no_grad()
def _grid_index_of_coords(coords: torch.Tensor, grid_item_dev, H: int, W: int):
    x0, y0, dx, dy, in_shell = grid_item_dev
    device = coords.device; dtype = coords.dtype
    x0 = torch.as_tensor(x0, device=device, dtype=dtype)
    y0 = torch.as_tensor(y0, device=device, dtype=dtype)
    dx = torch.as_tensor(dx, device=device, dtype=dtype)
    dy = torch.as_tensor(dy, device=device, dtype=dtype)
    gx = torch.floor((coords[:, 0] - x0) / dx).long()
    gy = torch.floor((coords[:, 1] - y0) / dy).long()
    inb = (gx >= 0) & (gx < int(W)) & (gy >= 0) & (gy < int(H))
    if inb.any():
        ish = in_shell.bool().to(device=device)
        inb[inb.clone()] = ish[gy[inb], gx[inb]]
    return gx, gy, inb, x0, y0, dx, dy

@torch.no_grad()
def gather_map_at_coords_fast(map2d: torch.Tensor, coords: torch.Tensor, grid_item_dev, H: int, W: int, default: float = 0.0):
    gx, gy, ok, *_ = _grid_index_of_coords(coords, grid_item_dev, H, W)
    out = torch.full((coords.shape[0],), float(default), device=coords.device, dtype=map2d.dtype)
    if ok.any():
        idx = ok.nonzero(as_tuple=True)[0]
        out[idx] = map2d[gy[idx], gx[idx]]
    return out

@torch.no_grad()
def sample_centers_map(need_map_2d: torch.Tensor, grid_item_dev, H: int, W: int, k: int, *, gamma: float = 1.0, eps: float = 1e-10, seed: int = 2025, smooth_k: int = 1):
    if k <= 0: return None
    x0, y0, dx, dy, in_shell = grid_item_dev
    need = need_map_2d.clamp_min(0).float()
    if smooth_k and smooth_k > 1:
        need4 = need.view(1, 1, H, W)
        need = F.avg_pool2d(need4, kernel_size=smooth_k, stride=1, padding=smooth_k // 2).view(H, W)
    in_shell_f = in_shell.reshape(-1).float()
    need_flat = (need.reshape(-1) * in_shell_f).clamp_min(0)
    if float(need_flat.sum().detach().cpu().item()) <= eps: return None
    gen = torch.Generator(device=need.device); gen.manual_seed(int(seed))
    picks, need_work = [], need_flat.clone()
    for _ in range(int(k)):
        S = need_work.sum()
        if float(S.detach().cpu().item()) <= eps: break
        w = need_work.pow(float(gamma)) if gamma != 1.0 else need_work
        Sw = w.sum()
        if float(Sw.detach().cpu().item()) <= eps: break
        p = (w / Sw).detach()
        idx = torch.multinomial(p, num_samples=1, replacement=False, generator=gen).view(-1)
        picks.append(idx)
        need_work[idx] = (need_work[idx] - 1.0).clamp_min(0.0)
    if len(picks) == 0: return None
    idx = torch.cat(picks, dim=0)
    gy = (idx // W).long(); gx = (idx % W).long()
    cx = x0 + (gx.float() + 0.5) * dx
    cy = y0 + (gy.float() + 0.5) * dy
    return torch.stack([cx, cy], dim=1)

@torch.no_grad()
def grid_occupancy_by_layer(coords: torch.Tensor, layers: torch.Tensor, grid_item_dev, H: int, W: int, n_layers: int, normalize: bool = True):
    x0, y0, dx, dy, in_shell = grid_item_dev
    device = coords.device; dtype = coords.dtype
    in_shell = in_shell.bool()
    x0 = torch.as_tensor(x0, device=device, dtype=dtype)
    y0 = torch.as_tensor(y0, device=device, dtype=dtype)
    dx = torch.as_tensor(dx, device=device, dtype=dtype)
    dy = torch.as_tensor(dy, device=device, dtype=dtype)
    gx = torch.floor((coords[:, 0] - x0) / dx).long()
    gy = torch.floor((coords[:, 1] - y0) / dy).long()
    m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    keep = torch.empty((0,), device=device, dtype=torch.long)
    if m.any():
        gx_m = gx[m]; gy_m = gy[m]
        m2 = in_shell[gy_m, gx_m]
        keep = m.nonzero(as_tuple=True)[0][m2]
    HW = H * W
    if keep.numel() == 0: return torch.zeros((n_layers, H, W), device=device, dtype=torch.float32)
    gx_k = gx[keep]; gy_k = gy[keep]
    ly_k = layers[keep].long().clamp(0, n_layers - 1)
    cell = (gy_k * W + gx_k).long()
    flat = (ly_k * HW + cell).long()
    cnt = torch.bincount(flat, minlength=n_layers * HW).float().view(n_layers, H, W)
    if normalize:
        denom = cnt.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        cnt = cnt / denom
    return cnt

@torch.no_grad()
def get_occ_and_crowd(coords: torch.Tensor, layers: torch.Tensor, grid_item_dev, H: int, W: int, n_layers: int, cap: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = coords.device
    x0, y0, dx, dy, in_shell = grid_item_dev
    gx = torch.floor((coords[:, 0] - x0) / dx).long()
    gy = torch.floor((coords[:, 1] - y0) / dy).long()
    m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    valid_idx = m.nonzero(as_tuple=True)[0]
    if valid_idx.numel() > 0:
        is_in = in_shell[gy[valid_idx], gx[valid_idx]].bool()
        valid_idx = valid_idx[is_in]
    occ = torch.zeros((n_layers, H, W), device=device, dtype=torch.float32)
    crowd = torch.zeros(coords.shape[0], device=device, dtype=torch.float32)
    if valid_idx.numel() == 0: return occ, crowd
    gx_v, gy_v = gx[valid_idx], gy[valid_idx]
    ly_v = layers[valid_idx].long().clamp(0, n_layers - 1)
    flat_idx = ly_v * (H * W) + gy_v * W + gx_v
    counts = torch.bincount(flat_idx, minlength=n_layers * H * W).float()
    occ = counts.view(n_layers, H, W)
    total = occ.sum(dim=0)
    vals = total[gy_v, gx_v]
    crowd_val = (vals - float(cap)).clamp_min(0.0) / max(float(cap), 1e-6)
    crowd[valid_idx] = crowd_val
    return occ, crowd

def sample_no_replace(probs: torch.Tensor, k: int, eps: float = 1e-12):
    k = int(k); n = int(probs.numel()); device = probs.device
    if k <= 0 or n <= 0: return torch.empty(0, dtype=torch.long, device=device), torch.zeros(1, device=device)
    w = probs.clamp_min(eps); S = w.sum()
    if float(S.detach().cpu().item()) <= eps: return torch.empty(0, dtype=torch.long, device=device), torch.zeros(1, device=device)
    k = min(k, n)
    idx = torch.multinomial((w / S).detach(), num_samples=k, replacement=False)
    w_sel = w.gather(0, idx)
    prev_sum = torch.cumsum(w_sel, dim=0) - w_sel
    denom = (S - prev_sum).clamp_min(eps)
    p_cond = (w_sel / denom).clamp_min(eps)
    logp = torch.log(p_cond).sum().view(1)
    return idx, logp


@torch.no_grad()
def sample_birth_locations(centers: torch.Tensor, 
    mu_raw: torch.Tensor,        # (B,2) -> (mu_parallel, mu_perp)
    rho_raw: torch.Tensor,       # (B,2) -> (std_parallel, std_perp)
    dir_vec: torch.Tensor,       
    dx, dy, seed: int = 2025,min_std: float = 1e-3, eps: float = 1e-8,
):
    device = centers.device
    dtype = centers.dtype

    dx = torch.tensor(float(dx), device=device, dtype=dtype) if not torch.is_tensor(dx) else dx.to(device=device, dtype=dtype)
    dy = torch.tensor(float(dy), device=device, dtype=dtype) if not torch.is_tensor(dy) else dy.to(device=device, dtype=dtype)
    min_d = torch.minimum(dx, dy)

    # 1) normalize direction
    v = dir_vec.to(device=device, dtype=dtype)
    v = v / (v.norm(dim=1, keepdim=True) + eps)               # (B,2)
    n = torch.stack([-v[:, 1], v[:, 0]], dim=1)               # perpendicular

    # 2) bounded mean/std in (parallel, perp)
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))
    max_mu = 0.45 * min_d
    mu_par  = max_mu * torch.tanh(mu_raw[:, 0:1])
    mu_perp = max_mu * torch.tanh(mu_raw[:, 1:2])

    std_par  = (0.25 * min_d) * torch.sigmoid(rho_raw[:, 0:1])
    std_perp = (0.25 * min_d) * torch.sigmoid(rho_raw[:, 1:2])
    std_par  = std_par.clamp_min(float(min_std))
    std_perp = std_perp.clamp_min(float(min_std))

    # 3) sample in local frame then rotate back
    eps2 = torch.randn((centers.shape[0], 2), device=device, dtype=dtype, generator=gen)
    off_par  = mu_par  + std_par  * eps2[:, 0:1]
    off_perp = mu_perp + std_perp * eps2[:, 1:2]

    offset = off_par * v + off_perp * n                       # (B,2)
    new_coords = centers + offset
    return new_coords, torch.zeros(1, device=device, dtype=dtype)

@torch.no_grad()
def sample_local_centers(
    parent_coords: torch.Tensor, parent_layers: torch.Tensor,
    need_map_by_layer: torch.Tensor, grid_item_dev, H: int, W: int,
    *, radius: int = 6, gamma: float = 2.0, seed: int = 2025,
    fallback_to_parent: bool = True, kcand: int | None = None
):
    device, dtype = parent_coords.device, parent_coords.dtype
    gx, gy, ok, x0, y0, dx, dy = _grid_index_of_coords(parent_coords, grid_item_dev, H, W)
    gx = gx.long(); gy = gy.long(); ok = ok.bool()
    in_shell = grid_item_dev[4].to(device=device).bool()

    out = parent_coords.clone()
    B = int(parent_coords.shape[0])
    if B == 0: return out

    r = int(radius)
    off = torch.arange(-r, r + 1, device=device)
    offy, offx = torch.meshgrid(off, off, indexing="ij")
    offx = offx.reshape(1, -1); offy = offy.reshape(1, -1)
    K = int(offx.shape[1])

    gen = torch.Generator(device=device); gen.manual_seed(int(seed))
    if kcand is not None and int(kcand) < K:
        sel = torch.randperm(K, generator=gen, device=device)[:int(kcand)]
        offx = offx[:, sel]; offy = offy[:, sel]; K = int(kcand)

    gx0 = gx.view(B, 1); gy0 = gy.view(B, 1)
    xi = gx0 + offx; yi = gy0 + offy
    valid = (xi >= 0) & (xi < int(W)) & (yi >= 0) & (yi < int(H)) & ok.view(B, 1)

    xi = xi.clamp(0, int(W) - 1); yi = yi.clamp(0, int(H) - 1)
    li = parent_layers.long().view(B, 1).expand(B, K)

    w = need_map_by_layer[li, yi, xi].clamp_min(0.0)
    w = w * in_shell[yi, xi].float() * valid.float()
    if float(gamma) != 1.0: w = w.pow(float(gamma))

    eps = 1e-12
    row_sum = w.sum(dim=1)
    good = row_sum > eps
    if not bool(good.any()):
        return out if fallback_to_parent else out 

    wg = w[good]
    pg = wg / wg.sum(dim=1, keepdim=True).clamp_min(eps)  
    pick = torch.multinomial(pg, 1, replacement=False, generator=gen).squeeze(1)

    ar = torch.arange(int(wg.shape[0]), device=device)
    xi_g = xi[good]; yi_g = yi[good]
    xi_sel = xi_g[ar, pick]; yi_sel = yi_g[ar, pick]

    cx = x0 + (xi_sel.to(dtype) + 0.5) * dx
    cy = y0 + (yi_sel.to(dtype) + 0.5) * dy

    if fallback_to_parent:
        out_idx = good.nonzero(as_tuple=True)[0]
        out[out_idx, 0] = cx
        out[out_idx, 1] = cy
    else:
        out[:, 0] = out[:, 0] 
        out[:, 1] = out[:, 1]
    return out

@torch.no_grad()
def project_to_allowed_mask(
    coords: torch.Tensor, allow_mask_hw: torch.Tensor, grid_item_dev, H: int, W: int, *,
    center_offset: float = 0.5, chunk_q: int = 4096, max_ref: int | None = None, seed: int = 123
) -> torch.Tensor:
    if coords.numel() == 0:
        return coords
    dev, dtype = coords.device, coords.dtype
    allow = allow_mask_hw.to(device=dev, dtype=torch.bool)
    ys, xs = torch.nonzero(allow, as_tuple=True)
    if ys.numel() == 0:
        return coords

    x0, y0, dx, dy = grid_item_dev[0], grid_item_dev[1], grid_item_dev[2], grid_item_dev[3]
    ref = torch.stack([
        x0 + (xs.to(dtype) + float(center_offset)) * dx,
        y0 + (ys.to(dtype) + float(center_offset)) * dy
    ], dim=1)

    if (max_ref is not None) and (ref.shape[0] > int(max_ref)):
        gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
        perm = torch.randperm(ref.shape[0], device=dev, generator=gen)[:int(max_ref)]
        ref = ref[perm]

    j = torch.round((coords[:, 0] - x0) / dx).to(torch.long)
    i = torch.round((coords[:, 1] - y0) / dy).to(torch.long)
    inside = (i >= 0) & (i < int(H)) & (j >= 0) & (j < int(W))
    ii, jj = i.clamp(0, int(H) - 1), j.clamp(0, int(W) - 1)
    good = inside & allow[ii, jj]
    bad_idx = (~good).nonzero(as_tuple=True)[0]
    if bad_idx.numel() == 0:
        return coords

    out = coords.clone()
    for s in range(0, bad_idx.numel(), int(chunk_q)):
        idx = bad_idx[s:s + int(chunk_q)]
        D = torch.cdist(coords[idx], ref)
        out[idx] = ref[torch.argmin(D, dim=1)]
    return out

# -------------------- mass->quotas --------------------
def compute_N_tilde_from_mass(Ms: np.ndarray, N0_real: int, N_target_total: int) -> np.ndarray:
    M0, MT = float(Ms[0]), float(Ms[-1])
    deltaM = MT - M0
    deltaN = float(N_target_total - N0_real)
    N_tilde = np.empty_like(Ms, dtype=np.float32)
    if abs(deltaM) < 1e-6:
        T_local = len(Ms) - 1
        for k in range(len(Ms)): N_tilde[k] = N0_real + deltaN * (k / max(T_local, 1))
    else:
        scale = deltaN / deltaM
        for k, Mk in enumerate(Ms): N_tilde[k] = N0_real + (float(Mk) - M0) * scale
    return N_tilde

def _round_to_sum(x_float: np.ndarray, target_sum: int) -> np.ndarray:
    x_float = np.asarray(x_float, dtype=np.float64)
    target_sum = int(target_sum)
    x = np.floor(x_float).astype(np.int64)
    cur = int(x.sum())
    diff = target_sum - cur
    if diff == 0: return x
    if diff > 0:
        frac = x_float - np.floor(x_float)
        order = np.argsort(-frac)
        x[order[:diff]] += 1
        return x
    need_remove = -diff
    w = x.astype(np.float64) + 1e-12
    w /= w.sum()
    dec = np.floor(need_remove * w).astype(np.int64)
    dec = np.minimum(dec, x)
    x -= dec
    rem = need_remove - int(dec.sum())
    if rem > 0:
        resid = (need_remove * w) - dec
        order = np.argsort(-resid)
        for i in order:
            if rem == 0: break
            take = min(rem, int(x[i]))
            x[i] -= take
            rem -= take
    return x

def build_layer_plan_and_quotas(
    N_tilde: np.ndarray, counts0: np.ndarray, countsT: np.ndarray,
    death_frac: Optional[float] = 0.0, turnover_frac: Optional[float] = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.isinf(N_tilde).any() or np.isnan(N_tilde).any():
        print("[Warning] N_tilde contains INF/NAN! emergency truncation...")
        N_start = float(counts0.sum()); N_end = float(countsT.sum())
        total_steps = len(N_tilde) - 1
        for t in range(len(N_tilde)):
            if np.isinf(N_tilde[t]) or np.isnan(N_tilde[t]):
                progress = t / max(total_steps, 1)
                N_tilde[t] = N_start + progress * (N_end - N_start)

    T = len(N_tilde) - 1
    L = counts0.shape[0]
    N0 = int(counts0.sum()); NT = int(countsT.sum())
    df = 0.0 if death_frac is None else float(death_frac)
    tf = 0.0 if turnover_frac is None else float(turnover_frac)

    N_plan = np.zeros((T + 1, L), dtype=np.int64)
    for t in range(T + 1):
        den = float(NT - N0)
        alpha = (t / max(T, 1)) if abs(den) < 1e-6 else (float(N_tilde[t]) - float(N0)) / den
        alpha = float(np.clip(alpha, 0.0, 1.0))
        exp_layer = counts0 + alpha * (countsT - counts0)
        N_plan[t] = _round_to_sum(exp_layer, int(round(float(N_tilde[t]))))

    B_base = np.zeros((T, L), dtype=np.int64)
    D_base = np.zeros((T, L), dtype=np.int64)
    for t in range(T):
        delta = N_plan[t + 1] - N_plan[t]
        B_base[t] = np.maximum(delta, 0)
        D_base[t] = np.maximum(-delta, 0)

    B_step = B_base.copy(); D_step = D_base.copy()
    for t in range(T):
        step_delta = int(round(float(N_tilde[t + 1] - N_tilde[t])))
        turn_from_delta = int(round(df * abs(step_delta)))
        turn_from_level = int(round(tf * float(N_plan[t].sum())))
        D_extra_total = int(turn_from_delta + turn_from_level)
        if D_extra_total <= 0: continue

        wD = (N_plan[t].astype(np.float64) + 1e-3); wD = wD / wD.sum()
        D_extra = np.floor(D_extra_total * wD).astype(np.int64)
        rem = D_extra_total - int(D_extra.sum())
        if rem > 0:
            frac = D_extra_total * wD - D_extra
            for i in np.argsort(-frac)[:rem]: D_extra[i] += 1

        wB = (N_plan[t + 1].astype(np.float64) + 1e-3); wB = wB / wB.sum()
        B_extra = np.floor(D_extra_total * wB).astype(np.int64)
        rem = D_extra_total - int(B_extra.sum())
        if rem > 0:
            frac = D_extra_total * wB - B_extra
            for i in np.argsort(-frac)[:rem]: B_extra[i] += 1

        D_step[t] += D_extra
        B_step[t] += B_extra
    return N_plan, B_step, D_step

# -------------------- decoder + LR --------------------
@dataclass
class DecoderPack:
    genes_union: List[str]                 
    g2i: Dict[str, int]                    
    model_dec: nn.Module                   
    lr_cfg_torch: Dict[str, torch.Tensor]  
    pair_id: str = ""

def prepare_lr_tensors(lr_pairs_df: pd.DataFrame, g2i: Dict[str, int], device: torch.device) -> Dict[str, torch.Tensor]:
    df = lr_pairs_df.copy()
    if not {"ligand", "receptor"}.issubset(df.columns): raise ValueError("LR pairs must contain columns: ligand, receptor")
    if "sign" in df.columns: sign = df["sign"].map({"+": 1.0, "-": -1.0}).fillna(df["sign"]).astype(float).to_numpy(np.float32)
    else: sign = np.ones(len(df), dtype=np.float32)
    weight = df["weight"].astype(float).to_numpy(np.float32) if "weight" in df.columns else np.ones(len(df), dtype=np.float32)
    KL = df["KL"].astype(float).to_numpy(np.float32) if "KL" in df.columns else np.full(len(df), 5.0, np.float32)
    KR = df["KR"].astype(float).to_numpy(np.float32) if "KR" in df.columns else np.full(len(df), 5.0, np.float32)
    nH = df["hill_n"].astype(float).to_numpy(np.float32) if "hill_n" in df.columns else np.ones(len(df), np.float32)

    lig = df["ligand"].astype(str).to_numpy()
    rec = df["receptor"].astype(str).to_numpy()
    lig_idx = np.array([g2i.get(x, -1) for x in lig], dtype=np.int64)
    rec_idx = np.array([g2i.get(x, -1) for x in rec], dtype=np.int64)
    keep = (lig_idx >= 0) & (rec_idx >= 0)
    lig_idx, rec_idx = lig_idx[keep], rec_idx[keep]
    sign, weight, KL, KR, nH = sign[keep], weight[keep], KL[keep], KR[keep], nH[keep]

    w = weight * sign
    w_pos = np.clip(w, 0, None)
    w_neg = np.clip(-w, 0, None)
    return dict(
        lig_idx=torch.as_tensor(lig_idx, dtype=torch.long, device=device),
        rec_idx=torch.as_tensor(rec_idx, dtype=torch.long, device=device),
        w_pos=torch.as_tensor(w_pos, dtype=torch.float32, device=device),
        w_neg=torch.as_tensor(w_neg, dtype=torch.float32, device=device),
        KL=torch.as_tensor(KL, dtype=torch.float32, device=device),
        KR=torch.as_tensor(KR, dtype=torch.float32, device=device),
        nH=torch.as_tensor(nH, dtype=torch.float32, device=device),
    )

def build_decoder_pack(*, device: torch.device, stat_json: str, ckpt_path_dec: str,
                       lr_pairs_df: pd.DataFrame, latent_dim: int) -> DecoderPack:
    with open(stat_json, "r") as f:
        cfg = json.load(f)

    
    if "genes" not in cfg:
        raise KeyError(
            f"[build_decoder_pack] stat_json is missing the 'genes' field."
            f"This pipeline expects the pair-count decoder JSON with genes/in_dim/out_dim."
        )

    genes_union = [str(g) for g in cfg["genes"]]
    if len(genes_union) == 0:
        raise ValueError("[build_decoder_pack] cfg['genes'] is empty")

    in_dim = int(cfg.get("in_dim", latent_dim))
    if int(in_dim) != int(latent_dim):
        
        raise ValueError(f"[build_decoder_pack] latent_dim mismatch: json in_dim={in_dim} vs passed latent_dim={latent_dim}")

    out_dim_json = int(cfg.get("out_dim", len(genes_union)))
    if out_dim_json != len(genes_union):
        
        raise ValueError(f"[build_decoder_pack] out_dim mismatch: json out_dim={out_dim_json} vs len(genes)={len(genes_union)}")

    h = tuple(cfg.get("h", (256, 256)))
    dropout = float(cfg.get("dropout", 0.2))
    noise_in = float(cfg.get("noise_in", 0.0))
    noise_h  = float(cfg.get("noise_h", 0.0))

    model_dec = NoisyCountDecoder(
        in_dim=in_dim,
        out_dim=len(genes_union),
        h=h,
        dropout=dropout,
        noise_in=noise_in,
        noise_h=noise_h,
    ).to(device)

    state_dec = torch.load(ckpt_path_dec, map_location=device)
    model_dec.load_state_dict(state_dec, strict=True)
    model_dec.eval()

    g2i = {g: i for i, g in enumerate(genes_union)}
    lr_cfg_torch = prepare_lr_tensors(lr_pairs_df, g2i, device)

    pair_id = str(cfg.get("pair_id", ""))
    return DecoderPack(
        genes_union=genes_union,
        g2i=g2i,
        model_dec=model_dec,
        lr_cfg_torch=lr_cfg_torch,
        pair_id=pair_id,
    )

@torch.no_grad()
def decode_expr_union_torch(
    dp: DecoderPack,
    latent: torch.Tensor,
    *,
    clamp_min: float = 0.0,
    round_int: bool = False,
) -> torch.Tensor:

    expr = dp.model_dec(latent)  # (N, G)
    expr = expr.to(dtype=torch.float32)

    if clamp_min is not None:
        expr = torch.clamp(expr, min=float(clamp_min))

    if round_int:
        expr = torch.round(expr)

    return expr

@torch.no_grad()
def recompute_lr_factory(dp: DecoderPack):
    def _fn(latent: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        expr_union_t = decode_expr_union_torch(dp, latent, clamp_min=0.0, round_int=False)
        return compute_lr_potential_gpu(
            expr_union_t, coords,lr_cfg=dp.lr_cfg_torch,
            k=16,use_cpm=True,lib_power=0.5,lib_clip_q=(5, 95),
            hill=True,combine_mode="zdiff",pair_chunk=64,knn_chunk_size=2048
        )
    return _fn

def recompute_lr_factory_none(device: torch.device):
    def _fn(_: Optional[torch.Tensor], coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros((coords.shape[0],), device=device, dtype=torch.float32)
    return _fn

def recompute_lr_factory_counts(*, device: torch.device, lr_pairs_df: pd.DataFrame, genes_union: List[str], compute_lr_potential_gpu):
    g2i = {g: i for i, g in enumerate(genes_union)}
    lr_cfg_torch = prepare_lr_tensors(lr_pairs_df, g2i, device)
    has_pairs = int(lr_cfg_torch["lig_idx"].numel()) > 0
    def _fn(expr_union: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if (expr_union is None) or (not has_pairs):
            return torch.zeros((coords.shape[0],), device=device, dtype=torch.float32)
        return compute_lr_potential_gpu(expr_union, coords, lr_cfg=lr_cfg_torch, k=16, use_cpm=True, lib_power=0.5, lib_clip_q=(5, 95),
                                        hill=True, combine_mode="zdiff", pair_chunk=64, knn_chunk_size=2048)
    return _fn

# -------------------- advection + reanchor --------------------
@torch.no_grad()
def advect_by_knn_deterministic(q: torch.Tensor, base_t: torch.Tensor, disp: torch.Tensor, *, k: int = 8, max_ref: int = 20000,
                               seed: int = 2025, eps: float = 1e-6, chunk_q: int = 4096) -> torch.Tensor:
    if q.numel() == 0: return q
    Nbase = base_t.shape[0]
    if (max_ref is not None) and (Nbase > max_ref):
        gen = torch.Generator(device=base_t.device); gen.manual_seed(int(seed))
        idx_ref = torch.randperm(Nbase, device=base_t.device, generator=gen)[:max_ref]
        bt = base_t[idx_ref]; dp = disp[idx_ref]
    else:
        bt = base_t; dp = disp
    Nref = bt.shape[0]; k = min(int(k), int(Nref))
    out = q.clone()
    for s in range(0, q.shape[0], int(chunk_q)):
        qs = q[s:s + chunk_q]
        D = torch.cdist(qs, bt)
        vals, idx = torch.topk(D, k, largest=False)
        w = 1.0 / (vals + eps); w = w / w.sum(dim=1, keepdim=True)
        delta = (w.unsqueeze(-1) * dp[idx]).sum(dim=1)
        out[s:s + chunk_q] = qs + delta
    return out

@torch.no_grad()
def _quantile_fast(x: torch.Tensor, q: float):
    q = float(q)
    x = x.float()
    try:
        return torch.quantile(x, q)
    except Exception:
        k = max(1, int(round(q * (x.numel() - 1))) + 1)
        return x.kthvalue(k).values

@torch.no_grad()
def advect_coords_hybrid_soft(coords: torch.Tensor, anchor: torch.Tensor, base_t: torch.Tensor, base_tp1: torch.Tensor, *,
                             dist_q: float = 0.90, dist_min: float = 0.02, dist_max: float = 0.35,
                             blend_temp: float = 0.02, knn_k: int = 8, knn_max_ref: int = 20000, knn_seed: int = 2025,
                             knn_band: float = 3.0, return_alpha: bool = False):
    dev=coords.device
    if coords.numel()==0:
        if return_alpha: return coords, torch.zeros((0,), device=dev, dtype=torch.float32)
        return coords

    disp=base_tp1-base_t
    out_anchor=coords+disp[anchor]
    drift=(coords-base_t[anchor]).norm(dim=1)  # drift BEFORE move

    thr=_quantile_fast(drift, dist_q).clamp(min=float(dist_min), max=float(dist_max))
    temp=max(float(blend_temp), 1e-6)

    alpha=torch.sigmoid((drift-thr)/temp)  # (N,)
    band=float(knn_band)
    alpha_min=float(1.0/(1.0+np.exp(band)))  # sigmoid(-band)
    need_knn = alpha > alpha_min
    alpha = torch.where(need_knn, alpha, torch.zeros_like(alpha))

    if not bool(need_knn.any()):
        if return_alpha: return out_anchor, alpha
        return out_anchor

    out=out_anchor.clone()
    out_knn=advect_by_knn_deterministic(coords[need_knn], base_t, disp, k=int(knn_k), max_ref=int(knn_max_ref), seed=int(knn_seed))
    a=alpha[need_knn].view(-1,1)
    out[need_knn]=(1.0-a)*out_anchor[need_knn] + a*out_knn

    if return_alpha: return out, alpha
    return out

@torch.no_grad()
def quota_from_weights(total: int, w: torch.Tensor, *, max_cap: Optional[torch.Tensor] = None, eps: float = 1e-12):
    total = int(max(total, 0))
    L = int(w.numel())
    if total == 0 or L == 0: 
        q = torch.zeros((L,), device=w.device, dtype=torch.long)
        return q
    w = w.float().clamp_min(0.0)
    if max_cap is not None: max_cap = max_cap.long().clamp_min(0)
    if float(w.sum().item()) <= eps:
        # fallback: uniform over feasible layers
        if max_cap is None:
            w = torch.ones_like(w)
        else:
            w = (max_cap > 0).float()
            if float(w.sum().item()) <= eps: return torch.zeros((L,), device=w.device, dtype=torch.long)
    qf = total * (w / (w.sum().clamp_min(eps)))
    q = torch.floor(qf).long()
    if max_cap is not None: q = torch.minimum(q, max_cap)
    diff = total - int(q.sum().item())
    if diff == 0: return q
    frac = (qf - torch.floor(qf)).float()
    # +diff: add to largest frac among not-at-cap
    if diff > 0:
        if max_cap is None: can_add = torch.ones_like(q, dtype=torch.bool)
        else: can_add = q < max_cap
        if not bool(can_add.any()):
            return q  # already saturated
        frac2 = frac.clone()
        frac2[~can_add] = -1e9
        order = torch.argsort(frac2, descending=True)
        for i in order.tolist():
            if diff <= 0: break
            if not bool(can_add[i]): continue
            q[i] += 1; diff -= 1
            if max_cap is not None and q[i] >= max_cap[i]: can_add[i] = False
        return q
    # -diff: remove from smallest frac but q>0
    need = -diff
    can_sub = q > 0
    if not bool(can_sub.any()): return q
    frac2 = frac.clone()
    frac2[~can_sub] = 1e9
    order = torch.argsort(frac2, descending=False)
    for i in order.tolist():
        if need <= 0: break
        if q[i] <= 0: continue
        take = min(int(q[i].item()), need)
        q[i] -= take; need -= take
    return q