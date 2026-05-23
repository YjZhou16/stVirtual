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
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from model.lr_dec import NoisyCountDecoder
from utils.compute_lr import compute_lr_potential_gpu

# -------------------- small utils --------------------
def _round_to_base(v: float, base: int) -> int: return int(np.ceil(v / base) * base)
def _filter_kwargs(cls, kw):
    sig = inspect.signature(cls.__init__); ok = set(sig.parameters.keys()); ok.discard("self")
    return {k: v for k, v in kw.items() if k in ok}
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
        paths = glob.glob(os.path.join(bound_dir, "bound_z*.csv")); paths = [p for p in paths if "_loop" not in os.path.basename(p)]
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
        name = os.path.basename(p); m = re.search(r"bound_z(\d+)", name); z = int(m.group(1))
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
                if mt == "scanvi": m = scvi.model.SCANVI.load(model_dir)
                elif mt == "scvi": m = scvi.model.SCVI.load(model_dir)
                else: raise ValueError(f"Unknown model_type={model_type}")
            except Exception as e_no_adata:
                print(f"[warn] load(model_dir) failed without adata: {e_no_adata}")
                if load_ref_adata is None:
                    raise RuntimeError("Model load without adata failed. Save with save_anndata=True or pass load_ref_adata=training_schema_adata.") from e_no_adata
                if mt == "scanvi": m = scvi.model.SCANVI.load(model_dir, adata=load_ref_adata)
                elif mt == "scvi": m = scvi.model.SCVI.load(model_dir, adata=load_ref_adata)
                else: raise ValueError(f"Unknown model_type={model_type}")
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
    lig = lr_pairs_df["ligand"].astype(str).tolist(); rec = lr_pairs_df["receptor"].astype(str).tolist()
    union = sorted(set(lig).union(set(rec)))
    return [g for g in union if g in var_names]

def dense_from_layer_by_genes(adata, layer: str, genes: List[str]) -> np.ndarray:
    if len(genes) == 0: return np.zeros((adata.n_obs, 0), dtype=np.float32)
    if layer not in adata.layers: raise KeyError(f"Layer '{layer}' not in adata.layers")
    X = adata.layers[layer]; idx = adata.var_names.get_indexer(genes)
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

def build_shell_grid_cache(shells_norm: List[np.ndarray], H: int, W: int, margin: float):
    cache = [None] * len(shells_norm)
    for k, shell_xy in enumerate(shells_norm):
        x_min, x_max = shell_xy[:, 0].min(), shell_xy[:, 0].max()
        y_min, y_max = shell_xy[:, 1].min(), shell_xy[:, 1].max()
        dx = (x_max - x_min) * (1.0 + 2 * margin) / W; dy = (y_max - y_min) * (1.0 + 2 * margin) / H
        x0 = x_min - margin * (x_max - x_min); y0 = y_min - margin * (y_max - y_min)
        x_centers = x0 + (np.arange(W) + 0.5) * dx; y_centers = y0 + (np.arange(H) + 0.5) * dy
        Xc, Yc = np.meshgrid(x_centers, y_centers); pts = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
        poly = MplPath(shell_xy); in_shell = poly.contains_points(pts).reshape(H, W)
        cache[k] = (float(x0), float(y0), float(dx), float(dy), in_shell.astype(bool))
    return cache

# -------------------- map sampling / gather --------------------
@torch.no_grad()
def _grid_index_of_coords(coords: torch.Tensor, grid_item_dev, H: int, W: int):
    x0, y0, dx, dy, in_shell = grid_item_dev
    device = coords.device; dtype = coords.dtype
    x0 = torch.as_tensor(x0, device=device, dtype=dtype); y0 = torch.as_tensor(y0, device=device, dtype=dtype)
    dx = torch.as_tensor(dx, device=device, dtype=dtype); dy = torch.as_tensor(dy, device=device, dtype=dtype)
    gx = torch.floor((coords[:, 0] - x0) / dx).long(); gy = torch.floor((coords[:, 1] - y0) / dy).long()
    inb = (gx >= 0) & (gx < int(W)) & (gy >= 0) & (gy < int(H))
    if inb.any():
        ish = in_shell.bool().to(device=device); inb[inb.clone()] = ish[gy[inb], gx[inb]]
    return gx, gy, inb, x0, y0, dx, dy

@torch.no_grad()
def gather_map_at_coords_fast(map2d: torch.Tensor, coords: torch.Tensor, grid_item_dev, H: int, W: int, default: float = 0.0):
    gx, gy, ok, *_ = _grid_index_of_coords(coords, grid_item_dev, H, W)
    out = torch.full((coords.shape[0],), float(default), device=coords.device, dtype=map2d.dtype)
    if ok.any():
        idx = ok.nonzero(as_tuple=True)[0]; out[idx] = map2d[gy[idx], gx[idx]]
    return out

@torch.no_grad()
def sample_centers_map(need_map_2d: torch.Tensor, grid_item_dev, H: int, W: int, k: int, *, gamma: float = 1.0, eps: float = 1e-10, seed: int = 2025, smooth_k: int = 1):
    if k <= 0: return None
    x0, y0, dx, dy, in_shell = grid_item_dev
    need = need_map_2d.clamp_min(0).float()
    if smooth_k and smooth_k > 1:
        need4 = need.view(1, 1, H, W); need = F.avg_pool2d(need4, kernel_size=smooth_k, stride=1, padding=smooth_k // 2).view(H, W)
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
        picks.append(idx); need_work[idx] = (need_work[idx] - 1.0).clamp_min(0.0)
    if len(picks) == 0: return None
    idx = torch.cat(picks, dim=0)
    gy = (idx // W).long(); gx = (idx % W).long()
    cx = x0 + (gx.float() + 0.5) * dx; cy = y0 + (gy.float() + 0.5) * dy
    return torch.stack([cx, cy], dim=1)

@torch.no_grad()
def grid_occupancy_by_layer(coords: torch.Tensor, layers: torch.Tensor, grid_item_dev, H: int, W: int, n_layers: int, normalize: bool = True):
    x0, y0, dx, dy, in_shell = grid_item_dev
    device = coords.device; dtype = coords.dtype; in_shell = in_shell.bool()
    x0 = torch.as_tensor(x0, device=device, dtype=dtype); y0 = torch.as_tensor(y0, device=device, dtype=dtype)
    dx = torch.as_tensor(dx, device=device, dtype=dtype); dy = torch.as_tensor(dy, device=device, dtype=dtype)
    gx = torch.floor((coords[:, 0] - x0) / dx).long(); gy = torch.floor((coords[:, 1] - y0) / dy).long()
    m = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    keep = torch.empty((0,), device=device, dtype=torch.long)
    if m.any():
        gx_m = gx[m]; gy_m = gy[m]; m2 = in_shell[gy_m, gx_m]
        keep = m.nonzero(as_tuple=True)[0][m2]
    HW = H * W
    if keep.numel() == 0: return torch.zeros((n_layers, H, W), device=device, dtype=torch.float32)
    gx_k = gx[keep]; gy_k = gy[keep]
    ly_k = layers[keep].long().clamp(0, n_layers - 1)
    cell = (gy_k * W + gx_k).long()
    flat = (ly_k * HW + cell).long()
    cnt = torch.bincount(flat, minlength=n_layers * HW).float().view(n_layers, H, W)
    if normalize:
        denom = cnt.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6); cnt = cnt / denom
    return cnt

@torch.no_grad()
def get_occ_and_crowd(coords: torch.Tensor, layers: torch.Tensor, grid_item_dev, H: int, W: int, n_layers: int, cap: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = coords.device
    x0, y0, dx, dy, in_shell = grid_item_dev
    gx = torch.floor((coords[:, 0] - x0) / dx).long(); gy = torch.floor((coords[:, 1] - y0) / dy).long()
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
def sample_birth_locations(centers: torch.Tensor, mu_raw: torch.Tensor, rho_raw: torch.Tensor, dir_vec: torch.Tensor, dx, dy, seed: int = 2025, min_std: float = 1e-3, eps: float = 1e-8):
    device = centers.device; dtype = centers.dtype
    dx = torch.tensor(float(dx), device=device, dtype=dtype) if not torch.is_tensor(dx) else dx.to(device=device, dtype=dtype)
    dy = torch.tensor(float(dy), device=device, dtype=dtype) if not torch.is_tensor(dy) else dy.to(device=device, dtype=dtype)
    min_d = torch.minimum(dx, dy)
    v = dir_vec.to(device=device, dtype=dtype); v = v / (v.norm(dim=1, keepdim=True) + eps)
    n = torch.stack([-v[:, 1], v[:, 0]], dim=1)
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))
    max_mu = 0.45 * min_d
    mu_par = max_mu * torch.tanh(mu_raw[:, 0:1]); mu_perp = max_mu * torch.tanh(mu_raw[:, 1:2])
    std_par = (0.25 * min_d) * torch.sigmoid(rho_raw[:, 0:1]); std_perp = (0.25 * min_d) * torch.sigmoid(rho_raw[:, 1:2])
    std_par = std_par.clamp_min(float(min_std)); std_perp = std_perp.clamp_min(float(min_std))
    eps2 = torch.randn((centers.shape[0], 2), device=device, dtype=dtype, generator=gen)
    off_par = mu_par + std_par * eps2[:, 0:1]; off_perp = mu_perp + std_perp * eps2[:, 1:2]
    offset = off_par * v + off_perp * n
    return centers + offset, torch.zeros(1, device=device, dtype=dtype)

@torch.no_grad()
def sample_local_centers(parent_coords: torch.Tensor, parent_layers: torch.Tensor, need_map_by_layer: torch.Tensor, grid_item_dev, H: int, W: int, *, radius: int = 6, gamma: float = 2.0, seed: int = 2025, fallback_to_parent: bool = True, kcand: int | None = None):
    device, dtype = parent_coords.device, parent_coords.dtype
    gx, gy, ok, x0, y0, dx, dy = _grid_index_of_coords(parent_coords, grid_item_dev, H, W)
    gx = gx.long(); gy = gy.long(); ok = ok.bool()
    in_shell = grid_item_dev[4].to(device=device).bool()
    out = parent_coords.clone(); B = int(parent_coords.shape[0])
    if B == 0: return out
    r = int(radius)
    off = torch.arange(-r, r + 1, device=device)
    offy, offx = torch.meshgrid(off, off, indexing="ij")
    offx = offx.reshape(1, -1); offy = offy.reshape(1, -1); K = int(offx.shape[1])
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
    row_sum = w.sum(dim=1); good = row_sum > eps
    if not bool(good.any()): return out if fallback_to_parent else out
    wg = w[good]; pg = wg / wg.sum(dim=1, keepdim=True).clamp_min(eps)
    pick = torch.multinomial(pg, 1, replacement=False, generator=gen).squeeze(1)
    ar = torch.arange(int(wg.shape[0]), device=device)
    xi_g = xi[good]; yi_g = yi[good]
    xi_sel = xi_g[ar, pick]; yi_sel = yi_g[ar, pick]
    cx = x0 + (xi_sel.to(dtype) + 0.5) * dx; cy = y0 + (yi_sel.to(dtype) + 0.5) * dy
    if fallback_to_parent:
        out_idx = good.nonzero(as_tuple=True)[0]
        out[out_idx, 0] = cx; out[out_idx, 1] = cy
    return out

@torch.no_grad()
def project_to_allowed_mask(coords: torch.Tensor, allow_mask_hw: torch.Tensor, grid_item_dev, H: int, W: int, *, center_offset: float = 0.5, chunk_q: int = 4096, max_ref: int | None = None, seed: int = 123) -> torch.Tensor:
    if coords.numel() == 0: return coords
    dev, dtype = coords.device, coords.dtype
    allow = allow_mask_hw.to(device=dev, dtype=torch.bool)
    ys, xs = torch.nonzero(allow, as_tuple=True)
    if ys.numel() == 0: return coords
    x0, y0, dx, dy = grid_item_dev[0], grid_item_dev[1], grid_item_dev[2], grid_item_dev[3]
    ref = torch.stack([x0 + (xs.to(dtype) + float(center_offset)) * dx, y0 + (ys.to(dtype) + float(center_offset)) * dy], dim=1)
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
    if bad_idx.numel() == 0: return coords
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
    x_float = np.asarray(x_float, dtype=np.float64); target_sum = int(target_sum)
    x = np.floor(x_float).astype(np.int64); cur = int(x.sum()); diff = target_sum - cur
    if diff == 0: return x
    if diff > 0:
        frac = x_float - np.floor(x_float); order = np.argsort(-frac); x[order[:diff]] += 1; return x
    need_remove = -diff
    w = x.astype(np.float64) + 1e-12; w /= w.sum()
    dec = np.floor(need_remove * w).astype(np.int64); dec = np.minimum(dec, x); x -= dec
    rem = need_remove - int(dec.sum())
    if rem > 0:
        resid = (need_remove * w) - dec; order = np.argsort(-resid)
        for i in order:
            if rem == 0: break
            take = min(rem, int(x[i])); x[i] -= take; rem -= take
    return x

def build_layer_plan_and_quotas(N_tilde: np.ndarray, counts0: np.ndarray, countsT: np.ndarray, death_frac: Optional[float] = 0.2, turnover_frac: Optional[float] = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.isinf(N_tilde).any() or np.isnan(N_tilde).any():
        print("[Warning] N_tilde contains INF/NAN! emergency truncation...")
        N_start = float(counts0.sum()); N_end = float(countsT.sum()); total_steps = len(N_tilde) - 1
        for t in range(len(N_tilde)):
            if np.isinf(N_tilde[t]) or np.isnan(N_tilde[t]):
                progress = t / max(total_steps, 1); N_tilde[t] = N_start + progress * (N_end - N_start)

    T = len(N_tilde) - 1; L = counts0.shape[0]
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

    B_base = np.zeros((T, L), dtype=np.int64); D_base = np.zeros((T, L), dtype=np.int64)
    for t in range(T):
        delta = N_plan[t + 1] - N_plan[t]
        B_base[t] = np.maximum(delta, 0); D_base[t] = np.maximum(-delta, 0)

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

        D_step[t] += D_extra; B_step[t] += B_extra
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

    lig = df["ligand"].astype(str).to_numpy(); rec = df["receptor"].astype(str).to_numpy()
    lig_idx = np.array([g2i.get(x, -1) for x in lig], dtype=np.int64)
    rec_idx = np.array([g2i.get(x, -1) for x in rec], dtype=np.int64)
    keep = (lig_idx >= 0) & (rec_idx >= 0)
    lig_idx, rec_idx = lig_idx[keep], rec_idx[keep]
    sign, weight, KL, KR, nH = sign[keep], weight[keep], KL[keep], KR[keep], nH[keep]

    w = weight * sign; w_pos = np.clip(w, 0, None); w_neg = np.clip(-w, 0, None)
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

    # ---- important:new json must genes(LR) ----
    if "genes" not in cfg:
        raise KeyError(
            f"[build_decoder_pack] stat_json missing 'genes' ."
            f" pipeline expects pair count decoder  json( genes/in_dim/out_dim)."
        )

    genes_union = [str(g) for g in cfg["genes"]]
    if len(genes_union) == 0:
        raise ValueError("[build_decoder_pack] cfg['genes'] is empty")

    in_dim = int(cfg.get("in_dim", latent_dim))
    if int(in_dim) != int(latent_dim):
        # here, latent 
        raise ValueError(f"[build_decoder_pack] latent_dim mismatch: json in_dim={in_dim} vs passed latent_dim={latent_dim}")

    out_dim_json = int(cfg.get("out_dim", len(genes_union)))
    if out_dim_json != len(genes_union):
        # , json/training
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
        if (expr_union is None) or (not has_pairs): return torch.zeros((coords.shape[0],), device=device, dtype=torch.float32)
        return compute_lr_potential_gpu(expr_union, coords, lr_cfg=lr_cfg_torch, k=16, use_cpm=True, lib_power=0.5, lib_clip_q=(5, 95), hill=True, combine_mode="zdiff", pair_chunk=64, knn_chunk_size=2048)
    return _fn

# -------------------- advection --------------------
@torch.no_grad()
def advect_by_knn_deterministic(q: torch.Tensor, base_t: torch.Tensor, disp: torch.Tensor, *, k: int = 8, max_ref: int = 20000, seed: int = 2025, eps: float = 1e-6, chunk_q: int = 4096) -> torch.Tensor:
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
    q = float(q); x = x.float()
    try: return torch.quantile(x, q)
    except Exception:
        k = max(1, int(round(q * (x.numel() - 1))) + 1)
        return x.kthvalue(k).values

@torch.no_grad()
def advect_coords_hybrid_soft(coords: torch.Tensor, anchor: torch.Tensor, base_t: torch.Tensor, base_tp1: torch.Tensor, *, dist_q: float = 0.90, dist_min: float = 0.02, dist_max: float = 0.35, blend_temp: float = 0.02, knn_k: int = 8, knn_max_ref: int = 20000, knn_seed: int = 2025, knn_band: float = 3.0, return_alpha: bool = False):
    dev = coords.device
    if coords.numel() == 0:
        if return_alpha: return coords, torch.zeros((0,), device=dev, dtype=torch.float32)
        return coords
    disp = base_tp1 - base_t
    out_anchor = coords + disp[anchor]
    drift = (coords - base_t[anchor]).norm(dim=1)
    thr = _quantile_fast(drift, dist_q).clamp(min=float(dist_min), max=float(dist_max))
    temp = max(float(blend_temp), 1e-6)
    alpha = torch.sigmoid((drift - thr) / temp)
    band = float(knn_band); alpha_min = float(1.0 / (1.0 + np.exp(band)))
    need_knn = alpha > alpha_min
    alpha = torch.where(need_knn, alpha, torch.zeros_like(alpha))
    if not bool(need_knn.any()):
        if return_alpha: return out_anchor, alpha
        return out_anchor
    out = out_anchor.clone()
    out_knn = advect_by_knn_deterministic(coords[need_knn], base_t, disp, k=int(knn_k), max_ref=int(knn_max_ref), seed=int(knn_seed))
    a = alpha[need_knn].view(-1, 1)
    out[need_knn] = (1.0 - a) * out_anchor[need_knn] + a * out_knn
    if return_alpha: return out, alpha
    return out

# -------------------- quota allocator --------------------
@torch.no_grad()
def quota_from_weights(total: int, w: torch.Tensor, *, max_cap: Optional[torch.Tensor] = None, eps: float = 1e-12):
    total = int(max(total, 0)); L = int(w.numel())
    if total == 0 or L == 0: return torch.zeros((L,), device=w.device, dtype=torch.long)
    w = w.float().clamp_min(0.0)
    if max_cap is not None: max_cap = max_cap.long().clamp_min(0)
    if float(w.sum().item()) <= eps:
        if max_cap is None: w = torch.ones_like(w)
        else:
            w = (max_cap > 0).float()
            if float(w.sum().item()) <= eps: return torch.zeros((L,), device=w.device, dtype=torch.long)
    qf = total * (w / (w.sum().clamp_min(eps)))
    q = torch.floor(qf).long()
    if max_cap is not None: q = torch.minimum(q, max_cap)
    diff = total - int(q.sum().item())
    if diff == 0: return q
    frac = (qf - torch.floor(qf)).float()
    if diff > 0:
        can_add = torch.ones_like(q, dtype=torch.bool) if max_cap is None else (q < max_cap)
        if not bool(can_add.any()): return q
        frac2 = frac.clone(); frac2[~can_add] = -1e9
        order = torch.argsort(frac2, descending=True)
        for i in order.tolist():
            if diff <= 0: break
            if not bool(can_add[i]): continue
            q[i] += 1; diff -= 1
            if max_cap is not None and q[i] >= max_cap[i]: can_add[i] = False
        return q
    need = -diff
    can_sub = q > 0
    if not bool(can_sub.any()): return q
    frac2 = frac.clone(); frac2[~can_sub] = 1e9
    order = torch.argsort(frac2, descending=False)
    for i in order.tolist():
        if need <= 0: break
        if q[i] <= 0: continue
        take = min(int(q[i].item()), need)
        q[i] -= take; need -= take
    return q

@torch.no_grad()
def estimate_shell_need_cap_from_coords(coords: torch.Tensor, grid_item_dev, H: int, W: int, *, q: float = 0.90, min_cap: int = 1, max_cap: int = 8, ignore_zeros: bool = True):
    x0, y0, dx, dy, in_shell = grid_item_dev
    gx = torch.floor((coords[:, 0] - x0) / dx).long(); gy = torch.floor((coords[:, 1] - y0) / dy).long()
    inb = (gx >= 0) & (gx < int(W)) & (gy >= 0) & (gy < int(H))
    if not bool(inb.any()): return int(min_cap), {"reason": "no points in bbox"}
    gx = gx[inb]; gy = gy[inb]
    in_sh = in_shell[gy, gx].bool()
    if not bool(in_sh.any()): return int(min_cap), {"reason": "no points in shell"}
    gx = gx[in_sh]; gy = gy[in_sh]
    cell = (gy * int(W) + gx).long()
    counts = torch.bincount(cell, minlength=int(H) * int(W)).float()
    shell_mask = in_shell.reshape(-1).bool()
    counts_shell = counts[shell_mask]
    if ignore_zeros: counts_shell = counts_shell[counts_shell > 0]
    if counts_shell.numel() == 0: return int(min_cap), {"reason": "all-zero shell counts"}
    try: qv = torch.quantile(counts_shell, float(q)).item()
    except Exception:
        k = max(1, int(round(float(q) * (counts_shell.numel() - 1))) + 1)
        qv = counts_shell.kthvalue(k).values.item()
    cap = int(np.clip(int(np.ceil(qv)), int(min_cap), int(max_cap)))
    info = {"q": float(q), "quantile_value": float(qv), "mean": float(counts_shell.mean().item()), "max": float(counts_shell.max().item()), "n_cells_used": int(counts_shell.numel())}
    return cap, info

def auto_shell_need_cap_for_stage(coords_src: torch.Tensor, coords_tgt: torch.Tensor, grid_cache_dev: list, H: int, W: int, T: int, *, q: float = 0.90, min_cap: int = 1, max_cap: int = 8):
    cap0, info0 = estimate_shell_need_cap_from_coords(coords_src, grid_cache_dev[0], H, W, q=q, min_cap=min_cap, max_cap=max_cap)
    capT, infoT = estimate_shell_need_cap_from_coords(coords_tgt, grid_cache_dev[int(T)], H, W, q=q, min_cap=min_cap, max_cap=max_cap)
    cap = max(cap0, capT)
    return cap, {"src": info0, "tgt": infoT, "cap0": cap0, "capT": capT, "cap": cap}

# -------------------- differentiation CSV --------------------
def load_diff_csv_to_tensors(diff_csv: str, layer_to_idx: Dict[str, int], n_layers: int, device: torch.device):
    df = pd.read_csv(diff_csv)
    need_cols = {"src_layer", "tgt_layer"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"diff_csv must have columns {sorted(list(need_cols))}, got {list(df.columns)}")
    w = df["weight"].astype(float).to_numpy(np.float32) if "weight" in df.columns else np.ones(len(df), np.float32)
    src = df["src_layer"].astype(str).to_list(); tgt = df["tgt_layer"].astype(str).to_list()
    edges = [[] for _ in range(int(n_layers))]
    for s, t, ww in zip(src, tgt, w):
        if s not in layer_to_idx or t not in layer_to_idx: continue
        si = int(layer_to_idx[s]); ti = int(layer_to_idx[t])
        if si < 0 or si >= n_layers or ti < 0 or ti >= n_layers: continue
        if ww <= 0: continue
        edges[si].append((ti, float(ww)))
    deg = [len(x) for x in edges]; Kmax = max(deg) if max(deg) > 0 else 1
    tgt_idx = torch.full((n_layers, Kmax), -1, dtype=torch.long, device=device)
    tgt_w = torch.zeros((n_layers, Kmax), dtype=torch.float32, device=device)
    for si in range(n_layers):
        if len(edges[si]) == 0: continue
        lst = edges[si]
        for k, (ti, ww) in enumerate(lst[:Kmax]):
            tgt_idx[si, k] = int(ti); tgt_w[si, k] = float(ww)
    return tgt_idx, tgt_w

def build_old_policy_features(
    coords: torch.Tensor,
    density: torch.Tensor = None,
    lr: torch.Tensor = None,
    *,
    use_lr: bool = True,
):
    """
    old-mode features:
        [x, y, density, (lr)]
    coords:  (N, 2)
    density: (N, 1) or (N,)
    lr:      (N, 1) or (N,)
    """
    device = coords.device
    N = int(coords.shape[0])

    feats = [coords.to(torch.float32)]

    if density is None:
        density = torch.zeros((N, 1), device=device, dtype=torch.float32)
    elif density.ndim == 1:
        density = density.view(N, 1).to(torch.float32)
    else:
        density = density.to(torch.float32)
    feats.append(density)

    if use_lr:
        if lr is None:
            lr = torch.zeros((N, 1), device=device, dtype=torch.float32)
        elif lr.ndim == 1:
            lr = lr.view(N, 1).to(torch.float32)
        else:
            lr = lr.to(torch.float32)
        feats.append(lr)

    return torch.cat(feats, dim=1)

# ============================================================
# 1) GrowthPolicyNet (add diff head + learnable eta0)
# ============================================================
class GrowthPolicyNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        use_lr: bool = True,
        hidden_dim: int = 128,
        learn_eta0: bool = False,
        eta0_init: float = 0.0,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.use_lr = bool(use_lr)

        # coords(2) + density(1) + lr(1)
        input_dim = 2 + 1 + (1 if self.use_lr else 0)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 7),   # birth, death, diff, mu2, rho2
        )

        self.learn_eta0 = bool(learn_eta0)
        if self.learn_eta0:
            eta0_init = float(np.clip(float(eta0_init), 1e-4, 1 - 1e-4))
            self.eta0_logit = nn.Parameter(
                torch.tensor(np.log(eta0_init / (1 - eta0_init)), dtype=torch.float32)
            )
        else:
            self.register_buffer("eta0_const", torch.tensor(float(eta0_init), dtype=torch.float32))

    def eta0(self):
        if self.learn_eta0:
            return torch.sigmoid(self.eta0_logit)
        return self.eta0_const

    def forward(self, coords, lr=None, *, density=None):
        x = build_old_policy_features(
            coords=coords,
            density=density,
            lr=lr,
            use_lr=self.use_lr,
        )
        return self.net(x)

class AlphaTransitionNet(nn.Module):
    def __init__(self, use_lr=True, hidden_dim=128):
        super().__init__()
        self.use_lr = bool(use_lr)
        input_dim = 2 + 1 + (1 if self.use_lr else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, coords, lr=None, *, density=None):
        x = build_old_policy_features(coords=coords, density=density, lr=lr, use_lr=self.use_lr)
        return self.net(x)
    
# ============================================================
# 2) SimulationEnv: add differentiation
# ============================================================
class SimulationEnv:
    def __init__(
        self, state0: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor], device: torch.device,
        base_coords_seq: List[torch.Tensor], base_Z_seq: List[torch.Tensor], base_layers0: torch.Tensor,
        grid_cache_np: list, grid_cache_dev: list, shells_norm: list, T: int, n_layers: int, H: int, W: int,
        recompute_lr_fn, t0: int = 0, shell_need_cap: int = 2, use_lr: bool = True,
        advect_dist_q: float = 0.90, advect_dist_min: float = 0.02, advect_dist_max: float = 0.15, advect_blend_temp: float = 0.02,
        turnover_max_frac: float = 0.15, turnover_cap_frac: float = 0.20, turnover_p: float = 1.1, turnover_mismatch_mul: float = 2.0,
        w_need: float = 1.0, w_overlap: float = 2.0, w_gap: float = 1.0, gap_ema_beta: float = 0.90, alpha_net=None,
        # NEW: differentiation
        diff_tgt_idx: Optional[torch.Tensor] = None, diff_tgt_w: Optional[torch.Tensor] = None,
        diff_frac0: float = 0.00, diff_frac_min: float = 0.40, diff_pow: float = 1.6,
    ):
        self.device = device
        self.alpha_net = alpha_net
        self.base_coords_seq = base_coords_seq; self.base_Z_seq = base_Z_seq; self.base_layers0 = base_layers0.to(device)
        self.grid_cache_np = grid_cache_np; self.grid_cache_dev = grid_cache_dev; self.shells_norm = shells_norm
        self.shell_need_cap = int(shell_need_cap)
        self.T = int(T); self.t = int(t0)
        self.n_layers = int(n_layers); self.H = int(H); self.W = int(W)
        self.use_lr = bool(use_lr); self.recompute_lr = recompute_lr_fn

        self.advect_dist_q = float(advect_dist_q); self.advect_dist_min = float(advect_dist_min); self.advect_dist_max = float(advect_dist_max); self.advect_blend_temp = float(advect_blend_temp)
        self.turnover_max_frac = float(turnover_max_frac); self.turnover_cap_frac = float(turnover_cap_frac); self.turnover_p = float(turnover_p); self.turnover_mismatch_mul = float(turnover_mismatch_mul)
        self.w_need = float(w_need); self.w_overlap = float(w_overlap); self.w_gap = float(w_gap)
        self.gap_ema_beta = float(gap_ema_beta); self.gap_ema = torch.zeros((self.n_layers,), device=device, dtype=torch.float32)

        # diff transitions (si -> tj)
        self.diff_tgt_idx = None if diff_tgt_idx is None else diff_tgt_idx.to(device)
        self.diff_tgt_w = None if diff_tgt_w is None else diff_tgt_w.to(device)
        self.diff_frac0 = float(diff_frac0); self.diff_frac_min = float(diff_frac_min); self.diff_pow = float(diff_pow)

        self.tgt_coords = target_state["coords"].to(device); self.tgt_layers = target_state["layers"].to(device)
        self.tgt_latent = target_state.get("latent", None); self.tgt_latent = None if self.tgt_latent is None else self.tgt_latent.to(device)
        self.tgt_is_new = target_state.get("is_new", None); self.tgt_is_new = None if self.tgt_is_new is None else self.tgt_is_new.to(device)

        self.state = {
            "coords": state0["coords"].to(device).clone(),
            "layers": state0["layers"].to(device).clone(),
            "anchor": state0["anchor"].to(device).clone(),
        }
        lat0 = state0.get("latent", None)
        self.state["latent"] = None if lat0 is None else lat0.to(device).clone()
        ex0 = state0.get("expr_union", None)
        self.state["expr_union"] = None if ex0 is None else ex0.to(device).clone()
        lr0 = state0.get("lr", None)
        lr0 = torch.zeros((self.state["coords"].shape[0],), device=device, dtype=torch.float32) if lr0 is None else lr0.to(device)
        self.state["lr"] = lr0.clone()

        N0 = int(self.state["coords"].shape[0])
        self.state["is_birth"] = torch.zeros((N0,), dtype=torch.bool, device=device)
        self.state["born_step"] = torch.full((N0,), -1, dtype=torch.int32, device=device)
        self.state["is_diff"] = torch.zeros((N0,), dtype=torch.bool, device=device)
        self.state["diff_alpha"] = torch.zeros((N0,), dtype=torch.float32, device=device)   # publicdifferentiation score
        self.state["diff_mid"] = torch.zeros((N0,), dtype=torch.float32, device=device)     # internalintermediate state, trace
        self.state["diff_tgt_layer"] = torch.full((N0,), -1, dtype=torch.long, device=device)
        self.state["diff_enter_step"] = torch.full((N0,), -100000, dtype=torch.int32, device=device)
        self.state["commit_step"] = torch.full((N0,), -1, dtype=torch.int32, device=device)

        self.state["has_divided"] = torch.zeros((N0,), dtype=torch.bool, device=device)
        self.state["uid"] = torch.arange(N0, device=device, dtype=torch.long)
        self.state["parent_uid"] = torch.full((N0,), -1, device=device, dtype=torch.long)
        self._next_uid = int(N0)
        self._tgt_occ_cache = {}

        # target-latent layer means for fallback teacher
        if self.tgt_latent is not None:
            D = int(self.tgt_latent.shape[1])
            mean = torch.zeros((self.n_layers, D), device=device, dtype=torch.float32)
            cnt = torch.zeros((self.n_layers,), device=device, dtype=torch.float32)
            for li in range(self.n_layers):
                idx = (self.tgt_layers == li).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    mean[li] = self.tgt_latent[idx].mean(dim=0)
                    cnt[li] = float(idx.numel())
            self.tgt_latent_mean = mean
        else:
            self.tgt_latent_mean = None

        # dynamic per-layer pools for tid (teacher assignment, not identity)
        Nt = int(self.tgt_coords.shape[0])
        self.tgt_used = torch.zeros((Nt,), device=device, dtype=torch.bool)
        self.tgt_idx_by_layer = []
        for li in range(self.n_layers):
            self.tgt_idx_by_layer.append((self.tgt_layers == li).nonzero(as_tuple=True)[0].to(device))
        tid0 = state0.get("tid", None)
        if tid0 is None:
            tid0 = torch.full((N0,), -1, device=device, dtype=torch.long)
            self._assign_tid_for_indices(torch.arange(N0, device=device), self.state["layers"], tid0, seed=12345)
        else:
            tid0 = tid0.to(device).long().clone()
            ok = (tid0 >= 0) & (tid0 < Nt)
            if bool(ok.any()): self.tgt_used[tid0[ok]] = True
        self.state["tid"] = tid0

        last_parent_step = state0.get("last_parent_step", None)
        if last_parent_step is None or int(last_parent_step.numel()) != N0:
            last_parent_step = torch.full((N0,), -100000, dtype=torch.int32, device=device)
        self.state["last_parent_step"] = last_parent_step.clone()

    def _get_occ_tgt_raw(self, ti: int) -> torch.Tensor:
        if ti not in self._tgt_occ_cache:
            grid_item = self.grid_cache_dev[int(ti)]
            v = grid_occupancy_by_layer(self.tgt_coords, self.tgt_layers, grid_item, self.H, self.W, self.n_layers, normalize=False)
            self._tgt_occ_cache[ti] = v
        return self._tgt_occ_cache[ti]

    @torch.no_grad()
    def _assign_tid_for_indices(self, idx: torch.Tensor, layers_for_idx: torch.Tensor, tid_out: torch.Tensor, seed: int = 0):
        dev = idx.device
        idx = idx.long().view(-1)
        # safety clamp: avoid any accidental OOB
        idx = idx[(idx >= 0) & (idx < int(tid_out.numel()))]
        if idx.numel() == 0: return

        layers_for_idx = layers_for_idx.to(device=dev).long().view(-1)
        # support both "global layers" and "per-idx layers"
        if int(layers_for_idx.numel()) == int(idx.numel()):
            layers_sel = layers_for_idx
        else:
            layers_sel = layers_for_idx[idx]

        gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
        for li in range(self.n_layers):
            sel = idx[(layers_sel == li)]
            if sel.numel() == 0: continue
            pool = self.tgt_idx_by_layer[li]
            if pool.numel() == 0:
                tid_out[sel] = -1
                continue
            free = pool[~self.tgt_used[pool]]
            if free.numel() == 0:
                tid_out[sel] = -1
                continue
            perm = torch.randperm(free.numel(), generator=gen, device=dev)
            free = free[perm]
            take = min(int(sel.numel()), int(free.numel()))
            tid_out[sel[:take]] = free[:take]
            self.tgt_used[free[:take]] = True
            if take < int(sel.numel()): tid_out[sel[take:]] = -1

    @torch.no_grad()
    def _release_tid(self, tid: torch.Tensor):
        ok = (tid >= 0) & (tid < self.tgt_used.numel())
        if bool(ok.any()): self.tgt_used[tid[ok]] = False

    @torch.no_grad()
    def _pick_diff_targets(self, src_layers: torch.Tensor, gap_pos: torch.Tensor, seed: int = 0) -> torch.Tensor:
        # for each src layer, sample target layer using (edge_w * gap_pos[tgt]); fallback to argmax(gap_pos)
        dev = src_layers.device
        L = int(self.n_layers)
        gap_pos = gap_pos.float().clamp_min(0.0)
        fallback = int(torch.argmax(gap_pos).item()) if float(gap_pos.sum().item()) > 0 else 0
        if self.diff_tgt_idx is None or self.diff_tgt_w is None:
            return torch.full((src_layers.shape[0],), int(fallback), device=dev, dtype=torch.long)

        Kmax = int(self.diff_tgt_idx.shape[1])
        gen = torch.Generator(device=dev); gen.manual_seed(int(seed))
        out = torch.full((src_layers.shape[0],), int(fallback), device=dev, dtype=torch.long)
        for li in range(L):
            mask = (src_layers == li)
            if not bool(mask.any()): continue
            tgts = self.diff_tgt_idx[li]  # (Kmax,)
            ws = self.diff_tgt_w[li].clamp_min(0.0)
            ok = tgts >= 0
            if not bool(ok.any()): continue
            tgts_ok = tgts[ok]; ws_ok = ws[ok]
            w = ws_ok * gap_pos[tgts_ok].clamp_min(0.0)
            if float(w.sum().item()) <= 1e-12: w = ws_ok
            if float(w.sum().item()) <= 1e-12: continue
            p = (w / w.sum()).detach()
            n = int(mask.sum().item())
            pick = torch.multinomial(p, n, replacement=True, generator=gen)
            out[mask] = tgts_ok[pick]
        return out

    def step(
        self, policy_net, birth_quota_layer: np.ndarray, death_quota_layer: np.ndarray,
        dir_step: int, advect_latent: bool = True, seed_step: int = 2025, *,
        TAU_BIRTH: float, TAU_DEATH: float, LATENT_NOISE_SCALE: float, NEED_GAMMA: float = 2.0,
        LOCAL_NEED_RADIUS: int = 16, PARENT_COOLDOWN: int = 0, BIRTH_HOT_FRAC: float = 0.6,
        CUR_DILATE_MAX: int = 3, LAMBDA_PARENT: float = 1.0, CROWD_DEATH_W: float = 2.0,  TAU_DIFF: float = 1.0,
    ):
        dev = self.device
        assert dir_step in (+1, -1)
        t_now = int(self.t)
        t_next = t_now + int(dir_step)
        if t_next < 0 or t_next > int(self.T):
            z0 = torch.zeros(1, device=dev).squeeze()
            return z0, z0

        coords = self.state["coords"]
        layers = self.state["layers"]
        anchor = self.state["anchor"]
        lr_vals = self.state["lr"]
        latent = self.state.get("latent", None)
        expr_union = self.state.get("expr_union", None)
        born_step = self.state["born_step"]
        uid = self.state["uid"]
        parent_uid = self.state["parent_uid"]
        has_divided = self.state.get("has_divided", torch.zeros((coords.shape[0],), dtype=torch.bool, device=dev))
        last_parent_step = self.state.get("last_parent_step", torch.full((coords.shape[0],), -100000, dtype=torch.int32, device=dev))

        N = int(coords.shape[0])
        if N <= 0:
            self.t = t_next
            z0 = torch.zeros(1, device=dev).squeeze()
            return z0, z0

        def _state_or_default(key, default_tensor):
            x = self.state.get(key, None)
            if x is None or int(x.numel()) != int(default_tensor.numel()):
                return default_tensor.clone()
            return x.to(dev).clone()

        def _fix_centers_len(centers, ref_xy):
            if centers is None or (not torch.is_tensor(centers)) or centers.ndim != 2 or centers.shape[1] != 2:
                return ref_xy
            m, n = int(centers.shape[0]), int(ref_xy.shape[0])
            if m == n:
                return centers
            if m > n:
                return centers[:n]
            return torch.cat([centers, ref_xy[m:n]], dim=0)

        def _dilate_hw(mask_hw_bool: torch.Tensor, px: int):
            px = int(px)
            if px <= 0:
                return mask_hw_bool
            k = 2 * px + 1
            x = mask_hw_bool.to(torch.float32)[None, None, :, :]
            y = F.max_pool2d(x, kernel_size=k, stride=1, padding=px)
            return (y[0, 0] > 0)

        def _diff_active_mask(is_diff_, diff_tgt_layer_, commit_step_):
            return is_diff_ & (diff_tgt_layer_ >= 0) & (commit_step_ < 0)

        def _effective_counts(layers_, is_diff_, diff_alpha_, diff_tgt_layer_, commit_step_):
            hard = torch.bincount(layers_.long().clamp(0, self.n_layers - 1), minlength=self.n_layers).float()
            act = _diff_active_mask(is_diff_, diff_tgt_layer_, commit_step_)
            if bool(act.any()):
                src_dec = torch.bincount(
                    layers_[act].long().clamp(0, self.n_layers - 1),
                    weights=diff_alpha_[act].float(),
                    minlength=self.n_layers,
                ).float()
                tgt_inc = torch.bincount(
                    diff_tgt_layer_[act].long().clamp(0, self.n_layers - 1),
                    weights=diff_alpha_[act].float(),
                    minlength=self.n_layers,
                ).float()
                hard = hard - src_dec + tgt_inc
            return hard.clamp_min(0.0)

        def _soft_occ_by_layer_diff(coords_, layers_, is_diff_, diff_alpha_, diff_tgt_layer_, commit_step_, grid_item_dev_, H_, W_, n_layers_):
            x0_, y0_, dx_, dy_, in_shell_ = grid_item_dev_
            gx = torch.floor((coords_[:, 0] - x0_) / dx_).long()
            gy = torch.floor((coords_[:, 1] - y0_) / dy_).long()
            m = (gx >= 0) & (gx < W_) & (gy >= 0) & (gy < H_)
            valid_idx = m.nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 0:
                valid_idx = valid_idx[in_shell_[gy[valid_idx], gx[valid_idx]].bool()]
            occ = torch.zeros((n_layers_, H_, W_), device=coords_.device, dtype=torch.float32)
            if valid_idx.numel() == 0:
                return occ
            HW = H_ * W_
            act_all = _diff_active_mask(is_diff_, diff_tgt_layer_, commit_step_)
            act_v = act_all[valid_idx]
            idx_non = valid_idx[~act_v]
            if idx_non.numel() > 0:
                flat = (layers_[idx_non].long().clamp(0, n_layers_ - 1) * HW + gy[idx_non] * W_ + gx[idx_non]).long()
                occ += torch.bincount(flat, minlength=n_layers_ * HW).float().view(n_layers_, H_, W_)
            idx_act = valid_idx[act_v]
            if idx_act.numel() > 0:
                cell_a = (gy[idx_act] * W_ + gx[idx_act]).long()
                a = diff_alpha_[idx_act].float().clamp(0.0, 1.0)
                src = layers_[idx_act].long().clamp(0, n_layers_ - 1)
                tgt = diff_tgt_layer_[idx_act].long().clamp(0, n_layers_ - 1)
                flat_s = (src * HW + cell_a).long()
                flat_t = (tgt * HW + cell_a).long()
                occ += torch.bincount(flat_s, weights=(1.0 - a), minlength=n_layers_ * HW).float().view(n_layers_, H_, W_)
                occ += torch.bincount(flat_t, weights=a, minlength=n_layers_ * HW).float().view(n_layers_, H_, W_)
            return occ

        @torch.no_grad()
        def _knn_mean_target_latent(q_xy: torch.Tensor, li: int, *, k: int, chunk_q: int, max_ref: int, seed: int):
            tgt_coords = getattr(self, "tgt_coords", None)
            tgt_layers = getattr(self, "tgt_layers", None)
            tgt_latent = getattr(self, "tgt_latent", None)
            if tgt_coords is None or tgt_layers is None or tgt_latent is None or tgt_latent.numel() == 0:
                return None
            ref_idx = (tgt_layers == int(li)).nonzero(as_tuple=True)[0]
            if ref_idx.numel() == 0:
                return None
            if ref_idx.numel() > int(max_ref):
                g = torch.Generator(device=dev)
                g.manual_seed(int(seed))
                perm = torch.randperm(ref_idx.numel(), generator=g, device=dev)[:int(max_ref)]
                ref_idx = ref_idx[perm]
            ref_xy = tgt_coords[ref_idx].to(dev)
            ref_z = tgt_latent[ref_idx].to(dev)
            kk = min(int(k), int(ref_xy.shape[0]))
            if kk <= 0:
                return None
            out = torch.empty((q_xy.shape[0], ref_z.shape[1]), device=dev, dtype=ref_z.dtype)
            for s in range(0, int(q_xy.shape[0]), int(chunk_q)):
                qq = q_xy[s:s + int(chunk_q)].to(dev)
                d2 = (qq[:, None, :] - ref_xy[None, :, :]).pow(2).sum(dim=2)
                idx = torch.topk(d2, k=kk, largest=False).indices
                out[s:s + int(chunk_q)] = ref_z[idx].mean(dim=1)
            return out

        def _diff_target_mask():
            mask = torch.zeros((self.n_layers,), device=dev, dtype=torch.bool)
            diff_tgt_idx = getattr(self, "diff_tgt_idx", None)
            if diff_tgt_idx is None:
                return mask
            for s in range(self.n_layers):
                tgts = diff_tgt_idx[s]
                ok = tgts >= 0
                if bool(ok.any()):
                    mask[tgts[ok].long().clamp(0, self.n_layers - 1)] = True
            return mask

        def _build_source_birth_drive_map(need_map_base: torch.Tensor, occ_soft_now: torch.Tensor, free_hw_: torch.Tensor):
            out = need_map_base.clone()
            if self.diff_tgt_idx is None or self.diff_tgt_w is None:
                return out
            precursor_birth_w = float(getattr(self, "precursor_birth_w", 0.65))
            precursor_frontier_px = int(getattr(self, "precursor_frontier_px", max(4, LOCAL_NEED_RADIUS // 2)))
            precursor_dilate_px = int(getattr(self, "precursor_dilate_px", 4))
            tissue_hw = (occ_soft_now.sum(dim=0) > 0)
            frontier_hw = _dilate_hw(tissue_hw, precursor_frontier_px) & allow_shell
            frontier_w = frontier_hw.float() * (0.25 + 0.75 * free_hw_)
            for src in range(self.n_layers):
                tgts = self.diff_tgt_idx[src]
                ws = self.diff_tgt_w[src].float().clamp_min(0.0)
                ok = (tgts >= 0)
                if not bool(ok.any()):
                    continue
                tgts_ok = tgts[ok].long()
                ws_ok = ws[ok]
                if float(ws_ok.sum().item()) <= 1e-12:
                    continue
                ws_ok = ws_ok / ws_ok.sum().clamp_min(1e-12)
                borrowed = torch.zeros((self.H, self.W), device=dev, dtype=torch.float32)
                for k in range(int(tgts_ok.numel())):
                    tgt = int(tgts_ok[k].item())
                    w = float(ws_ok[k].item())
                    need_t = need_map_base[tgt]
                    if float(need_t.sum().item()) <= 0:
                        continue
                    borrow_mask = _dilate_hw(need_t > 0, precursor_dilate_px).float()
                    borrowed = borrowed + w * need_t * borrow_mask
                if float(borrowed.sum().item()) <= 0:
                    continue
                out[src] = out[src] + precursor_birth_w * borrowed * frontier_w
            return out

        @torch.no_grad()
        def _pick_diff_targets_local(src_layers_: torch.Tensor, src_coords_: torch.Tensor, need_map_: torch.Tensor, seed: int = 0):
            B = int(src_layers_.shape[0])
            out = torch.full((B,), -1, device=dev, dtype=torch.long)
            if B == 0:
                return out
            eps = 1e-12
            global_need = need_map_.sum(dim=(1, 2)).float().clamp_min(0.0)
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(seed))
            for li in range(self.n_layers):
                row_idx = (src_layers_ == li).nonzero(as_tuple=True)[0]
                if row_idx.numel() == 0:
                    continue
                if self.diff_tgt_idx is not None and self.diff_tgt_w is not None:
                    tgts = self.diff_tgt_idx[li]
                    ws = self.diff_tgt_w[li].float().clamp_min(0.0)
                    ok = (tgts >= 0)
                    tgts = tgts[ok].long()
                    ws = ws[ok]
                else:
                    tgts = torch.arange(self.n_layers, device=dev, dtype=torch.long)
                    tgts = tgts[tgts != int(li)]
                    ws = torch.ones((tgts.numel(),), device=dev, dtype=torch.float32)
                if tgts.numel() == 0:
                    continue
                local_cols = []
                for tgt in tgts.tolist():
                    v = gather_map_at_coords_fast(need_map_[int(tgt)], src_coords_[row_idx], grid_item_dev, self.H, self.W, default=0.0).float()
                    local_cols.append(v)
                local_need = torch.stack(local_cols, dim=1)
                score = ws.view(1, -1) * (1e-3 + local_need)
                row_sum = score.sum(dim=1)
                bad = row_sum <= eps
                if bool(bad.any()):
                    base = ws.view(1, -1) * (1e-3 + global_need[tgts].view(1, -1))
                    score[bad] = base.expand(int(bad.sum().item()), -1)
                    row_sum = score.sum(dim=1)
                good = row_sum > eps
                if bool(good.any()):
                    p = score[good] / row_sum[good].unsqueeze(1).clamp_min(eps)
                    pick = torch.multinomial(p, num_samples=1, replacement=True, generator=gen).squeeze(1)
                    out[row_idx[good]] = tgts[pick]
            return out

        def _make_density_map(total_occ_hw: torch.Tensor, smooth_k: int = 5):
            smooth_k = int(max(1, smooth_k))
            if smooth_k <= 1:
                return total_occ_hw.float()
            x = total_occ_hw.float().view(1, 1, self.H, self.W)
            y = F.avg_pool2d(x, kernel_size=smooth_k, stride=1, padding=smooth_k // 2)
            return y[0, 0]

        is_diff = _state_or_default("is_diff", torch.zeros((N,), dtype=torch.bool, device=dev))
        diff_alpha = _state_or_default("diff_alpha", torch.zeros((N,), dtype=torch.float32, device=dev))
        diff_mid = _state_or_default("diff_mid", torch.zeros((N,), dtype=torch.float32, device=dev))
        diff_tgt_layer = _state_or_default("diff_tgt_layer", torch.full((N,), -1, dtype=torch.long, device=dev))
        diff_enter_step = _state_or_default("diff_enter_step", torch.full((N,), -100000, dtype=torch.int32, device=dev))
        commit_step = _state_or_default("commit_step", torch.full((N,), -1, dtype=torch.int32, device=dev))

        bq_plan_total = int(np.asarray(birth_quota_layer, np.int64).sum()) if birth_quota_layer is not None else 0
        dq_plan_total = int(np.asarray(death_quota_layer, np.int64).sum()) if death_quota_layer is not None else 0
        B_base = int(max(bq_plan_total, 0))
        D_base = int(max(dq_plan_total, 0))
        D_base = min(D_base, N)
        deltaN = int(B_base - D_base)

        base_t = self.base_coords_seq[t_now]
        base_tp1 = self.base_coords_seq[t_next]
        coords = advect_coords_hybrid_soft(
            coords,
            anchor=anchor,
            base_t=base_t,
            base_tp1=base_tp1,
            dist_q=self.advect_dist_q,
            dist_min=self.advect_dist_min,
            dist_max=self.advect_dist_max,
            blend_temp=self.advect_blend_temp,
            knn_k=6,
            knn_max_ref=20000,
            knn_seed=100000 + int(seed_step),
        )

        is_new_now = (born_step.to(dev).view(-1) == int(t_now))
        grid_item_dev = self.grid_cache_dev[t_next]
        occ_tgt_raw = self._get_occ_tgt_raw(t_next)
        allow_shell = grid_item_dev[4].bool()
        tgt_allow_hw = (occ_tgt_raw.sum(dim=0) > 0) & allow_shell
        if not bool(tgt_allow_hw.any()):
            tgt_allow_hw = allow_shell

        x0, y0, dx, dy = grid_item_dev[0], grid_item_dev[1], grid_item_dev[2], grid_item_dev[3]
        j = torch.round((coords[:, 0] - x0) / dx).to(torch.long)
        i = torch.round((coords[:, 1] - y0) / dy).to(torch.long)
        inside = (i >= 0) & (i < int(self.H)) & (j >= 0) & (j < int(self.W))
        ii, jj = i.clamp(0, int(self.H) - 1), j.clamp(0, int(self.W) - 1)
        good = inside & allow_shell[ii, jj]
        bad_idx = (~good).nonzero(as_tuple=True)[0]

        if bad_idx.numel() > 0:
            good_idx = good.nonzero(as_tuple=True)[0]
            if good_idx.numel() > 0:
                occ_good, _ = get_occ_and_crowd(coords[good_idx], layers[good_idx], grid_item_dev, self.H, self.W, self.n_layers, cap=self.shell_need_cap)
                tissue = (occ_good.sum(dim=0) > 0)
                tissue_d = _dilate_hw(tissue, px=2) & allow_shell
                ring = (tissue_d & (~tissue)) & allow_shell
                target_hw = ring if bool(ring.any()) else (tissue_d if bool(tissue_d.any()) else allow_shell)
            else:
                target_hw = allow_shell
            target_hw = (target_hw & tgt_allow_hw)
            if not bool(target_hw.any()):
                target_hw = tgt_allow_hw if bool(tgt_allow_hw.any()) else allow_shell
            coords_bad = project_to_allowed_mask(coords[bad_idx], target_hw, grid_item_dev, self.H, self.W, chunk_q=4096, max_ref=12000, seed=int(123 + seed_step))
            coords = coords.clone()
            coords[bad_idx] = coords_bad

        _, crowd_all = get_occ_and_crowd(coords, layers, grid_item_dev, self.H, self.W, self.n_layers, cap=self.shell_need_cap)
        occ_cur_soft = _soft_occ_by_layer_diff(coords, layers, is_diff, diff_alpha, diff_tgt_layer, commit_step, grid_item_dev, self.H, self.W, self.n_layers)
        need_map = (occ_tgt_raw - occ_cur_soft).clamp_min(0.0)
        overlap_map = (occ_cur_soft - occ_tgt_raw).clamp_min(0.0)

        density_map = _make_density_map(occ_cur_soft.sum(dim=0), smooth_k=5)
        density_here = gather_map_at_coords_fast(density_map, coords, grid_item_dev, self.H, self.W, default=0.0).float()

        desired_layer = occ_tgt_raw.sum(dim=(1, 2)).float()
        cur_layer_eff = _effective_counts(layers, is_diff, diff_alpha, diff_tgt_layer, commit_step)
        gap = desired_layer - cur_layer_eff
        beta = float(self.gap_ema_beta)
        self.gap_ema = beta * self.gap_ema + (1.0 - beta) * gap.detach()

        desired_total = float(desired_layer.sum().clamp_min(1.0).item())
        mismatch = float((need_map.sum() + overlap_map.sum()).detach().cpu().item()) / desired_total
        mismatch01 = float(np.clip(mismatch, 0.0, 1.0))
        t_frac = float(t_now) / max(int(self.T), 1)
        warm_start, warm_end = 0.10, 0.70
        warm = float(np.clip((t_frac - warm_start) / max(warm_end - warm_start, 1e-6), 0.0, 1.0))
        warm = warm * warm

        cap = float(self.shell_need_cap)
        total_occ = occ_cur_soft.sum(dim=0)
        overcap_map = (total_occ - cap).clamp_min(0.0)
        overcap = float(overcap_map.sum().detach().cpu().item())
        missing = float(need_map.sum().detach().cpu().item())
        excess = float(overlap_map.sum().detach().cpu().item())
        move_cap = int(max(0.0, round(min(missing, excess + overcap))))
        K_goal = int(round(move_cap * (0.10 + 0.90 * mismatch01)))
        K_cap = int(round(self.turnover_cap_frac * N))
        turn_sched = float(0.2 + 0.2 * t_frac ** 1.5)
        K_raw = int(round(turn_sched * K_goal))
        K_extra = int(np.clip(K_raw - min(B_base, D_base), 0, K_cap))
        D_total = min(N, D_base + K_extra)
        B_total = int(B_base + K_extra)
        paired = int(min(B_total, D_total))

        overlap_here = torch.zeros((N,), device=dev, dtype=torch.float32)
        for li in range(self.n_layers):
            idx = (layers == li).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            overlap_here[idx] = gather_map_at_coords_fast(overlap_map[li], coords[idx], grid_item_dev, self.H, self.W, default=0.0).float()

        lr_in = None
        if self.use_lr and (lr_vals is not None) and (lr_vals.numel() > 0):
            lr_z = (lr_vals - lr_vals.mean()) / (lr_vals.std() + 1e-6)
            lr_in = lr_z.detach().view(N, 1)

        out = policy_net(coords.detach(), lr=lr_in, density=torch.log1p(density_here).view(N, 1))
        birth_logits = out[:, 0]
        death_logits = out[:, 1]
        diff_logits = out[:, 2]
        place_mu_raw = out[:, 3:5]
        place_rho_raw = out[:, 5:7]
        ent = torch.zeros(1, device=dev)

        use_diff = (getattr(self, "tgt_latent", None) is not None) and (getattr(self, "tgt_coords", None) is not None) and (getattr(self, "tgt_layers", None) is not None)
        pre_diff_mask = torch.zeros((N,), dtype=torch.bool, device=dev)
        pre_diff_tgt = torch.full((N,), -1, dtype=torch.long, device=dev)
        diff_logp = torch.zeros(1, device=dev)
        K_diff = 0

        if use_diff and paired > 0:
            can_diff_pre = (~is_new_now) & (~is_diff)
            idx_pool = can_diff_pre.nonzero(as_tuple=True)[0]
            if idx_pool.numel() > 0:
                p_cell = torch.sigmoid((diff_logits[idx_pool] - 0.2) / TAU_DIFF).clamp_min(1e-8)
                K_cap_model = int(torch.round(p_cell.sum()).item())
                K_cap_model = int(min(K_cap_model, paired, int(idx_pool.numel())))
                if K_cap_model > 0:
                    probs = p_cell / p_cell.sum().clamp_min(1e-12)
                    idx_local, logpDiff = sample_no_replace(probs, K_cap_model)
                    idx_sel = idx_pool[idx_local]
                    tgt_sel = _pick_diff_targets_local(layers[idx_sel], coords[idx_sel], need_map, seed=int(330000 + seed_step))
                    ok = (tgt_sel >= 0) & (tgt_sel != layers[idx_sel])
                    idx_sel = idx_sel[ok]
                    tgt_sel = tgt_sel[ok]
                    if idx_sel.numel() > 0:
                        pre_diff_mask[idx_sel] = True
                        pre_diff_tgt[idx_sel] = tgt_sel
                        diff_logp = diff_logp + logpDiff
                        K_diff = int(idx_sel.numel())

        overlap_sum_layer = overlap_map.sum(dim=(1, 2)).float()
        hard_count_layer = torch.bincount(layers.long().clamp(0, self.n_layers - 1), minlength=self.n_layers).float()
        if bool(pre_diff_mask.any()):
            pre_diff_count_layer = torch.bincount(layers[pre_diff_mask].long().clamp(0, self.n_layers - 1), minlength=self.n_layers).float()
        else:
            pre_diff_count_layer = torch.zeros((self.n_layers,), device=dev)
        death_cap_layer = (hard_count_layer.long() - pre_diff_count_layer.long()).clamp_min(0)

        crowd_sum_layer = torch.zeros((self.n_layers,), device=dev, dtype=torch.float32)
        for li in range(self.n_layers):
            idx = (layers == li).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                crowd_sum_layer[li] = crowd_all[idx].sum()

        D_exec_plan = int(max(D_total - K_diff, 0))
        wD = self.w_gap * (-self.gap_ema).clamp_min(0.0) + self.w_overlap * overlap_sum_layer + float(CROWD_DEATH_W) * crowd_sum_layer
        if float(wD.sum().item()) <= 1e-8:
            wD = (death_cap_layer > 0).float()

        dq_t = quota_from_weights(D_exec_plan, wD, max_cap=death_cap_layer)
        D_exec = int(dq_t.sum().item())
        B_exec = int(max(D_exec + deltaN, 0))

        death_logp = torch.zeros(1, device=dev)
        keep_mask = torch.ones(N, dtype=torch.bool, device=dev)

        if D_exec > 0:
            idx_die_all, D_done = [], 0
            for li in range(self.n_layers):
                d_need = int(dq_t[li].item())
                if d_need <= 0:
                    continue
                idx_li = (layers == li).nonzero(as_tuple=True)[0]
                if idx_li.numel() == 0:
                    continue
                idx_li = idx_li[~pre_diff_mask[idx_li]]
                if idx_li.numel() == 0:
                    continue
                take = min(d_need, int(idx_li.numel()))
                ov = overlap_here[idx_li]
                cr = crowd_all[idx_li]
                score = ov + float(CROWD_DEATH_W) * cr
                cand = idx_li[(score > 0)]
                if int(cand.numel()) < take:
                    k2 = min(int(max(take * 3, take)), int(idx_li.numel()))
                    cand = idx_li[torch.topk(score, k=k2, largest=True).indices]
                if cand.numel() == 0:
                    continue
                probs = F.softmax(death_logits[cand] / float(TAU_DEATH), dim=0)
                idx_local, logpD = sample_no_replace(probs, take)
                death_logp = death_logp + logpD
                if idx_local.numel() > 0:
                    idx_die_all.append(cand[idx_local])
                    D_done += int(idx_local.numel())

            D_left = int(D_exec - D_done)
            if D_left > 0:
                killed = torch.cat(idx_die_all, dim=0) if len(idx_die_all) else torch.empty(0, device=dev, dtype=torch.long)
                cand_mask = torch.ones(N, dtype=torch.bool, device=dev)
                if killed.numel() > 0:
                    cand_mask[killed] = False
                cand_mask[pre_diff_mask] = False
                idx_cand = cand_mask.nonzero(as_tuple=True)[0]
                if idx_cand.numel() > 0:
                    ov = overlap_here[idx_cand]
                    cr = crowd_all[idx_cand]
                    score = ov + float(CROWD_DEATH_W) * cr
                    cand = idx_cand[(score > 0)]
                    if int(cand.numel()) < D_left:
                        k2 = min(int(max(D_left * 3, D_left)), int(idx_cand.numel()))
                        cand = idx_cand[torch.topk(score, k=k2, largest=True).indices]
                    take = min(D_left, int(cand.numel()))
                    if take > 0:
                        probs = F.softmax(death_logits[cand] / float(TAU_DEATH), dim=0)
                        idx_local, logpD = sample_no_replace(probs, take)
                        death_logp = death_logp + logpD
                        if idx_local.numel() > 0:
                            idx_die_all.append(cand[idx_local])

            if len(idx_die_all) > 0:
                idx_die = torch.cat(idx_die_all, dim=0)
                keep_mask[idx_die] = False

        surv_idx = keep_mask.nonzero(as_tuple=True)[0]
        coords_surv = coords[surv_idx]
        layers_surv = layers[surv_idx]
        anchor_surv = anchor[surv_idx]
        birth_logits_surv = birth_logits[surv_idx]
        mu_surv = place_mu_raw[surv_idx]
        rho_surv = place_rho_raw[surv_idx]
        latent_surv = None if latent is None else latent[surv_idx]
        expr_surv = None if expr_union is None else expr_union[surv_idx]
        uid_surv = uid[surv_idx]
        parent_uid_surv = parent_uid[surv_idx]
        born_step_surv = born_step[surv_idx].clone()
        has_div_surv = has_divided[surv_idx].clone()
        last_parent_surv = last_parent_step[surv_idx].clone()
        is_new_surv_now = is_new_now[surv_idx]
        is_diff_surv = is_diff[surv_idx].clone()
        diff_alpha_surv = diff_alpha[surv_idx].clone()
        diff_mid_surv = diff_mid[surv_idx].clone()
        diff_tgt_layer_surv = diff_tgt_layer[surv_idx].clone()
        diff_enter_step_surv = diff_enter_step[surv_idx].clone()
        commit_step_surv = commit_step[surv_idx].clone()

        if K_diff > 0 and surv_idx.numel() > 0:
            sel_diff_surv = pre_diff_mask[surv_idx]
            if bool(sel_diff_surv.any()):
                tgt_surv = pre_diff_tgt[surv_idx][sel_diff_surv]
                is_diff_surv[sel_diff_surv] = True
                diff_alpha_surv[sel_diff_surv] = 0.0
                diff_mid_surv[sel_diff_surv] = 0.0
                diff_tgt_layer_surv[sel_diff_surv] = tgt_surv
                diff_enter_step_surv[sel_diff_surv] = int(t_next)
                commit_step_surv[sel_diff_surv] = -1

        occ_cur_soft_surv = _soft_occ_by_layer_diff(coords_surv, layers_surv, is_diff_surv, diff_alpha_surv, diff_tgt_layer_surv, commit_step_surv, grid_item_dev, self.H, self.W, self.n_layers)
        total_occ_surv = occ_cur_soft_surv.sum(dim=0)
        need_map_surv = (occ_tgt_raw - occ_cur_soft_surv).clamp_min(0.0)

        free_hw = (cap - total_occ_surv).clamp_min(0.0) / cap
        tgt_any_hw = (occ_tgt_raw.sum(dim=0) > 0)
        hole_hw = (free_hw > 0) & tgt_any_hw
        cur_allow = (occ_cur_soft_surv > 0)

        dil_px = int(max(1, round(float(CUR_DILATE_MAX) * (1.0 - warm))))
        cur_allow_d = torch.zeros_like(cur_allow)
        for li in range(self.n_layers):
            cur_allow_d[li] = _dilate_hw(cur_allow[li], dil_px)

        tgt_allow = (occ_tgt_raw > 0)
        allow = cur_allow_d.float() + (1.0 - warm) * hole_hw.float().unsqueeze(0) + warm * tgt_allow.float()
        allow = allow.clamp(0.0, 1.0)
        need_map_birth = need_map_surv * allow * (0.10 + free_hw).pow(2.0).unsqueeze(0)
        birth_drive_map = _build_source_birth_drive_map(need_map_birth, occ_cur_soft_surv, free_hw)

        tissue_surv = (total_occ_surv > 0)
        tissue_surv_d = _dilate_hw(tissue_surv, px=2) & allow_shell
        ring_surv = (tissue_surv_d & (~tissue_surv)) & allow_shell
        target_hw_surv = ring_surv if bool(ring_surv.any()) else (tissue_surv_d if bool(tissue_surv_d.any()) else allow_shell)

        desired_layer_surv = occ_tgt_raw.sum(dim=(1, 2)).float()
        cur_hard_count_surv = torch.bincount(layers_surv.long().clamp(0, self.n_layers - 1), minlength=self.n_layers).float()
        gap_hard_surv = (desired_layer_surv - cur_hard_count_surv).clamp_min(0.0)
        need_sum_layer_surv = birth_drive_map.sum(dim=(1, 2)).float()

        wB = self.w_gap * gap_hard_surv + self.w_need * need_sum_layer_surv
        if float(wB.sum().item()) <= 1e-8:
            wB = desired_layer_surv.clamp_min(0.0)

        base_has_layer = torch.bincount(self.base_layers0.long().clamp(0, self.n_layers - 1), minlength=self.n_layers).bool()
        diff_tgt_mask = _diff_target_mask()
        has_hard_seed = cur_hard_count_surv >= 1

        wB_eff = wB.clone()
        forbid_from_nothing = diff_tgt_mask & (~base_has_layer) & (~has_hard_seed)
        wB_eff[forbid_from_nothing] = 0.0
        if float(wB_eff.sum().item()) <= 1e-8:
            wB_eff = wB

        cool_ok = (t_now - last_parent_surv.to(torch.int32)) > int(PARENT_COOLDOWN)
        elig = cool_ok & (~is_new_surv_now) 
        elig_count_layer = torch.bincount(layers_surv[elig].long().clamp(0, self.n_layers - 1), minlength=self.n_layers).long()

        base_seed_birth_cap = int(getattr(self, "base_seed_birth_cap", 2))
        base_seed_cap_layer = torch.where(
            (cur_hard_count_surv.long() == 0) & base_has_layer,
            torch.full((self.n_layers,), base_seed_birth_cap, device=dev, dtype=torch.long),
            torch.zeros((self.n_layers,), device=dev, dtype=torch.long),
        )
        birth_cap_layer = elig_count_layer + base_seed_cap_layer

        def _quota_from_weights_capped(total: int, weights: torch.Tensor, max_cap: torch.Tensor):
            total = int(max(total, 0))
            cap_local = max_cap.long().clone()
            out = torch.zeros_like(cap_local)
            if total <= 0 or int(cap_local.sum().item()) <= 0:
                return out
            total = min(total, int(cap_local.sum().item()))
            remain = total
            while remain > 0:
                resid = (cap_local - out).clamp_min(0)
                active = resid > 0
                if not bool(active.any()):
                    break
                w = weights.float().clone()
                w[~active] = 0.0
                if float(w.sum().item()) <= 1e-12:
                    w = active.float()
                probs = w / w.sum().clamp_min(1e-12)
                ideal = probs * float(remain)
                base_add = torch.floor(ideal).long()
                base_add = torch.minimum(base_add, resid)
                added = int(base_add.sum().item())
                if added > 0:
                    out += base_add
                    remain -= added
                    if remain <= 0:
                        break
                resid = (cap_local - out).clamp_min(0)
                active = resid > 0
                if not bool(active.any()):
                    break
                frac = (ideal - torch.floor(ideal)).float()
                frac[~active] = -1.0
                order = torch.argsort(frac, descending=True)
                gave = 0
                for kk in order.tolist():
                    if resid[kk].item() <= 0:
                        continue
                    out[kk] += 1
                    gave += 1
                    if gave >= remain:
                        break
                if gave == 0:
                    break
                remain -= gave
            return out

        bq_t = _quota_from_weights_capped(B_exec, wB_eff, birth_cap_layer)
        B_alloc = int(bq_t.sum().item())

        birth_logp = torch.zeros(1, device=dev)
        centers_list, anchors_list, new_layers_list, mu_list, rho_list = [], [], [], [], []
        latent_src_list, expr_src_list, parent_mark_local_all, parent_uid_list = [], [], [], []
        child_is_diff_list, child_diff_alpha_list, child_diff_mid_list, child_diff_tgt_layer_list, child_diff_enter_step_list = [], [], [], [], []
        B_done_by_layer = torch.zeros((self.n_layers,), device=dev, dtype=torch.long)

        if B_alloc > 0 and int(coords_surv.shape[0]) > 0 and bool(elig.any()):
            idx_elig = elig.nonzero(as_tuple=True)[0]
            hot_frac = float(BIRTH_HOT_FRAC) * (1.0 - warm) + 0.2 * warm

            for li in range(self.n_layers):
                b_need = int(bq_t[li].item())
                if b_need <= 0:
                    continue
                idx_li = idx_elig[(layers_surv[idx_elig] == li)]
                if idx_li.numel() == 0:
                    continue
                take = min(b_need, int(idx_li.numel()))
                if take <= 0:
                    continue

                pneed = gather_map_at_coords_fast(birth_drive_map[li], coords_surv[idx_li], grid_item_dev, self.H, self.W, default=0.0).float()
                logits_adj = birth_logits_surv[idx_li] + float(LAMBDA_PARENT) * torch.log1p(pneed)
                probsB = F.softmax(logits_adj / float(TAU_BIRTH), dim=0)
                parent_local, logpB = sample_no_replace(probsB, take)
                birth_logp = birth_logp + logpB
                if parent_local.numel() == 0:
                    continue

                parent_idx = idx_li[parent_local]
                g = torch.Generator(device=dev)
                g.manual_seed(int(700000 + seed_step + 1000 * li))
                perm = torch.randperm(parent_idx.numel(), generator=g, device=dev)
                parent_idx = parent_idx[perm]

                cnt = int(parent_idx.numel())
                cnt_hot = int(round(cnt * hot_frac))
                cnt_loc = cnt - cnt_hot
                idx_loc = parent_idx[:cnt_loc]
                idx_hot = parent_idx[cnt_loc:]
                parent_idx_all = torch.cat([idx_loc, idx_hot], dim=0) if (cnt_loc > 0 and cnt_hot > 0) else (idx_loc if cnt_hot == 0 else idx_hot)
                if parent_idx_all.numel() == 0:
                    continue

                parent_mark_local_all.append(parent_idx_all)
                B_done_by_layer[li] += int(parent_idx_all.numel())

                rad_eff = int(LOCAL_NEED_RADIUS)
                gam_eff = float(NEED_GAMMA) * (0.50 + 0.50 * warm)

                if cnt_loc > 0:
                    pc = coords_surv[idx_loc]
                    pl = layers_surv[idx_loc]
                    cent_loc = sample_local_centers(
                        parent_coords=pc,
                        parent_layers=pl,
                        need_map_by_layer=birth_drive_map,
                        grid_item_dev=grid_item_dev,
                        H=self.H,
                        W=self.W,
                        radius=rad_eff,
                        gamma=gam_eff,
                        seed=int(100000 + seed_step + 1000 * li),
                    )
                    cent_loc = _fix_centers_len(cent_loc, pc)
                else:
                    cent_loc = torch.empty((0, 2), device=dev, dtype=coords_surv.dtype)

                if cnt_hot > 0:
                    cent_hot = sample_centers_map(
                        birth_drive_map[li],
                        grid_item_dev,
                        self.H,
                        self.W,
                        cnt_hot,
                        gamma=gam_eff * 1.5,
                        seed=int(110000 + seed_step + 1000 * li),
                        smooth_k=3,
                    )
                    if cent_hot is None or int(cent_hot.shape[0]) != cnt_hot:
                        cent_hot = coords_surv[idx_hot].clone()
                else:
                    cent_hot = torch.empty((0, 2), device=dev, dtype=coords_surv.dtype)

                centers_all = torch.cat([cent_loc, cent_hot], dim=0) if (cnt_loc > 0 and cnt_hot > 0) else (cent_loc if cnt_hot == 0 else cent_hot)
                centers_all = _fix_centers_len(centers_all, coords_surv[parent_idx_all])

                need_at = gather_map_at_coords_fast(birth_drive_map[li], centers_all, grid_item_dev, self.H, self.W, default=0.0).float()
                bad = (need_at <= 0.0)
                if bool(bad.any()):
                    cnt_bad = int(bad.sum().item())
                    cen2 = sample_centers_map(
                        birth_drive_map[li],
                        grid_item_dev,
                        self.H,
                        self.W,
                        cnt_bad,
                        gamma=gam_eff * 1.5,
                        seed=int(120000 + seed_step + 1000 * li),
                        smooth_k=3,
                    )
                    if cen2 is not None:
                        cen2 = _fix_centers_len(cen2, centers_all[bad])
                        centers_all = centers_all.clone()
                        centers_all[bad] = cen2

                centers_list.append(centers_all)
                anchors_list.append(anchor_surv[parent_idx_all])
                new_layers_list.append(layers_surv[parent_idx_all].clone())
                mu_list.append(mu_surv[parent_idx_all])
                rho_list.append(rho_surv[parent_idx_all])
                parent_uid_list.append(uid_surv[parent_idx_all].clone())
                if latent_surv is not None:
                    latent_src_list.append(latent_surv[parent_idx_all])
                if expr_surv is not None:
                    expr_src_list.append(expr_surv[parent_idx_all])

                child_is_diff_list.append(is_diff_surv[parent_idx_all].clone())
                child_diff_alpha_list.append(diff_alpha_surv[parent_idx_all].clone())
                child_diff_mid_list.append(diff_mid_surv[parent_idx_all].clone())
                child_diff_tgt_layer_list.append(diff_tgt_layer_surv[parent_idx_all].clone())
                child_diff_enter_step_list.append(diff_enter_step_surv[parent_idx_all].clone())
        
        left = (bq_t - B_done_by_layer).clamp_min(0)
        seed_left = torch.minimum(left, base_seed_cap_layer)

        if int(seed_left.sum().item()) > 0:
            gA = torch.Generator(device=dev)
            gA.manual_seed(int(900000 + seed_step))
            for lj in range(self.n_layers):
                cnt = int(seed_left[lj].item())
                if cnt <= 0:
                    continue
                pool = (self.base_layers0 == lj).nonzero(as_tuple=True)[0]
                if pool.numel() == 0:
                    continue
                ridx = torch.randint(0, pool.numel(), (cnt,), generator=gA, device=dev)
                seed_anchor_li = pool[ridx].long()
                gam_eff = float(NEED_GAMMA) * (0.50 + 0.50 * warm)
                cen = sample_centers_map(
                    birth_drive_map[lj],
                    grid_item_dev,
                    self.H,
                    self.W,
                    cnt,
                    gamma=gam_eff * 1.5,
                    seed=int(910000 + seed_step + 1000 * lj),
                    smooth_k=3,
                )
                if cen is None or int(cen.shape[0]) != cnt:
                    cen = self.base_coords_seq[t_next][seed_anchor_li].clone()

                centers_list.append(cen)
                anchors_list.append(seed_anchor_li)
                new_layers_list.append(torch.full((cnt,), lj, dtype=torch.long, device=dev))
                mu_list.append(torch.zeros((cnt, 2), device=dev, dtype=coords.dtype))
                rho_list.append(torch.zeros((cnt, 2), device=dev, dtype=coords.dtype))
                parent_uid_list.append(torch.full((cnt,), -1, device=dev, dtype=torch.long))

                child_is_diff_list.append(torch.zeros((cnt,), dtype=torch.bool, device=dev))
                child_diff_alpha_list.append(torch.zeros((cnt,), dtype=torch.float32, device=dev))
                child_diff_mid_list.append(torch.zeros((cnt,), dtype=torch.float32, device=dev))
                child_diff_tgt_layer_list.append(torch.full((cnt,), -1, dtype=torch.long, device=dev))
                child_diff_enter_step_list.append(torch.full((cnt,), -100000, dtype=torch.int32, device=dev))
                
                if latent_surv is not None and latent_surv.numel() > 0:
                    src_pool = (layers_surv == lj).nonzero(as_tuple=True)[0]
                    if src_pool.numel() == 0:
                        latent_src_list.append(self.base_Z_seq[t_next][seed_anchor_li].to(latent_surv.dtype))
                    else:
                        ridx2 = torch.randint(0, src_pool.numel(), (cnt,), generator=gA, device=dev)
                        latent_src_list.append(latent_surv[src_pool[ridx2]])

                if expr_surv is not None and expr_surv.numel() > 0:
                    src_pool = (layers_surv == lj).nonzero(as_tuple=True)[0]
                    if src_pool.numel() > 0:
                        ridx2 = torch.randint(0, src_pool.numel(), (cnt,), generator=gA, device=dev)
                        expr_src_list.append(expr_surv[src_pool[ridx2]])

        did_birth = (len(centers_list) > 0)
        disp = (self.base_coords_seq[t_next] - self.base_coords_seq[t_now])

        if not did_birth:
            next_coords, next_layers, next_anchor = coords_surv, layers_surv, anchor_surv
            next_is_birth = torch.zeros(next_coords.shape[0], dtype=torch.bool, device=dev)
            next_latent, next_expr = latent_surv, expr_surv
            next_born_step = born_step_surv
            next_uid, next_parent_uid = uid_surv, parent_uid_surv
            next_has_div, next_last_parent = has_div_surv, last_parent_surv
            next_is_diff, next_diff_alpha, next_diff_mid, next_diff_tgt_layer = is_diff_surv, diff_alpha_surv, diff_mid_surv, diff_tgt_layer_surv
            next_diff_enter_step, next_commit_step = diff_enter_step_surv, commit_step_surv
        else:
            centers_all = torch.cat(centers_list, dim=0)
            new_anchor = torch.cat(anchors_list, dim=0)
            new_layers = torch.cat(new_layers_list, dim=0)
            mu_all = torch.cat(mu_list, dim=0)
            rho_all = torch.cat(rho_list, dim=0)
            new_parent_uid = torch.cat(parent_uid_list, dim=0) if parent_uid_list else torch.full((new_anchor.numel(),), -1, device=dev, dtype=torch.long)

            Bc, Bm, Br, Ba = int(centers_all.shape[0]), int(mu_all.shape[0]), int(rho_all.shape[0]), int(new_anchor.shape[0])
            B = min(Bc, Bm, Br, Ba)
            if Bc != B:
                centers_all = centers_all[:B]
            if Bm != B:
                mu_all = mu_all[:B]
            if Br != B:
                rho_all = rho_all[:B]
            if Ba != B:
                new_anchor = new_anchor[:B]
            if int(new_layers.shape[0]) != B:
                new_layers = new_layers[:B]
            if int(new_parent_uid.shape[0]) != B:
                new_parent_uid = new_parent_uid[:B]

            nb = int(new_anchor.numel())
            new_uid = torch.arange(self._next_uid, self._next_uid + nb, device=dev, dtype=torch.long)
            self._next_uid += nb

            new_coords, logp_place = sample_birth_locations(centers=centers_all, mu_raw=mu_all, rho_raw=rho_all, dir_vec=disp[new_anchor], dx=dx, dy=dy, seed=int(2025 + int(seed_step)))
            birth_logp = birth_logp + logp_place

            jn = torch.round((new_coords[:, 0] - x0) / dx).to(torch.long)
            in_ = torch.round((new_coords[:, 1] - y0) / dy).to(torch.long)
            inside_n = (in_ >= 0) & (in_ < int(self.H)) & (jn >= 0) & (jn < int(self.W))
            iin, jjn = in_.clamp(0, int(self.H) - 1), jn.clamp(0, int(self.W) - 1)
            good_n = inside_n & allow_shell[iin, jjn]
            bad_n_idx = (~good_n).nonzero(as_tuple=True)[0]
            if bad_n_idx.numel() > 0:
                new_bad = project_to_allowed_mask(new_coords[bad_n_idx], target_hw_surv, grid_item_dev, self.H, self.W, chunk_q=4096, max_ref=12000, seed=int(456 + seed_step))
                new_coords = new_coords.clone()
                new_coords[bad_n_idx] = new_bad

            next_latent = latent_surv
            if latent_surv is not None and len(latent_src_list) > 0:
                latent_src = torch.cat(latent_src_list, dim=0)
                if int(latent_src.shape[0]) != nb:
                    latent_src = latent_src[:nb]
                if latent_src.numel() > 0:
                    lat_std = latent_surv.std(dim=0, keepdim=True).clamp_min(1e-6) if int(latent_surv.shape[0]) > 1 else torch.ones((1, latent_src.shape[1]), device=dev, dtype=latent_src.dtype)
                    noise = torch.randn_like(latent_src) * (float(LATENT_NOISE_SCALE) * lat_std)
                    next_latent = torch.cat([latent_surv, latent_src + noise], dim=0)

            next_expr = expr_surv
            if expr_surv is not None and len(expr_src_list) > 0:
                expr_src = torch.cat(expr_src_list, dim=0)
                if int(expr_src.shape[0]) != nb:
                    expr_src = expr_src[:nb]
                if expr_src.numel() > 0:
                    next_expr = torch.cat([expr_surv, expr_src], dim=0)

            next_coords = torch.cat([coords_surv, new_coords], dim=0)
            next_layers = torch.cat([layers_surv, new_layers], dim=0)
            next_anchor = torch.cat([anchor_surv, new_anchor], dim=0)
            next_is_birth = torch.cat([torch.zeros(coords_surv.shape[0], dtype=torch.bool, device=dev), torch.ones(new_coords.shape[0], dtype=torch.bool, device=dev)], dim=0)
            next_born_step = torch.cat([born_step_surv, torch.full((new_coords.shape[0],), int(t_next), device=dev, dtype=torch.int32)], dim=0)
            next_has_div = torch.cat([has_div_surv, torch.zeros((new_coords.shape[0],), dtype=torch.bool, device=dev)], dim=0)
            next_last_parent = torch.cat([last_parent_surv, torch.full((new_coords.shape[0],), -100000, dtype=torch.int32, device=dev)], dim=0)

            if len(parent_mark_local_all) > 0:
                parent_mark_local = torch.cat(parent_mark_local_all, dim=0)
                if parent_mark_local.numel() > 0:
                    next_has_div[parent_mark_local] = True
                    next_last_parent[parent_mark_local] = int(t_now)

            next_uid = torch.cat([uid_surv, new_uid], dim=0)
            next_parent_uid = torch.cat([parent_uid_surv, new_parent_uid], dim=0)
            child_is_diff = torch.cat(child_is_diff_list, dim=0) if len(child_is_diff_list) > 0 else torch.zeros((nb,), dtype=torch.bool, device=dev)
            child_diff_alpha = torch.cat(child_diff_alpha_list, dim=0) if len(child_diff_alpha_list) > 0 else torch.zeros((nb,), dtype=torch.float32, device=dev)
            child_diff_mid = torch.cat(child_diff_mid_list, dim=0) if len(child_diff_mid_list) > 0 else torch.zeros((nb,), dtype=torch.float32, device=dev)
            child_diff_tgt_layer = torch.cat(child_diff_tgt_layer_list, dim=0) if len(child_diff_tgt_layer_list) > 0 else torch.full((nb,), -1, dtype=torch.long, device=dev)
            child_diff_enter_step = torch.cat(child_diff_enter_step_list, dim=0) if len(child_diff_enter_step_list) > 0 else torch.full((nb,), -100000, dtype=torch.int32, device=dev)

            if int(child_is_diff.shape[0]) != nb:
                child_is_diff = child_is_diff[:nb]
            if int(child_diff_alpha.shape[0]) != nb:
                child_diff_alpha = child_diff_alpha[:nb]
            if int(child_diff_mid.shape[0]) != nb:
                child_diff_mid = child_diff_mid[:nb]
            if int(child_diff_tgt_layer.shape[0]) != nb:
                child_diff_tgt_layer = child_diff_tgt_layer[:nb]
            if int(child_diff_enter_step.shape[0]) != nb:
                child_diff_enter_step = child_diff_enter_step[:nb]

            next_is_diff = torch.cat([is_diff_surv, child_is_diff], dim=0)
            next_diff_alpha = torch.cat([diff_alpha_surv, child_diff_alpha], dim=0)
            next_diff_mid = torch.cat([diff_mid_surv, child_diff_mid], dim=0)
            next_diff_tgt_layer = torch.cat([diff_tgt_layer_surv, child_diff_tgt_layer], dim=0)
            next_diff_enter_step = torch.cat([diff_enter_step_surv, child_diff_enter_step], dim=0)
            next_commit_step = torch.cat([commit_step_surv, torch.full((nb,), -1, dtype=torch.int32, device=dev)], dim=0)
        if advect_latent and (next_latent is not None) and (next_latent.numel() > 0):
            teacher_all = self.base_Z_seq[t_next][next_anchor].to(next_latent.dtype)
            legacy_mask = (~next_is_diff) & (next_born_step < 0)
            if bool(legacy_mask.any()):
                next_latent = next_latent.clone()
                next_latent[legacy_mask] = teacher_all[legacy_mask]

            born_mask = (~next_is_diff) & (next_born_step >= 0)
            if bool(born_mask.any()) and (self.tgt_latent is not None) and (self.tgt_latent.numel() > 0):
                alpha0 = float(getattr(self, "newborn_target_alpha0", 0.97))
                knn_k = int(getattr(self, "newborn_target_knn_k", 16))
                max_ref = int(getattr(self, "newborn_target_knn_max_ref", 20000))
                chunk_q = int(getattr(self, "newborn_target_knn_chunk_q", 2048))
                idx_born = born_mask.nonzero(as_tuple=True)[0]
                age = (int(t_next) - next_born_step[idx_born].to(torch.int64)).clamp_min(0).to(torch.float32)
                a_cell = torch.clamp(torch.pow(torch.tensor(alpha0, device=dev), age + 1.0), 0.0, 1.0).to(next_latent.dtype)
                next_latent = next_latent.clone()
                for li in range(self.n_layers):
                    idx_li = idx_born[(next_layers[idx_born] == li)]
                    if idx_li.numel() == 0:
                        continue
                    z_tgt = _knn_mean_target_latent(next_coords[idx_li], li, k=knn_k, chunk_q=chunk_q, max_ref=max_ref, seed=int(91000 + seed_step + 77 * li))
                    if z_tgt is None:
                        continue
                    a = a_cell[(next_layers[idx_born] == li)].view(-1, 1)
                    next_latent[idx_li] = next_latent[idx_li] * a + z_tgt.to(next_latent.dtype) * (1.0 - a)

        occ_alpha_soft = _soft_occ_by_layer_diff(next_coords, next_layers, next_is_diff, next_diff_alpha, next_diff_tgt_layer, next_commit_step, grid_item_dev, self.H, self.W, self.n_layers)
        density_map_alpha = _make_density_map(occ_alpha_soft.sum(dim=0), smooth_k=5)
        density_here_alpha = gather_map_at_coords_fast(density_map_alpha, next_coords, grid_item_dev, self.H, self.W, default=0.0).float()

        lr_alpha = None
        if self.use_lr and (self.recompute_lr is not None):
            if next_expr is not None:
                lr_alpha_raw = self.recompute_lr(next_expr, next_coords)
            elif next_latent is not None:
                lr_alpha_raw = self.recompute_lr(next_latent, next_coords)
            else:
                lr_alpha_raw = None
            if lr_alpha_raw is not None and lr_alpha_raw.numel() > 0:
                lr_alpha = (lr_alpha_raw - lr_alpha_raw.mean()) / (lr_alpha_raw.std() + 1e-6)
                lr_alpha = lr_alpha.view(-1, 1)

        diff_active = _diff_active_mask(next_is_diff, next_diff_tgt_layer, next_commit_step) & (next_diff_enter_step < int(t_next))
        if bool(diff_active.any()):
            idx_d = diff_active.nonzero(as_tuple=True)[0]

            if self.alpha_net is not None:
                raw_h = self.alpha_net(
                    next_coords[idx_d],
                    lr=None if lr_alpha is None else lr_alpha[idx_d],
                    density=torch.log1p(density_here_alpha[idx_d]).view(-1, 1),
                ).float()
                assert raw_h.ndim == 2 and raw_h.shape[1] == 2, "AlphaTransitionNet mustoutput (N,2)"
                h01 = F.softplus(raw_h[:, 0])
                h12 = F.softplus(raw_h[:, 1])
            else:
                h01 = torch.full((idx_d.numel(),), 0.10, device=dev, dtype=torch.float32)
                h12 = torch.full((idx_d.numel(),), 0.06, device=dev, dtype=torch.float32)

            age_d = (int(t_next) - next_diff_enter_step[idx_d].to(torch.int64)).clamp_min(0).float()
            gate12 = ((age_d - 1.0) / 3.0).clamp(0.0, 1.0)
            gate01 = ((age_d + 1.0) / 3.0).clamp(0.0, 1.0)
            gate01 = gate01 * gate01
            q01 = (1.0 - torch.exp(-h01)) * gate01

            q12 = (1.0 - torch.exp(-h12)) * gate12

            next_diff_mid = next_diff_mid.clone()
            next_diff_alpha = next_diff_alpha.clone()

            p_mid = next_diff_mid[idx_d].clamp(0.0, 1.0)
            p_tgt = (next_diff_alpha[idx_d] - 0.5 * p_mid).clamp(0.0, 1.0)
            p_src = (1.0 - p_mid - p_tgt).clamp(0.0, 1.0)

            flow01 = p_src * q01
            flow12 = p_mid * q12

            p_src_new = p_src - flow01
            p_mid_new = p_mid + flow01 - flow12
            p_tgt_new = p_tgt + flow12

            s = (p_src_new + p_mid_new + p_tgt_new).clamp_min(1e-12)
            p_mid_new = (p_mid_new / s).clamp(0.0, 1.0)
            p_tgt_new = (p_tgt_new / s).clamp(0.0, 1.0)

            next_diff_mid[idx_d] = p_mid_new
            next_diff_alpha[idx_d] = (0.5 * p_mid_new + p_tgt_new).clamp(0.0, 1.0)

            if (next_latent is not None) and (self.tgt_latent is not None) and (self.tgt_latent.numel() > 0):
                next_latent = next_latent.clone()
                diff_knn_k = int(getattr(self, "diff_target_knn_k", 16))
                diff_max_ref = int(getattr(self, "diff_target_knn_max_ref", 20000))
                diff_chunk_q = int(getattr(self, "diff_target_knn_chunk_q", 2048))
                for li in torch.unique(next_diff_tgt_layer[idx_d]).tolist():
                    idx_li = idx_d[(next_diff_tgt_layer[idx_d] == int(li))]
                    if idx_li.numel() == 0:
                        continue
                    z_tgt = _knn_mean_target_latent(next_coords[idx_li], int(li), k=diff_knn_k, chunk_q=diff_chunk_q, max_ref=diff_max_ref, seed=int(191000 + seed_step + 91 * int(li)))
                    if z_tgt is None:
                        continue
                    a = next_diff_alpha[idx_li].view(-1, 1).to(next_latent.dtype)
                    next_latent[idx_li] = next_latent[idx_li] * (1.0 - a) + z_tgt.to(next_latent.dtype) * a

        late_prob_all = (next_diff_alpha - 0.5 * next_diff_mid).clamp(0.0, 1.0)
        diff_age_all = (int(t_next) - next_diff_enter_step.to(torch.int64)).clamp_min(0)
        commit_mask = _diff_active_mask(next_is_diff, next_diff_tgt_layer, next_commit_step) & (late_prob_all >= 0.80) & (diff_age_all >= 2)

        if bool(commit_mask.any()):
            next_layers = next_layers.clone()
            next_layers[commit_mask] = next_diff_tgt_layer[commit_mask]
            next_is_diff = next_is_diff.clone()
            next_is_diff[commit_mask] = False
            next_diff_alpha = next_diff_alpha.clone()
            next_diff_alpha[commit_mask] = 1.0
            next_diff_mid = next_diff_mid.clone()
            next_diff_mid[commit_mask] = 0.0
            next_commit_step = next_commit_step.clone()
            next_commit_step[commit_mask] = int(t_next)
            next_diff_tgt_layer = next_diff_tgt_layer.clone()
            next_diff_tgt_layer[commit_mask] = -1
            next_diff_enter_step = next_diff_enter_step.clone()
            next_diff_enter_step[commit_mask] = -100000

        if (not self.use_lr) or (self.recompute_lr is None):
            next_lr = torch.zeros((next_coords.shape[0],), device=dev, dtype=torch.float32)
        else:
            next_lr = self.recompute_lr(next_expr, next_coords) if (next_expr is not None) else self.recompute_lr(next_latent, next_coords)

        self.state["coords"] = next_coords
        self.state["layers"] = next_layers
        self.state["anchor"] = next_anchor
        self.state["lr"] = next_lr
        self.state["is_birth"] = next_is_birth
        self.state["latent"] = next_latent
        self.state["expr_union"] = next_expr
        self.state["born_step"] = next_born_step
        self.state["has_divided"] = next_has_div
        self.state["uid"] = next_uid
        self.state["parent_uid"] = next_parent_uid
        self.state["last_parent_step"] = next_last_parent
        self.state["is_diff"] = next_is_diff
        self.state["diff_alpha"] = next_diff_alpha
        self.state["diff_mid"] = next_diff_mid
        self.state["diff_tgt_layer"] = next_diff_tgt_layer
        self.state["diff_enter_step"] = next_diff_enter_step
        self.state["commit_step"] = next_commit_step
        self.t = t_next

        if hasattr(self, "trace") and isinstance(self.trace, list):
            self.trace.append({
                "t": int(self.t),
                "uid": next_uid.detach().cpu(),
                "parent_uid": next_parent_uid.detach().cpu(),
                "anchor": next_anchor.detach().cpu(),
                "layers": next_layers.detach().cpu(),
                "coords": next_coords.detach().cpu(),
                "is_birth": next_is_birth.detach().cpu(),
                "is_diff": next_is_diff.detach().cpu(),
                "diff_tgt_layer": next_diff_tgt_layer.detach().cpu(),
                "diff_alpha": next_diff_alpha.detach().cpu(),
            })

        return (birth_logp + death_logp + diff_logp).squeeze(), ent.squeeze()

# ============================================================
# 3) Training (unchanged except env kwargs support diff tensors)
# ============================================================
def train_policy_rl(
    *, state0, target_state, shells_norm, coords_seq_torch, Z_seq_torch, base_layers0,
    grid_cache_np, grid_cache_dev, B_step_layer, D_step_layer, T, device, n_layers, H, W, recompute_lr_fn,
    policy_cls, env_cls, USE_LR=True, EPOCHS=300, LR=1e-4,
    W_ENT=0.1, EMA_BETA=0.9, ADVECT_LATENT=True, shell_need_cap=3, hidden_dim=128,
    best_ckpt_path=None, save_best=True, empty_cache_every=1, return_history=True,
    W_XY_TGT=2.0, W_Z_TGT=0.05, W_OCC=0.1, W_OCC_IOU=0.1,
    TAU_BIRTH=1.0, TAU_DEATH=1.0, TAU_DIFF=1.0, LATENT_NOISE_SCALE=0.01,
    OLD_MASS_SCALE=0.1,
    diff_tgt_idx: Optional[torch.Tensor] = None, diff_tgt_w: Optional[torch.Tensor] = None,
):
    device = torch.device(device)

    if state0.get("latent", None) is not None:
        latent_dim = int(state0["latent"].shape[1])
    elif Z_seq_torch is not None and len(Z_seq_torch) > 0 and Z_seq_torch[0] is not None:
        latent_dim = int(Z_seq_torch[0].shape[1])
    else:
        raise ValueError("cannotdetermine latent_dim:state0['latent']  Z_seq_torch ")

    kw = dict(
        n_layers=int(n_layers),
        use_lr=bool(USE_LR),
        hidden_dim=int(hidden_dim),
        learn_eta0=False,
        eta0_init=0.0,
    )
    policy_net = policy_cls(**_filter_kwargs(policy_cls, kw)).to(device)
    alpha_net = AlphaTransitionNet( use_lr=bool(USE_LR), hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(alpha_net.parameters()),
        lr=float(LR)
    )

    best_reward = -1e18
    reward_ema = None
    best_ckpt_path = str(best_ckpt_path) if best_ckpt_path is not None else None
    history = {"reward": [], "adv": [], "tgt": [], "occ": [], "iou": [], "ent": [], "logp": []} if return_history else None

    wfr_xy = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9, backend="multiscale")
    wfr_z = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.9, backend="online")

    tgt_is_new = target_state.get("is_new", None)
    if tgt_is_new is not None:
        tgt_is_new = tgt_is_new.to(device).bool()

    xt = target_state["coords"].to(device)
    lt = target_state["layers"].to(device)
    zt = target_state.get("latent", None)
    zt = None if zt is None else zt.to(device)
    x0 = state0["coords"].to(device)
    grid_item_T = grid_cache_dev[int(T)]

    pbar = trange(1, int(EPOCHS) + 1, desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        policy_net.train()
        alpha_net.train()

        env_f = env_cls(
            state0=state0,
            target_state=target_state,
            device=device,
            alpha_net=alpha_net,
            base_coords_seq=coords_seq_torch,
            base_Z_seq=Z_seq_torch,
            base_layers0=base_layers0,
            grid_cache_np=grid_cache_np,
            grid_cache_dev=grid_cache_dev,
            shells_norm=shells_norm,
            T=int(T),
            n_layers=int(n_layers),
            H=int(H),
            W=int(W),
            recompute_lr_fn=recompute_lr_fn,
            t0=0,
            shell_need_cap=int(shell_need_cap),
            use_lr=bool(USE_LR),
            diff_tgt_idx=diff_tgt_idx,
            diff_tgt_w=diff_tgt_w,
        )

        logp_f_list, ent_f_list = [], []
        for t in range(int(T)):
            logp, ent = env_f.step(
                policy_net,
                birth_quota_layer=B_step_layer[t],
                death_quota_layer=D_step_layer[t],
                dir_step=+1,
                advect_latent=bool(ADVECT_LATENT),
                seed_step=int(epoch * 1000 + t),
                TAU_BIRTH=float(TAU_BIRTH),
                TAU_DEATH=float(TAU_DEATH),
                TAU_DIFF=float(TAU_DIFF),
                LATENT_NOISE_SCALE=float(LATENT_NOISE_SCALE),
            )
            logp_f_list.append(logp)
            ent_f_list.append(ent)

        with torch.no_grad():
            xs = env_f.state["coords"]
            ls = env_f.state["layers"]
            zs = env_f.state.get("latent", None)
            born_step = env_f.state.get("born_step", None)
            new_s = (born_step >= 0) if born_step is not None else env_f.state.get(
                "is_birth", torch.zeros(xs.shape[0], device=device, dtype=torch.bool)
            ).bool()

            if tgt_is_new is not None:
                new_t = tgt_is_new
            else:
                N_new_expected = max(int(xt.shape[0] - x0.shape[0]), 0)
                if N_new_expected <= 0:
                    new_t = torch.zeros((xt.shape[0],), device=device, dtype=torch.bool)
                else:
                    mind = torch.full((xt.shape[0],), 1e9, device=device)
                    chunk = 2048
                    for s in range(0, xt.shape[0], chunk):
                        D = torch.cdist(xt[s:s + chunk], x0)
                        mind[s:s + chunk] = D.min(dim=1).values
                    _, idx = torch.topk(mind, k=min(N_new_expected, xt.shape[0]), largest=True)
                    new_t = torch.zeros((xt.shape[0],), device=device, dtype=torch.bool)
                    new_t[idx] = True

            ms = torch.ones((xs.shape[0],), device=device, dtype=torch.float32)
            mt = torch.ones((xt.shape[0],), device=device, dtype=torch.float32)
            ms[~new_s] = float(OLD_MASS_SCALE)
            mt[~new_t] = float(OLD_MASS_SCALE)
            ms = ms / ms.sum().clamp_min(1e-12)
            mt = mt / mt.sum().clamp_min(1e-12)

            tgt_xy = wfr_xy(ms, xs, mt, xt).view(1)
            tgt_z = wfr_z(ms, zs, mt, zt).view(1) if (zs is not None and zt is not None) else torch.zeros(1, device=device)
            tgt_loss = float(W_XY_TGT) * tgt_xy + float(W_Z_TGT) * tgt_z

            occ_cur_cap, _ = get_occ_and_crowd(xs, ls, grid_item_T, int(H), int(W), int(n_layers), cap=int(shell_need_cap))
            occ_tgt_raw = grid_occupancy_by_layer(xt, lt, grid_item_T, int(H), int(W), int(n_layers), normalize=False)
            occ_tgt_cap = occ_tgt_raw.clamp_max(float(shell_need_cap))
            occ_l1 = (occ_cur_cap - occ_tgt_cap).abs().sum() / occ_tgt_cap.sum().clamp_min(1.0)

            iou_loss = torch.zeros(1, device=device)
            if float(W_OCC_IOU) > 0:
                cur_bin = (occ_cur_cap.sum(dim=0) > 0).float()
                tgt_bin = (occ_tgt_cap.sum(dim=0) > 0).float()
                inter = (cur_bin * tgt_bin).sum()
                uni = ((cur_bin + tgt_bin) > 0).float().sum().clamp_min(1.0)
                iou = inter / uni
                iou_loss = (1.0 - iou).view(1)

            occ_loss = occ_l1.view(1) + float(W_OCC_IOU) * iou_loss
            final_loss = tgt_loss + float(W_OCC) * occ_loss
            reward = -final_loss

        r_det = reward.detach()
        reward_ema = r_det if reward_ema is None else float(EMA_BETA) * reward_ema + (1.0 - float(EMA_BETA)) * r_det
        adv = (reward - reward_ema).detach()

        logp_total = torch.stack(logp_f_list).sum()
        ent_total = torch.stack(ent_f_list).sum()
        loss = -logp_total * adv - float(W_ENT) * ent_total

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            r_val = float(reward.item())
            if r_val > best_reward:
                best_reward = r_val
                if save_best and (best_ckpt_path is not None):
                    p_args = {
                        "n_layers": int(n_layers),
                        "use_lr": bool(USE_LR),
                        "hidden_dim": int(hidden_dim),
                        "learn_eta0": False,
                        "eta0_init": 0.0,
                    }
                    a_args = {
                        "use_lr": bool(USE_LR),
                        "hidden_dim": 128,
                    }
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "model_state": policy_net.state_dict(),
                            "alpha_state": alpha_net.state_dict(),
                            "best_reward": float(best_reward),
                            "grid_size": (int(H), int(W)),
                            "advect_latent": bool(ADVECT_LATENT),
                            "T": int(T),
                            "use_lr": bool(USE_LR),
                            "policy_args": p_args,
                            "alpha_args": a_args,
                        },
                        best_ckpt_path,
                    )

        if return_history:
            history["reward"].append(float(reward.item()))
            history["adv"].append(float(adv.item()))
            history["tgt"].append(float(tgt_loss.item()))
            history["occ"].append(float(occ_l1.item()))
            history["iou"].append(float((1.0 - iou_loss).item()) if float(W_OCC_IOU) > 0 else float("nan"))
            history["ent"].append(float(ent_total.item()))
            history["logp"].append(float(logp_total.item()))

        pbar.set_postfix({
            "rew": f"{float(reward.item()):.4f}",
            "best": f"{best_reward:.4f}",
            "tgt": f"{float(tgt_loss.item()):.4f}",
            "occ": f"{float(occ_l1.item()):.4f}",
        })

        del env_f, logp_f_list, ent_f_list, loss, reward
        if empty_cache_every and empty_cache_every > 0 and (epoch % int(empty_cache_every) == 0):
            torch.cuda.empty_cache()
            gc.collect()

    out = {
        "policy": policy_net,
        "alpha_net": alpha_net,
        "best_reward": float(best_reward),
        "best_ckpt_path": best_ckpt_path,
    }
    if return_history:
        out["history"] = history
    return out

# -------------------- orchestration --------------------
@dataclass
class GlobalCtx:
    device: torch.device
    adata_all: sc.AnnData
    lr_pairs: pd.DataFrame
    xy_mu: np.ndarray
    xy_s: np.ndarray
    layers_list: List[str]
    layer_to_idx: Dict[str, int]
    latent_cache: dict
    layer_col: str

@dataclass
class StageCfg:
    src: str
    tgt: str
    out_npz_path: str
    bound_dir: str
    stat_json: Optional[str] = None
    ckpt_path_dec: Optional[str] = None
    scanvi_dir: Optional[str] = None
    model_type: str = "scanvi"
    latent_fallback_obsm: str = "X_scVI"
    latent_fallback_layer: Optional[str] = None
    z_csv_offset: int = 0
    layer_col: Optional[str] = None
    use_latent: bool = True
    use_lr: bool = True
    lr_source: str = "decoder"
    counts_layer: str = "counts"
    # NEW: diff csv
    diff_csv: Optional[str] = None

@dataclass
class StagePack:
    cfg: StageCfg
    n_layers: int
    T: int
    H: int
    W: int
    shells_norm: List[np.ndarray]
    coords_seq_torch: List[torch.Tensor]
    Z_seq_torch: List[torch.Tensor]
    base_layers0: torch.Tensor
    grid_cache_np: list
    grid_cache_dev: list
    B_step_layer: np.ndarray
    D_step_layer: np.ndarray
    state0: Dict[str, torch.Tensor]
    target_state: Dict[str, torch.Tensor]
    recompute_lr_fn: Any
    shell_need_cap: int = 3
    diff_tgt_idx: Optional[torch.Tensor] = None
    diff_tgt_w: Optional[torch.Tensor] = None

def build_global_ctx(*, adata_path: str, lr_pairs_path: str, ckpt_3dslice: str, device: torch.device, layer_col: str) -> GlobalCtx:
    ckpt = torch.load(ckpt_3dslice, map_location="cpu")
    adata_all = sc.read_h5ad(adata_path)
    lr_pairs = pd.read_csv(lr_pairs_path)
    xy_mu, xy_s = get_xy_norm_from_ckpt(ckpt)
    layers_all = adata_all.obs[layer_col].astype(str).values
    layers_list = sorted(pd.unique(layers_all).tolist())
    layer_to_idx = {name: i for i, name in enumerate(layers_list)}
    return GlobalCtx(device=device, adata_all=adata_all, lr_pairs=lr_pairs, xy_mu=xy_mu, xy_s=xy_s, layers_list=layers_list, layer_to_idx=layer_to_idx, latent_cache={}, layer_col=str(layer_col))

def get_slice(ctx: GlobalCtx, sample_id: str, sample_key: str = "sample") -> sc.AnnData:
    m = ctx.adata_all.obs[sample_key].astype(str) == str(sample_id)
    return ctx.adata_all[m].copy()

def get_layer_idx(ctx: GlobalCtx, ad: sc.AnnData, layer_col: str) -> np.ndarray:
    lab = ad.obs[layer_col].astype(str).to_numpy()
    return np.array([ctx.layer_to_idx.get(s, 0) for s in lab], dtype=np.int64)

def prepare_one_stage(
    ctx, cfg: StageCfg, *, layer_col: str, sample_key: str = "sample",
    x_key: str = "cx_aligned", y_key: str = "cy_aligned",
    GRID_MARGIN: float = 0.02, GRID_HW_MAX: int = 512, GRID_BASE: int = 4,
    DEATH_FRAC: Optional[float] = 0.1, TURNOVER_FRAC: Optional[float] = 0.0,
    SHELL_NEED_CAP: int = 2, name_col: Optional[str] = None,
    ad_src_override=None, ad_tgt_override=None,
    coords_seq_override=None, Z_seq_override=None, mass_frames_override=None,
    **kwargs
):
    device = ctx.device

    ad_src = ad_src_override.copy() if ad_src_override is not None else get_slice(ctx, f"{cfg.src}", sample_key=sample_key)
    ad_tgt = ad_tgt_override.copy() if ad_tgt_override is not None else get_slice(ctx, f"{cfg.tgt}", sample_key=sample_key)

    xysrc_norm = normalize_xy_from_obs(ad_src, x_key=x_key, y_key=y_key, xy_mu=ctx.xy_mu, xy_s=ctx.xy_s)
    xytgt_norm = normalize_xy_from_obs(ad_tgt, x_key=x_key, y_key=y_key, xy_mu=ctx.xy_mu, xy_s=ctx.xy_s)

    layer_idx_src = get_layer_idx(ctx, ad_src, layer_col=layer_col)
    layer_idx_tgt = get_layer_idx(ctx, ad_tgt, layer_col=layer_col)
    n_layers = len(ctx.layers_list)
    base_layers0 = torch.from_numpy(layer_idx_src).to(device=device, dtype=torch.long)

    if coords_seq_override is None or mass_frames_override is None:
        if cfg.out_npz_path is None:
            raise ValueError("cfg.out_npz_path is empty,missing coords_seq_override/mass_frames_override")
        out_npz = np.load(cfg.out_npz_path, allow_pickle=True)
        coords_frames = [np.asarray(c, dtype=np.float32) for c in out_npz["coords_frames"]]
        mass_frames = [np.asarray(m, dtype=np.float32) for m in out_npz["mass_frames"]]
        Z_frames_in_npz = [np.asarray(z, dtype=np.float32) for z in out_npz["Z_frames"]] if "Z_frames" in out_npz.files else None
    else:
        coords_frames = [c.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(c) else np.asarray(c, dtype=np.float32) for c in coords_seq_override]
        mass_frames = [m.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(m) else np.asarray(m, dtype=np.float32) for m in mass_frames_override]
        Z_frames_in_npz = None if Z_seq_override is None else [z.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(z) else np.asarray(z, dtype=np.float32) for z in Z_seq_override]

    def _pad_or_repeat_Z(Z0: torch.Tensor, n: int) -> torch.Tensor:
        n0 = int(Z0.shape[0]); D = int(Z0.shape[1])
        if n == n0: return Z0.clone()
        if n < n0: return Z0[:n].clone()
        idx = (torch.arange(n - n0, device=Z0.device) % n0).long()
        return torch.cat([Z0, Z0[idx]], dim=0).reshape(n, D)

    coords_seq_torch = [torch.as_tensor(c, device=device, dtype=torch.float32) for c in coords_frames]
    Tp1 = len(coords_frames); T = Tp1 - 1

    target_state = {
        "coords": torch.from_numpy(xytgt_norm).to(device=device, dtype=torch.float32),
        "layers": torch.from_numpy(layer_idx_tgt).to(device=device, dtype=torch.long),
    }

    if bool(cfg.use_latent):
        Zt, ctx.latent_cache = get_latent_tensor(
            ad_tgt, device=device, model_dir=cfg.scanvi_dir, model_type=cfg.model_type,
            fallback_obsm=cfg.latent_fallback_obsm, fallback_layer=cfg.latent_fallback_layer,
            cache=ctx.latent_cache, load_ref_adata=getattr(ctx, "adata", None)
        )
    else:
        Zt = torch.zeros((ad_tgt.n_obs, 1), device=device)
    target_state["latent"] = Zt

    if bool(cfg.use_latent):
        if Z_seq_override is not None:
            Z_seq_torch = [torch.as_tensor(z, device=device, dtype=torch.float32) for z in Z_frames_in_npz]
        elif (Z_frames_in_npz is not None) and (len(Z_frames_in_npz) == Tp1):
            Z_seq_torch = [torch.as_tensor(z, device=device, dtype=torch.float32) for z in Z_frames_in_npz]
        else:
            Z0_src, ctx.latent_cache = get_latent_tensor(
                ad_src, device=device, model_dir=cfg.scanvi_dir, model_type=cfg.model_type,
                fallback_obsm=cfg.latent_fallback_obsm, fallback_layer=cfg.latent_fallback_layer,
                cache=ctx.latent_cache, load_ref_adata=getattr(ctx, "adata", None)
            )
            Z_seq_torch = [_pad_or_repeat_Z(Z0_src, int(c.shape[0])) for c in coords_seq_torch]
    else:
        Z_seq_torch = [torch.zeros((c.shape[0], 1), device=device) for c in coords_seq_torch]

    shells_norm = load_shells_from_dir(cfg.bound_dir, include_loops=False)
    H, W, _ = auto_choose_grid_size(xysrc_norm, shells_norm[0], pts_per_cell=0.3, base=int(GRID_BASE), max_hw=int(GRID_HW_MAX))
    print(f"[Grid] H={H}, W={W}, H×W={H*W}")

    grid_cache_np = build_shell_grid_cache(shells_norm, H=int(H), W=int(W), margin=float(GRID_MARGIN))
    grid_cache_dev = []
    for (x0, y0, dx, dy, in_shell_np) in grid_cache_np:
        grid_cache_dev.append((
            torch.tensor(x0, device=device, dtype=torch.float32),
            torch.tensor(y0, device=device, dtype=torch.float32),
            torch.tensor(dx, device=device, dtype=torch.float32),
            torch.tensor(dy, device=device, dtype=torch.float32),
            torch.from_numpy(in_shell_np).to(device=device),
        ))

    coords_src_t = torch.from_numpy(xysrc_norm).to(device=device, dtype=torch.float32)
    coords_tgt_t = torch.from_numpy(xytgt_norm).to(device=device, dtype=torch.float32)
    cap_auto, cap_info = auto_shell_need_cap_for_stage(coords_src_t, coords_tgt_t, grid_cache_dev, H=int(H), W=int(W), T=int(T), q=0.90, min_cap=1, max_cap=8)
    print("[auto cap]", cap_auto, cap_info)
    SHELL_NEED_CAP = int(cap_auto)

    Ms = np.array([np.asarray(m, np.float32).reshape(-1).sum() for m in mass_frames], dtype=np.float32)
    N_tilde = compute_N_tilde_from_mass(Ms, int(ad_src.n_obs), int(ad_tgt.n_obs))

    counts0 = np.bincount(layer_idx_src, minlength=n_layers)
    countsT = np.bincount(layer_idx_tgt, minlength=n_layers)
    _, B_step_layer, D_step_layer = build_layer_plan_and_quotas(
        N_tilde=N_tilde, counts0=counts0, countsT=countsT,
        death_frac=DEATH_FRAC, turnover_frac=TURNOVER_FRAC
    )

    lr_source = str(cfg.lr_source).lower()
    use_lr = bool(cfg.use_lr) and (lr_source != "none")
    state0 = {
        "coords": coords_seq_torch[0].clone(),
        "layers": base_layers0.clone(),
        "anchor": torch.arange(coords_seq_torch[0].shape[0], device=device, dtype=torch.long),
        "latent": Z_seq_torch[0].clone(),
        "expr_union": None,
        "lr": torch.zeros(coords_seq_torch[0].shape[0], device=device),
        "tid": None,
    }
    recompute_lr_fn = recompute_lr_factory_none(device)

    if use_lr and lr_source == "decoder":
        dp = build_decoder_pack(
            device=device, stat_json=str(cfg.stat_json), ckpt_path_dec=str(cfg.ckpt_path_dec),
            lr_pairs_df=ctx.lr_pairs, latent_dim=int(state0["latent"].shape[1])
        )
        recompute_lr_fn = recompute_lr_factory(dp)
        with torch.no_grad():
            state0["lr"] = recompute_lr_fn(state0["latent"], state0["coords"])
    elif use_lr and lr_source == "counts":
        genes_union = genes_union_from_lr_pairs(ctx.lr_pairs, ad_src.var_names)
        if len(genes_union) > 0:
            state0["expr_union"] = torch.from_numpy(dense_from_layer_by_genes(ad_src, str(cfg.counts_layer), genes_union)).to(device=device, dtype=torch.float32)
            target_state["expr_union"] = torch.from_numpy(dense_from_layer_by_genes(ad_tgt, str(cfg.counts_layer), genes_union)).to(device=device, dtype=torch.float32)
            recompute_lr_fn = recompute_lr_factory_counts(
                device=device, lr_pairs_df=ctx.lr_pairs, genes_union=genes_union,
                compute_lr_potential_gpu=compute_lr_potential_gpu
            )
            with torch.no_grad():
                state0["lr"] = recompute_lr_fn(state0["expr_union"], state0["coords"])
        else:
            recompute_lr_fn = recompute_lr_factory_none(device)

    diff_tgt_idx, diff_tgt_w = None, None
    if getattr(cfg, "diff_csv", None) is not None and str(cfg.diff_csv).strip() != "" and Path(str(cfg.diff_csv)).exists():
        diff_tgt_idx, diff_tgt_w = load_diff_csv_to_tensors(str(cfg.diff_csv), ctx.layer_to_idx, n_layers=n_layers, device=device)
        print("[diff] loaded", cfg.diff_csv, "Kmax=", int(diff_tgt_idx.shape[1]))

    return StagePack(
        cfg=cfg, n_layers=n_layers, T=T, H=int(H), W=int(W),
        shells_norm=shells_norm, coords_seq_torch=coords_seq_torch, Z_seq_torch=Z_seq_torch,
        base_layers0=base_layers0, grid_cache_np=grid_cache_np, grid_cache_dev=grid_cache_dev,
        B_step_layer=B_step_layer, D_step_layer=D_step_layer,
        state0=state0, target_state=target_state, recompute_lr_fn=recompute_lr_fn,
        shell_need_cap=int(SHELL_NEED_CAP), diff_tgt_idx=diff_tgt_idx, diff_tgt_w=diff_tgt_w
    )

def run_multi_stages(*, ctx, W_XY_TGT, W_Z_TGT, sample_key, stages: List[StageCfg], best_ckpt_dir: str, train_kwargs: Optional[dict] = None) -> List[Dict[str, Any]]:
    train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
    Path(best_ckpt_dir).mkdir(parents=True, exist_ok=True)
    outs = []
    for cfg in stages:
        lc = (cfg.layer_col if getattr(cfg, "layer_col", None) else getattr(ctx, "layer_col", "annotation"))
        pack = prepare_one_stage(ctx, cfg, sample_key=sample_key, layer_col=lc)
        best_ckpt_path = str(Path(best_ckpt_dir) / f"policy_{cfg.src}_to_{cfg.tgt}.pt")
        out = train_policy_rl(
            state0=pack.state0, target_state=pack.target_state, shells_norm=pack.shells_norm,
            coords_seq_torch=pack.coords_seq_torch, Z_seq_torch=pack.Z_seq_torch, base_layers0=pack.base_layers0,
            grid_cache_np=pack.grid_cache_np, grid_cache_dev=pack.grid_cache_dev,
            B_step_layer=pack.B_step_layer, D_step_layer=pack.D_step_layer, T=pack.T,
            device=ctx.device, n_layers=pack.n_layers, H=pack.H, W=pack.W, recompute_lr_fn=pack.recompute_lr_fn,
            policy_cls=GrowthPolicyNet, env_cls=SimulationEnv, USE_LR=bool(cfg.use_lr) and (str(cfg.lr_source).lower() != "none"),
            best_ckpt_path=best_ckpt_path, W_XY_TGT=W_XY_TGT, W_Z_TGT=W_Z_TGT, save_best=True, shell_need_cap=pack.shell_need_cap,
            diff_tgt_idx=pack.diff_tgt_idx, diff_tgt_w=pack.diff_tgt_w, **train_kwargs
        )
        outs.append({"stage": cfg, "best_ckpt_path": best_ckpt_path, "train_out": out})
    return outs

def _infer_policy_args_smart(in_features: int, n_layers: int, hidden_dim: int):
    for use_lr in (False, True):
        for use_t in (False, True):
            d = 3 + (1 if use_lr else 0) + (1 if use_t else 0)
            if d == in_features:
                return {"n_layers": n_layers, "use_lr": use_lr, "use_t": use_t, "hidden_dim": hidden_dim, "learn_eta0": True, "eta0_init": 0.2}
    raise RuntimeError(f"cannot {in_features}  GrowthPolicyNet  (new-mode).")

def _load_policy_and_alpha_net(s2, ckpt_path, *, n_layers, device, latent_dim=None):
    ckpt = torch.load(ckpt_path, map_location=device)

    sd = ckpt.get("model_state", None) or ckpt.get("state_dict", None) or ckpt

    # ---- load policy ----
    policy_args = ckpt.get("policy_args", None)
    if policy_args is not None:
        args = dict(policy_args)
        args["n_layers"] = int(n_layers)
    else:
        if "net.0.weight" in sd:
            w = sd["net.0.weight"]
            in_features = int(w.shape[1])
            hidden_dim = int(w.shape[0])
        else:
            keys = list(sd.keys())
            w = sd[keys[0]]
            in_features = int(w.shape[1])
            hidden_dim = int(w.shape[0])
        args = _infer_policy_args_smart(in_features, n_layers, hidden_dim)

    PolicyCls = getattr(s2, "PolicyNet", None)
    if PolicyCls is None:
        PolicyCls = getattr(s2, "GrowthPolicyNet")

    policy_net = PolicyCls(**_filter_kwargs(PolicyCls, args)).to(device)
    policy_net.load_state_dict(sd, strict=False)
    policy_net.eval()

    # ---- load alpha ----
    alpha_args = ckpt.get("alpha_args", None)
    if alpha_args is not None:
        alpha_args = dict(alpha_args)
        alpha_args.pop("latent_dim", None)   
    else:
        alpha_args = {
            "use_lr": bool(ckpt.get("use_lr", True)),
            "hidden_dim": 128,
        }

    alpha_net = AlphaTransitionNet(**_filter_kwargs(AlphaTransitionNet, alpha_args)).to(device)

    if "alpha_state" in ckpt:
        alpha_net.load_state_dict(ckpt["alpha_state"], strict=False)
    else:
        raise KeyError("ckpt missing alpha_state,cannot alpha_net")

    alpha_net.eval()
    return policy_net, alpha_net

@torch.no_grad()
def rollout_policy_one_stage(
    s2, ctx, cfg, ckpt_path, *,
    seed=0, sample_key='sample',
    ADVECT_LATENT=True,
    TAU_BIRTH=1.0, TAU_DEATH=1.0, TAU_DIFF=1.0,
    LATENT_NOISE_SCALE=0.02,
    AUTO_PRINT: bool = True
):
    lc = (cfg.layer_col if getattr(cfg, "layer_col", None) else getattr(ctx, "layer_col", "annotation"))
    pack = s2.prepare_one_stage(ctx, cfg, layer_col=lc, sample_key=sample_key, AUTO_PRINT=AUTO_PRINT)
    device = ctx.device

    if pack.state0.get("latent", None) is not None:
        latent_dim = int(pack.state0["latent"].shape[1])
    elif pack.Z_seq_torch is not None and len(pack.Z_seq_torch) > 0 and pack.Z_seq_torch[0] is not None:
        latent_dim = int(pack.Z_seq_torch[0].shape[1])
    else:
        raise ValueError("cannotdetermine latent_dim")

    policy_net, alpha_net = _load_policy_and_alpha_net(
        s2,
        ckpt_path,
        n_layers=pack.n_layers,
        device=device,
        latent_dim=latent_dim,
    )

    EnvCls = s2.SimulationEnv
    env_kw = dict(
        state0=pack.state0,
        target_state=pack.target_state,
        base_coords_seq=pack.coords_seq_torch,
        base_Z_seq=pack.Z_seq_torch,
        base_layers0=pack.base_layers0,
        grid_cache_np=pack.grid_cache_np,
        grid_cache_dev=pack.grid_cache_dev,
        shells_norm=pack.shells_norm,
        T=pack.T,
        device=device,
        n_layers=pack.n_layers,
        H=pack.H,
        W=pack.W,
        recompute_lr_fn=pack.recompute_lr_fn,
        shell_need_cap=getattr(pack, "shell_need_cap", 2),
        diff_tgt_idx=pack.diff_tgt_idx,
        diff_tgt_w=pack.diff_tgt_w,
        alpha_net=alpha_net,
    )
    env = EnvCls(**_filter_kwargs(EnvCls, env_kw))

    if "uid" not in env.state or env.state.get("uid", None) is None:
        N0 = int(env.state["coords"].shape[0])
        env.state["uid"] = torch.arange(N0, device=device, dtype=torch.long)
        env.state["parent_uid"] = torch.full((N0,), -1, device=device, dtype=torch.long)
        env._next_uid = int(N0)

    N0 = int(env.state["coords"].shape[0])

    def _ensure_state_vec(key, default):
        x = env.state.get(key, None)
        if x is None or int(x.numel()) != N0:
            env.state[key] = default

    _ensure_state_vec("is_diff", torch.zeros((N0,), device=device, dtype=torch.bool))
    _ensure_state_vec("diff_alpha", torch.zeros((N0,), device=device, dtype=torch.float32))
    _ensure_state_vec("diff_alpha_post", torch.zeros((N0,), device=device, dtype=torch.float32))
    _ensure_state_vec("diff_tgt_layer", torch.full((N0,), -1, device=device, dtype=torch.long))
    _ensure_state_vec("diff_enter_step", torch.full((N0,), -100000, device=device, dtype=torch.int32))
    _ensure_state_vec("commit_step", torch.full((N0,), -1, device=device, dtype=torch.int32))
    _ensure_state_vec("is_birth", torch.zeros((N0,), device=device, dtype=torch.bool))
    _ensure_state_vec("born_step", torch.full((N0,), -1, device=device, dtype=torch.int32))

    def _np(x, dt=None):
        if x is None:
            return None
        y = x.detach().cpu().numpy()
        return y.astype(dt) if dt is not None else y

    def _get_state(e):
        st = e.state
        N = int(st["coords"].shape[0])
        dev0 = st["coords"].device

        def _get_valid(name, default):
            x = st.get(name, None)
            if x is None or int(x.numel()) != N:
                return default
            return x

        c = st["coords"]
        l = st["layers"]
        z = st.get("latent", None)
        ib = _get_valid("is_birth", torch.zeros((N,), device=dev0, dtype=torch.bool))
        a = st.get("anchor", None)
        lr = st.get("lr", None)
        uid = _get_valid("uid", torch.arange(N, device=dev0, dtype=torch.long))
        puid = _get_valid("parent_uid", torch.full((N,), -1, device=dev0, dtype=torch.long))
        bs = _get_valid("born_step", torch.full((N,), -1, device=dev0, dtype=torch.int32))
        ev = _get_valid("event", torch.zeros((N,), device=dev0, dtype=torch.int8))

        is_diff = _get_valid("is_diff", torch.zeros((N,), device=dev0, dtype=torch.bool))
        diff_alpha = _get_valid("diff_alpha", torch.zeros((N,), device=dev0, dtype=torch.float32))
        diff_alpha_post = _get_valid("diff_alpha_post", torch.zeros((N,), device=dev0, dtype=torch.float32))
        diff_tgt_layer = _get_valid("diff_tgt_layer", torch.full((N,), -1, device=dev0, dtype=torch.long))
        diff_enter_step = _get_valid("diff_enter_step", torch.full((N,), -100000, device=dev0, dtype=torch.int32))
        commit_step = _get_valid("commit_step", torch.full((N,), -1, device=dev0, dtype=torch.int32))

        return c, l, z, ib, a, lr, uid, puid, bs, ev, is_diff, diff_alpha, diff_alpha_post, diff_tgt_layer, diff_enter_step, commit_step

    coords_list, layers_list, latent_list = [], [], []
    is_birth_list, anchor_list, lr_list = [], [], []
    uid_list, parent_uid_list, born_step_list, event_list = [], [], [], []
    is_diff_list, diff_alpha_list, diff_alpha_post_list = [], [], []
    diff_tgt_layer_list, diff_enter_step_list, commit_step_list = [], [], []

    c0, l0, z0, ib0, a0, lr0, uid0, puid0, bs0, ev0, id0, da0, dap0, dt0, de0, cs0 = _get_state(env)

    coords_list.append(_np(c0, np.float32))
    layers_list.append(_np(l0, np.int64))
    latent_list.append(_np(z0, np.float32))
    is_birth_list.append(_np(ib0, np.bool_))
    anchor_list.append(_np(a0, np.int64))
    lr_list.append(_np(lr0, np.float32))
    uid_list.append(_np(uid0, np.int64))
    parent_uid_list.append(_np(puid0, np.int64))
    born_step_list.append(_np(bs0, np.int32))
    event_list.append(_np(ev0, np.int8))
    is_diff_list.append(_np(id0, np.bool_))
    diff_alpha_list.append(_np(da0, np.float32))
    diff_alpha_post_list.append(_np(dap0, np.float32))
    diff_tgt_layer_list.append(_np(dt0, np.int64))
    diff_enter_step_list.append(_np(de0, np.int32))
    commit_step_list.append(_np(cs0, np.int32))

    with torch.no_grad():
        for t in range(pack.T):
            env.step(
                policy_net,
                birth_quota_layer=pack.B_step_layer[t],
                death_quota_layer=pack.D_step_layer[t],
                dir_step=+1,
                advect_latent=ADVECT_LATENT,
                seed_step=seed * 1000 + t,
                TAU_BIRTH=TAU_BIRTH,
                TAU_DEATH=TAU_DEATH,
                TAU_DIFF=TAU_DIFF,
                LATENT_NOISE_SCALE=LATENT_NOISE_SCALE,
            )

            c, l, z, ib, a, lr, uid, puid, bs, ev, isd, da, dap, dtl, des, cs = _get_state(env)

            coords_list.append(_np(c, np.float32))
            layers_list.append(_np(l, np.int64))
            latent_list.append(_np(z, np.float32))
            is_birth_list.append(_np(ib, np.bool_))
            anchor_list.append(_np(a, np.int64))
            lr_list.append(_np(lr, np.float32))
            uid_list.append(_np(uid, np.int64))
            parent_uid_list.append(_np(puid, np.int64))
            born_step_list.append(_np(bs, np.int32))
            event_list.append(_np(ev, np.int8))
            is_diff_list.append(_np(isd, np.bool_))
            diff_alpha_list.append(_np(da, np.float32))
            diff_alpha_post_list.append(_np(dap, np.float32))
            diff_tgt_layer_list.append(_np(dtl, np.int64))
            diff_enter_step_list.append(_np(des, np.int32))
            commit_step_list.append(_np(cs, np.int32))

    t_arr = np.linspace(0.0, 1.0, len(coords_list), dtype=np.float32)

    return dict(
        coords=coords_list,
        layers=layers_list,
        latent=latent_list,
        lr=lr_list,
        anchor=anchor_list,
        is_birth=is_birth_list,
        born_step=born_step_list,
        uid=uid_list,
        parent_uid=parent_uid_list,
        event=event_list,
        is_diff=is_diff_list,
        diff_alpha=diff_alpha_list,
        diff_alpha_post=diff_alpha_post_list,
        diff_tgt_layer=diff_tgt_layer_list,
        diff_enter_step=diff_enter_step_list,
        commit_step=commit_step_list,
        t=t_arr,
        T=pack.T,
        n_layers=pack.n_layers,
    )

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[device]", device)