import gc
import os
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import ot
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

def set_seed(seed: int = 2025) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _to_np(x, dtype=None):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        arr = x
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr

def save_guide_pack_npz(
    path: str,
    *,
    y_bar, f_bar, a_hat, std_0, std_T,
    eps_used=None,
    ctx0=None, ctxT=None, lam_context=0.0,
    ctx_dtype="float16",
    allowed=None, near_y=None, near_x=None, bbox=None,
    grid=None, dilate=None, diag=None,
    schedule="cosine", var_correction=True, var_clip=(0.95, 1.02),
    topk=None, target_cov=None, temp=None, tau=None, lam_x=None, lam_f=None,
):
    payload = {}

    payload["y_bar"] = _to_np(y_bar, np.float32)
    payload["f_bar"] = _to_np(f_bar, np.float32)
    payload["a_hat"] = _to_np(a_hat, np.float32)
    payload["std_0"] = _to_np(std_0, np.float32)
    payload["std_T"] = _to_np(std_T, np.float32)

    if eps_used is not None:
        payload["eps_used"] = np.array([float(eps_used)], dtype=np.float32)

    if ctx0 is not None:
        payload["ctx0"] = _to_np(ctx0, np.float16 if ctx_dtype == "float16" else np.float32)
    if ctxT is not None:
        payload["ctxT"] = _to_np(ctxT, np.float16 if ctx_dtype == "float16" else np.float32)
    payload["lam_context"] = np.array([float(lam_context)], dtype=np.float32)
    if allowed is not None:
        payload["allowed"] = _to_np(allowed, np.uint8)
    if near_y is not None:
        payload["near_y"] = _to_np(near_y, np.int32)
    if near_x is not None:
        payload["near_x"] = _to_np(near_x, np.int32)
    if bbox is not None:
        payload["bbox"] = _to_np(np.array(bbox, dtype=np.float32), np.float32)
    if grid is not None:
        payload["grid"] = np.array([int(grid)], dtype=np.int32)
    if dilate is not None:
        payload["dilate"] = np.array([int(dilate)], dtype=np.int32)
    if diag is not None:
        payload["diag"] = np.array([int(bool(diag))], dtype=np.int32)

    payload["schedule"] = np.array([str(schedule)], dtype=object)
    payload["var_correction"] = np.array([int(bool(var_correction))], dtype=np.int32)
    payload["var_clip"] = np.array([float(var_clip[0]), float(var_clip[1])], dtype=np.float32)

    for k, v in dict(topk=topk, target_cov=target_cov, temp=temp, tau=tau, lam_x=lam_x, lam_f=lam_f).items():
        if v is not None:
            payload[k] = np.array([float(v)], dtype=np.float32)

    np.savez_compressed(path, **payload)

def load_guide_pack_npz(path: str, *, device="cpu", dtype=torch.float32):
    z = np.load(path, allow_pickle=True)

    def get(name, default=None):
        return z[name] if name in z.files else default

    pack = {}
    # core
    pack["y_bar"] = torch.as_tensor(get("y_bar"), device=device, dtype=dtype)
    pack["f_bar"] = torch.as_tensor(get("f_bar"), device=device, dtype=dtype)
    pack["a_hat"] = torch.as_tensor(get("a_hat"), device=device, dtype=dtype)
    pack["std_0"] = torch.as_tensor(get("std_0"), device=device, dtype=dtype)
    pack["std_T"] = torch.as_tensor(get("std_T"), device=device, dtype=dtype)

    pack["eps_used"] = float(get("eps_used", np.array([np.nan], np.float32))[0])

    # ctx
    pack["lam_context"] = float(get("lam_context", np.array([0.0], np.float32))[0])
    if "ctx0" in z.files:
        pack["ctx0"] = torch.as_tensor(get("ctx0"), device=device, dtype=dtype)
    else:
        pack["ctx0"] = None
    if "ctxT" in z.files:
        pack["ctxT"] = torch.as_tensor(get("ctxT"), device=device, dtype=dtype)
    else:
        pack["ctxT"] = None

    # obstacle pack
    if "allowed" in z.files:
        pack["allowed_t"] = torch.as_tensor(get("allowed").astype(bool), device=device)
        pack["near_y_t"] = torch.as_tensor(get("near_y").astype(np.int64), device=device)
        pack["near_x_t"] = torch.as_tensor(get("near_x").astype(np.int64), device=device)
        bbox = get("bbox").astype(np.float32).tolist()
        pack["bbox_t"] = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        pack["grid_i"] = int(get("grid")[0])
    else:
        pack["allowed_t"] = None
        pack["near_y_t"] = None
        pack["near_x_t"] = None
        pack["bbox_t"] = None
        pack["grid_i"] = None

    # meta
    pack["schedule"] = str(get("schedule", np.array(["cosine"], dtype=object))[0])
    pack["var_correction"] = bool(int(get("var_correction", np.array([1], np.int32))[0]))
    vc = get("var_clip", np.array([0.95, 1.02], np.float32))
    pack["var_clip"] = (float(vc[0]), float(vc[1]))

    return pack


# ============================================================
# Guide builder (keep your logic)
# ============================================================
def _build_sparse_W_sklearn(
    coords_np: np.ndarray,
    *,
    method: str = "knn",         # "knn" or "radius"
    k: int = 15,
    radius: float = 1.0,
    weight: str = "gaussian",    # "uniform" | "gaussian" | "softmax"
    sigma: float | None = None,  # for gaussian; None -> auto from distances
    tau: float = 1.0,            # for softmax
    exclude_self: bool = True,
    eps: float = 1e-12,
    n_jobs: int = -1,
):
    coords_np = np.asarray(coords_np, dtype=np.float32)
    N, dim = coords_np.shape

    algo = "kd_tree" if dim <= 10 else "ball_tree"
    nn = NearestNeighbors(algorithm=algo, n_jobs=n_jobs)

    if method == "knn":
        # +1 是为了方便去 self
        nn.set_params(n_neighbors=int(k) + (1 if exclude_self else 0))
        nn.fit(coords_np)
        dists, inds = nn.kneighbors(coords_np, return_distance=True)

        # dists/inds: (N, kk)
        rows = np.repeat(np.arange(N, dtype=np.int64), inds.shape[1])
        cols = inds.reshape(-1).astype(np.int64)
        d    = dists.reshape(-1).astype(np.float32)

    elif method == "radius":
        nn.set_params(radius=float(radius))
        nn.fit(coords_np)
        dlist, ilist = nn.radius_neighbors(coords_np, return_distance=True, sort_results=True)

        # list-of-arrays -> COO
        deg  = np.fromiter((len(x) for x in ilist), count=N, dtype=np.int64)
        rows = np.repeat(np.arange(N, dtype=np.int64), deg)
        cols = np.concatenate(ilist).astype(np.int64)
        d    = np.concatenate(dlist).astype(np.float32)

    else:
        raise ValueError("method must be 'knn' or 'radius'")

    if exclude_self:
        mask = (rows != cols)
        rows, cols, d = rows[mask], cols[mask], d[mask]

    if d.size == 0:
        W = sp.csr_matrix((N, N), dtype=np.float32)
        return W

    d2 = d * d

    if weight == "uniform":
        w = np.ones_like(d, dtype=np.float32)

    elif weight == "gaussian":
        if sigma is None:
            sig = np.median(d).astype(np.float32)
            sig = float(max(sig, 1e-6))
        else:
            sig = float(sigma)
        w = np.exp(-0.5 * d2 / (sig * sig)).astype(np.float32)

    elif weight == "softmax":
        t = float(max(tau, 1e-6))
        w = np.exp(-d2 / t).astype(np.float32)

    else:
        raise ValueError("weight must be 'uniform'|'gaussian'|'softmax'")

    # build sparse adjacency
    W = sp.coo_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32).tocsr()

    # row-normalize: W[i,:] sum to 1
    rs = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    inv = 1.0 / np.maximum(rs, eps)
    W = sp.diags(inv) @ W
    return W

@torch.no_grad()
def compute_context_features(
    coords: torch.Tensor,
    features: torch.Tensor,
    *,
    method: str = "knn",          # "knn" or "radius"
    k: int = 15,
    radius: float = 1.0,
    weight: str = "gaussian",     # "uniform" | "gaussian" | "softmax"
    sigma: float | None = None,
    tau: float = 1.0,
    n_scales: int = 1,
    exclude_self: bool = True,
    norm: str = "l2",             # "l2" | "l1" | "none"
    verbose: bool = True,
    out_device: str | torch.device | None = None,
):
    # ---- to CPU numpy ----
    coords_np = coords.detach().cpu().numpy().astype(np.float32)
    feats = features.detach()

    if norm == "l2":
        cur = F.normalize(feats, p=2, dim=1).cpu().numpy().astype(np.float32)
    elif norm == "l1":
        x = feats.clamp_min(0)
        s = x.sum(dim=1, keepdim=True).clamp_min(1e-12)
        cur = (x / s).cpu().numpy().astype(np.float32)
    else:
        cur = feats.cpu().numpy().astype(np.float32)

    # ---- build neighbor weight matrix once ----
    W = _build_sparse_W_sklearn(
        coords_np,
        method=method, k=k, radius=radius,
        weight=weight, sigma=sigma, tau=tau,
        exclude_self=exclude_self,
    )

    # ---- multi-scale propagation ----
    for si in range(int(n_scales)):
        if verbose:
            print(f"[ctx] scale {si+1}/{n_scales}: sparse matmul ...  (E={W.nnz})")
        cur = W @ cur  # (N,D), scipy sparse dot numpy

        # normalize each scale like your original
        if norm == "l2":
            nrm = np.linalg.norm(cur, axis=1, keepdims=True)
            cur = cur / np.maximum(nrm, 1e-12)
        elif norm == "l1":
            s = cur.sum(axis=1, keepdims=True)
            cur = cur / np.maximum(s, 1e-12)

    ctx = torch.tensor(cur, dtype=features.dtype)

    if out_device is None:
        out_device = features.device
    ctx = ctx.to(out_device)
    return ctx
 
def get_neighbor_features(
    adata_src, adata_tgt,
    cell_type_key,
    *,
    # --- ctx---
    method="knn",          # "knn" or "radius"
    k=15,
    radius=50.0,           
    weight="gaussian",    
    sigma=None,            
    tau=1.0,               
    n_scales=2,
    exclude_self=True,
    norm="l2",            
    device="cuda",
):
    coords_src = torch.tensor(adata_src.obsm["spatial"], dtype=torch.float32, device="cpu")
    coords_tgt = torch.tensor(adata_tgt.obsm["spatial"], dtype=torch.float32, device="cpu")

    ct_src = adata_src.obs[cell_type_key].astype(str)
    ct_tgt = adata_tgt.obs[cell_type_key].astype(str)

    all_types = pd.concat([ct_src, ct_tgt], axis=0).unique()
    all_types.sort()

    feat_src_df = pd.get_dummies(pd.Categorical(ct_src, categories=all_types))
    feat_tgt_df = pd.get_dummies(pd.Categorical(ct_tgt, categories=all_types))

    features_src = torch.tensor(feat_src_df.values, dtype=torch.float32, device="cpu")
    features_tgt = torch.tensor(feat_tgt_df.values, dtype=torch.float32, device="cpu")

    ctx_src = compute_context_features(
        coords_src, features_src,
        method=method, k=k, radius=radius,
        weight=weight, sigma=sigma, tau=tau,
        n_scales=n_scales,
        exclude_self=exclude_self,
        norm=norm,
        out_device=device,        
    )

    ctx_tgt = compute_context_features(
        coords_tgt, features_tgt,
        method=method, k=k, radius=radius,
        weight=weight, sigma=sigma, tau=tau,
        n_scales=n_scales,
        exclude_self=exclude_self,
        norm=norm,
        out_device=device,
    )

    return ctx_src, ctx_tgt

def _sinkhorn_unbalanced_retry(a, b, C_np, *, eps0, reg_m, step=0.001, max_eps=0.2, max_tries=300):
    eps_used = float(eps0)
    last_err = None
    for it in range(int(max_tries)):
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")  
                P = ot.unbalanced.sinkhorn_unbalanced(
                    a, b, C_np,
                    reg=float(eps_used),
                    reg_m=float(reg_m),
                )
                if len(wlist) > 0:
                    msg = str(wlist[0].message)
                    raise RuntimeError(f"sinkhorn_unbalanced warning: {msg}")

            return P, eps_used, it  

        except Exception as e:
            last_err = e
            eps_used = float(eps_used + step)
            if eps_used > float(max_eps):
                raise RuntimeError(
                    f"sinkhorn_unbalanced failed even after increasing eps to {eps_used:.4f}. "
                    f"Last error: {last_err}"
                ) from last_err

    raise RuntimeError(f"sinkhorn_unbalanced failed after {max_tries} tries. Last error: {last_err}") from last_err

@torch.no_grad()
def topk_barycentric_adaptive(
    P_raw: torch.Tensor, xT_cpu: torch.Tensor, fT_cpu: torch.Tensor, *,
    Kmax: int = 256, target_cov: float = 0.99,
    temp: float | None = None,
    clip_min: float = 1e-12,
):
    ns, nt = P_raw.shape
    K = min(int(Kmax), nt)

    v, idx = torch.topk(P_raw, k=K, dim=1)
    row = P_raw.sum(1, keepdim=True).clamp_min(clip_min)

    csum = torch.cumsum(v, dim=1)
    cfrac = csum / row
    hit = (cfrac >= float(target_cov))
    k_i = hit.float().argmax(dim=1) + 1
    k_i = torch.where(hit.any(dim=1), k_i, torch.full_like(k_i, K))

    j = torch.arange(K, device=v.device).view(1, K)
    mask = (j < k_i.view(ns, 1)).to(v.dtype)

    v_keep = v * mask

    if (temp is not None) and (temp > 0):
        logits = torch.log(v_keep.clamp_min(clip_min))
        logits = torch.where(mask > 0, logits, logits.new_full(logits.shape, -1e9))
        w = torch.softmax(logits / float(temp), dim=1)
    else:
        w = v_keep

    w = w / (w.sum(1, keepdim=True).clamp_min(clip_min))
    cov = (v_keep.sum(1) / row.view(-1)).clamp(0, 1)

    xT_k = xT_cpu[idx]
    fT_k = fT_cpu[idx]
    y_bar_cpu = (w.unsqueeze(-1) * xT_k).sum(1)
    f_bar_cpu = (w.unsqueeze(-1) * fT_k).sum(1)

    return y_bar_cpu, f_bar_cpu, idx, w, k_i, cov

def build_guide(
    xs0, fs0, ms0, xT, fT, mT,
    *,
    eps=0.05, tau=1.0, lam_x=1.0, lam_f=0.5,
    topk: int = 256, target_cov: float = 0.99,
    temp: float | None = 0.2,
    schedule: str = "cosine",
    var_correction: bool = True,
    var_clip: tuple[float, float] = (0.95, 1.02),
    clip_min: float = 1e-12,
    verbose: bool = False,
    ctx0=None, ctxT=None, lam_context=0.0,

    # ---------- obstacle-aware ----------
    obstacle: bool = True,
    grid: int = 64,
    dilate: int = 2,
    diag: bool = True,
    bigM: float = 1e6,
    src_chunk: int = 128,

    # ---------- NEW ----------
    terrain_gap: bool | None = None,          # ✅ 开关：True 禁止跨洞；False 走原路线
    save_pack_path: str | None = None,        # ✅ 保存 ctx + y_bar 等
    pack_ctx_dtype: str = "float16",          # "float16" or "float32"
):
    # ---- NEW: terrain_gap override ----
    if terrain_gap is not None:
        obstacle = bool(terrain_gap)

    dev, dtype = xs0.device, xs0.dtype
    tiny = torch.tensor(float(clip_min), device=dev, dtype=dtype)

    def _phi(u):
        u = torch.as_tensor(u, device=dev, dtype=dtype).clamp(0, 1)
        if schedule == "cosine":
            return 0.5 * (1 - torch.cos(torch.tensor(np.pi, device=dev, dtype=dtype) * u))
        if schedule == "sigmoid":
            return 1.0 / (1.0 + torch.exp(-6 * (u - 0.5)))
        if schedule == "power2":
            return u * u
        if schedule == "power3":
            return u * u * u
        return u

    # ---------- helpers (kept minimal) ----------
    def _xy_to_ij_np(xy, bbox, g):
        xmin, ymin, xmax, ymax = bbox
        gx = (xy[:, 0] - xmin) / (xmax - xmin + 1e-9) * (g - 1)
        gy = (xy[:, 1] - ymin) / (ymax - ymin + 1e-9) * (g - 1)
        ix = np.clip(np.rint(gx).astype(np.int32), 0, g - 1)
        iy = np.clip(np.rint(gy).astype(np.int32), 0, g - 1)
        return iy, ix

    def _xy_to_ij_t(xy_t, bbox, g):
        xmin, ymin, xmax, ymax = bbox
        gx = (xy_t[:, 0] - xmin) / (xmax - xmin + 1e-9) * (g - 1)
        gy = (xy_t[:, 1] - ymin) / (ymax - ymin + 1e-9) * (g - 1)
        ix = gx.round().clamp(0, g - 1).long()
        iy = gy.round().clamp(0, g - 1).long()
        return iy, ix

    def _ij_to_xy_t(iy, ix, bbox, g, dtype_):
        xmin, ymin, xmax, ymax = bbox
        x = xmin + (ix.to(dtype_) / (g - 1)) * (xmax - xmin)
        y = ymin + (iy.to(dtype_) / (g - 1)) * (ymax - ymin)
        return x, y

    def _build_allowed_and_graph(xy_all, g, dil, diag_):
        # bbox
        xmin, ymin = xy_all.min(0)
        xmax, ymax = xy_all.max(0)
        bbox = (float(xmin), float(ymin), float(xmax), float(ymax))

        # rasterize occupancy
        iy, ix = _xy_to_ij_np(xy_all, bbox, g)
        occ = np.zeros((g, g), dtype=bool)
        occ[iy, ix] = True

        # allowed = dilation(occ)  (注意：不 fill_holes，洞保持不可达)
        allowed = binary_dilation(occ, iterations=int(dil)) if dil > 0 else occ

        # nearest allowed (for projection)
        disallowed = ~allowed
        _, inds = distance_transform_edt(disallowed, return_indices=True)
        near_y = inds[0].astype(np.int64)
        near_x = inds[1].astype(np.int64)

        # build grid graph on allowed pixels
        dx = (bbox[2] - bbox[0]) / (g - 1)
        dy = (bbox[3] - bbox[1]) / (g - 1)
        H = W = g
        n = H * W

        def nid(y, x): return y * W + x

        R, C, Ww = [], [], []
        allow = allowed

        # right
        m = allow[:, :-1] & allow[:, 1:]
        ys, xs = np.where(m)
        u = nid(ys, xs); v = nid(ys, xs + 1)
        w = np.full(u.shape, dx, dtype=np.float64)
        R += [u, v]; C += [v, u]; Ww += [w, w]

        # down
        m = allow[:-1, :] & allow[1:, :]
        ys, xs = np.where(m)
        u = nid(ys, xs); v = nid(ys + 1, xs)
        w = np.full(u.shape, dy, dtype=np.float64)
        R += [u, v]; C += [v, u]; Ww += [w, w]

        if diag_:
            d = float(np.sqrt(dx * dx + dy * dy))
            # down-right
            m = allow[:-1, :-1] & allow[1:, 1:]
            ys, xs = np.where(m)
            u = nid(ys, xs); v = nid(ys + 1, xs + 1)
            w = np.full(u.shape, d, dtype=np.float64)
            R += [u, v]; C += [v, u]; Ww += [w, w]
            # down-left
            m = allow[:-1, 1:] & allow[1:, :-1]
            ys, xs = np.where(m)
            u = nid(ys, xs); v = nid(ys + 1, xs - 1)
            w = np.full(u.shape, d, dtype=np.float64)
            R += [u, v]; C += [v, u]; Ww += [w, w]

        R = np.concatenate(R); C = np.concatenate(C); Ww = np.concatenate(Ww)
        G = coo_matrix((Ww, (R, C)), shape=(n, n)).tocsr()
        return allowed, near_y, near_x, bbox, G

    # ---------- precompute (CPU) ----------
    with torch.no_grad():
        xs_cpu = xs0.detach().cpu().double()
        xT_cpu = xT.detach().cpu().double()
        fs_cpu = fs0.detach().cpu().double()
        fT_cpu = fT.detach().cpu().double()
        ns, nt = int(xs_cpu.shape[0]), int(xT_cpu.shape[0])
        K = min(int(topk), nt)

        # candidates by Euclidean NN in 2D
        nbrs = NearestNeighbors(n_neighbors=K, algorithm="auto").fit(xT_cpu[:, :2].numpy())
        cand = nbrs.kneighbors(xs_cpu[:, :2].numpy(), return_distance=False)
        cand_t = torch.from_numpy(cand).long()

        eps_used = float(eps)  # NEW: record, will be overwritten by retry result

        if obstacle and lam_x > 0:
            # ---- 你的 obstacle 分支原样 ----
            xy_all = np.vstack([xs_cpu[:, :2].numpy(), xT_cpu[:, :2].numpy()])
            allowed_np, near_y_np, near_x_np, bbox, G = _build_allowed_and_graph(
                xy_all, int(grid), int(dilate), bool(diag)
            )

            iy_s, ix_s = _xy_to_ij_np(xs_cpu[:, :2].numpy(), bbox, int(grid))
            iy_t, ix_t = _xy_to_ij_np(xT_cpu[:, :2].numpy(), bbox, int(grid))

            bad_s = ~allowed_np[iy_s, ix_s]
            if bad_s.any():
                iy_s[bad_s] = near_y_np[iy_s[bad_s], ix_s[bad_s]]
                ix_s[bad_s] = near_x_np[iy_s[bad_s], ix_s[bad_s]]

            bad_t = ~allowed_np[iy_t, ix_t]
            if bad_t.any():
                iy_t[bad_t] = near_y_np[iy_t[bad_t], ix_t[bad_t]]
                ix_t[bad_t] = near_x_np[iy_t[bad_t], ix_t[bad_t]]

            src_nodes = (iy_s * int(grid) + ix_s).astype(np.int64)
            tgt_nodes = (iy_t * int(grid) + ix_t).astype(np.int64)

            C_np = np.full((ns, nt), float(bigM), dtype=np.float64)

            use_ctx = (lam_context > 0 and ctx0 is not None and ctxT is not None)
            c0_cpu = ctx0.detach().cpu().double() if use_ctx else None
            cT_cpu = ctxT.detach().cpu().double() if use_ctx else None

            for s in range(0, ns, int(src_chunk)):
                e = min(s + int(src_chunk), ns)
                B = e - s
                cand_be = cand[s:e]
                tgt_be_nodes = tgt_nodes[cand_be]

                dist_all = dijkstra(G, indices=src_nodes[s:e], directed=False)
                row = np.arange(B)[:, None]
                d_geo = dist_all[row, tgt_be_nodes]
                d_geo = np.where(np.isfinite(d_geo), d_geo, float(bigM))
                C_space = d_geo * d_geo

                C_f = 0.0
                if lam_f > 0:
                    f_be = fs_cpu[s:e]
                    fT_be = fT_cpu[cand_t[s:e]]
                    C_f = ((f_be[:, None, :] - fT_be) ** 2).sum(-1).numpy()

                C_ctx = 0.0
                if use_ctx:
                    c_be = c0_cpu[s:e]
                    cT_be = cT_cpu[cand_t[s:e]]
                    C_ctx = ((c_be[:, None, :] - cT_be) ** 2).sum(-1).numpy()

                C_be = lam_x * C_space + lam_f * C_f + lam_context * C_ctx
                rows = np.arange(s, e)[:, None]
                C_np[rows, cand_be] = C_be

            allowed_t = torch.from_numpy(allowed_np).to(device=dev)
            near_y_t = torch.from_numpy(near_y_np).to(device=dev)
            near_x_t = torch.from_numpy(near_x_np).to(device=dev)
            bbox_t = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            grid_i = int(grid)

        else:
            # ---- 原路线：欧式距离（仍然可加 ctx）----
            C_t = lam_x * torch.cdist(xs_cpu, xT_cpu).pow(2)
            if lam_f > 0:
                C_t = C_t + lam_f * torch.cdist(fs_cpu, fT_cpu).pow(2)
            if lam_context > 0 and ctx0 is not None and ctxT is not None:
                C_t = C_t + lam_context * torch.cdist(
                    ctx0.detach().cpu().double(),
                    ctxT.detach().cpu().double()
                ).pow(2)
            C_np = C_t.numpy().astype("float64", copy=False)
            allowed_t = near_y_t = near_x_t = None
            bbox_t = None
            grid_i = None
            allowed_np = near_y_np = near_x_np = bbox = None  # NEW: for saving

        # masses
        a = (ms0 / ms0.sum().clamp_min(1e-8)).detach().cpu().numpy().astype("float64")
        b = (mT  / mT.sum().clamp_min(1e-8)).detach().cpu().numpy().astype("float64")
        reg_m = float(1.0 / max(tau, 1e-6))

        P_np, eps_used, n_retry = _sinkhorn_unbalanced_retry(
            a, b, C_np, eps0=eps, reg_m=reg_m,
            step=0.001, max_eps=0.5, max_tries=500,
        )
        if verbose and n_retry > 0:
            print(f"[Guide] sinkhorn_unbalanced eps {float(eps):.4f} -> {eps_used:.4f} (retries={n_retry})")

        P_raw = torch.from_numpy(P_np).to(torch.float64)
        del P_np
        gc.collect()

        # barycentric targets
        y_bar_cpu, f_bar_cpu, idx, w, k_i, cov = topk_barycentric_adaptive(
            P_raw, xT_cpu, fT_cpu,
            Kmax=K, target_cov=float(target_cov),
            temp=(float(temp) if (temp is not None) else None),
            clip_min=float(clip_min),
        )

        a0_cpu = (ms0 / ms0.sum().clamp_min(clip_min)).to(torch.float64).cpu()
        a_hat_cpu = P_raw.sum(1)
        a_hat_cpu = a_hat_cpu / (a_hat_cpu.sum() + 1e-12)

        std_0_cpu = xs_cpu.std(0, unbiased=False)
        std_T_cpu = xT_cpu.std(0, unbiased=False)

    # move to device
    y_bar = y_bar_cpu.to(device=dev, dtype=dtype)
    f_bar = f_bar_cpu.to(device=dev, dtype=dtype)
    a0 = a0_cpu.to(device=dev, dtype=dtype)
    a_hat = a_hat_cpu.to(device=dev, dtype=dtype)
    std_0 = std_0_cpu.to(device=dev, dtype=dtype)
    std_T = std_T_cpu.to(device=dev, dtype=dtype)

    # ---- NEW: save pack (ctx + obstacle pack + core) ----
    if save_pack_path is not None:
        save_guide_pack_npz(
            save_pack_path,
            y_bar=y_bar_cpu, f_bar=f_bar_cpu, a_hat=a_hat_cpu, std_0=std_0_cpu, std_T=std_T_cpu,
            eps_used=eps_used,
            ctx0=ctx0, ctxT=ctxT, lam_context=lam_context, ctx_dtype=pack_ctx_dtype,
            allowed=allowed_np, near_y=near_y_np, near_x=near_x_np, bbox=bbox,
            grid=grid_i, dilate=dilate, diag=diag,
            schedule=schedule, var_correction=var_correction, var_clip=var_clip,
            topk=topk, target_cov=target_cov, temp=(temp if temp is not None else np.nan),
            tau=tau, lam_x=lam_x, lam_f=lam_f,
        )
        if verbose:
            print(f"[Guide] saved pack -> {save_pack_path}")

    # free temps
    try:
        del P_raw, xs_cpu, xT_cpu, fs_cpu, fT_cpu, C_np
    except Exception:
        pass
    gc.collect()

    # guide(t)
    def guide(t: float):
        t = torch.as_tensor(t, device=dev, dtype=dtype).clamp(0, 1)
        u = _phi(t)

        x_t = (1 - u) * xs0 + u * y_bar
        f_t = (1 - u) * fs0 + u * f_bar

        if var_correction:
            mu = x_t.mean(0, keepdim=True)
            s_now = x_t.std(0, unbiased=False) + 1e-8
            u_var = u * u
            s_tar = (1 - u_var) * std_0 + u_var * std_T
            scale = (s_tar / s_now).clamp(var_clip[0], var_clip[1])
            x_t = mu + (x_t - mu) * scale

        # only when terrain_gap/on (obstacle pack exists)
        if obstacle and (allowed_t is not None):
            xy = x_t[:, :2]
            iy, ix = _xy_to_ij_t(xy, bbox_t, grid_i)
            bad = ~allowed_t[iy, ix]
            if bad.any():
                iy2 = near_y_t[iy[bad], ix[bad]]
                ix2 = near_x_t[iy[bad], ix[bad]]
                x_new, y_new = _ij_to_xy_t(iy2, ix2, bbox_t, grid_i, dtype)
                x_t = x_t.clone()
                x_t[bad, 0] = x_new
                x_t[bad, 1] = y_new

        total_mass_t = (1 - u) * ms0.sum() + u * mT.sum()
        shape = (1 - u) * a0 + u * a_hat
        shape = shape / (shape.sum() + tiny)
        m_t = (total_mass_t * shape).to(ms0.dtype)

        return x_t.detach(), f_t.detach(), m_t.detach()

    return guide