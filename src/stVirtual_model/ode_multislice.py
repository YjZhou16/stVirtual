import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import sys 
sys.path.append("/home/zhouyj/stVirtual/src")
from utils.ode_utils import *
from torch import optim
from torchdiffeq import odeint
from pathlib import Path
from tqdm import tqdm
from geomloss import SamplesLoss

# ============================================================
# UOTEntLoss 
# ============================================================
class UOTEntLoss:
    def __init__(self, lam_x=1.0, lam_f=0.5, *,
                 blur=0.05, reach=1.0, p=2, scaling=0.9,
                 backend_x="multiscale", backend_f="online",
                 debias=True, dtype=torch.float32, device="cuda"):
        self.lam_x = float(lam_x)
        self.lam_f = float(lam_f)
        self.dtype = dtype
        self.p = p
        self.device = torch.device(device)

        kw_x = dict(loss="sinkhorn", p=p, blur=blur, reach=reach,
                    scaling=scaling, backend=backend_x, debias=debias)
        kw_f = dict(loss="sinkhorn", p=p, blur=blur, reach=reach,
                    scaling=scaling, backend=backend_f, debias=debias)

        self.loss_x = SamplesLoss(**kw_x)
        self._kw_f  = kw_f
        self.loss_f = SamplesLoss(**kw_f)

    def _ensure_feature_backend(self, Xf):
        D = Xf.shape[-1]
        backend = getattr(self.loss_f, "backend", None)
        if D > 3 and backend == "multiscale":
            self._kw_f["backend"] = "online"
            self.loss_f = SamplesLoss(**self._kw_f)

    def __call__(self, xs, fs, ms, xt, ft, mt, normalize="none"):
        a = ms.clamp_min(1e-12).to(self.dtype).to(self.device)
        b = mt.clamp_min(1e-12).to(self.dtype).to(self.device)

        scale_x = self.lam_x ** (1.0 / self.p)
        Xx = (scale_x * xs).to(self.dtype).to(self.device)
        Yx = (scale_x * xt).to(self.dtype).to(self.device)

        if Xx.dim() == 2:
            Xx = Xx.unsqueeze(0); Yx = Yx.unsqueeze(0)
            a_ = a.view(1, -1);   b_ = b.view(1, -1)
        else:
            a_, b_ = a, b

        loss_x = self.loss_x(a_, Xx, b_, Yx)

        loss_f = Xx.new_tensor(0.0)
        if (self.lam_f > 0.0) and (fs is not None) and (ft is not None):
            scale_f = self.lam_f ** (1.0 / self.p)
            Xf = (scale_f * fs).to(self.dtype).to(self.device)
            Yf = (scale_f * ft).to(self.dtype).to(self.device)
            if Xf.dim() == 2:
                Xf = Xf.unsqueeze(0); Yf = Yf.unsqueeze(0)

            self._ensure_feature_backend(Xf)
            try:
                loss_f = self.loss_f(a_, Xf, b_, Yf)
            except NotImplementedError:
                self._kw_f["backend"] = "online"
                self.loss_f = SamplesLoss(**self._kw_f)
                loss_f = self.loss_f(a_, Xf, b_, Yf)

        loss = loss_x + loss_f

        if normalize == "N":
            loss = loss / Xx.shape[-2]
        elif normalize == "mass":
            mass_ref = 0.5 * (a.sum() + b.sum()).clamp_min(1e-12)
            loss = loss / mass_ref
        elif normalize == "mass_cost":
            mass_ref = 0.5 * (a.sum() + b.sum()).clamp_min(1e-12)
            with torch.no_grad():
                X2 = Xx if Xx.dim()==2 else Xx.squeeze(0)
                Y2 = Yx if Yx.dim()==2 else Yx.squeeze(0)
                D = torch.cdist(X2, Y2, p=2)
                dmed = D[D>0].median().clamp_min(1e-12)
                cost_ref = (dmed ** self.p)
            loss = loss / (mass_ref * cost_ref)

        return loss

def build_uot_keops(xs0, fs0, xt, ft, *,
                    lam_x=1.0, lam_f=0.0,
                    uot_eps=0.05, uot_tau=1.0,
                    device="cuda", dtype=torch.float32):
    with torch.no_grad():
        xs = xs0.detach().cpu().float()
        xt = xt.detach().cpu().float()

        n = min(2000, xs.size(0))
        m = min(2000, xt.size(0))

        i = torch.randperm(xs.size(0))[:n]
        j = torch.randperm(xt.size(0))[:m]

        d2_med = torch.cdist(xs[i], xt[j]).pow(2).median().clamp_min(1e-12).item()
        eps_eff = max(float(uot_eps) / (lam_x * d2_med + 1e-12), 1e-4)

        blur  = float(np.sqrt(eps_eff))
        reach = float(1.0 / max(uot_tau, 1e-6))

    return UOTEntLoss(lam_x=lam_x, lam_f=lam_f, blur=blur, reach=reach,
                       backend_x="multiscale", backend_f="online",
                       dtype=dtype, device=device)

# ============================================================
# Residual Neural ODE 
# ============================================================
class ResidualDynamicsNet(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 256,
                 residual_scale_x: float = 0.20,
                 residual_scale_f: float = 0.20,
                 residual_scale_s: float = 0.20,
                 dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.residual_scale_x = residual_scale_x
        self.residual_scale_f = residual_scale_f
        self.residual_scale_s = residual_scale_s

        in_dim = (
            2 + latent_dim + 1 +
            2 + latent_dim + 1 +
            2 + latent_dim + 1 +
            1
        )

        layers = []
        dims = [in_dim, hidden, hidden, hidden]
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(hidden, 2 + latent_dim + 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, q, s, xg, fg, logmg, t_scalar: torch.Tensor):
        if t_scalar.dim() == 0:
            tcol = t_scalar.view(1, 1).expand(x.size(0), 1)
        elif t_scalar.dim() == 1 and t_scalar.numel() == 1:
            tcol = t_scalar.view(1, 1).expand(x.size(0), 1)
        else:
            tcol = t_scalar.view(-1, 1)

        dq = q - fg
        dx = x - xg
        ds = s - logmg

        inp = torch.cat([x, q, s, xg, fg, logmg, dx, dq, ds, tcol], dim=1)
        h = self.backbone(inp)
        out = self.head(h)

        dx_res = torch.tanh(out[:, :2]) * self.residual_scale_x
        dq_res = torch.tanh(out[:, 2:2+self.latent_dim]) * self.residual_scale_f
        ds_res = torch.tanh(out[:, 2+self.latent_dim:]) * self.residual_scale_s

        return dx_res, dq_res, ds_res

class GuideCache:
    def __init__(self, guide_fn, *, n=128, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

        with torch.no_grad():
            t = torch.linspace(0.0, 1.0, n, device=self.device, dtype=self.dtype)
            X, F_, LM = [], [], []
            for tt in t.tolist():
                xg, fg, mg = guide_fn(float(tt))
                X.append(xg.to(self.device, self.dtype))
                F_.append(fg.to(self.device, self.dtype))
                LM.append(torch.log(mg.clamp_min(1e-12)).to(self.device, self.dtype).view(-1, 1))

            self.t = t
            self.xg = torch.stack(X, dim=0)
            self.fg = torch.stack(F_, dim=0)
            self.logmg = torch.stack(LM, dim=0)

            dt = (t[1] - t[0]).clamp_min(1e-12)

            self.vg = torch.zeros_like(self.xg)
            self.wg = torch.zeros_like(self.fg)
            self.ag = torch.zeros_like(self.logmg)

            self.vg[1:-1] = (self.xg[2:] - self.xg[:-2]) / (2*dt)
            self.wg[1:-1] = (self.fg[2:] - self.fg[:-2]) / (2*dt)
            self.ag[1:-1] = (self.logmg[2:] - self.logmg[:-2]) / (2*dt)

            self.vg[0]  = (self.xg[1] - self.xg[0]) / dt
            self.wg[0]  = (self.fg[1] - self.fg[0]) / dt
            self.ag[0]  = (self.logmg[1] - self.logmg[0]) / dt
            self.vg[-1] = (self.xg[-1] - self.xg[-2]) / dt
            self.wg[-1] = (self.fg[-1] - self.fg[-2]) / dt
            self.ag[-1] = (self.logmg[-1] - self.logmg[-2]) / dt

    def interp(self, t_scalar: torch.Tensor):
        t = t_scalar.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.t, t).clamp(1, self.t.numel()-1) - 1
        t0 = self.t[idx]
        t1 = self.t[idx+1]
        w = ((t - t0) / (t1 - t0 + 1e-12)).to(self.dtype)

        def lerp(A):
            return (1-w)*A[idx] + w*A[idx+1]

        return lerp(self.xg), lerp(self.fg), lerp(self.logmg), lerp(self.vg), lerp(self.wg), lerp(self.ag)

class FusedODEFunc(nn.Module):
    def __init__(self, guide_cache: GuideCache, net: nn.Module):
        super().__init__()
        self.gc = guide_cache
        self.net = net

    def forward(self, t, state):
        x, q, s = state
        xg, fg, logmg, vg, wg, ag = self.gc.interp(t)

        dx_res, dq_res, ds_res = self.net(x, q, s, xg, fg, logmg, t)
        dx = vg + dx_res
        dq = wg + dq_res
        ds = ag + ds_res
        return (dx, dq, ds)

def rollout_neuralode_dopri5(
    func: FusedODEFunc,
    coords0: torch.Tensor,
    Z0: torch.Tensor,
    lib0: torch.Tensor,
    *,
    steps: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_num_steps: int = 20000,
):
    device, dtype = coords0.device, coords0.dtype
    N = coords0.size(0)

    y0 = (coords0, Z0, torch.log(lib0.clamp_min(1e-12)).view(N, 1))
    t_eval = torch.linspace(0.0, 1.0, steps+1, device=device, dtype=dtype)

    xs, qs, ss = odeint(
        func, y0, t_eval,
        method="dopri5",
        rtol=rtol, atol=atol,
        options={"max_num_steps": max_num_steps}
    )
    return t_eval, xs, qs, ss

# ============================================================
# kNN losses 
# ============================================================

@torch.no_grad()
def knn_idx(query_x: torch.Tensor, all_x: torch.Tensor, k: int):
    D = torch.cdist(query_x, all_x)
    _, idx = torch.topk(D, k=min(k, all_x.size(0)), largest=False, dim=1)
    return idx

def knn_velocity_smoothness(
    x: torch.Tensor,
    v: torch.Tensor,
    *,
    k: int = 16,
    n_sample: int = 2048,
):
    device = x.device
    N = x.size(0)
    B = min(int(n_sample), N)

    idx_q = torch.randperm(N, device=device)[:B]
    xq = x.detach()[idx_q]

    with torch.no_grad():
        idx_nn = knn_idx(xq, x.detach(), k=k)

    vq  = v[idx_q]
    vnn = v[idx_nn]
    return ((vq[:, None, :] - vnn) ** 2).sum(-1).mean()

# ============================================================
# Helpers
# ============================================================
def _to_dense_float32(X):
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if np.isnan(X).any() or np.isinf(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def _get_latent_np(adata, model=None, latent_layer="scaled"):
    if model is not None:
        Z = model.get_latent_representation(adata=adata)
        return np.asarray(Z, dtype=np.float32)

    if latent_layer is not None and latent_layer in adata.layers:
        X = adata.layers[latent_layer]
    else:
        X = adata.X
    return _to_dense_float32(X)

def _get_library_torch(adata, device, dtype=torch.float32, lib_layer="counts"):
    if lib_layer is not None and lib_layer in adata.layers:
        X = adata.layers[lib_layer]
    else:
        X = adata.X

    if sp.issparse(X):
        s = np.ravel(X.sum(axis=1)).astype("float32")
    else:
        s = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
    return torch.tensor(s, device=device, dtype=dtype).clamp_min(1e-12)

def _median_pairwise_distance_torch(all_xy, sample_pairs=200_000, pdist_threshold=6000):
    x = all_xy.float().contiguous()
    N = x.size(0)
    if N <= pdist_threshold:
        return torch.pdist(x).median().clamp_min(1e-12)

    S = min(sample_pairs, max(1, (N*(N-1))//2))
    i = torch.randint(0, N, (S,), device=x.device)
    j = torch.randint(0, N, (S,), device=x.device)
    m = (i != j)
    i, j = i[m], j[m]
    return (x[i] - x[j]).norm(dim=1).median().clamp_min(1e-12)

def _subset_adata_by_slice(adata_all, slice_key: str, slice_id):
    if slice_key not in adata_all.obs.columns:
        raise KeyError(f"adata_all.obs has no `{slice_key}`, exsiting cols:{list(adata_all.obs.columns)[:30]} ...")
    sid = str(slice_id)
    m = (adata_all.obs[slice_key].astype(str).values == sid)
    if m.sum() == 0:
        raise ValueError(f"can't find slice_id={slice_id} in slice_key={slice_key}  (as '{sid}')")
    return adata_all[m].copy()

def compute_global_norm_stats(
    adata_all,
    *,
    slice_key="sample",
    route_ids=None,               
    x_key="cx_aligned",
    y_key="cy_aligned",
    model=None,
    latent_layer="scaled",
    device="cuda",
    seed=2025,
    sample_pairs=200_000,
    pdist_threshold=6000,
    max_cells_for_f=200_000,
):
    set_seed(seed)
    dev = torch.device(device)
    DTYPE = torch.float32

    if route_ids is not None:
        route_ids = [str(x) for x in route_ids]
        m = adata_all.obs[slice_key].astype(str).isin(route_ids).values
        ad = adata_all[m]
        if ad.n_obs == 0:
            raise ValueError("No cells after filtering by route_ids, check slice_key/route_ids")
    else:
        ad = adata_all

    # coords
    xy_np = np.c_[ad.obs[x_key].to_numpy(), ad.obs[y_key].to_numpy()].astype(np.float32)
    all_xy = torch.tensor(xy_np, device=dev, dtype=DTYPE)
    xy_mu = all_xy.mean(0, keepdim=True)
    xy_s  = _median_pairwise_distance_torch(all_xy, sample_pairs=sample_pairs, pdist_threshold=pdist_threshold)

    # latent
    Z_np = _get_latent_np(ad, model=model, latent_layer=latent_layer)  # (N,D)
    N = Z_np.shape[0]
    if N > max_cells_for_f:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=max_cells_for_f, replace=False)
        Z_use = Z_np[idx]
    else:
        Z_use = Z_np
    all_f = torch.tensor(Z_use, device=dev, dtype=DTYPE)
    f_mu  = all_f.mean(0, keepdim=True)
    f_std = all_f.std(0, unbiased=False).clamp_min(1e-6)

    return {
        "xy_mu": xy_mu.detach(),
        "xy_s":  xy_s.detach(),
        "f_mu":  f_mu.detach(),
        "f_std": f_std.detach(),
        "latent_dim": int(Z_np.shape[1]),
        "slice_key": slice_key,
        "route_ids": None if route_ids is None else list(route_ids),
        "x_key": x_key,
        "y_key": y_key,
        "latent_layer": latent_layer,
    }

# ============================================================
# Single-pair training (Global-normalized)
# ============================================================
def train_model(
    model, adata_all, adata_src, adata_tgt,
    *,
    # keys
    x_key="cx_aligned", y_key="cy_aligned", latent_layer="scaled", lib_layer="counts",             
    # normalization
    norm_stats=None, slice_key="sample",route_ids=None,
    # dims & train
    latent_dim=None, steps=10, epochs=300, lr_init=2e-4, device="cuda", ode_hidden=256,
    # uot
    uot_eps=0.05, uot_tau=1.0, uot_lam_x=2.0, uot_lam_f=0.5, normalize="mass",
    # guide
    guide_eps=0.001,guide_topk=8, guide_temp=0.2, guide_schedule="linear",
    var_correction=False, lam_context=10.0, cell_type_key="cell_type",
    save_pack_path="stage1_pack.npz",terrain_gap=False,
    # dopri5
    rtol=1e-3, atol=1e-4, max_num_steps=20000, n_cache=256,
    # velocity smoothness
    lam_vsmooth=0.1, knn_k=16, n_speed_sample=2048, n_time_samples=10,
    # other
    lam_uot=1.0, lam_residual=1e-2, save_dir=None, verbose=False,
):
    DTYPE = torch.float32
    device = torch.device(device)
    set_seed(2025)

    # ---- global norm stats ----
    if norm_stats is None:
        norm_stats = compute_global_norm_stats(
            adata_all,
            slice_key=slice_key,
            route_ids=route_ids,
            x_key=x_key, y_key=y_key,
            model=model,
            latent_layer=latent_layer,
            device=str(device),
        )

    xy_mu = norm_stats["xy_mu"].to(device)
    xy_s  = norm_stats["xy_s"].to(device)
    f_mu  = norm_stats["f_mu"].to(device)
    f_std = norm_stats["f_std"].to(device)

    def scale_x(x): return (x - xy_mu) / xy_s
    def scale_f(f): return (f - f_mu) / f_std

    if lam_context > 0:
        print("Pre-computing neighbor context features...")
        ctx0, ctxT  = get_neighbor_features(
            adata_src, adata_tgt, cell_type_key,
            method="knn", k=16,
            weight="gaussian", sigma=None,   
            n_scales=1,
            device="cuda",
        )

    else:
        ctx0, ctxT = None, None

    # ---- data ----
    coords0 = torch.tensor(
        np.c_[adata_src.obs[x_key].to_numpy(), adata_src.obs[y_key].to_numpy()].astype(np.float32),
        dtype=DTYPE, device=device
    )
    coords_tgt = torch.tensor(
        np.c_[adata_tgt.obs[x_key].to_numpy(), adata_tgt.obs[y_key].to_numpy()].astype(np.float32),
        dtype=DTYPE, device=device
    )

    Z0_np = _get_latent_np(adata_src, model=model, latent_layer=latent_layer)
    Zt_np = _get_latent_np(adata_tgt, model=model, latent_layer=latent_layer)
    Z0 = torch.tensor(Z0_np, dtype=DTYPE, device=device)
    Zt = torch.tensor(Zt_np, dtype=DTYPE, device=device)

    if latent_dim is None:
        latent_dim = int(Z0.shape[1])
    else:
        assert Z0.shape[1] == latent_dim, f"Z dim={Z0.shape[1]} != latent_dim={latent_dim}"

    lib0    = _get_library_torch(adata_src, device=device, dtype=DTYPE, lib_layer=lib_layer)
    lib_tgt = _get_library_torch(adata_tgt, device=device, dtype=DTYPE, lib_layer=lib_layer)

    # ---- apply GLOBAL normalization ----
    coords0    = scale_x(coords0)
    coords_tgt = scale_x(coords_tgt)
    Z0         = scale_f(Z0)
    Zt         = scale_f(Zt)

    # ---- build guide & uot ----
    guide_fn = build_guide(
        coords0, Z0, lib0,
        coords_tgt, Zt, lib_tgt,
        eps=guide_eps, tau=uot_tau,
        lam_x=uot_lam_x, lam_f=uot_lam_f,
        topk=guide_topk, temp=guide_temp,
        schedule=guide_schedule,
        ctx0=ctx0, ctxT=ctxT, lam_context=lam_context,
        var_correction=var_correction,
        terrain_gap=terrain_gap,
        save_pack_path=save_pack_path,
        verbose=verbose,
        )


    uot_keops = build_uot_keops(
        coords0, Z0, coords_tgt, Zt,
        lam_x=uot_lam_x, lam_f=uot_lam_f,
        uot_eps=uot_eps, uot_tau=uot_tau,
        device=device.type, dtype=DTYPE
    )

    # ---- dynamics net ----
    net = ResidualDynamicsNet(
        latent_dim=latent_dim,
        hidden=ode_hidden,
        residual_scale_x=0.20,
        residual_scale_f=0.05,
        residual_scale_s=0.05
    ).to(device)

    opt = optim.AdamW(net.parameters(), lr=lr_init)

    guide_cache = GuideCache(guide_fn, n=n_cache, device=device, dtype=DTYPE)
    func = FusedODEFunc(guide_cache, net).to(device)

    # ---- save dir ----
    if save_dir is not None:
        sd = Path(save_dir); sd.mkdir(parents=True, exist_ok=True)
        (sd / "checkpoints").mkdir(parents=True, exist_ok=True)

    best = float("inf")
    pbar = tqdm(range(1, epochs + 1), dynamic_ncols=True, desc="Train (NeuralODE dopri5)")

    for epoch in pbar:
        opt.zero_grad(set_to_none=True)

        # ---- forward rollout ----
        t_eval, xs, qs, ss = rollout_neuralode_dopri5(
            func, coords0, Z0, lib0,
            steps=steps, rtol=rtol, atol=atol, max_num_steps=max_num_steps
        )

        x1, q1, s1 = xs[-1], qs[-1], ss[-1]
        m1 = torch.exp(s1.view(-1)).clamp_min(1e-12)

        # ---- terminal uot ----
        loss_uot = uot_keops(x1, q1, m1, coords_tgt, 
                             Zt, lib_tgt, normalize=normalize)

        # ---- (A) kNN velocity smoothness ----
        loss_vsmooth = x1.new_tensor(0.0)
        if n_time_samples >= xs.size(0):
            time_ids = torch.arange(xs.size(0), device=xs.device)
        else:
            time_ids = torch.randint(0, xs.size(0), (n_time_samples,), device=xs.device)

        for i in time_ids:
            dx, dq, ds = func(t_eval[i], (xs[i], qs[i], ss[i]))
            loss_vsmooth = loss_vsmooth + knn_velocity_smoothness(
                xs[i], dx, k=knn_k, n_sample=n_speed_sample
            )
        loss_vsmooth = loss_vsmooth / max(1, time_ids.numel())

        # ---- (B) residual energy ----
        loss_residual = x1.new_tensor(0.0)
        for i in time_ids:
            xg, fg, logmg, vg, wg, ag = guide_cache.interp(t_eval[i])
            dx_res, dq_res, ds_res = net(xs[i], qs[i], ss[i], xg, fg, logmg, t_eval[i])
            loss_residual = loss_residual + (
                (dx_res**2).sum(-1).mean() +
                0.5 * (dq_res**2).sum(-1).mean() +
                0.1 * (ds_res**2).mean()
            )
        loss_residual = loss_residual / max(1, time_ids.numel())

        # ---- total loss ----
        loss = (
            lam_uot * loss_uot
            + lam_residual * loss_residual
            + lam_vsmooth * loss_vsmooth
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        val = float(loss.detach().item())

        pbar.set_postfix({
            "loss": f"{val:.4f}",
            "uot": f"{float(loss_uot.detach().item()):.4f}",
            "vs": f"{float(loss_vsmooth.detach().item()):.4f}",
        })

        # ---- save ckpt ----
        if save_dir is not None:
            sd = Path(save_dir)
            (sd / "checkpoints").mkdir(parents=True, exist_ok=True)

            ckpt = {
                "epoch": epoch,
                "best": best,
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "global_norm": { 
                    "xy_mu": norm_stats["xy_mu"].detach().cpu(),
                    "xy_s":  norm_stats["xy_s"].detach().cpu(),
                    "f_mu":  norm_stats["f_mu"].detach().cpu(),
                    "f_std": norm_stats["f_std"].detach().cpu(),
                    "meta": {
                        "slice_key": norm_stats.get("slice_key"),
                        "route_ids": norm_stats.get("route_ids"),
                        "x_key": norm_stats.get("x_key"),
                        "y_key": norm_stats.get("y_key"),
                        "latent_layer": norm_stats.get("latent_layer"),
                        "latent_dim": norm_stats.get("latent_dim"),
                    }
                },
                "uot_hparams": {
                    "eps": uot_eps, "tau": uot_tau, "lam_x": uot_lam_x, "lam_f": uot_lam_f,
                    "backend": "multiscale", "debias": True
                },
                "ode_hparams": {
                    "latent_dim": latent_dim,
                    "hidden": ode_hidden,
                    "steps": steps,
                    "rtol": rtol, "atol": atol,
                    "method": "dopri5",
                    "n_cache": n_cache,
                },
                "guide_hparams": {
                    "topk": guide_topk,
                    "temp": guide_temp,
                    "schedule": guide_schedule,
                },
                "loss_hparams": {
                    "lam_uot": lam_uot, 
                    "lam_vsmooth": lam_vsmooth, "lam_residual": lam_residual,
                }
            }

            torch.save(ckpt, sd / "checkpoints" / "last.pt")
            if val < best:
                best = val
                ckpt["best"] = best
                torch.save(ckpt, sd / "checkpoints" / "best.pt")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return {
        "guide_fn": guide_fn,
        "guide_cache": guide_cache,
        "net": net,
        "func": func,
        "coords0": coords0, "Z0": Z0, "lib0": lib0,
        "coords_tgt": coords_tgt, "Zt": Zt, "lib_tgt": lib_tgt,
        "best": best,
        "norm_stats": norm_stats,  
    }

# ============================================================
# Multi-slice wrapper
# ============================================================
def train_model_multislice(
    model,
    adata_all,
    steps,
    *,
    slice_key="sample",
    guide_eps = 0.01,
    guide_topk=512,
    guide_temp=0.2,
    guide_schedule="linear",
    uot_eps=0.05, uot_tau=1.0, 
    uot_lam_x=2.0, uot_lam_f=0.5,
    lam_residual=1e-2,
    lam_vsmooth=0.1,
    lam_uot=1.0,
    lam_context=10.0,   
    cell_type_key="cell_type",
    save_pack_path="stage1_pack.npz",
    terrain_gap=False,
    route_ids=None,                
    start_id=None,
    target_ids=None,
    save_root=None,
    verbose=False,
    **train_kwargs,
):
    if route_ids is None:
        if start_id is None or target_ids is None:
            raise ValueError("plz upload route_ids=[...] or (start_id=..., target_ids=[...])")
        route_ids = [start_id] + list(target_ids)

    route_ids = list(route_ids)
    if len(route_ids) < 2:
        raise ValueError("route_ids need at least 2 slices, e.g. [1,3]")

    norm_stats = compute_global_norm_stats(
        adata_all,
        slice_key=slice_key,
        route_ids=route_ids,
        x_key=train_kwargs.get("x_key", "cx_aligned"),
        y_key=train_kwargs.get("y_key", "cy_aligned"),
        model=model,
        latent_layer=train_kwargs.get("latent_layer", "scaled"),
        device=train_kwargs.get("device", "cuda"),
    )


    if train_kwargs.get("latent_dim", None) is None:
        train_kwargs["latent_dim"] = norm_stats["latent_dim"]

    pairs = list(zip(route_ids[:-1], route_ids[1:]))

    results = {}
    for sid0, sid1 in pairs:
        ad_src = _subset_adata_by_slice(adata_all, slice_key, sid0)
        ad_tgt = _subset_adata_by_slice(adata_all, slice_key, sid1)

        seg_name = f"{sid0}_to_{sid1}"
        seg_save = None
        if save_root is not None:
            seg_save = str(Path(save_root) / seg_name)

        print(f"\n==============================")
        print(f"[MultiSlice] Train segment: {seg_name} (n_src={ad_src.n_obs}, n_tgt={ad_tgt.n_obs})")
        print(f"  Global norm route_ids={route_ids}  latent_dim={train_kwargs['latent_dim']}")
        print(f"==============================\n")

        out = train_model(
            model,
            adata_all,
            ad_src, ad_tgt,
            steps=steps,
            guide_eps=guide_eps,
            guide_temp=guide_temp,
            guide_topk=guide_topk,
            guide_schedule=guide_schedule,
            uot_eps=uot_eps, uot_tau=uot_tau, 
            uot_lam_x=uot_lam_x, uot_lam_f=uot_lam_f,
            slice_key=slice_key,
            route_ids=route_ids,
            norm_stats=norm_stats,    
            save_dir=seg_save,
            verbose=verbose,
            lam_residual=lam_residual,
            lam_vsmooth=lam_vsmooth,
            lam_uot=lam_uot,
            lam_context=lam_context,   
            cell_type_key=cell_type_key,
            save_pack_path=save_pack_path,
            terrain_gap=terrain_gap,
            **train_kwargs
        )
        results[seg_name] = out

    return results