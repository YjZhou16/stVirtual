import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torchdiffeq import odeint
from pathlib import Path
import sys
sys.path.append('/home/zhouyj/stVirtual/src')
import stVirtual_model.ode_multislice as p
import matplotlib as mpl

def plot_snapshots_2d(trace, x_tgt, *, n_show=6000, cols=5, s_src=2, s_tgt=1, title="Trajectory snapshots"):
    xs = trace["x"]
    ts = trace["t"]
    Kp = len(xs)

    N = xs[0].size(0)
    idx = torch.randperm(N, device=xs[0].device)[:min(n_show, N)]
    x_tgt_show = x_tgt.detach().cpu().numpy()
    if x_tgt_show.shape[0] > n_show:
        ridx = np.random.choice(x_tgt_show.shape[0], size=n_show, replace=False)
        x_tgt_show = x_tgt_show[ridx]

    ncols = cols
    nrows = int(np.ceil(Kp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i in range(nrows*ncols):
        ax = axes[i]
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if i >= Kp:
            ax.axis("off")
            continue

        x_i = xs[i][idx].detach().cpu().numpy()
        ax.scatter(x_tgt_show[:,0], x_tgt_show[:,1], s=s_tgt, alpha=0.25)  # target 背景
        ax.scatter(x_i[:,0], x_i[:,1], s=s_src, alpha=0.70)
        ax.set_title(f"t={ts[i]:.2f}")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def rollout_trace_dopri5(
    guide_fn, net, coords0, Z0, lib0,
    *, steps=10, n_cache=256,
    rtol=1e-4, atol=1e-6, max_num_steps=20000
):
    device, dtype = coords0.device, coords0.dtype
    gc = p.GuideCache(guide_fn, n=n_cache, device=device, dtype=dtype)
    func = p.FusedODEFunc(gc, net).to(device)

    N = coords0.size(0)
    y0 = (coords0, Z0, torch.zeros((N,1), device=device, dtype=dtype))
    t_eval = torch.linspace(0.0, 1.0, steps+1, device=device, dtype=dtype)

    xs, qs, ss = odeint(
        func, y0, t_eval,
        method="dopri5",
        rtol=rtol, atol=atol,
        options={"max_num_steps": max_num_steps}
    )

    ms = lib0.view(1, -1, 1) * torch.exp(ss).clamp_min(1e-12)  # [T,N,1]

    trace = {
        "t": t_eval.detach().cpu().numpy(),
        "x": [xs[i] for i in range(xs.size(0))],
        "m": [ms[i].view(-1) for i in range(ms.size(0))],
    }
    return trace

def _make_discrete_cmap(n, base="tab20"):
    base_cmap = mpl.cm.get_cmap(base)
    base_colors = base_cmap(np.linspace(0, 1, base_cmap.N))
    rep = int(np.ceil(n / base_colors.shape[0]))
    colors = np.vstack([base_colors] * rep)[:n]
    if n > 0:
        colors[-1] = np.array([0.7, 0.7, 0.7, 1.0])
    return mpl.colors.ListedColormap(colors)

def plot_snapshots_2d_by_annotation(
    trace, x_tgt, y_src, y_tgt, cats=None,
    *,
    n_show=50000, cols=6, s_src=5, s_tgt=5, alpha_src=0.75,
    alpha_tgt=0.2, cmap_name="tab20", show_legend=False,
):
    xs = trace["x"]
    ts = trace["t"]
    Kp = len(xs)

    device = xs[0].device
    N = xs[0].shape[0]
    M = x_tgt.shape[0]

    idx_src = torch.randperm(N, device=device)[:min(n_show, N)]
    idx_tgt = torch.randperm(M, device=device)[:min(n_show, M)]

    y_src = torch.as_tensor(y_src, device=device)[idx_src].detach().cpu().numpy()
    y_tgt_show = torch.as_tensor(y_tgt, device=device)[idx_tgt].detach().cpu().numpy()

    x_tgt_show = x_tgt[idx_tgt].detach().cpu().numpy()

    n_class = int(max(y_src.max(initial=0), y_tgt_show.max(initial=0)) + 1)
    cmap = _make_discrete_cmap(n_class, base=cmap_name)

    ncols = cols
    nrows = int(np.ceil(Kp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for i in range(nrows*ncols):
        ax = axes[i]
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if i >= Kp:
            ax.axis("off")
            continue

        x_i = xs[i][idx_src].detach().cpu().numpy()

        ax.scatter(
            x_tgt_show[:,0], x_tgt_show[:,1],
            c=y_tgt_show, cmap=cmap, vmin=0, vmax=n_class-1,
            s=s_tgt, alpha=alpha_tgt, linewidths=0
        )

        ax.scatter(
            x_i[:,0], x_i[:,1],
            c=y_src, cmap=cmap, vmin=0, vmax=n_class-1,
            s=s_src, alpha=alpha_src, linewidths=0
        )

        ax.set_title(f"t={ts[i]:.2f}")

    if show_legend and (cats is not None) and (len(cats) <= 20):
        handles = []
        for k, name in enumerate(list(cats)[:n_class]):
            handles.append(mpl.patches.Patch(color=cmap(k), label=str(name)))
        fig.legend(handles=handles, loc="center right", frameon=False)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
    else:
        plt.tight_layout()

    plt.show()


@torch.no_grad()
def rollout_trace_from_out(
    out: dict,
    *,
    steps: int = 10, n_cache: int = 256,
    rtol: float = 1e-4, atol: float = 1e-6,
    max_num_steps: int = 20000, unnormalize: bool = False, 
):

    guide_fn = out["guide_fn"]
    net      = out["net"]
    coords0  = out["coords0"]
    Z0       = out["Z0"]
    lib0     = out["lib0"]

    device, dtype = coords0.device, coords0.dtype
    N = coords0.size(0)

    gc = p.GuideCache(guide_fn, n=n_cache, device=device, dtype=dtype)
    func = p.FusedODEFunc(gc, net).to(device)

    t_eval = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

    y0 = (coords0, Z0, torch.log(lib0.clamp_min(1e-12)).view(N, 1))

    xs, qs, ss = odeint(
        func, y0, t_eval,
        method="dopri5",
        rtol=rtol, atol=atol,
        options={"max_num_steps": max_num_steps},
    )

    ms = torch.exp(ss).clamp_min(1e-12)  

    if unnormalize:
        ns = out.get("norm_stats", None)

        xy_mu = ns["xy_mu"].to(device=device, dtype=dtype)          
        xy_s  = ns["xy_s"].to(device=device, dtype=dtype)         
        f_mu  = ns["f_mu"].to(device=device, dtype=dtype)          
        f_std = ns["f_std"].to(device=device, dtype=dtype)       

        xs = xs * xy_s + xy_mu
        qs = qs * f_std + f_mu

    trace = {
        "t": t_eval.detach().cpu().numpy(),
        "x": [xs[i].detach().cpu() for i in range(xs.size(0))],
        "q": [qs[i].detach().cpu() for i in range(qs.size(0))],
        "s": [ss[i].detach().cpu().view(-1) for i in range(ss.size(0))],
        "m": [ms[i].detach().cpu().view(-1) for i in range(ms.size(0))],
    }
    return trace

