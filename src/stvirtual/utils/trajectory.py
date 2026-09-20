import re
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from torchdiffeq import odeint
from pathlib import Path

import sys
from stvirtual.models import stage1_2d as p
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
    trace,
    x_tgt,
    y_src,
    y_tgt,
    cats=None,
    *,
    n_show=50000,
    cols=6,
    s_src=5,
    s_tgt=5,
    alpha_src=0.75,
    alpha_tgt=0.2,
    cmap_name="tab20",
    show_legend=False,
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
    steps: int = 10,
    n_cache: int = 256,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_num_steps: int = 20000,
    unnormalize: bool = False, 
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

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from pathlib import Path
# from matplotlib.colors import ListedColormap, BoundaryNorm

# def _make_custom_discrete_cmap(n_class: int, hex_colors):
#     base = [mpl.colors.to_rgba(c) for c in hex_colors]
#     if n_class <= len(base):
#         colors = base[:n_class]
#     else:
#         colors = (base * (n_class // len(base) + 1))[:n_class]
#     cmap = ListedColormap(colors, name="custom_discrete")
#     boundaries = np.arange(n_class + 1) - 0.5
#     norm = BoundaryNorm(boundaries, ncolors=n_class)
#     return cmap, norm

# def save_snapshots_2d_by_annotation_transparent(
#     trace,
#     y_src,
#     cats=None,
#     *,
#     out_dir="snapshots_transparent",
#     n_show=50000,
#     s_src=5,
#     alpha_src=0.75,
#     dpi=300,
#     fmt="png",
#     prefix="frame",
#     seed=None,
#     show_legend=False,
#     transparent=True,   # 关键：透明底
#     pad_inches=0.02,
# ):
#     xs = trace["x"]
#     ts = trace.get("t", np.arange(len(xs)))

#     device = xs[0].device
#     N = xs[0].shape[0]

#     # 固定采样（所有帧同一批点）
#     if seed is not None:
#         g = torch.Generator(device=device)
#         g.manual_seed(int(seed))
#         idx_src = torch.randperm(N, generator=g, device=device)[:min(n_show, N)]
#     else:
#         idx_src = torch.randperm(N, device=device)[:min(n_show, N)]

#     y_src_np = torch.as_tensor(y_src, device=device)[idx_src].detach().cpu().numpy()
#     n_class = int(y_src_np.max(initial=0) + 1)

#     hex_colors = ["#2B54C7", "#25D690", "#EE8C0C", "#EE416C"]
#     cmap, norm = _make_custom_discrete_cmap(n_class, hex_colors)

#     ts_np = ts.detach().cpu().numpy() if torch.is_tensor(ts) else np.asarray(ts)

#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     saved = []
#     for i, x in enumerate(xs):
#         x_i = x[idx_src].detach().cpu().numpy()

#         fig = plt.figure(figsize=(6, 6))
#         # 关键：figure 和 axes 都设为透明
#         fig.patch.set_alpha(0.0)
#         ax = fig.add_subplot(111)
#         ax.set_aspect("equal")
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.set_facecolor((0, 0, 0, 0))  # 透明 axes 背景
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#         ax.scatter(
#             x_i[:, 0], x_i[:, 1],
#             c=y_src_np, cmap=cmap, norm=norm,
#             s=s_src, alpha=alpha_src, linewidths=0
#         )
#         ax.set_title(f"t={float(ts_np[i]):.2f}")

#         if show_legend and (cats is not None) and (len(cats) <= 20):
#             handles = [mpl.patches.Patch(color=cmap(k), label=str(name))
#                        for k, name in enumerate(list(cats)[:n_class])]
#             leg = ax.legend(handles=handles, loc="best", frameon=False)
#             # legend 也尽量透明（frameon=False 已经没底了）

#         fname = f"{prefix}_{i:03d}_t{float(ts_np[i]):.3f}.{fmt}"
#         fpath = out_dir / fname
#         fig.savefig(
#             fpath,
#             dpi=dpi,
#             bbox_inches="tight",
#             pad_inches=pad_inches,
#             transparent=transparent  # 关键：输出 PNG 透明通道
#         )
#         plt.close(fig)
#         saved.append(fpath)

#     return saved

# save_snapshots_2d_by_annotation_transparent(
#     trace=trace,
#     y_src=y_src,
#     cats=cats,
#     out_dir="./snap_png_transparent",
#     seed=0,
#     dpi=300,
# )