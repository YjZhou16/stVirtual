import re
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from torchdiffeq import odeint
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys

from stvirtual.models import stage1_3d as p
import matplotlib as mpl


def _make_discrete_cmap(n, base="tab20"):
    base_cmap = mpl.cm.get_cmap(base)
    base_colors = base_cmap(np.linspace(0, 1, base_cmap.N))
    rep = int(np.ceil(n / base_colors.shape[0]))
    colors = np.vstack([base_colors] * rep)[:n]
    if n > 0:
        colors[-1] = np.array([0.7, 0.7, 0.7, 1.0])
    return mpl.colors.ListedColormap(colors)


def _to_xyz_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expect 2D array, got shape={x.shape}")
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 2:
        z = np.zeros((x.shape[0], 1), dtype=x.dtype)
        return np.concatenate([x, z], axis=1)
    raise ValueError(f"expect (N,2) or (N,3), got shape={x.shape}")


def _set_ax_3d(ax, xlim, ylim, zlim, elev=25, azim=-60, invert_y=False):
    ax.set_xlim(*xlim)
    if invert_y:
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    xr = max(float(xlim[1] - xlim[0]), 1e-6)
    yr = max(float(ylim[1] - ylim[0]), 1e-6)
    zr = max(float(zlim[1] - zlim[0]), 1e-6)
    ax.set_box_aspect((xr, yr, zr))

    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_snapshots_3d(
    trace,
    x_tgt,
    *,
    n_show=6000,
    cols=5,
    s_src=2,
    s_tgt=1,
    alpha_src=0.70,
    alpha_tgt=0.25,
    title="Trajectory snapshots (3D)",
    elev=25,
    azim=-60,
    invert_y=False,
):
    xs = trace["x"]
    ts = trace["t"]
    Kp = len(xs)

    x0 = _to_xyz_np(xs[0])
    N = x0.shape[0]
    idx = torch.randperm(N, device=xs[0].device if torch.is_tensor(xs[0]) else "cpu")[:min(n_show, N)]
    idx = idx.detach().cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)

    x_tgt_show = _to_xyz_np(x_tgt)
    if x_tgt_show.shape[0] > n_show:
        ridx = np.random.choice(x_tgt_show.shape[0], size=n_show, replace=False)
        x_tgt_show = x_tgt_show[ridx]

    mins = [x_tgt_show.min(0)]
    maxs = [x_tgt_show.max(0)]
    for x in xs:
        xi = _to_xyz_np(x)
        xi = xi[idx]
        mins.append(xi.min(0))
        maxs.append(xi.max(0))

    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    pad = 0.03 * (mx - mn + 1e-12)
    xlim = (mn[0] - pad[0], mx[0] + pad[0])
    ylim = (mn[1] - pad[1], mx[1] + pad[1])
    zlim = (mn[2] - pad[2], mx[2] + pad[2])

    ncols = cols
    nrows = int(np.ceil(Kp / ncols))
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows), dpi=150)
    axes = []

    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        axes.append(ax)

        if i >= Kp:
            ax.axis("off")
            continue

        x_i = _to_xyz_np(xs[i])[idx]

        ax.scatter(
            x_tgt_show[:, 0], x_tgt_show[:, 1], x_tgt_show[:, 2],
            s=s_tgt, alpha=alpha_tgt, linewidths=0, depthshade=False
        )
        ax.scatter(
            x_i[:, 0], x_i[:, 1], x_i[:, 2],
            s=s_src, alpha=alpha_src, linewidths=0, depthshade=False
        )
        ax.set_title(f"t={ts[i]:.2f}")
        _set_ax_3d(ax, xlim, ylim, zlim, elev=elev, azim=azim, invert_y=invert_y)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_snapshots_3d_by_annotation(
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
    elev=25,
    azim=-60,
    invert_y=False,
):
    xs = trace["x"]
    ts = trace["t"]
    Kp = len(xs)

    x0 = _to_xyz_np(xs[0])
    x_tgt_np = _to_xyz_np(x_tgt)

    device = xs[0].device if torch.is_tensor(xs[0]) else "cpu"
    N = x0.shape[0]
    M = x_tgt_np.shape[0]

    idx_src = torch.randperm(N, device=device)[:min(n_show, N)]
    idx_tgt = torch.randperm(M, device=device)[:min(n_show, M)]
    idx_src_np = idx_src.detach().cpu().numpy() if torch.is_tensor(idx_src) else np.asarray(idx_src)
    idx_tgt_np = idx_tgt.detach().cpu().numpy() if torch.is_tensor(idx_tgt) else np.asarray(idx_tgt)

    y_src = torch.as_tensor(y_src, device=device)[idx_src].detach().cpu().numpy()
    y_tgt_show = torch.as_tensor(y_tgt, device=device)[idx_tgt].detach().cpu().numpy()

    x_tgt_show = x_tgt_np[idx_tgt_np]

    n_class = int(max(y_src.max(initial=0), y_tgt_show.max(initial=0)) + 1)
    cmap = _make_discrete_cmap(n_class, base=cmap_name)

    mins = [x_tgt_show.min(0)]
    maxs = [x_tgt_show.max(0)]
    for x in xs:
        xi = _to_xyz_np(x)[idx_src_np]
        mins.append(xi.min(0))
        maxs.append(xi.max(0))

    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    pad = 0.03 * (mx - mn + 1e-12)
    xlim = (mn[0] - pad[0], mx[0] + pad[0])
    ylim = (mn[1] - pad[1], mx[1] + pad[1])
    zlim = (mn[2] - pad[2], mx[2] + pad[2])

    ncols = cols
    nrows = int(np.ceil(Kp / ncols))
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows), dpi=150)
    axes = []

    for i in range(nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        axes.append(ax)

        if i >= Kp:
            ax.axis("off")
            continue

        x_i = _to_xyz_np(xs[i])[idx_src_np]

        ax.scatter(
            x_tgt_show[:, 0], x_tgt_show[:, 1], x_tgt_show[:, 2],
            c=y_tgt_show, cmap=cmap, vmin=0, vmax=n_class - 1,
            s=s_tgt, alpha=alpha_tgt, linewidths=0, depthshade=False
        )

        ax.scatter(
            x_i[:, 0], x_i[:, 1], x_i[:, 2],
            c=y_src, cmap=cmap, vmin=0, vmax=n_class - 1,
            s=s_src, alpha=alpha_src, linewidths=0, depthshade=False
        )

        ax.set_title(f"t={ts[i]:.2f}")
        _set_ax_3d(ax, xlim, ylim, zlim, elev=elev, azim=azim, invert_y=invert_y)

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
def rollout_trace_dopri5(
    guide_fn, net, coords0, Z0, lib0,
    *, steps=10, n_cache=256,
    rtol=1e-4, atol=1e-6, max_num_steps=20000
):
    device, dtype = coords0.device, coords0.dtype
    gc = p.GuideCache(guide_fn, n=n_cache, device=device, dtype=dtype)
    func = p.FusedODEFunc(gc, net).to(device)

    N = coords0.size(0)
    y0 = (coords0, Z0, torch.zeros((N, 1), device=device, dtype=dtype))
    t_eval = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

    xs, qs, ss = odeint(
        func, y0, t_eval,
        method="dopri5",
        rtol=rtol, atol=atol,
        options={"max_num_steps": max_num_steps}
    )

    ms = lib0.view(1, -1, 1) * torch.exp(ss).clamp_min(1e-12)

    trace = {
        "t": t_eval.detach().cpu().numpy(),
        "x": [xs[i] for i in range(xs.size(0))],
        "m": [ms[i].view(-1) for i in range(ms.size(0))],
    }
    return trace


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
    net = out["net"]
    coords0 = out["coords0"]
    Z0 = out["Z0"]
    lib0 = out["lib0"]

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
        if ns is None:
            raise ValueError("unnormalize=True but out['norm_stats'] is missing")

        # 优先 3D
        if ("xyz_mu" in ns) and ("xyz_s" in ns):
            xyz_mu = ns["xyz_mu"].to(device=device, dtype=dtype)
            xyz_s = ns["xyz_s"].to(device=device, dtype=dtype)
            xs = xs * xyz_s + xyz_mu
        # 兼容旧 2D
        elif ("xy_mu" in ns) and ("xy_s" in ns):
            xy_mu = ns["xy_mu"].to(device=device, dtype=dtype)
            xy_s = ns["xy_s"].to(device=device, dtype=dtype)
            xs = xs * xy_s + xy_mu
        else:
            raise KeyError("norm_stats must contain xyz_mu/xyz_s or xy_mu/xy_s")

        f_mu = ns["f_mu"].to(device=device, dtype=dtype)
        f_std = ns["f_std"].to(device=device, dtype=dtype)
        qs = qs * f_std + f_mu

    trace = {
        "t": t_eval.detach().cpu().numpy(),
        "x": [xs[i].detach().cpu() for i in range(xs.size(0))],
        "q": [qs[i].detach().cpu() for i in range(qs.size(0))],
        "s": [ss[i].detach().cpu().view(-1) for i in range(ss.size(0))],
        "m": [ms[i].detach().cpu().view(-1) for i in range(ms.size(0))],
    }
    return trace