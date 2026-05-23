import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.traj_ana import rollout_trace_from_out
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

def _normalize_xy_with_out(out, xy_raw_np, *, device="cpu", dtype=torch.float32):
    ns = out["norm_stats"]
    xy_mu = ns["xy_mu"].to(device=device, dtype=dtype)   # (1,2)
    xy_s  = ns["xy_s"].to(device=device, dtype=dtype)    # scalar
    xy = torch.tensor(xy_raw_np, device=device, dtype=dtype)
    xy = (xy - xy_mu) / (xy_s + 1e-12)
    return xy.detach().cpu().numpy()


def _subset_adata(adata_all, sid, sample_key="sample"):
    return adata_all[adata_all.obs[sample_key].astype(str) == str(sid)].copy()


def _pick(n, n_show, seed):
    if n <= n_show:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=n_show, replace=False)


def build_label_palette(adata_all, ann_key="leiden", fallback_cmap="tab20"):
    obs = adata_all.obs[ann_key]
    if pd.api.types.is_categorical_dtype(obs):
        cats = obs.cat.categories.astype(str).to_numpy()
    else:
        cats = np.unique(obs.astype(str).values)
        try:
            cats = np.array(sorted(cats, key=lambda x: int(x)))
        except Exception:
            cats = np.array(sorted(cats))

    ck = f"{ann_key}_colors"
    if ck in adata_all.uns and len(adata_all.uns[ck]) >= len(cats):
        colors = list(adata_all.uns[ck])[:len(cats)]
    else:
        base = plt.get_cmap(fallback_cmap)
        colors = [base(i % base.N) for i in range(len(cats))]

    palette = {lab: plt.matplotlib.colors.to_rgba(col) for lab, col in zip(cats, colors)}
    unknown = (0.6, 0.6, 0.6, 1.0)
    return palette, unknown, cats


def labels_to_rgba(labels, palette, unknown):
    return np.array([palette.get(str(l), unknown) for l in labels], dtype=float)


# ============================================================
# (A) one chosen frame: even(real) / virtual(chosen) / odd(real)
# ============================================================
def plot_oneframe(
    res, adata_all, *,
    route_ids,                 # now expects [head, tail]
    real_mid_id=None,          # e.g. "U22"; if None, try to infer midpoint like U22
    t_mid=0.5, frame_idx=None,
    steps=10, n_cache=256,
    sample_key="sample",
    ann_key="leiden",
    x_key="cx_aligned", y_key="cy_aligned",
    n_show=20000, s=2, alpha=0.85, seed=0,
    save_path=None, transparent=False,
):

    palette, unknown, _ = build_label_palette(adata_all, ann_key=ann_key)

    route_ids = [str(x) for x in route_ids]
    if len(route_ids) != 2:
        raise ValueError("plot_oneframe expects route_ids=[head, tail], e.g. ['U12','U32'].")

    head, tail = route_ids[0], route_ids[1]
    any_out = next(iter(res.values()))

    def _find_seg_key(a, b):
        k1 = f"{a}_to_{b}"
        if k1 in res: return k1
        a2 = "".join([c for c in str(a) if c.isdigit()])
        b2 = "".join([c for c in str(b) if c.isdigit()])
        k2 = f"{a2}_to_{b2}"
        if k2 in res: return k2
        return None

    def _infer_mid_id(a, b):
        # infer like U12 + U32 -> U22 (only if exists in adata)
        def split_prefix_num(s):
            s = str(s)
            pref = "".join([c for c in s if not c.isdigit()])
            num  = "".join([c for c in s if c.isdigit()])
            return pref, (int(num) if num != "" else None)

        pa, na = split_prefix_num(a)
        pb, nb = split_prefix_num(b)
        if na is None or nb is None or pa != pb:
            return None
        mid = int(round((na + nb) / 2))
        cand = f"{pa}{mid}"
        if cand in set(adata_all.obs[sample_key].astype(str).values):
            return cand
        return None

    if real_mid_id is None:
        real_mid_id = _infer_mid_id(head, tail)
    if real_mid_id is not None:
        real_mid_id = str(real_mid_id)

    seg_key = _find_seg_key(head, tail)
    if seg_key is None:
        raise KeyError(f"cannot find segment key for {head}->{tail} in res")

    fig, axes = plt.subplots(3, 1, figsize=(4.2, 10.2), dpi=150)
    axes = np.array(axes).reshape(3, 1)

    # -------- head real (row0) --------
    ad_head = _subset_adata(adata_all, head, sample_key=sample_key)
    xy_head_raw = np.c_[ad_head.obs[x_key].to_numpy(),
                        ad_head.obs[y_key].to_numpy()].astype(np.float32)
    xy_head = _normalize_xy_with_out(any_out, xy_head_raw, device="cpu")

    lab_head = ad_head.obs[ann_key].astype(str).values
    col_head = labels_to_rgba(lab_head, palette, unknown)

    idx_h = _pick(xy_head.shape[0], n_show, seed + 1)
    xh, ch = xy_head[idx_h], col_head[idx_h]

    # -------- virtual chosen (row1) --------
    out = res[seg_key]
    trace = rollout_trace_from_out(out, steps=steps, n_cache=n_cache, unnormalize=False)
    t = np.asarray(trace["t"])

    if frame_idx is None:
        mid_i = int(np.argmin(np.abs(t - float(t_mid))))
    else:
        mid_i = max(0, min(int(frame_idx), len(trace["x"]) - 1))

    x_mid = trace["x"][mid_i]
    if torch.is_tensor(x_mid):
        x_mid = x_mid.detach().cpu().numpy()

    idx_v = _pick(x_mid.shape[0], n_show, seed + 2)
    xv = x_mid[idx_v]
    if x_mid.shape[0] == len(col_head):
        cv = col_head[idx_v]
    else:
        cv = np.tile(np.array(unknown)[None, :], (len(idx_v), 1))

    # -------- real mid (row2) optional --------
    xm = cm = None
    if real_mid_id is not None:
        ad_mid = _subset_adata(adata_all, real_mid_id, sample_key=sample_key)
        xy_mid_raw = np.c_[ad_mid.obs[x_key].to_numpy(),
                           ad_mid.obs[y_key].to_numpy()].astype(np.float32)
        xy_mid = _normalize_xy_with_out(any_out, xy_mid_raw, device="cpu")
        lab_mid = ad_mid.obs[ann_key].astype(str).values
        col_mid = labels_to_rgba(lab_mid, palette, unknown)
        idx_m = _pick(xy_mid.shape[0], n_show, seed + 3)
        xm, cm = xy_mid[idx_m], col_mid[idx_m]

    # -------- shared limits --------
    stacks = [xh, xv] + ([xm] if xm is not None else [])
    all_xy = np.concatenate(stacks, axis=0)
    mn, mx = all_xy.min(0), all_xy.max(0)
    pad = 0.03 * (mx - mn + 1e-12)
    xlim = (mn[0]-pad[0], mx[0]+pad[0])
    ylim = (mn[1]-pad[1], mx[1]+pad[1])

    ax0, ax1, ax2 = axes[0, 0], axes[1, 0], axes[2, 0]

    ax0.scatter(xh[:,0], xh[:,1], c=ch, s=s, alpha=alpha, linewidths=0)
    ax0.set_title(f"head real\nsample {head}", fontsize=10)

    ax1.scatter(xv[:,0], xv[:,1], c=cv, s=s, alpha=alpha, linewidths=0)
    ax1.set_title(f"virtual\n{seg_key}\nt≈{t[mid_i]:.2f}", fontsize=10)

    if xm is not None:
        ax2.scatter(xm[:,0], xm[:,1], c=cm, s=s, alpha=alpha, linewidths=0)
        ax2.set_title(f"mid real\nsample {real_mid_id}", fontsize=10)
    else:
        ax2.set_title("mid real\nN/A", fontsize=10)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Annotation-colored (ann_key={ann_key})", y=1.01, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=transparent)
        plt.close(fig)
        print("[OK] saved:", str(save_path))

    return fig

# ============================================================
# (B) all frames for all current results (segments)
# Each column = one segment (even_i -> even_{i+1})
# Rows = even(real) + all rollout frames + odd(real)
# ============================================================
def plot_allframes(
    res, adata_all, *,
    route_ids,                 # real chain like ['U2','U12','U22','U32']
    layout="vertical",
    steps=10, n_cache=256,
    sample_key="sample",
    ann_key="leiden",
    x_key="cx_aligned", y_key="cy_aligned",
    n_show=20000, s=2, alpha=0.85, seed=0,
    save_path=None, transparent=False,
):

    palette, unknown, _ = build_label_palette(adata_all, ann_key=ann_key)

    route_ids = [str(x) for x in route_ids]
    K = len(route_ids)
    if K < 2:
        raise ValueError("route_ids must have at least 2 ids, e.g. ['U2','U12'].")

    n_cols = K - 1
    n_frames = steps + 1
    n_rows = 1 + n_frames + 1   # start real + all virtual frames + end real

    if layout == "vertical":
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6*n_cols, 2.7*n_rows), dpi=150)
        if n_cols == 1:
            axes = np.array(axes).reshape(n_rows, 1)
        def AX(ti, si): 
            return axes[ti, si]
    else:
        fig, axes = plt.subplots(n_cols, n_rows, figsize=(2.7*n_rows, 3.6*n_cols), dpi=150)
        if n_cols == 1:
            axes = np.array(axes).reshape(1, n_rows)   
        def AX(ti, si):
            return axes[si, ti]


    any_out = next(iter(res.values()))

    def _find_seg_key(a, b):
        k1 = f"{a}_to_{b}"
        if k1 in res: return k1
        a2 = "".join([c for c in str(a) if c.isdigit()])
        b2 = "".join([c for c in str(b) if c.isdigit()])
        k2 = f"{a2}_to_{b2}"
        if k2 in res: return k2
        return None

    for col in range(n_cols):
        sid_start = route_ids[col]
        sid_end   = route_ids[col + 1]

        seg_key = _find_seg_key(sid_start, sid_end)
        if seg_key is None:
            for r in range(n_rows):
                AX(r, col).axis("off")
            AX(0, col).set_title(f"{sid_start}→{sid_end}\n(missing)", fontsize=10)
            continue

        # -------- start real --------
        ad_s = _subset_adata(adata_all, sid_start, sample_key=sample_key)
        xy_s_raw = np.c_[ad_s.obs[x_key].to_numpy(),
                         ad_s.obs[y_key].to_numpy()].astype(np.float32)
        xy_s = _normalize_xy_with_out(any_out, xy_s_raw, device="cpu")
        lab_s = ad_s.obs[ann_key].astype(str).values
        col_s = labels_to_rgba(lab_s, palette, unknown)

        # -------- end real --------
        ad_e = _subset_adata(adata_all, sid_end, sample_key=sample_key)
        xy_e_raw = np.c_[ad_e.obs[x_key].to_numpy(),
                         ad_e.obs[y_key].to_numpy()].astype(np.float32)
        xy_e = _normalize_xy_with_out(any_out, xy_e_raw, device="cpu")
        lab_e = ad_e.obs[ann_key].astype(str).values
        col_e = labels_to_rgba(lab_e, palette, unknown)

        # -------- trace --------
        out = res[seg_key]
        trace = rollout_trace_from_out(out, steps=steps, n_cache=n_cache, unnormalize=False)
        t = np.asarray(trace["t"])

        x0 = trace["x"][0]
        if torch.is_tensor(x0):
            N0 = int(x0.shape[0])
        else:
            N0 = int(np.asarray(x0).shape[0])

        idx = _pick(N0, n_show, seed + 1000 + col)

        # virtual colors inherit from start if possible else gray
        if len(col_s) == N0:
            c_virtual = col_s[idx]
        else:
            c_virtual = np.tile(np.array(unknown)[None, :], (len(idx), 1))

        idx_s = _pick(xy_s.shape[0], n_show, seed + 2000 + col)
        idx_e = _pick(xy_e.shape[0], n_show, seed + 3000 + col)

        xs, cs = xy_s[idx_s], col_s[idx_s]
        xe, ce = xy_e[idx_e], col_e[idx_e]

        # -------- limits per column --------
        mins, maxs = [], []
        mins.append(xs.min(0)); maxs.append(xs.max(0))
        for i in range(n_frames):
            xi = trace["x"][i]
            if torch.is_tensor(xi):
                xi = xi.detach().cpu().numpy()
            xi = np.asarray(xi)[idx]
            mins.append(xi.min(0)); maxs.append(xi.max(0))
        mins.append(xe.min(0)); maxs.append(xe.max(0))

        mn = np.min(np.stack(mins, 0), 0)
        mx = np.max(np.stack(maxs, 0), 0)
        pad = 0.03 * (mx - mn + 1e-12)
        xlim = (mn[0]-pad[0], mx[0]+pad[0])
        ylim = (mn[1]-pad[1], mx[1]+pad[1])

        # row0: start real
        # ax = axes[0, col]
        ax = AX(0, col)
        ax.scatter(xs[:,0], xs[:,1], c=cs, s=s, alpha=alpha, linewidths=0)
        ax.set_title(f"{seg_key}\nstart real {sid_start}", fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # rows 1..n_frames: all virtual frames
        for i in range(n_frames):
            # ax = axes[1 + i, col]
            ax = AX(1 + i, col)
            xi = trace["x"][i]
            if torch.is_tensor(xi):
                xi = xi.detach().cpu().numpy()
            xi = np.asarray(xi)[idx]
            ax.scatter(xi[:,0], xi[:,1], c=c_virtual, s=s, alpha=alpha, linewidths=0)
            ax.set_title(f"virtual t={t[i]:.2f} (i={i})", fontsize=9)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

        # last row: end real
        # ax = axes[-1, col]
        ax = AX(n_rows - 1, col)
        ax.scatter(xe[:,0], xe[:,1], c=ce, s=s, alpha=alpha, linewidths=0)
        ax.set_title(f"end real {sid_end}", fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Annotation-colored segments (ann_key={ann_key})", y=1.002, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=transparent)
        plt.close(fig)
        print("[OK] saved:", str(save_path))

    return fig


def plot_spatial_stack_3d( adata, sample_key="sample", spatial_key="spatial", *, z_step=1.0,
    max_points_per_sample=8000, s=2, alpha=0.75, elev=25, azim=-60, seed=0, 
    out_png="spatial_stack_3d.png", transparent=True,
):
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm missing {spatial_key}")
    xy = np.asarray(adata.obsm[spatial_key])
    if xy.shape[1] < 2:
        raise ValueError(f"{spatial_key} invalid shape:{xy.shape},requires (n,2)")

    if sample_key not in adata.obs:
        raise KeyError(f"adata.obs missing {sample_key}")
    samp = adata.obs[sample_key].astype(str).to_numpy()

    uniq = np.unique(samp)
    def _sort_key(x):
        return (0, int(x)) if x.isdigit() else (1, x)
    uniq = sorted(list(uniq), key=_sort_key)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab20")
    for i, sid in enumerate(uniq):
        idx = np.where(samp == sid)[0]
        if max_points_per_sample is not None and len(idx) > max_points_per_sample:
            rng = np.random.default_rng(seed + i)
            idx = rng.choice(idx, size=max_points_per_sample, replace=False)

        x = xy[idx, 0]
        y = xy[idx, 1]
        z = np.full_like(x, fill_value=i * z_step, dtype=float)

        ax.scatter(
            x, y, z,
            s=s, alpha=alpha,
            color=cmap(i % 20), depthshade=False, label=sid,
        )

    ax.set_xlabel("spatial_x")
    ax.set_ylabel("spatial_y")
    ax.set_zlabel(sample_key)

    zticks = [i * z_step for i in range(len(uniq))]
    ax.set_zticks(zticks)
    ax.set_zticklabels(uniq)

    ax.view_init(elev=elev, azim=azim)
    ax.legend( bbox_to_anchor=(1.02, 1.0), loc="upper left", title=sample_key,
        markerscale=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, transparent=transparent, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png} | samples: {len(uniq)} | points: {adata.n_obs}")


#-------------------------------------
#             Prolifer
#-------------------------------------

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def _normalize_xy_with_ctx(ctx, xy_raw_np):
    xy_mu = _to_numpy(ctx.xy_mu).astype(np.float32)   
    xy_s  = _to_float(ctx.xy_s)

    xy_mu = xy_mu.reshape(1, 2) if xy_mu.size == 2 else xy_mu
    return ((xy_raw_np.astype(np.float32) - xy_mu) / (xy_s + 1e-12)).astype(np.float32)

def plot_oneframe_policy(
    rollouts, adata_all, ctx, *,
    route_ids,                 # [head, tail] e.g. ["U12","U22"]
    real_mid_id=None,          # optional real mid sample id
    t_mid=0.5, frame_idx=None,
    sample_key="sample",
    ann_key="His_anno",
    x_key="cx_aligned", y_key="cy_aligned",
    n_show=20000, s=2, alpha=0.85, seed=0,
    show_birth=False,          # ✅ whether overlay birth points
    birth_scale=6.0,           # marker scale for birth points
    birth_edgecolor="r",
    birth_lw=0.6,
    birth_facecolors="none",   # "none" -> hollow circle; None -> filled
    layout="vertical",         # ✅ "vertical" (3x1) or "horizontal" (1x3)
    save_path=None, transparent=False,
):
    def _pick(n, k, sd):
        n = int(n)
        if k is None or k <= 0 or n <= k:
            return np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(int(sd))
        return rng.choice(n, size=int(k), replace=False).astype(np.int64)

    def _find_seg_key(a, b):
        k1 = f"{a}_to_{b}"
        if k1 in rollouts: return k1
        a2 = "".join([c for c in str(a) if c.isdigit()])
        b2 = "".join([c for c in str(b) if c.isdigit()])
        k2 = f"{a2}_to_{b2}"
        if k2 in rollouts: return k2
        return None

    def _infer_mid_id(a, b):
        def split_prefix_num(s):
            s = str(s)
            pref = "".join([c for c in s if not c.isdigit()])
            num  = "".join([c for c in s if c.isdigit()])
            return pref, (int(num) if num != "" else None)
        pa, na = split_prefix_num(a)
        pb, nb = split_prefix_num(b)
        if na is None or nb is None or pa != pb:
            return None
        mid = int(round((na + nb) / 2))
        cand = f"{pa}{mid}"
        if cand in set(adata_all.obs[sample_key].astype(str).values):
            return cand
        return None

    def _colors_from_layer_idx(layers_idx, palette, unknown):
        layers_idx = np.asarray(layers_idx).astype(np.int64)
        L = np.asarray(ctx.layers_list).astype(str)
        unk_lab = "__UNKNOWN__"
        lab = np.full(layers_idx.shape, unk_lab, dtype=object)
        m = (layers_idx >= 0) & (layers_idx < len(L))
        lab[m] = L[layers_idx[m]]
        return labels_to_rgba(lab.astype(str), palette, unknown)

    route_ids = [str(x) for x in route_ids]
    if len(route_ids) != 2:
        raise ValueError("plot_oneframe_policy expects route_ids=[head, tail], e.g. ['U12','U22'].")

    head, tail = route_ids
    seg_key = _find_seg_key(head, tail)
    if seg_key is None:
        raise KeyError(f"cannot find rollout segment for {head}->{tail}")

    if real_mid_id is None:
        real_mid_id = _infer_mid_id(head, tail)
    if real_mid_id is not None:
        real_mid_id = str(real_mid_id)

    palette, unknown, _ = build_label_palette(adata_all, ann_key=ann_key)

    ro = rollouts[seg_key]
    coords_seq = ro["coords"]
    layers_seq = ro["layers"]
    Tp1 = len(coords_seq)

    # choose mid frame
    if frame_idx is None:
        if "t" in ro:
            t_arr = np.asarray(ro["t"], dtype=float)
            mid_i = int(np.argmin(np.abs(t_arr - float(t_mid))))
        else:
            mid_i = int(round(float(t_mid) * (Tp1 - 1)))
            mid_i = max(0, min(mid_i, Tp1 - 1))
    else:
        mid_i = max(0, min(int(frame_idx), Tp1 - 1))

    # layout
    layout = str(layout).lower()
    if layout not in ("vertical", "horizontal"):
        raise ValueError("layout must be 'vertical' or 'horizontal'")
    if layout == "vertical":
        fig, axes = plt.subplots(3, 1, figsize=(4.2, 10.2), dpi=150)
        ax0, ax1, ax2 = np.array(axes).reshape(3,)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=150)
        ax0, ax1, ax2 = np.array(axes).reshape(3,)

    # -------- head real --------
    ad_head = _subset_adata(adata_all, head, sample_key=sample_key)
    xy_head_raw = np.c_[ad_head.obs[x_key].to_numpy(),
                        ad_head.obs[y_key].to_numpy()].astype(np.float32)
    xy_head = _normalize_xy_with_ctx(ctx, xy_head_raw)
    lab_head = ad_head.obs[ann_key].astype(str).values
    col_head = labels_to_rgba(lab_head, palette, unknown)
    idx_h = _pick(xy_head.shape[0], n_show, seed + 1)
    xh, ch = xy_head[idx_h], col_head[idx_h]

    # -------- virtual chosen --------
    x_mid = np.asarray(coords_seq[mid_i], dtype=np.float32)
    l_mid = np.asarray(layers_seq[mid_i], dtype=np.int64)
    idx_v = _pick(x_mid.shape[0], n_show, seed + 2)
    xv = x_mid[idx_v]
    cv = _colors_from_layer_idx(l_mid[idx_v], palette, unknown)

    # -------- real mid optional --------
    xm = cm = None
    if real_mid_id is not None:
        ad_mid = _subset_adata(adata_all, real_mid_id, sample_key=sample_key)
        xy_mid_raw = np.c_[ad_mid.obs[x_key].to_numpy(),
                           ad_mid.obs[y_key].to_numpy()].astype(np.float32)
        xy_mid = _normalize_xy_with_ctx(ctx, xy_mid_raw)
        lab_mid = ad_mid.obs[ann_key].astype(str).values
        col_mid = labels_to_rgba(lab_mid, palette, unknown)
        idx_m = _pick(xy_mid.shape[0], n_show, seed + 3)
        xm, cm = xy_mid[idx_m], col_mid[idx_m]

    # -------- shared limits --------
    stacks = [xh, xv] + ([xm] if xm is not None else [])
    all_xy = np.concatenate(stacks, axis=0)
    mn, mx = all_xy.min(0), all_xy.max(0)
    pad = 0.03 * (mx - mn + 1e-12)
    xlim = (mn[0]-pad[0], mx[0]+pad[0])
    ylim = (mn[1]-pad[1], mx[1]+pad[1])

    # row0
    ax0.scatter(xh[:, 0], xh[:, 1], c=ch, s=s, alpha=alpha, linewidths=0)
    ax0.set_title(f"head real\nsample {head}", fontsize=10)

    # row1
    ax1.scatter(xv[:, 0], xv[:, 1], c=cv, s=s, alpha=alpha, linewidths=0)

    birth_n = None
    if show_birth and ("is_birth" in ro):
        isb_full = np.asarray(ro["is_birth"][mid_i]).astype(bool)
        isb = isb_full[idx_v]
        birth_n = int(isb.sum())
        if birth_n > 0:
            kw = dict(
                s=float(s) * float(birth_scale),
                edgecolors=birth_edgecolor,
                linewidths=float(birth_lw),
                alpha=0.95,
            )
            if birth_facecolors is not None:
                kw["facecolors"] = birth_facecolors
            ax1.scatter(xv[isb, 0], xv[isb, 1], **kw)

    # title virtual
    if "t" in ro:
        t_show = float(np.asarray(ro["t"])[mid_i])
        base_title = f"virtual policy\n{seg_key}\nt≈{t_show:.2f} (i={mid_i})"
    else:
        base_title = f"virtual policy\n{seg_key}\nframe i={mid_i}/{Tp1-1}"
    if show_birth and birth_n is not None:
        base_title += f"\n+b={birth_n}"
    ax1.set_title(base_title, fontsize=10)

    # row2
    if xm is not None:
        ax2.scatter(xm[:, 0], xm[:, 1], c=cm, s=s, alpha=alpha, linewidths=0)
        ax2.set_title(f"mid real\nsample {real_mid_id}", fontsize=10)
    else:
        ax2.set_title("mid real\nN/A", fontsize=10)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Policy rollout colored by {ann_key}", y=1.01 if layout=="vertical" else 1.02, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=transparent)
        plt.close(fig)
        print("[OK] saved:", str(save_path))

    return fig


def plot_allframes_policy(
    rollouts, adata_all, ctx, *,
    route_ids,                 # chain like ['U2','U12','U22','U32']
    steps=None,                # if None -> infer max Tp1 across segments; else enforce steps+1 virtual frames
    sample_key="sample",
    ann_key="His_anno",
    x_key="cx_aligned", y_key="cy_aligned",
    n_show=20000, s=5, alpha=0.85, seed=0,
    show_birth=False,
    birth_scale=6.0,
    birth_edgecolor="r",
    birth_lw=0.6,
    birth_facecolors="none",
    layout="vertical",         # ✅ "vertical": time on rows (old style). "horizontal": time on cols.
    save_path=None, transparent=False,
):
    def _pick(n, k, sd):
        n = int(n)
        if k is None or k <= 0 or n <= k:
            return np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(int(sd))
        return rng.choice(n, size=int(k), replace=False).astype(np.int64)

    def _find_seg_key(a, b):
        k1 = f"{a}_to_{b}"
        if k1 in rollouts: return k1
        a2 = "".join([c for c in str(a) if c.isdigit()])
        b2 = "".join([c for c in str(b) if c.isdigit()])
        k2 = f"{a2}_to_{b2}"
        if k2 in rollouts: return k2
        return None

    def _colors_from_layer_idx(layers_idx, palette, unknown):
        layers_idx = np.asarray(layers_idx).astype(np.int64)
        L = np.asarray(ctx.layers_list).astype(str)
        unk_lab = "__UNKNOWN__"
        lab = np.full(layers_idx.shape, unk_lab, dtype=object)
        m = (layers_idx >= 0) & (layers_idx < len(L))
        lab[m] = L[layers_idx[m]]
        return labels_to_rgba(lab.astype(str), palette, unknown)

    palette, unknown, _ = build_label_palette(adata_all, ann_key=ann_key)

    route_ids = [str(x) for x in route_ids]
    K = len(route_ids)
    if K < 2:
        raise ValueError("route_ids must have at least 2 ids, e.g. ['U2','U12'].")

    n_seg = K - 1  # segments count

    # virtual frames count
    if steps is not None:
        n_frames = int(steps) + 1
    else:
        tps = []
        for i in range(n_seg):
            seg_key = _find_seg_key(route_ids[i], route_ids[i+1])
            if seg_key is None:
                continue
            tps.append(len(rollouts[seg_key]["coords"]))
        n_frames = max(tps) if len(tps) > 0 else 1

    layout = str(layout).lower()
    if layout not in ("vertical", "horizontal"):
        raise ValueError("layout must be 'vertical' or 'horizontal'")

    # vertical (old): rows = start + frames + end ; cols = segments
    # horizontal (new): rows = segments ; cols = start + frames + end
    if layout == "vertical":
        n_rows = 1 + n_frames + 1
        n_cols = n_seg
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6*n_cols, 2.7*n_rows), dpi=150)
        if n_cols == 1:
            axes = np.array(axes).reshape(n_rows, 1)

        def _ax(r, c): return axes[r, c]
    else:
        n_rows = n_seg
        n_cols = 1 + n_frames + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7*n_cols, 3.6*n_rows), dpi=150)
        if n_rows == 1:
            axes = np.array(axes).reshape(1, n_cols)

        def _ax(r, c): return axes[r, c]

    for seg_i in range(n_seg):
        sid_start = route_ids[seg_i]
        sid_end   = route_ids[seg_i + 1]
        seg_key = _find_seg_key(sid_start, sid_end)

        # choose drawing row/col index
        if layout == "vertical":
            col = seg_i
            row_base = 0
        else:
            row = seg_i
            col_base = 0

        if seg_key is None:
            # turn off all axes in this segment slot
            if layout == "vertical":
                for r in range(n_rows):
                    _ax(r, col).axis("off")
                _ax(0, col).set_title(f"{sid_start}→{sid_end}\n(missing)", fontsize=10)
            else:
                for c in range(n_cols):
                    _ax(row, c).axis("off")
                _ax(row, 0).set_title(f"{sid_start}→{sid_end}\n(missing)", fontsize=10)
            continue

        ro = rollouts[seg_key]
        coords_seq = ro["coords"]
        layers_seq = ro["layers"]
        Tp1 = len(coords_seq)

        # ----- start/end real -----
        ad_s = _subset_adata(adata_all, sid_start, sample_key=sample_key)
        xy_s_raw = np.c_[ad_s.obs[x_key].to_numpy(), ad_s.obs[y_key].to_numpy()].astype(np.float32)
        xy_s = _normalize_xy_with_ctx(ctx, xy_s_raw)
        lab_s = ad_s.obs[ann_key].astype(str).values
        col_s = labels_to_rgba(lab_s, palette, unknown)

        ad_e = _subset_adata(adata_all, sid_end, sample_key=sample_key)
        xy_e_raw = np.c_[ad_e.obs[x_key].to_numpy(), ad_e.obs[y_key].to_numpy()].astype(np.float32)
        xy_e = _normalize_xy_with_ctx(ctx, xy_e_raw)
        lab_e = ad_e.obs[ann_key].astype(str).values
        col_e = labels_to_rgba(lab_e, palette, unknown)

        # ----- limits per segment -----
        mins, maxs = [], []
        idx_s = _pick(xy_s.shape[0], n_show, seed + 2000 + seg_i)
        xs = xy_s[idx_s]
        mins.append(xs.min(0)); maxs.append(xs.max(0))

        for i in range(n_frames):
            j = min(i, Tp1 - 1)
            xi = np.asarray(coords_seq[j], dtype=np.float32)
            if xi.shape[0] == 0:
                continue
            idx_i = _pick(xi.shape[0], n_show, seed + 10000 + seg_i*100 + i)
            xii = xi[idx_i]
            mins.append(xii.min(0)); maxs.append(xii.max(0))

        idx_e = _pick(xy_e.shape[0], n_show, seed + 3000 + seg_i)
        xe = xy_e[idx_e]
        mins.append(xe.min(0)); maxs.append(xe.max(0))

        mn = np.min(np.stack(mins, 0), 0)
        mx = np.max(np.stack(maxs, 0), 0)
        pad = 0.03 * (mx - mn + 1e-12)
        xlim = (mn[0]-pad[0], mx[0]+pad[0])
        ylim = (mn[1]-pad[1], mx[1]+pad[1])

        # ----- draw start real -----
        if layout == "vertical":
            ax = _ax(0, col)
        else:
            ax = _ax(row, 0)
        cs = col_s[idx_s]
        ax.scatter(xs[:, 0], xs[:, 1], c=cs, s=s, alpha=alpha, linewidths=0)
        ax.set_title(f"{seg_key}\nstart {sid_start}", fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # ----- draw virtual frames -----
        for i in range(n_frames):
            j = min(i, Tp1 - 1)
            xi = np.asarray(coords_seq[j], dtype=np.float32)
            li = np.asarray(layers_seq[j], dtype=np.int64)

            if layout == "vertical":
                ax = _ax(1 + i, col)
            else:
                ax = _ax(row, 1 + i)

            if xi.shape[0] == 0:
                ax.set_title("virtual (empty)", fontsize=9)
                ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            idx_i = _pick(xi.shape[0], n_show, seed + 10000 + seg_i*100 + i)
            xii = xi[idx_i]
            lii = li[idx_i] # here lii  layer 
            cii = _colors_from_layer_idx(lii, palette, unknown)

            # ---  ---
            #  label( "Red_Type")
            #  cii .
            # :
            # here priority_mask ,/ True
            # , layer 
            sort_idx = np.argsort(lii) #  layer 
            # :
            target_layer_idx = 3
            sort_idx = np.argsort(lii == target_layer_idx) 

            xii = xii[sort_idx]
            cii = cii[sort_idx]
            # ------------------

            ax.scatter(xii[:, 0], xii[:, 1], c=cii, s=s, alpha=alpha, linewidths=0)

            birth_n = None
            if show_birth and ("is_birth" in ro):
                isb = np.asarray(ro["is_birth"][j]).astype(bool)
                isb = isb[idx_i]
                birth_n = int(isb.sum())
                if birth_n > 0:
                    kw = dict(
                        s=float(s) * float(birth_scale),
                        edgecolors=birth_edgecolor,
                        linewidths=float(birth_lw),
                        alpha=0.95,
                    )
                    if birth_facecolors is not None:
                        kw["facecolors"] = birth_facecolors
                    ax.scatter(xii[isb, 0], xii[isb, 1], **kw)

            if "t" in ro:
                t_show = float(np.asarray(ro["t"])[j])
                tt = f"t={t_show:.2f} (i={j})"
            else:
                tt = f"i={j}"
            if show_birth and birth_n is not None:
                tt += f" +b={birth_n}"
            ax.set_title(tt, fontsize=9)

            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

        # ----- draw end real -----
        if layout == "vertical":
            ax = _ax(n_rows - 1, col)
        else:
            ax = _ax(row, n_cols - 1)
        ce = col_e[idx_e]
        ax.scatter(xe[:, 0], xe[:, 1], c=ce, s=s, alpha=alpha, linewidths=0)
        ax.set_title(f"end {sid_end}", fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Policy rollouts colored by {ann_key}", y=1.002 if layout=="vertical" else 1.01, fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", transparent=transparent)
        plt.close(fig)
        print("[OK] saved:", str(save_path))

    return fig
