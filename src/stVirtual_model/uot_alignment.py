import numpy as np
import pandas as pd
import ot
from tqdm import trange
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# ============================================================
# 0) Rigid utilities (row-vector convention): X' = X @ R + t
# ============================================================

def _theta_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _barycentric_map(P: np.ndarray, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    P: (ns, nt) transport plan
    Xt: (nt, d)
    return:
      Y: (ns, d) barycentric mapped points
      w: (ns,) row mass (useful as weights)
    """
    row_sum = P.sum(1)  # (ns,)
    denom = np.maximum(row_sum[:, None], 1e-12)
    Y = (P @ Xt) / denom
    return Y, row_sum


def _weighted_rigid(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted Procrustes (no scaling), row-vector convention.
    Solve: minimize sum_i w_i || X_i @ R + t - Y_i ||^2
    Returns R (dxd), t (1xd)
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    assert X.shape == Y.shape and X.ndim == 2
    n, d = X.shape

    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, np.float64).reshape(-1)
        w = np.maximum(w, 0.0)

    ws = w.sum()
    if ws < 1e-12:
        # fallback to unweighted
        w = np.ones(n, dtype=np.float64)
        ws = float(n)

    w_norm = w / ws

    muX = (w_norm[:, None] * X).sum(0, keepdims=True)  # (1,d)
    muY = (w_norm[:, None] * Y).sum(0, keepdims=True)

    Xc = X - muX
    Yc = Y - muY

    # Cov for row-vector formulation:
    # want Xc @ R close to Yc
    # objective -> maximize trace(R^T (Xc^T W Yc))
    C = Xc.T @ (w_norm[:, None] * Yc)  # (d,d)

    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt

    # fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = muY - muX @ R  # (1,d)  IMPORTANT: @R (NOT R.T)
    return R, t


def _apply_rigid(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (X @ R) + t


def _compose_rigid(R: np.ndarray, t: np.ndarray, dR: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose transforms under row-vector convention:
      f(X)=X@R + t
      g(X)=X@dR + dt
      g(f(X)) = X@(R@dR) + (t@dR + dt)
    """
    R2 = R @ dR
    t2 = (t @ dR) + dt
    return R2, t2


def _robust_center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return X - X.mean(0, keepdims=True)


# ============================================================
# 1) Unbalanced OT init (more stable + multi-start + trimming)
# ============================================================

def _uot_init_once(
    Xs: np.ndarray,
    Xt: np.ndarray,
    *,
    n_sub: int,
    reg: float | None,
    reg_m: float,
    seed: int,
    trim_q: float = 0.10,   # drop worst residuals when fitting rigid
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    One OT init trial:
      - subsample points
      - compute UOT plan on centered coordinates (translation-invariant cost)
      - barycentric map -> tentative correspondences
      - robust weighted rigid (use OT row mass as weights) + trim outliers
    return (R, t, info)
    """
    rng = np.random.default_rng(seed)
    Ns, Nt = Xs.shape[0], Xt.shape[0]
    ns = min(int(n_sub), Ns)
    nt = min(int(n_sub), Nt)

    idx_s = rng.choice(Ns, size=ns, replace=False)
    idx_t = rng.choice(Nt, size=nt, replace=False)

    Xs_sub = Xs[idx_s].astype(np.float64)
    Xt_sub = Xt[idx_t].astype(np.float64)

    # cost in centered space to reduce translation sensitivity
    Xs_c = _robust_center(Xs_sub)
    Xt_c = _robust_center(Xt_sub)

    M = cdist(Xs_c, Xt_c, metric="sqeuclidean")

    if reg is None:
        # median heuristic
        reg0 = 0.05 * float(np.median(M + 1e-12))
        reg = max(reg0, 1e-8)

    a = np.full(ns, 1.0 / ns, dtype=np.float64)
    b = np.full(nt, 1.0 / nt, dtype=np.float64)

    P = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m)
    Y_match, w_row = _barycentric_map(P, Xt_sub)  # correspondences in original coordinate

    # initial weighted fit
    R0, t0 = _weighted_rigid(Xs_sub, Y_match, w=w_row)

    # robust trimming by residual
    Xw = _apply_rigid(Xs_sub, R0, t0)
    resid = np.sqrt(((Xw - Y_match) ** 2).sum(1))
    if 0.0 < trim_q < 0.5 and len(resid) >= 10:
        thr = np.quantile(resid, 1.0 - trim_q)
        keep = resid <= thr
        if keep.sum() >= max(10, int(0.2 * len(resid))):
            R1, t1 = _weighted_rigid(Xs_sub[keep], Y_match[keep], w=w_row[keep])
            R0, t0 = R1, t1

    # score: mean trimmed residual
    Xw2 = _apply_rigid(Xs_sub, R0, t0)
    resid2 = np.sqrt(((Xw2 - Y_match) ** 2).sum(1))
    score = float(np.mean(np.sort(resid2)[: max(10, int(0.8 * len(resid2))) ]))

    info = dict(
        seed=int(seed),
        reg=float(reg),
        reg_m=float(reg_m),
        score=float(score),
        theta=float(_theta_from_R(R0)),
        t=t0.reshape(-1).tolist(),
        ns=int(ns),
        nt=int(nt),
    )
    return R0, t0, info


def uot_global_rigid_init(
    Xs: np.ndarray,
    Xt: np.ndarray,
    *,
    n_sub: int = 4000,
    reg: float | None = None,
    reg_m: float = 1.0,
    n_starts: int = 6,
    seed: int = 2025,
    trim_q: float = 0.10,
    reg_grid: tuple[float, ...] = (0.5, 1.0, 2.0),
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Multi-start UOT init: try several seeds and reg scalings, pick best score.
    Returns: (R, t, best_info)
    """
    Xs = np.asarray(Xs, np.float64)
    Xt = np.asarray(Xt, np.float64)
    assert Xs.shape[1] == Xt.shape[1] == 2, "This implementation assumes 2D points."

    # if reg not provided, estimate from a quick small sample
    if reg is None:
        rng = np.random.default_rng(seed)
        ns = min(1024, Xs.shape[0])
        nt = min(1024, Xt.shape[0])
        xs = Xs[rng.choice(Xs.shape[0], size=ns, replace=False)]
        xt = Xt[rng.choice(Xt.shape[0], size=nt, replace=False)]
        M = cdist(_robust_center(xs), _robust_center(xt), metric="sqeuclidean")
        reg = max(0.05 * float(np.median(M + 1e-12)), 1e-8)

    best = None
    best_score = np.inf

    base_seed = int(seed)
    total = int(n_starts) * len(reg_grid)
    pbar = tqdm(total=total, disable=(not verbose), desc="UOT init", dynamic_ncols=True)

    tried = 0
    for si in range(int(n_starts)):
        for g in reg_grid:
            tried += 1
            R0, t0, info = _uot_init_once(
                Xs, Xt,
                n_sub=n_sub,
                reg=float(reg) * float(g),
                reg_m=reg_m,
                seed=base_seed + 97 * si + int(13 * g),
                trim_q=trim_q,
            )

            if info["score"] < best_score:
                best_score = info["score"]
                best = (R0, t0, info)

            pbar.set_postfix(
                trial=tried,
                score=f"{info['score']:.4g}",
                best=f"{best_score:.4g}",
                reg=f"{info['reg']:.3g}",
                th=f"{info['theta']:.3g}",
            )

    pbar.close()

    R_best, t_best, info_best = best
    if verbose:
        print(f"[UOT init] BEST score={info_best['score']:.6f} reg={info_best['reg']:.6g} theta={info_best['theta']:.4f} t={info_best['t']}")
    return R_best, t_best, info_best

# ============================================================
# 2) Robust ICP refine (closed-form, trimming, label gating)
# ============================================================

def _build_nn_index(X: np.ndarray, ann: np.ndarray | None, *, max_points_per_class: int | None = 200000, seed: int = 0):
    """
    Build per-class NN and global NN.
    To avoid huge memory/time, can subsample each class for NN index.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)

    global_nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X)

    class_nns = {}
    if ann is None:
        return global_nn, class_nns

    ann = np.asarray(ann)
    uniq = pd.unique(ann)
    for cls in uniq:
        m = (ann == cls)
        Xc = X[m]
        if Xc.shape[0] == 0:
            continue
        if max_points_per_class is not None and Xc.shape[0] > int(max_points_per_class):
            idx = rng.choice(Xc.shape[0], size=int(max_points_per_class), replace=False)
            Xc = Xc[idx]
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Xc)
        class_nns[cls] = (nn, Xc)
    return global_nn, class_nns


def _match_nn(
    Xq: np.ndarray,
    lab_q: np.ndarray | None,
    *,
    global_nn,
    class_nns: dict,
    Xt_full: np.ndarray,
    use_class: bool = True,
) -> np.ndarray:
    """
    For each query point, find nearest neighbor in target.
    If use_class and class exists -> per-class nn, else global.
    """
    Xq = np.asarray(Xq, np.float64)
    B = Xq.shape[0]
    Y = np.empty_like(Xq)

    if lab_q is None or (not use_class) or len(class_nns) == 0:
        _, idx = global_nn.kneighbors(Xq, n_neighbors=1)
        Y[:] = Xt_full[idx[:, 0]]
        return Y

    lab_q = np.asarray(lab_q)
    # batch by class
    for cls in pd.unique(lab_q):
        ii = np.where(lab_q == cls)[0]
        if ii.size == 0:
            continue
        sub = Xq[ii]
        if cls in class_nns:
            nn, Xc = class_nns[cls]
            _, idx = nn.kneighbors(sub, n_neighbors=1)
            Y[ii] = Xc[idx[:, 0]]
        else:
            _, idx = global_nn.kneighbors(sub, n_neighbors=1)
            Y[ii] = Xt_full[idx[:, 0]]
    return Y


def icp_refine_rigid(
    Xs: np.ndarray,
    Xt: np.ndarray,
    ann_s: np.ndarray | None,
    ann_t: np.ndarray | None,
    *,
    R0: np.ndarray,
    t0: np.ndarray,
    n_iter: int = 40,
    sub_size: int = 20000,
    trim_q: float = 0.15,
    use_class_after: int = 10,
    seed: int = 2025,
    verbose: bool = True,
    max_points_per_class_nn: int | None = 200000,
) -> tuple[np.ndarray, np.ndarray, dict]:

    rng = np.random.default_rng(seed)

    Xs = np.asarray(Xs, np.float64)
    Xt = np.asarray(Xt, np.float64)
    if ann_s is not None:
        ann_s = np.asarray(ann_s)
    if ann_t is not None:
        ann_t = np.asarray(ann_t)

    global_nn, class_nns = _build_nn_index(Xt, ann_t, max_points_per_class=max_points_per_class_nn, seed=seed)

    R = np.asarray(R0, np.float64).copy()
    t = np.asarray(t0, np.float64).reshape(1, -1).copy()

    Ns = Xs.shape[0]
    history = []

    pbar = trange(int(n_iter), disable=(not verbose), desc="ICP refine", dynamic_ncols=True)

    for it in pbar:
        # sample queries
        if sub_size is None or sub_size <= 0 or sub_size >= Ns:
            idx = np.arange(Ns)
        else:
            idx = rng.choice(Ns, size=int(sub_size), replace=False)

        Xq = Xs[idx]
        lab_q = ann_s[idx] if ann_s is not None else None

        Xw = _apply_rigid(Xq, R, t)

        use_class = (it >= int(use_class_after))
        Y = _match_nn(Xw, lab_q, global_nn=global_nn, class_nns=class_nns, Xt_full=Xt, use_class=use_class)

        # residual + trimming
        resid = np.sqrt(((Xw - Y) ** 2).sum(1))
        if 0.0 < trim_q < 0.5 and resid.size >= 10:
            thr = np.quantile(resid, 1.0 - trim_q)
            keep = resid <= thr
        else:
            keep = np.ones_like(resid, dtype=bool)

        Xw_k = Xw[keep]
        Y_k = Y[keep]

        # fit delta rigid from Xw -> Y
        dR, dt = _weighted_rigid(Xw_k, Y_k, w=None)

        # update (compose): X -> X@R + t; then -> (... )@dR + dt
        R, t = _compose_rigid(R, t, dR, dt)

        # diagnostics
        Xw2 = _apply_rigid(Xq, R, t)
        Y2 = _match_nn(Xw2, lab_q, global_nn=global_nn, class_nns=class_nns, Xt_full=Xt, use_class=use_class)
        resid2 = np.sqrt(((Xw2 - Y2) ** 2).sum(1))

        med = float(np.median(resid2))
        p90 = float(np.quantile(resid2, 0.90))
        history.append((med, p90))

        theta = float(_theta_from_R(R))
        tv = t.reshape(-1)

        # ✅ 单行动态更新（不刷屏）
        pbar.set_postfix(
            med=f"{med:.4g}",
            p90=f"{p90:.4g}",
            cls=int(use_class),
            th=f"{theta:.3g}",
            tx=f"{tv[0]:.3g}",
            ty=f"{tv[1]:.3g}",
        )

    info = {
        "theta": float(_theta_from_R(R)),
        "t": t.reshape(-1).tolist(),
        "history": history,
    }
    return R, t, info

# ============================================================
# 3) One-call robust registration
# ============================================================

def register_slice_to_ref_robust(
    X_src: np.ndarray,
    X_ref: np.ndarray,
    ann_src: np.ndarray | None = None,
    ann_ref: np.ndarray | None = None,
    *,
    # UOT init
    n_sub: int = 4000,
    reg: float | None = None,
    reg_m: float = 1.0,
    n_starts: int = 6,
    seed: int = 2025,
    init_trim_q: float = 0.10,
    reg_grid: tuple[float, ...] = (0.5, 1.0, 2.0),
    # ICP refine
    icp_iter: int = 40,
    icp_sub: int = 20000,
    icp_trim_q: float = 0.15,
    use_class_after: int = 10,
    max_points_per_class_nn: int | None = 200000,
    verbose: bool = True,
):
    """
    Robust pipeline:
      (1) multi-start UOT init -> best (R,t)
      (2) robust ICP refine -> refined (R,t)
      (3) output aligned source
    """
    X_src = np.asarray(X_src, np.float64)
    X_ref = np.asarray(X_ref, np.float64)

    # ----- init -----
    R0, t0, info0 = uot_global_rigid_init(
        X_src, X_ref,
        n_sub=n_sub,
        reg=reg,
        reg_m=reg_m,
        n_starts=n_starts,
        seed=seed,
        trim_q=init_trim_q,
        reg_grid=reg_grid,
        verbose=verbose,
    )

    # ----- refine (NO double-transform; refine acts on original X_src via composed transforms) -----
    R1, t1, info1 = icp_refine_rigid(
        X_src, X_ref,
        ann_s=ann_src,
        ann_t=ann_ref,
        R0=R0,
        t0=t0,
        n_iter=icp_iter,
        sub_size=icp_sub,
        trim_q=icp_trim_q,
        use_class_after=use_class_after,
        seed=seed,
        verbose=verbose,
        max_points_per_class_nn=max_points_per_class_nn,
    )

    X_aligned = _apply_rigid(X_src, R1, t1)

    info = {
        "init": info0,
        "refine": info1,
        "R": R1,
        "t": t1,
    }
    return X_aligned, info


# ============================================================
# 4) Visualization helper (overlay by annotation)
# ============================================================

def overlay_scatter_by_ann(
    adata,
    x_col: str,
    y_col: str,
    ann_col: str = "leiden_scVI",
    sample_col: str = "sample",
    title: str = "",
    s: float = 3,
    alpha: float = 0.6,
    cmap_name: str = "tab20",
    dpi: int = 200,
    pad_frac: float = 0.02,
    invert_y: bool = False,
    axis_off: bool = True,
    legend: bool = True,
):
    df = pd.DataFrame({
        "x": adata.obs[x_col].to_numpy(),
        "y": adata.obs[y_col].to_numpy(),
        "ann": adata.obs[ann_col].to_numpy(),
        "sample": adata.obs[sample_col].to_numpy(),
    }).dropna(subset=["x", "y"])

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)

    base = 6.0
    ratio = w / h
    figsize = (base * ratio, base) if ratio >= 1 else (base, base / ratio)

    ann_cats = pd.Categorical(df["ann"]).categories.tolist()
    cmap = plt.get_cmap(cmap_name)
    ann_color_map = {ann: cmap(i % cmap.N) for i, ann in enumerate(ann_cats)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for ann in ann_cats:
        m = (df["ann"] == ann)
        ax.scatter(df.loc[m, "x"], df.loc[m, "y"],
                   s=s, c=[ann_color_map[ann]], alpha=alpha, linewidths=0, label=str(ann))

    ax.set_aspect("equal", adjustable="datalim")

    padx = w * pad_frac
    pady = h * pad_frac
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)

    if invert_y:
        ax.invert_yaxis()

    ax.set_title(title)

    if axis_off:
        ax.axis("off")

    if legend:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=ann_color_map[ann], markersize=5, label=str(ann))
            for ann in ann_cats
        ]
        ax.legend(handles=handles, title=ann_col, bbox_to_anchor=(1.02, 1),
                  loc="upper left", borderaxespad=0.0, fontsize=8)

    plt.tight_layout()
    plt.show()
