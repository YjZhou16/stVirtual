import numpy as np
import pandas as pd
import ot
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 0) Rigid utilities (row-vector convention): X' = X @ R + t
# ============================================================

def _theta_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))

def _barycentric_map(P: np.ndarray, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_sum = P.sum(1)  # (ns,)
    denom = np.maximum(row_sum[:, None], 1e-12)
    Y = (P @ Xt) / denom
    return Y, row_sum

def _weighted_rigid(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
        w = np.ones(n, dtype=np.float64)
        ws = float(n)

    w_norm = w / ws

    muX = (w_norm[:, None] * X).sum(0, keepdims=True)
    muY = (w_norm[:, None] * Y).sum(0, keepdims=True)

    Xc = X - muX
    Yc = Y - muY

    C = Xc.T @ (w_norm[:, None] * Yc)
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = muY - muX @ R
    return R, t

def _apply_rigid(X: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (X @ R) + t

def _compose_rigid(R: np.ndarray, t: np.ndarray, dR: np.ndarray, dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R2 = R @ dR
    t2 = (t @ dR) + dt
    return R2, t2

def _robust_center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return X - X.mean(0, keepdims=True)

def _center_scale(F: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    z-score (per-dim) to make feature distances more stable across dims.
    return: Fz, mu, std
    """
    F = np.asarray(F, np.float64)
    mu = F.mean(0, keepdims=True)
    sd = F.std(0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (F - mu) / sd, mu, sd

def _estimate_scale(X: np.ndarray, seed: int = 0, n: int = 2048) -> float:
    """
    Robust-ish scale for distance normalization: median(||x-mean||)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)
    N = X.shape[0]
    if N <= 2:
        return 1.0
    m = min(int(n), N)
    idx = rng.choice(N, size=m, replace=False)
    Xs = X[idx]
    c = Xs.mean(0, keepdims=True)
    r = np.sqrt(((Xs - c) ** 2).sum(1))
    s = float(np.median(r))
    return max(s, 1e-6)

# ============================================================
# 1) Feature-driven UOT init
# ============================================================

def _uot_init_once_feat(
    Xs: np.ndarray,
    Xt: np.ndarray,
    Fs: np.ndarray,
    Ft: np.ndarray,
    *,
    n_sub: int,
    reg: float | None,
    reg_m: float,
    seed: int,
    trim_q: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    One trial:
      - subsample points
      - UOT in FEATURE space
      - barycentric map to TARGET spatial coords
      - robust weighted rigid fit in SPATIAL space
    """
    rng = np.random.default_rng(seed)
    Ns, Nt = Xs.shape[0], Xt.shape[0]
    ns = min(int(n_sub), Ns)
    nt = min(int(n_sub), Nt)

    idx_s = rng.choice(Ns, size=ns, replace=False)
    idx_t = rng.choice(Nt, size=nt, replace=False)

    Xs_sub = np.asarray(Xs[idx_s], np.float64)
    Xt_sub = np.asarray(Xt[idx_t], np.float64)
    Fs_sub = np.asarray(Fs[idx_s], np.float64)
    Ft_sub = np.asarray(Ft[idx_t], np.float64)

    # normalize features (important)
    Fs_z, mu_s, sd_s = _center_scale(Fs_sub)
    Ft_z, mu_t, sd_t = _center_scale(Ft_sub)

    M = cdist(Fs_z, Ft_z, metric="sqeuclidean")

    if reg is None:
        reg0 = 0.05 * float(np.median(M + 1e-12))
        reg = max(reg0, 1e-8)

    a = np.full(ns, 1.0 / ns, dtype=np.float64)
    b = np.full(nt, 1.0 / nt, dtype=np.float64)

    P = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m)

    # correspondences in spatial domain
    Y_match, w_row = _barycentric_map(P, Xt_sub)

    # initial weighted fit
    R0, t0 = _weighted_rigid(Xs_sub, Y_match, w=w_row)

    # robust trimming by spatial residual
    Xw = _apply_rigid(Xs_sub, R0, t0)
    resid = np.sqrt(((Xw - Y_match) ** 2).sum(1))
    if 0.0 < trim_q < 0.5 and len(resid) >= 10:
        thr = np.quantile(resid, 1.0 - trim_q)
        keep = resid <= thr
        if keep.sum() >= max(10, int(0.2 * len(resid))):
            R1, t1 = _weighted_rigid(Xs_sub[keep], Y_match[keep], w=w_row[keep])
            R0, t0 = R1, t1

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

def uot_global_rigid_init_feat(
    Xs: np.ndarray,
    Xt: np.ndarray,
    Fs: np.ndarray,
    Ft: np.ndarray,
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
    Multi-start UOT init in FEATURE space, rigid fit in SPATIAL space.
    """
    Xs = np.asarray(Xs, np.float64)
    Xt = np.asarray(Xt, np.float64)
    Fs = np.asarray(Fs, np.float64)
    Ft = np.asarray(Ft, np.float64)
    assert Xs.shape[1] == Xt.shape[1] == 2, "assumes 2D spatial coords"
    assert Fs.shape[0] == Xs.shape[0] and Ft.shape[0] == Xt.shape[0], "features must align with coords"

    # reg heuristic on feature distances
    if reg is None:
        rng = np.random.default_rng(seed)
        ns = min(1024, Fs.shape[0])
        nt = min(1024, Ft.shape[0])
        fs = Fs[rng.choice(Fs.shape[0], size=ns, replace=False)]
        ft = Ft[rng.choice(Ft.shape[0], size=nt, replace=False)]
        fs_z, _, _ = _center_scale(fs)
        ft_z, _, _ = _center_scale(ft)
        M = cdist(fs_z, ft_z, metric="sqeuclidean")
        reg = max(0.05 * float(np.median(M + 1e-12)), 1e-8)

    best = None
    best_score = np.inf

    base_seed = int(seed)
    total = int(n_starts) * len(reg_grid)
    pbar = tqdm(total=total, disable=(not verbose), desc="UOT init (feat)", dynamic_ncols=True)

    tried = 0
    for si in range(int(n_starts)):
        for g in reg_grid:
            tried += 1
            R0, t0, info = _uot_init_once_feat(
                Xs, Xt, Fs, Ft,
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
        print(f"[UOT init feat] BEST score={info_best['score']:.6f} reg={info_best['reg']:.6g} theta={info_best['theta']:.4f} t={info_best['t']}")
    return R_best, t_best, info_best


# ============================================================
# 2) Feature-gated ICP refine (hard/soft)
# ============================================================

def _build_feat_index(Ft: np.ndarray, ann: np.ndarray | None, *, max_points_per_class: int | None = 200000, seed: int = 0):
    rng = np.random.default_rng(seed)
    Ft = np.asarray(Ft, np.float64)

    global_nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Ft)

    class_nns = {}
    if ann is None:
        return global_nn, class_nns

    ann = np.asarray(ann)
    for cls in pd.unique(ann):
        m = (ann == cls)
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        if max_points_per_class is not None and idx.size > int(max_points_per_class):
            idx = rng.choice(idx, size=int(max_points_per_class), replace=False)
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Ft[idx])
        class_nns[cls] = (nn, idx)  # nn on Ft[idx], plus mapping to original indices
    return global_nn, class_nns

def _feat_knn_candidates(
    Fq: np.ndarray,
    lab_q: np.ndarray | None,
    *,
    global_nn,
    class_nns: dict,
    k_feat: int,
    use_class: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return candidate indices in target + their feature distances.
    idx: (B,k), dist: (B,k)
    """
    Fq = np.asarray(Fq, np.float64)
    B = Fq.shape[0]
    k = int(max(1, k_feat))
    idx_out = np.empty((B, k), dtype=np.int64)
    dist_out = np.empty((B, k), dtype=np.float64)

    if lab_q is None or (not use_class) or len(class_nns) == 0:
        dist, idx = global_nn.kneighbors(Fq, n_neighbors=k)
        idx_out[:] = idx
        dist_out[:] = dist
        return idx_out, dist_out

    lab_q = np.asarray(lab_q)
    for cls in pd.unique(lab_q):
        ii = np.where(lab_q == cls)[0]
        if ii.size == 0:
            continue
        sub = Fq[ii]
        if cls in class_nns:
            nn, idx_map = class_nns[cls]
            dist, jj = nn.kneighbors(sub, n_neighbors=k)
            idx_out[ii] = idx_map[jj]
            dist_out[ii] = dist
        else:
            dist, jj = global_nn.kneighbors(sub, n_neighbors=k)
            idx_out[ii] = jj
            dist_out[ii] = dist
    return idx_out, dist_out

def _match_from_feat_candidates(
    Xw: np.ndarray,
    Xt: np.ndarray,
    cand_idx: np.ndarray,
    cand_dfeat: np.ndarray,
    *,
    mode: str = "soft",          # "soft" or "hard"
    alpha_xy: float = 1.0,       # weight for spatial
    beta_feat: float = 1.0,      # weight for feature
    tau: float = 1.0,            # soft temperature
    xy_scale: float = 1.0,
    feat_scale: float = 1.0,
    chunk: int = 4096,
) -> np.ndarray:
    """
    Given warped Xw and candidate target indices (by feature),
    choose/average them to get matched spatial Y.
    """
    Xw = np.asarray(Xw, np.float64)
    Xt = np.asarray(Xt, np.float64)
    cand_idx = np.asarray(cand_idx, np.int64)
    cand_dfeat = np.asarray(cand_dfeat, np.float64)

    B, k = cand_idx.shape
    Y = np.empty_like(Xw)

    xy_scale = float(max(xy_scale, 1e-8))
    feat_scale = float(max(feat_scale, 1e-8))
    tau = float(max(tau, 1e-8))

    for s in range(0, B, int(chunk)):
        e = min(B, s + int(chunk))
        Xb = Xw[s:e]                          # (b,2)
        idxb = cand_idx[s:e]                  # (b,k)
        dfeatb = cand_dfeat[s:e] / feat_scale # (b,k)

        # coords of candidates: (b,k,2)
        C = Xt[idxb]

        # spatial squared distances in normalized scale
        dxy2 = ((C - Xb[:, None, :]) / xy_scale)
        dxy2 = (dxy2 ** 2).sum(-1)            # (b,k)

        if mode == "hard":
            cost = alpha_xy * dxy2 + beta_feat * (dfeatb ** 2)
            j = np.argmin(cost, axis=1)
            Y[s:e] = C[np.arange(e - s), j]
        else:
            cost = alpha_xy * dxy2 + beta_feat * (dfeatb ** 2)
            w = np.exp(-cost / tau)           # (b,k)
            ws = np.maximum(w.sum(1, keepdims=True), 1e-12)
            w = w / ws
            Y[s:e] = (w[..., None] * C).sum(1)

    return Y

def icp_refine_rigid_feat(
    Xs: np.ndarray,
    Xt: np.ndarray,
    Fs: np.ndarray,
    Ft: np.ndarray,
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
    max_points_per_class_feat: int | None = 200000,
    k_feat: int = 32,
    match_mode: str = "soft",     # "soft"/"hard"
    alpha_xy: float = 1.0,
    beta_feat: float = 1.0,
    tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:

    rng = np.random.default_rng(seed)

    Xs = np.asarray(Xs, np.float64)
    Xt = np.asarray(Xt, np.float64)
    Fs = np.asarray(Fs, np.float64)
    Ft = np.asarray(Ft, np.float64)

    if ann_s is not None:
        ann_s = np.asarray(ann_s)
    if ann_t is not None:
        ann_t = np.asarray(ann_t)

    # feature index on target
    global_feat_nn = NearestNeighbors(n_neighbors=int(max(1, k_feat)), algorithm="auto").fit(Ft)
    _, class_feat_nns = _build_feat_index(Ft, ann_t, max_points_per_class=max_points_per_class_feat, seed=seed)

    R = np.asarray(R0, np.float64).copy()
    t = np.asarray(t0, np.float64).reshape(1, -1).copy()

    Ns = Xs.shape[0]
    history = []

    # normalize scales for combined cost
    xy_scale = _estimate_scale(Xt, seed=seed, n=2048)
    feat_scale = _estimate_scale(Ft, seed=seed, n=2048)  # same helper works for any dim

    pbar = trange(int(n_iter), disable=(not verbose), desc="ICP refine (feat)", dynamic_ncols=True)

    for it in pbar:
        # sample queries
        if sub_size is None or sub_size <= 0 or sub_size >= Ns:
            idx = np.arange(Ns)
        else:
            idx = rng.choice(Ns, size=int(sub_size), replace=False)

        Xq = Xs[idx]
        Fq = Fs[idx]
        lab_q = ann_s[idx] if ann_s is not None else None

        Xw = _apply_rigid(Xq, R, t)

        use_class = (it >= int(use_class_after))

        # feature candidates (kNN in feature space)
        if lab_q is None or (not use_class) or len(class_feat_nns) == 0:
            dfeat, cand = global_feat_nn.kneighbors(Fq, n_neighbors=int(max(1, k_feat)))
        else:
            cand, dfeat = _feat_knn_candidates(
                Fq, lab_q,
                global_nn=global_feat_nn,
                class_nns=class_feat_nns,
                k_feat=int(max(1, k_feat)),
                use_class=True,
            )

        # match in spatial using feature candidates (+ optional soft barycenter)
        Y = _match_from_feat_candidates(
            Xw, Xt, cand, dfeat,
            mode=str(match_mode),
            alpha_xy=float(alpha_xy),
            beta_feat=float(beta_feat),
            tau=float(tau),
            xy_scale=xy_scale,
            feat_scale=feat_scale,
        )

        # residual + trimming (spatial)
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

        # compose update
        R, t = _compose_rigid(R, t, dR, dt)

        # diagnostics
        Xw2 = _apply_rigid(Xq, R, t)
        # reuse same candidates from this iter (ok), or recompute; here recompute for more accurate metrics
        if lab_q is None or (not use_class) or len(class_feat_nns) == 0:
            dfeat2, cand2 = global_feat_nn.kneighbors(Fq, n_neighbors=int(max(1, k_feat)))
        else:
            cand2, dfeat2 = _feat_knn_candidates(
                Fq, lab_q,
                global_nn=global_feat_nn,
                class_nns=class_feat_nns,
                k_feat=int(max(1, k_feat)),
                use_class=True,
            )
        Y2 = _match_from_feat_candidates(
            Xw2, Xt, cand2, dfeat2,
            mode=str(match_mode),
            alpha_xy=float(alpha_xy),
            beta_feat=float(beta_feat),
            tau=float(tau),
            xy_scale=xy_scale,
            feat_scale=feat_scale,
        )

        resid2 = np.sqrt(((Xw2 - Y2) ** 2).sum(1))
        med = float(np.median(resid2))
        p90 = float(np.quantile(resid2, 0.90))
        history.append((med, p90))

        theta = float(_theta_from_R(R))
        tv = t.reshape(-1)

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
        "xy_scale": float(xy_scale),
        "feat_scale": float(feat_scale),
        "k_feat": int(k_feat),
        "match_mode": str(match_mode),
        "alpha_xy": float(alpha_xy),
        "beta_feat": float(beta_feat),
        "tau": float(tau),
    }
    return R, t, info

# ============================================================
# 3) One-call robust registration (FEATURE-driven)
# ============================================================

def register_slice_to_ref_by_features(
    X_src: np.ndarray,
    X_ref: np.ndarray,
    F_src: np.ndarray,
    F_ref: np.ndarray,
    ann_src: np.ndarray | None = None,
    ann_ref: np.ndarray | None = None,
    *,
    # UOT init (feature)
    n_sub: int = 4000,
    reg: float | None = None,
    reg_m: float = 1.0,
    n_starts: int = 6,
    seed: int = 2025,
    init_trim_q: float = 0.10,
    reg_grid: tuple[float, ...] = (0.5, 1.0, 2.0),
    # ICP refine (feature-gated)
    icp_iter: int = 40,
    icp_sub: int = 20000,
    icp_trim_q: float = 0.15,
    use_class_after: int = 10,
    max_points_per_class_feat: int | None = 200000,
    k_feat: int = 32,
    match_mode: str = "soft",   # "soft"/"hard"
    alpha_xy: float = 1.0,
    beta_feat: float = 1.0,
    tau: float = 1.0,
    verbose: bool = True,
):
    """
    Pipeline:
      (1) UOT init in FEATURE space -> (R0,t0)
      (2) ICP refine with FEATURE-gated candidates -> (R1,t1)
      (3) return aligned spatial coords
    """
    X_src = np.asarray(X_src, np.float64)
    X_ref = np.asarray(X_ref, np.float64)
    F_src = np.asarray(F_src, np.float64)
    F_ref = np.asarray(F_ref, np.float64)

    R0, t0, info0 = uot_global_rigid_init_feat(
        X_src, X_ref, F_src, F_ref,
        n_sub=n_sub,
        reg=reg,
        reg_m=reg_m,
        n_starts=n_starts,
        seed=seed,
        trim_q=init_trim_q,
        reg_grid=reg_grid,
        verbose=verbose,
    )

    R1, t1, info1 = icp_refine_rigid_feat(
        X_src, X_ref, F_src, F_ref,
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
        max_points_per_class_feat=max_points_per_class_feat,
        k_feat=k_feat,
        match_mode=match_mode,
        alpha_xy=alpha_xy,
        beta_feat=beta_feat,
        tau=tau,
    )

    X_aligned = _apply_rigid(X_src, R1, t1)

    info = {
        "init": info0,
        "refine": info1,
        "R": R1,
        "t": t1,
    }
    return X_aligned, info

# import numpy as np
# import torch
# import ot
# from tqdm import trange
# from scipy.spatial.distance import cdist
# from sklearn.neighbors import NearestNeighbors


# def _l2norm(x, eps=1e-12):
#     x = np.asarray(x, dtype=np.float64)
#     n = np.linalg.norm(x, axis=1, keepdims=True)
#     return x / np.maximum(n, eps)


# def _umeyama_rigid(X, Y):
#     Xc = X - X.mean(0, keepdims=True)
#     Yc = Y - Y.mean(0, keepdims=True)

#     C = Xc.T @ Yc / X.shape[0]
#     U, S, Vt = np.linalg.svd(C)

#     R = U @ Vt
#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = U @ Vt

#     t = Y.mean(0, keepdims=True) - X.mean(0, keepdims=True) @ R.T
#     return R, t


# def _theta_from_R(R):
#     return np.arctan2(R[1, 0], R[0, 0])


# def _barycentric_map(P, Xt):
#     row_sum = P.sum(1, keepdims=True)
#     Y = (P @ Xt) / np.maximum(row_sum, 1e-12)
#     return Y


# def align_slice_uot_transcript_rigid(
#     Xs, Xt,          # 坐标 (Ns,2), (Nt,2)
#     Zs, Zt,          # 转录嵌入 (Ns,d), (Nt,d)
#     *,
#     n_sub=4000,
#     reg=None,
#     reg_m=1.0,
#     seed=0,
#     metric="cosine",   # "cosine" or "sqeuclidean"
# ):
#     """
#     用转录相似性做 UOT，对应到 target 坐标后求刚体初始变换
#     """
#     rng = np.random.default_rng(seed)

#     Ns, Nt = Xs.shape[0], Xt.shape[0]
#     idx_s = rng.choice(Ns, size=min(n_sub, Ns), replace=False)
#     idx_t = rng.choice(Nt, size=min(n_sub, Nt), replace=False)

#     Xs_sub = Xs[idx_s].astype(np.float64)
#     Xt_sub = Xt[idx_t].astype(np.float64)
#     Zs_sub = Zs[idx_s].astype(np.float64)
#     Zt_sub = Zt[idx_t].astype(np.float64)

#     if metric == "cosine":
#         Zs_n = _l2norm(Zs_sub)
#         Zt_n = _l2norm(Zt_sub)
#         M = 1.0 - (Zs_n @ Zt_n.T)  # cosine distance
#         M = np.maximum(M, 0.0)
#     else:
#         M = cdist(Zs_sub, Zt_sub, metric="sqeuclidean")

#     if reg is None:
#         reg = 0.05 * np.median(M + 1e-12)

#     a = np.full(Xs_sub.shape[0], 1.0 / Xs_sub.shape[0], dtype=np.float64)
#     b = np.full(Xt_sub.shape[0], 1.0 / Xt_sub.shape[0], dtype=np.float64)

#     P = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m)

#     # 注意：barycentric 的目标仍然是 target 的坐标 Xt_sub
#     Y_match = _barycentric_map(P, Xt_sub)

#     R, t = _umeyama_rigid(Xs_sub, Y_match)
#     Xs_rigid = (Xs @ R.T) + t

#     init_theta = _theta_from_R(R)
#     init_t_vec = t.reshape(-1)

#     return Xs_rigid, init_theta, init_t_vec

# def train_global_rigid_refine_by_transcript(
#     X_src, X_tgt,        # 坐标
#     Z_src, Z_tgt,        # 转录嵌入
#     ann_src=None,
#     ann_tgt=None,
#     init_theta=0.0,
#     init_t_vec=(0.0, 0.0),
#     *,
#     steps=1000,
#     batch_size=2048,
#     lr=1e-2,
#     rot_reg=1e-5,
#     trans_reg=1e-6,
#     use_class_gate=True,     # 是否同类约束
#     z_metric="cosine",
#     verbose=True
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     Xs0 = torch.tensor(X_src, dtype=torch.float32, device=device)
#     Xt0 = torch.tensor(X_tgt, dtype=torch.float32, device=device)

#     Ns = Xs0.shape[0]

#     theta = torch.tensor(float(init_theta), dtype=torch.float32, device=device, requires_grad=True)
#     t_vec = torch.tensor(np.asarray(init_t_vec, dtype=np.float32), dtype=torch.float32, device=device, requires_grad=True)

#     # ---------- 预建 transcript NN 索引 ----------
#     Z_src_np = np.asarray(Z_src, dtype=np.float32)
#     Z_tgt_np = np.asarray(Z_tgt, dtype=np.float32)

#     if z_metric == "cosine":
#         Z_src_np = _l2norm(Z_src_np).astype(np.float32)
#         Z_tgt_np = _l2norm(Z_tgt_np).astype(np.float32)
#         nn_metric = "cosine"
#     else:
#         nn_metric = "euclidean"

#     ann_src_np = None if ann_src is None else np.asarray(ann_src)
#     ann_tgt_np = None if ann_tgt is None else np.asarray(ann_tgt)

#     class_nns = {}
#     if use_class_gate and (ann_src_np is not None) and (ann_tgt_np is not None):
#         for cls in np.unique(ann_tgt_np):
#             mask = (ann_tgt_np == cls)
#             if mask.sum() == 0:
#                 continue
#             nbrs = NearestNeighbors(n_neighbors=1, metric=nn_metric).fit(Z_tgt_np[mask])
#             class_nns[cls] = (nbrs, X_tgt[mask])

#     global_nn = NearestNeighbors(n_neighbors=1, metric=nn_metric).fit(Z_tgt_np)

#     # ---------- 优化 ----------
#     opt = torch.optim.Adam([theta, t_vec], lr=lr)

#     for step in trange(steps):
#         batch_idx_torch = torch.randperm(Ns, device=device)[:batch_size]
#         batch_idx = batch_idx_torch.detach().cpu().numpy()

#         xb0 = Xs0[batch_idx_torch]           # (B,2)
#         zb = Z_src_np[batch_idx]             # (B,d)
#         lab_b = None if ann_src_np is None else ann_src_np[batch_idx]

#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)
#         R = torch.stack([
#             torch.stack([cos_t, -sin_t]),
#             torch.stack([sin_t,  cos_t])
#         ])
#         xb_warp = (xb0 @ R.T) + t_vec

#         # transcript-based target matching (目标坐标)
#         yt_match_np = np.zeros((len(batch_idx), 2), dtype=np.float32)

#         if (lab_b is not None) and use_class_gate:
#             for cls in np.unique(lab_b):
#                 idxs = np.where(lab_b == cls)[0]
#                 z_sub = zb[idxs]

#                 if cls in class_nns:
#                     nbrs_cls, X_cls = class_nns[cls]
#                     _, nn_idx = nbrs_cls.kneighbors(z_sub, n_neighbors=1)
#                     yt_match_np[idxs] = X_cls[nn_idx[:, 0]]
#                 else:
#                     _, nn_idx = global_nn.kneighbors(z_sub, n_neighbors=1)
#                     yt_match_np[idxs] = X_tgt[nn_idx[:, 0]]
#         else:
#             _, nn_idx = global_nn.kneighbors(zb, n_neighbors=1)
#             yt_match_np[:] = X_tgt[nn_idx[:, 0]]

#         yt_match = torch.tensor(yt_match_np, dtype=torch.float32, device=device)

#         loss_fit = ((xb_warp - yt_match) ** 2).sum(dim=1).mean()
#         loss_reg_rot = (theta ** 2) * rot_reg
#         loss_reg_trans = (t_vec ** 2).mean() * trans_reg
#         loss = loss_fit + loss_reg_rot + loss_reg_trans

#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         if verbose and ((step + 1) % 400 == 0 or step == 0):
#             print(f"[transcript rigid refine] step {step+1}/{steps} loss={loss.item():.4f}")

#     with torch.no_grad():
#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)
#         R = torch.stack([
#             torch.stack([cos_t, -sin_t]),
#             torch.stack([sin_t,  cos_t])
#         ])
#         X_full = (Xs0 @ R.T) + t_vec

#     return X_full.cpu().numpy()

# def register_slice_to_ref_by_transcript(
#     X_src, X_ref,
#     Z_src, Z_ref,            # 新增：转录嵌入
#     ann_src=None, ann_ref=None,
#     *,
#     n_sub=4000,
#     reg=None,
#     reg_m=1.0,
#     seed=2025,
#     steps=1000,
#     batch_size=2048,
#     lr=1e-2,
#     rot_reg=1e-5,
#     trans_reg=1e-6,
#     z_metric="cosine",
#     use_class_gate=True,
#     verbose=True
# ):
#     # 1) transcript-UOT 初始化
#     X_src_rigid, init_theta, init_t_vec = align_slice_uot_transcript_rigid(
#         X_src, X_ref,
#         Z_src, Z_ref,
#         n_sub=n_sub,
#         reg=reg,
#         reg_m=reg_m,
#         seed=seed,
#         metric=z_metric
#     )

#     # 2) transcript-NN refine
#     X_src_refined = train_global_rigid_refine_by_transcript(
#         X_src=X_src_rigid,
#         X_tgt=X_ref,
#         Z_src=Z_src,
#         Z_tgt=Z_ref,
#         ann_src=ann_src,
#         ann_tgt=ann_ref,
#         init_theta=0.0,              # 这里建议用 0（因为 X_src_rigid 已经变换过）
#         init_t_vec=np.zeros(2),      # 同理
#         steps=steps,
#         batch_size=batch_size,
#         lr=lr,
#         rot_reg=rot_reg,
#         trans_reg=trans_reg,
#         use_class_gate=use_class_gate,
#         z_metric=z_metric,
#         verbose=verbose
#     )
#     return X_src_refined