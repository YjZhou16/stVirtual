import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from typing import Optional, Sequence


# -------------------------
# 1) Extract the representation/latent 
# from a specific frame in the rollout.
# -------------------------
def _get_first_existing_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def _as_2d_float32(X):
    if X is None:
        return None
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    return X.astype(np.float32, copy=False)


def _maybe_to_sparse(X, make_sparse=False, sparse_threshold=0.7):
    if (X is None) or (not make_sparse):
        return X
    zero_frac = float((X == 0).mean()) if X.size else 0.0
    if zero_frac >= sparse_threshold:
        return sparse.csr_matrix(X)
    return X


def _frame_pick(v, fi, n_frames):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == n_frames:
            return v[fi]
        if len(v) == 1:
            return v[0]
        return v
    return v


def _safe_float_tag(x: float, nd: int = 4) -> str:
    return f"{float(x):.{nd}f}".replace(".", "p")


def _get_t_array(rollout):
    if "t" in rollout:
        return np.asarray(rollout["t"], np.float32)
    return np.linspace(0, 1, len(rollout["coords"]), dtype=np.float32)


def _layer_idx_to_name_arr(layer_idx, layer_names):
    if layer_names is None:
        return None
    layer_names = list(layer_names)
    out = []
    for li in np.asarray(layer_idx).reshape(-1).tolist():
        if 0 <= int(li) < len(layer_names):
            out.append(layer_names[int(li)])
        else:
            out.append(str(li))
    return np.asarray(out, dtype=object)


def rollout_frame_to_adata(
    rollout,
    frame_idx,
    *,
    seg=None,
    t=None,
    layer_names=None,                 # e.g. ctx.layers_list
    prefer_expr_keys=("expr", "X", "x", "gene_expr", "counts", "fs", "scaled"),
    prefer_latent_keys=("latent", "z", "Z", "X_latent", "emb", "scaled"),
    var_names=None,
    make_sparse=False,
):
    coords = np.asarray(rollout["coords"][frame_idx], np.float32)
    layers = np.asarray(rollout["layers"][frame_idx], np.int64)

    # expr
    k_expr = _get_first_existing_key(rollout, prefer_expr_keys)
    X = None
    if k_expr is not None:
        X = _as_2d_float32(rollout[k_expr][frame_idx])
        X = _maybe_to_sparse(X, make_sparse=make_sparse)

    # latent
    k_lat = _get_first_existing_key(rollout, prefer_latent_keys)
    Z = None
    if k_lat is not None:
        Z = _as_2d_float32(rollout[k_lat][frame_idx])

    if X is None:
        X = np.zeros((coords.shape[0], 0), dtype=np.float32)

    if var_names is None:
        var_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        var_names = list(var_names)
        if len(var_names) != X.shape[1]:
            var_names = [f"f{i}" for i in range(X.shape[1])]

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)

    # coords
    d = coords.shape[1]
    if d >= 2:
        adata.obsm["spatial"] = coords[:, :2].astype(np.float32, copy=False)
        adata.obs["cx"] = coords[:, 0].astype(np.float32, copy=False)
        adata.obs["cy"] = coords[:, 1].astype(np.float32, copy=False)
    if d >= 3:
        adata.obs["cz"] = coords[:, 2].astype(np.float32, copy=False)

    # labels
    adata.obs["layer_idx"] = layers.astype(np.int64, copy=False)
    if layer_names is not None:
        lname = _layer_idx_to_name_arr(layers, layer_names)
        adata.obs["layer_name"] = pd.Categorical(lname)

    # meta
    if seg is not None:
        adata.uns["seg"] = str(seg)
    if t is not None:
        adata.uns["t"] = float(t)

    # latent
    if Z is not None:
        adata.obsm["X_latent"] = Z

    return adata


# -------------------------
# 2) Intermediate frame selection
# -------------------------
def pick_mid_frame_index(rollout, t_mid=0.5):
    t_arr = _get_t_array(rollout)
    dif = np.abs(t_arr - float(t_mid))
    cand = np.where(dif == dif.min())[0]
    idx = int(cand[-1])
    return idx, float(t_arr[idx])


def lerp_mid_cell_level(rollout, t_mid=0.5):
    t_arr = _get_t_array(rollout)
    if t_mid <= t_arr[0]:
        i0 = i1 = 0
        alpha = 0.0
    elif t_mid >= t_arr[-1]:
        i0 = i1 = len(t_arr) - 1
        alpha = 0.0
    else:
        i1 = int(np.searchsorted(t_arr, t_mid, side="right"))
        i0 = i1 - 1
        denom = float(t_arr[i1] - t_arr[i0])
        alpha = 0.0 if denom <= 1e-12 else float((t_mid - t_arr[i0]) / denom)

    c0 = np.asarray(rollout["coords"][i0], np.float32)
    c1 = np.asarray(rollout["coords"][i1], np.float32)
    if c0.shape[0] != c1.shape[0]:
        return None

    coords_mid = (1 - alpha) * c0 + alpha * c1
    layers_mid = np.asarray(rollout["layers"][i0], np.int64)

    k_expr = _get_first_existing_key(rollout, ("expr", "X", "x", "gene_expr", "counts", "fs"))
    X_mid = None
    if k_expr is not None:
        X0 = _as_2d_float32(rollout[k_expr][i0])
        X1 = _as_2d_float32(rollout[k_expr][i1])
        if X0.shape == X1.shape:
            X_mid = (1 - alpha) * X0 + alpha * X1

    return coords_mid, layers_mid, X_mid, (i0, i1, alpha), (float(t_arr[i0]), float(t_arr[i1]))


# -------------------------
# 3) hist_interp
# -------------------------
def _make_edges_from_union(sim_c0, sim_c1, grid_size, pad_frac=0.01):
    allc = np.vstack([sim_c0, sim_c1]).astype(np.float32)
    mn = np.nanmin(allc, axis=0)
    mx = np.nanmax(allc, axis=0)
    span = np.maximum(mx - mn, 1e-6)
    mn = mn - pad_frac * span
    mx = mx + pad_frac * span
    return [np.linspace(mn[j], mx[j], int(grid_size[j]) + 1, dtype=np.float32) for j in range(allc.shape[1])]


def _hist_counts(coords, edges_list):
    if coords.shape[0] == 0:
        shape = tuple(len(e) - 1 for e in edges_list)
        return np.zeros(shape, dtype=np.float32)
    H, _ = np.histogramdd(coords, bins=edges_list)
    return H.astype(np.float32)


def save_mid_density_grid_h5ad(
    rollout, out_path,
    *,
    t_mid=0.5,
    grid_size=(112, 80),
    layers=None,
    pad_frac=0.01,
    normalize=True,
):
    t_arr = _get_t_array(rollout)

    if t_mid <= t_arr[0]:
        i0 = i1 = 0
        alpha = 0.0
    elif t_mid >= t_arr[-1]:
        i0 = i1 = len(t_arr) - 1
        alpha = 0.0
    else:
        i1 = int(np.searchsorted(t_arr, t_mid, side="right"))
        i0 = i1 - 1
        denom = float(t_arr[i1] - t_arr[i0])
        alpha = 0.0 if denom <= 1e-12 else float((t_mid - t_arr[i0]) / denom)

    c0 = np.asarray(rollout["coords"][i0], np.float32)[:, :2]
    c1 = np.asarray(rollout["coords"][i1], np.float32)[:, :2]
    l0 = np.asarray(rollout["layers"][i0], np.int64)
    l1 = np.asarray(rollout["layers"][i1], np.int64)

    if layers is None:
        layers = sorted(set(l0.tolist()) | set(l1.tolist()))
    layers = [int(L) for L in layers]

    edges = _make_edges_from_union(c0, c1, grid_size, pad_frac=pad_frac)
    xcent = (edges[0][:-1] + edges[0][1:]) * 0.5
    ycent = (edges[1][:-1] + edges[1][1:]) * 0.5
    xx, yy = np.meshgrid(xcent, ycent, indexing="ij")
    H, W = xx.shape

    obs_list = []
    X_list = []

    for L in layers:
        H0 = _hist_counts(c0[l0 == L], edges)
        H1 = _hist_counts(c1[l1 == L], edges)
        Hm = (1 - alpha) * H0 + alpha * H1
        if normalize:
            s = float(Hm.sum())
            if s > 0:
                Hm = Hm / s

        obs = pd.DataFrame({
            "layer_idx": np.full(H * W, L, dtype=np.int64),
            "ix": np.repeat(np.arange(H), W),
            "iy": np.tile(np.arange(W), H),
            "cx": xx.reshape(-1).astype(np.float32),
            "cy": yy.reshape(-1).astype(np.float32),
        })
        obs_list.append(obs)
        X_list.append(Hm.reshape(-1, 1).astype(np.float32))

    obs_all = pd.concat(obs_list, axis=0, ignore_index=True)
    X_all = np.vstack(X_list)

    adata = ad.AnnData(X=X_all, obs=obs_all, var=pd.DataFrame(index=["density"]))
    adata.obsm["spatial"] = obs_all[["cx", "cy"]].to_numpy(np.float32)
    adata.uns["t_mid"] = float(t_mid)
    adata.uns["bracket"] = (int(i0), int(i1))
    adata.uns["alpha"] = float(alpha)
    adata.uns["bracket_t"] = (float(t_arr[i0]), float(t_arr[i1]))
    adata.write_h5ad(out_path)


# -------------------------
# 4) general attach helpers
# -------------------------
def attach_rollout_metadata_to_adata(
    ad,
    r,
    fi,
    n_frames,
    *,
    layer_names=None,
    uid_keys=("uid", "uids", "cell_uid", "cell_uids"),
    uid_obs_col="uid",
    set_obs_names_from_uid=True,
    parent_uid_keys=("parent_uid", "parent_uids"),
    parent_uid_obs_col="parent_uid",
    is_birth_keys=("is_birth", "birth", "born", "newborn", "is_new"),
    is_birth_obs_col="is_birth",
    is_diff_keys=("is_diff",),
    is_diff_obs_col="is_diff",
    diff_alpha_keys=("diff_alpha",),
    diff_alpha_obs_col="diff_alpha",
    diff_tgt_layer_keys=("diff_tgt_layer",),
    diff_tgt_layer_obs_col="diff_tgt_layer",
    diff_tgt_layer_name_obs_col="diff_tgt_layer_name",
):
    def _pick_first_existing_key(keys):
        for k in keys:
            if k in r:
                return k
        return None

    def _get_frame_vec(keys):
        k = _pick_first_existing_key(keys)
        if k is None:
            return None
        arr = r.get(k, None)
        if arr is None or fi >= len(arr):
            return None
        x = arr[fi]
        if x is None:
            return None
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.reshape(-1)
        return x

    def _attach_obs(col, vec, dtype=None):
        if vec is None:
            return
        if len(vec) != ad.n_obs:
            print(f"[warn] {col} len={len(vec)} != n_obs={ad.n_obs} @ frame={fi}, skip {col} attach")
            return
        if dtype is not None:
            vec = vec.astype(dtype, copy=False)
        ad.obs[col] = vec

    def _attach_layer_name_from_idx(idx_vec, idx_col, name_col):
        if idx_vec is None:
            return
        if len(idx_vec) != ad.n_obs:
            print(f"[warn] {idx_col} len={len(idx_vec)} != n_obs={ad.n_obs} @ frame={fi}, skip {name_col} attach")
            return
        idx_vec = np.asarray(idx_vec).astype(np.int64, copy=False)
        ad.obs[idx_col] = idx_vec
        if layer_names is None:
            return
        cats = list(layer_names)
        vals = [cats[int(v)] if 0 <= int(v) < len(cats) else pd.NA for v in idx_vec.tolist()]
        ad.obs[name_col] = pd.Categorical(vals, categories=cats)

    uid_vec = _get_frame_vec(uid_keys)
    if uid_vec is not None:
        _attach_obs(uid_obs_col, uid_vec, np.int64)
        if set_obs_names_from_uid and len(uid_vec) == ad.n_obs:
            ad.obs_names = pd.Index([str(x) for x in uid_vec], dtype="object")

    _attach_obs(parent_uid_obs_col, _get_frame_vec(parent_uid_keys), np.int64)
    _attach_obs(is_birth_obs_col, _get_frame_vec(is_birth_keys), np.bool_)
    _attach_obs(is_diff_obs_col, _get_frame_vec(is_diff_keys), np.bool_)
    _attach_obs(diff_alpha_obs_col, _get_frame_vec(diff_alpha_keys), np.float32)
    _attach_layer_name_from_idx(_get_frame_vec(diff_tgt_layer_keys), diff_tgt_layer_obs_col, diff_tgt_layer_name_obs_col)

    ad.uns["frame_idx"] = int(fi)
    ad.uns["n_frames"] = int(n_frames)


def save_all_rollouts_as_h5ad(
    rollouts,
    stages,
    ctx,
    out_dir,
    *,
    sample_key="sample",
    make_sparse=False,
    save_which="all",
    frame_stride=1,
    t_tag_ndigits=4,
    add_frame_meta=True,
    uid_keys=("uid", "uids", "cell_uid", "cell_uids"),
    uid_obs_col="uid",
    set_obs_names_from_uid=True,
    parent_uid_keys=("parent_uid", "parent_uids"),
    parent_uid_obs_col="parent_uid",
    is_birth_keys=("is_birth", "birth", "born", "newborn", "is_new"),
    is_birth_obs_col="is_birth",
    is_diff_keys=("is_diff",),
    is_diff_obs_col="is_diff",
    diff_alpha_keys=("diff_alpha",),
    diff_alpha_obs_col="diff_alpha",
    diff_tgt_layer_keys=("diff_tgt_layer",),
    diff_tgt_layer_obs_col="diff_tgt_layer",
    diff_tgt_layer_name_obs_col="diff_tgt_layer_name",
):
    os.makedirs(out_dir, exist_ok=True)
    layer_names = getattr(ctx, "layers_list", None)
    var_names = getattr(getattr(ctx, "adata_all", None), "var_names", None)

    def _attach_common(ad_i, r, fi, n_frames):
        attach_rollout_metadata_to_adata(
            ad_i, r, fi, n_frames,
            layer_names=layer_names,
            uid_keys=uid_keys,
            uid_obs_col=uid_obs_col,
            set_obs_names_from_uid=set_obs_names_from_uid,
            parent_uid_keys=parent_uid_keys,
            parent_uid_obs_col=parent_uid_obs_col,
            is_birth_keys=is_birth_keys,
            is_birth_obs_col=is_birth_obs_col,
            is_diff_keys=is_diff_keys,
            is_diff_obs_col=is_diff_obs_col,
            diff_alpha_keys=diff_alpha_keys,
            diff_alpha_obs_col=diff_alpha_obs_col,
            diff_tgt_layer_keys=diff_tgt_layer_keys,
            diff_tgt_layer_obs_col=diff_tgt_layer_obs_col,
            diff_tgt_layer_name_obs_col=diff_tgt_layer_name_obs_col,
        )
        if add_frame_meta:
            ad_i.uns["frame_idx"] = int(fi)
            ad_i.uns["n_frames"] = int(n_frames)

    for cfg in stages:
        seg = f"{cfg.src}_to_{cfg.tgt}"
        r = rollouts[seg]
        n_frames = len(r["coords"])
        t_arr = _get_t_array(r)

        if save_which == "all":
            for fi in range(0, n_frames, int(frame_stride)):
                tfi = float(t_arr[fi]) if (t_arr is not None and len(t_arr) == n_frames) else (fi / max(n_frames - 1, 1))
                ad_i = rollout_frame_to_adata(
                    r,
                    fi,
                    seg=seg,
                    t=tfi,
                    layer_names=layer_names,
                    var_names=var_names,
                    make_sparse=make_sparse,
                )
                _attach_common(ad_i, r, fi, n_frames)
                fname = f"{seg}_f{fi:04d}_t{_safe_float_tag(tfi, nd=t_tag_ndigits)}.h5ad"
                ad_i.write_h5ad(os.path.join(out_dir, fname))

        elif save_which == "last_mid":
            last_idx = n_frames - 1
            last_t = float(t_arr[last_idx]) if (t_arr is not None and len(t_arr) == n_frames) else 1.0
            ad_last = rollout_frame_to_adata(
                r,
                last_idx,
                seg=seg,
                t=last_t,
                layer_names=layer_names,
                var_names=var_names,
                make_sparse=make_sparse,
            )
            _attach_common(ad_last, r, last_idx, n_frames)
            ad_last.write_h5ad(os.path.join(out_dir, f"{seg}_last.h5ad"))

            mid_idx, mid_t = pick_mid_frame_index(r, t_mid=0.5)
            ad_mid = rollout_frame_to_adata(
                r,
                mid_idx,
                seg=seg,
                t=mid_t,
                layer_names=layer_names,
                var_names=var_names,
                make_sparse=make_sparse,
            )
            _attach_common(ad_mid, r, mid_idx, n_frames)
            ad_mid.write_h5ad(os.path.join(out_dir, f"{seg}_mid_nearest_t0.5.h5ad"))

        else:
            raise ValueError("save_which must be 'all' or 'last_mid'")

    print("[done] saved to", out_dir)