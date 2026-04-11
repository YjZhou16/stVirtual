import numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import issparse, csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional

from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------- helpers --------
def _get_counts(adata, layer="counts"):
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X
    return X if issparse(X) else csr_matrix(X)

def _hill(L, R, KL, KR, n=1.0, eps=1e-8):
    # L/R: nonneg arrays, broadcasting supported
    Ln = np.power(np.clip(L, 0, None), n)
    Rn = np.power(np.clip(R, 0, None), n)
    return (Ln/(np.power(KL, n)+Ln+eps)) * (Rn/(np.power(KR, n)+Rn+eps))

def _prep_lr_pairs(lr_pairs: pd.DataFrame):
    df = lr_pairs.copy()
    # standardize column names
    assert {"ligand","receptor"}.issubset(df.columns), "lr_pairs 需要包含 'ligand' 和 'receptor' 列"
    if "sign" in df.columns:
        # map "+"/"-" → 1/-1
        df["sign"] = df["sign"].map({"+":1, "-":-1}).fillna(df["sign"]).astype(float)
    else:
        df["sign"] = 1.0
    df["weight"] = df.get("weight", 1.0)
    # default Hill params (可改：也可从表里读 KL/KR/n)
    df["KL"] = df.get("KL", 5.0)
    df["KR"] = df.get("KR", 5.0)
    df["hill_n"] = df.get("hill_n", 1.0)
    return df.reset_index(drop=True)

def _knn(coords, k=16):
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords, return_distance=True)
    # drop self
    return dist[:,1:], idx[:,1:]

def _z_by_sample(values: np.ndarray, samples: np.ndarray):
    out = np.zeros_like(values, dtype=float)
    for s in pd.unique(samples.astype(str)):
        mask = (samples == s)
        v = values[mask]
        mu, sd = np.nanmean(v), np.nanstd(v)
        sd = sd if sd > 1e-6 else 1.0
        out[mask] = (v - mu) / sd
    return out

# -------- main: compute LR potential per sample --------
def compute_lr_potential_per_sample(
    adata, lr_pairs,
    sample_key="sample", coords_key="spatial",
    layer="counts",
    k=16, use_cpm=True,
    hill=True, pair_chunk=64, verbose=False,
    # ---- new: quantile thresholds ----
    thr_q=0.7,                 # 非零分位数：0.5=中位数，0.7/0.8更严格
    min_thr=1.0,               # 阈值下限（raw counts 推荐 1.0；TP10k 也可 1~2）
    klkr_mode="max",           # "max"=max(base, data); "data"=只用data; "base"=只用表里KL/KR
    scale_target=1e4,          # use_cpm=True 时的目标库容量（TP10k）
):
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse, csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # ---------------- helpers (standalone) ----------------
    def _get_counts(adata, layer="counts"):
        X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
        return X if issparse(X) else csr_matrix(X)

    def _prep_lr_pairs(df: pd.DataFrame):
        df = df.copy()
        assert {"ligand", "receptor"}.issubset(df.columns), "lr_pairs 需要包含 ligand/receptor"
        if "sign" in df.columns:
            df["sign"] = df["sign"].map({"+": 1, "-": -1}).fillna(df["sign"]).astype(float)
        else:
            df["sign"] = 1.0
        df["weight"] = df.get("weight", 1.0)
        df["KL"] = df.get("KL", 5.0)
        df["KR"] = df.get("KR", 5.0)
        df["hill_n"] = df.get("hill_n", 1.0)
        return df.reset_index(drop=True)

    def _hill(L, R, KL, KR, n=1.0, eps=1e-8):
        Ln = np.power(np.clip(L, 0, None), n)
        Rn = np.power(np.clip(R, 0, None), n)
        return (Ln / (np.power(KL, n) + Ln + eps)) * (Rn / (np.power(KR, n) + Rn + eps))

    def _knn(coords, kk=16):
        nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
        nn.fit(coords)
        dist, idx = nn.kneighbors(coords, return_distance=True)
        return dist[:, 1:], idx[:, 1:]  # drop self

    def _z_by_sample(values: np.ndarray, samples: np.ndarray):
        out = np.zeros_like(values, dtype=float)
        for s in pd.unique(samples.astype(str)):
            m = (samples == s)
            v = values[m]
            mu, sd = np.nanmean(v), np.nanstd(v)
            sd = sd if sd > 1e-6 else 1.0
            out[m] = (v - mu) / sd
        return out

    def _per_gene_nonzero_quantile_scaled(Xs_csr, gene_idx, scale, q, min_thr):
        """
        对当前 sample 内指定基因集合 gene_idx（全局索引）：
          thr_g = quantile( (counts*scale)_{:,g} | >0 )
        """
        M = Xs_csr[:, gene_idx].toarray().astype(np.float32)   # (N, Guse)
        M = (M.T * scale).T                                    # scale per cell
        thr = np.full((len(gene_idx),), float(min_thr), dtype=np.float32)
        for t in range(len(gene_idx)):
            v = M[:, t]
            v = v[v > 0]
            if v.size:
                thr[t] = max(float(np.quantile(v, q)), float(min_thr))
        return thr

    # ---------------- main ----------------
    assert sample_key in adata.obs, f"obs['{sample_key}'] 不存在"
    assert coords_key in adata.obsm, f"obsm['{coords_key}'] 不存在"

    lrdf0 = _prep_lr_pairs(lr_pairs)
    X = _get_counts(adata, layer=layer)  # CSR
    var = np.array(adata.var_names)
    coords_all = np.asarray(adata.obsm[coords_key], dtype=float)
    samples = adata.obs[sample_key].astype(str).values
    uniq_samples = pd.unique(samples)

    # gene -> index；过滤不存在的 pair
    gene_to_idx = {g: i for i, g in enumerate(var)}
    lig_idx, rec_idx, keep = [], [], []
    for r in lrdf0.itertuples():
        iL = gene_to_idx.get(r.ligand, None)
        iR = gene_to_idx.get(r.receptor, None)
        if iL is not None and iR is not None:
            lig_idx.append(iL); rec_idx.append(iR); keep.append(True)
        else:
            keep.append(False)
    lrdf = lrdf0.loc[keep].reset_index(drop=True)
    lig_idx = np.array(lig_idx, dtype=int)
    rec_idx = np.array(rec_idx, dtype=int)
    P = len(lrdf)
    if verbose:
        print(f"[LR] 有效 LR 配对 {P} 条（在 var_names 中均存在）")

    U_all = np.zeros(adata.n_obs, dtype=np.float32)

    for s in uniq_samples:
        msk = (samples == s)
        idx = np.where(msk)[0]
        N = idx.size
        if N == 0:
            continue
        if verbose:
            print(f"  - sample '{s}': {N} cells")

        coords = coords_all[idx]
        dist, nbr = _knn(coords, kk=k)                           # (N,k), (N,k)
        sigma = np.clip(dist[:, -1], 1e-6, None)                 # (N,)
        K_sp = np.exp(-(dist**2) / (2.0 * (sigma[:, None]**2)))  # (N,k)

        Xs = X[idx, :]  # (N,G) CSR
        if use_cpm:
            lib = np.asarray(Xs.sum(axis=1)).ravel().astype(np.float32)
            lib = np.clip(lib, 1.0, None)
            scale = (float(scale_target) / lib).astype(np.float32)  # TP10k
        else:
            scale = np.ones(N, dtype=np.float32)

        # ---- per-gene quantile thresholds (scaled space) ----
        genes_use = np.unique(np.concatenate([lig_idx, rec_idx]))
        thr_use = _per_gene_nonzero_quantile_scaled(Xs, genes_use, scale, q=thr_q, min_thr=min_thr)
        pos = {int(g): t for g, t in zip(genes_use.tolist(), thr_use.tolist())}
        KL_data_all = np.array([pos[int(g)] for g in lig_idx], dtype=np.float32)  # (P,)
        KR_data_all = np.array([pos[int(g)] for g in rec_idx], dtype=np.float32)  # (P,)

        U_s = np.zeros(N, dtype=np.float64)

        for start in range(0, P, pair_chunk):
            end = min(start + pair_chunk, P)
            sl = slice(start, end)

            # dense block of (scaled) ligand/receptor expression
            L_mat = Xs[:, lig_idx[sl]].toarray().astype(np.float32)  # (N,p)
            R_mat = Xs[:, rec_idx[sl]].toarray().astype(np.float32)  # (N,p)
            L_mat = (L_mat.T * scale).T
            R_mat = (R_mat.T * scale).T
            L_mat = np.clip(L_mat, 0, None)
            R_mat = np.clip(R_mat, 0, None)

            sub = lrdf.iloc[sl]  # 注意：放这里，hill=False 也能用 weight/sign

            if hill:
                KL_base = sub["KL"].to_numpy(np.float32)
                KR_base = sub["KR"].to_numpy(np.float32)
                nH = sub["hill_n"].to_numpy(np.float32)

                KL_data = KL_data_all[sl]
                KR_data = KR_data_all[sl]

                if klkr_mode == "max":
                    KL_used = np.maximum(KL_base, KL_data)
                    KR_used = np.maximum(KR_base, KR_data)
                elif klkr_mode == "data":
                    KL_used = KL_data
                    KR_used = KR_data
                elif klkr_mode == "base":
                    KL_used = KL_base
                    KR_used = KR_base
                else:
                    raise ValueError("klkr_mode must be one of {'max','data','base'}")

                L_j = L_mat[nbr]          # (N,k,p)
                R_i = R_mat[:, None, :]   # (N,1,p)
                h = _hill(
                    L_j, R_i,
                    KL_used.reshape(1, 1, -1),
                    KR_used.reshape(1, 1, -1),
                    n=nH.reshape(1, 1, -1),
                )
            else:
                L_j = L_mat[nbr]
                R_i = R_mat[:, None, :]
                h_raw = L_j * R_i
                denom = np.maximum(np.percentile(h_raw, 99), 1e-6)
                h = np.clip(h_raw / denom, 0, 1)

            w = (sub["weight"].to_numpy(np.float32) * sub["sign"].to_numpy(np.float32))  # (p,)
            A = (h * w.reshape(1, 1, -1)).sum(axis=2)  # (N,k)

            U_s += np.sum(A * K_sp, axis=1).astype(np.float64)

        U_all[idx] = U_s.astype(np.float32)

    adata.obs["LR_potential"] = U_all
    adata.obs["LR_potential_z"] = _z_by_sample(U_all, samples)

    if verbose:
        print("[LR] 完成：写入 adata.obs['LR_potential'] 与 'LR_potential_z'")
    return adata


def compute_lr_from_decoder(
    adata, lr_pairs,
    sample_key="sample", coords_key="spatial",
    layer="counts",
    k=16, use_cpm=True,
    hill=True, pair_chunk=64, verbose=True,
    lib_power=0.5,                 
    lib_clip_q=(5,95),            
    combine_mode="zdiff",          
    hill_ref="fixed"             
):
    import numpy as np
    from scipy.sparse import csr_matrix, issparse
    from sklearn.neighbors import NearestNeighbors

    def _get_counts(adata, layer="counts"):
        X = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
        return X if issparse(X) else csr_matrix(X)

    def _prep_lr_pairs(df):
        df = df.copy()
        assert {"ligand","receptor"}.issubset(df.columns)
        if "sign" in df.columns:
            df["sign"] = df["sign"].map({"+":1, "-":-1}).fillna(df["sign"]).astype(float)
        else:
            df["sign"] = 1.0
        df["weight"] = df.get("weight", 1.0)
        df["KL"] = df.get("KL", 5.0); df["KR"] = df.get("KR", 5.0)
        df["hill_n"] = df.get("hill_n", 1.0)
        return df.reset_index(drop=True)

    def _hill(L, R, KL, KR, n=1.0, eps=1e-8):
        Ln = np.power(np.clip(L, 0, None), n)
        Rn = np.power(np.clip(R, 0, None), n)
        return (Ln/(np.power(KL, n)+Ln+eps)) * (Rn/(np.power(KR, n)+Rn+eps))

    def _z_by_sample(values: np.ndarray, samples: np.ndarray):
        out = np.zeros_like(values, dtype=float)
        for s in pd.unique(samples.astype(str)):
            msk = (samples == s)
            v = values[msk]
            mu, sd = np.nanmean(v), np.nanstd(v)
            sd = sd if sd > 1e-6 else 1.0
            out[msk] = (v - mu) / sd
        return out

    lrdf = _prep_lr_pairs(lr_pairs)
    X = _get_counts(adata, layer=layer)  # CSR
    var = np.array(adata.var_names)
    coords_all = np.asarray(adata.obsm[coords_key], dtype=float)
    samples = adata.obs[sample_key].astype(str).values
    uniq_samples = pd.unique(samples)

    gene_to_idx = {g:i for i,g in enumerate(var)}
    lig_idx, rec_idx, keep = [], [], []
    for r in lrdf.itertuples():
        iL = gene_to_idx.get(r.ligand, None)
        iR = gene_to_idx.get(r.receptor, None)
        if iL is not None and iR is not None:
            lig_idx.append(iL); rec_idx.append(iR); keep.append(True)
        else:
            keep.append(False)
    lrdf = lrdf.loc[keep].reset_index(drop=True)
    lig_idx = np.array(lig_idx, dtype=int)
    rec_idx = np.array(rec_idx, dtype=int)
    P = len(lrdf)
    if verbose: print(f"[LR] 有效 LR 配对 {P} 条")

    U_all = np.zeros(adata.n_obs, dtype=np.float32)
    U_all_pos = np.zeros_like(U_all)
    U_all_neg = np.zeros_like(U_all)

    for s in uniq_samples:
        msk = (samples == s)
        idx = np.where(msk)[0]
        N = idx.size
        if verbose: print(f"  - sample '{s}': {N} cells")

        coords = coords_all[idx]
        nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(coords)
        dist, nbr = nn.kneighbors(coords, return_distance=True)
        dist, nbr = dist[:,1:], nbr[:,1:]
        sigma = np.clip(dist[:, -1], 1e-6, None)
        K = np.exp(-(dist**2) / (2.0 * (sigma[:,None]**2)))

        Xs = X[idx, :]
        if use_cpm:
            lib = np.asarray(Xs.sum(axis=1)).ravel().astype(np.float32)
            if lib_clip_q is not None:
                ql, qh = np.percentile(lib, lib_clip_q)
                lib = np.clip(lib, max(1.0, ql), max(ql+1.0, qh))
            scale = (1e4 / lib)**float(lib_power)
        else:
            scale = np.ones(N, dtype=np.float32)

        U_s_pos = np.zeros(N, dtype=np.float64)
        U_s_neg = np.zeros(N, dtype=np.float64)

        for start in range(0, P, pair_chunk):
            end = min(start + pair_chunk, P)
            sl = slice(start, end)
            L_mat = Xs[:, lig_idx[sl]].toarray().astype(np.float32)
            R_mat = Xs[:, rec_idx[sl]].toarray().astype(np.float32)
            L_mat = (L_mat.T * scale).T;  R_mat = (R_mat.T * scale).T
            L_mat = np.clip(L_mat, 0, None); R_mat = np.clip(R_mat, 0, None)

            sub = lrdf.iloc[sl]
            if hill:
                KL = sub["KL"].to_numpy(np.float32)
                KR = sub["KR"].to_numpy(np.float32)
                nH = sub["hill_n"].to_numpy(np.float32)
                if hill_ref == "adaptive":
                    KL = np.maximum(KL, np.median(L_mat[L_mat>0]) if (L_mat>0).any() else 1.0)
                    KR = np.maximum(KR, np.median(R_mat[R_mat>0]) if (R_mat>0).any() else 1.0)
                L_j = L_mat[nbr]              
                R_i = R_mat[:,None,:]         
                h = _hill(L_j, R_i, KL.reshape(1,1,-1), KR.reshape(1,1,-1), n=nH.reshape(1,1,-1))
            else:
                L_j = L_mat[nbr]; R_i = R_mat[:,None,:]
                h_raw = L_j * R_i
                denom = np.maximum(np.percentile(h_raw, 99), 1e-6)
                h = np.clip(h_raw / denom, 0, 1)

            w = (sub["weight"].to_numpy(np.float32) * sub["sign"].to_numpy(np.float32))  # [p]
            w_pos = np.clip(w, 0, None); w_neg = np.clip(-w, 0, None)

            A_pos = (h * w_pos.reshape(1,1,-1)).sum(axis=2)   # [N,k]
            A_neg = (h * w_neg.reshape(1,1,-1)).sum(axis=2)

            U_s_pos += np.sum(A_pos * K, axis=1).astype(np.float64)
            U_s_neg += np.sum(A_neg * K, axis=1).astype(np.float64)

        if combine_mode == "sum":
            U_s = U_s_pos - U_s_neg
        elif combine_mode == "diff":
            U_s = (U_s_pos - U_s_neg)
        elif combine_mode == "zdiff":
            up = (U_s_pos - U_s_pos.mean()) / (U_s_pos.std()+1e-6)
            un = (U_s_neg - U_s_neg.mean()) / (U_s_neg.std()+1e-6)
            U_s = up - un
        else:
            raise ValueError("combine_mode not supported")

        U_all[idx]     = U_s.astype(np.float32)
        U_all_pos[idx] = U_s_pos.astype(np.float32)
        U_all_neg[idx] = U_s_neg.astype(np.float32)

    adata.obs["LR_pos"] = U_all_pos
    adata.obs["LR_neg"] = U_all_neg
    adata.obs["LR_potential"]   = U_all
    adata.obs["LR_potential_z"] = _z_by_sample(U_all, samples)
    if verbose:
        print("[LR] 完成：写入 obs['LR_pos'], ['LR_neg'], ['LR_potential(_z)']")

    return adata

def knn_self_chunked(
    coords: torch.Tensor,
    k: int = 16,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 GPU 上做 self-KNN，但是按 chunk 计算，避免 N×N 距离矩阵一次性占满显存。

    输入:
      coords: (N, d) GPU tensor
      k: 邻居数（不含自己）
      chunk_size: 每次处理的 cell 数（比如 1024/2048）

    输出:
      dist_all: (N, k) 每个点到 k 个近邻的距离
      idx_all:  (N, k) 每个点近邻的索引
    """
    device = coords.device
    N = coords.shape[0]
    k = min(k, N - 1)

    dist_all = torch.empty((N, k), device=device, dtype=coords.dtype)
    idx_all  = torch.empty((N, k), device=device, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = coords[start:end]                  # (B, d)

            # (B, N) 距离矩阵，B << N，显存可控
            D = torch.cdist(chunk, coords)

            vals, idx = torch.topk(D, k + 1, largest=False)  # 包含自己
            dist_all[start:end] = vals[:, 1:]                # 去掉自己
            idx_all[start:end]  = idx[:, 1:]

            del D, vals, idx
            torch.cuda.empty_cache()

    return dist_all, idx_all

def compute_lr_potential_gpu(
    expr_union_t: torch.Tensor,  # (N, G) decoder 输出的表达，GPU tensor
    coords_t: torch.Tensor,      # (N, 2) 当前步的归一化坐标，GPU tensor
    lr_cfg: Dict[str, torch.Tensor],
    k: int = 16,
    use_cpm: bool = True,
    lib_power: float = 0.5,
    lib_clip_q: Tuple[float, float] = (5.0, 95.0),
    hill: bool = True,
    combine_mode: str = "zdiff",
    pair_chunk: int = 64,
    knn_chunk_size: int = 2048,
) -> torch.Tensor:
    """
    单 sample 的 GPU 版 LR_potential 计算（分块 KNN 版，不会分配 N×N 距离矩阵）。

    步骤：
      1）分块 self-KNN 得到 dist,nbr (N,k)
      2）可选 CPM-like 缩放
      3）按 LR 配对分 chunk 做 Hill / dot / 加权
    """
    device = expr_union_t.device
    N = expr_union_t.shape[0]
    if N == 0:
        return torch.zeros(0, device=device)

    # ---------- 1) 分块 self-KNN ----------
    dist, nbr = knn_self_chunked(coords_t, k=k, chunk_size=knn_chunk_size)
    # sigma 用第 k 个邻居距离，和你原来一样
    sigma = torch.clamp(dist[:, -1], min=1e-6)        # (N,)
    K = torch.exp(-dist**2 / (2.0 * sigma[:, None]**2))  # (N, k)

    # ---------- 2) CPM-like 库容量缩放 ----------
    if use_cpm:
        lib = expr_union_t.sum(dim=1).to(torch.float32)  # (N,)
        if lib_clip_q is not None:
            ql, qh = torch.quantile(
                lib,
                torch.tensor(
                    [lib_clip_q[0] / 100.0, lib_clip_q[1] / 100.0],
                    device=device,
                ),
            )
            ql = torch.maximum(ql, torch.tensor(1.0, device=device))
            qh = torch.maximum(qh, ql + 1.0)
            lib = torch.clamp(lib, min=ql, max=qh)
        scale = (1e4 / lib).pow(float(lib_power))  # (N,)
    else:
        scale = torch.ones(N, device=device)

    # ---------- 3) 提取 L/R 表达 ----------
    lig_idx = lr_cfg["lig_idx"]     # (P,)
    rec_idx = lr_cfg["rec_idx"]     # (P,)
    KL      = lr_cfg["KL"]          # (P,)
    KR      = lr_cfg["KR"]          # (P,)
    nH      = lr_cfg["nH"]          # (P,)
    w_pos   = lr_cfg["w_pos"]       # (P,)
    w_neg   = lr_cfg["w_neg"]       # (P,)

    L_full = expr_union_t[:, lig_idx] * scale.view(-1, 1)   # (N, P)
    R_full = expr_union_t[:, rec_idx] * scale.view(-1, 1)   # (N, P)

    U_pos = torch.zeros(N, device=device, dtype=torch.float32)
    U_neg = torch.zeros(N, device=device, dtype=torch.float32)

    P = L_full.shape[1]
    for start in range(0, P, pair_chunk):
        end = min(start + pair_chunk, P)
        L_mat = L_full[:, start:end]  # (N, p)
        R_mat = R_full[:, start:end]  # (N, p)

        # KNN 组合：L_j(i,邻居,p), R_i(i,1,p)
        L_j = L_mat[nbr]               # (N, k, p)
        R_i = R_mat.unsqueeze(1)       # (N, 1, p)

        if hill:
            KL_chunk = KL[start:end].view(1, 1, -1)   # (1,1,p)
            KR_chunk = KR[start:end].view(1, 1, -1)
            nH_chunk = nH[start:end].view(1, 1, -1)

            L_clip = torch.clamp(L_j, min=0.0)
            R_clip = torch.clamp(R_i, min=0.0)
            Ln = L_clip.pow(nH_chunk)
            Rn = R_clip.pow(nH_chunk)

            h = (Ln / (KL_chunk.pow(nH_chunk) + Ln + 1e-8)) * \
                (Rn / (KR_chunk.pow(nH_chunk) + Rn + 1e-8))  # (N,k,p)
        else:
            h_raw = L_j * R_i   # (N,k,p)
            denom = torch.quantile(h_raw.reshape(-1), 0.99)
            denom = torch.clamp(denom, min=1e-6)
            h = torch.clamp(h_raw / denom, min=0.0, max=1.0)

        w_pos_c = w_pos[start:end].view(1, 1, -1)
        w_neg_c = w_neg[start:end].view(1, 1, -1)

        A_pos = (h * w_pos_c).sum(dim=2)   # (N, k)
        A_neg = (h * w_neg_c).sum(dim=2)

        U_pos = U_pos + (A_pos * K).sum(dim=1)
        U_neg = U_neg + (A_neg * K).sum(dim=1)

        del L_mat, R_mat, L_j, R_i, h, A_pos, A_neg
        torch.cuda.empty_cache()

    # ---------- 4) combine_mode ----------
    if combine_mode == "sum":
        U = U_pos - U_neg
    elif combine_mode == "diff":
        U = U_pos - U_neg
    elif combine_mode == "zdiff":
        up = (U_pos - U_pos.mean()) / (U_pos.std() + 1e-6)
        un = (U_neg - U_neg.mean()) / (U_neg.std() + 1e-6)
        U = up - un
    else:
        raise ValueError(f"combine_mode '{combine_mode}' not supported")

    return U.to(torch.float32)  # (N,)

def render_lr_potential_smooth(
    adata,
    key="LR_potential_z",
    sample_key="sample",
    coords_key="spatial",
    obs_xy=None,
    bins=256,
    sigma_mode="knn",
    k=16,
    sigma_px=None,
    invert_y=True,
    cmap="viridis",
    obs_out=None,                 # 平滑值写回列名，默认 f"{key}_smooth"
    write_weight=True,
    thin_mask_percentile=5.0,
    show_plot=True,
    verbose=True,

    # ---------- NEW: 0-1 normalization ----------
    norm_out=None,                # 归一化写回列名，默认 f"{obs_out}_01"
    norm_mode="minmax",           # "minmax" or "quantile"
    norm_q=(2.0, 98.0),           # norm_mode="quantile" 时用
    norm_scope="per_sample",      # "per_sample" or "global"
    plot_use_norm=True,           # 画图是否用归一化场
):
    """
    对每个 sample：
      1) (x,y,v) -> W_grid / VW_grid
      2) 高斯平滑 -> W_s / VW_s
      3) field = VW_s / (W_s + eps)
      4) 双线性采样写回 obs_out
      5) NEW: 生成 0-1 归一化列 norm_out，并且画图用 norm_field（可选）
    """
    assert sample_key in adata.obs, f"obs['{sample_key}'] not found"
    if obs_out is None:
        obs_out = f"{key}_smooth"
    if norm_out is None:
        norm_out = f"{obs_out}_01"

    # ---- coords ----
    if obs_xy is not None:
        x_all = np.asarray(adata.obs[obs_xy[0]].values, dtype=float)
        y_all = np.asarray(adata.obs[obs_xy[1]].values, dtype=float)
    elif coords_key is not None and hasattr(adata, "obsm") and (coords_key in getattr(adata, "obsm_keys", lambda: [])()):
        xy = np.asarray(adata.obsm[coords_key], dtype=float)
        assert xy.shape[1] >= 2, f"obsm['{coords_key}'] must have at least 2 columns"
        x_all, y_all = xy[:, 0], xy[:, 1]
    else:
        raise ValueError("请提供 obs_xy 或有效的 coords_key 来获取坐标")

    vals_all = np.asarray(adata.obs[key].values, dtype=float)
    samples  = adata.obs[sample_key].astype(str).values
    uniq_samples = pd.unique(samples)

    if isinstance(bins, int):
        H = W = int(bins)
    else:
        H, W = int(bins[0]), int(bins[1])

    # ---- init output cols ----
    adata.obs[obs_out] = np.nan
    adata.obs[norm_out] = np.nan
    if write_weight:
        adata.obs[f"{obs_out}_w"] = np.nan

    eps = 1e-8

    def _bilinear_sample(field, xs, ys, xmin, xmax, ymin, ymax):
        HH, WW = field.shape
        u = (xs - xmin) / (xmax - xmin + 1e-12) * (WW - 1)
        v = (ys - ymin) / (ymax - ymin + 1e-12) * (HH - 1)
        ok = (u >= 0) & (u <= WW - 1) & (v >= 0) & (v <= HH - 1)
        out = np.full_like(xs, np.nan, dtype=float)
        if not np.any(ok):
            return out

        u0 = np.floor(u[ok]).astype(int)
        v0 = np.floor(v[ok]).astype(int)
        u1 = np.clip(u0 + 1, 0, WW - 1)
        v1 = np.clip(v0 + 1, 0, HH - 1)
        du = u[ok] - u0
        dv = v[ok] - v0

        f00 = field[v0, u0]
        f01 = field[v1, u0]
        f10 = field[v0, u1]
        f11 = field[v1, u1]
        out_ok = (1 - du) * (1 - dv) * f00 + (1 - du) * dv * f01 + du * (1 - dv) * f10 + du * dv * f11
        out[ok] = out_ok
        return out

    # --- for global norm stats (optional) ---
    global_vmin = np.inf
    global_vmax = -np.inf

    # 第一遍：如果要 global 范围，就先统计 smooth 的范围（按你的 mask 逻辑）
    if norm_scope == "global":
        for sname in uniq_samples:
            m = (samples == sname)
            x = x_all[m]; y = y_all[m]; v = vals_all[m]
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
            x, y, v = x[ok], y[ok], v[ok]
            if x.size == 0:
                continue

            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            pad_x = 0.02 * (xmax - xmin + 1e-8)
            pad_y = 0.02 * (ymax - ymin + 1e-8)
            xmin, xmax = xmin - pad_x, xmax + pad_x
            ymin, ymax = ymin - pad_y, ymax + pad_y

            W_grid  = np.histogram2d(y, x, bins=[H, W], range=[[ymin, ymax], [xmin, xmax]])[0]
            VW_grid = np.histogram2d(y, x, bins=[H, W], range=[[ymin, ymax], [xmin, xmax]], weights=v)[0]

            if sigma_mode == "knn":
                nnk = min(int(k) + 1, max(2, len(x)))
                nn = NearestNeighbors(n_neighbors=nnk).fit(np.c_[x, y])
                d, _ = nn.kneighbors()
                kk = min(int(k), d.shape[1] - 1)
                sig = float(np.median(d[:, kk])) + 1e-12
                sigma_x = sig / (xmax - xmin + 1e-8) * W
                sigma_y = sig / (ymax - ymin + 1e-8) * H
            else:
                if sigma_px is None:
                    base = max(1.0, min(H, W) / 60.0)
                    sigma_y = sigma_x = base
                else:
                    if np.isscalar(sigma_px):
                        sigma_y = sigma_x = float(sigma_px)
                    else:
                        sigma_y, sigma_x = float(sigma_px[0]), float(sigma_px[1])

            W_s  = gaussian_filter(W_grid,  sigma=(sigma_y, sigma_x), mode="nearest")
            VW_s = gaussian_filter(VW_grid, sigma=(sigma_y, sigma_x), mode="nearest")
            field = VW_s / (W_s + eps)

            if (W_s > 0).any():
                th = np.percentile(W_s[W_s > 0], thin_mask_percentile)
                mask = (W_s < th)
            else:
                mask = np.ones_like(W_s, dtype=bool)

            valid = (~mask) & np.isfinite(field)
            if not valid.any():
                continue

            if norm_mode == "quantile":
                qlo, qhi = np.array(norm_q, dtype=float) / 100.0
                vmin, vmax = np.quantile(field[valid], [qlo, qhi])
            else:
                vmin, vmax = np.nanmin(field[valid]), np.nanmax(field[valid])

            if np.isfinite(vmin): global_vmin = min(global_vmin, float(vmin))
            if np.isfinite(vmax): global_vmax = max(global_vmax, float(vmax))

        if not np.isfinite(global_vmin) or not np.isfinite(global_vmax) or global_vmax <= global_vmin + 1e-12:
            global_vmin, global_vmax = 0.0, 1.0

    # 第二遍：正式写回
    for sname in uniq_samples:
        m = (samples == sname)
        x = x_all[m]; y = y_all[m]; v = vals_all[m]
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        x, y, v = x[ok], y[ok], v[ok]
        idx_obs = np.where(m)[0][ok]
        if x.size == 0:
            if verbose:
                print(f"[warn] sample={sname} 没有有效点，跳过")
            continue

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        pad_x = 0.02 * (xmax - xmin + 1e-8)
        pad_y = 0.02 * (ymax - ymin + 1e-8)
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y

        W_grid  = np.histogram2d(y, x, bins=[H, W], range=[[ymin, ymax], [xmin, xmax]])[0]
        VW_grid = np.histogram2d(y, x, bins=[H, W], range=[[ymin, ymax], [xmin, xmax]], weights=v)[0]

        if sigma_mode == "knn":
            nnk = min(int(k) + 1, max(2, len(x)))
            nn = NearestNeighbors(n_neighbors=nnk).fit(np.c_[x, y])
            d, _ = nn.kneighbors()
            kk = min(int(k), d.shape[1] - 1)
            sig = float(np.median(d[:, kk])) + 1e-12
            sigma_x = sig / (xmax - xmin + 1e-8) * W
            sigma_y = sig / (ymax - ymin + 1e-8) * H
        else:
            if sigma_px is None:
                base = max(1.0, min(H, W) / 60.0)
                sigma_y = sigma_x = base
            else:
                if np.isscalar(sigma_px):
                    sigma_y = sigma_x = float(sigma_px)
                else:
                    sigma_y, sigma_x = float(sigma_px[0]), float(sigma_px[1])

        W_s  = gaussian_filter(W_grid,  sigma=(sigma_y, sigma_x), mode="nearest")
        VW_s = gaussian_filter(VW_grid, sigma=(sigma_y, sigma_x), mode="nearest")
        field = VW_s / (W_s + eps)

        # mask（用于 plot & 用于 quantile norm）
        if (W_s > 0).any():
            th = np.percentile(W_s[W_s > 0], thin_mask_percentile)
            mask = (W_s < th)
        else:
            mask = np.ones_like(W_s, dtype=bool)

        valid = (~mask) & np.isfinite(field)

        # ---- per-sample norm stats ----
        if norm_scope == "per_sample":
            if valid.any():
                if norm_mode == "quantile":
                    qlo, qhi = np.array(norm_q, dtype=float) / 100.0
                    vmin, vmax = np.quantile(field[valid], [qlo, qhi])
                else:
                    vmin, vmax = np.nanmin(field[valid]), np.nanmax(field[valid])
            else:
                vmin, vmax = np.nanmin(field), np.nanmax(field)
        else:
            vmin, vmax = global_vmin, global_vmax

        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin + 1e-12: vmax = vmin + 1.0

        # ---- build norm_field for plotting ----
        norm_field = (np.clip(field, vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
        # 可选：把低密度区域置 NaN（和你原 show_plot 一致）
        norm_field_vis = norm_field.copy()
        norm_field_vis[~valid] = np.nan

        # ---- 写回 smooth ----
        vals_smooth = _bilinear_sample(field, x, y, xmin, xmax, ymin, ymax)
        adata.obs.iloc[idx_obs, adata.obs.columns.get_loc(obs_out)] = vals_smooth

        # ---- 写回 0-1 ----
        vals_01 = (np.clip(vals_smooth, vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
        adata.obs.iloc[idx_obs, adata.obs.columns.get_loc(norm_out)] = vals_01

        if write_weight:
            w_samp = _bilinear_sample(W_s, x, y, xmin, xmax, ymin, ymax)
            adata.obs.iloc[idx_obs, adata.obs.columns.get_loc(f"{obs_out}_w")] = w_samp

        # ---- plot ----
        if show_plot:
            fig_w = 4.2
            fig_h = fig_w * (H / W)
            fig, ax = plt.subplots(figsize=(fig_w + 0.55, fig_h), dpi=100)

            fig.patch.set_alpha(0)     # 整个 figure 背景透明
            ax.patch.set_alpha(0)      # 坐标轴背景透明

            if plot_use_norm:
                img = norm_field_vis
                vmin_p, vmax_p = 0.0, 1.0
            else:
                field_vis = field.copy()
                field_vis[~valid] = np.nan
                img = field_vis
                vmin_p, vmax_p = np.nanmin(img), np.nanmax(img)

            cm = plt.get_cmap(cmap).copy()
            cm.set_bad((1, 1, 1, 0))

            img_ma = np.ma.masked_invalid(img)
            im = ax.imshow(
                img_ma,
                origin="lower",
                interpolation="bilinear",
                cmap=cm,
                vmin=vmin_p, vmax=vmax_p,
                aspect="equal",
            )

            if invert_y:
                ax.invert_yaxis()
            ax.set_axis_off()

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4.5%", pad=0.12)
            cax.patch.set_alpha(0)     # colorbar 背景也透明
            cb = fig.colorbar(im, cax=cax)

            cb.set_label("")
            cb.ax.tick_params(labelsize=8, length=2, width=0.6)
            cb.outline.set_linewidth(0.6)

            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.show()

    return adata


