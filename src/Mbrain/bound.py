import os
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull, cKDTree
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.path import Path as MplPath

def expand_shell_centroid(shell: np.ndarray, expand: float) -> np.ndarray:
    shell = np.asarray(shell, np.float32)
    c = shell.mean(0, keepdims=True)
    return c + (shell - c) * float(expand)

def inside_frac(xy: np.ndarray, shell: np.ndarray) -> tuple[float, np.ndarray]:
    xy = np.asarray(xy, np.float32); shell = np.asarray(shell, np.float32)
    poly = MplPath(shell)
    inside = poly.contains_points(xy)
    return float(inside.mean()), np.where(~inside)[0]

def save_shell_csv(shell: np.ndarray, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x_norm": shell[:, 0], "y_norm": shell[:, 1]}).to_csv(csv_path, index=False)

def plot_shell_coverage(xy: np.ndarray, shell: np.ndarray, out_idx: np.ndarray, png_path: Path, *, max_pts=80000, s_in=1, s_out=6):
    xy = np.asarray(xy, np.float32); shell = np.asarray(shell, np.float32)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    N = xy.shape[0]
    if N > max_pts:
        rng = np.random.default_rng(0)
        keep = rng.choice(N, size=max_pts, replace=False)
        xy_plot = xy[keep]
        out_mask = np.isin(keep, out_idx)
        out_plot = xy_plot[out_mask]
        in_plot = xy_plot[~out_mask]
    else:
        out_plot = xy[out_idx]
        mask = np.ones(N, dtype=bool); mask[out_idx] = False
        in_plot = xy[mask]
    plt.figure(figsize=(6, 6))
    if in_plot.shape[0] > 0: plt.scatter(in_plot[:, 0], in_plot[:, 1], s=s_in, alpha=0.25)
    if out_plot.shape[0] > 0: plt.scatter(out_plot[:, 0], out_plot[:, 1], s=s_out, alpha=0.9)
    plt.plot(np.r_[shell[:, 0], shell[0, 0]], np.r_[shell[:, 1], shell[0, 1]], linewidth=2)
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(png_path, dpi=200, transparent=True)
    plt.close()

def make_shell_adaptive(
    xy: np.ndarray, *, alpha_factor: float, seed: int, n_resample: int = 256,
    target_frac: float = 0.99, expand0: float = 1.02, expand_step: float = 0.01, expand_max: float = 1.25,
    fallback_expand: float = 1.05
):
    shell0 = alpha_shell_delaunay(xy, alpha=None, auto_alpha=True, alpha_factor=float(alpha_factor), n_resample=int(n_resample), seed=int(seed))
    expand = float(expand0)
    shell = expand_shell_centroid(shell0, expand)
    frac, out_idx = inside_frac(xy, shell)
    it = 0
    while frac < float(target_frac) and expand < float(expand_max):
        it += 1
        expand = min(float(expand_max), expand + float(expand_step))
        shell = expand_shell_centroid(shell0, expand)
        frac, out_idx = inside_frac(xy, shell)
        if it > 200: break
    # 兜底：还不够就 convex（避免被极少数离群点/多团块逼到很大 expand）
    if frac < float(target_frac):
        try:
            shell = convex_shell(xy, expand=float(fallback_expand), n_resample=int(n_resample))
            frac, out_idx = inside_frac(xy, shell)
            expand = np.nan  # 表示走了 convex fallback
        except Exception:
            pass
    return shell, frac, out_idx, expand

def _poly_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _resample_closed_polyline(xy: np.ndarray, n: int = 256) -> np.ndarray:
    xy = np.asarray(xy, np.float32)
    if xy.shape[0] < 3:
        return xy
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])

    seg = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(1))
    L = float(seg.sum())
    if L < 1e-8:
        return xy[:-1]

    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, L, n + 1)[:-1]

    out = np.zeros((n, 2), np.float32)
    j = 0
    for i, ti in enumerate(t):
        while j + 1 < len(s) and s[j + 1] < ti:
            j += 1
        w = (ti - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out[i] = (1 - w) * xy[j] + w * xy[j + 1]
    return out

def downsample_points_grid(xy: np.ndarray, max_n: int = 8000, grid: int = 256, seed: int = 0) -> np.ndarray:

    xy = np.asarray(xy, np.float32)
    N = xy.shape[0]
    if N <= max_n:
        return xy

    xmin, ymin = xy.min(0)
    xmax, ymax = xy.max(0)
    bw = max(xmax - xmin, 1e-6)
    bh = max(ymax - ymin, 1e-6)

    gx = np.floor((xy[:, 0] - xmin) / bw * grid).astype(np.int32)
    gy = np.floor((xy[:, 1] - ymin) / bh * grid).astype(np.int32)
    gx = np.clip(gx, 0, grid - 1)
    gy = np.clip(gy, 0, grid - 1)
    key = gy.astype(np.int64) * grid + gx.astype(np.int64)

    _, idx = np.unique(key, return_index=True)
    xy2 = xy[idx]

    if xy2.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        pick = rng.choice(xy2.shape[0], size=max_n, replace=False)
        xy2 = xy2[pick]
    return xy2


def convex_shell(xy: np.ndarray, expand: float = 1.08, n_resample: int = 256) -> np.ndarray:
    xy = np.asarray(xy, np.float32)
    if xy.shape[0] < 3:
        return xy
    hull = ConvexHull(xy)
    poly = xy[hull.vertices]
    c = poly.mean(0, keepdims=True)
    poly = c + (poly - c) * float(expand)
    return _resample_closed_polyline(poly, n=n_resample)

def _triangle_circumradius(a, b, c):
    ab = b - a
    ac = c - a
    bc = c - b
    lab = np.sqrt((ab * ab).sum())
    lac = np.sqrt((ac * ac).sum())
    lbc = np.sqrt((bc * bc).sum())
    area = 0.5 * np.abs(np.cross(ab, ac))
    if area < 1e-12:
        return np.inf
    return (lab * lac * lbc) / (4.0 * area)

def _edges_from_tri(tri):
    i, j, k = tri
    return [(i, j), (j, k), (k, i)]

def _build_loops_from_boundary_edges(edges):
    # adjacency
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited = set()
    loops = []

    # helper: pick next neighbor not visited edge
    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    # build loops
    for start in adj.keys():
        # start from a node that still has unvisited incident edges
        has_unvisited = any(edge_key(start, nb) not in visited for nb in adj[start])
        if not has_unvisited:
            continue

        # walk
        loop = [start]
        prev = None
        cur = start

        while True:
            # choose next
            nbs = adj.get(cur, [])
            # prefer continuing direction: pick neighbor whose edge not visited
            nxt = None
            for nb in nbs:
                ek = edge_key(cur, nb)
                if ek not in visited:
                    nxt = nb
                    break
            if nxt is None:
                break

            visited.add(edge_key(cur, nxt))
            prev, cur = cur, nxt

            if cur == start:
                break
            loop.append(cur)

        if len(loop) >= 3 and loop[0] == start:
            # sometimes loop ends with start already; keep without duplicate
            if loop[-1] == start:
                loop = loop[:-1]
            loops.append(loop)

    return loops

def suggest_alpha_from_knn(xy: np.ndarray, k: int = 10, factor: float = 2.5, n_probe: int = 5000, seed: int = 0) -> float:
    xy = np.asarray(xy, np.float32)
    N = xy.shape[0]
    if N <= k + 2:
        return 1.0

    if N > n_probe:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_probe, replace=False)
        pts = xy[idx]
    else:
        pts = xy

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=min(k + 1, pts.shape[0]))  
    dk = np.median(dists[:, -1]) 
    R_th = max(float(factor * dk), 1e-6)
    return 1.0 / R_th

def alpha_shell_delaunay(
    xy: np.ndarray,
    *,
    alpha: float = None,
    auto_alpha: bool = True,
    alpha_factor: float = 2.5,
    downsample_max: int = 8000,
    downsample_grid: int = 256,
    expand_fallback: float = 1.03,
    n_resample: int = 256,
    seed: int = 0,
):

    xy = np.asarray(xy, np.float32)
    if xy.shape[0] < 10:
        return convex_shell(xy, expand=expand_fallback, n_resample=n_resample)

    # downsample to make Delaunay feasible
    xy_ds = downsample_points_grid(xy, max_n=downsample_max, grid=downsample_grid, seed=seed)

    # choose alpha
    if alpha is None and auto_alpha:
        alpha = suggest_alpha_from_knn(xy_ds, k=10, factor=alpha_factor, seed=seed)
    if alpha is None:
        alpha = 1.0

    # Delaunay
    try:
        tri = Delaunay(xy_ds)
    except Exception as e:
        print("[alpha_shell] Delaunay failed, fallback convex. err=", repr(e))
        return convex_shell(xy, expand=expand_fallback, n_resample=n_resample)

    # filter triangles by circumradius < 1/alpha
    R_th = 1.0 / max(float(alpha), 1e-12)

    simplices = tri.simplices  # (T,3)
    keep_tris = []
    for t in simplices:
        a, b, c = xy_ds[t[0]], xy_ds[t[1]], xy_ds[t[2]]
        R = _triangle_circumradius(a, b, c)
        if R < R_th:
            keep_tris.append(t)

    if len(keep_tris) < 5:
        return convex_shell(xy, expand=expand_fallback, n_resample=n_resample)

    # boundary edges = edges appearing only once among kept triangles
    edge_count = {}
    for t in keep_tris:
        for u, v in _edges_from_tri(t):
            ek = (u, v) if u < v else (v, u)
            edge_count[ek] = edge_count.get(ek, 0) + 1

    boundary = [ek for ek, c in edge_count.items() if c == 1]
    if len(boundary) < 10:
        return convex_shell(xy, expand=expand_fallback, n_resample=n_resample)

    # build loops, pick the largest area loop
    loops = _build_loops_from_boundary_edges(boundary)
    if len(loops) == 0:
        return convex_shell(xy, expand=expand_fallback, n_resample=n_resample)

    best_poly = None
    best_area = -1
    for loop in loops:
        poly = xy_ds[np.array(loop, dtype=np.int64)]
        if poly.shape[0] < 3:
            continue
        area = _poly_area(poly)
        if area > best_area:
            best_area = area
            best_poly = poly

    if best_poly is None or best_poly.shape[0] < 3:
        return convex_shell(xy, expand=expand_fallback, n_resample=n_resample)

    # resample for stability
    shell = _resample_closed_polyline(best_poly, n=n_resample)
    return shell


# -----------------------------
# batch: frames -> shells, and save csv
# -----------------------------
def compute_shells_for_frames(
    coords_frames,
    *,
    bound_dir: str = None,
    z_offset: int = 20,
    use_lib_alphashape: bool = False,
    alpha: float = None,
    auto_alpha: bool = True,
    alpha_factor: float = 2.5,
    n_resample: int = 256,
    seed: int = 0,
    expand: float = 1.10,
):
    shells = []
    if bound_dir is not None:
        os.makedirs(bound_dir, exist_ok=True)

    for k, xy in enumerate(coords_frames):
        xy = np.asarray(xy, np.float32)

        shell = alpha_shell_delaunay(
            xy, alpha=alpha, auto_alpha=auto_alpha, alpha_factor=alpha_factor,
            n_resample=n_resample, seed=seed + k
        )
        shell = shell.mean(0, keepdims=True) + (shell - shell.mean(0, keepdims=True)) * float(expand)
        shells.append(shell)

        if bound_dir is not None:
            csv_path = os.path.join(bound_dir, f"bound_z{z_offset + k:03d}.csv")
            pd.DataFrame({"x_norm": shell[:, 0], "y_norm": shell[:, 1]}).to_csv(csv_path, index=False)

    return shells

def debug_plot_shell(xy: np.ndarray, shell: np.ndarray, save_path: str, s=1):
    import matplotlib.pyplot as plt
    xy = np.asarray(xy, np.float32)
    shell = np.asarray(shell, np.float32)
    plt.figure(figsize=(6, 6))
    plt.scatter(xy[:, 0], xy[:, 1], s=s, alpha=0.4)
    plt.plot(np.r_[shell[:, 0], shell[0, 0]], np.r_[shell[:, 1], shell[0, 1]], linewidth=2)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, transparent=True)
    plt.close()
