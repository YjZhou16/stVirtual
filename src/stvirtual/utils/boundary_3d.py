import os
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def compute_bbox_3d_from_frames(coords_frames, margin=0.05):
    all_xyz = np.concatenate(coords_frames, axis=0).astype(np.float32)
    mn = all_xyz.min(0)
    mx = all_xyz.max(0)
    span = np.maximum(mx - mn, 1e-6)

    x0 = mn[0] - margin * span[0]
    y0 = mn[1] - margin * span[1]
    z0 = mn[2] - margin * span[2]

    x1 = mx[0] + margin * span[0]
    y1 = mx[1] + margin * span[1]
    z1 = mx[2] + margin * span[2]
    return (float(x0), float(y0), float(z0), float(x1), float(y1), float(z1))


import numpy as np
from scipy import ndimage as ndi

def voxelize_points_3d_fixed_bbox_dense(
    xyz: np.ndarray,
    *,
    bbox,
    D: int,
    H: int,
    W: int,
    splat_radius: int = 1,
    dilate_iter: int = 1,
    close_iter: int = 2,
    fill_holes: bool = True,
    keep_lcc: bool = False,
    min_count: int = 1,
):
    xyz = np.asarray(xyz, np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3

    x0, y0, z0, x1, y1, z1 = bbox
    dx = max((x1 - x0) / W, 1e-6)
    dy = max((y1 - y0) / H, 1e-6)
    dz = max((z1 - z0) / D, 1e-6)

    gx = np.floor((xyz[:, 0] - x0) / dx).astype(np.int32)
    gy = np.floor((xyz[:, 1] - y0) / dy).astype(np.int32)
    gz = np.floor((xyz[:, 2] - z0) / dz).astype(np.int32)

    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H) & (gz >= 0) & (gz < D)
    gx, gy, gz = gx[ok], gy[ok], gz[ok]

    occ = np.zeros((D, H, W), dtype=np.uint16)
    np.add.at(occ, (gz, gy, gx), 1)

    mask = occ >= int(min_count)

    if splat_radius > 0:
        rad = int(splat_radius)
        structure = np.ones((2 * rad + 1, 2 * rad + 1, 2 * rad + 1), dtype=bool)
        mask = ndi.binary_dilation(mask, structure=structure)

    if dilate_iter > 0:
        mask = ndi.binary_dilation(mask, iterations=int(dilate_iter))

    if close_iter > 0:
        mask = ndi.binary_closing(mask, iterations=int(close_iter))

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    if keep_lcc and mask.any():
        lab, nlab = ndi.label(mask)
        if nlab > 1:
            cnt = np.bincount(lab.ravel())
            cnt[0] = 0
            mask = (lab == cnt.argmax())

    return {
        "mask": mask.astype(np.uint8),
        "occ": occ,
        "x0": float(x0), "y0": float(y0), "z0": float(z0),
        "dx": float(dx), "dy": float(dy), "dz": float(dz),
    }




def save_voxel_preview_png(pack, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = pack["mask"].astype(bool)   # (D,H,W)

    proj_xy = m.max(axis=0)  # (H,W)
    proj_xz = m.max(axis=1)  # (D,W)
    proj_yz = m.max(axis=2)  # (D,H)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    axes[0].imshow(proj_xy, origin="lower", cmap="gray")
    axes[0].set_title("max proj: XY")
    axes[1].imshow(proj_xz, origin="lower", cmap="gray", aspect="auto")
    axes[1].set_title("max proj: XZ")
    axes[2].imshow(proj_yz, origin="lower", cmap="gray", aspect="auto")
    axes[2].set_title("max proj: YZ")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

def voxelize_points_3d(
    xyz: np.ndarray,
    *,
    D: int,
    H: int,
    W: int,
    margin: float = 0.05,
    dilate_iter: int = 1,
    close_iter: int = 1,
    keep_lcc: bool = True,
):
    xyz = np.asarray(xyz, np.float32)
    assert xyz.ndim == 2 and xyz.shape[1] == 3

    mn = xyz.min(0)
    mx = xyz.max(0)
    span = np.maximum(mx - mn, 1e-6)

    x0 = mn[0] - margin * span[0]
    y0 = mn[1] - margin * span[1]
    z0 = mn[2] - margin * span[2]

    dx = span[0] * (1 + 2 * margin) / W
    dy = span[1] * (1 + 2 * margin) / H
    dz = span[2] * (1 + 2 * margin) / D

    gx = np.floor((xyz[:, 0] - x0) / dx).astype(np.int32)
    gy = np.floor((xyz[:, 1] - y0) / dy).astype(np.int32)
    gz = np.floor((xyz[:, 2] - z0) / dz).astype(np.int32)

    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H) & (gz >= 0) & (gz < D)
    gx, gy, gz = gx[ok], gy[ok], gz[ok]

    mask = np.zeros((D, H, W), dtype=bool)
    mask[gz, gy, gx] = True

    if dilate_iter > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate_iter)

    if close_iter > 0:
        mask = ndi.binary_closing(mask, iterations=close_iter)

    mask = ndi.binary_fill_holes(mask)

    if keep_lcc and mask.any():
        lab, nlab = ndi.label(mask)
        if nlab > 1:
            cnt = np.bincount(lab.ravel())
            cnt[0] = 0
            mask = (lab == cnt.argmax())

    return {
        "mask": mask.astype(np.uint8),
        "x0": float(x0), "y0": float(y0), "z0": float(z0),
        "dx": float(dx), "dy": float(dy), "dz": float(dz),
    }

def save_voxel_volume(pack, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mask=pack["mask"],
        x0=pack["x0"], y0=pack["y0"], z0=pack["z0"],
        dx=pack["dx"], dy=pack["dy"], dz=pack["dz"],
    )
