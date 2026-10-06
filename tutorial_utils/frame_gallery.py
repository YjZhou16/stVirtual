"""Show real endpoints and saved simulation steps with one color map."""
from pathlib import Path
import re
from functools import lru_cache
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def frame_number(path):
    match = re.search(r'_f(\d+)', Path(path).name)
    if not match:
        raise ValueError(f'Frame number missing from {path}')
    return int(match.group(1))


@lru_cache(maxsize=64)
def load_norm(checkpoint, dimension):
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    norm = state.get('global_norm', state.get('norm_stats'))
    key = 'xyz' if dimension == 3 else 'xy'
    def array(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return array(norm[f'{key}_mu']).reshape(1, dimension), float(array(norm[f'{key}_s']).ravel()[0])


def real_panel(path, sample_key, sample, label_key, dimension):
    data = ad.read_h5ad(path, backed='r')
    try:
        mask = data.obs[sample_key].astype(str).to_numpy() == str(sample)
        if not mask.any():
            raise ValueError(f'{sample!r} is missing from {path}')
        keys = ['cx_aligned', 'cy_aligned', 'cz_aligned'][:dimension]
        xyz = data.obs.loc[mask, keys].to_numpy(dtype=np.float32)
        labels = data.obs.loc[mask, label_key].astype(str).to_numpy()
        return {'coords': xyz, 'labels': labels, 'title': f'{sample} real'}
    finally:
        data.file.close()


def simulation_panel(path, checkpoint, dimension, step):
    data = ad.read_h5ad(path, backed='r')
    try:
        keys = ['cx', 'cy', 'cz'][:dimension]
        xyz = data.obs[keys].to_numpy(dtype=np.float32) if set(keys) <= set(data.obs) else np.asarray(data.obsm['spatial'])[:, :dimension]
        mu, scale = load_norm(checkpoint, dimension)
        xyz = xyz * scale + mu
        key = 'celltype' if 'celltype' in data.obs else 'layer_name'
        labels = data.obs[key].astype(str).to_numpy()
        return {'coords': xyz, 'labels': labels, 'title': f'Step {step}'}
    finally:
        data.file.close()


def load_gallery(input_path, sample_key, label_key, segments, dimension=2, interval=1):
    """segments contain src, tgt, frame_dir and the matching Stage-1 checkpoint."""
    panels = [real_panel(input_path, sample_key, segments[0]['src'], label_key, dimension)]
    offset = 0
    for segment in segments:
        paths = sorted(Path(segment['frame_dir']).glob('*.h5ad'), key=frame_number)
        if not paths:
            raise FileNotFoundError(f"No simulation frames in {segment['frame_dir']}")
        numbered = {frame_number(path): path for path in paths}
        last = max(numbered)
        if sorted(numbered) != list(range(last + 1)):
            raise ValueError('Simulation frames are incomplete or duplicated')
        # Step 0 is represented by the real source, not counted as a generated step.
        selected = [step for step in range(1, last + 1) if (offset + step) % interval == 0]
        if last not in selected:
            selected.append(last)
        panels.extend(simulation_panel(numbered[step], segment['checkpoint'], dimension, offset + step) for step in selected)
        offset += last
    panels.append(real_panel(input_path, sample_key, segments[-1]['tgt'], label_key, dimension))
    return panels


def plot_gallery(panels, palette, dimension=2, flip_y=False, columns=4, point_size=None):
    labels = sorted(set().union(*(set(panel['labels']) for panel in panels)))
    missing = set(labels) - set(palette)
    if missing:
        raise ValueError(f'Colors missing for {sorted(missing)}')
    all_coords = np.concatenate([panel['coords'] for panel in panels])
    if not np.isfinite(all_coords).all():
        raise ValueError('Coordinates contain NaN or Inf')
    lo, hi = all_coords.min(0), all_coords.max(0)
    pad = np.maximum(hi - lo, 1e-6) * 0.03
    rows = int(np.ceil(len(panels) / columns))
    legend_columns = min(2 if max(map(len, labels)) > 30 else 4, len(labels))
    legend_rows = int(np.ceil(len(labels) / legend_columns))
    legend_height = 0.24 * legend_rows + 0.35
    figure_height = 3.0 * rows + legend_height
    fig = plt.figure(figsize=(3.2 * columns, figure_height), facecolor='#111111')
    for index, panel in enumerate(panels):
        coords = np.asarray(panel['coords']).copy()
        if flip_y:
            coords[:, 1] *= -1
        ax = fig.add_subplot(rows, columns, index + 1, projection='3d' if dimension == 3 else None)
        ax.set_facecolor('#111111')
        colors = [palette[label] for label in panel['labels']]
        if dimension == 3:
            ax.scatter(*coords.T, c=colors, s=0.3 if point_size is None else point_size, depthshade=False, rasterized=True)
            ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
            ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
            ax.view_init(elev=18, azim=-55)
        else:
            ax.scatter(*coords.T, c=colors, s=0.5 if point_size is None else point_size, linewidths=0, rasterized=True)
            ax.set_aspect('equal')
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim((-hi[1] - pad[1], -lo[1] + pad[1]) if flip_y else (lo[1] - pad[1], hi[1] + pad[1]))
        ax.set_title(panel['title'], fontsize=11, color='white')
        ax.set_axis_off()
    handles = [Line2D([], [], color=palette[label], marker='o', linestyle='', markersize=4, label=label) for label in labels]
    bottom = legend_height / figure_height
    fig.legend(handles=handles, loc='lower center', ncol=legend_columns, frameon=False, fontsize=8, labelcolor='white')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=bottom, wspace=0.10, hspace=0.18)
    return fig
