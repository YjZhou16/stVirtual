"""Shared perturbation workflow for GP1, LUAD and Mcardiac."""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import importlib
from importlib import import_module
from pathlib import Path
import os
import sys
import zipfile

import anndata as ad
import h5py
import json
import numpy as np
import pandas as pd
from scipy import sparse
import torch
import yaml

from stvirtual.data.contracts import read_latent

LATENT_KEY = "X_scanvi_pert"
REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
TASKS = {"GP1": ("cell_removal", 2), "LUAD": ("gene_knockdown", 2),
         "Mcardiac": ("cell_state_transition_lr", 3)}



def _gene_indices(adata, genes):
    genes = list(dict.fromkeys(genes))
    if not genes:
        raise ValueError('Select at least one gene')
    indices = adata.var_names.get_indexer(genes)
    if (indices < 0).any():
        raise KeyError(f'Missing genes: {[g for g, i in zip(genes, indices) if i < 0]}')
    return indices


def _scaled(matrix, rows, columns, factor):
    if not np.isfinite(factor) or factor < 0:
        raise ValueError('Expression factor must be finite and non-negative')
    if sparse.issparse(matrix):
        result = matrix.astype(np.float32).tocsc(copy=True)
        selected = np.zeros(result.shape[0], dtype=bool)
        selected[rows] = True
        for column in columns:
            start, stop = result.indptr[column:column + 2]
            positions = np.arange(start, stop)[selected[result.indices[start:stop]]]
            result.data[positions] *= factor
        return result.tocsr()
    result = np.array(matrix, dtype=np.float32, copy=True)
    result[np.ix_(rows, columns)] *= factor
    return result


def remove_high_cells(adata, *, gene, sample_key, source, fraction, layer='counts'):
    """Remove exact top-k source cells, using input order to break expression ties."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    column = _gene_indices(adata, [gene])[0]
    values = adata.layers[layer][:, column]
    values = values.toarray() if sparse.issparse(values) else values
    values = np.asarray(values).ravel()
    if not np.isfinite(values).all():
        raise ValueError('Marker expression contains nonfinite values')
    source_rows = np.flatnonzero(adata.obs[sample_key].astype(str).eq(source))
    if not len(source_rows):
        raise ValueError(f'No source cells for {source!r}')
    count = int(np.floor(len(source_rows) * fraction + 0.5))
    selected = source_rows[np.argsort(-values[source_rows], kind='stable')[:count]]
    keep = np.ones(adata.n_obs, dtype=bool)
    keep[selected] = False
    return adata[keep].copy(), {
        'removed': count, 'source_cells': len(source_rows),
        'removed_obs_names': adata.obs_names[selected].astype(str).tolist(),
    }


def knockdown_genes(adata, *, genes, celltype_key, celltypes, factor, layer='counts', sample_key=None, samples=None):
    columns = _gene_indices(adata, genes)
    rows = np.flatnonzero(adata.obs[celltype_key].astype(str).isin(list(map(str, celltypes))))
    if samples is not None:
        if not samples or sample_key is None or sample_key not in adata.obs:
            raise ValueError('A sample key and nonempty sample selection are required')
        sample_mask = adata.obs[sample_key].astype(str).isin(list(map(str, samples))).to_numpy()
        rows = rows[sample_mask[rows]]
    if not len(rows):
        raise ValueError('No cells match the selected cell types and samples')
    result = adata.copy()
    result.layers[layer] = _scaled(adata.layers[layer], rows, columns, factor)
    report = {'affected_cells': len(rows), 'genes': list(genes), 'factor': factor,
              'celltypes': list(map(str, celltypes)), 'affected_obs_names': adata.obs_names[rows].astype(str).tolist()}
    if sample_key is not None:
        report['affected_cells_by_sample'] = {str(k): int(v) for k, v in
            adata.obs.iloc[rows][sample_key].astype(str).value_counts().items()}
        report['intervention_samples'] = list(map(str, samples)) if samples is not None else list(report['affected_cells_by_sample'])
    return result, report


def scale_cell_state_transition_genes(adata, *, cell_state_transitions, ventricular_genes, atrial_genes,
                     down=0.1, up=10.0, uid_key='uid', layer='counts'):
    """Apply the source notebook's reciprocal LR-gene scaling in each cell-state transition group.

    Shared genes receive both multipliers; both/neither cell-state transition groups are unchanged.
    """
    vent = _gene_indices(adata, ventricular_genes)
    atrial = _gene_indices(adata, atrial_genes)
    result = adata.copy()
    groups = result.obs[uid_key].astype(str).map(cell_state_transitions).fillna('other')
    result.obs['cell_state_transition_class'] = groups.to_numpy()
    matrix = result.layers[layer]
    for cell_state_transition, vent_factor, atrial_factor in [
        ('future_ventricular', down, up), ('future_atrial', up, down),
    ]:
        rows = np.flatnonzero(groups.eq(cell_state_transition))
        matrix = _scaled(matrix, rows, vent, vent_factor)
        matrix = _scaled(matrix, rows, atrial, atrial_factor)
    result.layers[layer] = matrix
    return result, {'cell_state_transition_counts': groups.value_counts().to_dict(), 'down': down, 'up': up}


def remove_quantile_cells(adata, *, gene, sample_key, source, quantile=0.7, layer='counts'):
    """Original GP1 notebook: remove source marker counts >= source quantile."""
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError('quantile must be in [0, 1]')
    col = _gene_indices(adata, [gene])[0]
    values = adata.layers[layer][:, col]
    values = values.toarray() if sparse.issparse(values) else values
    values = np.asarray(values).ravel()
    source_mask = adata.obs[sample_key].astype(str).eq(source).to_numpy()
    if not source_mask.any() or not np.isfinite(values[source_mask]).all():
        raise ValueError('Source is empty or has nonfinite marker counts')
    threshold = float(np.quantile(values[source_mask], quantile))
    removed = source_mask & (values >= threshold)
    return adata[~removed].copy(), {
        'selection': 'quantile', 'quantile': quantile, 'threshold': threshold,
        'removed': int(removed.sum()), 'source_cells': int(source_mask.sum()),
        'removed_obs_names': adata.obs_names[removed].astype(str).tolist(),
    }


@torch.no_grad()
def infer_transport(source, target, *, checkpoint, dimension, latent_key, device,
                    celltype_key, lam_context=0.0, guide_topk=None,
                    residual_scales=(0.2, 0.05, 0.05)):
    if dimension not in (2, 3):
        raise ValueError('dimension must be 2 or 3')
    module = import_module(f'stvirtual.models.stage1_{dimension + 1}d')
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    norm, ode, guide = ckpt['global_norm'], ckpt['ode_hparams'], ckpt['guide_hparams']
    uot = ckpt['uot_hparams']
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    coord_names = ['cx_aligned', 'cy_aligned', 'cz_aligned'][:dimension]
    prefix = 'xyz' if dimension == 3 else 'xy'
    if prefix + '_mu' not in norm or prefix + '_s' not in norm:
        raise ValueError(f'Checkpoint is missing {dimension}D coordinate normalization')
    mu, scale = tensor(norm[prefix + '_mu']), tensor(norm[prefix + '_s']).clamp_min(1e-12)
    f_mu, f_std = tensor(norm['f_mu']), tensor(norm['f_std']).clamp_min(1e-12)
    if mu.numel() != dimension or scale.numel() != dimension:
        raise ValueError('Coordinate normalization dimension does not match the experiment')
    if f_mu.numel() != int(ode['latent_dim']) or f_std.numel() != int(ode['latent_dim']):
        raise ValueError('Latent normalization dimension does not match the model')
    def arrays(data):
        coords = tensor(data.obs[coord_names].to_numpy(dtype=np.float32))
        latent = tensor(read_latent(data, latent_key=latent_key))
        if latent.shape[1] != int(ode['latent_dim']):
            raise ValueError('Perturbation latent dimension does not match the transport model')
        library = tensor(np.asarray(data.layers['counts'].sum(axis=1)).ravel()).clamp_min(1e-12)
        if not torch.isfinite(coords).all():
            raise ValueError('Coordinates contain nonfinite values')
        return (coords - mu) / scale, (latent - f_mu) / f_std, library
    x0, z0, m0 = arrays(source)
    xt, zt, mt = arrays(target)
    context = {}
    if lam_context > 0:
        source_context, target_context = source.copy(), target.copy()
        source_context.obsm['spatial'] = source_context.obs[coord_names].to_numpy(dtype=np.float32)
        target_context.obsm['spatial'] = target_context.obs[coord_names].to_numpy(dtype=np.float32)
        src_ctx, tgt_ctx = module.get_neighbor_features(source_context, target_context, cell_type_key=celltype_key)
        context = dict(ctx0=torch.nn.functional.normalize(tensor(src_ctx), dim=1),
                       ctxT=torch.nn.functional.normalize(tensor(tgt_ctx), dim=1))
    make_guide = module.build_guide if dimension == 2 else module.build_guide_3d
    guide_fn = make_guide(
        x0, z0, m0, xt, zt, mt, eps=guide.get('eps', 0.01), tau=uot['tau'],
        lam_x=uot['lam_x'], lam_f=uot['lam_f'],
        topk=guide_topk or guide['topk'], temp=guide['temp'], schedule=guide['schedule'],
        var_correction=False, terrain_gap=False, lam_context=lam_context, **context,
    )
    kwargs = {'spatial_dim': 3} if dimension == 3 else {}
    net = module.ResidualDynamicsNet(
        latent_dim=ode['latent_dim'], hidden=ode['hidden'],
        residual_scale_x=residual_scales[0], residual_scale_f=residual_scales[1],
        residual_scale_s=residual_scales[2], **kwargs,
    ).to(device)
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    cache = module.GuideCache(guide_fn, n=ode['n_cache'], device=device, dtype=torch.float32)
    func = module.FusedODEFunc(cache, net).to(device).eval()
    times, coords, features, log_mass = module.simulation_neuralode_dopri5(
        func, x0, z0, m0, steps=ode['steps'], rtol=ode['rtol'], atol=ode['atol'],
    )
    to_numpy = lambda values: [value.detach().cpu().numpy().astype(np.float32) for value in values]
    return {'t': times.detach().cpu().numpy(), 'coords_frames': to_numpy(coords),
            'Z_frames': to_numpy(features), 'mass_frames': to_numpy([torch.exp(s) for s in log_mass])}


def save_trace(trace, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in trace.items()})


def make_boundaries(frames, directory, *, dimension, settings, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if dimension == 2:
        from stvirtual.utils import boundary_2d as boundary
        keys = ['alpha_factor', 'target_frac', 'expand0', 'expand_step', 'expand_max', 'fallback_expand', 'n_resample']
        for i, coords in enumerate(frames):
            shell, *_ = boundary.make_shell_adaptive(coords, seed=seed + i, **{k:settings[k] for k in keys})
            boundary.save_shell_csv(shell, directory / f'bound_z{i:03d}.csv')
    else:
        from stvirtual.utils import boundary_3d as boundary
        from stvirtual.models import stage2_4d_transition as model
        bbox = boundary.compute_bbox_3d_from_frames(frames, margin=settings['margin'])
        d, h, w, _ = model.auto_choose_grid_size_3d(
            np.concatenate(frames), margin_xyz=settings['margin'],
            **{k:settings[k] for k in ['pts_per_cell','base','max_hw','min_hw','max_d','min_d']},
        )
        for i in range(len(frames)):
            coords = np.concatenate(frames[max(0,i-1):min(len(frames),i+2)])
            volume = boundary.voxelize_points_3d_fixed_bbox_dense(
                coords, bbox=bbox, D=d, H=h, W=w,
                **{k:settings[k] for k in ['splat_radius','dilate_iter','close_iter','fill_holes','keep_lcc','min_count']},
            )
            boundary.save_voxel_volume(volume, directory / f'bound_z{i:03d}.npz')


def baseline_cell_state_transitions(source, pattern, checkpoint, *, celltype_key, start_label,
                   atrial_label='Atrial cardiomyocytes', ventricular_label='Ventricular cardiomyocytes'):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No baseline simulation frames match {pattern}')
    from pathlib import Path
    if len({str(Path(p).parent) for p in paths}) != 1:
        raise ValueError('Baseline pattern matches multiple runs; select one simulation directory')
    first = ad.read_h5ad(paths[0])
    # Public simulation rows correspond to source rows, but never assume that silently.
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    norm = checkpoint['global_norm']
    prefix = 'xyz' if 'xyz_mu' in norm else 'xy'
    mu = np.asarray(norm[prefix + '_mu'])
    scale = np.asarray(norm[prefix + '_s'])
    coordinates = source.obs[['cx_aligned','cy_aligned','cz_aligned']].to_numpy()
    normalized = (coordinates - mu) / (scale + 1e-12)
    labels = source.obs[celltype_key].astype(str).to_numpy()
    if (len(first) != len(source) or first.obsm['spatial'].shape != normalized.shape
            or not np.allclose(first.obsm['spatial'], normalized, atol=1e-4)
            or not np.array_equal(first.obs['celltype'].astype(str).to_numpy(), labels)):
        raise ValueError('Baseline frame 0 does not match source row order, labels and normalized coordinates')
    uids = first.obs['uid'].astype(str).to_numpy()
    if len(set(uids)) != len(uids):
        raise ValueError('Baseline source UIDs must be unique')
    source = source.copy()
    source.obs['uid'] = uids
    selected = set(uids[labels == start_label])
    if not selected:
        raise ValueError(f'No source cells match {start_label!r}')
    seen_atrial, seen_ventricular = set(), set()
    for path in paths[1:]:
        frame = ad.read_h5ad(path)
        ids = frame.obs['uid'].astype(str)
        types = frame.obs['celltype'].astype(str)
        seen_atrial.update(set(ids[types.eq(atrial_label)]) & selected)
        seen_ventricular.update(set(ids[types.eq(ventricular_label)]) & selected)
    cell_state_transitions = {}
    for uid in selected:
        a, v = uid in seen_atrial, uid in seen_ventricular
        cell_state_transitions[uid] = 'both' if a and v else 'future_atrial' if a else 'future_ventricular' if v else 'neither'
    return source, cell_state_transitions


def calc_alpha_gt_threshold_by_t(
    adata,
    t_col="t",
    alpha_col="diff_alpha",
    is_diff_col="is_diff",
    only_diff=True,
    t_min=0,
    t_max=100,
    threshold=0.6,
):
    obs = adata.obs.copy()

    # ---------- Check columns. ----------
    need_cols = [t_col, alpha_col]
    if only_diff:
        need_cols.append(is_diff_col)
    miss = [c for c in need_cols if c not in obs.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # ---------- Convert to numeric values. ----------
    obs[t_col] = pd.to_numeric(obs[t_col], errors="coerce")
    obs[alpha_col] = pd.to_numeric(obs[alpha_col], errors="coerce")

    # ---------- Keep differentiated cells only. ----------
    if only_diff:
        v = obs[is_diff_col]
        keep = (
            (v == True) |
            (v == 1) |
            (v.astype(str).str.lower().isin(["true", "1"]))
        )
        obs = obs.loc[keep].copy()

    # ---------- Filter rows. ----------
    obs = obs.loc[
        obs[t_col].notna() &
        obs[alpha_col].notna() &
        (obs[t_col] >= t_min) &
        (obs[t_col] <= t_max)
    ].copy()

    # ---------- Round t to integers. ----------
    obs["t_step"] = pd.to_numeric(obs[t_col], errors="coerce").round().astype("Int64")
    obs = obs.loc[
        obs["t_step"].notna() &
        (obs["t_step"] >= t_min) &
        (obs["t_step"] <= t_max)
    ].copy()

    # ---------- Check whether values exceed the threshold. ----------
    obs["alpha_gt_thr"] = obs[alpha_col] > threshold

    order = list(range(t_min, t_max + 1))

    # ---------- Summary table. ----------
    res_df = (
        obs.groupby("t_step", observed=True)
        .agg(
            n_cells=(alpha_col, "size"),
            n_alpha_gt_thr=("alpha_gt_thr", "sum"),
        )
        .reindex(order, fill_value=0)
        .reset_index()
    )

    res_df["prop_alpha_gt_thr"] = np.where(
        res_df["n_cells"] > 0,
        res_df["n_alpha_gt_thr"] / res_df["n_cells"],
        np.nan
    )

    # Use descriptive column names.
    res_df = res_df.rename(columns={
        "n_alpha_gt_thr": f"n_alpha_gt_{threshold}",
        "prop_alpha_gt_thr": f"prop_alpha_gt_{threshold}",
    })

    return res_df


def find_first_positive_and_plateau_by_piecewise(
    prop_df,
    t_col="t_step",
    prop_col="prop_alpha_gt_0.6",
    positive_eps=1e-12,
    smooth=5,
    min_plateau_len=15,   # Minimum number of points in the plateau segment.
):
    df = prop_df[[t_col, prop_col]].copy().sort_values(t_col).reset_index(drop=True)
    df[prop_col] = pd.to_numeric(df[prop_col], errors="coerce").astype(float)

    # 1) First value greater than zero.
    idx_pos = df.index[df[prop_col] > positive_eps].tolist()
    first_positive_t = None if len(idx_pos) == 0 else df.loc[idx_pos[0], t_col]
    first_positive_idx = None if len(idx_pos) == 0 else idx_pos[0]

    # 2) Smooth the series.
    y = df[prop_col].to_numpy(dtype=float)
    t = df[t_col].to_numpy(dtype=float)
    y_sm = pd.Series(y).rolling(smooth, center=True, min_periods=1).mean().to_numpy()

    n = len(df)
    if n < min_plateau_len + 8:
        raise ValueError("Too few time points for a piecewise fit.")

    # Search range for the split point.
    k_min = max(3, 0 if first_positive_idx is None else first_positive_idx + 2)
    k_max = n - min_plateau_len - 1

    if k_min > k_max:
        raise ValueError("No valid split range; reduce min_plateau_len.")

    best = None

    for k in range(k_min, k_max + 1):
        # Fit a line to the first segment.
        coef = np.polyfit(t[:k+1], y_sm[:k+1], deg=1)
        y1_hat = np.polyval(coef, t[:k+1])

        # Fit a constant plateau to the second segment.
        c = np.median(y_sm[k+1:])
        y2_hat = np.full(n - (k + 1), c, dtype=float)

        sse1 = np.sum((y_sm[:k+1] - y1_hat) ** 2)
        sse2 = np.sum((y_sm[k+1:] - y2_hat) ** 2)
        sse = sse1 + sse2

        if (best is None) or (sse < best["sse"]):
            best = {
                "k": k,
                "sse": sse,
                "coef": coef,
                "plateau_level": c,
                "y1_hat": y1_hat,
                "y2_hat": y2_hat,
            }

    plateau_t = df.loc[best["k"], t_col]

    diag_df = df.copy()
    diag_df["y_sm"] = y_sm
    diag_df["fit_left"] = np.nan
    diag_df.loc[:best["k"], "fit_left"] = best["y1_hat"]
    diag_df["fit_right"] = np.nan
    diag_df.loc[best["k"]+1:, "fit_right"] = best["y2_hat"]

    summary_df = pd.DataFrame({
        "first_positive_t": [first_positive_t],
        "plateau_t": [plateau_t],
        "plateau_level": [best["plateau_level"]],
        "sse": [best["sse"]],
        "slope_left": [best["coef"][0]],
        "intercept_left": [best["coef"][1]],
    })

    return first_positive_t, plateau_t, diag_df, summary_df


def prepare_runtime(experiment_dir):
    sys.dont_write_bytecode=True
    repo=Path(__file__).resolve().parents[2]
    experiment=Path(experiment_dir).resolve()
    if not experiment.is_relative_to(repo/'experiments'):
        raise ValueError('Runtime must belong to an experiment')
    root=experiment/'artifacts/perturb'
    root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'src'))
    cache=root/'artifacts/.cache'
    for key,sub in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('PYKEOPS_BUILD_FOLDER','keops'),('NUMBA_CACHE_DIR','numba'),('TORCH_HOME','torch')]:
        os.environ[key]=str(cache/sub)
    if importlib.util.find_spec('_stvirtual_core') is not None:
        core = importlib.import_module('_stvirtual_core')
        if getattr(core, '__version__', None) != '0.2.0':
            raise ImportError('Install the stvirtual-core 0.2.0 wheel from wheels/ before running this release')
        return
    if sys.version_info[:2]!=(3,12) or sys.platform!='linux':
        raise RuntimeError('Bundled stVirtual models require Linux and Python 3.12')
    wheels=repo/'wheels'
    wheel=wheels/'stvirtual_core-0.2.0-cp312-cp312-linux_x86_64.whl'
    expected=next(line.split()[0] for line in (wheels/'SHA256SUMS').read_text().splitlines() if wheel.name in line)
    actual=hashlib.sha256(wheel.read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError('Bundled model wheel checksum mismatch')
    runtime=root/'artifacts/.runtime'/actual
    if not (runtime/'READY').exists():
        runtime.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if not name.startswith('_stvirtual_core/') or name.endswith('/'):
                    continue
                target=(runtime/name).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise ValueError('Invalid wheel member path')
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(name))
        (runtime/'READY').write_text(actual+'\n')
    sys.path.insert(0,str(runtime))


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config['experiment_config'] = str((path.parent / config['experiment_config']).resolve())
    experiment_path = Path(config['experiment_config'])
    experiment = yaml.safe_load(experiment_path.read_text())
    source, target = config['source'], config['target']
    route = f'{source}_to_{target}'
    def resolve(value): return str((experiment_path.parent / value).resolve())
    paths = {
        'input': resolve(experiment['scanvi_dir'] + '/adata.h5ad'),
        'scanvi': resolve(experiment['scanvi_dir']),
        'stage1': resolve(experiment['stage1_checkpoint'] + f'/{route}/checkpoints/best.pt'),
        'stage2': resolve(experiment['stage2_checkpoint'] + f'/policy_{route}.pt'),
        'decoder': resolve(experiment['decoder_checkpoint'].format(src=source,tgt=target)),
        'lr_pairs': resolve(experiment['lr_pairs_path']), 'diff_map': resolve(experiment['diff_map_path']),
        'output': str((path.parent / config['output']).resolve()),
    }
    if 'baseline_frames' in config:
        config['baseline_frames'] = str((path.parent / config['baseline_frames']).resolve())
    if 'lr_gene_sets' in config:
        config['lr_gene_sets'] = str((path.parent / config['lr_gene_sets']).resolve())
    dataset = config.get('dataset')
    if dataset not in TASKS:
        raise ValueError(f'Unsupported perturbation dataset: {dataset!r}')
    mode, dimension = TASKS[dataset]
    if config['mode'] != mode or int(experiment['spatial_dimension']) != dimension:
        raise ValueError(f'{dataset} requires mode={mode!r} and spatial_dimension={dimension}')
    if experiment['stage1_module'] != f'stvirtual.models.stage1_{dimension + 1}d':
        raise ValueError('Stage-1 module does not match the spatial dimension')
    if experiment['stage2_module'] != f'stvirtual.models.stage2_{dimension + 1}d_transition':
        raise ValueError('Stage-2 module does not match the transition dimension')
    if mode == 'gene_knockdown':
        samples = config.setdefault('intervention_samples', [source])
        if not isinstance(samples, list) or not samples or not set(samples) <= {source, target}:
            raise ValueError('intervention_samples must select source and/or target samples')
    return config, experiment, paths


def preflight(config_path):
    config, experiment, paths = load_config(config_path)
    status = {k: {'path': v, 'exists': Path(v).exists()} for k, v in paths.items() if k != 'output'}
    baseline = sorted(glob.glob(config.get('baseline_frames', ''))) if 'baseline_frames' in config else []
    missing = [k for k, v in status.items() if not v['exists']]
    if config['mode'] == 'cell_state_transition_lr' and not baseline:
        missing.append('baseline_frames')
    if 'lr_gene_sets' in config and not Path(config['lr_gene_sets']).exists():
        missing.append('lr_gene_sets')
    try:
        importlib.import_module(experiment['stage1_module'])
        importlib.import_module(experiment['stage2_module'])
        models = 'ready'
    except ImportError as error:
        models = str(error)
    return {'config':config,'paths':paths,'inputs':status,'baseline_frame_count':len(baseline),
            'missing_inputs':missing,'model_import':models,'ready':not missing and models=='ready'}


def read_frames(paths):
    import anndata as ad
    paths = sorted(map(Path, paths))
    if not paths:
        raise FileNotFoundError('No simulation/decoded frames; complete the matching run first')
    frames = {}
    for i,p in enumerate(paths):
        a = ad.read_h5ad(p)
        if 'celltype' in a.obs:
            a.obs['layer_name'] = a.obs['celltype'].astype(str)
            a.obs['cluster'] = a.obs['celltype'].astype(str)
        a.obs['frame'] = i
        a.obs['t'] = i
        frames[p.stem] = a
    merged = ad.concat(frames, label='sample', index_unique=':')
    merged.obs_names_make_unique()
    return merged


def find_gene_name(adata, gene):
    matches = [g for g in adata.var_names if str(g).lower()==str(gene).lower()]
    if len(matches)!=1:
        raise KeyError(f'Gene {gene!r} has {len(matches)} matches')
    return matches[0]



def counts_signature(data):
    """Fingerprint counts, gene order and cell order for the inferred representation."""
    digest = hashlib.sha256()
    for values in (data.obs_names, data.var_names):
        for value in values.astype(str):
            encoded = value.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
    matrix = data.layers['counts']
    digest.update(str(matrix.shape).encode())
    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        digest.update(b'csr')
        for values in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(str(values.dtype).encode())
            for start in range(0, len(values), 1_000_000):
                digest.update(np.ascontiguousarray(values[start:start + 1_000_000]).tobytes())
    else:
        digest.update(str(matrix.dtype).encode())
        for start in range(0, len(matrix), 4096):
            digest.update(np.ascontiguousarray(matrix[start:start + 4096]).tobytes())
    return digest.hexdigest()


def validate_perturb_latent(data, *, expected_dim=None, check_signature=True):
    values = read_latent(data, latent_key=LATENT_KEY)
    if expected_dim is not None and values.shape[1] != expected_dim:
        raise ValueError('Perturbation latent dimension does not match the reference model')
    if check_signature:
        provenance = data.uns.get('perturbation_latent', {})
        if provenance.get('key') != LATENT_KEY or provenance.get('counts_signature') != counts_signature(data):
            raise ValueError('Recompute X_scanvi_pert for the current perturbed counts before inference')
    return values


def save_perturbed_input(data, path):
    """Save and verify the representation passed to transport and transition models."""
    values = validate_perturb_latent(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(path)
    with h5py.File(path, 'r') as handle:
        key = f'obsm/{LATENT_KEY}'
        if key not in handle or handle[key].shape != values.shape:
            raise ValueError('Saved H5AD is missing the expected perturbation representation')
        saved_obs = ad.io.read_elem(handle['obs'])
        saved_var = ad.io.read_elem(handle['var'])
        if not saved_obs.index.equals(data.obs_names) or not saved_var.index.equals(data.var_names):
            raise ValueError('Saved H5AD changed the cell or gene ordering')
        for start in range(0, data.n_obs, 4096):
            block = np.asarray(handle[key][start:start + 4096])
            if not np.isfinite(block).all() or not np.array_equal(block, values[start:start + 4096]):
                raise ValueError('Saved perturbation latent differs from the inferred values')
    return path


def apply_intervention(original, config, paths, *, genes=None):
    """Apply each task's intervention before reference-model representation inference."""
    sample_key, type_key = config['sample_key'], config['celltype_key']
    source, target = config['source'], config['target']
    original = original[original.obs[sample_key].astype(str).isin([source, target])].copy()
    mode = config['mode']
    if mode == 'cell_removal':
        kwargs = dict(gene=config['gene'], sample_key=sample_key, source=source)
        selection = config.get('selection', 'quantile')
        if selection == 'quantile':
            query, report = remove_quantile_cells(original, quantile=config.get('quantile', 0.7), **kwargs)
        elif selection == 'exact_topk':
            query, report = remove_high_cells(original, fraction=config['fraction'], **kwargs)
            report['selection'] = selection
        else:
            raise ValueError('selection must be quantile or exact_topk')
    elif mode == 'gene_knockdown':
        if genes is None:
            genes = config.get('genes')
            if not genes:
                count = int(config['random_gene_count'])
                if not 1 <= count <= original.n_vars:
                    raise ValueError('random_gene_count must be between 1 and the number of genes')
                genes = np.random.default_rng(int(config.get('seed', 2025))).choice(
                    original.var_names.to_numpy(), size=count, replace=False).tolist()
        samples = config.get('intervention_samples', [source])
        if not isinstance(samples, list) or not samples or not set(samples) <= {source, target}:
            raise ValueError('intervention_samples must select source and/or target samples')
        query, report = knockdown_genes(
            original, genes=genes, celltype_key=type_key, celltypes=config['celltypes'],
            factor=config['factor'], sample_key=sample_key, samples=samples)
        report['intervention_kind'] = 'initial_expression_suppression'
    elif mode == 'cell_state_transition_lr':
        source_data = original[original.obs[sample_key].astype(str).eq(source)].copy()
        source_data, cell_state_transitions = baseline_cell_state_transitions(
            source_data, config['baseline_frames'], paths['stage1'],
            celltype_key=type_key, start_label=config['start_label'])
        pairs = json.loads(Path(config['lr_gene_sets']).read_text())
        groups = {name: list(dict.fromkeys(gene for pair in values for gene in pair.split('|')))
                  for name, values in pairs.items()}
        missing = {name: [gene for gene in values if gene not in source_data.var_names]
                   for name, values in groups.items()}
        groups = {name: [gene for gene in values if gene in source_data.var_names]
                  for name, values in groups.items()}
        changed, report = scale_cell_state_transition_genes(
            source_data, cell_state_transitions=cell_state_transitions, ventricular_genes=groups['ventricular'],
            atrial_genes=groups['atrial'], down=config['down'], up=config['up'])
        report['missing_genes'] = missing
        report['intervention_samples'] = [source]
        target_data = original[original.obs[sample_key].astype(str).eq(target)].copy()
        target_data.obs['uid'] = 'target:' + target_data.obs_names.astype(str)
        target_data.obs['cell_state_transition_class'] = 'target'
        query = ad.concat([changed, target_data], merge='same', uns_merge='same')
    else:
        raise ValueError(f'Unsupported mode {mode!r}')
    if not query.obs[sample_key].astype(str).eq(source).any() or not query.obs[sample_key].astype(str).eq(target).any():
        raise ValueError('Perturbation leaves an empty source or target')
    query.X = query.layers['counts'].copy()
    query.obsm.pop(LATENT_KEY, None)
    query.uns.pop('perturbation_latent', None)
    report['mode'] = mode
    return query, report



class PerturbationSession:
    def __init__(self, dataset, config_path=None, device=None):
        dataset = 'Mcardiac' if dataset == 'Heart' else dataset
        config_path = Path(config_path) if config_path else EXPERIMENTS / {'GP1': 'human_gastric_cancer', 'LUAD': 'human_lung_cancer'}.get(dataset, dataset) / 'perturb/config.yaml'
        self._configure(*load_config(config_path), device=device)

    @classmethod
    def from_settings(cls, config, experiment, paths, *, device):
        session = cls.__new__(cls)
        session._configure(config, experiment, paths, device=device)
        return session

    def _configure(self, config, experiment, paths, *, device):
        self.config, self.experiment, self.paths = copy.deepcopy((config, experiment, paths))
        self.dataset = self.config['dataset']
        mode, dimension = TASKS[self.dataset]
        if self.config['mode'] != mode or int(self.experiment['spatial_dimension']) != dimension:
            raise ValueError('Perturbation mode or dimension does not match the dataset')
        for stage, expected in [('stage1', f'stvirtual.models.stage1_{dimension + 1}d'),
                                ('stage2', f'stvirtual.models.stage2_{dimension + 1}d_transition')]:
            if self.experiment[stage + '_module'] != expected:
                raise ValueError(f'{stage} module does not match the experiment dimension')
        self.experiment_dir = Path(self.config['experiment_config']).parent.resolve()
        self.output = Path(self.paths['output']).resolve()
        if not self.output.is_relative_to(self.experiment_dir):
            raise ValueError('Perturbation outputs must be under the dataset experiment')
        for key, value in self.paths.items():
            if key != 'output':
                protected = Path(value).resolve()
                if protected == self.output or self.output in protected.parents or protected in self.output.parents:
                    raise ValueError(f'Output overlaps an input: {key}')
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        seed = int(self.config.get('seed', 2025))
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._started = False
        self._model = None
        self._prepared_query = None
        self.perturbation_report = None

    def start(self):
        if not self._started:
            self.output.mkdir(parents=True, exist_ok=False)
            self._started = True
        return self.output

    def apply_intervention(self, original, *, genes=None):
        query, report = apply_intervention(original, self.config, self.paths, genes=genes)
        self.perturbation_report = report
        return query, report

    def reinfer(self, query, *, model=None):
        """Recompute the representation with the fitted reference SCANVI; no training."""
        import scvi
        model = self.scanvi() if model is None else model
        data = query.copy()
        if 'counts' not in data.layers:
            raise KeyError('Perturbation input is missing counts')
        if not data.obs_names.is_unique or not data.var_names.is_unique:
            raise ValueError('Cell and gene names must be unique')
        data.X = data.layers['counts'].copy()
        original_signature = counts_signature(data)
        registry = model.adata_manager.data_registry['X']
        attr = registry['attr_name'] if isinstance(registry, dict) else registry.attr_name
        key = registry.get('attr_key') if isinstance(registry, dict) else getattr(registry, 'attr_key', None)
        if attr == 'layers':
            data.layers[key] = data.layers['counts'].copy()
        elif attr != 'X':
            raise ValueError(f'Unsupported reference expression registry: {attr}')
        scvi.settings.seed = int(self.config.get('seed', 2025))
        scvi.model.SCANVI.prepare_query_anndata(data, model, inplace=True)
        if counts_signature(data) != original_signature:
            raise ValueError('Reference query preparation changed counts or cell/gene order')
        values = np.asarray(model.get_latent_representation(data), dtype=np.float32)
        if counts_signature(data) != original_signature:
            raise ValueError('Reference latent inference changed the perturbed counts')
        data.obsm[LATENT_KEY] = values
        data.uns['perturbation_latent'] = {
            'key': LATENT_KEY, 'method': 'reference_scanvi_inference',
            'reference': str(self.paths['scanvi']), 'counts_signature': counts_signature(data)}
        expected_dim = getattr(getattr(model, 'module', None), 'n_latent', None)
        validate_perturb_latent(data, expected_dim=expected_dim)
        self._prepared_query = data
        return data

    def save_input(self, query, path=None):
        path = Path(path) if path is not None else self.start() / 'model_input.h5ad'
        if not path.resolve().is_relative_to(self.start()):
            raise ValueError('Perturbation input snapshots must be under the current run')
        return save_perturbed_input(query, path)

    def run(self):
        missing = [key for key, value in self.paths.items()
                   if key != 'output' and not Path(value).exists()]
        if missing:
            raise FileNotFoundError('Missing experiment inputs: ' + ', '.join(missing))
        self.start()
        query, _ = self.apply_intervention(self.input())
        query.write_h5ad(self.output / 'perturbed_counts.h5ad')
        query = self.reinfer(query)
        source = query[query.obs[self.config['sample_key']].astype(str).eq(self.config['source'])].copy()
        target = query[query.obs[self.config['sample_key']].astype(str).eq(self.config['target'])].copy()
        transport = self.transport(source, target)
        self.transition(query, transport)
        return self.output

    def input(self):
        import anndata as ad
        a = ad.read_h5ad(self.paths['input'])
        a = a[a.obs[self.config['sample_key']].astype(str).isin([self.config['source'],self.config['target']])].copy()
        a.X = a.layers['counts'].copy()
        return a

    def scanvi(self):
        import scvi
        if self._model is None:
            scvi.settings.seed = int(self.config.get('seed', 2025))
            self._model = scvi.model.SCANVI.load(
                self.paths['scanvi'], accelerator='gpu' if str(self.device).startswith('cuda') else 'cpu')
        return self._model

    def baseline_paths(self):
        paths = sorted(glob.glob(self.config.get('baseline_frames','')))
        if not paths:
            raise FileNotFoundError('No baseline frames. Set baseline_frames in perturb/config.yaml to one complete Mcardiac run.')
        if len({str(Path(p).parent) for p in paths}) != 1:
            raise ValueError('Select one baseline simulation run in perturb/config.yaml')
        return paths

    def transport(self, source, target):
        if self._prepared_query is not None:
            validate_perturb_latent(self._prepared_query)
        validate_perturb_latent(source, check_signature=False)
        validate_perturb_latent(target, check_signature=False)
        trace = infer_transport(source,target,checkpoint=self.paths['stage1'],
            dimension=self.experiment['spatial_dimension'],latent_key=LATENT_KEY,
            device=self.device,celltype_key=self.config['celltype_key'],
            lam_context=self.config.get('lam_context',0),guide_topk=self.config.get('guide_topk'),
            residual_scales=self.config.get('residual_scales',[0.2,0.05,0.05]))
        ckpt = torch.load(self.paths['stage1'],map_location='cpu',weights_only=False)
        norm = ckpt['global_norm']
        prefix = 'xyz' if int(self.experiment['spatial_dimension']) == 3 else 'xy'
        tensor = lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float32)
        mu,scale = tensor(norm[prefix+'_mu']),tensor(norm[prefix+'_s'])
        fmu,fstd = tensor(norm['f_mu']),tensor(norm['f_std'])
        keys=['cx_aligned','cy_aligned','cz_aligned'][:self.experiment['spatial_dimension']]
        xs,qs,ss = [tensor(x) for x in trace['coords_frames']],[tensor(x) for x in trace['Z_frames']],[tensor(np.log(np.maximum(x,1e-12))) for x in trace['mass_frames']]
        return {'trace':trace,'ckpt':ckpt,'t_eval':tensor(trace['t']),
                'xs_norm':xs,'qs_norm':qs,'ss':ss,
                'coords0':tensor(source.obs[keys].to_numpy()),'coords_tgt':tensor(target.obs[keys].to_numpy()),
                'coords0_norm':xs[0],'coords_tgt_norm':(tensor(target.obs[keys].to_numpy())-mu)/scale,
                'x1':xs[-1]*scale+mu,'q1':qs[-1]*fstd+fmu,'m1':tensor(trace['mass_frames'][-1])}

    def transition(self, query, transport):
        out=self.start()
        query=query.copy()
        query.X=query.layers['counts'].copy()
        self.save_input(query, out/'model_input.h5ad')
        trace=transport['trace']
        save_trace(trace,out/'stage1_trace.npz')
        make_boundaries(trace['coords_frames'],out/'bound',dimension=self.experiment['spatial_dimension'],
                        settings=self.experiment['boundary'],seed=self.config.get('seed',2025))
        s2=importlib.import_module(self.experiment['stage2_module'])
        ctx=s2.build_global_ctx(adata_path=str(out/'model_input.h5ad'),lr_pairs_path=self.paths['lr_pairs'],
                               ckpt_3dslice=self.paths['stage1'],device=torch.device(self.device),
                               layer_col=self.config['celltype_key'])
        stage=s2.StageCfg(src=self.config['source'],tgt=self.config['target'],out_npz_path=str(out/'stage1_trace.npz'),
                          bound_dir=str(out/'bound'),decoder_checkpoint=self.paths['decoder'],latent_key=LATENT_KEY,
                          layer_col=self.config['celltype_key'],diff_csv=self.paths['diff_map'],use_lr=True,lr_source='decoder')
        pack=s2.prepare_one_stage(ctx,stage,sample_key=self.config['sample_key'],layer_col=self.config['celltype_key'])
        result=s2.simulation_policy_one_stage(s2,ctx,stage,self.paths['stage2'],sample_key=self.config['sample_key'],
                    seed=self.config.get('seed',2025),ADVECT_LATENT=True,
                    LATENT_NOISE_SCALE=self.config.get('latent_noise_scale',0.01),
                    output_dir=out/'simulation',output_prefix=f"{stage.src}_to_{stage.tgt}")
        import anndata as ad
        source_query=query[query.obs[self.config['sample_key']].astype(str).eq(self.config['source'])]
        if 'uid' in source_query.obs:
            initial_uids=np.asarray(result['uid'][0]).astype(str)
            baseline_uids=source_query.obs['uid'].astype(str).to_numpy()
            if len(initial_uids)!=len(baseline_uids) or not np.allclose(np.asarray(result['coords'][0]),trace['coords_frames'][0],atol=1e-5):
                raise ValueError('Initial simulation UID count does not match baseline source')
            self.baseline_to_simulation_uid=dict(zip(baseline_uids,initial_uids))
            if self.perturbation_report is not None:
                self.perturbation_report['baseline_uid_to_simulation_uid'] = self.baseline_to_simulation_uid
        states=[]
        for i,p in enumerate(result['output_paths']):
            frame=ad.read_h5ad(p)
            state={}
            for field,value in result.items():
                if isinstance(value,list) and len(value)==len(result['output_paths']) and field not in ['output_paths']:
                    state[field]=np.asarray(value[i])
            if 'latent' not in state:
                if 'Z' in state:state['latent']=state['Z']
                elif 'X_latent' in frame.obsm:state['latent']=np.asarray(frame.obsm['X_latent'])
                else:raise KeyError('Current simulation has no normalized latent')
            for field in ['uid','parent_uid','is_diff','diff_alpha','diff_tgt_layer','diff_enter_step','commit_step','born_step']:
                if field in state:frame.obs[field]=state[field]
            frame.obs['layer_name']=frame.obs['celltype'].astype(str)
            frame.write_h5ad(p,compression='gzip')
            states.append(state)
        (out/'report.json').write_text(json.dumps({
            'config': self.config, 'inputs': self.paths, 'frames': len(states),
            'latent_key': LATENT_KEY, 'representation_method': 'reference_scanvi_inference',
            'perturbation': self.perturbation_report}, indent=2) + '\n')
        self.simulation=result
        return ctx,pack,{'state_final':states[-1],'history':{'state':states},'public_simulation':result}

    def decode(self, baseline=False):
        from stvirtual.decoder.infer import decode_h5ad
        import anndata as ad
        paths=self.baseline_paths() if baseline else self.simulation['output_paths']
        out=self.start()/('baseline_decoded' if baseline else 'decoded')
        out.mkdir(exist_ok=False)
        for i,p in enumerate(paths):
            frame=ad.read_h5ad(p)
            latent_key='X_latent' if 'X_latent' in frame.obsm else 'X'
            q=out/f'counts_pred_sample{i:02d}.h5ad'
            decode_h5ad(input_path=Path(p),output_path=q,checkpoint_path=Path(self.paths['decoder']),
                        latent_key=latent_key,latent_is_normalized=True,normalization_checkpoint=Path(self.paths['stage1']),
                        output_key='layers/counts_hat',device_name=self.device)
            a=ad.read_h5ad(q)
            a.X=a.layers['counts_hat'].copy()
            from scipy import sparse
            x=a.layers['counts_hat']
            if sparse.issparse(x):
                x=x.copy();x.data=np.log1p(x.data)
            else:x=np.log1p(x)
            a.layers['log1p_hat']=x
            if 'celltype' in a.obs:
                a.obs['layer_name']=a.obs['celltype'].astype(str)
                a.obs['cluster']=a.obs['celltype'].astype(str)
            a.write_h5ad(q,compression='gzip')
        return str(out)



def run(config, experiment, paths, *, device):
    return PerturbationSession.from_settings(config, experiment, paths, device=device).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config, experiment, paths = load_config(args.config)
    prepare_runtime(Path(config['experiment_config']).parent)
    if args.dry_run:
        print(json.dumps(preflight(args.config), indent=2))
        return
    print(run(config, experiment, paths, device=args.device))


if __name__ == '__main__':
    main()
