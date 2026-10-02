"""Infer a cell-type transition map from joint UOT and source-label permutations."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import ot


def sampled_sqdist_median(a, b, seed, n_pairs=200_000):
    rng = np.random.default_rng(seed)
    n = min(n_pairs, max(len(a), len(b)) * 20)
    ia = rng.integers(0, len(a), size=n)
    ib = rng.integers(0, len(b), size=n)
    return float(np.median(np.square(a[ia] - b[ib]).sum(axis=1)))


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return q


def infer_transition_map(adata, *, source='E9.5h', target='E11.5h',
                         sample_key='stage', label_key='mapped_celltype',
                         latent_key='X_scanVI', seed=2026, permutations=500,
                         spatial_weight=0.1, latent_weight=1.0, reg=0.1,
                         source_reg=1.0, target_reg=1.0, chunk_size=256,
                         fdr=0.05, min_fraction=0.05, min_excess=0.02,
                         min_enrichment=1.1, device='cpu'):
    """Return all candidate edges and discovered edges; no prior map is read."""
    if source == target or permutations < 1 or chunk_size < 1:
        raise ValueError('Use distinct stages and positive permutation/chunk counts')
    if min(reg, source_reg, target_reg) <= 0 or min(spatial_weight, latent_weight) < 0 or spatial_weight + latent_weight <= 0:
        raise ValueError('Invalid UOT regularization or cost weights')
    columns = ['cx_aligned', 'cy_aligned', 'cz_aligned']
    for key in [sample_key, label_key, *columns]:
        if key not in adata.obs:
            raise KeyError(key)
    selected = adata[adata.obs[sample_key].astype(str).isin([source, target])]
    if selected.obs[label_key].isna().any():
        raise ValueError('Cell-type labels contain missing values')
    stages = selected.obs[sample_key].astype(str).to_numpy()
    src, tgt = stages == source, stages == target
    if not src.any() or not tgt.any():
        raise ValueError('Both endpoint samples must contain cells')
    xyz = selected.obs[columns].to_numpy(dtype=np.float32)
    z = np.asarray(selected.obsm[latent_key], dtype=np.float32)
    if z.ndim != 2 or not z.shape[1] or not np.isfinite(xyz).all() or not np.isfinite(z).all():
        raise ValueError('Coordinates and latent states must be finite matrices')
    xyz = (xyz - xyz.mean(0)) / xyz.std(0).clip(1e-6)
    z = (z - z.mean(0)) / z.std(0).clip(1e-6)
    xs, xt, zs, zt = xyz[src], xyz[tgt], z[src], z[tgt]
    xscale = max(sampled_sqdist_median(xs, xt, seed), 1e-8)
    zscale = max(sampled_sqdist_median(zs, zt, seed + 1), 1e-8)
    cost = np.empty((len(xs), len(xt)), dtype=np.float32)
    tx, tz = torch.as_tensor(xt, device=device), torch.as_tensor(zt, device=device)
    for start in range(0, len(xs), chunk_size):
        end = min(start + chunk_size, len(xs))
        sx = torch.as_tensor(xs[start:end], device=device)
        sz = torch.as_tensor(zs[start:end], device=device)
        cost[start:end] = (spatial_weight * torch.cdist(sx, tx).square() / xscale
                           + latent_weight * torch.cdist(sz, tz).square() / zscale).cpu().numpy()
    plan = ot.unbalanced.sinkhorn_unbalanced(
        np.full(len(xs), 1.0 / len(xs), dtype=np.float32),
        np.full(len(xt), 1.0 / len(xt), dtype=np.float32), cost,
        reg=reg, reg_m=(source_reg, target_reg), method='sinkhorn_stabilized',
        reg_type='kl', numItermax=3000, stopThr=1e-7)
    plan = np.asarray(plan, dtype=np.float32)
    if not np.isfinite(plan).all() or plan.sum(dtype=np.float64) <= 0:
        raise FloatingPointError('Invalid or empty UOT plan')
    labels = selected.obs[label_key].astype(str).to_numpy()
    source_types, target_types = sorted(set(labels[src])), sorted(set(labels[tgt]))
    sc = pd.Categorical(labels[src], categories=source_types).codes
    tc = pd.Categorical(labels[tgt], categories=target_types).codes
    mass = plan @ np.eye(len(target_types), dtype=np.float32)[tc]
    del plan, cost
    def fractions(codes):
        rows = np.array([mass[codes == i].sum(axis=0, dtype=np.float64)
                         for i in range(len(source_types))])
        return rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-12)
    observed = fractions(sc)
    rng = np.random.default_rng(seed + 100)
    null = np.asarray([fractions(rng.permutation(sc)) for _ in range(permutations)], dtype=np.float32)
    rows = []
    for i, s in enumerate(source_types):
        for j, t in enumerate(target_types):
            obs = float(observed[i, j]); background = null[:, i, j].astype(float)
            mean = float(background.mean())
            rows.append(dict(source_type=s, target_type=t, observed_transport_fraction=obs,
                             perm_mean=mean, excess_over_null=obs - mean,
                             enrichment_over_null=obs / max(mean, 1e-12),
                             p_enrichment=float((1 + (background >= obs).sum()) / (permutations + 1))))
    stats = pd.DataFrame(rows)
    stats['q_value'] = bh_fdr(stats.p_enrichment)
    stats['discovered'] = ((stats.q_value < fdr) & (stats.observed_transport_fraction >= min_fraction)
                           & (stats.excess_over_null >= min_excess) & (stats.enrichment_over_null >= min_enrichment))
    edges = stats.loc[stats.discovered, ['source_type', 'target_type', 'observed_transport_fraction']].rename(
        columns={'source_type': 'src_layer', 'target_type': 'tgt_layer', 'observed_transport_fraction': 'weight'})
    return stats, edges.reset_index(drop=True)


def main():
    import anndata as ad
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/transition_uot'))
    parser.add_argument('--source', default='E9.5h')
    parser.add_argument('--target', default='E11.5h')
    parser.add_argument('--label-key', default='mapped_celltype')
    parser.add_argument('--permutations', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else config_path.parent / path
    data_path = resolve(config['scanvi_dir']) / 'adata.h5ad'
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    adata = ad.read_h5ad(data_path)
    settings = dict(source=args.source, target=args.target, label_key=args.label_key,
                    latent_key=config.get('latent_key', 'X_scanVI'), seed=args.seed,
                    permutations=args.permutations, device=args.device)
    stats, edges = infer_transition_map(adata, **settings)
    output.mkdir(parents=True, exist_ok=False)
    stats.to_csv(output / 'edge_statistics.csv', index=False)
    (output / 'settings.json').write_text(json.dumps(dict(input=str(data_path), **settings), indent=2) + '\n')
    if edges.empty or not (edges.src_layer != edges.tgt_layer).any():
        raise ValueError(f'No non-self transition was discovered; inspect {output / "edge_statistics.csv"}')
    edges.to_csv(output / 'discovered_transition.csv', index=False)
    print(output / 'discovered_transition.csv')


if __name__ == '__main__':
    main()
