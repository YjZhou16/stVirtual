"""Read upstream Open-ST files under their HMLN sample names."""
from pathlib import Path


def validate_mapping(mapping):
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise ValueError('Sample mapping must be nonempty and one-to-one')
    return dict(mapping)


def load_hmln(raw_dir, mapping):
    """Map original filenames once, preserving original IDs for provenance."""
    import anndata as ad
    mapping = validate_mapping(mapping)
    paths = {old: Path(raw_dir) / f'Reconstructed_{old}.h5ad' for old in mapping}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing upstream HMLN slices: ' + ', '.join(missing))
    pieces = {}
    for old, path in paths.items():
        data = ad.read_h5ad(path)
        data.var_names_make_unique()
        data.obs['sample_original'] = old
        pieces[mapping[old]] = data
    result = ad.concat(pieces, label='sample', index_unique='-')
    result.obs_names_make_unique()
    result.uns['sample_mapping'] = mapping
    return result
