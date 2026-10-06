"""Expose measured protein expression through the model feature interface."""
from .contracts import _as_valid_latent


def prepare_protein(adata, *, layer='scaled', feature_key='X_protein'):
    """Copy scaled protein channels without fitting a latent representation."""
    if layer not in adata.layers:
        raise KeyError(f'Missing protein expression layer {layer!r}')
    adata.obsm[feature_key] = _as_valid_latent(
        adata.layers[layer], n_obs=adata.n_obs, key=feature_key
    ).copy()
    adata.uns['protein_features'] = {
        'layer': layer, 'feature_key': feature_key,
        'names': adata.var_names.astype(str).to_numpy(),
        'representation': 'measured_scaled_protein',
    }
    return adata
