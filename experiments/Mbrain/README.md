# Mbrain

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#mbrain).

Mbrain uses the shared 3D Stage-1 model and its own 3D Stage-2 RL checkpoint. The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to use another `obsm` representation.

Start from `config.yaml`. The simulation call accepts `output_dir` and writes one AnnData file per frame plus `celltype_mapping.csv`. Each frame stores latent vectors in `obsm["X_latent"]`, coordinates in `obsm["spatial"]`, and annotations in `obs["celltype"]` and `obs["celltype_id"]`.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Paths

All config paths are resolved relative to this directory.
