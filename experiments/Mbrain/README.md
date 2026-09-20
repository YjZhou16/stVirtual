# Mbrain

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#mbrain).

Mbrain uses the shared time-free 2D Stage-1 model and its own 2D Stage-2 RL checkpoint. The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to use another `obsm` representation.

Start from `config.yaml`. Paths are repository-relative placeholders and no data or checkpoints are distributed here. The rollout call accepts `output_dir` and writes one AnnData file per frame plus `celltype_mapping.csv`. Each frame stores latent vectors in `obsm["X_latent"]`, coordinates in `obsm["spatial"]`, and annotations in `obs["celltype"]` and `obs["celltype_id"]`.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Local runtime layout

All paths in `config.yaml` are resolved relative to this dataset directory,
independent of the notebook server's working directory. Running the notebooks
creates `data/` for user-provided and processed AnnData, and `artifacts/` for
checkpoints and generated results. Both directories are excluded from Git.

