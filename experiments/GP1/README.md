# GP1

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#gp1).

GP1 uses the shared time-free 2D Stage-1 model and its own lineage-aware 2D Stage-2 RL checkpoint. The audited historical source is recorded in notebook metadata and is not a runtime path.

The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to use another `obsm` representation. Start from `config.yaml`. The rollout `output_dir` contains one AnnData file per frame and `celltype_mapping.csv`, with `X_latent`, `spatial`, `celltype`, and `celltype_id` following the public contract.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Local runtime layout

All paths in `config.yaml` are resolved relative to this dataset directory,
independent of the notebook server's working directory. Running the notebooks
creates `data/` for user-provided and processed AnnData, and `artifacts/` for
checkpoints and generated results. Both directories are excluded from Git.

