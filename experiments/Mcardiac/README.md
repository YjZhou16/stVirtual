# Mcardiac

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#mcardiac).

Mcardiac is the public name for the `heart_9_11` workflow. Its sole public baseline is the corrected-lineage implementation in `stvirtual.models.stage2_3d_lineage`; historical 3D lineage variants must not replace or override this behavior. It uses the shared time-free 3D Stage-1 model and a dataset-specific corrected-lineage Stage-2 RL checkpoint.

The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to select another `obsm` representation. Start from `config.yaml`. The rollout `output_dir` contains one AnnData file per frame and `celltype_mapping.csv`, with `X_latent`, `spatial`, `celltype`, and `celltype_id` following the public contract.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Local runtime layout

All paths in `config.yaml` are resolved relative to this dataset directory,
independent of the notebook server's working directory. Running the notebooks
creates `data/` for user-provided and processed AnnData, and `artifacts/` for
checkpoints and generated results. Both directories are excluded from Git.

