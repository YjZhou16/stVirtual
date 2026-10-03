# Membryo

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#membryo).

Membryo reconstructs mouse embryonic development using full MOSTA embryo sections. Preprocessing retains all cells and tissue annotations and aligns each timepoint using the centroid of all its cells, with E16.5 as the reference. It follows the developmental reconstruction across time workflow and uses `stage1_3d` with a dataset-specific `stage2_3d` RL checkpoint. The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to select another `obsm` representation.

Start from `config.yaml`. The simulation call accepts `output_dir` and writes one AnnData file per frame plus `celltype_mapping.csv`. Each frame stores latent vectors in `obsm["X_latent"]`, coordinates in `obsm["spatial"]`, and annotations in `obs["celltype"]` and `obs["celltype_id"]`.

Run `preprocess.ipynb`, `decoder.ipynb`, and `train.ipynb` in that order. The example uses E15.5 to E16.5; edit `route_ids` and `steps` in `config.yaml` to select a different route.

## Paths

All config paths are resolved relative to this directory.
