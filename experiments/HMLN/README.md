# HMLN

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#hmln).

HMLN uses the shared 3D Stage-1 model and its own 3D Stage-2 RL checkpoint. The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to use another `obsm` representation.

Start from `config.yaml`. The simulation call accepts `output_dir` and writes one AnnData file per frame plus `celltype_mapping.csv`. Each frame stores latent vectors in `obsm["X_latent"]`, coordinates in `obsm["spatial"]`, and annotations in `obs["celltype"]` and `obs["celltype_id"]`.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Paths

All config paths are resolved relative to this directory.

## Upstream slice correspondence

| Upstream Open-ST ID | HMLN ID |
|---|---|
| S2 | S1 |
| S3 | S2 |
| S4 | S3 |
| S17 | S4 |
| S18 | S5 |
| S19 | S6 |
| S24 | S7 |
| S25 | S8 |
| S26 | S9 |

Keep upstream filenames unchanged under `data/raw/`, for example `Reconstructed_S17.h5ad`. Preprocessing reads all nine original files, creates the HMLN `sample` labels once at ingestion, and preserves `sample_original` and `uns["sample_mapping"]`. Use `sample_original` and `uns["sample_mapping"]` to recover upstream identifiers. Re-run preprocessing and retrain route checkpoints when updating this mapping.

Both decoder and transport training read the three routes from `config.yaml`: S1→S3, S4→S6, S7→S9. Each middle slice is retained as an observed intermediate sample; each of the three groups is trained as an independent route.
