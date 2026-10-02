# Human lung cancer

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#human-lung-cancer).

Human lung cancer uses the shared 3D Stage-1 model and its own transition-aware 3D Stage-2 RL checkpoint.

The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to use another `obsm` representation. Start from `config.yaml`. The simulation `output_dir` contains one AnnData file per frame and `celltype_mapping.csv`, with `X_latent`, `spatial`, `celltype`, and `celltype_id` following the public contract.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Paths

All config paths are resolved relative to this directory.

## Perturbation

Use [`perturb/config.yaml`](perturb/config.yaml) after completing this experiment. The [perturbation guide](perturb/README.md) explains the intervention, checkpoint inputs and output layout.
