# Mcardiac

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#mcardiac).

Mcardiac models mouse cardiac development from E9.5 to E11.5. It uses `stage1_4d` and `stage2_4d_transition`.

The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to select another `obsm` representation. Start from `config.yaml`. The simulation `output_dir` contains one AnnData file per frame and `celltype_mapping.csv`, with `X_latent`, `spatial`, `celltype`, and `celltype_id` following the public contract.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Paths

All config paths are resolved relative to this directory.

## Infer transitions without a supplied diff map

This variant retains cell-type labels and infers transition relationships from joint UOT using XYZ coordinates and `X_scanVI`. Source-label permutations identify enriched transitions; the discovered edges become the Stage-2 transition map.

After preprocessing, run from the repository root:

```bash
python experiments/Mcardiac/infer_transition_map.py
```

The command writes `edge_statistics.csv` and `discovered_transition.csv` under `experiments/Mcardiac/artifacts/transition_uot/`. It uses all cell types present in the two endpoint samples. Labels group transported mass for the permutation analysis; the UOT cost uses coordinates and latent features.

In `experiments/Mcardiac/config.yaml`, set:

```yaml
diff_map_path: artifacts/transition_uot/discovered_transition.csv
```

Continue with `train.ipynb` using `stage1_4d` and `stage2_4d_transition`. The notebook passes this generated file to `StageCfg.diff_csv`. Set separate checkpoint and result paths before training this variant. The generated table retains discovered self-edges for inspection; Stage 2 treats them as maintenance and loads non-self transitions. No supplied transition map or reference-edge list is used to select the discovered edges.

## Perturbation

Use [`perturb/config.yaml`](perturb/config.yaml) after completing this experiment. The [perturbation guide](perturb/README.md) explains the intervention, checkpoint inputs and output layout.
