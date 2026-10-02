# Mcardiac

Download instructions and the official upstream source are listed in [Data Availability](../../docs/Data-Availability.md#mcardiac).

Mcardiac models mouse cardiac development from E9.5 to E11.5. It uses `stage1_4d` and `stage2_4d_transition`.

The recommended input embedding is `adata.obsm["X_scanVI"]`; set `latent_key` explicitly to select another `obsm` representation. Start from `config.yaml`. The simulation `output_dir` contains one AnnData file per frame and `celltype_mapping.csv`, with `X_latent`, `spatial`, `celltype`, and `celltype_id` following the public contract.

Run `preprocess.ipynb` first, then `train.ipynb`.

## Paths

All config paths are resolved relative to this directory.

## No-label variant

For reconstruction without cell-type labels, use the 4D `nolabel` models with the same XYZ coordinates and `X_scanVI` representation:

```python
from stvirtual.models.nolabel import stage1_4d as s1
from stvirtual.models.nolabel import stage2_4d as s2
```

Adapt the training cells in `train.ipynb` as follows:

- Set `stage1_module` to `stvirtual.models.nolabel.stage1_4d` and `stage2_module` to `stvirtual.models.nolabel.stage2_4d` in the experiment configuration. Keep `spatial_dimension: 3` and the `stage` sample key.
- Call `s1.train_model_multislice` without `cell_type_key` or `lam_context`, then export the Stage-1 trace and build the voxel boundaries.
- Call `s2.build_global_ctx` without `layer_col`. Create `s2.StageCfg` without `layer_col` or `diff_csv`, and call `s2.prepare_one_stage` with `sample_key="stage"`. Omit transition-specific checkpoint, cell-type, and differentiation-map assertions.
- Train with `s2.run_multi_stages`, then call `s2.simulation_policy_one_stage` with `sample_key="stage"` and `output_dir`, without `TAU_DIFF`. Use separate checkpoint and output paths for this variant. Its saved frames contain coordinates, latent states, and cell identities; omit the transition-specific metadata export cells.

## Perturbation

Use [`perturb/config.yaml`](perturb/config.yaml) after completing this experiment. The [perturbation guide](perturb/README.md) explains the intervention, checkpoint inputs and output layout.
