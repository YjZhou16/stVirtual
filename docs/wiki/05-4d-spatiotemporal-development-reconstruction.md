# 4D spatiotemporal development reconstruction

This tutorial covers 4D spatiotemporal development reconstruction with **Mcardiac** as the worked example. Other volumetric datasets can use the same models with XYZ coordinates, cell annotations, expression features, and a dataset-specific differentiation map.

```bash
conda activate stvirtual
source env/activate.sh
```

## Result

### Mcardiac — E9.5h to E11.5h

![Mcardiac E9.5h-to-E11.5h transition simulation](../assets/results/mcardiac.gif)

## 1. Input data

Use `experiments/Mcardiac`. The model requires stage labels, cell annotations, raw counts, X/Y/Z coordinates, the mouse LR table, and `data/diff_map.csv`. `config.yaml` fixes `spatial_dimension: 3`, `stage1_4d`, and `stage2_4d_transition`.

## 2. Preprocessing

Run `preprocess.ipynb` to create consistent 3D coordinates, categorical cell types, and `obsm["X_scanVI"]`. The registered model-ready AnnData is saved to `artifacts/checkpoints/scanvi/adata.h5ad`.

## 3. Expression decoder

Run `decoder.ipynb`. It creates route-specific latent-to-expression decoder checkpoints under `artifacts/checkpoints/decoder/checkpoints/`.

## 4. Train the model

### 4.1 Stage 1

`stvirtual.models.stage1_4d` learns the shared volumetric transport route and writes checkpoints under `artifacts/checkpoints/stage1/`.

### 4.2 Volumetric boundaries

The notebook builds per-frame 3D boundaries under `artifacts/results/bound/`. Rebuild them after any coordinate or Stage-1 change.

### 4.3 Transition Stage 2

`stvirtual.models.stage2_4d_transition` is the 4D spatiotemporal development model. It loads the differentiation map, trains the transition policy, and saves checkpoints under `artifacts/checkpoints/stage2/`.

## 5. Transition output and quality control

Frames record `uid`, `parent_uid`, `diff_alpha`, source and target layers, committed cell type, latent state, and XYZ coordinates. Confirm `[diff] loaded` appears, rotate all axes together, inspect boundary coverage, and validate UID continuity separately from parent-child ancestry.

## Output contract

Each simulation route is written to `artifacts/results/simulation/<src>_to_<tgt>/`. Every frame is an AnnData file with coordinates in `obsm["spatial"]`, latent states in `obsm["X_latent"]`, and cell-type annotations in `obs["celltype"]` and `obs["celltype_id"]`. The adjacent `celltype_mapping.csv` maps IDs to names.

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

## Recommended order

1. Review and edit `config.yaml`.
2. Run `preprocess.ipynb`.
3. Run `decoder.ipynb`.
4. Run `train.ipynb` from top to bottom.
5. Inspect boundary coverage and the first and last simulation frames before downstream analysis.

Input data are placed under `data/`, and generated checkpoints and results are written to `artifacts/`.
