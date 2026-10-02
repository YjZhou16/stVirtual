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

## Recommended order

1. Review and edit `config.yaml`.
2. Run `preprocess.ipynb`.
3. Run `decoder.ipynb`.
4. Run `train.ipynb` from top to bottom.
5. Inspect boundary coverage and the first and last simulation frames before downstream analysis.

Input data are placed under `data/`, and generated checkpoints and results are written to `artifacts/`.
