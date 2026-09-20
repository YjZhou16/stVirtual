# VolumeFlow 3D Tutorial

This tutorial describes how to add a non-lineage volumetric dataset. No public benchmark dataset is assigned to this workflow yet.

```bash
conda activate stvirtual-3dslice
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

## 1. Create the experiment

```text
experiments/My3D/
├── config.yaml
├── preprocess.ipynb
├── decoder.ipynb
├── train.ipynb
├── data/
└── artifacts/
```

Configure:

```yaml
spatial_dimension: 3
stage1_module: stvirtual.models.stage1_3d
stage2_module: stvirtual.models.stage2_3d
latent_key: X_scanVI
```

## 2. Input data

The AnnData must contain categorical stage and cell-type columns, `layers["counts"]`, a latent representation, and consistent X/Y/Z coordinates. Use one coordinate system and one unit across stages. Time labels select source and target samples but are never neural-network inputs.

## 3. Preprocessing and decoder

Train scanVI with `save_anndata=True` and save the canonical input as `artifacts/checkpoints/scanvi/adata.h5ad`. Then train the latent-only expression decoder in `decoder.ipynb`.

## 4. Stage 1 and volumetric boundaries

Train `stage1_3d` and export its interpolated trace. Convert every frame into a volumetric occupancy/boundary file under `artifacts/results/bound/`. Check that source and target cells fall inside their corresponding volumes.

## 5. Stage 2 and rollout

Train `stage2_3d`, save policies under `artifacts/checkpoints/stage2/`, and export AnnData frames under `artifacts/results/rollout/`.

## 6. Validation

- Rotate X, Y, and Z together in visualization; do not rotate only a 2D projection.
- Verify coordinate orientation and units.
- Compare source/target counts with the first/last rollout frames.
- Inspect volumetric boundary coverage before interpreting dynamics.

## Output contract

Each rollout route is written to `artifacts/results/rollout/<src>_to_<tgt>/`. Every frame is an AnnData file with coordinates in `obsm["spatial"]`, latent states in `obsm["X_latent"]`, and public annotations in `obs["celltype"]` and `obs["celltype_id"]`. The adjacent `celltype_mapping.csv` maps IDs to names.

## Recommended order

1. Review and edit `config.yaml`.
2. Run `preprocess.ipynb`.
3. Run `decoder.ipynb`.
4. Run `train.ipynb` from top to bottom.
5. Inspect boundary coverage and the first and last rollout frames before downstream analysis.

Input data may be symlinked under `data/`. All generated files must remain under `artifacts/`; the workflow must never write into the linked source-data directory.
