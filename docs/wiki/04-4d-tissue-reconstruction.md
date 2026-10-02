# 4D tissue reconstruction

This tutorial describes 4D tissue reconstruction from volumetric measurements along an ordered progression axis. It provides a workflow template for a user-supplied dataset.

```bash
conda activate stvirtual
source env/activate.sh
```

## 1. Create the experiment

```text
experiments/My4D/
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
stage1_module: stvirtual.models.stage1_4d
stage2_module: stvirtual.models.stage2_4d
latent_key: X_scanVI
```

## 2. Input data

Prepare AnnData with categorical stage and cell-type columns, `layers["counts"]`, a latent representation, and consistent X/Y/Z coordinates. Use one coordinate system and one unit across stages. Stage labels select the source and target samples.

## 3. Preprocessing and decoder

Train scanVI with `save_anndata=True` and save the canonical input as `artifacts/checkpoints/scanvi/adata.h5ad`. Then train the latent-to-expression decoder in `decoder.ipynb`.

## 4. Stage 1 and volumetric boundaries

Train `stage1_4d` and export its interpolated trace. Convert every frame into a volumetric occupancy/boundary file under `artifacts/results/bound/`. Check that source and target cells fall inside their corresponding volumes.

## 5. Stage 2 and simulation

Train `stage2_4d`, save policies under `artifacts/checkpoints/stage2/`, and export AnnData frames under `artifacts/results/simulation/`.

## 6. Inspect the results

- Rotate X, Y, and Z together when viewing the generated tissue.
- Verify coordinate orientation and units.
- Compare source/target counts with the first/last simulation frames.
- Inspect volumetric boundary coverage before interpreting dynamics.

## Output contract

Each simulation route is written to `artifacts/results/simulation/<src>_to_<tgt>/`. Every frame is an AnnData file with coordinates in `obsm["spatial"]`, latent states in `obsm["X_latent"]`, and cell-type annotations in `obs["celltype"]` and `obs["celltype_id"]`. The adjacent `celltype_mapping.csv` maps IDs to names.

## Recommended order

1. Review and edit `config.yaml`.
2. Run `preprocess.ipynb`.
3. Run `decoder.ipynb`.
4. Run `train.ipynb` from top to bottom.
5. Inspect boundary coverage and the first and last simulation frames before downstream analysis.

Input data are placed under `data/`, and generated checkpoints and results are written to `artifacts/`.
