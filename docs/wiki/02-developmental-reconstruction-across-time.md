# Developmental reconstruction across time

This tutorial follows **Membryo** and **Mcardiac_EA** across developmental time. Both workflows use planar spatial measurements and simulate intermediate tissue states. Their Stage-2 configurations follow the biological annotations available for each dataset.

```bash
conda activate stvirtual
source env/activate.sh
```

## Results

### Membryo — E9.5 to E15.5

The GIF concatenates E9.5→E11.5, E11.5→E13.5, and E13.5→E15.5.

![Membryo three-stage simulation](../assets/results/membryo.gif)

### Mcardiac_EA — E10.5 CD1 to E12.5 CD1

![Mcardiac_EA developmental simulation](../assets/results/Mcardiac_EA.gif)

## Membryo workflow

Use `experiments/Membryo` and the MOSTA data described in [Data Availability](../Data-Availability.md). The notebook example uses full embryo sections from E15.5 to E16.5. Set `route_ids` and `steps` in `config.yaml` to select the developmental route.

1. Review `config.yaml`, including input paths, the mouse LR table, and the configured latent key.
2. Run `preprocess.ipynb` to retain all cells and tissue annotations, preserve raw counts, and train scanVI. Align each timepoint by translating its all-cell centroid to the E16.5 reference centroid. The registered input is saved at `artifacts/checkpoints/scanvi/adata.h5ad` with `obsm["X_scanVI"]`.
3. Run `decoder.ipynb` to train the route-specific expression decoders. Checkpoints are written to `artifacts/checkpoints/decoder/checkpoints/<src>_<tgt>.pt`.
4. Run `train.ipynb`: `stvirtual.models.stage1_3d` learns each developmental route, the notebook builds planar boundaries from the Stage-1 trace, and `stvirtual.models.stage2_3d` models movement, birth, death, signaling, and latent-state dynamics.
5. Inspect the generated frames along each route before comparing developmental stages.

The Membryo configuration uses `stage2_3d`. Mcardiac_EA uses `stage2_3d_transition` with an explicit epicardial-to-fibroblast transition prior.

## Mcardiac_EA workflow

Use `experiments/Mcardiac_EA` to generate the E10.5 CD1 → E12.5 CD1 transition using `stage1_3d` and `stage2_3d_transition`, planar spatial coordinates, and a configured cell-state transition prior.

1. Set raw input and coordinate directories in `config.yaml`.
2. Run `preprocess.ipynb`: QC, isolated-bead removal, final tissue ROI selection, nearby epicardial-cell inclusion, scanVI, and UOT rotation/translation to the E12.5 reference. Unaligned coordinates and transform parameters are preserved.
3. Run `train.ipynb` for Stage 1, automatic CD1 decoder training, 2D boundaries, transition Stage 2, and transition output frames.
4. Continue with the [Mcardiac_EA perturbation workflow](06-perturbation.md#mcardiac_ea-epicardial-cell-ablation) for epicardial-cell ablation and free extrapolation. The training manifest supplies its input paths.

The transition table contains epicardial → fibroblasts. The Wt1-DTA sample provides the perturbation evaluation condition. Output frames preserve UID, ancestry, differentiation state, coordinates and expression embeddings. Generated files are written under the experiment's `artifacts/` directory.

Run preprocessing followed by training to prepare the extrapolation inputs. Use `decoder.ipynb` for standalone decoder training.

## Simulation output

Generated frames are saved under `artifacts/results/simulation/<src>_to_<tgt>/`. Coordinates are stored in `obsm["spatial"]`, expression representations in `obsm["X_latent"]`, and cell annotations in `obs["celltype"]` and `obs["celltype_id"]`. The adjacent `celltype_mapping.csv` maps IDs to names. Use `uid` to follow cells between frames and `parent_uid` to trace birth ancestry. Mcardiac_EA additionally records the configured cell-state transitions.

Compare the first and last frames with the measured tissues, inspect boundary coverage, and keep each simulation paired with its Stage-1, decoder, and Stage-2 checkpoints.
