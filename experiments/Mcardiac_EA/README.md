# Mcardiac_EA

This experiment generates E10.5 CD1 → E12.5 CD1 using `stage1_3d` and `stage2_3d_transition`, with 20 integration steps. Its transition prior permits epicardial → fibroblasts. E12.5 Wt1-DTA is included in preprocessing for perturbation evaluation.

Run in order:

1. `preprocess.ipynb`: read the three GSE282547 samples and barcode-matched coordinates; require ≥100 UMIs, ≥50 genes and singlet status; retain four tissue types with at least five neighbors within 100 coordinate units; apply the final source ROI polygons; add epicardial cells within 200 units of the retained tissue. Train 10D scanVI, then rotate/translate each sample to E12.5 CD1 with UOT registration. Write `artifacts/checkpoints/scanvi/adata.h5ad` and preserve the unaligned coordinates and alignment transforms.
2. `train.ipynb`: train Stage 1, train the expression decoder automatically, generate 2D boundaries, train transition Stage 2, and export frames plus `artifacts/training/E10.5_CD1_to_E12.5_CD1.json`.
3. `python perturb/run.py`: run the epicardial-cell perturbation and free extrapolation using that training manifest.

`train.ipynb` includes decoder training. Use `decoder.ipynb` for standalone decoder training.

All config paths are resolved relative to this directory. Set the raw-data and barcode-coordinate directories in `config.yaml`. Results and checkpoints are written to `artifacts/`.

See the [perturbation workflow](perturb/README.md) and [Wiki 05](../../docs/wiki/06-perturbation.md#mcardiac_ea-epicardial-cell-ablation) for the intervention and execution order.
