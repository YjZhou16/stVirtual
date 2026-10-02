# Human breast cancer

This experiment models measured protein expression with the 3D Stage-1 and Stage-2 models. The input contains 25 protein channels. Preprocessing applies arcsinh transformation and channel-wise standardization, then stores the resulting protein features in `layers["scaled"]` and `obsm["X_protein"]`.

1. Use [GASTON](https://github.com/raphael-group/GASTON) to annotate spatial regions and store the labels in `obs["gaston_region"]`.
2. Place `imc_all.h5ad` under `data/` with `sample`, `gaston_region`, and 2D coordinates in `obsm["spatial"]`.
3. Run `preprocess.ipynb` to prepare the protein features and write `artifacts/preprocessing/imc.h5ad`.
4. Run `train.ipynb` to train the routes connecting slices 0→2→4→6→8→10→12→14.

Both stages use `X_protein`. Stage 2 is configured with `use_lr=False` and `lr_source="none"`. Generated frames contain the normalized protein state. The prepared input's `protein_features` metadata records channel names and order; the Stage-1 feature normalization converts simulation states back to scaled-protein units.

All config paths are resolved relative to this directory.
