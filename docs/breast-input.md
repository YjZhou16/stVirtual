# Prepare the Human breast cancer input

Use the downloaded breast cancer H5AD data for this example.

1. Use [GASTON](https://github.com/raphael-group/GASTON) to annotate spatial regions and save the labels in `obs["gaston_region"]`.
2. Place the annotated `imc_all.h5ad` in `experiments/human_breast_cancer/data/`. Keep section IDs in `obs["sample"]`, XY coordinates in `obsm["spatial"]`, and the measured protein features.
3. Run `preprocess.ipynb`, then `train.ipynb`.

This workflow uses measured proteins directly, without scanVI, an expression decoder, or an LR table.
