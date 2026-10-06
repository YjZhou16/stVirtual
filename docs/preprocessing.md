# Data preprocessing guide

Mbrain is the example dataset below. We recommend combining all sections or timepoints into one AnnData object and recording their names in `obs["sample"]`.

## Combine sections and normalize expression

Combine T168–T171. `join="inner"` keeps the genes shared by all sections.

```python
from pathlib import Path
import anndata as ad
import scanpy as sc

DATA_ROOT = Path("experiments/Mbrain/data")
pieces = {}
for sample in ["T168", "T169", "T170", "T171"]:
    section = ad.read_h5ad(DATA_ROOT / f"Mouse1_{sample}.h5ad")
    section.var_names_make_unique()
    pieces[sample] = section
adata = ad.concat(pieces, label="sample", join="inner", index_unique="-")

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
```

Here `X` contains raw counts. If your counts are stored in a layer, copy that layer instead. Keep raw counts for scanVI and decoder training.

## Learn compact features

High-dimensional data need a compact representation as model input. For low-dimensional data, use the measurements directly without learning another representation.

For Mbrain, train scanVI on the combined data:

```python
import scvi

scvi.model.SCANVI.setup_anndata(
    adata, layer="counts", batch_key="sample",
    labels_key="annotation", unlabeled_category="Unknown",
)
model = scvi.model.SCANVI(adata)
model.train()
adata.obsm["X_scanVI"] = model.get_latent_representation()
```

## Spatial alignment

| Dataset | Spatial alignment | Alignment input | Method |
| --- | --- | --- | --- |
| Mbrain | Yes | Coordinates | `uot_alignment.py` |
| HMLN | No (already aligned) | Coordinates | Use the input's `spatial_3d_aligned` |
| Human breast cancer | No | — | Use the input coordinates |
| Membryo | Yes | Coordinates | Centroid translation to E16.5 |
| Mcardiac_EA | Yes | Coordinates | `uot_alignment.py` |
| Human lung cancer | Yes | Features (`X_umap`) | `uot_alignment_trans.py` |
| Human gastric cancer | Yes | Features (`X_pca`) | `uot_alignment_trans.py` |
| Mcardiac | Yes | Coordinates | Scale each XYZ axis to [0, 1] within each timepoint |

Feature-based matching uses jointly computed features and transforms the coordinates. Membryo uses all cells from all eight sections for centroid alignment. Mcardiac applies scaling independently to each timepoint.

If no alignment is needed, copy the coordinates into the model fields:

```python
adata.obs["cx_aligned"] = adata.obsm["spatial"][:, 0]
adata.obs["cy_aligned"] = adata.obsm["spatial"][:, 1]
# For XYZ data, also set cz_aligned from the third column.
```

## Transition prior

`transition_prior.csv` lists prior cell-state transitions. Each row maps `src_layer` (source cell type or state) to `tgt_layer` (target cell type or state). Use label names or IDs that match the input annotations.

For example:

```text
src_layer,tgt_layer
Proepicardium,Cardiac fibroblasts
```

The listed transition is allowed; it does not require every source cell to change type.

Mcardiac, Human lung cancer, and Human gastric cancer keep this file in `data/`. Mcardiac_EA includes it in the experiment directory, with an additional `weight` column. Set its path with `transition_prior_path` in `config.yaml`.

For Mcardiac, the optional UOT workflow can infer the prior and write `discovered_transition.csv`; set `transition_prior_path` to that file.
