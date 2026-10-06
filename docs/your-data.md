# Use your own data

Copy the experiment closest to your data into a new dataset folder. Keep `experiments/_shared/` and the LR tables alongside it.

## Prepare input

Combine your sections or timepoints into one AnnData object. Store their names in `obs["sample"]`, cell-type labels in an annotation column, feature names in `var_names`, and coordinates in `obsm["spatial"]`. Preserve the original counts in `layers["counts"]`.

Update the notebooks' experiment folder, input paths, sample key, feature key and feature dimension to match your data.

## Spatial alignment

First decide whether your sections or timepoints need alignment. Choose a method from [Data preprocessing guide](preprocessing.md#spatial-alignment). If no alignment is needed, copy the coordinates into `cx_aligned`/`cy_aligned`; for XYZ data, also set `cz_aligned`.

## Set model features

For low-dimensional data, use the original counts directly as model features; no additional representation is needed:

```python
import numpy as np
from scipy import sparse

counts = adata.layers["counts"]
adata.obsm["X_features"] = (
    counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
)
```

For high-dimensional data, learn a compact representation such as scanVI or PCA. The model takes these features as input, rather than the full high-dimensional count matrix. Set `latent_key` to the corresponding `obsm` key.

## Edit config

Set the following fields in your experiment's `config.yaml`:

| Field | Set to |
| --- | --- |
| `data_root` | Your input directory |
| `route_ids` or `routes` | Source and target names from the saved sample metadata |
| `celltype_source_key` | Your annotation column |
| `latent_key` | `X_features` for direct input, or your learned representation key |
| Checkpoint and result paths | Output directories for this dataset |
| `lr_pairs_path` | The matching LR table, when LR inputs are used |
| `transition_prior_path` | `transition_prior.csv`, when explicit cell-state transitions are used |

If you do not use LR inputs, follow the configuration with `use_lr: false` in the Human breast cancer example. Use the XY or XYZ model matching your coordinates.

## Train decoder

When using a learned representation, run `decoder.ipynb` before model training. Train one decoder per selected sample pair using all features retained in the input. The decoder and model must use the same representation.

Direct low-dimensional inputs do not need a decoder.

## Train model

Run `train.ipynb` with your saved input and configuration. Models and simulation frames are saved under the configured checkpoint and result directories in `artifacts/`.

## View results

In the Tutorial, open `results.ipynb` and set its input and frame paths to your new run. In the main repository, the last cell of `train.ipynb` saves the frame overview under `artifacts/results/overview/`.

### Read and decode simulation results

Read a saved frame with `anndata.read_h5ad`. The frame contains simulated coordinates, cell-type labels, and model features in `obsm["X_latent"]`.

```python
from pathlib import Path
import anndata as ad

frame_path = Path("path/to/your/simulation/frame.h5ad")
frame = ad.read_h5ad(frame_path)
print(frame)
```

If you trained a decoder, convert the saved model features back to predicted measurements after simulation:

```bash
python experiments/decode_simulation.py \
  --experiment experiments/your_dataset \
  --source SOURCE --target TARGET \
  --frame-dir path/to/your/simulation \
  --output-dir path/to/your/decoded
```

The decoded files store predictions in `layers["reconstructed_expression"]` with feature names in `var_names`. Use the decoder and Stage-1 checkpoints belonging to that run. For direct low-dimensional inputs, use the matching Stage-1 feature transform to return simulated values to their input scale.
