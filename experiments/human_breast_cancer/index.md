# Human breast cancer

Use measured proteins and GASTON region labels directly, without scanVI, LR inputs, or an expression decoder.

**Example route:** Sections 0 to 14 in increments of 2.

**Sample field:** `sample`. **Annotation field:** `gaston_region`.

## Input

Place imc_all.h5ad with sample, gaston_region, measured protein features, and spatial coordinates under this experiment's `data/` directory. See [Data availability](../../docs/data.md) for upstream sources.

Set paths in {download}`config.yaml <config.yaml>`. Paths are relative to this experiment folder.

## Workflow

```{toctree}
:maxdepth: 1

../../docs/breast-input
preprocess
train
results
```

## Download

- {download}`colors.json <colors.json>`

- {download}`preprocess.ipynb <preprocess.ipynb>`
- {download}`train.ipynb <train.ipynb>`

- {download}`results.ipynb <results.ipynb>`
