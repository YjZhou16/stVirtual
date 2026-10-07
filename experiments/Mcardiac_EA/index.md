# Mcardiac_EA

Use Slide-seq tissue regions for epicardial cell removal.

**Example route:** E10.5_CD1 to E12.5_CD1.

**Sample field:** `sample`. **Annotation field:** `cell_type`.

## Input

Place the expression H5AD files in `data/GSE282547/` and barcode-matched CSVs in `data/coordinates/`. Keep `roi_polygons.json` in this experiment folder. See [Data availability](../../docs/data.md) for upstream sources.

Set paths in {download}`config.yaml <config.yaml>`. Paths are relative to this experiment folder.

## Transition prior

`transition_prior.csv` provides the **prior cell-state transitions**. Each row maps `src_layer` (source cell type or state) to `tgt_layer` (target cell type or state); labels must match the experiment annotations. Set its path with `transition_prior_path` in `config.yaml`. See [Transition prior](../../docs/preprocessing.md#transition-prior).

## Workflow

Run preprocessing, decoder training, then transport training.

```{toctree}
:maxdepth: 1

preprocess
decoder
train
results
```

## Download

- {download}`colors.json <colors.json>`

- {download}`preprocess.ipynb <preprocess.ipynb>`
- {download}`train.ipynb <train.ipynb>`
- {download}`decoder.ipynb <decoder.ipynb>`

## Continue with perturbation

[Open the perturbation walkthrough](perturb/index.md).

- {download}`results.ipynb <results.ipynb>`

## Decode saved frames

After simulation, run this command from the repository root:

```bash
python experiments/decode_simulation.py --experiment experiments/Mcardiac_EA
```

The command reads `config.yaml` and saves predicted expression in `reconstructed_expression`. Use `--frame-dir` to select a specific simulation run.

{download}`Download the decoding program <../decode_simulation.py>`.
