# Membryo

Combine all timepoints. Align section centers to E16.5.

**Example route:** E9.5 to E11.5.

**Sample field:** `timepoint`. **Annotation field:** `annotation`.

## Input

Place Mouse_embryo_all_stage.h5ad with raw counts, timepoint, annotation, and spatial coordinates under this experiment's `data/` directory. See [Data availability](../../docs/data.md) for upstream sources.

Set paths in {download}`config.yaml <config.yaml>`. Paths are relative to this experiment folder.

## Workflow

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
- {download}`decoder.ipynb <decoder.ipynb>`
- {download}`train.ipynb <train.ipynb>`

- {download}`results.ipynb <results.ipynb>`

## Decode saved frames

After simulation, run this command from the repository root:

```bash
python experiments/decode_simulation.py --experiment experiments/Membryo
```

The command reads `config.yaml` and saves predicted expression in `reconstructed_expression`. Use `--frame-dir` to select a specific simulation run.

{download}`Download the decoding program <../decode_simulation.py>`.
