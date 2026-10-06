# Mbrain

Combine T168-T171 and align to T171. The example reconstructs T168 to T170 with 20 simulation steps.

**Example route:** T168 to T170.

**Sample field:** `sample`. **Annotation field:** `annotation`.

## Input

Place Mouse1_T168.h5ad, Mouse1_T169.h5ad, Mouse1_T170.h5ad, Mouse1_T171.h5ad under this experiment's `data/` directory. See [Data availability](../../docs/data.md) for upstream sources.

Set paths in {download}`config.yaml <config.yaml>`. Paths are relative to this experiment folder.

## Notebooks

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
python experiments/decode_simulation.py --experiment experiments/Mbrain
```

The command reads `config.yaml` and saves predicted expression in `reconstructed_expression`. Use `--frame-dir` to select a specific simulation run.

{download}`Download the decoding program <../decode_simulation.py>`.
