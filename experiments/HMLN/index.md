# HMLN

Use sections S1-S3 to reconstruct S1 to S3 with 10 simulation steps.

**Example route:** S1 to S3.

**Sample field:** `sample`. **Annotation field:** `annotation`.

## Input

Place `raw/Reconstructed_S2.h5ad`, `raw/Reconstructed_S3.h5ad`, and `raw/Reconstructed_S4.h5ad` under this experiment's `data/` directory. See [Data availability](../../docs/data.md) for upstream sources.

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
python experiments/decode_simulation.py --experiment experiments/HMLN
```

The command reads `config.yaml` and saves predicted expression in `reconstructed_expression`. Use `--frame-dir` to select a specific simulation run.

{download}`Download the decoding program <../decode_simulation.py>`.
