# Human lung cancer

Model AAH to LUAD with clone labels. Shared cell features guide coordinate alignment.

**Example route:** AAH to LUAD.

**Sample field:** `status`. **Annotation field:** `_clone_s`.

## Input

Place source.h5ad with counts, status, _clone_s, spatial; transition_prior.csv under this experiment's `data/` directory. See [Data availability](../../docs/data.md) for upstream sources.

Set paths in {download}`config.yaml <config.yaml>`. Paths are relative to this experiment folder.

## Transition prior

`data/transition_prior.csv` provides the **prior cell-state transitions**. Each row maps `src_layer` (source cell type or state) to `tgt_layer` (target cell type or state); labels must match the experiment annotations. Set its path with `transition_prior_path` in `config.yaml`. See [Transition prior](../../docs/preprocessing.md#transition-prior).

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

## Continue with perturbation

[Open the perturbation walkthrough](perturb/index.md).

- {download}`results.ipynb <results.ipynb>`

## Decode saved frames

After simulation, run this command from the repository root:

```bash
python experiments/decode_simulation.py --experiment experiments/human_lung_cancer
```

The command reads `config.yaml` and saves predicted expression in `reconstructed_expression`. Use `--frame-dir` to select a specific simulation run.

{download}`Download the decoding program <../decode_simulation.py>`.
