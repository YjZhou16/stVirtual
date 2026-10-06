# Mcardiac

Model E9.5h to E11.5h using XYZ coordinates. A transition map can be supplied or estimated with UOT.

**Example route:** E9.5h to E11.5h.

**Sample field:** `stage`. **Annotation field:** `mapped_celltype`.

## Input

Place Mosta_heart_9h.h5ad, Mosta_heart_11h_full.h5ad, transition_prior.csv under this experiment's `data/` directory. See [Data availability](../../docs/data.md) for upstream sources.

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

## Optional inferred transition prior

Run this after preprocessing and before training to derive a prior with joint UOT. Select the generated table using `transition_prior_path`.

```{toctree}
:maxdepth: 1

infer_prior
```

- {download}`infer_prior.ipynb <infer_prior.ipynb>`
- {download}`infer_transition_map.py <infer_transition_map.py>`

## Decode saved frames

After simulation, run this command from the repository root:

```bash
python experiments/decode_simulation.py --experiment experiments/Mcardiac
```

The command reads `config.yaml` and saves predicted expression in `reconstructed_expression`. Use `--frame-dir` to select a specific simulation run.

{download}`Download the decoding program <../decode_simulation.py>`.
