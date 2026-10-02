# Human gastric cancer perturbation experiment

Remove ALDH3A1-high normal cells and generate the normal-to-cancer trajectory with the trained scanVI, transport, decoder and transition models.

## Prepare and run

Complete the dataset's preprocessing, decoder training and model training. Run from the repository root:

```bash
python experiments/human_gastric_cancer/perturb/run.py --dry-run
python experiments/human_gastric_cancer/perturb/run.py --device cuda:0
```

The entry reads `config.yaml`. Use `--config` to select another condition. All config paths are resolved relative to this directory; training inputs use the dataset's `../config.yaml`. The dry run displays input paths and readiness.

## Intervention and analysis

The command entry and notebook use the shared `stvirtual.perturb.PerturbationSession`. After the intervention, the fitted scanVI model recomputes the representation stored in `obsm["X_scanvi_pert"]` for transport and transition inference.

Use `perturb_ana.ipynb` for cell removal, simulation of cell-state transitions and expression analysis. The default quantile selection removes source cells at or above the configured expression threshold, including ties. Use `exact_topk.yaml` to select an exact fraction of high-expression cells.

For a matched control, use `selection: exact_topk` and `fraction: 0` with the same seed and checkpoints. Set `baseline_frames` to the corresponding baseline simulation for the before/after comparison.

## Results

Set `output` in `config.yaml` to a separate directory for each condition. `run.py` writes `perturbed_counts.h5ad`, `model_input.h5ad`, `stage1_trace.npz`, `bound/`, `simulation/` and `report.json` under the experiment's `artifacts/perturb`.

The notebook also writes `decoded/` with predicted expression in `counts_hat` and log-transformed values in `log1p_hat`.
