# Mcardiac perturbation experiment

Apply cell-state transition-specific reciprocal LR-gene scaling to the E9.5h source and generate the E9.5h-to-E11.5h trajectory with the trained scanVI, transport, decoder and 4D transition models.

## Prepare and run

Complete the dataset's preprocessing, decoder training and model training. Run from the repository root:

```bash
python experiments/Mcardiac/perturb/run.py --dry-run
python experiments/Mcardiac/perturb/run.py --device cuda:0
```

The entry reads `config.yaml`. Use `--config` to select another condition. All config paths are resolved relative to this directory; training inputs use the dataset's `../config.yaml`. The dry run displays input paths and readiness.

## Intervention and analysis

The command entry and notebook use the shared `stvirtual.perturb.PerturbationSession`. After the intervention, the fitted scanVI model recomputes the representation stored in `obsm["X_scanvi_pert"]` for transport and transition inference.

Use `perturb_ana.ipynb` for cell-state transition-specific LR-gene perturbation and transition analysis. Set `baseline_frames` to one complete baseline simulation. The initial frame identifies source cells; their trajectories define future atrial, future ventricular, shared and remaining cell-state transition groups.

`lr_gene_sets.json` defines the LR-gene groups. The `down` and `up` settings scale their expression; set both to 1 for a matched control. Shared genes receive both multipliers. The notebook records the baseline-to-simulation UID mapping and writes baseline expression to `baseline_decoded/`.

## Results

Set `output` in `config.yaml` to a separate directory for each condition. `run.py` writes `perturbed_counts.h5ad`, `model_input.h5ad`, `stage1_trace.npz`, `bound/`, `simulation/` and `report.json` under the experiment's `artifacts/perturb`.

The notebook also writes `decoded/` with predicted expression in `counts_hat` and log-transformed values in `log1p_hat`.
