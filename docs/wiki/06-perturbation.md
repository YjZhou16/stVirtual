# Perturbation

Each dataset keeps its perturbation configuration and entry points inside its own experiment.

| Dataset | Workflow | Config |
| --- | --- | --- |
| Human gastric cancer | [Human gastric cancer perturbation](../../experiments/human_gastric_cancer/perturb/README.md) | `experiments/human_gastric_cancer/perturb/config.yaml` |
| Human lung cancer | [Human lung cancer perturbation](../../experiments/human_lung_cancer/perturb/README.md) | `experiments/human_lung_cancer/perturb/config.yaml` |
| Mcardiac | [Mcardiac perturbation](../../experiments/Mcardiac/perturb/README.md) | `experiments/Mcardiac/perturb/config.yaml` |
| Mcardiac_EA | [Mcardiac_EA perturbation](../../experiments/Mcardiac_EA/perturb/README.md) | `experiments/Mcardiac_EA/perturb/config.json` |

## Human gastric cancer, Human lung cancer and Mcardiac

The command entries and notebooks share one perturbation workflow. Each task applies its own intervention, then uses the fitted scanVI model to recompute `X_scanvi_pert`. Human gastric cancer and Human lung cancer use 3D transition models; Mcardiac uses the 4D transition model.

- **Human gastric cancer:** remove ALDH3A1-high normal cells before normal-to-cancer inference.
- **Human lung cancer:** suppress selected gene expression in the configured AAH clones, then simulate the AAH-to-LUAD trajectory against the original LUAD reference.
- **Mcardiac:** modify ventricular/atrial LR-gene expression in baseline cell-state transition groups before E9.5h-to-E11.5h inference.

Complete the corresponding dataset's preprocessing and model training first. These perturbation analyses reuse its trained scanVI, transport, decoder and transition checkpoints. They re-encode perturbed counts, recompute transport and boundaries, and generate transition frames.

Run from the repository root:

```bash
python experiments/human_gastric_cancer/perturb/run.py --dry-run
python experiments/human_gastric_cancer/perturb/run.py
python experiments/human_lung_cancer/perturb/run.py
python experiments/Mcardiac/perturb/run.py
```

Each entry defaults to its adjacent `config.yaml`. Use `--config` for another condition. Inputs and output paths are resolved relative to their configuration files; generated results are written under the corresponding experiment's `artifacts/perturb`.

Mcardiac uses a complete baseline simulation to identify source-cell cell-state transition groups and link their UIDs to the perturbed trajectories. Outputs include perturbed input, transport traces, boundaries, annotated transition frames and a report of selected cells or genes.

For matched controls, retain the same training artifacts and sampling setup while disabling the configured intervention. Use a separate output directory for every condition. Numerical settings are defined in the experiment configs.

## Mcardiac_EA: epicardial-cell ablation

Use `experiments/Mcardiac_EA/perturb` for the epicardial-cell perturbation experiment. It starts from E10.5 CD1, selects epicardial cells using Wt1-expression-weighted sampling and removes the selected cells at initialization. The remaining population undergoes local niche and LR updates, cell growth/death and epicardial-to-fibroblast transition progression.

The simulation follows the current simulated state and local neighborhood. E12.5 Wt1-DTA provides the perturbation evaluation condition.

### Prepare and run

1. Complete `experiments/Mcardiac_EA/preprocess.ipynb`.
2. Run `experiments/Mcardiac_EA/train.ipynb`. This produces the trained models, decoder, transport trace, boundaries and training manifest.
3. Run the perturbation entry from the repository root:

```bash
python experiments/Mcardiac_EA/perturb/run.py --dry-run
python experiments/Mcardiac_EA/perturb/run.py
```

The entry reads `config.json` and obtains input paths from `artifacts/training/E10.5_CD1_to_E12.5_CD1.json`. It uses the current public transition-model and decoder interfaces. The dry run displays the planned commands.

### Results and comparisons

Results are written under `experiments/Mcardiac_EA/artifacts/extrapolation`, with separate directories for configured intervention conditions and repeats. Each condition includes:

- `ablation_free_simulation.npz`: coordinates, expression embeddings, transition state, UIDs and parent UIDs across frames.
- `cell_counts_by_frame.csv`: cell-type counts over the simulated trajectory.
- `ablation_targets.csv` and `ablation_selection_by_cell.csv`: selected cells, Wt1 expression and selection annotations.
- `E10.5_counterfactual_partial_ablation_selection.h5ad`: the annotated initial selection.
- `config.json`: the settings and input provenance for that run.

The default batch exports perturbed trajectories. The underlying `simulation.py` also supports an unperturbed trajectory for matched comparisons. Keep the same training artifacts and sampling setup for each comparison, and use separate output roots for distinct runs.

See the [Mcardiac_EA training workflow](02-developmental-reconstruction-across-time.md#mcardiac_ea-workflow) for preprocessing and training, and the [experiment guide](../../experiments/Mcardiac_EA/perturb/README.md) for configuration details.
