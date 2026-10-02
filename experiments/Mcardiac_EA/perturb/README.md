# Mcardiac_EA perturbation

This experiment applies Wt1-expression-weighted partial epicardial-cell ablation to the initial E10.5 CD1 state, then follows local niche/LR dynamics, cell growth/death and epicardial-to-fibroblast transition progression with the trained public model. E12.5 Wt1-DTA is held out for evaluation.

Run `../preprocess.ipynb` and `../train.ipynb` first. Training writes `../artifacts/training/E10.5_CD1_to_E12.5_CD1.json` with the prepared AnnData, Stage-1 trace and checkpoint, boundaries, decoder, transition policy and LR table.

From the repository root:

```bash
python experiments/Mcardiac_EA/perturb/run.py --dry-run
python experiments/Mcardiac_EA/perturb/run.py --device cuda:0
```

`config.json` defines the perturbation conditions and repeats. All config paths are resolved relative to this directory. `--dry-run` displays the planned commands.

`simulation.py` uses `stvirtual.models.stage2_3d_transition` and its decoder-backed LR computation. Results are written to `../artifacts/extrapolation`.

Each condition writes `ablation_free_simulation.npz`, `cell_counts_by_frame.csv`, `ablation_targets.csv`, `ablation_selection_by_cell.csv`, `E10.5_counterfactual_partial_ablation_selection.h5ad` and `config.json`. The default sweep exports the perturbation arm; direct `simulation.py` use can also export `normal_free_simulation.npz`. Preserve matching seeds and training artifacts when comparing perturbation with controls. Use separate output roots to retain separate runs.
