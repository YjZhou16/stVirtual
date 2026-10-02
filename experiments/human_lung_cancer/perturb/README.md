# Human lung cancer perturbation experiment

Suppress selected gene expression in the configured AAH clones and generate the AAH-to-LUAD trajectory with the trained scanVI, transport, decoder and transition models.

## Prepare and run

Complete the dataset's preprocessing, decoder training and model training. Run from the repository root:

```bash
python experiments/human_lung_cancer/perturb/run.py --dry-run
python experiments/human_lung_cancer/perturb/run.py --device cuda:0
```

The entry reads `config.yaml`. Use `--config` to select another condition. All config paths are resolved relative to this directory; training inputs use the dataset's `../config.yaml`. The dry run displays input paths and readiness.

## Intervention and analysis

The command entry and notebook use the shared `stvirtual.perturb.PerturbationSession`. After the intervention, the fitted scanVI model recomputes the representation stored in `obsm["X_scanvi_pert"]` for transport and transition inference.

Use `insilico_knockout.ipynb` for gene perturbation, simulation of cell-state transitions and expression analysis. Set `celltypes` and either an explicit `genes` list or `random_gene_count` in `config.yaml`. The `factor` setting scales the selected counts; `factor: 1` gives a matched control.

`intervention_samples: [AAH]` selects the starting sample. The LUAD reference retains its original expression. This experiment applies initial expression suppression; downstream decoded expression can recover nonzero values. The run report records the affected samples, cells and genes.

Repeat random gene perturbations with:

```bash
python experiments/human_lung_cancer/perturb/perturb_gene.py --device cuda:0
```

The batch entry writes per-seed gene lists, diagnostic curves and summaries under `../artifacts/perturb/perturbation_test`, together with `all_runs_overview.csv`, `all_gene_lists.csv`, `all_diag_piece.csv` and `all_alpha_threshold_summary.csv`.

Use `draw_paper2.ipynb` for condition comparisons and repeated-run figures. It reads `origin/diag_piece1.csv`, `top10_results/diag_piece1.csv` and the batch summaries under `../artifacts/perturb`.

## Results

Set `output` in `config.yaml` to a separate directory for each condition. `run.py` writes `perturbed_counts.h5ad`, `model_input.h5ad`, `stage1_trace.npz`, `bound/`, `simulation/` and `report.json` under the experiment's `artifacts/perturb`.

The notebook also writes `decoded/` with predicted expression in `counts_hat` and log-transformed values in `log1p_hat`.
