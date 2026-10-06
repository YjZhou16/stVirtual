# Perturbation

## 1. Input data

Complete the dataset workflow first. Keep the trained scanVI, decoder, Stage 1 and Stage 2 models.

| Dataset | Data and training | Additional input |
| --- | --- | --- |
| Human gastric cancer | [Workflow](03-cancer-progression-across-stages.md) | Full P1 counts in `data/raw/` and `data/P1_cluster.csv` |
| Human lung cancer | [Workflow](03-cancer-progression-across-stages.md) | Prepared AAH/LUAD counts |
| Mcardiac | [Workflow](04-4d-spatiotemporal-development-reconstruction.md) | One baseline run's `.h5ad` frames; `perturb/lr_gene_sets.json` |
| Mcardiac_EA | [Workflow](02-developmental-reconstruction-across-time.md) | `artifacts/training/E10.5_CD1_to_E12.5_CD1.json` |

## 2. Edit config

| Dataset | Config | Edit |
| --- | --- | --- |
| Human gastric cancer | [config.yaml](../../experiments/human_gastric_cancer/perturb/config.yaml) | `gene`, `selection`, `fraction`/`quantile`, `output` |
| Human lung cancer | [config.yaml](../../experiments/human_lung_cancer/perturb/config.yaml) | `genes`, `celltypes`, `factor`, `output` |
| Mcardiac | [config.yaml](../../experiments/Mcardiac/perturb/config.yaml) | `baseline_frames` (one run), `lr_gene_sets`, `down`, `up`, `output` |
| Mcardiac_EA | [config.json](../../experiments/Mcardiac_EA/perturb/config.json) | `fractions`, `seeds`, `training_manifest`, `output_root` |

## 3. Preprocessing

| Dataset | Run before perturbation |
| --- | --- |
| Human gastric cancer | [preprocess.ipynb](../../experiments/human_gastric_cancer/preprocess.ipynb) |
| Human lung cancer | [preprocess.ipynb](../../experiments/human_lung_cancer/preprocess.ipynb) |
| Mcardiac | [preprocess.ipynb](../../experiments/Mcardiac/preprocess.ipynb) |
| Mcardiac_EA | [preprocess.ipynb](../../experiments/Mcardiac_EA/preprocess.ipynb) |

## 4. Train decoder

| Dataset | Run before perturbation |
| --- | --- |
| Human gastric cancer | [decoder.ipynb](../../experiments/human_gastric_cancer/decoder.ipynb) |
| Human lung cancer | [decoder.ipynb](../../experiments/human_lung_cancer/decoder.ipynb) |
| Mcardiac | [decoder.ipynb](../../experiments/Mcardiac/decoder.ipynb) |
| Mcardiac_EA | [decoder.ipynb](../../experiments/Mcardiac_EA/decoder.ipynb) |

Reuse the saved decoder for perturbation.

## 5. Train model and run perturbation

| Dataset | Train first | Run perturbation |
| --- | --- | --- |
| Human gastric cancer | [train.ipynb](../../experiments/human_gastric_cancer/train.ipynb) | [perturb.ipynb](../../experiments/human_gastric_cancer/perturb/perturb.ipynb) |
| Human lung cancer | [train.ipynb](../../experiments/human_lung_cancer/train.ipynb) | [perturb.ipynb](../../experiments/human_lung_cancer/perturb/perturb.ipynb) |
| Mcardiac | [train.ipynb](../../experiments/Mcardiac/train.ipynb) | [perturb.ipynb](../../experiments/Mcardiac/perturb/perturb.ipynb) |
| Mcardiac_EA | [train.ipynb](../../experiments/Mcardiac_EA/train.ipynb) | [perturb.ipynb](../../experiments/Mcardiac_EA/perturb/perturb.ipynb) |

Reuse the trained models for perturbation.

## 6. Results

Paths below are relative to `experiments/<dataset>/`. Each notebook prints its output directory.

| Dataset | Result folder | Files |
| --- | --- | --- |
| Human gastric cancer | `artifacts/perturb/<run>/` | `simulation/*.h5ad`, `stage1_trace.npz`, `report.json` |
| Human lung cancer | `artifacts/perturb/<run>/` | `simulation/*.h5ad`, `stage1_trace.npz`, `report.json` |
| Mcardiac | `artifacts/perturb/<run>/` | `simulation/*.h5ad`, `stage1_trace.npz`, `report.json` |
| Mcardiac_EA | `artifacts/extrapolation/<run>/` | `ablation_free_simulation.npz`, `cell_counts_by_frame.csv`, `config.json` |

## 7. Result preview

Open the saved execution output in each `perturb/perturb.ipynb`. No result figures are included.

<a id="mcardiac_ea-epicardial-cell-ablation"></a>
