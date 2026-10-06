# 3D tissue reconstruction

## 1. Input data

| Dataset | Folder | Required files |
| --- | --- | --- |
| [Mbrain](https://db.cngb.org/stomics/cbmsta/download/) | `experiments/Mbrain/data/` | `Mouse1_T168.h5ad`, `Mouse1_T169.h5ad`, `Mouse1_T170.h5ad`, `Mouse1_T171.h5ad` |
| [HMLN](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) | `experiments/HMLN/data/raw/` | `Reconstructed_S2.h5ad`, `Reconstructed_S3.h5ad`, `Reconstructed_S4.h5ad` (raw counts, annotations, aligned coordinates) |
| [Human breast cancer](https://doi.org/10.5281/zenodo.4752030) | `experiments/human_breast_cancer/data/` | `imc_all.h5ad` (`sample`, `gaston_region`, protein measurements, spatial coordinates) |

Human breast cancer: add `gaston_region` labels with [GASTON](https://github.com/raphael-group/GASTON).

Included LR tables: `lrpairs/mouse/LR_pairs.csv` for Mbrain; `lrpairs/human/LR_pairs.csv` for HMLN. Human breast cancer does not use an LR table.

## 2. Edit config

Paths are relative to each dataset directory. Set input and output paths in the files below.

| Dataset | Config | Edit |
| --- | --- | --- |
| Mbrain | [config.yaml](../../experiments/Mbrain/config.yaml) | `data_root`, `route_ids` |
| HMLN | [config.yaml](../../experiments/HMLN/config.yaml) | `data_root`, `sample_mapping`, `routes` |
| Human breast cancer | [config.yaml](../../experiments/human_breast_cancer/config.yaml) | `input_path`, `prepared_path`, `routes` |

HMLN: `sample_mapping` maps source S2/S3/S4 to S1/S2/S3.

## 3. Preprocessing

| Dataset | Run | Task |
| --- | --- | --- |
| Mbrain | [preprocess.ipynb](../../experiments/Mbrain/preprocess.ipynb) | Counts, scanVI, spatial alignment |
| HMLN | [preprocess.ipynb](../../experiments/HMLN/preprocess.ipynb) | Counts, scanVI |
| Human breast cancer | [preprocess.ipynb](../../experiments/human_breast_cancer/preprocess.ipynb) | Protein features |

HMLN uses the input’s `spatial_3d_aligned` coordinates without additional spatial alignment.

## 4. Train decoder

| Dataset | Run |
| --- | --- |
| Mbrain | [decoder.ipynb](../../experiments/Mbrain/decoder.ipynb) |
| HMLN | [decoder.ipynb](../../experiments/HMLN/decoder.ipynb) |
| Human breast cancer | Skip |

Human breast cancer uses the measured protein features directly because they are already low-dimensional.

Train one decoder per selected sample pair using all genes retained in the input.

## 5. Train model

| Dataset | Run |
| --- | --- |
| Mbrain | [train.ipynb](../../experiments/Mbrain/train.ipynb) |
| HMLN | [train.ipynb](../../experiments/HMLN/train.ipynb) |
| Human breast cancer | [train.ipynb](../../experiments/human_breast_cancer/train.ipynb) |

<a id="output-contract"></a>

## 6. Results

Paths below are relative to `experiments/<dataset>/`.

| Files | Folder |
| --- | --- |
| Trained models | `artifacts/checkpoints/` |
| Simulation frames | `artifacts/results/simulation/<src>_to_<tgt>/` |
| Overview image | `artifacts/results/overview/<src>_to_<tgt>.png` |

Run the last cell of `train.ipynb` to save the overview image. Open it from `artifacts/results/overview/`.

## 7. Existing result preview

### Mbrain — T168 to T171

![Mbrain simulation](../assets/results/mbrain.gif)

### HMLN — S1 to S3, S4 to S6, S7 to S9

![HMLN simulation](../assets/results/hmln.gif)

### Human breast cancer — Slices 0 to 14

![Human breast cancer simulation](../assets/results/human_breast_cancer.gif)

<a id="human-breast-cancer-data-workflow"></a>
<a id="human-breast-cancer-protein-workflow"></a>
