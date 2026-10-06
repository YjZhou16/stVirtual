# Data Availability

Download each dataset from its official repository, review its license or access conditions, and place or symlink the required files under the corresponding `experiments/<dataset>/data/` directory. Preprocessing notebooks write generated checkpoints and results under `artifacts/`.

## Public experiment datasets

| stVirtual experiment | Dataset used in the manuscript | Species | Official source | Access notes |
|---|---|---|---|---|
| <a id="hmln"></a>`HMLN` | HMLN / Open-ST | Human | [GEO GSE251926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE251926&format=file) | Use reconstructed sections S2, S3 and S4; see the exact input list below. |
| <a id="human-gastric-cancer"></a>`human_gastric_cancer` | Human gastric cancer | Human | [OMIX010346](https://ngdc.cncb.ac.cn/omix/release/OMIX010346) | The OMIX record is controlled-access; request authorization through the repository before downloading. |
| <a id="human-lung-cancer"></a>`human_lung_cancer` | Human lung adenocarcinoma progression | Human | [GEO GSE307534](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE307534&format=file) | Use patient P2 AAH and P2 LUAD, with matched clone annotations. |
| <a id="membryo"></a>`Membryo` | Mouse embryo development | Mouse | [MOSTA portal](https://db.cngb.org/stomics/mosta/) · [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/) | Download the required embryo stages, including the E9.5–E16.5 stages used in the tutorial. |
| <a id="mcardiac"></a>`Mcardiac` | Mouse cardiac development | Mouse | [MOSTA portal](https://db.cngb.org/stomics/mosta/) · [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/) | Download the heart-stage files required by the Mcardiac workflow. |
| <a id="mbrain"></a>`Mbrain` | Mouse cerebellum atlas | Mouse | [CBMSTA portal](https://db.cngb.org/stomics/cbmsta/) · [CBMSTA downloads](https://db.cngb.org/stomics/cbmsta/download/) | The tutorial uses the Mouse1 T168–T171 series available from the download page. |
| <a id="human-breast-cancer"></a>`human_breast_cancer` | Human breast-cancer imaging mass cytometry (IMC) | Human | [Zenodo DOI 10.5281/zenodo.4752030](https://doi.org/10.5281/zenodo.4752030) | Prepare the annotated `imc_all.h5ad` required by `experiments/human_breast_cancer/preprocess.ipynb`, including measured protein features, spatial coordinates and `gaston_region` labels annotated with [GASTON](https://github.com/raphael-group/GASTON); see the [Human breast cancer input contract](../experiments/human_breast_cancer/index.md). |
| <a id="mcardiac_ea"></a>`Mcardiac_EA` | Developing mouse heart Slide-seq | Mouse | [GEO GSE282547](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282547) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE282547&format=file) | Use GSM9065770, GSM8645803 and GSM8645808 with barcode-matched coordinates. The CD1 samples define the training route; Wt1-DTA is retained for perturbation evaluation. See the [Mcardiac_EA input guide](../experiments/Mcardiac_EA/index.md). |

## Files required before running

All paths below are relative to `experiments/<dataset>/`. A filename in this table is the notebook's expected input; reconstructed or annotated inputs may require upstream preparation rather than a simple rename.

| Experiment | Required local inputs | Preparation before the notebook |
| --- | --- | --- |
| Mbrain | `data/Mouse1_T168.h5ad`, `Mouse1_T169.h5ad`, `Mouse1_T170.h5ad`, `Mouse1_T171.h5ad` in the same directory | Download these four annotated H5AD files from CBMSTA Stereo-seq data. Keep their original names. |
| HMLN | `data/raw/Reconstructed_S2.h5ad`, `Reconstructed_S3.h5ad`, `Reconstructed_S4.h5ad` | Reconstructed Open-ST sections with `layers["raw"]`, `obsm["spatial_3d_aligned"]`, and `obs["annotation"]`. These are processed section inputs, not the raw sequencing archive. |
| Human lung cancer | `data/source.h5ad` | P2 AAH and P2 LUAD counts, coordinates, `status`, and `_clone_s` annotations. |
| Human gastric cancer | `data/raw/filtered_feature_bc_matrix.h5`, the associated `data/raw/spatial/` directory, and `data/P1_cluster.csv` | Obtain GP1/P1 Visium data through OMIX access. Keep barcode-matched `Cell` and `cluster` annotations. The notebook reads the Visium input and performs its region selection. |
| Membryo | `data/Mouse_embryo_all_stage.h5ad` | Combined, annotated full sections from E9.5, E10.5, E11.5, E12.5, E13.5, E14.5, E15.5 and E16.5, with `timepoint`, `annotation` and `obsm["spatial"]`. Preserve counts and stage identity when assembling MOSTA sections. |
| Mcardiac | `data/Mosta_heart_9h.h5ad`, `data/Mosta_heart_11h_full.h5ad` | Prepared E9.5 and E11.5 heart volumes with cell annotations and XYZ coordinates. These are heart-specific derived files, not generic whole-embryo inputs. |
| Mcardiac_EA | `data/GSE282547/GSM9065770_SlideSeq_e10.5_1.h5ad`, `GSM8645803_SlideSeq_e12_2.h5ad`, `GSM8645808_SlideSeq_e12_wt1dta_1.h5ad` in that directory; coordinate CSVs with the same stems under `data/coordinates/` | Convert source count objects to AnnData while preserving barcodes and annotations. Match coordinate rows by barcode. Keep `roi_polygons.json` in the experiment directory. |
| Human breast cancer | `data/imc_all.h5ad` | Use the downloaded H5AD data and add GASTON region annotations; see [input preparation](breast-input.md). |


## Ligand–receptor tables

The public notebooks expect species-specific LR tables at:

```text
lrpairs/human/LR_pairs.csv
lrpairs/mouse/LR_pairs.csv
```

Human gastric cancer, Human lung cancer, and HMLN use the human table. Mbrain, Membryo, Mcardiac, and Mcardiac_EA use the mouse table. Human breast cancer uses measured protein features.

## Recommended local layout

```text
experiments/<dataset>/
├── data/                         # downloaded files or symlinks
├── config.yaml
├── preprocess.ipynb
├── decoder.ipynb                 # when required by the experiment
├── train.ipynb
└── artifacts/                    # generated checkpoints and results
```

Verify checksums supplied by the upstream repository when available, and record the exact upstream file names used for reproducibility.
