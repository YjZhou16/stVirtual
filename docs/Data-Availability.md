# Data Availability

Download each dataset from its official repository, review its license or access conditions, and place or symlink the required files under the corresponding `experiments/<dataset>/data/` directory. Preprocessing notebooks write generated checkpoints and results under `artifacts/`.

## Public experiment datasets

| stVirtual experiment | Dataset used in the manuscript | Species | Official source | Access notes |
|---|---|---|---|---|
| <a id="hmln"></a>`HMLN` | HMLN / Open-ST | Human | [GEO GSE251926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE251926&format=file) | Download the Open-ST files required by `experiments/HMLN/preprocess.ipynb`. |
| <a id="human-gastric-cancer"></a>`human_gastric_cancer` | Human gastric cancer | Human | [OMIX010346](https://ngdc.cncb.ac.cn/omix/release/OMIX010346) | The OMIX record is controlled-access; request authorization through the repository before downloading. |
| <a id="human-lung-cancer"></a>`human_lung_cancer` | Human lung adenocarcinoma progression | Human | [GEO GSE307534](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE307534&format=file) | Select the AAH and LUAD samples used by the preprocessing notebook. |
| <a id="membryo"></a>`Membryo` | Mouse embryo development | Mouse | [MOSTA portal](https://db.cngb.org/stomics/mosta/) · [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/) | Download the required embryo stages, including the E9.5–E15.5 stages used in the tutorial. |
| <a id="mcardiac"></a>`Mcardiac` | Mouse cardiac development | Mouse | [MOSTA portal](https://db.cngb.org/stomics/mosta/) · [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/) | Download the heart-stage files required by the Mcardiac workflow. |
| <a id="mbrain"></a>`Mbrain` | Mouse cerebellum atlas | Mouse | [CBMSTA portal](https://db.cngb.org/stomics/cbmsta/) · [CBMSTA downloads](https://db.cngb.org/stomics/cbmsta/download/) | The tutorial uses the Mouse1 T167–T171 series available from the download page. |
| <a id="human-breast-cancer"></a>`human_breast_cancer` | Human breast-cancer imaging mass cytometry (IMC) | Human | [Zenodo DOI 10.5281/zenodo.4752030](https://doi.org/10.5281/zenodo.4752030) | Prepare the annotated `imc_all.h5ad` required by `experiments/human_breast_cancer/preprocess.ipynb`, including measured protein features, spatial coordinates and `gaston_region` labels annotated with [GASTON](https://github.com/raphael-group/GASTON); see the [Human breast cancer input contract](../experiments/human_breast_cancer/README.md). |
| <a id="mcardiac_ea"></a>`Mcardiac_EA` | Developing mouse heart Slide-seq | Mouse | [GEO GSE282547](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282547) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE282547&format=file) | Use GSM9065770, GSM8645803 and GSM8645808 with barcode-matched coordinates. The CD1 samples define the training route; Wt1-DTA is retained for perturbation evaluation. See the [Mcardiac_EA input guide](../experiments/Mcardiac_EA/README.md). |

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
