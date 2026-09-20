# Data Availability

stVirtual does not redistribute the source datasets. Download each dataset from its official repository, review its license or access conditions, and place or symlink the required files under the corresponding `experiments/<dataset>/data/` directory. Preprocessing notebooks write generated files only under `artifacts/` and must not modify downloaded source data.

## Public experiment datasets

| stVirtual experiment | Dataset used in the manuscript | Species | Official source | Access notes |
|---|---|---|---|---|
| <a id="openst"></a>`OpenST` | HMLN / Open-ST | Human | [GEO GSE251926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE251926&format=file) | Download the Open-ST files required by `experiments/OpenST/preprocess.ipynb`. |
| <a id="gp1"></a>`GP1` | Human gastric cancer | Human | [OMIX010346](https://ngdc.cncb.ac.cn/omix/release/OMIX010346) | The OMIX record is controlled-access; request authorization through the repository before downloading. |
| <a id="luad"></a>`LUAD` | Human lung adenocarcinoma progression | Human | [GEO GSE307534](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307534) · [GEO files](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE307534&format=file) | Select the AAH and LUAD samples used by the preprocessing notebook. GEO access conditions may differ between processed and human raw sequence files. |
| <a id="membryo"></a>`Membryo` | Mouse embryo development | Mouse | [MOSTA portal](https://db.cngb.org/stomics/mosta/) · [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/) | Download the required embryo stages, including the E9.5–E15.5 stages used in the tutorial. |
| <a id="mcardiac"></a>`Mcardiac` | Mouse cardiac development | Mouse | [MOSTA portal](https://db.cngb.org/stomics/mosta/) · [MOSTA downloads](https://db.cngb.org/stomics/mosta/download/) | Download the heart-stage files required by the corrected `heart_9_11` workflow. |
| <a id="mbrain"></a>`Mbrain` | Mouse cerebellum atlas | Mouse | [CBMSTA portal](https://db.cngb.org/stomics/cbmsta/) · [CBMSTA downloads](https://db.cngb.org/stomics/cbmsta/download/) | The tutorial uses the Mouse1 T167–T171 series available from the download page. |

## Additional manuscript dataset

The human breast-cancer spatial dataset cited in the manuscript is available through [Zenodo DOI 10.5281/zenodo.4752030](https://doi.org/10.5281/zenodo.4752030). It is not currently assigned to one of the six public experiment directories above; consult the repository record and its license before reuse.

## Ligand–receptor tables

The public notebooks expect species-specific LR tables at:

```text
lrpairs/human/LR_pairs.csv
lrpairs/mouse/LR_pairs.csv
```

GP1, LUAD, and OpenST use the human table. Mbrain, Membryo, and Mcardiac use the mouse table.

## Recommended local layout

```text
experiments/<dataset>/
├── data/                         # downloaded files or read-only symlinks
├── config.yaml
├── preprocess.ipynb
├── decoder.ipynb
├── train.ipynb
└── artifacts/                    # generated checkpoints and results
```

Large source files and generated artifacts are intentionally excluded from Git. Verify checksums supplied by the upstream repository when available, and record the exact upstream file names used for reproducibility.
