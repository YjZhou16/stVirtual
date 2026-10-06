# Quick start

[Notebook tutorial](https://stVirtual-tutorial.readthedocs.io/en/latest/docs/quickstart.html).

**Recommended memory for the Mbrain example: at least 64 GB RAM and 24 GB GPU memory.**

## 1. Install

```bash
git clone https://github.com/YjZhou16/stVirtual.git
cd stVirtual
conda create -n stvirtual python=3.12.11 -y
conda activate stvirtual
pip install -r requirements.txt
source env/activate.sh
```

[Installation details](../../README.md#installation).

<a id="mbrain-example"></a>

## 2. Input data

Download the four H5AD files from [CBMSTA Stereo-seq data](https://db.cngb.org/stomics/cbmsta/download/) and place them here:

```text
experiments/Mbrain/data/
    Mouse1_T168.h5ad
    Mouse1_T169.h5ad
    Mouse1_T170.h5ad
    Mouse1_T171.h5ad
```

## 3. Edit config

Edit [experiments/Mbrain/config.yaml](../../experiments/Mbrain/config.yaml):

```yaml
data_root: data
route_ids: [T168, T170]
```

## 4. Run notebooks

Open the notebooks locally, select the **stvirtual** environment as the kernel, and run them in order:

1. [preprocess.ipynb](../../experiments/Mbrain/preprocess.ipynb)
2. [decoder.ipynb](../../experiments/Mbrain/decoder.ipynb)
3. [train.ipynb](../../experiments/Mbrain/train.ipynb)

## 5. Results

Simulation files: `experiments/Mbrain/artifacts/results/simulation/T168_to_T170/`.

Run the last cell of `train.ipynb`, then open `experiments/Mbrain/artifacts/results/overview/T168_to_T170.png`.

Other datasets: [3D tissue reconstruction](01-3d-tissue-reconstruction.md), [development across time](02-developmental-reconstruction-across-time.md), [cancer progression](03-cancer-progression-across-stages.md), and [4D development](04-4d-spatiotemporal-development-reconstruction.md).
