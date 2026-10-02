# stVirtual

stVirtual is a niche-driven multi-agent generative framework for reconstructing spatiotemporal 3D and 4D tissue dynamics from sparse and static measurements across space, time, and disease progression.

![Mcardiac cardiac development from E9.5 to E11.5](docs/assets/results/mcardiac.gif)

The simulation above illustrates mouse cardiac development from E9.5 to E11.5. stVirtual reconstructs intermediate three-dimensional tissue states throughout development, capturing the dynamic changes in tissue organization and the spatial distribution of distinct cell populations.


## Installation

The code was tested on a workstation equipped with a 208-core Intel(R) Xeon(R) Platinum 8473C CPU, 512 GB of RAM, and an NVIDIA RTX PRO 6000 GPU with 96 GB of RAM, running Ubuntu 24.04.3 LTS and Python 3.12.11. If possible, stVirtual should be run with CUDA acceleration.

### Model runtime

The model runtime is distributed as a precompiled wheel for CPython 3.12 on Linux x86_64.

The runtime supports CPython 3.12 on Linux x86_64.

### Install with conda

Install [conda](https://docs.anaconda.com/anaconda/install/index.html), then create the stVirtual environment:

```bash
cd stVirtual
conda create -n stvirtual python=3.12.11 -y
conda activate stvirtual
pip install -r requirements.txt
source env/activate.sh
python -c "from stvirtual.models import stage1_3d, stage2_3d; print('stVirtual models ready')"
```

`requirements.txt` installs the bundled `stvirtual-core` wheel before installing
the public package in editable mode. The wheel checksum is listed in `wheels/SHA256SUMS`.

For CUDA, install a PyTorch build that matches your CUDA version, then install the remaining dependencies with `pip install -r requirements.txt`.

### Typical installation time:

- Existing CUDA/PyTorch-compatible environment: 5-15 minutes
- Fresh Linux workstation with package downloads: 20-60 minutes
- CPU desktop: 15-45 minutes

## Data availability

Download links, accessions, access conditions, and the expected local layout are listed in [Data Availability](docs/Data-Availability.md).

## Tutorials

Browse the [stVirtual Wiki](https://github.com/YjZhou16/stVirtual/wiki) for dataset-specific tutorials.

- [3D tissue reconstruction](https://github.com/YjZhou16/stVirtual/wiki/3D-tissue-reconstruction) — HMLN, Mbrain, and Human breast cancer.
- [Developmental reconstruction across time](https://github.com/YjZhou16/stVirtual/wiki/Developmental-reconstruction-across-time) — Membryo and Mcardiac_EA.
- [Cancer progression across stages](https://github.com/YjZhou16/stVirtual/wiki/Cancer-progression-across-stages) — Human gastric cancer and Human lung cancer.
- [4D tissue reconstruction](https://github.com/YjZhou16/stVirtual/wiki/4D-tissue-reconstruction) — Workflow template.
- [4D spatiotemporal development reconstruction](https://github.com/YjZhou16/stVirtual/wiki/4D-spatiotemporal-development-reconstruction) — Mcardiac.

Each tutorial documents input data, preprocessing, decoder training where applicable, Stage 1, boundary generation, Stage 2, simulation outputs, and the recommended execution order.

## Perturbation experiments

[Human gastric cancer](experiments/human_gastric_cancer/perturb/README.md), [Human lung cancer](experiments/human_lung_cancer/perturb/README.md), [Mcardiac](experiments/Mcardiac/perturb/README.md) and [Mcardiac_EA](experiments/Mcardiac_EA/perturb/README.md) perturbation workflows cover cell removal, gene knockdown, cell-state transition-specific LR expression changes and epicardial-cell ablation with free extrapolation. See the [perturbation tutorial](https://github.com/YjZhou16/stVirtual/wiki/Perturbation) for inputs and commands.
