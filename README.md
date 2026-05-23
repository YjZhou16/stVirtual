# stVirtual

<img width="1280" height="720" alt="幻灯片1" src="https://github.com/user-attachments/assets/97230574-6e20-4eea-9da8-84917e4962bb" />

stVirtual is a computational framework for virtual tissue generation from spatial transcriptomics data. By integrating spatial organization, gene expression, and local niche signals, it supports cross-slice simulation, intermediate tissue reconstruction, and 3D virtual representation of dynamic tissue evolution.

## Highlight

- Cross-slice spatial transcriptomics simulation
- Virtual intermediate tissue generation
- Integration of spatial coordinates, gene expression, and niche signals
- Support for 3D/4D tissue representation
- Flexible modeling framework for developmental and disease studies
- In silico perturbation analysis

## System requirements

### Operating system

The code is intended for Linux. It has been used on:

- Ubuntu 24.04.1 LTS, Linux kernel 6.17.0, x86_64
- NVIDIA driver 580.126.09
- CUDA 12.9 runtime packages through PyTorch wheels

macOS and Windows have not been tested. Windows users should use WSL2 with an NVIDIA CUDA-enabled GPU if running the full training workflow.

### Python and software dependencies

Use Python 3.10 or later. The development environment was exported with Python 3.13.2. Major runtime dependencies are pinned in `requirements.txt`; key versions include:

- `torch==2.8.0+cu129`
- `torchvision==0.23.0+cu129`
- `torchdiffeq==0.2.5`
- `scanpy==1.11.4`
- `scvi-tools==1.4.0`
- `anndata==0.12.2`
- `numpy==2.1.2`
- `pandas==2.3.2`
- `scipy==1.16.2`
- `scikit-learn==1.7.2`
- `matplotlib==3.10.6`
- `POT==0.9.6.post1`
- `geomloss==0.2.6`
- `pykeops==2.3`
- `squidpy==1.7.0`
- `spatialdata==0.6.1`

See `requirements.txt` for the full dependency list and exact versions.

Note: the current `requirements.txt` was exported from a conda-based environment and contains several local `@ file://...` build paths. If `pip` cannot resolve those entries on a new machine, install the named package from conda-forge or PyPI with the same version, or regenerate a clean requirements file from your environment.

KeOps is required for efficient large-scale kernel and OT computations. Please refer to the official KeOps documentation for installation instructions: https://www.kernel-operations.io/keops/index.html

### Hardware

For full model training, a CUDA-capable NVIDIA GPU is strongly recommended and is effectively required for practical runtime on the provided full-size data. The workflow has been tested on NVIDIA RTX PRO 6000 Blackwell GPUs with 96 GB VRAM.

Recommended hardware:

- CPU: 8 or more cores
- RAM: 32 GB minimum; 64 GB or more recommended for full mouse brain / IMC data
- GPU: NVIDIA CUDA GPU with at least 16 GB VRAM for small examples; 40 GB or more recommended for full training
- Disk: at least 10 GB for the included processed data and outputs; more if running all case studies

No non-standard hardware is needed for reading data, preprocessing, and plotting small subsets, but full training requires a CUDA GPU for feasible runtime.

## Installation guide

Clone or unpack the repository, then install the dependencies in a fresh environment.

```bash
cd /home/zhouyj/stVirtual

conda create -n stvirtual python=3.13 -y
conda activate stvirtual

pip install -r requirements.txt
```

If CUDA-specific PyTorch packages fail to install through the default index, install PyTorch and PyG packages using the wheel indexes appropriate for your CUDA version, then rerun `pip install -r requirements.txt` for the remaining packages.

Before running scripts or notebooks, expose the local source directory:

```bash
export PYTHONPATH=/home/zhouyj/stVirtual/src:$PYTHONPATH
```

Typical installation time:

- Existing CUDA/PyTorch-compatible environment: 5-15 minutes
- Fresh Linux workstation with package downloads: 20-60 minutes
- CPU-only desktop: 15-45 minutes, but full training is not recommended

## Repository layout

```text
stVirtual/
├── data/
│   ├── Mbrain/                 # mouse brain example data
│   ├── imc/                    # IMC example data
│   └── lrpairs/                # ligand-receptor pair tables
├── src/
│   ├── stVirtual_model/        # core ODE, UOT, decoder, and RL models
│   ├── utils/                  # rollout, saving, LR, and analysis utilities
│   ├── Mbrain/                 # mouse brain notebooks and plotting code
│   └── imc/                    # IMC notebooks and plotting code
├── other/                      # additional case studies
├── requirements.txt
└── README.md
```

## Source Data

Mbrain: https://db.cngb.org/stomics/cbmsta      Mouse1 T167-T171 
   
IMC: https://zenodo.org/records/4752030


## Demo

The quickest complete demo is the mouse brain route from slice `T170` to `T171` in `src/Mbrain/train_mouse.ipynb`.

Input data:

- `data/Mbrain/processed/Mbrain.h5ad`
- `data/lrpairs/mouse/LR_pairs.csv`

The notebook expects the processed AnnData object to contain:

- `obs["sample"]` with slice IDs such as `T170` and `T171`
- aligned coordinates in `obs["cx_aligned"]` and `obs["cy_aligned"]`
- annotations in `obs["annotation"]`
- latent representation in `obsm["X_scanVI"]`
- counts in `layers["counts"]`

Run the demo:

```bash
cd /home/zhouyj/stVirtual
conda activate stvirtual
export PYTHONPATH=/home/zhouyj/stVirtual/src:$PYTHONPATH
jupyter notebook src/Mbrain/train_mouse.ipynb
```

In the notebook, the demo configuration is:

```python
data_path = "/home/zhouyj/stVirtual/data/Mbrain/processed"
ckpt_path = "/home/zhouyj/stVirtual/src/Mbrain/Mbrain_ckpt"
lrpr_path = "/home/zhouyj/stVirtual/data/lrpairs/mouse/LR_pairs.csv"
route_ids = ["T170", "T171"]
steps = 10
```

The stage-1 ODE/UOT demo call is:

```python
res1 = s1.train_model_multislice(
    model=model,
    adata_all=adata,
    slice_key="sample",
    route_ids=route_ids,
    save_root=f"{ckpt_path}/res1_all",
    x_key="cx_aligned",
    y_key="cy_aligned",
    latent_layer="X_scanVI",
    steps=steps,
    cell_type_key="annotation",
    lib_layer="counts",
    latent_dim=10,
    epochs=100,
    device="cuda:0",
)
```

Save the stage-1 rollout:

```python
saved = sv1.save_res(
    res1=res1,
    adata_all=adata,
    out_dir=f"{ckpt_path}/stage1_res",
    slice_key="sample",
    ann_key="annotation",
    save_prefix="rollout_stage1",
    steps=10,
    n_cache=256,
    unnormalize=False,
)
```

Expected output:

- stage-1 checkpoints under `src/Mbrain/Mbrain_ckpt/res1_all/T170_to_T171/checkpoints/`
- a compressed rollout file such as `src/Mbrain/Mbrain_ckpt/stage1_res/rollout_stage1_T170_to_T171.npz`
- optional visualization figures generated by `plot_mouse.py`
- stage-2 policy checkpoints under `src/Mbrain/Mbrain_ckpt/policy_ckpt_mouse/` when the RL cells are run
- optional virtual-slice `.h5ad` files when using utilities in `utils/save_rl.py`

Expected demo runtime:

- For Stage-1 mouse brain `T170_to_T171`, the OT computation is relatively time-consuming due to context graph construction, taking about 10 minutes on a modern high-memory NVIDIA GPU. In contrast, the subsequent ODE training takes only about 2 minutes.
- Full stage-1 + stage-2 workflow: tens of minutes to several hours depending on GPU, data size, route length, and epoch settings.
- CPU-only runtime is expected to be much longer and is not recommended for the full demo.

## Instructions for use on your own data

Prepare a single `.h5ad` file containing all slices. At minimum, the object should include:

- `obs[slice_key]`: slice or time-point identifier, for example `sample`
- `obs[x_key]`, `obs[y_key]`: aligned spatial coordinates
- `obs[cell_type_key]`: cell type, region, layer, or annotation labels
- `obsm[latent_layer]`: low-dimensional latent features, for example `X_scanVI`
- `layers[lib_layer]` or `X`: count/expression matrix used by decoder and ligand-receptor utilities

Train stage 1 on a route of observed slices:

```python
import scanpy as sc
import stVirtual_model.ode_multislice as s1
import utils.save_uot as sv1

adata = sc.read_h5ad("path/to/your_data.h5ad")
route_ids = ["slice_a", "slice_b", "slice_c"]

res1 = s1.train_model_multislice(
    model=model,
    adata_all=adata,
    slice_key="sample",
    route_ids=route_ids,
    save_root="outputs/stage1_ckpt",
    x_key="cx_aligned",
    y_key="cy_aligned",
    latent_layer="X_scanVI",
    steps=10,
    cell_type_key="annotation",
    lib_layer="counts",
    latent_dim=adata.obsm["X_scanVI"].shape[1],
    epochs=100,
    device="cuda:0",
)

saved = sv1.save_res(
    res1=res1,
    adata_all=adata,
    out_dir="outputs/stage1_res",
    slice_key="sample",
    ann_key="annotation",
    save_prefix="rollout_stage1",
)
```

Train stage 2 using the saved stage-1 rollouts:

```python
import torch
import stVirtual_model.rl_multislice as s2

ctx = s2.build_global_ctx(
    adata_path="path/to/your_data.h5ad",
    lr_pairs_path="data/lrpairs/mouse/LR_pairs.csv",
    ckpt_3dslice="outputs/stage1_ckpt/slice_a_to_slice_b/checkpoints/best.pt",
    device=torch.device("cuda:0"),
    layer_col="annotation",
)

stages = [
    s2.StageCfg(
        src="slice_a",
        tgt="slice_b",
        out_npz_path=saved["slice_a_to_slice_b"],
        bound_dir="path/to/boundary_files",
        stat_json=None,
        ckpt_path_dec=None,
        scanvi_dir=None,
        layer_col="annotation",
        use_latent=True,
        use_lr=False,
        lr_source="none",
    )
]

outs = s2.run_multi_stages(
    ctx=ctx,
    W_XY_TGT=1.0,
    W_Z_TGT=1.0,
    sample_key="sample",
    stages=stages,
    best_ckpt_dir="outputs/stage2_policy_ckpt",
)
```
- Stage-2 mouse brain `T170_to_T171`, 100 epochs: about 16 minutes on a modern high-memory NVIDIA GPU. 
- Boundary files for stage 2 can be produced with the dataset-specific `bound.py` helpers, for example `src/Mbrain/bound.py` or `src/imc/bound.py`.

## Reproduction instructions

To reproduce the quantitative and visual results from the included examples, run the notebooks in order for the dataset of interest.

Mouse brain:

1. `src/Mbrain/preprocess.ipynb`
2. `src/Mbrain/align_lr_compute.ipynb`
3. `src/Mbrain/train_mouse.ipynb`
4. `src/Mbrain/exp_decoder.ipynb`

IMC:

1. `src/imc/preprocess.ipynb`
2. `src/imc/train_imc.ipynb`


For reproducibility, keep the route IDs, random seeds, epoch counts, latent keys, coordinate keys, and ligand-receptor tables unchanged from the notebooks. GPU nondeterminism in PyTorch/CUDA may cause small numerical differences between runs.

## Common outputs

The workflow produces several output types:

- `.pt` PyTorch checkpoints for ODE and RL models
- `.npz` rollout files containing virtual coordinates, latent states, masses, time points, annotations, and normalization statistics
- `.h5ad` virtual slice files generated from rollouts
- `.png` visualization figures from dataset-specific plotting utilities

## Troubleshooting

- If imports fail, confirm that `PYTHONPATH` contains `.../src`.
- If CUDA is unavailable, verify the NVIDIA driver, CUDA-compatible PyTorch wheel, and `torch.cuda.is_available()`.
- If `pip install -r requirements.txt` fails on `@ file://` entries, install the same package name and version from conda-forge or PyPI.
- If a notebook references an absolute path, update it to match your local checkout and data location.
