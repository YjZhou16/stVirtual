# Quick start

Use Mbrain T168 to T170 as the first example. Complete [environment setup](getting_started.md#download-stvirtual-and-set-up-the-environment) and extract the tutorial bundle. The paths below are relative to the extracted tutorial folder.

Recommended memory for this example: **at least 64 GB RAM and 24 GB GPU memory**. See the [memory measurements](runtime-and-resources.md#mbrain-memory-check).

## 1. Prepare data

Download the four annotated H5AD files from [CBMSTA Stereo-seq data](https://db.cngb.org/stomics/cbmsta/download/) and place them here:

```text
experiments/Mbrain/data/
    Mouse1_T168.h5ad
    Mouse1_T169.h5ad
    Mouse1_T170.h5ad
    Mouse1_T171.h5ad
```

Preprocessing uses all four sections; model training uses T168 and T170.

## 2. Edit config

Set the input folder and sample pair in {download}`experiments/Mbrain/config.yaml <../experiments/Mbrain/config.yaml>`:

```yaml
data_root: data
route_ids: [T168, T170]
```

## 3. Run notebooks

Open these notebooks locally with the **stvirtual** kernel, in this order:

1. [preprocess.ipynb](../experiments/Mbrain/preprocess.ipynb): prepare expression features and aligned coordinates.
2. [decoder.ipynb](../experiments/Mbrain/decoder.ipynb): train the expression decoder.
3. [train.ipynb](../experiments/Mbrain/train.ipynb): train the model and generate simulation frames.

## 4. View results

Open [results.ipynb](../experiments/Mbrain/results.ipynb) to view the real sections and simulated frames together.

Generated files are saved under:

```text
experiments/Mbrain/artifacts/
    checkpoints/                         # trained models
    results/simulation/T168_to_T170/      # simulation frames
    results/overview/T168_to_T170.png     # frame overview
```

Continue with the [Mbrain tutorial](../experiments/Mbrain/index.md), another dataset under **Data workflow**, or [Use your own data](your-data.md).
