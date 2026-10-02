# Cancer progression across stages

This tutorial covers cancer progression across stages for **Human gastric cancer** and **Human lung cancer**. Both workflows model spatial transport and cell-state transitions between the source and target tissues.

```bash
conda activate stvirtual
source env/activate.sh
```

## Results

### Human gastric cancer — normal to cancer

![Human gastric cancer transition simulation](../assets/results/human_gastric_cancer.gif)

### Human lung cancer — AAH to LUAD

![Human lung cancer transition simulation](../assets/results/human_lung_cancer.gif)

## 1. Input data

Use `experiments/human_gastric_cancer` or `experiments/human_lung_cancer`. Both workflows require spatial counts, stage labels, cell annotations, a human LR table, and `data/diff_map.csv`. Human gastric cancer uses `annotation`; Human lung cancer uses `_clone_s` as its configured source label.

Define the differentiation map with these columns:

```text
src_layer,tgt_layer,weight
7,6,1.0
6,1,1.0
```

Multi-hop paths are executed sequentially; a committed intermediate cell may later enter another permitted transition.

## 2. Preprocessing

Run `preprocess.ipynb`. Human gastric cancer attaches metadata before training scanVI. Human lung cancer additionally aligns AAH coordinates to the LUAD reference using feature-guided rigid UOT and writes `cx_aligned`, `cy_aligned`, and `obsm["spatial_aligned"]`.

The canonical output is:

```text
artifacts/checkpoints/scanvi/adata.h5ad
```

The prepared input stores `layers["counts"]` and `obsm["X_scanVI"]`.

## 3. Expression decoder

Run `decoder.ipynb`. Route-specific checkpoints are stored as:

```text
artifacts/checkpoints/decoder/checkpoints/<src>_<tgt>.pt
```

The decoder maps latent vectors to expression. Simulation time is recorded as frame metadata.

## 4. Train the model

Open `train.ipynb`.

### 4.1 Stage 1

`stvirtual.models.stage1_3d` learns the source-to-target transport. Checkpoints and Stage-1 traces stay under `artifacts/checkpoints/stage1/`.

### 4.2 Boundary files

Per-frame boundaries are generated under `artifacts/results/bound/`. Rebuild them whenever Stage 1, alignment, or boundary settings change.

### 4.3 Transition-aware Stage 2

`stvirtual.models.stage2_3d_transition` loads `diff_map.csv`, trains the RL policy, and writes `artifacts/checkpoints/stage2/policy_<src>_to_<tgt>.pt`. Confirm the log contains `[diff] loaded`.

## 5. Transition output

In addition to the standard contract, frames contain `uid`, `parent_uid`, `diff_alpha`, `src_layer`, and `tgt_layer`. `celltype` is the authoritative committed state. Use `uid` to follow each cell across frames and `parent_uid` to reconstruct birth ancestry. Source and target layers describe each transition along a multi-hop transition.

## 6. Reproducibility

Keep decoder checkpoint, Stage-2 epochs, boundary seed, simulation seed, and differentiation map fixed when comparing runs. For Human gastric cancer, 6/9 appear before 1, as required by the configured graph.

## Output contract

Each simulation route is written to `artifacts/results/simulation/<src>_to_<tgt>/`. Every frame is an AnnData file with coordinates in `obsm["spatial"]`, latent states in `obsm["X_latent"]`, and cell-type annotations in `obs["celltype"]` and `obs["celltype_id"]`. The adjacent `celltype_mapping.csv` maps IDs to names.

## Recommended order for Human gastric cancer and Human lung cancer

1. Review and edit `config.yaml`.
2. Run `preprocess.ipynb`.
3. Run `decoder.ipynb`.
4. Run `train.ipynb` from top to bottom.
5. Inspect boundary coverage and the first and last simulation frames before downstream analysis.

Input data are placed under `data/`, and generated checkpoints and results are written to `artifacts/`.

