# Hardware and training time

## Choosing a machine

Use Linux x86_64, CPython 3.12, and an NVIDIA GPU with a compatible PyTorch/CUDA environment. For the Mbrain example, we recommend **at least 64 GB system RAM and 24 GB GPU memory**. Memory use depends on cell numbers, genes, neighborhoods, and model settings.

The completed runs below used a workstation with **512 GB RAM**, a **208-core Intel Xeon Platinum 8473C CPU**, and an **NVIDIA RTX PRO 6000 with 96 GB VRAM**, running Ubuntu 24.04.3 and Python 3.12.11. These are tested machine specifications, not minimum requirements.

Use an otherwise idle GPU for Mcardiac Stage 2. Its peak VRAM and minimum GPU memory requirement have not been measured.

### Mbrain memory check

Mbrain T168 to T170 completed 5 epochs for each training stage. The measured peaks were:

| Stage | RAM (GiB) | GPU memory (GiB) |
| --- | ---: | ---: |
| Stage 1 | 42.17 | 7.24 |
| Stage 2 | 11.31 | 23.63 |

This check used the 96 GB GPU with PyTorch allocation limited to 23 GiB. It does not verify a physical 24 GB GPU, preprocessing, decoder training, simulation, or a full 100-epoch run.

## Measured training time

Each row covers **one window, 100 Stage-1 epochs and 100 Stage-2 epochs**, run sequentially on GPU 1 of that workstation. Times are wall-clock durations of the training call/cell, including its local setup and checkpoint operations. They exclude preprocessing, decoder training, boundary generation and simulation. They are reference measurements, not projected runtimes on other GPUs.

| Dataset | Window | Stage 1 (minutes) | Stage 2 (minutes) |
| --- | --- | ---: | ---: |
| Mbrain | T168 to T170 | 6.8 | 18.5 |
| HMLN | S1 to S3 | 16.0 | 38.9 |
| Human breast cancer | 0 to 2 | 0.9 | 2.3 |
| Membryo | E9.5 to E11.5 | 5.0 | 15.9 |
| Mcardiac_EA | E10.5_CD1 to E12.5_CD1 | 0.9 | 4.2 |
| Human lung cancer | AAH to LUAD | 0.6 | 16.5 |
| Human gastric cancer | normal to cancer | 0.6 | 9.8 |
| Mcardiac | E9.5h to E11.5h | 33.8 | 210.3 |

These measurements were collected in October 2026. Mbrain training took about **25 minutes combined**; Mcardiac took about **4.1 hours combined**. Allow additional time for preprocessing and decoding. Installation estimates in the README refer only to environment setup.
