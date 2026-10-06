# Getting started

## Download stVirtual and set up the environment

Use Linux x86_64 with Python 3.12. From a terminal:

```bash
git clone https://github.com/YjZhou16/stVirtual.git
cd stVirtual
conda create -n stvirtual python=3.12.11 -y
conda activate stvirtual
pip install -r requirements.txt
source env/activate.sh
```

`requirements.txt` installs the model runtime, stVirtual, and the notebook dependencies. Select this environment as the notebook kernel.

Download the {download}`tutorial bundle <../.work/package/stVirtual_tutorial.zip>` and extract it into a separate folder. Keep its notebooks, configs, shared helpers, and LR tables together.

## Tutorial structure

- **Data workflow:** eight dataset examples, each with preprocessing, decoder training when needed, model training, and results. [Use your own data](your-data.md) follows the same workflow.
- **Perturbation experiments:** four examples showing how to change cells or gene expression and run a new simulation with trained models.

[Data Availability](data.md) lists the input files and download sources. [Data preprocessing guide](preprocessing.md) shows the shared preparation steps.

[Quick start](quickstart.md) takes you through the Mbrain example from input files to simulation results.
