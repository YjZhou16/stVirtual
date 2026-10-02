# Installation

Use Linux x86_64 with Python 3.12. Clone the repository, then create the environment from the repository root:

```bash
git clone https://github.com/YjZhou16/stVirtual.git
cd stVirtual
conda create -n stvirtual python=3.12.11 -y
conda activate stvirtual
pip install -r requirements.txt
source env/activate.sh
```

The model runtime is distributed as a precompiled wheel for CPython 3.12 on Linux x86_64.

## CUDA environment

For GPU workflows, install a CUDA development toolkit compatible with your PyTorch environment. The toolkit supplies the compiler, runtime compilation library, and CUDA headers used by KeOps.

After activating your environment, run `source env/activate.sh` from the repository root. This configures the Python source path and detects CUDA headers, including the `targets/x86_64-linux/include` layout used by Conda toolkits. Start Jupyter from that shell so its kernels inherit the configuration.

## Build the tutorials locally

```bash
python -m venv docs/.work/docs-venv
docs/.work/docs-venv/bin/python -m pip install -r docs/requirements.txt
docs/.work/docs-venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Open `http://127.0.0.1:8000` on the machine running the preview server. The documentation build renders the saved notebook outputs.
