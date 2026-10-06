# stVirtual tutorial collection

Eight data workflows and four perturbation experiments using the current stVirtual public interfaces.

## Run notebooks

Install stVirtual and its core wheel in a Linux Python 3.12 scientific environment. Launch Jupyter from this folder and follow each dataset index. Input and output locations are configured per experiment. See docs/getting_started.md.

## Build and preview documentation

```bash
python -m venv .venv
.venv/bin/python -m pip install -r docs/requirements.txt
.venv/bin/python docs/build_bundle.py
.venv/bin/sphinx-build -a -W --keep-going -b html . _build/html
.venv/bin/python -m http.server 8766 --directory _build/html --bind 127.0.0.1
```

The HTML preview is http://127.0.0.1:8766/. For a remote server, forward local port 8766 to the server's 127.0.0.1:8766. Documentation builds render saved notebook outputs without executing scientific workflows.
