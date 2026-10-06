"""Sphinx configuration for the stVirtual tutorials."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'docs'))
from prepare_docs import prepare, rewrite
prepare()
project = 'stVirtual'
author = 'stVirtual contributors'
extensions = ['myst_nb']
html_theme = 'sphinx_rtd_theme'
html_title = 'stVirtual Wiki'
root_doc = 'wiki/00-quick-start'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {'navigation_depth': 3, 'collapse_navigation': False}
exclude_patterns = ['.work/**', '_build/**', '**/.ipynb_checkpoints/**', '**/*.orig', 'assets/**', 'requirements.txt']
nb_execution_mode = 'off'
myst_enable_extensions = ['colon_fence', 'dollarmath', 'amsmath']
myst_heading_anchors = 4

def process_links(app, docname, source):
    if docname.startswith('wiki/'):
        original = ROOT / 'docs' / (docname + '.md')
        source[0] = rewrite(source[0], original, original)
    if docname == root_doc:
        pages = sorted('/wiki/' + p.stem for p in (ROOT / 'docs/wiki').glob('*.md') if p.stem != '00-quick-start')
        source[0] += '\n\n```{toctree}\n:hidden:\n\n' + '\n'.join(pages + ['/_generated/notebooks']) + '\n```\n'

def setup(app):
    app.connect('source-read', process_links)
