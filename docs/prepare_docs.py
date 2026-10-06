"""Generate documentation copies without executing or changing notebooks."""
from pathlib import Path
import os
import re
import json

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
GENERATED = DOCS / '_generated'
REPO_URL = 'https://github.com/YjZhou16/stVirtual/blob/main/'

def rewrite(text, original, destination):
    def link(match):
        value = match.group(1)
        if value.startswith(('http:', 'https:', '#', 'mailto:')):
            return match.group(0)
        path, sep, anchor = value.partition('#')
        target = (original.parent / path).resolve()
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            return match.group(0)
        if relative.parts[0] == 'experiments':
            if target.name == 'README.md':
                mapped = GENERATED / relative.parent.relative_to('experiments') / 'index.md'
            elif target.suffix == '.ipynb':
                mapped = GENERATED / relative.relative_to('experiments')
            else:
                return '](' + REPO_URL + relative.as_posix() + (sep + anchor if sep else '') + ')'
        elif relative.parts[0] == 'docs':
            mapped = target
        else:
            return '](' + REPO_URL + relative.as_posix() + (sep + anchor if sep else '') + ')'
        new = os.path.relpath(mapped, destination.parent).replace(os.sep, '/')
        return '](' + new + (sep + anchor if sep else '') + ')'
    return re.sub(r'\]\(([^\s)]+)\)', link, text)

def prepare():
    GENERATED.mkdir(exist_ok=True)
    entries = []
    for experiment in sorted((ROOT / 'experiments').iterdir()):
        notebooks = sorted(experiment.glob('**/*.ipynb'))
        notebooks = [p for p in notebooks if not any(x in p.parts for x in ('artifacts', '.ipynb_checkpoints', 'data'))]
        if not notebooks:
            continue
        groups = {}
        for notebook in notebooks:
            output = GENERATED / notebook.relative_to(ROOT / 'experiments')
            output.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads(notebook.read_text())
            if not any(c.get('cell_type') == 'markdown' and re.search(r'^# ', ''.join(c.get('source', [])), re.M) for c in data['cells']):
                data['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'id': 'documentation-title', 'source': ['# ' + notebook.stem.replace('_', ' ').title()]})
            output.write_text(json.dumps(data, ensure_ascii=False))
            groups.setdefault(notebook.parent, []).append(notebook)
        for readme in experiment.glob('**/README.md'):
            if not any(x in readme.parts for x in ('artifacts', 'data', '.ipynb_checkpoints')):
                groups.setdefault(readme.parent, [])
        for folder, items in groups.items():
            destination = GENERATED / folder.relative_to(ROOT / 'experiments') / 'index.md'
            readme = folder / 'README.md'
            body = rewrite(readme.read_text(), readme, destination) if readme.exists() else '# ' + folder.name + '\n'
            children = [p.name for p in items]
            if folder == experiment:
                children += [str(p.relative_to(folder) / 'index') for p in groups if p != folder]
            body += '\n\n```{toctree}\n:maxdepth: 1\n:caption: Notebooks\n\n' + '\n'.join(children) + '\n```\n'
            body += '\n## Download notebooks\n\n' + '\n'.join('- {download}`' + p.stem + ' <' + p.name + '>`' for p in items) + '\n'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(body)
        entries.append(experiment.name + '/index')
    (GENERATED / 'notebooks.md').write_text('# Notebook examples\n\nBrowse the analysis steps, code, and saved visualizations for each experiment.\n\n```{toctree}\n:maxdepth: 2\n\n' + '\n'.join(entries) + '\n```\n')

if __name__ == '__main__':
    prepare()
