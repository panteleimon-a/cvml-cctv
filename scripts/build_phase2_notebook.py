"""Regenerate notebooks/CCT20_PHASE2_EXPERIMENTS.ipynb from scripts/cct20_phase2_experiments.py.

The .py is the source of truth. `# %%` starts a code cell and `# %% [markdown]` a markdown cell.
Run: python scripts/build_phase2_notebook.py   (needs nbformat)
"""
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts' / 'cct20_phase2_experiments.py'
DST = ROOT / 'notebooks' / 'CCT20_PHASE2_EXPERIMENTS.ipynb'


def build():
    parts = re.split(r'^# %%(.*)$', SRC.read_text(), flags=re.M)
    cells = []
    for i in range(1, len(parts), 2):
        tag, body = parts[i].strip(), parts[i + 1].strip('\n')
        if tag == '[markdown]':
            text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    for k, c in enumerate(cells):
        c['id'] = f'cell-{k:02d}'  # stable ids: rebuilding does not churn the diff
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {'accelerator': 'GPU',
                   'colab': {'name': DST.name, 'provenance': [], 'gpuType': 'T4'},
                   'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}}
    nbformat.validate(nb)
    data = json.loads(nbformat.writes(nb))
    for c in data['cells']:  # Jupyter-style line lists keep git diffs readable
        c['source'] = ''.join(c['source']).splitlines(keepends=True)
    DST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    return len(cells)


if __name__ == '__main__':
    print(f'{build()} cells -> {DST}')
