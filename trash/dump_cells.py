import json

with open('notebooks/classification.ipynb') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']).strip()
    preview = src[:120].replace('\n', ' | ')
    print(f"Cell {i:3d} [{c['cell_type']:8s}]  {preview}")
