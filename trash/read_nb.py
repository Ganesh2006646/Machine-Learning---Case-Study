import json

with open('notebooks/classification.ipynb') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if src.strip():
        print(f"\n=== Cell {i} ({c['cell_type']}) ===")
        print(src[:3000])
