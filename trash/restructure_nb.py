"""
1. Crop SVM ROC from the combined image (only one missing)
2. Delete heatmap, macro comparison, and combined image
3. Restructure notebook: insert individual ROC under each classifier, remove combined section
"""
from PIL import Image
from pathlib import Path
import json

out_dir = Path("notebooks/figures")

# ── 1. Crop SVM ROC from combined image ──────────────────────────────────────
img = Image.open(out_dir / "roc_curves_all_classifiers.png")
w, h = img.size
col_w = w // 3
row_h = h // 2
# SVM is row 1, col 1 (0-indexed)
left = 1 * col_w
top = 1 * row_h
cropped = img.crop((left, top, left + col_w, top + row_h))
cropped.save(out_dir / "svm_rbf_roc.png")
print("Cropped SVM ROC from combined image.")

# ── 2. Delete old combined images ────────────────────────────────────────────
for old in ["roc_curves_all_classifiers.png", "roc_macro_avg_comparison.png", "auc_heatmap_classifier_floor.png"]:
    p = out_dir / old
    if p.exists():
        p.unlink()
        print(f"Deleted: {p}")

# ── 3. Restructure notebook ─────────────────────────────────────────────────
with open("notebooks/classification.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove the combined ROC section at the end (cell 42)
nb['cells'] = [c for i, c in enumerate(nb['cells']) if i != 42]

# Insert individual ROC image cells after each classifier's inference cell
# Original indices (before any inserts):
#   18 = LR inference,  21 = KNN inference,  24 = NB inference
#   27 = DT inference,  30 = SVM inference
inserts = [
    (18, "logistic_regression_roc.png", "Logistic Regression"),
    (21, "knn_roc.png",                 "KNN (k=5)"),
    (24, "naive_bayes_roc.png",         "Gaussian Naive Bayes"),
    (27, "decision_tree_roc.png",       "Decision Tree"),
    (30, "svm_rbf_roc.png",             "SVM (RBF)"),
]

offset = 0
for after_idx, img_file, model_name in inserts:
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"**ROC Curve (One-vs-Rest) - {model_name}:**\n",
            "\n",
            f"![{model_name} ROC](figures/{img_file})\n"
        ]
    }
    nb['cells'].insert(after_idx + 1 + offset, cell)
    offset += 1

with open("notebooks/classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nDone! Notebook restructured with individual ROC curves under each classifier.")
