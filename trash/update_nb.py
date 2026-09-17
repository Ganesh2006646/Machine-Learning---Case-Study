import json

nb_path = 'notebooks/classification.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown cell content to append
md_content = [
    "---\n",
    "## 5. ROC and AUC Curve Visualizations\n",
    "\n",
    "The following visualizations detail the One-vs-Rest ROC/AUC curves for all 5 floor classifiers evaluated in this notebook.\n",
    "\n",
    "### 5.1 Individual ROC Curves\n",
    "![ROC Curves](figures/roc_curves_all_classifiers.png)\n",
    "\n",
    "### 5.2 Macro-Average ROC Comparison\n",
    "![Macro-Average ROC Comparison](figures/roc_macro_avg_comparison.png)\n",
    "\n",
    "### 5.3 AUC Score Heatmap\n",
    "![AUC Score Heatmap](figures/auc_heatmap_classifier_floor.png)\n"
]

new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": md_content
}

nb['cells'].append(new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Appended images to notebook!")
