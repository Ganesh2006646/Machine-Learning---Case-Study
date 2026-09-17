"""
ROC / AUC Curve Generator — Classification Notebook (Floor Prediction)
Generates one-vs-rest ROC curves (per class + micro / macro average) for
all 5 classifiers used in the classification notebook.

Run from the project root:
    python trash/generate_roc_auc.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path("notebooks/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (one per floor class) ─────────────────────────────────────
CLASS_COLORS = ['#6C63FF', '#FF6584', '#43B89C', '#F5A623', '#2196F3']
FLOOR_LABELS = ['Floor 0', 'Floor 1', 'Floor 2', 'Floor 3', 'Floor 4']
CLASSES = [0, 1, 2, 3, 4]
N_CLASSES = len(CLASSES)

# ── 1. Load & preprocess data (mirror classification.ipynb exactly) ───────────
print("Loading data …")
df = pd.read_csv("data/raw/trainingData.csv")

WAP_COLS = [c for c in df.columns if c.startswith("WAP")]

# Replace +100 (undetected) sentinel with -110 dBm
df[WAP_COLS] = df[WAP_COLS].replace(100, -110)

# Feature: variance of WAP values per row (signal diversity)
df["WAP_VAR"] = df[WAP_COLS].var(axis=1)

# Drop rows where variance == 0 (all WAPs undetected)
df = df[df["WAP_VAR"] > 0].reset_index(drop=True)

# Remove WAPs that are almost always undetected (>99 % rows at -110)
threshold = 0.99 * len(df)
always_off = [c for c in WAP_COLS if (df[c] == -110).sum() > threshold]
df.drop(columns=always_off, inplace=True)
WAP_COLS = [c for c in df.columns if c.startswith("WAP")]

X = df[WAP_COLS].values
y = df["FLOOR"].values

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_sc = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
X_test_sc  = np.nan_to_num(scaler.transform(X_test),  nan=0.0)

# Binarise labels for OvR ROC
y_test_bin = label_binarize(y_test, classes=CLASSES)

print(f"Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
print(f"Floor classes: {CLASSES}\n")

# ── 2. Classifiers ────────────────────────────────────────────────────────────
classifiers = [
    ("Logistic Regression",
     OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42)),
     X_train_sc, X_test_sc),

    ("KNN (k=5)",
     OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)),
     X_train_sc, X_test_sc),

    ("Gaussian Naive Bayes",
     OneVsRestClassifier(GaussianNB()),
     X_train_sc, X_test_sc),

    ("Decision Tree",
     OneVsRestClassifier(DecisionTreeClassifier(max_depth=15, random_state=42)),
     X_train_sc, X_test_sc),

    ("SVM (RBF)",
     OneVsRestClassifier(SVC(kernel='rbf', C=10, random_state=42, probability=True)),
     X_train_sc, X_test_sc),
]

# ── 3. Compute ROC curves for every classifier ────────────────────────────────

def compute_roc(model, X_tr, X_te, y_te_bin):
    """Fit model and return per-class + micro + macro ROC data."""
    model.fit(X_tr, y_train)                    # y_train is global

    # predict_proba on the OvR wrapper gives (n_samples, n_classes)
    y_score = model.predict_proba(X_te)

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class curves
    for i, cls in enumerate(CLASSES):
        fpr[cls], tpr[cls], _ = roc_curve(y_te_bin[:, i], y_score[:, i])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    # Micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(
        y_te_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Macro-average (interpolated)
    all_fpr = np.unique(np.concatenate([fpr[c] for c in CLASSES]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in CLASSES:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= N_CLASSES
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return fpr, tpr, roc_auc


print("Training classifiers and computing ROC curves …")
results = []
for name, model, X_tr, X_te in classifiers:
    print(f"  {name} …", flush=True)
    fpr, tpr, roc_auc = compute_roc(model, X_tr, X_te, y_test_bin)
    results.append((name, fpr, tpr, roc_auc))
print("Done.\n")

# ── 4. Plot A — Individual ROC panels (one per classifier) ────────────────────

def plot_individual_roc(name, fpr, tpr, roc_auc, ax):
    """Draw per-class + micro + macro ROC curves on ax."""
    for i, cls in enumerate(CLASSES):
        ax.plot(fpr[cls], tpr[cls],
                color=CLASS_COLORS[i], lw=1.8,
                label=f"{FLOOR_LABELS[i]}  (AUC = {roc_auc[cls]:.3f})")

    ax.plot(fpr['micro'], tpr['micro'],
            color='#E74C3C', lw=2.5, linestyle='--',
            label=f"Micro-avg  (AUC = {roc_auc['micro']:.3f})")
    ax.plot(fpr['macro'], tpr['macro'],
            color='#2C3E50', lw=2.5, linestyle=':',
            label=f"Macro-avg  (AUC = {roc_auc['macro']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance (AUC=0.5)')
    ax.fill_between(fpr['macro'], tpr['macro'], alpha=0.07, color='#2C3E50')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.85)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(2, 3, figsize=(21, 13))
fig.patch.set_facecolor('#F8F9FB')

axes_flat = axes.flatten()
for idx, (name, fpr, tpr, roc_auc) in enumerate(results):
    axes_flat[idx].set_facecolor('#FDFDFE')
    plot_individual_roc(name, fpr, tpr, roc_auc, axes_flat[idx])

axes_flat[5].set_visible(False)   # 5th slot hidden (we have 5 classifiers)

fig.suptitle(
    "ROC Curves (One-vs-Rest) — Floor Classification (All 5 Classifiers)",
    fontsize=16, fontweight='bold', y=1.01
)
plt.tight_layout(rect=[0, 0, 1, 1])
out_path_a = OUT_DIR / "roc_curves_all_classifiers.png"
fig.savefig(out_path_a, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out_path_a}")

# ── 5. Plot B — Macro-average ROC comparison (all classifiers on one axes) ────

fig2, ax2 = plt.subplots(figsize=(9, 7))
fig2.patch.set_facecolor('#F8F9FB')
ax2.set_facecolor('#FDFDFE')

cmap = ['#6C63FF', '#FF6584', '#43B89C', '#F5A623', '#2196F3']
for idx, (name, fpr, tpr, roc_auc) in enumerate(results):
    ax2.plot(fpr['macro'], tpr['macro'],
             color=cmap[idx], lw=2.4,
             label=f"{name}  (Macro AUC = {roc_auc['macro']:.3f})")
    ax2.fill_between(fpr['macro'], tpr['macro'], alpha=0.06, color=cmap[idx])

ax2.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Chance (AUC = 0.5)')
ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([-0.01, 1.05])
ax2.set_xlabel("False Positive Rate", fontsize=12)
ax2.set_ylabel("True Positive Rate", fontsize=12)
ax2.set_title("Macro-Average ROC Comparison — All Classifiers",
              fontsize=14, fontweight='bold', pad=10)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.35)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

out_path_b = OUT_DIR / "roc_macro_avg_comparison.png"
fig2.savefig(out_path_b, dpi=180, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"Saved: {out_path_b}")

# ── 6. Plot C — AUC heatmap (model × floor) ───────────────────────────────────

auc_matrix = np.array([
    [roc_auc[c] for c in CLASSES] for _, _, _, roc_auc in results
])
model_names = [name for name, *_ in results]

fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.patch.set_facecolor('#F8F9FB')

im = ax3.imshow(auc_matrix, cmap='RdYlGn', vmin=0.85, vmax=1.0, aspect='auto')
plt.colorbar(im, ax=ax3, label='AUC Score', shrink=0.8)

ax3.set_xticks(range(N_CLASSES))
ax3.set_xticklabels(FLOOR_LABELS, fontsize=10)
ax3.set_yticks(range(len(model_names)))
ax3.set_yticklabels(model_names, fontsize=10)
ax3.set_title("AUC Score Heatmap — Classifier × Floor (One-vs-Rest)",
              fontsize=13, fontweight='bold', pad=10)

for i in range(len(model_names)):
    for j in range(N_CLASSES):
        val = auc_matrix[i, j]
        color = 'white' if val < 0.94 else '#1A1A2E'
        ax3.text(j, i, f"{val:.3f}", ha='center', va='center',
                 fontsize=9, fontweight='bold', color=color)

out_path_c = OUT_DIR / "auc_heatmap_classifier_floor.png"
fig3.savefig(out_path_c, dpi=180, bbox_inches='tight', facecolor=fig3.get_facecolor())
plt.close()
print(f"Saved: {out_path_c}")

print("\n✅ All plots saved to:", OUT_DIR.resolve())
