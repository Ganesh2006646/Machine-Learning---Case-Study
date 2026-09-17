"""
Generate one individual ROC curve image per classifier.
Saves to notebooks/figures/<model_name>_roc.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

OUT_DIR = Path("notebooks/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = ['#6C63FF', '#FF6584', '#43B89C', '#F5A623', '#2196F3']
FLOOR_LABELS = ['Floor 0', 'Floor 1', 'Floor 2', 'Floor 3', 'Floor 4']
CLASSES = [0, 1, 2, 3, 4]
N_CLASSES = len(CLASSES)

# ── Load & preprocess ────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv("data/raw/trainingData.csv")
WAP_COLS = [c for c in df.columns if c.startswith("WAP")]
df[WAP_COLS] = df[WAP_COLS].replace(100, -110)
df["WAP_VAR"] = df[WAP_COLS].var(axis=1)
df = df[df["WAP_VAR"] > 0].reset_index(drop=True)
threshold = 0.99 * len(df)
always_off = [c for c in WAP_COLS if (df[c] == -110).sum() > threshold]
df.drop(columns=always_off, inplace=True)
WAP_COLS = [c for c in df.columns if c.startswith("WAP")]

X = df[WAP_COLS].values
y = df["FLOOR"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
X_test_sc  = np.nan_to_num(scaler.transform(X_test), nan=0.0)
y_test_bin = label_binarize(y_test, classes=CLASSES)

# ── Classifiers ───────────────────────────────────────────────────────────────
classifiers = [
    ("logistic_regression", "Logistic Regression",
     OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))),
    ("knn", "KNN (k=5)",
     OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))),
    ("naive_bayes", "Gaussian Naive Bayes",
     OneVsRestClassifier(GaussianNB())),
    ("decision_tree", "Decision Tree",
     OneVsRestClassifier(DecisionTreeClassifier(max_depth=15, random_state=42))),
    ("svm_rbf", "SVM (RBF)",
     OneVsRestClassifier(SVC(kernel='rbf', C=10, random_state=42, probability=True))),
]

for slug, name, model in classifiers:
    print(f"  Training {name} ...", flush=True)
    model.fit(X_train_sc, y_train)
    y_score = model.predict_proba(X_test_sc)

    fpr, tpr, roc_auc = {}, {}, {}
    for i, cls in enumerate(CLASSES):
        fpr[cls], tpr[cls], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    # Micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#F8F9FB')
    ax.set_facecolor('#FDFDFE')

    for i, cls in enumerate(CLASSES):
        ax.plot(fpr[cls], tpr[cls], color=CLASS_COLORS[i], lw=2,
                label=f"{FLOOR_LABELS[i]}  (AUC = {roc_auc[cls]:.3f})")

    ax.plot(fpr['micro'], tpr['micro'], color='#E74C3C', lw=2.5, linestyle='--',
            label=f"Micro-avg  (AUC = {roc_auc['micro']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Chance (AUC = 0.5)')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve (One-vs-Rest) - {name}", fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    out_path = OUT_DIR / f"{slug}_roc.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved: {out_path}")

print("\nAll individual ROC curves saved.")
