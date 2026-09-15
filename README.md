# Wi-Fi Fingerprint-Based Indoor Localization Using Machine Learning

> **23CSE301 Machine Learning — Capstone Project | B.Tech CSE III Year | Academic Year 2026–27**

## Team

| Name | Roll Number |
|------|-------------|
| K Ganesh Giridhar | 519 |
| G R Balaji | 510 |
| A Suhas Reddy | 503 |

---

## Problem Statement

Estimate the **indoor location** of a device from its Wi-Fi RSSI (Received Signal Strength Indicator) fingerprint. The project addresses three machine-learning tracks using a single real-world dataset:

| Track | Target Variable | Question Answered |
|-------|----------------|-------------------|
| **Regression** | Longitude, Latitude (X, Y coordinates) | *Where exactly is the device?* |
| **Classification** | Floor | *Which floor is the device on?* |
| **Clustering** *(Review 2)* | Unsupervised grouping of fingerprints | *Do Wi-Fi fingerprints naturally form spatial clusters?* |

```
Wi-Fi Access Points  →  RSSI Fingerprint  →  Preprocessing  →  ML Model  →  Indoor Location
```

---

## Dataset

**UJIndoorLoc** — sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).

| Property | Value |
|----------|-------|
| WAP features | 520 (WAP001 – WAP520) |
| Metadata columns | LONGITUDE, LATITUDE, FLOOR, BUILDINGID, SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP |
| Training samples | ~19,937 |
| Validation samples | 1,111 |
| RSSI encoding | Detected: –104 dBm to 0 dBm; Not detected: +100 |
| Environment | 3 buildings, 4–5 floors, Universitat Jaume I campus, Spain |

The raw data is stored in `data/raw/` and remains untouched. All cleaned and engineered versions are saved separately in `data/processed/`.

---

## Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Data Loading & Audit                                        │
│     Shape, dtypes, missing values, target distribution          │
├─────────────────────────────────────────────────────────────────┤
│  2. Exploratory Data Analysis (EDA)                             │
│     Feature distributions, correlation heatmap, scatter plots   │
├─────────────────────────────────────────────────────────────────┤
│  3. Preprocessing & Feature Engineering                         │
│     Handle +100 sentinel, remove zero-variance WAPs,            │
│     engineer RSSI summary features, encode, scale               │
├─────────────────────────────────────────────────────────────────┤
│  4. Train/Test Split (80:20, random_state=42)                   │
│     Fit preprocessing on TRAIN only → transform both            │
├─────────────────────────────────────────────────────────────────┤
│  5. Model Training & Evaluation                                 │
│     All required algorithms per track                           │
├─────────────────────────────────────────────────────────────────┤
│  6. Hyperparameter Tuning                                       │
│     GridSearchCV / RandomizedSearchCV on top models             │
├─────────────────────────────────────────────────────────────────┤
│  7. Result Visualisation & Comparison Tables                    │
│     Consolidated metrics, residual plots, confusion matrices    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Leakage Prevention

All scalers, imputers, and encoders are **fitted on the training set only** and then applied (transformed) to both training and test sets. This is enforced throughout every notebook.

---

## Algorithms & Metrics

### Regression Track (Review 1)

All ten algorithms are trained on the same preprocessed dataset and evaluated on the same held-out test set.

| # | Algorithm | Key Tuning / Notes |
|---|-----------|-------------------|
| 1 | Linear Regression | Baseline; interpret coefficients |
| 2 | Ridge Regression | L2 regularisation; tune alpha |
| 3 | Lasso Regression | L1 regularisation; observe feature sparsity |
| 4 | ElasticNet Regression | Combined L1 + L2; tune l1_ratio |
| 5 | Polynomial Regression | PolynomialFeatures + LinearRegression; compare degrees |
| 6 | Decision Tree Regressor | Tune max_depth; show feature importance |
| 7 | Random Forest Regressor | Ensemble baseline; tune n_estimators |
| 8 | Gradient Boosting Regressor | sklearn GBM / XGBoost; tune learning_rate |
| 9 | Support Vector Regressor (SVR) | Scale features first; tune C and kernel |
| 10 | K-Nearest Neighbors Regressor | Tune k; discuss impact of scaling |

**Metrics:** R², RMSE, MAE, 5-fold cross-validated R² (top-2 models), physical positioning error (metres).

### Classification Track — Part A (Review 1)

| # | Algorithm | Key Tuning / Notes |
|---|-----------|-------------------|
| 1 | Logistic Regression | Baseline classifier; interpret coefficients |
| 2 | K-Nearest Neighbors | Tune k; discuss distance metrics |
| 3 | Gaussian Naive Bayes | Discuss conditional independence assumption |
| 4 | Decision Tree Classifier | Tune max_depth; visualise the tree |
| 5 | Support Vector Machine (SVC) | Tune C and kernel; scale features |

**Metrics:** Accuracy, Precision, Recall, Weighted F1-score, Confusion Matrix.

### Classification Track — Part B (Review 2)

| # | Algorithm |
|---|-----------|
| 6 | Random Forest Classifier |
| 7 | AdaBoost Classifier |
| 8 | Gradient Boosting Classifier |
| 9 | Bagging Classifier |
| 10 | MLP Classifier (Neural Network) |

### Clustering Track (Review 2)

| # | Algorithm | Notes |
|---|-----------|-------|
| 1 | K-Means Clustering | Elbow curve (inertia vs. k) |
| 2 | Agglomerative Hierarchical Clustering | Dendrogram; compare linkage strategies |

**Metrics:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index. Visualisation via PCA (mandatory) and t-SNE (encouraged).

---

## Feature Engineering

Candidate engineered features derived from the RSSI fingerprint:

| Feature | Description |
|---------|-------------|
| `n_visible_aps` | Count of WAPs with RSSI ≠ +100 |
| `mean_rssi` | Mean RSSI of detected APs |
| `median_rssi` | Median RSSI of detected APs |
| `std_rssi` | Standard deviation of detected RSSI values |
| `max_rssi` | Strongest signal observed |
| `rssi_range` | Difference between strongest and weakest detected signal |

Final feature selection is driven by the EDA — only features that demonstrate empirical value are retained.

---

## Repository Structure

```
Machine-Learning---Case-Study/
│
├── README.md                   ← Project overview, results, setup instructions
├── requirements.txt            ← Python dependencies with versions
├── .gitignore                  ← Excludes large data, checkpoints, OS files
│
├── data/
│   ├── raw/                    ← Original UJIndoorLoc CSVs (read-only)
│   │   ├── trainingData.csv
│   │   └── validationData.csv
│   └── processed/              ← Cleaned & engineered datasets
│
├── notebooks/
│   ├── regression.ipynb        ← Full Regression track
│   ├── classification.ipynb    ← Full Classification track (Parts A & B)
│   └── clustering.ipynb        ← Full Clustering track
│
├── models/                     ← Saved model files (.pkl via joblib)
│
└── app/                        ← GUI / deployment code (bonus)
```

---

## Results Summary

> **Results tables will be populated after model training is complete.**

### Regression Results

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| Linear Regression | — | — | — |
| Ridge Regression | — | — | — |
| Lasso Regression | — | — | — |
| ElasticNet | — | — | — |
| Polynomial Regression | — | — | — |
| Decision Tree Regressor | — | — | — |
| Random Forest Regressor | — | — | — |
| Gradient Boosting Regressor | — | — | — |
| SVR | — | — | — |
| KNN Regressor | — | — | — |

### Classification Part A Results

| Model | Accuracy | Precision | Recall | Weighted F1 |
|-------|----------|-----------|--------|-------------|
| Logistic Regression | — | — | — | — |
| KNN | — | — | — | — |
| Gaussian Naive Bayes | — | — | — | — |
| Decision Tree | — | — | — | — |
| SVM | — | — | — | — |

---

## Environment Setup

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Ganesh2006646/Machine-Learning---Case-Study.git
cd Machine-Learning---Case-Study

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and run all cells top-to-bottom.

---

## Review Schedule

| Review | Date | Scope |
|--------|------|-------|
| **Review 1** | Before mid-semester exams | Full Regression track + Classification Part A |
| **Review 2** | End of semester | Classification Part B + Full Clustering track |

---

## Acknowledgements

- **Dataset:** Torres-Sospedra, J. et al. (2014). *UJIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization.* [UCI ML Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).
- **Course:** 23CSE301 Machine Learning, B.Tech CSE III Year, Academic Year 2026–27.
- **Tools:** Python 3, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook.

---

## License

This project is developed for academic purposes as part of the 23CSE301 Machine Learning capstone.
