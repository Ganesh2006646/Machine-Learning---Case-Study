# 📶 Wi-Fi Fingerprint-Based Indoor Localization Using Machine Learning

> **Course:** 23CSE301 Machine Learning — Capstone Project  
> **Degree:** B.Tech. Computer Science and Engineering (III Year) — Academic Year 2026–27  
> **Milestone:** Review 1 Deliverable (Regression Track + Classification Part A)

---

## 👨‍💻 Team & Contribution Breakdown

Contributions are tracked transparently via Git commit history across team members according to assigned technical tracks:

| Team Member | Roll No. | Role & Module Ownership | Key Git Commits |
| :--- | :--- | :--- | :--- |
| **K Ganesh Giridhar** *(Lead)* | 519 | Regressors 6–10 (Trees, Ensembles, SVR, KNN), Hyperparameter Tuning (`GridSearchCV`), Visualisations (Residuals, Feature Importances, Comparison Plots), PPT & Final Integration | `962625c`, `4acf7e3`, `909272d`, `bd58f53`, `5bfcab2`, `6e30ff4` |
| **A Suhas Reddy** | 503 | Feature Engineering (RSSI Aggregates), Stratified Splitting & Leakage-Free Scaling Pipeline, Classification Part A (5 Classifiers), One-vs-Rest ROC Curve Diagrams, Confusion Matrices | `6f1f198`, `809f3cb`, `de1f6e9`, `d901d30`, `d89edd3`, `dee519d` |
| **G R Balaji** | 510 | Dataset Audit & Exploratory Data Analysis (EDA), Data Cleaning (Sentinel treatment, Zero-Variance removal), Regression Models 1–5 (Linear Baselines), Comparative Summary Tables | `5722ffd`, `c3be9cd`, `f1b9ed9`, `fc06550`, `2c27d15` |

---

## 📌 Project Overview & Motivation

Indoor Positioning Systems (IPS) are essential because Satellite GPS signals degrade heavily inside multi-story concrete structures due to line-of-sight attenuation. This capstone develops an end-to-end Machine Learning pipeline predicting exact indoor **(X, Y) coordinates** (Longitude & Latitude) and **Floor levels** using Received Signal Strength Indicator (RSSI) fingerprints gathered from ubiquitous Wi-Fi Access Points (APs).

```
  ┌───────────────────────┐
  │  Wi-Fi Access Points  │
  └───────────┬───────────┘
              ▼
  ┌───────────────────────┐      ┌───────────────────────────────┐
  │   RSSI Fingerprint    ├─────►│ Preprocessing & Feature Eng.  │
  │ (520 WAP dBm Signals) │      │ (-105 Sentinel, Scaling, Agg) │
  └───────────────────────┘      └──────────────┬────────────────┘
                                                ▼
  ┌─────────────────────────────────────────────┴──────────────────────────────┐
  │                                                                            │
  ▼                                                                            ▼
┌─────────────────────────────────────────┐   ┌────────────────────────────────────────┐
│  Regression Pipeline (Longitude / Lat)  │   │   Classification Pipeline (Floor)      │
│  Outputs: (X, Y) Coordinates in Metres  │   │   Outputs: Floor Level (Floor 0 to 4)  │
└─────────────────────────────────────────┘   └────────────────────────────────────────┘
```

---

## 📊 Dataset Audit — UJIndoorLoc

The project utilizes the **UJIndoorLoc** benchmark dataset from the UCI Machine Learning Repository, captured across 3 buildings of Universitat Jaume I, Spain.

* **Total Samples:** 19,937 Training samples \| 1,111 Validation samples
* **Raw Features:** 520 WAP Signal Columns (`WAP001` to `WAP520`)
* **Metadata Columns:** `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`
* **RSSI Value Encoding:** Detected RSSI ranges from `-104 dBm` (weakest) to `0 dBm` (strongest). Non-detected Access Points are encoded as sentinel value `+100`.

---

## ⚙️ Data Preprocessing & Leakage Prevention

1. **Sentinel Value Rectification:** Replaced non-detected `+100` signals with `-105 dBm` (below the physical receiver sensitivity floor). This preserves ordinal continuity for distance-based estimators.
2. **Zero-Variance Feature Elimination:** Identified and pruned WAPs that were never detected across the training corpus, reducing feature space dimensionality from 520 to 465 WAPs.
3. **RSSI Feature Engineering:**
   * `n_visible_aps`: Total count of active APs with RSSI > -105 dBm.
   * `mean_rssi`: Average signal strength of detected APs per sample.
   * `max_rssi`: Signal strength of the strongest visible AP.
   * `std_rssi`: Variance in signal strength across visible APs.
   * `rssi_range`: Signal spread between strongest and weakest visible APs.
4. **Strict Data Leakage Controls:** Train/Test split (80:20, `random_state=42`) is executed prior to feature transformation. All `StandardScaler` parameters are fitted **exclusively on the training split** and applied onto the test split.

---

## 📈 Review 1 Model Evaluation & Results

### 1. Regression Track — Indoor Coordinate Prediction (`LONGITUDE`)

Evaluated across all 10 mandated regression algorithms on the identical held-out test split (20% test set, 3,988 samples). Ranked by Coefficient of Determination ($R^2$).

| Rank | Model Name | $R^2$ Score | RMSE (metres) | MAE (metres) | Primary Hyperparameters / Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **KNN Regressor (Tuned)** | **0.9966** | **7.28** | **2.06** | $k=5$, `weights='distance'`, Euclidean metric |
| **2** | **Random Forest Regressor (Tuned)** | **0.9935** | **10.01** | **5.23** | $n\_estimators=100$, $max\_depth=25$, `random_state=42` |
| **3** | **Gradient Boosting Regressor (Tuned)** | **0.9800** | **17.60** | **7.66** | `HistGradientBoostingRegressor` ($max\_leaf\_nodes=15$, $lr=0.1$) |
| **4** | Support Vector Regressor (SVR) | 0.9755 | 19.46 | 8.17 | RBF Kernel, $C=100$, $\epsilon=0.1$ |
| **5** | Decision Tree Regressor | 0.9717 | 20.92 | 9.15 | $max\_depth=15$, `random_state=42` |
| **6** | Linear Regression | 0.9488 | 28.17 | 20.21 | Baseline OLS Model |
| **7** | Ridge Regression | 0.9488 | 28.17 | 20.21 | L2 Regularisation ($\alpha=1.0$) |
| **8** | Lasso Regression | 0.9419 | 29.99 | 22.25 | L1 Regularisation ($\alpha=0.5$), 293 Features Zeroed |
| **9** | ElasticNet Regression | 0.9339 | 32.00 | 23.88 | L1 + L2 Combination ($\alpha=0.5$, $l1\_ratio=0.5$) |
| **10** | Polynomial Regression | 0.8161 | 53.37 | 39.82 | Degree 2 with SelectKBest ($k=20$) Feature Pre-selection |

---

### 2. Classification Track — Floor Level Prediction (Part A)

Evaluated across all 5 mandated Part A classification algorithms predicting `FLOOR` (Floors 0–4) with stratified splitting and One-vs-Rest ROC analysis.

| Rank | Classifier Name | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Key Observations & ROC Diagrams |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Support Vector Machine (SVC)** | **0.9937** | **0.9938** | **0.9937** | **0.9937** | RBF Kernel ($C=10$), optimal decision boundary (`figures/svm_rbf_roc.png`) |
| **2** | **Logistic Regression** | **0.9890** | **0.9890** | **0.9890** | **0.9890** | Baseline One-vs-Rest, surprisingly strong linearity (`figures/logistic_regression_roc.png`) |
| **3** | **KNN Classifier** | **0.9887** | **0.9888** | **0.9887** | **0.9887** | $k=5$, distance weighting captures spatial proximity (`figures/knn_roc.png`) |
| **4** | Decision Tree Classifier | 0.7931 | 0.8622 | 0.7931 | 0.8017 | $max\_depth=15$, axis-aligned decision boundaries (`figures/decision_tree_roc.png`) |
| **5** | Gaussian Naive Bayes | 0.4982 | 0.6869 | 0.4982 | 0.5158 | Independence assumption violated by co-located WAPs (`figures/naive_bayes_roc.png`) |

---

## 🎤 Presentation Structure (PPT Content Outline)

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               SLIDE PRESENTATION DECK                             │
├─────────┬─────────────────────────────────┬───────────────────────────────────────┤
│ Slide # │ Title                           │ Key Content & Speaker Talking Points  │
├─────────┼─────────────────────────────────┼───────────────────────────────────────┤
│ Slide 1 │ Title & Team Introduction       │ Project Name, Course Code, Members    │
│ Slide 2 │ Motivation & Problem Statement  │ GPS Failure Indoors, RSSI Concept     │
│ Slide 3 │ Dataset Architecture & Audit    │ UJIndoorLoc, 520 WAPs, Sentinel +100  │
│ Slide 4 │ Preprocessing & Leakage Shield │ -105 Replacement, Scaling, Eng. Feats │
│ Slide 5 │ Regression Results (10 Models)  │ R², RMSE Table, KNN (0.9966) Champion │
│ Slide 6 │ Regression Diagnostic Plots     │ Residuals, Predicted vs Actual        │
│ Slide 7 │ Classification Part A (5 Models)│ Floor Accuracy, SVM (0.9937 F1)       │
│ Slide 8 │ Confusion Matrix & ROC Curves   │ One-vs-Rest ROC & Inter-floor Overlap │
│ Slide 9 │ Review 2 Roadmap & Conclusion   │ Clustering Track & Part B Models      │
└─────────┴─────────────────────────────────┴───────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```
Machine-Learning---Case-Study/
│
├── README.md                   ← Master Documentation & Final Results Summary
├── requirements.txt            ← Python Dependencies & Pins
├── .gitignore                  ← Git Exclusion Rules
│
├── data/
│   ├── raw/                    ← Original Untouched Dataset CSVs
│   │   ├── trainingData.csv
│   │   └── validationData.csv
│   └── processed/              ← Cleaned & Transformed Output Datasets
│
├── notebooks/
│   ├── regression.ipynb        ← Complete Regression Notebook (Models 1–10, Plots, Tuning)
│   ├── classification.ipynb    ← Complete Classification Notebook (Part A Models 1–5, ROCs)
│   └── figures/                ← Output ROC Curves and Comparative Plots (.png)
│
├── models/                     ← Serialized Trained Models (.pkl)
├── app/                        ← Streamlit / Web GUI Deployment Code (Review 2)
└── trash/
    └── work_allocation.md      ← Detailed Work Split & Team Task Rubric Mapping
```

---

## ⚡ Quick Start & Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/Ganesh2006646/Machine-Learning---Case-Study.git
cd Machine-Learning---Case-Study

# 2. Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebooks
jupyter notebook
```

---

## 📜 Citation & References

* **Dataset Source:** Torres-Sospedra, J., et al. (2014). *UJIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization.* International Conference on Indoor Positioning and Indoor Navigation (IPIN).
* **Guidelines:** 23CSE301 Machine Learning Capstone Guidelines, B.Tech. CSE III Year, Academic Year 2026–27.
