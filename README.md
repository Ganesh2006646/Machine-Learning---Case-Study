# 📶 Wi-Fi Fingerprint-Based Indoor Localization Using Machine Learning

> **Course:** 23CSE301 Machine Learning — Capstone Project  
> **Degree:** B.Tech. Computer Science and Engineering (III Year) — Academic Year 2026–27  
> **Milestone:** Review 1 Deliverable (Regression Track + Classification Part A)

---

## 👨‍💻 Team & Contribution Breakdown

Contributions are tracked transparently via Git commit history across team members according to assigned technical tracks:

| Team Member | Roll No. | Role & Module Ownership | Primary Git Commits |
| :--- | :--- | :--- | :--- |
| **K Ganesh Giridhar** *(Lead)* | 519 | Regressors 6–10 (Trees, Ensembles, SVR, KNN), Hyperparameter Tuning (`GridSearchCV`), Visualisations (Residuals, Feature Importances), PPT & Final Integration | `962625c`, `4acf7e3`, `909272d`, `bd58f53`, `1d2a829` |
| **A Suhas Reddy** | 503 | Feature Engineering (RSSI Aggregates), Stratified Splitting & Leakage-Free Scaling Pipeline, Classification Part A (5 Classifiers), Confusion Matrices | `6f1f198`, `809f3cb`, `de1f6e9`, `d901d30`, `d89edd3`, `ca59942` |
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
4. **Strict Data Leakage Controls:** Train/Test split (80:20, `random_state=42`) is executed prior to feature transformation. All `StandardScaler` and `MinMaxScaler` parameters are fitted **exclusively on the training split** and applied onto the test split.

---

## 📈 Review 1 Model Evaluation & Results

### 1. Regression Track — Indoor Coordinate Prediction (`LONGITUDE`)

Evaluated across all 10 mandated regression algorithms on the identical held-out test split (20% test set, 3,988 samples). Ranked by Coefficient of Determination ($R^2$).

| Rank | Model Name | $R^2$ Score | RMSE (metres) | MAE (metres) | Primary Hyperparameters / Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **KNN Regressor** | **0.9967** | **7.17** | **2.16** | $k=5$, `weights='distance'`, Euclidean metric |
| **2** | **Random Forest Regressor** | **0.9911** | **11.77** | **5.42** | $n\_estimators=100$, $max\_depth=20$, `random_state=42` |
| **3** | **Gradient Boosting Regressor** | **0.9896** | **12.73** | **8.41** | $n\_estimators=150$, $learning\_rate=0.1$, $max\_depth=5$ |
| **4** | Decision Tree Regressor | 0.9714 | 21.08 | 8.60 | $max\_depth=15$, `random_state=42` |
| **5** | Support Vector Regressor (SVR) | 0.9597 | 25.04 | 11.18 | RBF Kernel, $C=100$, $\epsilon=0.1$ |
| **6** | Linear Regression | 0.9511 | 27.59 | 20.07 | Baseline OLS Model |
| **7** | Ridge Regression | 0.9511 | 27.59 | 20.07 | L2 Regularisation ($\alpha=1.0$) |
| **8** | Lasso Regression | 0.9479 | 28.47 | 20.99 | L1 Regularisation ($\alpha=0.5$), Feature Sparsity Observed |
| **9** | ElasticNet Regression | 0.9437 | 29.60 | 21.88 | L1 + L2 Combination ($\alpha=0.5$, $l1\_ratio=0.5$) |
| **10** | Polynomial Regression | 0.8208 | 52.79 | 39.14 | Degree 2 with SelectKBest ($k=20$) Feature Pre-selection |

#### 🏆 Top 2 Cross-Validation Stability (5-Fold CV $R^2$)
* **KNN Regressor (Top 1):** Mean $R^2 = 0.9964 \pm 0.0005$
* **Random Forest Regressor (Top 2):** Mean $R^2 = 0.9908 \pm 0.0012$

---

### 2. Classification Track — Floor Level Prediction (Part A)

Evaluated across all 5 mandated Part A classification algorithms predicting `FLOOR` (Floors 0–4) with stratified splitting.

| Rank | Classifier Name | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Key Observations |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Support Vector Machine (SVC)** | **0.9937** | **0.9938** | **0.9937** | **0.9937** | RBF Kernel ($C=10$), optimal decision boundary |
| **2** | **Logistic Regression** | **0.9890** | **0.9890** | **0.9890** | **0.9890** | Baseline One-vs-Rest, surprisingly strong linearity |
| **3** | **KNN Classifier** | **0.9887** | **0.9888** | **0.9887** | **0.9887** | $k=5$, distance weighting captures spatial proximity |
| **4** | Decision Tree Classifier | 0.7931 | 0.8622 | 0.7931 | 0.8017 | $max\_depth=15$, prone to axis-aligned boundary splits |
| **5** | Gaussian Naive Bayes | 0.4982 | 0.6869 | 0.4982 | 0.5158 | Independence assumption violated by co-located WAPs |

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
│ Slide 5 │ Regression Results (10 Models)  │ R², RMSE Table, KNN (0.9967) Champion │
│ Slide 6 │ Regression Diagnostic Plots     │ Residuals, Predicted vs Actual        │
│ Slide 7 │ Classification Part A (5 Models)│ Floor Accuracy, SVM & Logistic Performance│
│ Slide 8 │ Confusion Matrix & Inferences   │ Inter-floor Overlap Analysis          │
│ Slide 9 │ Review 2 Roadmap & Conclusion   │ Clustering Track & Part B Models      │
└─────────┴─────────────────────────────────┴───────────────────────────────────────┘
```

### Detailed Slide Outline for Presentations

* **Slide 1: Title & Team**
  * *Title:* Wi-Fi Fingerprint-Based Indoor Localization Using Machine Learning
  * *Presenters:* K Ganesh Giridhar (519), A Suhas Reddy (503), G R Balaji (510)
  * *Course:* 23CSE301 Machine Learning (Academic Year 2026–27)

* **Slide 2: Background & Problem Statement**
  * *Problem:* GPS signals cannot penetrate concrete walls and multi-floor buildings.
  * *Solution:* Utilize existing campus Wi-Fi infrastructure (RSSI signals) to determine exact location.
  * *Dual Objectives:* (1) Regression for (X, Y) Coordinates, (2) Classification for Floor Number.

* **Slide 3: Dataset Audit & Preprocessing**
  * Real-world UJIndoorLoc dataset (19,937 train / 1,111 validation records).
  * Replaced non-detected `+100` sentinel values with `-105 dBm`.
  * Generated aggregate features: `n_visible_aps`, `mean_rssi`, `max_rssi`, `std_rssi`, `rssi_range`.
  * Scaled features strictly post train-test split to guarantee zero data leakage.

* **Slide 4: Regression Performance Analysis**
  * Compared 10 regression algorithms on identical test split.
  * **KNN Regressor** achieved top performance ($R^2 = 0.9967$, $RMSE = 7.17\text{ m}$, $MAE = 2.16\text{ m}$).
  * Tree ensembles (Random Forest $R^2=0.9911$, Gradient Boosting $R^2=0.9896$) closely followed.

* **Slide 5: Classification Performance Analysis**
  * Evaluated 5 Part A classification algorithms for Floor prediction.
  * **SVM (SVC)** achieved peak weighted F1-Score of **0.9937**, followed by Logistic Regression (**0.9890**).
  * Gaussian Naive Bayes underperformed (F1 = 0.5158) due to signal feature correlation violating independence assumptions.

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
│   ├── regression.ipynb        ← Complete Regression Notebook (Models 1–10, Plots)
│   └── classification.ipynb    ← Complete Classification Notebook (Part A Models 1–5)
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
