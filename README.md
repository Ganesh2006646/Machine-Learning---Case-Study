# 📶 Wi-Fi Fingerprint-Based Indoor Localization Using Machine Learning

> **Course:** 23CSE301 Machine Learning — Capstone Project  
> **Degree:** B.Tech. Computer Science and Engineering (III Year) — Academic Year 2026–27  
> **Scope:** Complete End-to-End Pipeline — **Regression Track + Classification Track (Parts A & B) + Clustering Track**

---

## 👨‍💻 Team & Contribution Breakdown

Contributions are tracked transparently via Git commit history across all three team members according to assigned technical tracks:

| Team Member | Roll No. | Role & Module Ownership | Key Git Commits |
| :--- | :---: | :--- | :--- |
| **K Ganesh Giridhar** *(Lead)* | 519 | Regressors 6–10 (Trees, Ensembles, SVR, KNN), Hyperparameter Tuning (`GridSearchCV`), Visualisations (Residuals, Feature Importances, Comparison Plots, ROC Curves), PPT & Final Integration | `962625c`, `4acf7e3`, `909272d`, `bd58f53`, `5a16a57`, `d3c95c0` |
| **A Suhas Reddy** | 503 | Feature Engineering (RSSI Aggregates), Stratified Splitting & Leakage-Free Scaling Pipeline, Classification Track Parts A & B (All 10 Classifiers), One-vs-Rest ROC-AUC Analysis, Confusion Matrices | `01faaf9`, `6f1f198`, `809f3cb`, `de1f6e9`, `d901d30`, `d89edd3`, `dee519d` |
| **G R Balaji** | 510 | Dataset Audit & Exploratory Data Analysis (EDA), Data Cleaning (Sentinel treatment, Zero-Variance removal), Regression Models 1–5 (Linear Baselines), Complete Clustering Track (K-Means, Hierarchical, Elbow, Dendrogram, PCA & t-SNE) | `12f87c2`, `5722ffd`, `c3be9cd`, `f1b9ed9`, `fc06550`, `2c27d15` |

---

## 📌 Project Overview & Motivation

Indoor Positioning Systems (IPS) are essential because Satellite GPS signals degrade heavily inside multi-story concrete structures due to line-of-sight attenuation and multipath fading. This capstone develops a complete, three-track Machine Learning pipeline using Received Signal Strength Indicator (RSSI) fingerprints gathered from 520 ubiquitous Wi-Fi Access Points (APs):

1. **Regression Track:** Predict continuous indoor **(X, Y) coordinates** (`LONGITUDE` & `LATITUDE` in metres).
2. **Classification Track (Parts A & B):** Predict discrete **Floor levels** (`FLOOR` 0 to 4) across multi-story buildings.
3. **Clustering Track (Unsupervised):** Discover natural spatial groupings (buildings and floor zones) purely from raw signal fingerprints without ground-truth labels during training.

```
  ┌───────────────────────┐
  │  Wi-Fi Access Points  │
  └───────────┬───────────┘
              ▼
  ┌───────────────────────┐      ┌───────────────────────────────┐
  │   RSSI Fingerprint    ├─────►│ Preprocessing & Feature Eng.  │
  │ (520 WAP dBm Signals) │      │ (-105 Sentinel, Scaling, Agg) │
  └───────────────────────┘      └──────────────┬────────────────┘
                                                │
       ┌────────────────────────────────────────┼────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│  1. Regression Track         │ │  2. Classification Track     │ │  3. Clustering Track         │
│  10 Algorithms (Longitude)   │ │  10 Algorithms (Floor 0–4)   │ │  K-Means & Hierarchical      │
│  Champion: KNN (R² = 0.9966) │ │  Champion: SVM (F1 = 0.9937) │ │  PCA, t-SNE, Elbow, Dendro   │
└──────────────────────────────┘ └──────────────────────────────┘ └──────────────────────────────┘
```

---

## 📊 Dataset Audit — UJIndoorLoc

The project utilizes the **UJIndoorLoc** benchmark dataset from the UCI Machine Learning Repository, captured across 3 buildings of Universitat Jaume I, Spain.

* **Total Samples:** 19,937 Training samples \| 1,111 Official Validation samples
* **Raw Features:** 520 WAP Signal Columns (`WAP001` to `WAP520`)
* **Active Features After Cleaning:** 465 active WAPs + 5 Engineered RSSI Features = **470 total features**
* **Metadata Columns:** `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID`, `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP`
* **RSSI Value Encoding:** Detected RSSI ranges from `-104 dBm` (weakest) to `0 dBm` (strongest). Non-detected Access Points are encoded as sentinel value `+100`.

---

## ⚙️ Data Preprocessing & Leakage Prevention

1. **Sentinel Value Rectification:** Replaced non-detected `+100` signals with `-105 dBm` (just below the `-104 dBm` physical receiver sensitivity floor). This preserves physical signal ordering for distance and linear estimators.
2. **Zero-Variance Feature Elimination:** Identified and pruned 55 WAPs that were never detected across the training corpus, reducing raw WAP features from 520 to 465.
3. **RSSI Feature Engineering (5 Aggregate Spatial Cues):**
   * `n_visible_aps`: Total count of active APs with RSSI > `-105 dBm`.
   * `mean_rssi`: Average signal strength of detected APs per sample.
   * `max_rssi`: Signal strength of the strongest visible AP.
   * `std_rssi`: Standard deviation of signal strength across visible APs.
   * `rssi_range`: Dynamic spread between strongest and weakest visible APs.
4. **Strict Data Leakage Shield:** Train/Test split (80:20, `random_state=42`, stratified for classification) is executed prior to scaling. `StandardScaler` is fitted **exclusively on the training split** and applied via `.transform()` to the test and validation sets.

---

## 📈 Complete Model Evaluation & Results

### Track 1: Regression — Indoor Coordinate Prediction (`LONGITUDE`)

All 10 required regression algorithms trained on the identical preprocessed training split (15,949 samples) and evaluated on the held-out test set (3,988 samples), ranked by $R^2$:

| Rank | Algorithm | $R^2$ Score | RMSE (m) | MAE (m) | Key Hyperparameters & Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **KNN Regressor (Tuned)** | **0.9966** | **7.28** | **2.06** | $k=5$, `weights='distance'`, Euclidean distance |
| **2** | **Random Forest Regressor (Tuned)** | **0.9935** | **10.01** | **5.23** | `n_estimators=100`, `max_depth=25`, `min_samples_split=2` |
| **3** | **Gradient Boosting Regressor (Tuned)** | **0.9800** | **17.60** | **7.66** | `HistGradientBoostingRegressor` (`max_leaf_nodes=15`, `lr=0.1`) |
| **4** | Support Vector Regressor (SVR) | 0.9755 | 19.46 | 8.17 | RBF Kernel, $C=100$, $\epsilon=0.1$ |
| **5** | Decision Tree Regressor | 0.9717 | 20.92 | 9.15 | `max_depth=15`, `random_state=42` |
| **6** | Linear Regression | 0.9488 | 28.17 | 20.21 | Baseline Ordinary Least Squares (OLS) |
| **7** | Ridge Regression | 0.9488 | 28.17 | 20.21 | L2 Regularisation ($\alpha=1.0$) |
| **8** | Lasso Regression | 0.9419 | 29.99 | 22.25 | L1 Regularisation ($\alpha=0.5$) — zeroes 293 of 470 features |
| **9** | ElasticNet Regression | 0.9339 | 32.00 | 23.88 | Combined L1 + L2 ($\alpha=0.5$, `l1_ratio=0.5`) |
| **10** | Polynomial Regression | 0.8161 | 53.37 | 39.82 | Degree 2 on top-20 `SelectKBest` features |

---

### Track 2: Classification — Consolidated 10-Algorithm Floor Prediction (Parts A & B)

All 10 required classification algorithms (5 from Review 1 Part A + 5 from Review 2 Part B) evaluated on the stratified 20% test split (3,988 samples) predicting `FLOOR` (Floors 0–4):

| Rank | Algorithm | Track Part | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | ROC-AUC (OvR) | Key Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Support Vector Machine (SVC)** | Part A | **0.9937** | **0.9938** | **0.9937** | **0.9937** | **0.9989** | RBF Kernel ($C=10$), best overall boundary separation |
| **2** | **MLP Classifier (Neural Network)** | Part B | **0.9932** | **0.9933** | **0.9932** | **0.9932** | **0.9996** | Hidden layers `(128, 64)`, ReLU, `max_iter=300` |
| **3** | **Gradient Boosting Classifier** | Part B | **0.9925** | **0.9925** | **0.9925** | **0.9925** | **0.9997** | `n_estimators=100`, `lr=0.1`, `max_depth=5` |
| **4** | **Random Forest Classifier** | Part B | **0.9895** | **0.9895** | **0.9895** | **0.9895** | **0.9997** | Tuned via `GridSearchCV` $\rightarrow$ **Test F1 = 0.9922** |
| **5** | **Logistic Regression** | Part A | **0.9890** | **0.9890** | **0.9890** | **0.9890** | **0.9998** | Strong One-vs-Rest linear baseline |
| **6** | **KNN Classifier ($k=5$)** | Part A | **0.9887** | **0.9888** | **0.9887** | **0.9887** | **0.9992** | Distance-weighted spatial neighbourhood voting |
| **7** | Bagging Classifier | Part B | 0.9155 | 0.9207 | 0.9155 | 0.9162 | 0.9924 | 50 Decision Tree base estimators (`max_depth=15`) |
| **8** | AdaBoost Classifier | Part B | 0.9122 | 0.9211 | 0.9122 | 0.9135 | 0.9827 | 100 shallow trees (`max_depth=3`), `lr=0.5` |
| **9** | Decision Tree Classifier | Part A | 0.7931 | 0.8622 | 0.7931 | 0.8017 | 0.9481 | Single tree (`max_depth=15`), axis-aligned splits |
| **10** | Gaussian Naive Bayes | Part A | 0.4982 | 0.6869 | 0.4982 | 0.5158 | 0.7531 | Violates conditional independence due to correlated WAPs |

* **Hyperparameter Tuning Highlight:** Tuning `RandomForestClassifier` with `GridSearchCV` (`n_estimators=200`, `max_depth=25`, `min_samples_split=2`) improved Test Weighted F1 from **0.9895** to **0.9922** (CV F1 = **0.9912**).
* **Official Validation Set Benchmark:** Evaluating the champion `SVM (RBF)` across the unseen 1,111 official validation samples (collected at a different time/device mix) achieves **85.24% accuracy** and **0.8637 weighted precision**.

---

### Track 3: Clustering — Unsupervised Spatial Discovery

Clustering algorithms were trained strictly on the unlabeled 470-feature scaled matrix (`X_scaled`). Ground-truth `BUILDINGID` and `FLOOR` labels were held out and used **only after fitting** for interpretation.

| Algorithm & Configuration | Silhouette Score $\uparrow$ | Davies-Bouldin Index $\downarrow$ | Calinski-Harabasz Index $\uparrow$ | Cluster Sizes | Structural Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means ($k=3$)** | 0.0684 | 3.2018 | **958.62** | `[344, 9314, 10279]` | Highest variance ratio; isolates Building 2 (`9,314` samples) cleanly |
| **K-Means ($k=5$)** | **0.0883** | **2.9513** | 789.03 | `[2580, 9435, 337, 2663, 4922]` | Best Silhouette & Davies-Bouldin among multi-group partitions |
| **Agglomerative ($k=3$, Ward)** | 0.0670 | 3.1815 | 936.98 | `[10343, 287, 9307]` | Matches K-Means building-level structure via bottom-up merging |
| **Agglomerative ($k=5$, Ward)** | 0.0784 | 2.9978 | 769.54 | `[5122, 2877, 9307, 287, 2344]` | Captures sub-building wing/floor groups |

* **Key Clustering Insights:**
  * **Elbow & Silhouette Analysis ($k=2\dots14$):** Inertia drops steeply from $k=2$ ($8.85\text{M}$, Silhouette $= 0.4735$) through $k=6$ ($7.95\text{M}$, Silhouette $= 0.1068$).
  * **Dendrogram Linkage Comparison:** Ward linkage yields balanced, well-separated hierarchical branches compared to Complete and Average linkage.
  * **PCA vs. t-SNE Visualisation:** 2D PCA ($10.50\%$ variance) separates building-level macro-clusters, while 2D t-SNE (`perplexity=30`) resolves tight local floor and corridor manifolds.

---

## 🎤 Presentation Structure (Review 1 & Review 2 Full Story Arc)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          CAPSTONE PRESENTATION SLIDE DECK                            │
├─────────┬──────────────────────────────────┬─────────────────────────────────────────┤
│ Slide # │ Title                            │ Key Content & Speaker Talking Points    │
├─────────┼──────────────────────────────────┼─────────────────────────────────────────┤
│ Slide 1 │ Title & Team Introduction        │ Project Name, Course 23CSE301, Members  │
│ Slide 2 │ Motivation & Problem Statement   │ GPS Failure Indoors, Wi-Fi RSSI Concept │
│ Slide 3 │ Dataset & Preprocessing Pipeline │ UJIndoorLoc, -105 Sentinel, 5 Eng Feats │
│ Slide 4 │ Track 1: Regression (10 Models)  │ KNN (R²=0.9966, MAE=2.06m) & Ensembles  │
│ Slide 5 │ Regression Diagnostics           │ Residuals, Feature Importance, Error CDF│
│ Slide 6 │ Track 2: Classification (All 10) │ Consolidated Part A+B Table, SVM & MLP  │
│ Slide 7 │ ROC-AUC & Confusion Matrices     │ 10-Model OvR ROC Curves & Floor Overlap │
│ Slide 8 │ Track 3: Unsupervised Clustering │ Elbow Curve, Ward Dendrogram, Metrics   │
│ Slide 9 │ PCA & t-SNE Cluster Projections  │ Post-hoc Building vs Floor Alignment    │
│ Slide 10│ Conclusion & Deployment Takeaways│ Summary of Best Models & Practical IPS  │
└─────────┴──────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```
Machine-Learning---Case-Study/
│
├── README.md                   ← Master Documentation, Results Tables & PPT Story Arc
├── requirements.txt            ← Python Dependencies & Version Pins
├── .gitignore                  ← Git Exclusion Rules
│
├── notebooks/
│   ├── regression.ipynb        ← Track 1: EDA, Cleaning, 10 Regressors, Tuning & Plots
│   ├── classification.ipynb    ← Track 2: Parts A & B (All 10 Classifiers), ROC-AUC & Tuning
│   ├── clustering.ipynb        ← Track 3: K-Means, Hierarchical, Elbow, Dendrogram, PCA & t-SNE
│   └── figures/                ← Saved High-Resolution ROC Curves & Visualisations (.png)
│
├── data/
│   ├── raw/                    ← UJIndoorLoc: trainingData.csv (19,937×529), validationData.csv (1,111×529)
│   └── campus/                 ← Live scan: raw_scans.csv, trainingData_campus.csv (524×219), wap_dictionary.json
│
├── scripts/
│   ├── campus_collector.py     ← Windows netsh collector (Block/Floor/Room + auto-GPS)
│   ├── build_campus_dataset.py ← raw_scans → wide ML matrix + WAP map
│   └── train_champions.py      ← trains UJI + campus champions → models/
│
├── app/
│   ├── app.py                  ← Bonus GUI entry (run: streamlit run app/app.py)
│   └── app_utils/              ← preprocess.py (shared clean/feats), scan_windows.py (live netsh)
│
├── models/                     ← regenerated artefacts (git-ignored *.pkl): uji_*/campus_* + room_centroids.csv
│
└── docs/
    └── STREAMLIT_BONUS_GUIDE.md ← +2 bonus: palette, scan flow, hosting, viva Q&A
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

# 4. Launch Jupyter Notebook
jupyter notebook
# open notebooks/regression.ipynb, classification.ipynb, clustering.ipynb (run top-to-bottom)

# 5. Retrain deployable champions (UJI + campus) — regenerates models/
python scripts/build_campus_dataset.py
python scripts/train_champions.py

# 6. Run bonus dashboard locally (Windows live scan works here)
streamlit run app/app.py

# 7. Campus live collection (Windows admin terminal, Wi-Fi on)
python scripts/campus_collector.py
```

---

## 📜 Citation & References

* **Dataset Source:** Torres-Sospedra, J., et al. (2014). *UJIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization.* International Conference on Indoor Positioning and Indoor Navigation (IPIN).
* **Course Rubric:** 23CSE301 Machine Learning Capstone Guidelines, B.Tech. CSE III Year, Academic Year 2026–27.
