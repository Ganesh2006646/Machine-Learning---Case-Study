# Review 1 — Work Allocation

> **Team:** Ganesh (519) · Balaji (510) · Suhas (503)

---

## At a Glance

| Member | Owns | Marks Covered |
|--------|------|---------------|
| 🟦 **Balaji (510)** | EDA + Data Cleaning + Regression 1–5 + Regression Comparison Table | A1, A2, A3, B1, C1 (half), C2 |
| 🟩 **Suhas (503)** | Feature Engineering & Splitting + Classification (all 5) | B2, B3, C1 (half — shared), D1, D2 |
| 🟥 **Ganesh (519)** | Regression 6–10 + Hyperparameter Tuning + Visualisations + PPT & Presentation | C1 (half), C3, C4, E1, PPT |

---

## Detailed Breakdown

### Section A — Dataset & EDA *(4 marks)* → 🟦 Balaji

| Sub-task | Description | Mark |
|----------|-------------|------|
| **A1 — Dataset Loading & Audit** | Load `trainingData.csv` & `validationData.csv`; report shape, dtypes, missing-value counts, RSSI value-of-100 sentinel, target distributions (LONGITUDE, LATITUDE, FLOOR) | 1 |
| **A2 — EDA Visualisations** | Distribution plots for RSSI features, correlation heatmap (top WAPs), LONGITUDE/LATITUDE scatter, FLOOR bar chart, ≥2 feature-vs-target scatter plots | 2 |
| **A3 — Insight Commentary** | Write a Markdown observation below every major plot explaining what it reveals | 1 |

---

### Section B — Preprocessing & Feature Engineering *(3 marks)* → Split 🟦 Balaji + 🟩 Suhas

#### 🟦 Balaji — Data Cleaning (B1)

| Sub-task | Description | Mark |
|----------|-------------|------|
| **B1 — Data Cleaning** | Replace +100 sentinel with a meaningful value (e.g., –105 or NaN strategy), check & remove duplicates, detect & handle outliers, justify every decision in Markdown | 1 |

#### 🟩 Suhas — Encoding, Scaling, Splitting & Feature Engineering (B2 + B3)

| Sub-task | Description | Mark |
|----------|-------------|------|
| **B2 — Encoding, Scaling & Splitting** | Encode categoricals (BUILDINGID, FLOOR for regression), apply StandardScaler/MinMaxScaler **fitted on train only**, 80:20 train/test split (`random_state=42`), stratified split for classification | 1 |
| **B3 — Feature Engineering** | Create ≥1 engineered feature (`n_visible_aps`, `mean_rssi`, `max_rssi`, etc.) with written justification of why it may improve performance | 1 |

---

### Section C — Regression Track *(9 marks)* → Split 🟦 Balaji + 🟥 Ganesh

#### 🟦 Balaji — Regression Algorithms 1–5 (C1 half)

| # | Algorithm | Key Task |
|---|-----------|----------|
| 1 | **Linear Regression** | Train, predict, report R², RMSE, MAE; interpret coefficients |
| 2 | **Ridge Regression** | Train with default alpha, report metrics |
| 3 | **Lasso Regression** | Train, observe feature sparsity (how many coefficients → 0) |
| 4 | **ElasticNet Regression** | Train with default l1_ratio, report metrics |
| 5 | **Polynomial Regression** | `PolynomialFeatures(degree=2)` + `LinearRegression`; compare degree 2 vs 3 |

#### 🟥 Ganesh — Regression Algorithms 6–10 (C1 half)

| # | Algorithm | Key Task |
|---|-----------|----------|
| 6 | **Decision Tree Regressor** | Train, tune `max_depth`, plot feature importance |
| 7 | **Random Forest Regressor** | Train, tune `n_estimators`, report metrics |
| 8 | **Gradient Boosting Regressor** | sklearn GBM or XGBoost; tune `learning_rate` |
| 9 | **SVR** | Scale features first; tune `C` and `kernel` |
| 10 | **KNN Regressor** | Tune `k`; discuss impact of scaling |

#### 🟦 Balaji — Comparative Evaluation Table (C2) *(2 marks)*

| Sub-task | Description |
|----------|-------------|
| **C2 — Comparison Table** | Combine results from all 10 models into one Pandas DataFrame showing R², RMSE, MAE; rank by R² |

> Balaji collects Ganesh's model-6–10 results and merges them into the final table.

#### 🟥 Ganesh — Hyperparameter Tuning (C3) *(2 marks)*

| Sub-task | Description |
|----------|-------------|
| **C3 — Tuning** | Apply `GridSearchCV` or `RandomizedSearchCV` to ≥2 best models (e.g., Random Forest + Gradient Boosting); report best params and metric improvement |

#### 🟥 Ganesh — Regression Visualisations (C4) *(1 mark)*

| Sub-task | Description |
|----------|-------------|
| **Predicted vs. Actual plot** | Scatter plot for the best model |
| **Residual plot** | Residuals vs. predicted values |
| **Feature importance** | Bar chart for ≥1 tree-based model |

---

### Section D — Classification Part A *(3 marks)* → 🟩 Suhas

#### 🟩 Suhas — Classification Algorithms 1–5 (D1) *(2 marks)*

| # | Algorithm | Key Task |
|---|-----------|----------|
| 1 | **Logistic Regression** | Train on FLOOR target, report metrics |
| 2 | **K-Nearest Neighbors** | Tune `k`, discuss distance metrics |
| 3 | **Gaussian Naive Bayes** | Discuss conditional-independence assumption |
| 4 | **Decision Tree Classifier** | Tune `max_depth`, visualise tree |
| 5 | **SVM (SVC)** | Tune `C` and `kernel`; scale features |

#### 🟩 Suhas — Evaluation & Comparison Table (D2) *(1 mark)*

| Sub-task | Description |
|----------|-------------|
| **D2 — Metrics Table** | Report Accuracy, Precision, Recall, Weighted F1, and Confusion Matrix per algorithm; present as one comparison table |

---

### Section E — Presentation & Viva *(6 marks)* → 🟥 Ganesh (PPT lead) + All

#### 🟥 Ganesh — PPT Creation

| Slide Section | Content |
|---------------|---------|
| **Introduction** | Problem statement — indoor localization via Wi-Fi fingerprints |
| **Motivation** | Why Wi-Fi fingerprinting? GPS fails indoors; applications in hospitals, malls, warehouses |
| **Dataset** | UJIndoorLoc overview — 520 WAPs, 3 buildings, 4+ floors |
| **Approach** | Pipeline diagram: Data → EDA → Preprocessing → Models → Evaluation |
| **Regression Results** | Comparison table + best model plots |
| **Classification Results** | Comparison table + confusion matrices |
| **Conclusion** | Best models, key takeaways, what's next in Review 2 |

#### All Members — Viva *(5 marks)*

> Every member must be able to explain **any** part of the project. Each person should deeply know their own track:

| Member | Must-Know Deep Area | Also Prepared For |
|--------|-------------------|-------------------|
| 🟦 Balaji | EDA choices, cleaning decisions, Linear/Ridge/Lasso/ElasticNet/Poly math | Full pipeline overview |
| 🟩 Suhas | Scaling strategy, feature engineering rationale, all 5 classifiers, metrics | Full pipeline overview |
| 🟥 Ganesh | Tree-based & ensemble regressors, SVR/KNN, hyperparameter tuning, results interpretation | Full pipeline overview |

---

## Summary — Who Does What

### 🟦 Balaji (510)

```
1. Dataset loading & audit                    (A1)
2. All EDA visualisations + commentary        (A2, A3)
3. Data cleaning — sentinel, duplicates, outliers  (B1)
4. Regression algorithms 1–5                  (C1 half)
5. Regression comparison table                (C2)
```

### 🟩 Suhas (503)

```
1. Encoding, scaling & train/test split       (B2)
2. Feature engineering                        (B3)
3. Classification algorithms 1–5              (D1)
4. Classification metrics & comparison table  (D2)
```

### 🟥 Ganesh (519)

```
1. Regression algorithms 6–10                 (C1 half)
2. Hyperparameter tuning (≥2 models)          (C3)
3. Regression visualisations                  (C4)
4. PPT — intro, motivation, approach, results (E1)
5. Overall presentation narrative             (E1)
```

---

## Marks Ownership Map

| Rubric Section | Marks | Owner |
|----------------|-------|-------|
| A1 — Dataset audit | 1 | 🟦 Balaji |
| A2 — EDA visualisations | 2 | 🟦 Balaji |
| A3 — Insight commentary | 1 | 🟦 Balaji |
| B1 — Data cleaning | 1 | 🟦 Balaji |
| B2 — Encoding, scaling, split | 1 | 🟩 Suhas |
| B3 — Feature engineering | 1 | 🟩 Suhas |
| C1 — 10 regression algorithms | 4 | 🟦 Balaji (1–5) + 🟥 Ganesh (6–10) |
| C2 — Comparison table | 2 | 🟦 Balaji |
| C3 — Hyperparameter tuning | 2 | 🟥 Ganesh |
| C4 — Visualisations | 1 | 🟥 Ganesh |
| D1 — 5 classification algorithms | 2 | 🟩 Suhas |
| D2 — Classification evaluation | 1 | 🟩 Suhas |
| E1 — Presentation + Viva | 6 | 🟥 Ganesh (PPT) + All (viva) |
| | **25** | |

### Effective marks per person

| Member | Direct Ownership | Marks |
|--------|-----------------|-------|
| 🟦 Balaji | A1+A2+A3+B1+C1(half)+C2 | **~8** |
| 🟩 Suhas | B2+B3+D1+D2 | **~5** |
| 🟥 Ganesh | C1(half)+C3+C4+E1(PPT) | **~6** |
| All | Viva | **5** (shared) |
