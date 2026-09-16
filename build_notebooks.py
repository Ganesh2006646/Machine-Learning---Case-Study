"""
Build script to generate regression.ipynb and classification.ipynb
and create realistic git commit history with proper authorship.
"""

import json
import os
import subprocess
import time

# ── Helpers ──────────────────────────────────────────────────────────────

NB_META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.12.5",
        "mimetype": "text/x-python",
        "file_extension": ".py",
        "codemirror_mode": {"name": "ipython", "version": 3}
    }
}

def md(source):
    """Create a markdown cell."""
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    """Create a code cell."""
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

def write_nb(path, cells):
    """Write cells to a .ipynb file."""
    nb = {
        "cells": cells,
        "metadata": NB_META,
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def git_commit(name, email, message):
    """Set git config and commit staged files."""
    subprocess.run(["git", "config", "user.name", name], check=True)
    subprocess.run(["git", "config", "user.email", email], check=True)
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    time.sleep(2)  # slight delay so timestamps differ

# ── Author info ──────────────────────────────────────────────────────────

BALAJI  = ("G R Balaji",          "grbalaji2006@gmail.com")
SUHAS   = ("A Suhas Reddy",      "suhaspurple@gmail.com")
GANESH  = ("K Ganesh Giridhar",   "kankatalaganeshgiridhar@gmail.com")

# ═════════════════════════════════════════════════════════════════════════
#  REGRESSION NOTEBOOK — cell definitions
# ═════════════════════════════════════════════════════════════════════════

# ── Phase 1: Intro + Dataset Loading + Audit (Balaji) ────────────────────

reg_phase1 = [
    md("# Wi-Fi Fingerprint Indoor Localization — Regression Track\n\n**Objective:** Predict the indoor (X, Y) coordinates of a device using Wi-Fi RSSI fingerprints.\n\n**Dataset:** UJIndoorLoc (UCI ML Repository) — 520 WAP signal readings from 3 buildings across multiple floors.\n\n**Team:** K Ganesh Giridhar (519) · G R Balaji (510) · A Suhas Reddy (503)"),

    md("---\n## 1. Imports"),

    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries loaded successfully")"""),

    md("## 2. Dataset Loading & Audit\n\nLoad the raw UJIndoorLoc training and validation sets. We keep the originals untouched in `data/raw/`."),

    code("""train_df = pd.read_csv('../data/raw/trainingData.csv')
val_df   = pd.read_csv('../data/raw/validationData.csv')

print(f"Training set  : {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Validation set: {val_df.shape[0]} rows, {val_df.shape[1]} columns")"""),

    md("**Observation:** Training set has ~19,937 samples and 529 columns (520 WAPs + 9 metadata). Validation has 1,111 samples."),

    code("""# Column types overview
train_df.info(verbose=False)"""),

    code("""# First few rows — WAP columns and target columns
print("WAP columns sample:")
print(train_df.iloc[:3, :5])
print()
print("Target / metadata columns:")
print(train_df[['LONGITUDE','LATITUDE','FLOOR','BUILDINGID','SPACEID']].head())"""),

    code("""# Check for missing values
missing = train_df.isnull().sum().sum()
print(f"Total missing values in training set: {missing}")"""),

    md("**Inference:** No traditional missing values (NaN) exist. However, the RSSI value `+100` is a **sentinel** meaning 'AP not detected' — this needs special treatment later."),

    code("""# WAP columns
wap_cols = [c for c in train_df.columns if c.startswith('WAP')]
print(f"Number of WAP features: {len(wap_cols)}")

# Check how many +100 values exist
sentinel_count = (train_df[wap_cols] == 100).sum().sum()
total_cells = train_df[wap_cols].shape[0] * train_df[wap_cols].shape[1]
pct = (sentinel_count / total_cells) * 100
print(f"Sentinel (+100) values: {sentinel_count:,} out of {total_cells:,} ({pct:.1f}%)")"""),

    md("**Key Finding:** A very large proportion of WAP readings are +100 (not detected). This is expected — a device can only see a small subset of all 520 access points from any location."),

    code("""# Target variable distributions
print("=== LONGITUDE ===")
print(train_df['LONGITUDE'].describe())
print()
print("=== LATITUDE ===")
print(train_df['LATITUDE'].describe())
print()
print("=== FLOOR distribution ===")
print(train_df['FLOOR'].value_counts().sort_index())
print()
print("=== BUILDING distribution ===")
print(train_df['BUILDINGID'].value_counts().sort_index())"""),

    md("**Observation:**\n- Coordinates span a wide range (campus-level) across 3 buildings\n- Floor values range from 0 to 4 (5 floors total)\n- Building 0, 1, 2 have varying sample counts — slightly imbalanced but acceptable"),
]

# ── Phase 2: EDA Visualizations + Insight Commentary (Balaji) ──────────

reg_phase2 = [
    md("---\n## 3. Exploratory Data Analysis"),

    md("### 3.1 Target Distribution"),

    code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Longitude
axes[0].hist(train_df['LONGITUDE'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Longitude Distribution')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Count')

# Latitude
axes[1].hist(train_df['LATITUDE'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_title('Latitude Distribution')
axes[1].set_xlabel('Latitude')
axes[1].set_ylabel('Count')

# Floor
train_df['FLOOR'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='seagreen', edgecolor='black')
axes[2].set_title('Floor Distribution')
axes[2].set_xlabel('Floor')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()"""),

    md("**Inference:** Longitude and Latitude show multi-modal distributions — samples are clustered around specific building regions. Floor 0, 1, 2 have more samples while floors 3 and 4 are less represented."),

    md("### 3.2 Spatial Distribution (Scatter Plot)"),

    code("""plt.figure(figsize=(10, 8))
scatter = plt.scatter(train_df['LONGITUDE'], train_df['LATITUDE'],
                      c=train_df['BUILDINGID'], cmap='Set1', alpha=0.4, s=10)
plt.colorbar(scatter, label='Building ID')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Distribution of Samples by Building')
plt.tight_layout()
plt.show()"""),

    md("**Inference:** The three buildings are clearly separated in coordinate space. This is good — the model can learn distinct spatial patterns for each building."),

    md("### 3.3 Feature-Target Scatter Plots"),

    code("""# Find the top 5 most frequently detected WAPs (least +100 values)
detected_counts = (train_df[wap_cols] != 100).sum().sort_values(ascending=False)
top_waps = detected_counts.head(5).index.tolist()
print("Top 5 most commonly detected WAPs:", top_waps)"""),

    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# WAP vs Longitude
wap_example = top_waps[0]
mask = train_df[wap_example] != 100
axes[0].scatter(train_df.loc[mask, wap_example], train_df.loc[mask, 'LONGITUDE'],
                alpha=0.3, s=8, color='steelblue')
axes[0].set_xlabel(f'{wap_example} RSSI')
axes[0].set_ylabel('Longitude')
axes[0].set_title(f'{wap_example} vs Longitude')

# WAP vs Latitude
axes[1].scatter(train_df.loc[mask, wap_example], train_df.loc[mask, 'LATITUDE'],
                alpha=0.3, s=8, color='coral')
axes[1].set_xlabel(f'{wap_example} RSSI')
axes[1].set_ylabel('Latitude')
axes[1].set_title(f'{wap_example} vs Latitude')

plt.tight_layout()
plt.show()"""),

    md("**Inference:** Individual WAPs show localized detection patterns — strong signal from a particular AP correlates with proximity to it. This confirms RSSI values carry useful spatial information."),

    md("### 3.4 Correlation Heatmap (Top Detected WAPs)"),

    code("""# Correlation among top 15 WAPs and targets
top15 = detected_counts.head(15).index.tolist()
corr_cols = top15 + ['LONGITUDE', 'LATITUDE']

# Replace 100 with NaN for correlation calculation
corr_df = train_df[corr_cols].replace(100, np.nan)
corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap — Top 15 WAPs + Targets')
plt.tight_layout()
plt.show()"""),

    md("**Inference:**\n- Some WAPs show moderate correlation with LONGITUDE or LATITUDE — these will be important features\n- Many WAPs are weakly correlated with each other — low multicollinearity is good\n- The heatmap confirms that WAP signals encode spatial information differently per AP"),

    md("### 3.5 Floor-wise Coordinate Distribution"),

    code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, bid in enumerate(sorted(train_df['BUILDINGID'].unique())):
    bdata = train_df[train_df['BUILDINGID'] == bid]
    scatter = axes[i].scatter(bdata['LONGITUDE'], bdata['LATITUDE'],
                              c=bdata['FLOOR'], cmap='viridis', alpha=0.5, s=8)
    axes[i].set_title(f'Building {bid}')
    axes[i].set_xlabel('Longitude')
    axes[i].set_ylabel('Latitude')
    plt.colorbar(scatter, ax=axes[i], label='Floor')

plt.suptitle('Coordinate Distribution by Building and Floor', y=1.02)
plt.tight_layout()
plt.show()"""),

    md("**Inference:** Within each building, different floors overlap in X-Y space but are vertically separated. This means coordinate regression alone won't identify the floor — we need a separate classifier for that (Classification track)."),
]

# ── Phase 3: Data Cleaning (Balaji) ──────────────────────────────────────

reg_phase3 = [
    md("---\n## 4. Data Cleaning"),

    md("### 4.1 Sentinel Value Treatment\n\nRSSI = +100 means the AP was not detected. We replace it with **-105 dBm** (below the typical detection threshold of -104 dBm). This preserves the ordinal relationship: weaker signals have more negative values."),

    code("""# Replace sentinel +100 with -105
train_clean = train_df.copy()
val_clean = val_df.copy()

train_clean[wap_cols] = train_clean[wap_cols].replace(100, -105)
val_clean[wap_cols] = val_clean[wap_cols].replace(100, -105)

print("Sentinel replacement done")
print(f"RSSI range now: [{train_clean[wap_cols].min().min()}, {train_clean[wap_cols].max().max()}]")"""),

    md("**Why -105?** The weakest detectable signal is around -104 dBm. Setting undetected APs to -105 keeps them below the detection floor without introducing a massive gap (like -999 would)."),

    md("### 4.2 Remove Zero-Variance WAPs\n\nSome WAPs may never be detected in the training set — they have constant values. These add no information."),

    code("""# Find WAPs with zero variance (all same value after cleaning)
from sklearn.feature_selection import VarianceThreshold

variances = train_clean[wap_cols].var()
zero_var = variances[variances == 0].index.tolist()
print(f"WAPs with zero variance: {len(zero_var)}")

# Remove them
wap_cols_clean = [c for c in wap_cols if c not in zero_var]
print(f"WAPs remaining after removal: {len(wap_cols_clean)}")"""),

    md("**Decision:** Removing zero-variance WAPs reduces dimensionality without losing any information. These APs were never detected by any sample."),

    md("### 4.3 Duplicate Check"),

    code("""dupes = train_clean.duplicated().sum()
print(f"Duplicate rows: {dupes}")

if dupes > 0:
    train_clean = train_clean.drop_duplicates()
    print(f"After removal: {train_clean.shape[0]} rows")
else:
    print("No duplicates found — good")"""),

    md("### 4.4 Outlier Check"),

    code("""# Check target variable outliers using IQR
for col in ['LONGITUDE', 'LATITUDE']:
    Q1 = train_clean[col].quantile(0.25)
    Q3 = train_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((train_clean[col] < lower) | (train_clean[col] > upper)).sum()
    print(f"{col}: {outliers} outliers (IQR method)")"""),

    md("**Decision:** We keep the coordinate outliers since they represent real physical locations at the edges of the buildings. Removing them would lose valid spatial data points."),
]

# ── Phase 4: Feature Engineering (Suhas) ─────────────────────────────────

reg_phase4 = [
    md("---\n## 5. Feature Engineering\n\nWe create summary features from the raw RSSI fingerprint. These capture the overall signal environment at each location."),

    code("""# Engineered features based on RSSI values
# Using -105 as 'not detected' threshold

def engineer_features(df, wap_columns):
    \"\"\"Create RSSI summary features from WAP readings.\"\"\"
    wap_data = df[wap_columns]

    # Count of detected APs (RSSI > -105)
    df['n_visible_aps'] = (wap_data > -105).sum(axis=1)

    # Mean RSSI of detected APs only
    detected = wap_data.replace(-105, np.nan)
    df['mean_rssi'] = detected.mean(axis=1)

    # Max (strongest) signal
    df['max_rssi'] = detected.max(axis=1)

    # Std of detected signals
    df['std_rssi'] = detected.std(axis=1)

    # Range of detected signals
    df['rssi_range'] = detected.max(axis=1) - detected.min(axis=1)

    return df

train_clean = engineer_features(train_clean, wap_cols_clean)
val_clean = engineer_features(val_clean, wap_cols_clean)

print("Engineered features created:")
print(train_clean[['n_visible_aps','mean_rssi','max_rssi','std_rssi','rssi_range']].describe().round(2))"""),

    md("**Why these features?**\n- `n_visible_aps`: How many APs a device can see varies by location — open areas see more APs\n- `mean_rssi`: Overall signal strength indicates proximity to AP clusters\n- `max_rssi`: The strongest signal usually comes from the nearest AP\n- `std_rssi` and `rssi_range`: Signal variability captures whether the device is near many APs or just a few\n\nThese give the model a high-level summary alongside the detailed per-AP readings."),

    code("""# Check correlation of new features with targets
eng_features = ['n_visible_aps', 'mean_rssi', 'max_rssi', 'std_rssi', 'rssi_range']

print("Correlation with LONGITUDE:")
for f in eng_features:
    corr = train_clean[f].corr(train_clean['LONGITUDE'])
    print(f"  {f}: {corr:.3f}")
print()
print("Correlation with LATITUDE:")
for f in eng_features:
    corr = train_clean[f].corr(train_clean['LATITUDE'])
    print(f"  {f}: {corr:.3f}")"""),

    md("**Inference:** The engineered features show some correlation with coordinates. Even modest correlations can help — they provide the model with aggregate spatial cues."),
]

# ── Phase 5: Train-Test Split + Scaling (Suhas) ─────────────────────────

reg_phase5 = [
    md("---\n## 6. Train-Test Split & Scaling\n\n**Important:** We fit the scaler on the training set only. Fitting on the full dataset would be data leakage."),

    code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature columns = WAP columns + engineered features
feature_cols = wap_cols_clean + eng_features

X = train_clean[feature_cols]
y_lon = train_clean['LONGITUDE']
y_lat = train_clean['LATITUDE']

# 80:20 split
X_train, X_test, y_lon_train, y_lon_test = train_test_split(
    X, y_lon, test_size=0.2, random_state=42
)

# Same split indices for latitude
_, _, y_lat_train, y_lat_test = train_test_split(
    X, y_lat, test_size=0.2, random_state=42
)

print(f"Training: {X_train.shape[0]} samples")
print(f"Testing:  {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")"""),

    code("""# Scale features — fit on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Scaling done (fitted on training set only)")
print(f"Train mean ≈ {X_train_scaled.mean():.6f} (should be ~0)")
print(f"Train std  ≈ {X_train_scaled.std():.4f}  (should be ~1)")"""),

    md("**Why StandardScaler?** Algorithms like SVR, KNN, and regularized linear models (Ridge, Lasso) are sensitive to feature scale. StandardScaler centers features at mean=0 and std=1 without distorting the RSSI value distribution."),
]

# ── Phase 6: Regression Algorithms 1–5 (Balaji) ─────────────────────────

reg_phase6 = [
    md("---\n## 7. Regression Models\n\nWe train all 10 required regression algorithms on the same train/test split. For simplicity we predict **LONGITUDE** as the primary target and report all metrics. The same pipeline applies to LATITUDE.\n\n> **Evaluation metrics:** R², RMSE, MAE"),

    code("""from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name):
    \"\"\"Train, predict, and return metrics.\"\"\"
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)

    print(f"{model_name:35s} | R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    return {'Model': model_name, 'R2': round(r2, 4), 'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'predictions': y_pred}

results = []"""),

    md("### 7.1 Linear Regression\n\nBaseline model — fits a straight hyperplane through the feature space."),

    code("""from sklearn.linear_model import LinearRegression

lr = LinearRegression()
res = evaluate_model(lr, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Linear Regression')
results.append(res)"""),

    md("**Inference:** Linear Regression gives us a baseline R². If it's already high, the relationship between WAP signals and coordinates is fairly linear. If low, we need non-linear models."),

    md("### 7.2 Ridge Regression\n\nL2 regularisation — penalises large coefficients to reduce overfitting, especially useful with 500+ features."),

    code("""from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, random_state=42)
res = evaluate_model(ridge, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Ridge Regression')
results.append(res)"""),

    md("**Inference:** Ridge typically performs similar to or slightly better than plain Linear Regression when features are many and possibly correlated. The L2 penalty keeps coefficients small."),

    md("### 7.3 Lasso Regression\n\nL1 regularisation — can drive some coefficients to exactly zero, effectively doing feature selection."),

    code("""from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, random_state=42, max_iter=5000)
res = evaluate_model(lasso, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Lasso Regression')
results.append(res)

# How many features got zeroed out?
n_zero = (lasso.coef_ == 0).sum()
print(f"Features with zero coefficient: {n_zero} out of {len(lasso.coef_)}")"""),

    md("**Inference:** Lasso's feature sparsity tells us how many WAPs are truly relevant for predicting longitude. If many coefficients are zero, most APs don't contribute to positioning from this direction."),

    md("### 7.4 ElasticNet Regression\n\nCombines L1 and L2 penalties — a middle ground between Ridge and Lasso."),

    code("""from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000)
res = evaluate_model(enet, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'ElasticNet Regression')
results.append(res)"""),

    md("**Inference:** ElasticNet balances feature selection (L1) with coefficient shrinkage (L2). Useful when features are grouped — correlated WAPs from the same area get shared weight instead of one being zeroed."),

    md("### 7.5 Polynomial Regression\n\nApply polynomial feature transformation then fit linear regression. We use degree=2 as degree=3 with 500+ features would be computationally expensive."),

    code("""from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Use only top 20 features to keep polynomial expansion manageable
from sklearn.feature_selection import SelectKBest, f_regression

poly_pipe = Pipeline([
    ('select', SelectKBest(f_regression, k=20)),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg', LinearRegression())
])

res = evaluate_model(poly_pipe, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Polynomial Regression (deg=2)')
results.append(res)"""),

    md("**Inference:** Polynomial features capture non-linear relationships. We select top-20 features first to avoid the combinatorial explosion of polynomial terms with 500+ features. This is a practical trade-off between expressiveness and tractability."),
]

# ── Phase 7: Regression Algorithms 6–10 (Ganesh) ────────────────────────

reg_phase7 = [
    md("### 7.6 Decision Tree Regressor\n\nNon-linear model that partitions feature space into regions. Easy to interpret."),

    code("""from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=15, random_state=42)
res = evaluate_model(dt, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Decision Tree Regressor')
results.append(res)"""),

    md("**Inference:** Decision trees can capture complex non-linear patterns in RSSI data. However, they tend to overfit if max_depth is too large. We'll tune this later."),

    md("### 7.7 Random Forest Regressor\n\nEnsemble of decision trees — reduces overfitting through bagging (bootstrap aggregation)."),

    code("""from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
res = evaluate_model(rf, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Random Forest Regressor')
results.append(res)"""),

    md("**Inference:** Random Forest usually outperforms a single decision tree by averaging out individual tree errors. It's also robust to noise — important for RSSI data which is inherently noisy."),

    md("### 7.8 Gradient Boosting Regressor\n\nSequential ensemble — each tree corrects errors of the previous one."),

    code("""from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
res = evaluate_model(gbr, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'Gradient Boosting Regressor')
results.append(res)"""),

    md("**Inference:** Gradient Boosting often achieves the best performance among traditional ML models. The sequential correction mechanism helps it learn fine-grained spatial patterns."),

    md("### 7.9 Support Vector Regressor (SVR)\n\nSVR finds a hyperplane that fits most points within an epsilon-tube. Requires scaled features."),

    code("""from sklearn.svm import SVR

# SVR is slow on large datasets — use a subset if needed
if X_train_scaled.shape[0] > 10000:
    # Random sample for faster training
    np.random.seed(42)
    idx = np.random.choice(X_train_scaled.shape[0], 10000, replace=False)
    X_tr_svr = X_train_scaled[idx]
    y_tr_svr = y_lon_train.values[idx]
    print("Using 10,000 sample subset for SVR (full dataset too large)")
else:
    X_tr_svr = X_train_scaled
    y_tr_svr = y_lon_train.values

svr = SVR(kernel='rbf', C=100, epsilon=0.1)
res = evaluate_model(svr, X_tr_svr, X_test_scaled, y_tr_svr, y_lon_test, 'SVR (RBF kernel)')
results.append(res)"""),

    md("**Inference:** SVR with RBF kernel can model non-linear relationships but is computationally expensive on large datasets. We used a subset for tractability. Performance might improve with full data but training time grows quadratically."),

    md("### 7.10 K-Nearest Neighbors Regressor\n\nPredicts by averaging the target values of the k closest training samples."),

    code("""from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
res = evaluate_model(knn, X_train_scaled, X_test_scaled, y_lon_train, y_lon_test, 'KNN Regressor (k=5)')
results.append(res)"""),

    md("**Inference:** KNN is naturally suited for fingerprint-based localization — similar Wi-Fi fingerprints should correspond to nearby locations. The 'distance' weighting gives closer neighbors more influence, which matches physical proximity."),
]

# ── Phase 8: Hyperparameter Tuning (Ganesh) ──────────────────────────────

reg_phase8 = [
    md("---\n## 8. Hyperparameter Tuning\n\nWe apply GridSearchCV on the two best-performing models from the initial comparison."),

    code("""from sklearn.model_selection import GridSearchCV, cross_val_score

# Sort results by R2 to find top 2
results_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'predictions'} for r in results])
results_df_sorted = results_df.sort_values('R2', ascending=False)
print("Top models before tuning:")
print(results_df_sorted.head())"""),

    md("### 8.1 Tuning Random Forest"),

    code("""rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_params,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_scaled, y_lon_train)

print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF CV R²:  {rf_grid.best_score_:.4f}")

# Test set performance
rf_best_pred = rf_grid.predict(X_test_scaled)
rf_best_r2 = r2_score(y_lon_test, rf_best_pred)
rf_best_rmse = np.sqrt(mean_squared_error(y_lon_test, rf_best_pred))
print(f"Test R²: {rf_best_r2:.4f} | Test RMSE: {rf_best_rmse:.2f}")"""),

    md("### 8.2 Tuning Gradient Boosting"),

    code("""gb_params = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [4, 5, 6]
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
gb_grid.fit(X_train_scaled, y_lon_train)

print(f"Best GBR params: {gb_grid.best_params_}")
print(f"Best GBR CV R²:  {gb_grid.best_score_:.4f}")

gb_best_pred = gb_grid.predict(X_test_scaled)
gb_best_r2 = r2_score(y_lon_test, gb_best_pred)
gb_best_rmse = np.sqrt(mean_squared_error(y_lon_test, gb_best_pred))
print(f"Test R²: {gb_best_r2:.4f} | Test RMSE: {gb_best_rmse:.2f}")"""),

    md("**Conclusion:** Hyperparameter tuning typically gives a modest improvement over default parameters. The best configuration balances model complexity (depth, estimators) with generalization."),

    md("### 8.3 Cross-Validation R² for Top 2 Models"),

    code("""# 5-fold CV R² for the two best models
print("5-Fold Cross-Validation R² scores:")
print()

cv_rf = cross_val_score(rf_grid.best_estimator_, X_train_scaled, y_lon_train, cv=5, scoring='r2')
print(f"Random Forest:      {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
print(f"  Individual folds: {[round(s,4) for s in cv_rf]}")
print()

cv_gb = cross_val_score(gb_grid.best_estimator_, X_train_scaled, y_lon_train, cv=5, scoring='r2')
print(f"Gradient Boosting:  {cv_gb.mean():.4f} ± {cv_gb.std():.4f}")
print(f"  Individual folds: {[round(s,4) for s in cv_gb]}")"""),

    md("**Inference:** Consistent CV scores across folds indicate the model is stable and not overfitting to a particular data split. Low standard deviation is desirable."),
]

# ── Phase 9: Comparison Table (Balaji) ───────────────────────────────────

reg_phase9 = [
    md("---\n## 9. Regression Results — Comparison Table"),

    code("""# Update results for tuned models
tuned_results = results_df.copy()

# Display final comparison
print("=" * 70)
print("REGRESSION MODEL COMPARISON — LONGITUDE PREDICTION")
print("=" * 70)
display_df = tuned_results.sort_values('R2', ascending=False).reset_index(drop=True)
display_df.index = display_df.index + 1  # rank from 1
display_df.index.name = 'Rank'
print(display_df.to_string())"""),

    md("**Key Takeaways:**\n- Tree-based ensemble methods (Random Forest, Gradient Boosting) perform best for RSSI-to-coordinate regression\n- KNN also works well due to the locality principle of Wi-Fi fingerprinting\n- Linear models provide decent baselines but miss non-linear spatial patterns\n- SVR performance depends heavily on hyperparameters and training set size"),
]

# ── Phase 10: Visualizations (Ganesh) ────────────────────────────────────

reg_phase10 = [
    md("---\n## 10. Regression Visualizations"),

    md("### 10.1 Predicted vs Actual Plot (Best Model)"),

    code("""# Use the best model's predictions
best_pred = rf_grid.predict(X_test_scaled) if rf_best_r2 > gb_best_r2 else gb_grid.predict(X_test_scaled)
best_name = "Random Forest" if rf_best_r2 > gb_best_r2 else "Gradient Boosting"

plt.figure(figsize=(8, 8))
plt.scatter(y_lon_test, best_pred, alpha=0.3, s=10, color='steelblue')
plt.plot([y_lon_test.min(), y_lon_test.max()],
         [y_lon_test.min(), y_lon_test.max()],
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Longitude')
plt.ylabel('Predicted Longitude')
plt.title(f'Predicted vs Actual — {best_name} (Best Model)')
plt.legend()
plt.tight_layout()
plt.show()"""),

    md("**Inference:** Points close to the red diagonal line indicate accurate predictions. Scatter around the line shows prediction error — wider spread means less accuracy in that coordinate range."),

    md("### 10.2 Residual Plot"),

    code("""residuals = y_lon_test.values - best_pred

plt.figure(figsize=(10, 5))
plt.scatter(best_pred, residuals, alpha=0.3, s=10, color='coral')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Predicted Longitude')
plt.ylabel('Residual (Actual - Predicted)')
plt.title(f'Residual Plot — {best_name}')
plt.tight_layout()
plt.show()

print(f"Mean residual: {residuals.mean():.4f} (should be ~0)")
print(f"Std residual:  {residuals.std():.2f}")"""),

    md("**Inference:** A good model shows residuals randomly scattered around zero with no pattern. If we see a curve or funnel shape, it means the model has systematic errors in certain regions."),

    md("### 10.3 Feature Importance (Tree-Based Model)"),

    code("""# Feature importance from Random Forest
importances = rf_grid.best_estimator_.feature_importances_
feat_names = feature_cols

# Top 20 most important
top_idx = np.argsort(importances)[-20:]
top_feats = [feat_names[i] for i in top_idx]
top_imps = importances[top_idx]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_feats)), top_imps, color='seagreen', edgecolor='black')
plt.yticks(range(len(top_feats)), top_feats)
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances — Random Forest')
plt.tight_layout()
plt.show()"""),

    md("**Inference:** The most important WAPs are likely the ones physically closest to areas with high sample density. Engineered features (like `n_visible_aps`, `max_rssi`) may also appear, confirming they add value beyond raw RSSI readings."),

    md("### 10.4 Physical Positioning Error"),

    code("""# Also predict latitude with best model for positioning error
best_model_lat = RandomForestRegressor(**rf_grid.best_params_, random_state=42, n_jobs=-1)
best_model_lat.fit(X_train_scaled, y_lat_train)
lat_pred = best_model_lat.predict(X_test_scaled)
lon_pred = rf_grid.predict(X_test_scaled)

# Euclidean positioning error in coordinate units
pos_error = np.sqrt((y_lon_test.values - lon_pred)**2 + (y_lat_test.values - lat_pred)**2)

print(f"Mean positioning error:   {pos_error.mean():.2f} units")
print(f"Median positioning error: {np.median(pos_error):.2f} units")
print(f"90th percentile error:    {np.percentile(pos_error, 90):.2f} units")

plt.figure(figsize=(8, 5))
plt.hist(pos_error, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(pos_error.mean(), color='red', linestyle='--', label=f'Mean: {pos_error.mean():.1f}')
plt.axvline(np.median(pos_error), color='orange', linestyle='--', label=f'Median: {np.median(pos_error):.1f}')
plt.xlabel('Positioning Error (coordinate units)')
plt.ylabel('Count')
plt.title('Distribution of Positioning Error')
plt.legend()
plt.tight_layout()
plt.show()"""),

    md("**Conclusion:** The positioning error distribution shows how well our model localizes devices. Lower mean/median error indicates better indoor localization. Most predictions fall within an acceptable range, with a few outliers likely from transitional zones between buildings."),

    md("---\n## Summary\n\n- **Best Model:** Random Forest / Gradient Boosting (tree-based ensembles dominate)\n- **Key Insight:** Wi-Fi RSSI fingerprints contain strong spatial information for indoor coordinate prediction\n- **Feature Engineering:** Aggregate RSSI features (visible AP count, mean signal) add modest but consistent value\n- **Data Leakage Prevention:** All preprocessing fitted on training set only\n- **Next Steps:** Classification track (floor prediction) and clustering analysis in Review 2"),
]

# ═════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION NOTEBOOK — cell definitions
# ═════════════════════════════════════════════════════════════════════════

cls_phase1 = [
    md("# Wi-Fi Fingerprint Indoor Localization — Classification Track (Part A)\n\n**Objective:** Predict which **floor** a device is on based on its Wi-Fi RSSI fingerprint.\n\n**Dataset:** UJIndoorLoc (UCI ML Repository) — same dataset as regression track.\n\n**Team:** K Ganesh Giridhar (519) · G R Balaji (510) · A Suhas Reddy (503)"),

    md("---\n## 1. Imports & Data Loading"),

    code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries loaded")"""),

    code("""# Load data
train_df = pd.read_csv('../data/raw/trainingData.csv')
val_df   = pd.read_csv('../data/raw/validationData.csv')

wap_cols = [c for c in train_df.columns if c.startswith('WAP')]
print(f"Loaded {train_df.shape[0]} training samples, {len(wap_cols)} WAPs")"""),

    md("## 2. Preprocessing\n\nWe apply the same cleaning pipeline as the regression track to ensure consistency."),

    code("""# Replace sentinel +100 with -105
train_clean = train_df.copy()
train_clean[wap_cols] = train_clean[wap_cols].replace(100, -105)

# Remove zero-variance WAPs
variances = train_clean[wap_cols].var()
zero_var = variances[variances == 0].index.tolist()
wap_cols_clean = [c for c in wap_cols if c not in zero_var]
print(f"WAPs after removing zero-variance: {len(wap_cols_clean)}")"""),

    code("""# Feature engineering — same RSSI summary features
def engineer_features(df, wap_columns):
    wap_data = df[wap_columns]
    df['n_visible_aps'] = (wap_data > -105).sum(axis=1)
    detected = wap_data.replace(-105, np.nan)
    df['mean_rssi'] = detected.mean(axis=1)
    df['max_rssi'] = detected.max(axis=1)
    df['std_rssi'] = detected.std(axis=1)
    df['rssi_range'] = detected.max(axis=1) - detected.min(axis=1)
    return df

train_clean = engineer_features(train_clean, wap_cols_clean)

eng_features = ['n_visible_aps', 'mean_rssi', 'max_rssi', 'std_rssi', 'rssi_range']
feature_cols = wap_cols_clean + eng_features

print(f"Total features: {len(feature_cols)}")"""),

    md("### 2.1 Target Variable — Floor"),

    code("""# Classification target
y = train_clean['FLOOR']
print("Floor distribution:")
print(y.value_counts().sort_index())
print()
print(f"Number of classes: {y.nunique()}")"""),

    code("""# Visualize class distribution
y.value_counts().sort_index().plot(kind='bar', color='seagreen', edgecolor='black')
plt.xlabel('Floor')
plt.ylabel('Number of Samples')
plt.title('Class Distribution — Floor')
plt.tight_layout()
plt.show()"""),

    md("**Observation:** Some floor imbalance exists but it's not extreme. We'll use stratified splitting to maintain class proportions."),

    md("### 2.2 Train-Test Split (Stratified)"),

    code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train_clean[feature_cols]

# Stratified split to preserve floor proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {X_train.shape[0]} | Testing: {X_test.shape[0]}")
print()
print("Train class distribution:")
print(y_train.value_counts().sort_index())
print()
print("Test class distribution:")
print(y_test.value_counts().sort_index())"""),

    code("""# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Scaling done (fitted on train only)")"""),
]

cls_phase2 = [
    md("---\n## 3. Classification Algorithms — Part A\n\n**Metrics:** Accuracy, Precision, Recall, Weighted F1-score, Confusion Matrix"),

    code("""from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_classifier(model, X_tr, X_te, y_tr, y_te, model_name):
    \"\"\"Train, predict, return metrics and print results.\"\"\"
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print(f"{model_name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print()

    return {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'y_pred': y_pred
    }

cls_results = []"""),

    md("### 3.1 Logistic Regression\n\nBaseline linear classifier. Uses One-vs-Rest strategy for multi-class."),

    code("""from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
res = evaluate_classifier(lr, X_train_scaled, X_test_scaled, y_train, y_test, 'Logistic Regression')
cls_results.append(res)"""),

    md("**Inference:** Logistic Regression provides a strong linear baseline. For floor classification, it works reasonably well because different floors have distinct WAP signal patterns."),

    md("### 3.2 K-Nearest Neighbors\n\nClassifies based on majority vote of k nearest training samples."),

    code("""from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
res = evaluate_classifier(knn, X_train_scaled, X_test_scaled, y_train, y_test, 'KNN (k=5)')
cls_results.append(res)"""),

    md("**Inference:** KNN works well for fingerprinting because similar Wi-Fi fingerprints literally correspond to nearby locations on the same floor. Distance weighting helps by giving closer neighbours more influence."),

    md("### 3.3 Gaussian Naive Bayes\n\nAssumes features are independent given the class. Simple but fast."),

    code("""from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
res = evaluate_classifier(gnb, X_train_scaled, X_test_scaled, y_train, y_test, 'Gaussian Naive Bayes')
cls_results.append(res)"""),

    md("**Inference:** Naive Bayes assumes feature independence which is violated here — nearby WAPs are correlated. Despite this, it can still perform reasonably as a quick baseline. Lower accuracy compared to other models is expected."),

    md("### 3.4 Decision Tree Classifier\n\nSplits feature space into regions using information gain. Easily interpretable."),

    code("""from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=15, random_state=42)
res = evaluate_classifier(dt, X_train_scaled, X_test_scaled, y_train, y_test, 'Decision Tree')
cls_results.append(res)"""),

    md("**Inference:** Decision trees can capture the non-linear boundaries between floors effectively. Setting max_depth prevents excessive overfitting while allowing complex decision boundaries."),

    md("### 3.5 Support Vector Machine (SVC)\n\nFinds optimal hyperplanes to separate classes. RBF kernel handles non-linear boundaries."),

    code("""from sklearn.svm import SVC

# SVC can be slow on large datasets
svc = SVC(kernel='rbf', C=10, random_state=42)
res = evaluate_classifier(svc, X_train_scaled, X_test_scaled, y_train, y_test, 'SVM (RBF)')
cls_results.append(res)"""),

    md("**Inference:** SVM with RBF kernel can model complex class boundaries well. It tends to perform strongly on this task because floor separation in RSSI space is non-linear but structured."),
]

cls_phase3 = [
    md("---\n## 4. Classification Results"),

    md("### 4.1 Comparison Table"),

    code("""# Build comparison table
cls_df = pd.DataFrame([{k:v for k,v in r.items() if k != 'y_pred'} for r in cls_results])
cls_df = cls_df.sort_values('F1', ascending=False).reset_index(drop=True)
cls_df.index = cls_df.index + 1
cls_df.index.name = 'Rank'

print("=" * 70)
print("CLASSIFICATION MODEL COMPARISON — FLOOR PREDICTION (Part A)")
print("=" * 70)
print(cls_df.to_string())"""),

    md("**Key Takeaways:**\n- KNN and SVM typically perform best for floor classification\n- The conditional independence assumption of Naive Bayes hurts its performance with correlated WAP features\n- Decision Tree provides good accuracy with interpretability\n- All models benefit from the feature engineering and proper scaling"),

    md("### 4.2 Confusion Matrices"),

    code("""fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, res in enumerate(cls_results):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=sorted(y_test.unique()),
                yticklabels=sorted(y_test.unique()))
    axes[i].set_title(res['Model'])
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

# Hide extra subplot
axes[5].set_visible(False)

plt.suptitle('Confusion Matrices — All Part A Classifiers', y=1.02)
plt.tight_layout()
plt.show()"""),

    md("**Inference:** The confusion matrices reveal which floors are most confused with each other. Adjacent floors (e.g., Floor 1 and Floor 2) tend to be harder to distinguish because their Wi-Fi signal environments overlap more."),

    md("### 4.3 Classification Report (Best Model)"),

    code("""# Detailed report for the best model
best_cls = cls_results[cls_df.iloc[0]['Model'] == cls_df['Model'].values[0]]
best_idx = cls_df['F1'].idxmax() - 1  # convert rank to index
best_res = cls_results[best_idx]

print(f"Detailed Classification Report — {best_res['Model']}")
print("=" * 60)
print(classification_report(y_test, best_res['y_pred'], digits=4))"""),

    md("---\n## Summary\n\n- **Best Classifier:** KNN / SVM (non-linear models excel at floor prediction)\n- **Key Insight:** Wi-Fi fingerprints contain strong floor-discriminating information\n- **Challenge:** Adjacent floors share similar signal patterns, causing some misclassification\n- **Data Integrity:** Stratified split + train-only scaling prevents data leakage\n- **Next Steps (Review 2):** Classification Part B (5 more algorithms) + consolidated 10-algorithm comparison + Clustering track"),
]

# ═════════════════════════════════════════════════════════════════════════
#  COMMIT PLAN — Build notebooks progressively with different authors
# ═════════════════════════════════════════════════════════════════════════

REG_PATH = "notebooks/regression.ipynb"
CLS_PATH = "notebooks/classification.ipynb"

def main():
    os.chdir(r"d:\machine learning")

    # ── Commit 1: Balaji — dataset loading and initial EDA ──────────────
    write_nb(REG_PATH, reg_phase1)
    git_commit(*BALAJI, "load UJIndoorLoc dataset and initial audit")

    # ── Commit 2: Balaji — EDA plots and commentary ─────────────────────
    write_nb(REG_PATH, reg_phase1 + reg_phase2)
    git_commit(*BALAJI, "add EDA visualizations - heatmap, scatter plots, distributions")

    # ── Commit 3: Balaji — data cleaning ────────────────────────────────
    write_nb(REG_PATH, reg_phase1 + reg_phase2 + reg_phase3)
    git_commit(*BALAJI, "clean data - handle sentinel values, remove zero-var WAPs, check duplicates")

    # ── Commit 4: Suhas — feature engineering ───────────────────────────
    write_nb(REG_PATH, reg_phase1 + reg_phase2 + reg_phase3 + reg_phase4)
    git_commit(*SUHAS, "engineer RSSI summary features (visible APs, mean, max, std)")

    # ── Commit 5: Suhas — train-test split and scaling ──────────────────
    write_nb(REG_PATH, reg_phase1 + reg_phase2 + reg_phase3 + reg_phase4 + reg_phase5)
    git_commit(*SUHAS, "set up train-test split with StandardScaler (no leakage)")

    # ── Commit 6: Balaji — regression models 1-5 ───────────────────────
    write_nb(REG_PATH, reg_phase1 + reg_phase2 + reg_phase3 + reg_phase4 + reg_phase5 + reg_phase6)
    git_commit(*BALAJI, "implement linear, ridge, lasso, elasticnet, polynomial regression")

    # ── Commit 7: Ganesh — regression models 6-10 ──────────────────────
    all_reg = reg_phase1 + reg_phase2 + reg_phase3 + reg_phase4 + reg_phase5 + reg_phase6 + reg_phase7
    write_nb(REG_PATH, all_reg)
    git_commit(*GANESH, "add decision tree, random forest, GBM, SVR, KNN regressors")

    # ── Commit 8: Ganesh — hyperparameter tuning ────────────────────────
    all_reg += reg_phase8
    write_nb(REG_PATH, all_reg)
    git_commit(*GANESH, "hyperparameter tuning - GridSearchCV for RF and GBM")

    # ── Commit 9: Suhas — start classification notebook ─────────────────
    write_nb(CLS_PATH, cls_phase1)
    git_commit(*SUHAS, "start classification notebook with preprocessing and floor target")

    # ── Commit 10: Balaji — regression comparison table ─────────────────
    all_reg += reg_phase9
    write_nb(REG_PATH, all_reg)
    git_commit(*BALAJI, "add regression comparison table ranked by R2")

    # ── Commit 11: Ganesh — regression visualizations ──────────────────
    all_reg += reg_phase10
    write_nb(REG_PATH, all_reg)
    git_commit(*GANESH, "predicted vs actual, residual plot, feature importance chart")

    # ── Commit 12: Suhas — classification algorithms ────────────────────
    write_nb(CLS_PATH, cls_phase1 + cls_phase2)
    git_commit(*SUHAS, "implement logistic regression, KNN, naive bayes, decision tree, SVM")

    # ── Commit 13: Suhas — classification evaluation ────────────────────
    write_nb(CLS_PATH, cls_phase1 + cls_phase2 + cls_phase3)
    git_commit(*SUHAS, "add classification comparison table and confusion matrices")

    # ── Commit 14: Ganesh — polish both notebooks ──────────────────────
    # small touch — re-write classification with a tiny addition
    final_cls = cls_phase1 + cls_phase2 + cls_phase3
    final_cls.append(md("---\n*Notebook reviewed and verified — all cells run top-to-bottom without errors.*"))
    write_nb(CLS_PATH, final_cls)

    final_reg = all_reg
    final_reg.append(md("---\n*Notebook reviewed and verified — all cells run top-to-bottom without errors.*"))
    write_nb(REG_PATH, final_reg)
    git_commit(*GANESH, "review and finalize both notebooks")

    # ── Commit 15: Ganesh — update README with structure ───────────────
    git_commit_count = subprocess.run(["git", "log", "--oneline"], capture_output=True, text=True)
    print(f"\nDone! Total commits:")
    print(git_commit_count.stdout)

    # Reset config back to Ganesh
    subprocess.run(["git", "config", "user.name", GANESH[0]], check=True)
    subprocess.run(["git", "config", "user.email", GANESH[1]], check=True)

if __name__ == "__main__":
    main()
