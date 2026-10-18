"""Train champions for UJI + campus. Run: python scripts/train_champions.py"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "app_utils"))
from preprocess import engineer, clean

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)


def save(obj, name):
    joblib.dump(obj, os.path.join(MODELS, name))


def train_uji(sample=6000, cv_sample=2500):
    """UJI champions. CV on a cv_sample subset (5-fold, rubric C1) then finals fit on `sample`."""
    raw_path = os.path.join(BASE, "data", "raw", "trainingData.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Keep UJI CSVs in data/raw/.")
    tr = pd.read_csv(raw_path)
    waps = [c for c in tr.columns if c.startswith("WAP")]
    tr = clean(tr, waps, 100, -105)
    zv = tr[waps].var()
    keep = zv[zv > 0].index.tolist()
    tr = engineer(tr, keep)
    feats = keep + ["n_visible_aps", "mean_rssi", "max_rssi", "std_rssi", "rssi_range"]
    if len(tr) > sample:  # speed guard for laptop training
        tr = tr.sample(sample, random_state=42)
    X = tr[feats].values
    sc = StandardScaler().fit(X)
    Xs = np.nan_to_num(sc.transform(X))
    y_floor = tr["FLOOR"].values
    # --- 5-fold CV for the two best (rubric: mandatory) on a smaller slice for speed ---
    cv_idx = np.random.RandomState(42).choice(len(tr), min(cv_sample, len(tr)), replace=False)
    Xcv, ycv = Xs[cv_idx], y_floor[cv_idx]
    cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = SVC(kernel="rbf", C=10)
    f1 = cross_val_score(clf, Xcv, ycv, cv=cv5, scoring="f1_weighted", n_jobs=-1)
    print(f"UJI floor SVM 5-fold F1 (n={len(Xcv)}): {f1.mean():.4f} +- {f1.std():.4f}")
    from sklearn.model_selection import GridSearchCV as GS
    gs = GS(KNeighborsRegressor(weights="distance", n_jobs=-1),
            {"n_neighbors": [3, 5, 7]}, cv=3, scoring="r2", n_jobs=-1)
    gs.fit(Xs[cv_idx], tr["LONGITUDE"].values[cv_idx])
    print(f"UJI KNN lon GridSearch best: {gs.best_params_} CV R2={gs.best_score_:.4f}")
    best_k = gs.best_params_["n_neighbors"]
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_predict
    pred = cross_val_predict(KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1),
                             Xcv, tr["LONGITUDE"].values[cv_idx], cv=3, n_jobs=-1)
    yt = tr["LONGITUDE"].values[cv_idx]
    print(f"UJI KNN lon 3-fold: R2={r2_score(yt, pred):.4f} RMSE={np.sqrt(mean_squared_error(yt, pred)):.2f} MAE={mean_absolute_error(yt, pred):.2f}")
    clf.fit(Xs, y_floor)
    knn_lon = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1).fit(Xs, tr["LONGITUDE"].values)
    knn_lat = KNeighborsRegressor(n_neighbors=best_k, weights="distance", n_jobs=-1).fit(Xs, tr["LATITUDE"].values)
    save(sc, "uji_scaler.pkl")
    save(clf, "uji_svm_floor.pkl")
    save(knn_lon, "uji_knn_lon.pkl")
    save(knn_lat, "uji_knn_lat.pkl")
    json.dump({"feature_order": feats, "wap_keep": keep, "best_k": int(best_k)},
              open(os.path.join(MODELS, "uji_meta.json"), "w"))
    print(f"saved UJI models (best_k={best_k}).")


def train_campus():
    p = os.path.join(BASE, "data", "campus", "trainingData_campus.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run python scripts/build_campus_dataset.py first.")
    df = pd.read_csv(p)
    waps = [c for c in df.columns if c.startswith("WAP")]
    if not waps:
        raise ValueError("No WAPxxx columns in campus CSV. Rebuild via build_campus_dataset.py.")
    df = clean(df, waps, 100, -105)
    df = engineer(df, waps)
    feats = waps + ["n_visible_aps", "mean_rssi", "max_rssi", "std_rssi", "rssi_range"]
    X = df[feats].values
    sc = StandardScaler().fit(X)
    Xs = np.nan_to_num(sc.transform(X))
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    rb = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    s = cross_val_score(rb, Xs, df["block"].values, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"Campus block RF 3-fold F1: {s.mean():.4f} +- {s.std():.4f}")
    sf = cross_val_score(RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1),
                         Xs, df["floor"].values, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"Campus floor RF 3-fold F1: {sf.mean():.4f} +- {sf.std():.4f}")
    rb.fit(Xs, df["block"].values)
    rf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1).fit(Xs, df["floor"].values)
    rm = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1).fit(Xs, df["room"].values)
    rg = KNeighborsRegressor(n_neighbors=3, weights="distance", n_jobs=-1).fit(Xs, df[["latitude", "longitude"]].values)
    save(sc, "campus_scaler.pkl")
    save(rb, "campus_rf_block.pkl")
    save(rf, "campus_rf_floor.pkl")
    save(rm, "campus_knn_room.pkl")
    save(rg, "campus_knn_reg.pkl")
    cent = df.groupby("room")[["latitude", "longitude"]].mean().reset_index()
    cent.to_csv(os.path.join(MODELS, "room_centroids.csv"), index=False)
    wmap = json.load(open(os.path.join(BASE, "data", "campus", "wap_dictionary.json"), encoding="utf-8"))
    b2w = {v["bssid"].lower(): k for k, v in wmap.items()}
    json.dump({"feature_order": feats, "wap_cols": waps, "bssid_to_wap": b2w},
              open(os.path.join(MODELS, "campus_meta.json"), "w"))
    print(f"saved campus models. rooms={df['room'].nunique()} blocks={sorted(df['block'].unique())}")


if __name__ == "__main__":
    train_uji()
    train_campus()
