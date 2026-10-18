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


def train_uji(sample=6000):
    tr = pd.read_csv(os.path.join(BASE, "data", "raw", "trainingData.csv"))
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
    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    clf = SVC(kernel="rbf", C=10)
    f1 = cross_val_score(clf, Xs, tr["FLOOR"].values, cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"UJI floor SVM 3-fold F1: {f1.mean():.4f} +- {f1.std():.4f}")
    clf.fit(Xs, tr["FLOOR"].values)
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
    knn_lon = knn.fit(Xs, tr["LONGITUDE"].values)
    knn_lat = KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1).fit(Xs, tr["LATITUDE"].values)
    save(sc, "uji_scaler.pkl")
    save(clf, "uji_svm_floor.pkl")
    save(knn_lon, "uji_knn_lon.pkl")
    save(knn_lat, "uji_knn_lat.pkl")
    json.dump({"feature_order": feats, "wap_keep": keep}, open(os.path.join(MODELS, "uji_meta.json"), "w"))
    print("saved UJI models.")


def train_campus():
    p = os.path.join(BASE, "data", "campus", "trainingData_campus.csv")
    df = pd.read_csv(p)
    waps = [c for c in df.columns if c.startswith("WAP")]
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
