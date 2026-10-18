"""Wi-Fi Indoor Localization Dashboard — UJI + Campus Live (clean, optimized)."""
from __future__ import annotations
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "app_utils"))
ROOT = os.path.dirname(BASE)
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")

from preprocess import PAL, CAT8, engineer, fingerprint_quality, haversine_m
from scan_windows import scan_windows, to_wap_vector
from page_uji_explorer import render as uji_render
from page_campus_cluster import campus_eda, clustering_tab
from page_extras import demo_fingerprint, push_history, report_csv, walk_path

plt.rcParams.update({"figure.facecolor": PAL["bg"], "axes.facecolor": PAL["card"]})
st.set_page_config(page_title="Wi-Fi Indoor Localization", layout="wide")
st.markdown(f"""<style>.stApp{{background:{PAL["bg"]};color:{PAL["ink"]}}}
div[data-testid="stMetric"]{{background:{PAL["card"]};border-radius:14px;padding:12px;border:1px solid #E8E2D6}}
section[data-testid="stSidebar"]{{background:#EFE9DC}}</style>""", unsafe_allow_html=True)
st.title("Wi-Fi Fingerprint Indoor Localization")
st.caption("UJI benchmark (19,937) + Amrita campus live scan (524) | 23CSE301 Capstone")

MODEL_FILES = ["uji_scaler.pkl", "uji_svm_floor.pkl", "uji_knn_lon.pkl", "uji_knn_lat.pkl",
               "campus_scaler.pkl", "campus_rf_block.pkl", "campus_rf_floor.pkl",
               "campus_knn_room.pkl", "campus_knn_reg.pkl"]


@st.cache_resource
def load_all() -> dict:
    out: dict = {}
    for n in MODEL_FILES:
        p = os.path.join(MODELS, n)
        try:
            out[n] = joblib.load(p) if os.path.exists(p) else None
        except Exception as e:
            out[n] = None
            out[n + "__err"] = str(e)
    for n in ("uji_meta.json", "campus_meta.json"):
        p = os.path.join(MODELS, n)
        try:
            out[n] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
        except Exception as e:
            out[n] = None
            out[n + "__err"] = str(e)
    p = os.path.join(MODELS, "room_centroids.csv")
    out["cent"] = pd.read_csv(p) if os.path.exists(p) else None
    return out


@st.cache_data
def uji_rows(n: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, "raw", "trainingData.csv"), nrows=n)


@st.cache_data
def campus_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, "campus", "trainingData_campus.csv"))


@st.cache_data
def val_rows(n: int) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, "raw", "validationData.csv"), nrows=n)


def missing(d: dict, *keys: str) -> list[str]:
    return [k for k in keys if d.get(k) is None]


def err_box(msg: str) -> None:
    st.error(msg)


M = load_all()
if missing(M, "uji_scaler.pkl", "campus_scaler.pkl"):
    st.warning("Models missing. Run: `python scripts/train_champions.py` then refresh.")

tabs = st.tabs(["Overview", "UJI Explorer", "UJI 10-Model Table", "UJI Predict Lab",
                "Campus EDA", "Campus Live Scan", "Simulated Walk",
                "Clustering", "Model Health", "Viva Q&A"])
# ---- generated pages A (overview..campus EDA) ----
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UJI samples", "19,937 + 1,111 val")
    c2.metric("UJI KNN (notebook)", "R2 0.9966, MAE 2.06m")
    c3.metric("UJI SVM floor", "F1 0.9937")
    c4.metric("Campus", "524 x 219 WAPs, 8 blocks")
    st.info("Scan (netsh) -> +100 to -105 dBm -> zero-var drop (520->465 UJI) -> 5 RSSI feats -> scaler fit train-only -> KNN/SVM/RF. UJI & campus use separate feature orders.")
    a, b = st.columns(2)
    with a:
        st.markdown("#### UJI champions (full-data notebooks)")
        st.table(pd.DataFrame([
            {"track": "Regression", "best": "KNN k=5 distance", "score": "R2 0.9966, RMSE 7.28, MAE 2.06"},
            {"track": "Classification", "best": "SVM RBF C=10", "score": "F1 0.9937, AUC 0.9989"},
            {"track": "Clustering", "best": "KMeans k=5", "score": "Sil 0.0883, DB 2.95"}]))
    with b:
        st.markdown("#### Campus deploy (models/*.pkl)")
        st.table(pd.DataFrame([
            {"model": "RF block", "cv": "3-fold F1 1.0000"},
            {"model": "RF floor", "cv": "3-fold F1 1.0000"},
            {"model": "KNN room/coords", "cv": "k=3 + haversine Top-3"}]))
with tabs[1]:
    st.subheader("UJI Explorer — spatial + RSSI (sampled)")
    n = st.slider("samples", 1000, 8000, 3000, 500, key="uji_n")
    try:
        df = uji_rows(n)
        uji_render(df, [c for c in df.columns if c.startswith("WAP")])
        f, ax = plt.subplots(figsize=(9, 3.4))
        wc = [c for c in df.columns if c.startswith("WAP")]
        eng = engineer(df[wc].replace(100, -105), wc)
        ax.scatter(eng["mean_rssi"], eng["n_visible_aps"], s=5, alpha=0.4, color=PAL["primary"])
        ax.set_title("Engineered: mean RSSI vs visible count"); ax.set_xlabel("mean_rssi"); ax.set_ylabel("n_visible_aps")
        plt.tight_layout(); st.pyplot(f); plt.close(f)
        st.caption("Insight: open areas see more APs with stronger mean — both feats help KNN separate buildings.")
    except Exception as e:
        err_box(str(e))
with tabs[2]:
    st.subheader("UJI 10-model comparison (from notebooks)")
    reg = pd.DataFrame([
        {"m": "KNN tuned", "R2": 0.9966, "RMSE": 7.28, "MAE": 2.06},
        {"m": "RandForest", "R2": 0.9935, "RMSE": 10.01, "MAE": 5.23},
        {"m": "GradBoost", "R2": 0.98, "RMSE": 17.6, "MAE": 7.66},
        {"m": "SVR RBF", "R2": 0.9755, "RMSE": 19.46, "MAE": 8.17},
        {"m": "Tree", "R2": 0.9717, "RMSE": 20.92, "MAE": 9.15},
        {"m": "Linear/Ridge", "R2": 0.9488, "RMSE": 28.17, "MAE": 20.21},
        {"m": "Lasso", "R2": 0.9419, "RMSE": 29.99, "MAE": 22.25},
        {"m": "ElasticNet", "R2": 0.9339, "RMSE": 32.0, "MAE": 23.88},
        {"m": "Poly d2", "R2": 0.8161, "RMSE": 53.37, "MAE": 39.82}])
    st.table(reg)
    f, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh(reg["m"], reg["R2"], color=PAL["sage"], edgecolor="black")
    ax.set_title("Regression R2 ranking"); plt.tight_layout(); st.pyplot(f); plt.close(f)
    clf = pd.DataFrame([
        {"m": "SVM RBF", "F1": 0.9937}, {"m": "MLP 128,64", "F1": 0.9932},
        {"m": "GradBoost", "F1": 0.9925}, {"m": "RandForest", "F1": 0.9895},
        {"m": "LogReg", "F1": 0.989}, {"m": "KNN k=5", "F1": 0.9887},
        {"m": "Bagging", "F1": 0.9155}, {"m": "AdaBoost", "F1": 0.9135},
        {"m": "Tree", "F1": 0.8017}, {"m": "NaiveBayes", "F1": 0.5158}])
    st.table(clf)
    f, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(clf["m"], clf["F1"], color=PAL["primary"], edgecolor="black")
    ax.set_title("Classifier F1 ranking"); plt.tight_layout(); st.pyplot(f); plt.close(f)
    st.caption("Insight: distance methods dominate; NB fails on correlated WAPs. Official val SVM 85.24% (shift).")
with tabs[3]:
    st.subheader("UJI Predict Lab — fingerprint in, floor + XY out")
    miss = missing(M, "uji_scaler.pkl", "uji_svm_floor.pkl", "uji_knn_lon.pkl", "uji_knn_lat.pkl", "uji_meta.json")
    if miss:
        err_box(f"Missing: {miss}. Run python scripts/train_champions.py.")
    else:
        n = st.slider("validation rows to test", 1, 20, 5, key="uji_test_n")
        if st.button("Run UJI demo"):
            try:
                v = val_rows(n)
                meta = M["uji_meta.json"]
                waps = [c for c in v.columns if c.startswith("WAP")]
                rows = v[waps].replace(100, -105)
                keep = [w for w in meta["wap_keep"] if w in rows.columns]
                X = np.nan_to_num(M["uji_scaler.pkl"].transform(engineer(rows, keep)[meta["feature_order"]].values))
                v["pred_floor"] = M["uji_svm_floor.pkl"].predict(X)
                v["pred_lon"] = M["uji_knn_lon.pkl"].predict(X)
                v["pred_lat"] = M["uji_knn_lat.pkl"].predict(X)
                v["err_m"] = np.sqrt((v["LONGITUDE"] - v["pred_lon"]) ** 2 + (v["LATITUDE"] - v["pred_lat"]) ** 2)
                c1, c2 = st.columns(2)
                c1.success(f"Floor acc: {(v['pred_floor'] == v['FLOOR']).mean():.3f} (k={meta.get('best_k','?')})")
                c2.success(f"Mean XY err: {v['err_m'].mean():.2f} m")
                st.dataframe(v[["FLOOR", "pred_floor", "LONGITUDE", "pred_lon", "LATITUDE", "pred_lat", "err_m"]])
                f, ax = plt.subplots(figsize=(9, 3.2))
                ax.hist(v["err_m"], bins=min(15, len(v)), color=PAL["gold"], edgecolor="black")
                ax.set_title("Error distribution"); ax.set_xlabel("metres"); plt.tight_layout(); st.pyplot(f); plt.close(f)
            except Exception as e:
                err_box(str(e))
with tabs[4]:
    st.subheader("Campus EDA — blocks, floors, rooms")
    try:
        campus_eda(campus_df())
    except Exception as e:
        err_box(str(e))
# ---- pages B1: live scan ----
with tabs[5]:
    st.subheader("Campus live scan -> block / floor / room + map")
    miss = missing(M, "campus_scaler.pkl", "campus_rf_block.pkl", "campus_rf_floor.pkl",
                   "campus_knn_room.pkl", "campus_knn_reg.pkl", "campus_meta.json")
    if miss:
        err_box(f"Missing: {miss}. Run python scripts/train_champions.py.")


    def _predict_campus(wide_df: pd.DataFrame, meta: dict):
        waps = meta["wap_cols"]
        for w in waps:
            if w not in wide_df.columns:
                wide_df[w] = -105
        d = engineer(wide_df[waps].replace(100, -105), waps)
        X = np.nan_to_num(M["campus_scaler.pkl"].transform(d[meta["feature_order"]].values))
        b = M["campus_rf_block.pkl"].predict(X)[0]
        f = M["campus_rf_floor.pkl"].predict(X)[0]
        r = M["campus_knn_room.pkl"].predict(X)[0]
        ll = M["campus_knn_reg.pkl"].predict(X)[0]
        return b, f, r, float(ll[0]), float(ll[1])


    def _show(b, f, r, lat, lon, wide) -> None:
        st.success(f"Block {b} | Floor {f} | Room {r} | Lat {lat:.7f} Lon {lon:.7f}")
        cent = M["cent"]
        if cent is not None:
            dm = haversine_m(lat, lon, cent["latitude"].values, cent["longitude"].values)
            top = cent.assign(dist_m=np.round(dm, 1)).sort_values("dist_m").head(3)
            st.dataframe(top[["room", "latitude", "longitude", "dist_m"]])
            st.caption(f"Insight: nearest {top.iloc[0]['room']} at {top.iloc[0]['dist_m']:.1f} m — walk there to verify.")
        push_history(str(r), lat, lon)
        tr = campus_df()
        fig = px.scatter(tr, x="longitude", y="latitude", color="block",
                         color_discrete_sequence=CAT8, title="Campus map — gold star is YOU")
        fig.add_scatter(x=[lon], y=[lat], mode="markers",
                        marker=dict(size=16, color="#D9A441", symbol="star"), name="YOU")
        if len(st.session_state.get("scan_history", [])) > 1:
            h = pd.DataFrame(st.session_state["scan_history"])
            fig.add_scatter(x=h["lon"], y=h["lat"], mode="lines+markers", name="trail",
                            line=dict(color="#C67C4E", width=2))
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor=PAL["bg"])
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Fingerprint quality: {fingerprint_quality(wide.iloc[0])} "
                   f"({int((wide.iloc[0] > -105).sum())} APs visible).")

    if not miss and st.button("Scan Now (Windows)"):
        aps = scan_windows()
        if not aps:
            st.warning("No APs (admin terminal + Wi-Fi on; Cloud has no netsh — use Demo/Upload).")
        else:
            try:
                meta = M["campus_meta.json"]
                wide = pd.DataFrame([to_wap_vector(aps, meta["bssid_to_wap"], meta["wap_cols"])], columns=meta["wap_cols"])
                _show(*_predict_campus(wide, meta), wide)
            except Exception as e:
                err_box(str(e))
    if not miss and st.button("Try assumed demo fingerprint (Cloud-safe)"):
        try:
            meta = M["campus_meta.json"]
            wide = demo_fingerprint(meta)
            _show(*_predict_campus(wide, meta), wide)
            st.caption("Assumed synthetic fingerprint — UI demo only.")
        except Exception as e:
            err_box(str(e))
    up = st.file_uploader("...or upload raw_scans.csv (Cloud demo)", type="csv")
    if up is not None and not miss:
        try:
            raw = pd.read_csv(up)
            raw.columns = [c.strip().lower() for c in raw.columns]
            if "bssid" not in raw.columns or "rssi_dbm" not in raw.columns:
                err_box("CSV needs columns bssid + rssi_dbm.")
            else:
                raw["bssid"] = raw["bssid"].astype(str).str.lower().str.strip()
                grp = raw.groupby("bssid")["rssi_dbm"].max().to_dict()
                aps = [{"bssid": k, "rssi_dbm": int(v)} for k, v in grp.items()]
                meta = M["campus_meta.json"]
                wide = pd.DataFrame([to_wap_vector(aps, meta["bssid_to_wap"], meta["wap_cols"])], columns=meta["wap_cols"])
                _show(*_predict_campus(wide, meta), wide)
        except Exception as e:
            err_box(str(e))
    if st.session_state.get("scan_history"):
        st.markdown("**Scan trail (this session)**")
        st.dataframe(pd.DataFrame(st.session_state["scan_history"]))
# ---- pages B2: walk, clustering, health, viva ----
with tabs[6]:
    st.subheader("Simulated walk — assumed movement trail")
    lat0 = st.number_input("start lat", value=10.90628, format="%.7f")
    lon0 = st.number_input("start lon", value=76.89784, format="%.7f")
    steps = st.slider("steps", 5, 30, 12)
    if st.button("Simulate walk"):
        pts = walk_path(lat0, lon0, steps)
        tr = campus_df()
        fig = px.scatter(tr, x="longitude", y="latitude", color="block",
                         color_discrete_sequence=CAT8, title="Assumed walk trail")
        la, lo = zip(*pts)
        fig.add_scatter(x=list(lo), y=list(la), mode="lines+markers", name="walk",
                        line=dict(color="#C67C4E", width=3), marker=dict(color="#D9A441", size=8))
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor=PAL["bg"])
        st.plotly_chart(fig, width='stretch')
with tabs[7]:
    try:
        clustering_tab()
    except Exception as e:
        err_box(str(e))
with tabs[8]:
    st.subheader("Model health (deploy build)")
    st.table(pd.DataFrame([
        {"model": "UJI SVM floor", "metric": "5-fold F1 0.9737+-0.0082 (n=2500)", "tuning": "C=10 RBF"},
        {"model": "UJI KNN lon", "metric": "R2 0.9901, RMSE 12.35, MAE 5.53", "tuning": "k in [3,5,7] -> 3"},
        {"model": "Campus RF block", "metric": "3-fold F1 1.0000", "tuning": "n=200 depth=14"},
        {"model": "Campus RF floor", "metric": "3-fold F1 1.0000", "tuning": "n=200 depth=14"},
        {"model": "Campus KNN room/coords", "metric": "k=3 + haversine Top-3", "tuning": "n=524 safe"}]))
    st.caption("Notebook full-data: KNN R2 0.9966 / SVM F1 0.9937. Deploy = 6k-row speed build.")
    if st.button("Download campus report (30 rows)"):
        try:
            df = campus_df().head(30)
            st.download_button("Save CSV", report_csv(df[["block", "floor", "room", "latitude", "longitude"]]), "campus_report.csv")
        except Exception as e:
            err_box(str(e))
with tabs[9]:
    st.subheader("Viva Q&A — one-line answers")
    st.markdown("- **Why -105?** Below -104 floor; preserves order.\n- **Why KNN wins?** Fingerprint neighbours = physical neighbours.\n- **Why 99% but 85% val?** Time/device shift; honest gap.\n- **Why separate campus models?** 219 vs 465 WAPs.\n- **Why RF + k=3 campus?** n=524; 3-fold honest.\n- **Assumed?** Demo print, walk, trail — labelled; core predictions real.")
    st.info("90-sec loop: Predict Lab (5 rows) -> Live Scan Upload/Demo -> Walk -> report.")
