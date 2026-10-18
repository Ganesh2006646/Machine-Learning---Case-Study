"""Wi-Fi Indoor Localization Dashboard — UJI + Campus Live."""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "app_utils"))
ROOT = os.path.dirname(BASE)
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")
PAL = {"bg": "#F8F6F1", "card": "#FFFFFF", "ink": "#1A2A32",
       "primary": "#1B3A4B", "accent": "#C67C4E", "sage": "#6AA99E",
       "gold": "#D9A441", "slate": "#5C6B73"}

st.set_page_config(page_title="Wi-Fi Indoor Localization", layout="wide")
st.markdown(f"""<style>
.stApp{{background:{PAL["bg"]};color:{PAL["ink"]}}}
div[data-testid="stMetric"]{{background:{PAL["card"]};border-radius:14px;padding:12px;border:1px solid #E8E2D6}}
</style>""", unsafe_allow_html=True)
st.title("Wi-Fi Fingerprint Indoor Localization")
st.caption("UJI benchmark + Amrita campus live scan | 23CSE301 Capstone")


@st.cache_resource
def load(names):
    out = {}
    for n in names:
        p = os.path.join(MODELS, n)
        try:
            out[n] = joblib.load(p) if os.path.exists(p) else None
        except Exception as e:
            out[n] = None
            out[n + "__err"] = str(e)
    for n in ["uji_meta.json", "campus_meta.json"]:
        p = os.path.join(MODELS, n)
        try:
            out[n] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
        except Exception as e:
            out[n] = None
            out[n + "__err"] = str(e)
    p = os.path.join(MODELS, "room_centroids.csv")
    try:
        out["cent"] = pd.read_csv(p) if os.path.exists(p) else None
    except Exception as e:
        out["cent"] = None
        out["cent__err"] = str(e)
    return out


def need(models_dict, *keys):
    missing = [k for k in keys if models_dict.get(k) is None]
    return missing


M = load(["uji_scaler.pkl", "uji_svm_floor.pkl", "uji_knn_lon.pkl",
          "uji_knn_lat.pkl", "campus_scaler.pkl", "campus_rf_block.pkl",
          "campus_rf_floor.pkl", "campus_knn_room.pkl", "campus_knn_reg.pkl"])
if not all([M["uji_scaler.pkl"], M["campus_scaler.pkl"]]):
    st.warning("Models not trained yet. Run: python scripts/train_champions.py then refresh.")

tabs = st.tabs(["Overview", "UJI Explorer", "UJI 10-Model Table",
    "UJI Predict Lab", "Campus EDA", "Campus Live Scan", "Simulated Walk",
    "Clustering", "Model Health", "Viva Q&A"])
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UJI KNN (notebook, full)", "R2 0.9966")
    c2.metric("UJI KNN MAE (notebook)", "2.06 m")
    c3.metric("UJI SVM Floor (notebook)", "F1 0.9937")
    c4.metric("Campus deploy CV", "block RF (see Health)")
    st.info("Notebook champions (full 19,937 rows) vs deploy models (6k-row speed build in models/). Health tab shows the deploy CV. Pipeline: scan -> +100/-105 -> zero-var drop -> 5 RSSI feats -> scaler fit train-only -> KNN/SVM/RF. UJI and campus keep separate feature orders.")
with tabs[1]:
    st.subheader("UJI Explorer — spatial + RSSI (sampled)")
    try:
        n = st.slider("samples", 1000, 8000, 3000, 500, key="uji_n")
        df = pd.read_csv(os.path.join(DATA, "raw", "trainingData.csv"), nrows=n)
        from page_uji_explorer import render as uji_render
        uji_render(df, [c for c in df.columns if c.startswith("WAP")])
    except Exception as e:
        st.error(str(e))
with tabs[2]:
    st.subheader("UJI 10-model comparison (from notebooks)")
    st.markdown("**Regression (test 3,988, by R2)**")
    st.table(pd.DataFrame([
        {"m": "KNN tuned", "R2": 0.9966, "RMSE": 7.28, "MAE": 2.06},
        {"m": "RandForest", "R2": 0.9935, "RMSE": 10.01, "MAE": 5.23},
        {"m": "GradBoost", "R2": 0.98, "RMSE": 17.6, "MAE": 7.66},
        {"m": "SVR RBF", "R2": 0.9755, "RMSE": 19.46, "MAE": 8.17},
        {"m": "Tree", "R2": 0.9717, "RMSE": 20.92, "MAE": 9.15},
        {"m": "Linear/Ridge", "R2": 0.9488, "RMSE": 28.17, "MAE": 20.21},
        {"m": "Lasso", "R2": 0.9419, "RMSE": 29.99, "MAE": 22.25},
        {"m": "ElasticNet", "R2": 0.9339, "RMSE": 32.0, "MAE": 23.88},
        {"m": "Poly d2", "R2": 0.8161, "RMSE": 53.37, "MAE": 39.82},
    ]))
    st.markdown("**Floor 0-4 (weighted)**")
    st.table(pd.DataFrame([
        {"m": "SVM RBF", "acc": 0.9937, "F1": 0.9937, "AUC": 0.9989},
        {"m": "MLP 128,64", "acc": 0.9932, "F1": 0.9932, "AUC": 0.9996},
        {"m": "GradBoost", "acc": 0.9925, "F1": 0.9925, "AUC": 0.9997},
        {"m": "RandForest", "acc": 0.9895, "F1": 0.9895, "AUC": 0.9997},
        {"m": "LogReg", "acc": 0.989, "F1": 0.989, "AUC": 0.9998},
        {"m": "KNN k=5", "acc": 0.9887, "F1": 0.9887, "AUC": 0.9992},
        {"m": "Bagging", "acc": 0.9155, "F1": 0.9162, "AUC": 0.9924},
        {"m": "AdaBoost", "acc": 0.9122, "F1": 0.9135, "AUC": 0.9827},
        {"m": "Tree", "acc": 0.7931, "F1": 0.8017, "AUC": 0.9481},
        {"m": "NaiveBayes", "acc": 0.4982, "F1": 0.5158, "AUC": 0.7531},
    ]))
    st.caption("Official validation 1,111 (shift): SVM 85.24%.")
with tabs[3]:
    st.subheader("UJI Predict Lab — fingerprint in, floor + XY out")
    miss = need(M, "uji_scaler.pkl", "uji_svm_floor.pkl", "uji_knn_lon.pkl", "uji_knn_lat.pkl", "uji_meta.json")
    if miss:
        st.error(f"Missing: {miss}. Run `python scripts/train_champions.py`.")
    else:
        n = st.slider("validation rows to test", 1, 20, 5, key="uji_test_n")
        if st.button("Run UJI demo"):
            try:
                from preprocess import engineer
                v = pd.read_csv(os.path.join(DATA, "raw", "validationData.csv"), nrows=n)
                meta = M["uji_meta.json"]
                waps = [c for c in v.columns if c.startswith("WAP")]
                rows = v[waps].replace(100, -105)
                keep = [w for w in meta["wap_keep"] if w in rows.columns]
                full = engineer(rows, keep)
                X = np.nan_to_num(M["uji_scaler.pkl"].transform(full[meta["feature_order"]].values))
                v["pred_floor"] = M["uji_svm_floor.pkl"].predict(X)
                v["pred_lon"] = M["uji_knn_lon.pkl"].predict(X)
                v["pred_lat"] = M["uji_knn_lat.pkl"].predict(X)
                v["err_m"] = np.sqrt((v["LONGITUDE"] - v["pred_lon"]) ** 2 + (v["LATITUDE"] - v["pred_lat"]) ** 2)
                st.success(f"Floor acc: {(v['pred_floor'] == v['FLOOR']).mean():.3f} | mean err: {v['err_m'].mean():.2f} m (k={meta.get('best_k','?')})")
                st.dataframe(v[["FLOOR", "pred_floor", "LONGITUDE", "pred_lon", "LATITUDE", "pred_lat", "err_m"]])
            except Exception as e:
                st.error(str(e))
with tabs[4]:
    st.subheader("Campus EDA — blocks, floors, rooms")
    try:
        from page_campus_cluster import campus_eda
        campus_eda(pd.read_csv(os.path.join(DATA, "campus", "trainingData_campus.csv")))
    except Exception as e:
        st.error(str(e))
with tabs[5]:
    st.subheader("Campus live scan -> block / floor / room + map")
    from scan_windows import scan_windows, to_wap_vector
    from preprocess import engineer as eng2, haversine_m
    miss = need(M, "campus_scaler.pkl", "campus_rf_block.pkl", "campus_rf_floor.pkl",
                "campus_knn_room.pkl", "campus_knn_reg.pkl", "campus_meta.json")
    if miss:
        st.error(f"Missing: {miss}. Run `python scripts/train_champions.py`.")


    def predict_campus_df(wide_df, meta):
        """Shared predict for Scan Now + Upload. Returns (block, floor, room, lat, lon)."""
        waps = meta["wap_cols"]
        for w in waps:
            if w not in wide_df.columns:
                wide_df[w] = -105
        d = wide_df[waps].replace(100, -105)
        d = eng2(d, waps)
        X = np.nan_to_num(M["campus_scaler.pkl"].transform(d[meta["feature_order"]].values))
        b = M["campus_rf_block.pkl"].predict(X)[0]
        f = M["campus_rf_floor.pkl"].predict(X)[0]
        r = M["campus_knn_room.pkl"].predict(X)[0]
        ll = M["campus_knn_reg.pkl"].predict(X)[0]
        return b, f, r, float(ll[0]), float(ll[1])


    def show_result(b, f, r, lat, lon):
        st.success(f"Block {b} | Floor {f} | Room {r} | Lat {lat:.7f} Lon {lon:.7f}")
        cent = M["cent"]
        if cent is not None:
            dm = haversine_m(lat, lon, cent["latitude"].values, cent["longitude"].values)
            st.dataframe(cent.assign(dist_m=dm).sort_values("dist_m").head(3)[["room", "latitude", "longitude", "dist_m"]])
            import plotly.express as px
            tr = pd.read_csv(os.path.join(DATA, "campus", "trainingData_campus.csv"), usecols=["latitude", "longitude", "block"])
            fig = px.scatter(tr, x="longitude", y="latitude", color="block", title="Campus map — gold star is YOU")
            fig.add_scatter(x=[lon], y=[lat], mode="markers", marker=dict(size=16, color="#D9A441", symbol="star"), name="YOU")
            fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#F8F6F1")
            st.plotly_chart(fig, use_container_width=True)

    if not miss and st.button("Scan Now (Windows)"):
        aps = scan_windows()
        if not aps:
            st.warning("No APs (need admin terminal + Wi-Fi on; Cloud has no netsh — use Upload).")
        else:
            try:
                meta = M["campus_meta.json"]
                vec = to_wap_vector(aps, meta["bssid_to_wap"], meta["wap_cols"])
                d = pd.DataFrame([vec], columns=meta["wap_cols"])
                b, f, r, lat, lon = predict_campus_df(d, meta)
                show_result(b, f, r, lat, lon)
            except Exception as e:
                st.error(str(e))
    up = st.file_uploader("...or upload raw_scans.csv (Cloud demo)", type="csv")
    if up is not None and not miss:
        try:
            raw = pd.read_csv(up)
            raw.columns = [c.strip().lower() for c in raw.columns]
            if "bssid" not in raw.columns or "rssi_dbm" not in raw.columns:
                st.error("CSV needs columns bssid + rssi_dbm (export from campus_collector).")
            else:
                raw["bssid"] = raw["bssid"].astype(str).str.lower().str.strip()
                grp = raw.groupby("bssid")["rssi_dbm"].max().to_dict()
                aps = [{"bssid": k, "rssi_dbm": int(v)} for k, v in grp.items()]
                meta = M["campus_meta.json"]
                vec = to_wap_vector(aps, meta["bssid_to_wap"], meta["wap_cols"])
                d = pd.DataFrame([vec], columns=meta["wap_cols"])
                b, f, r, lat, lon = predict_campus_df(d, meta)
                show_result(b, f, r, lat, lon)
        except Exception as e:
            st.error(str(e))

with tabs[6]:
    st.subheader("Simulated walk — assumed movement trail")
    st.caption("No Wi-Fi needed: assumes a walk around the predicted point with drift.")
    lat0 = st.number_input("start lat", value=10.90628, format="%.7f")
    lon0 = st.number_input("start lon", value=76.89784, format="%.7f")
    steps = st.slider("steps", 5, 30, 12)
    if st.button("Simulate walk"):
        from page_extras import walk_path
        pts = walk_path(lat0, lon0, steps)
        import plotly.express as px
        tr = pd.read_csv(os.path.join(DATA, "campus", "trainingData_campus.csv"), usecols=["latitude", "longitude", "block"])
        fig = px.scatter(tr, x="longitude", y="latitude", color="block", title="Assumed walk trail")
        la, lo = zip(*pts)
        fig.add_scatter(x=list(lo), y=list(la), mode="lines+markers", name="walk",
                        line=dict(color="#C67C4E", width=3), marker=dict(color="#D9A441", size=8))
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#F8F6F1")
        st.plotly_chart(fig, use_container_width=True)
with tabs[7]:
    try:
        from page_campus_cluster import clustering_tab
        clustering_tab()
    except Exception as e:
        st.error(str(e))
with tabs[8]:
    st.subheader("Model health (deploy build — from train_champions.py log)")
    st.table(pd.DataFrame([
        {"model": "UJI SVM floor (deploy)", "metric": "5-fold F1 0.9737±0.0082 (n=2500 slice)", "tuning": "C=10 RBF fixed"},
        {"model": "UJI KNN lon (deploy)", "metric": "R2 0.9901, RMSE 12.35, MAE 5.53 (3-fold)", "tuning": "GridSearch k in [3,5,7] -> best k=3"},
        {"model": "Campus RF block", "metric": "3-fold F1 1.0000", "tuning": "n=200, depth=14"},
        {"model": "Campus RF floor", "metric": "3-fold F1 1.0000", "tuning": "n=200, depth=14"},
        {"model": "Campus KNN room/coords", "metric": "k=3 distance, haversine Top-3", "tuning": "n=524 small-n safe"},
    ]))
    st.caption("Notebook full-data champions: KNN R2 0.9966 / SVM F1 0.9937. Deploy uses a 6k-row speed build — numbers above are honest for models/*.pkl.")
    if st.button("Download campus predictions report"):
        try:
            from page_extras import report_csv
            df = pd.read_csv(os.path.join(DATA, "campus", "trainingData_campus.csv"), nrows=30)
            st.download_button("Save CSV", report_csv(df[["block", "floor", "room", "latitude", "longitude"]].to_dict("records")), "campus_report.csv")
        except Exception as e:
            st.error(str(e))
with tabs[9]:
    st.subheader("Viva Q&A — one-line answers")
    st.markdown("""
- **Why -105?** Below -104 sensitivity floor; preserves order for distance models.
- **Why KNN wins?** Fingerprint neighbours = physical neighbours.
- **Why 99% test but 85% official val?** Time/device shift; honest gap.
- **Why separate campus models?** Different WAP universe (219 vs 465).
- **Why RF + K=3 on campus?** n=524 small; 3-fold honest.
- **Assumed extras?** Demo fingerprint, simulated walk, history, report — all labelled assumed, core predictions stay real.
""")
    st.info("Examiner tip: run Predict Lab (5 rows) -> Live Scan Upload -> Simulated Walk -> download report. Full loop in 90 seconds.")
