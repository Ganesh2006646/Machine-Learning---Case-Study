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

tabs = st.tabs(["Overview", "UJI Explorer", "UJI Predict", "Campus Live Scan", "Model Health"])
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UJI KNN (notebook, full)", "R2 0.9966")
    c2.metric("UJI KNN MAE (notebook)", "2.06 m")
    c3.metric("UJI SVM Floor (notebook)", "F1 0.9937")
    c4.metric("Campus deploy CV", "block RF (see Health)")
    st.info("Notebook champions (full 19,937 rows) vs deploy models (6k-row speed build in models/). Health tab shows the deploy CV. Pipeline: scan -> +100/-105 -> zero-var drop -> 5 RSSI feats -> scaler fit train-only -> KNN/SVM/RF. UJI and campus keep separate feature orders.")
with tabs[1]:
    st.subheader("UJI spatial structure (sample)")
    try:
        df = pd.read_csv(os.path.join(DATA, "raw", "trainingData.csv"), nrows=3000)
        import plotly.express as px
        fig = px.scatter(df, x="LONGITUDE", y="LATITUDE", color="BUILDINGID",
                         color_discrete_sequence=["#1B3A4B", "#C67C4E", "#6AA99E"],
                         title="Buildings separate cleanly in coordinate space")
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#F8F6F1")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(str(e))
with tabs[2]:
    st.subheader("UJI fingerprint -> floor + coordinates")
    miss = need(M, "uji_scaler.pkl", "uji_svm_floor.pkl", "uji_knn_lon.pkl", "uji_knn_lat.pkl", "uji_meta.json")
    if miss:
        st.error(f"Missing: {miss}. Run `python scripts/train_champions.py`.")
    elif st.button("Predict demo validation row 0"):
        try:
            from preprocess import engineer
            v = pd.read_csv(os.path.join(DATA, "raw", "validationData.csv"), nrows=1)
            meta = M["uji_meta.json"]
            waps = [c for c in v.columns if c.startswith("WAP")]
            row = v[waps].replace(100, -105)
            keep = [w for w in meta["wap_keep"] if w in row.columns]
            full = engineer(row, keep)
            X = np.nan_to_num(M["uji_scaler.pkl"].transform(full[meta["feature_order"]].values))
            fl = M["uji_svm_floor.pkl"].predict(X)[0]
            lo = M["uji_knn_lon.pkl"].predict(X)[0]
            la = M["uji_knn_lat.pkl"].predict(X)[0]
            st.success(f"Floor {fl} | Lon {lo:.2f} | Lat {la:.2f} | True floor {int(v['FLOOR'].iloc[0])} (k={meta.get('best_k','?')})")
        except Exception as e:
            st.error(str(e))
with tabs[3]:
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

with tabs[4]:
    st.subheader("Model health (deploy build — from train_champions.py log)")
    st.table(pd.DataFrame([
        {"model": "UJI SVM floor (deploy)", "metric": "5-fold F1 0.9737±0.0082 (n=2500 slice)", "tuning": "C=10 RBF fixed"},
        {"model": "UJI KNN lon (deploy)", "metric": "R2 0.9901, RMSE 12.35, MAE 5.53 (3-fold)", "tuning": "GridSearch k in [3,5,7] -> best k=3"},
        {"model": "Campus RF block", "metric": "3-fold F1 1.0000", "tuning": "n=200, depth=14"},
        {"model": "Campus RF floor", "metric": "3-fold F1 1.0000", "tuning": "n=200, depth=14"},
        {"model": "Campus KNN room/coords", "metric": "k=3 distance, haversine Top-3", "tuning": "n=524 small-n safe"},
    ]))
    st.caption("Notebook full-data champions: KNN R2 0.9966 / SVM F1 0.9937. Deploy uses a 6k-row speed build — numbers above are honest for models/*.pkl.")
