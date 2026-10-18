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
        out[n] = joblib.load(p) if os.path.exists(p) else None
    for n in ["uji_meta.json", "campus_meta.json"]:
        p = os.path.join(MODELS, n)
        out[n] = json.load(open(p)) if os.path.exists(p) else None
    p = os.path.join(MODELS, "room_centroids.csv")
    out["cent"] = pd.read_csv(p) if os.path.exists(p) else None
    return out


M = load(["uji_scaler.pkl", "uji_svm_floor.pkl", "uji_knn_lon.pkl",
          "uji_knn_lat.pkl", "campus_scaler.pkl", "campus_rf_block.pkl",
          "campus_rf_floor.pkl", "campus_knn_room.pkl", "campus_knn_reg.pkl"])
if not all([M["uji_scaler.pkl"], M["campus_scaler.pkl"]]):
    st.warning("Models not trained yet. Run: python scripts/train_champions.py then refresh.")

tabs = st.tabs(["Overview", "UJI Explorer", "UJI Predict", "Campus Live Scan", "Model Health"])
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UJI KNN R2", "0.9966")
    c2.metric("UJI KNN MAE", "2.06 m")
    c3.metric("UJI SVM Floor F1", "0.9937")
    c4.metric("Campus samples", "524 x 219 WAPs")
    st.info("Pipeline: scan -> +100/-105 sentinel -> zero-var drop -> 5 RSSI feats -> scaler fit train-only -> KNN/SVM/RF. UJI and campus keep separate feature orders.")
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
    if st.button("Predict demo validation row 0"):
        try:
            v = pd.read_csv(os.path.join(DATA, "raw", "validationData.csv"), nrows=1)
            meta = M["uji_meta.json"]
            waps = [c for c in v.columns if c.startswith("WAP")]
            row = v[waps].replace(100, -105)
            import sys
            sys.path.insert(0, os.path.join(BASE, "app_utils"))
            from preprocess import engineer
            full = engineer(row, meta["wap_keep"])
            X = np.nan_to_num(M["uji_scaler.pkl"].transform(full[meta["feature_order"]].values))
            fl = M["uji_svm_floor.pkl"].predict(X)[0]
            lo = M["uji_knn_lon.pkl"].predict(X)[0]
            la = M["uji_knn_lat.pkl"].predict(X)[0]
            st.success(f"Floor {fl} | Lon {lo:.2f} | Lat {la:.2f} | True floor {int(v['FLOOR'].iloc[0])}")
        except Exception as e:
            st.error(str(e))
with tabs[3]:
    st.subheader("Campus live scan -> block / floor / room + map")
    from scan_windows import scan_windows, to_wap_vector
    from preprocess import engineer as eng2, haversine_m
    if st.button("Scan Now (Windows)"):
        aps = scan_windows()
        if not aps:
            st.warning("No APs (need admin terminal + Wi-Fi on; Cloud has no netsh — use Upload).")
        else:
            meta = M["campus_meta.json"]
            vec = to_wap_vector(aps, meta["bssid_to_wap"], meta["wap_cols"])
            d = pd.DataFrame([vec], columns=meta["wap_cols"])
            d = eng2(d, meta["wap_cols"])
            X = np.nan_to_num(M["campus_scaler.pkl"].transform(d[meta["feature_order"]].values))
            b = M["campus_rf_block.pkl"].predict(X)[0]
            f = M["campus_rf_floor.pkl"].predict(X)[0]
            r = M["campus_knn_room.pkl"].predict(X)[0]
            ll = M["campus_knn_reg.pkl"].predict(X)[0]
            st.success(f"Block {b} | Floor {f} | Room {r} | Lat {ll[0]:.7f} Lon {ll[1]:.7f}")
            cent = M["cent"]
            if cent is not None:
                dm = haversine_m(ll[0], ll[1], cent["latitude"].values, cent["longitude"].values)
                st.dataframe(cent.assign(m=dm).sort_values("m").head(3)[["room", "latitude", "longitude", "m"]])
                import plotly.express as px
                tr = pd.read_csv(os.path.join(DATA, "campus", "trainingData_campus.csv"), usecols=["latitude", "longitude", "block"])
                fig = px.scatter(tr, x="longitude", y="latitude", color="block", title="Campus map — gold star is YOU")
                fig.add_scatter(x=[ll[1]], y=[ll[0]], mode="markers", marker=dict(size=16, color="#D9A441", symbol="star"), name="YOU")
                fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#F8F6F1")
                st.plotly_chart(fig, use_container_width=True)
    up = st.file_uploader("...or upload raw_scans.csv (Cloud demo)", type="csv")
    if up is not None:
        st.info("Upload path reuses the same parser: group by scan_id, max RSSI per BSSID, map to WAP order, predict.")

with tabs[4]:
    st.subheader("Model health")
    st.table(pd.DataFrame([
        {"model": "UJI KNN reg", "metric": "R2 0.9966, MAE 2.06m", "cv": "5-fold GridSearch k=5 distance"},
        {"model": "UJI SVM floor", "metric": "F1 0.9937, val-1111 85.24%", "cv": "stratified 5-fold C=10 RBF"},
        {"model": "Campus RF block/floor", "metric": "3-fold F1 (see train log)", "cv": "StratifiedKFold 3 + LORO"},
        {"model": "Campus KNN room/coords", "metric": "haversine Top-3 m", "cv": "k=3 distance, n=524 safe"},
    ]))
