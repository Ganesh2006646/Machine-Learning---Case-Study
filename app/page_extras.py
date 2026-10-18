"""Extra assumed features: demo fingerprints, history, report, simulated walk."""
import numpy as np
import pandas as pd
import streamlit as st


def demo_fingerprint(meta, strong=5, seed=7):
    rng = np.random.RandomState(seed)
    waps = meta["wap_cols"]
    vec = np.full(len(waps), -105)
    idx = rng.choice(len(waps), strong, replace=False)
    vec[idx] = rng.randint(-65, -45, strong)
    return pd.DataFrame([vec], columns=waps)


def push_history(room, lat, lon):
    h = st.session_state.get("scan_history", [])
    h.append({"room": room, "lat": lat, "lon": lon})
    st.session_state["scan_history"] = h[-20:]


def report_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode()


def walk_path(lat, lon, steps=12, drift=0.00003, seed=3):
    rng = np.random.RandomState(seed)
    la, lo, pts = lat, lon, []
    for _ in range(steps):
        la += rng.uniform(-drift, drift); lo += rng.uniform(-drift, drift)
        pts.append((la, lo))
    return pts
