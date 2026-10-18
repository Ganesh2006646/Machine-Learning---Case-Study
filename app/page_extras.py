"""Assumed demo helpers: fingerprint, history, report, walk. Core predictions stay real."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st


def demo_fingerprint(meta: dict, strong: int = 5, seed: int = 7) -> pd.DataFrame:
    """Assumed: synthetic campus fingerprint for Cloud demo (labelled assumed in UI)."""
    rng = np.random.RandomState(seed)
    waps = meta["wap_cols"]
    vec = np.full(len(waps), -105)
    idx = rng.choice(len(waps), min(strong, len(waps)), replace=False)
    vec[idx] = rng.randint(-65, -45, len(idx))
    return pd.DataFrame([vec], columns=waps)


def push_history(room: str, lat: float, lon: float, cap: int = 20) -> list[dict]:
    h = st.session_state.get("scan_history", [])
    h.append({"room": room, "lat": lat, "lon": lon})
    st.session_state["scan_history"] = h[-cap:]
    return st.session_state["scan_history"]


def report_csv(rows: list[dict] | pd.DataFrame) -> bytes:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    return df.to_csv(index=False).encode()


def walk_path(lat: float, lon: float, steps: int = 12,
              drift: float = 0.00003, seed: int = 3) -> list[tuple[float, float]]:
    """Assumed: random walk around a point for the trail visual."""
    rng = np.random.RandomState(seed)
    la, lo, pts = lat, lon, []
    for _ in range(max(1, steps)):
        la += rng.uniform(-drift, drift)
        lo += rng.uniform(-drift, drift)
        pts.append((la, lo))
    return pts
