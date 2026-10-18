"""Shared preprocessing + theme. Single source of truth for notebook/trainer/app."""
from __future__ import annotations
import numpy as np
import pandas as pd

PAL: dict[str, str] = {"bg": "#F8F6F1", "card": "#FFFFFF", "ink": "#1A2A32",
                       "primary": "#1B3A4B", "accent": "#C67C4E", "sage": "#6AA99E",
                       "gold": "#D9A441", "slate": "#5C6B73"}
CAT8: list[str] = ["#1B3A4B", "#C67C4E", "#6AA99E", "#8AA8C8",
                   "#E0C084", "#7A6C5D", "#4A7C7E", "#B7836B"]
ENG: list[str] = ["n_visible_aps", "mean_rssi", "max_rssi", "std_rssi", "rssi_range"]
SENTINEL_IN, FILL_DBM = 100, -105


def clean(df: pd.DataFrame, wap_cols: list[str], sentinel: int = SENTINEL_IN,
          fill: int = FILL_DBM) -> pd.DataFrame:
    """Replace not-detected sentinel with below-floor dBm. Copy-in, copy-out."""
    d = df.copy()
    d[wap_cols] = d[wap_cols].replace(sentinel, fill).clip(-105, 0)
    return d


def engineer(df: pd.DataFrame, waps: list[str]) -> pd.DataFrame:
    """5 vectorised RSSI summaries. NaN-safe: empty scans -> fill defaults."""
    d = df.copy()
    w = d[waps].astype(float)
    seen = w > FILL_DBM
    det = w.where(seen)
    d["n_visible_aps"] = seen.sum(axis=1).astype(int)
    d["mean_rssi"] = det.mean(axis=1).fillna(FILL_DBM)
    d["max_rssi"] = det.max(axis=1).fillna(FILL_DBM)
    d["std_rssi"] = det.std(axis=1).fillna(0.0)
    d["rssi_range"] = (det.max(axis=1) - det.min(axis=1)).fillna(0.0)
    return d


def fingerprint_quality(row: pd.Series) -> str:
    n = int((row > FILL_DBM).sum())
    if n == 0:
        return "empty — move near APs / rerun as admin"
    if n < 5:
        return "weak — prediction low confidence"
    if n < 12:
        return "ok — usable"
    return "strong — reliable"


def haversine_m(a1, o1, a2, o2) -> np.ndarray:
    """Great-circle distance in metres. Vectorised, NaN-safe."""
    R = 6371000.0
    la1, lo1, la2, lo2 = (np.asarray(x, dtype=float) for x in (a1, o1, a2, o2))
    p = np.radians
    h = np.sin(p(la2 - la1) / 2) ** 2 + np.cos(p(la1)) * np.cos(p(la2)) * np.sin(p(lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))

