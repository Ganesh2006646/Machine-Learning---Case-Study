"""Shared preprocessing: ONE function reused by notebook, trainer, app."""
import numpy as np
import pandas as pd

PAL = {"bg": "#F8F6F1", "card": "#FFFFFF", "ink": "#1A2A32",
       "primary": "#1B3A4B", "accent": "#C67C4E", "sage": "#6AA99E",
       "gold": "#D9A441", "slate": "#5C6B73"}
CAT8 = ["#1B3A4B", "#C67C4E", "#6AA99E", "#8AA8C8",
        "#E0C084", "#7A6C5D", "#4A7C7E", "#B7836B"]
ENG = ["n_visible_aps", "mean_rssi", "max_rssi", "std_rssi", "rssi_range"]


def engineer(df, waps):
    d = df.copy()
    w = d[waps]
    det = w.where(w > -105)
    d["n_visible_aps"] = (w > -105).sum(axis=1)
    d["mean_rssi"] = det.mean(axis=1).fillna(-105)
    d["max_rssi"] = det.max(axis=1).fillna(-105)
    d["std_rssi"] = det.std(axis=1).fillna(0)
    d["rssi_range"] = (det.max(axis=1) - det.min(axis=1)).fillna(0)
    return d


def clean(df, wap_cols, sentinel=100, fill=-105):
    d = df.copy()
    d[wap_cols] = d[wap_cols].replace(sentinel, fill)
    return d


def haversine_m(a1, o1, a2, o2):
    R = 6371000.0
    p = np.radians
    la1, lo1, la2, lo2 = map(np.asarray, [a1, o1, a2, o2])
    h = np.sin(p(la2 - la1) / 2) ** 2 + np.cos(p(la1)) * np.cos(p(la2)) * np.sin(p(lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))
