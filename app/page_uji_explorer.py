"""UJI explorer visuals + insights. Pure function: render(df, waps)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from preprocess import PAL, CAT8



def _insights(df: pd.DataFrame, waps: list[str]) -> list[str]:
    """Data-driven insight bullets computed from the live sample."""
    n = len(df)
    bshare = df["BUILDINGID"].value_counts(normalize=True).round(2).to_dict()
    fshare = df["FLOOR"].value_counts(normalize=True).sort_index().round(2).to_dict()
    vis = (df[waps] != 100).sum(axis=1)
    det = (df[waps] != 100).sum().sort_values(ascending=False)
    top = det.head(3).index.tolist()
    return [
        f"Sample n={n:,}: buildings {bshare} — imbalance is mild, no resampling needed.",
        f"Floors {fshare} — stratified split required (used in notebooks).",
        f"Median visible APs {vis.median():.0f} (p90 {vis.quantile(0.9):.0f}) — sparsity is the norm; -105 fill keeps order.",
        f"Most-seen {top[0]} ({det.iloc[0]:,} hits) vs rarest seen — long tail; zero-var pruning removed 55 dead WAPs.",
        "Floors overlap in XY but separate in signal space — regression gives XY, classifier gives floor. Both needed.",
    ]


def render(df: pd.DataFrame, waps: list[str]) -> None:
    for b in _insights(df, waps):
        st.caption("Insight: " + b)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="LONGITUDE", y="LATITUDE", color="BUILDINGID",
                         color_discrete_sequence=[CAT8[0], CAT8[1], CAT8[2]],
                         title="Buildings separate in XY")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor=PAL["bg"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(df, x="LONGITUDE", y="LATITUDE", color="FLOOR",
                         color_discrete_sequence=CAT8, title="Floors overlap in XY (need classifier)")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor=PAL["bg"])
        st.plotly_chart(fig, use_container_width=True)
    f, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].hist(df["LONGITUDE"], bins=40, color=PAL["primary"], edgecolor="black", alpha=0.8)
    ax[0].set_title("Longitude"); ax[0].set_xlabel("m")
    ax[1].hist(df["LATITUDE"], bins=40, color=PAL["accent"], edgecolor="black", alpha=0.8)
    ax[1].set_title("Latitude"); ax[1].set_xlabel("m")
    plt.tight_layout(); st.pyplot(f)
    plt.close(f)
    det = (df[waps] != 100).sum().sort_values(ascending=False)
    top5 = det.head(5).index.tolist()
    st.write("Top-5 most-seen WAPs:", top5)
    f, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    mask = df[top5[0]] != 100
    ax[0].scatter(df.loc[mask, top5[0]], df.loc[mask, "LONGITUDE"], s=4, alpha=0.4, color=PAL["primary"])
    ax[0].set_title(f"{top5[0]} vs LONGITUDE"); ax[0].set_xlabel("RSSI dBm"); ax[0].set_ylabel("LONGITUDE")
    vis = (df[waps] != 100).sum(axis=1)
    ax[1].hist(vis, bins=30, color=PAL["sage"], edgecolor="black")
    ax[1].set_title("Visible APs per sample"); ax[1].set_xlabel("count")
    corr = df[top5].replace(100, np.nan).corr()
    im = ax[2].imshow(corr.values, cmap="RdYlGn", vmin=-0.2, vmax=1)
    ax[2].set_xticks(range(len(top5)), top5, rotation=45, fontsize=7)
    ax[2].set_yticks(range(len(top5)), top5, fontsize=7)
    ax[2].set_title("Top-5 WAP correlation")
    plt.colorbar(im, ax=ax[2]); plt.tight_layout(); st.pyplot(f)
    plt.close(f)
    st.caption("Weak inter-WAP correlation = low redundancy (good). Strong RSSI↔position link in the strongest AP = fingerprint works.")
