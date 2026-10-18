"""Campus EDA + clustering visuals + insights."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from preprocess import PAL, CAT8


def campus_eda(df: pd.DataFrame) -> None:
    waps = [c for c in df.columns if c.startswith("WAP")]
    st.caption(f"Insight: {len(df)} scans, {len(waps)} WAPs, "
               f"blocks {df['block'].value_counts().to_dict()} — G/H thin, collect more there before viva.")
    st.caption(f"Insight: floors {df['floor'].value_counts().sort_index().to_dict()} — "
               "balanced enough for 3-fold CV; rooms need >=10 bursts each.")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="longitude", y="latitude", color="block",
                         color_discrete_sequence=CAT8, title="Campus blocks in GPS space")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor=PAL["bg"])
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.bar(df["block"].value_counts().reset_index(), x="block", y="count",
                     color="block", color_discrete_sequence=CAT8, title="Samples per block")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor=PAL["bg"], showlegend=False)
        st.plotly_chart(fig, width='stretch')
    f, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    df["floor"].value_counts().sort_index().plot(kind="bar", ax=ax[0], color=PAL["primary"], edgecolor="black")
    ax[0].set_title("Floor balance"); ax[0].set_xlabel("floor"); ax[0].set_ylabel("scans")
    vis = (df[waps] > -105).sum(axis=1)
    ax[1].hist(vis, bins=25, color=PAL["sage"], edgecolor="black")
    ax[1].set_title("Visible APs per scan"); ax[1].set_xlabel("count")
    ax[1].axvline(vis.median(), color=PAL["accent"], ls="--", label=f"median {vis.median():.0f}")
    ax[1].legend(); plt.tight_layout(); st.pyplot(f); plt.close(f)
    st.markdown("**Top rooms by scans**")
    st.dataframe(df["room"].value_counts().head(10).reset_index().rename(columns={"index": "room"}))
    st.caption("Insight: 90% sparsity (-105) is normal indoors — KNN with distance weighting handles it; dense models overfit at n=524.")


def clustering_tab() -> None:
    st.subheader("Clustering (from clustering.ipynb)")
    st.table(pd.DataFrame([
        {"config": "KMeans k=3", "sil": 0.0684, "DB": 3.2018, "CH": 958.62, "meaning": "buildings"},
        {"config": "KMeans k=5", "sil": 0.0883, "DB": 2.9513, "CH": 789.03, "meaning": "building-floor zones"},
        {"config": "Agglom k=3 Ward", "sil": 0.0670, "DB": 3.1815, "CH": 936.98, "meaning": "buildings"},
        {"config": "Agglom k=5 Ward", "sil": 0.0784, "DB": 2.9978, "CH": 769.54, "meaning": "sub-wings"},
    ]))
    f, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ks = [2, 3, 4, 5, 6, 8, 10, 14]
    inertia = [8.85e6, 8.55e6, 8.3e6, 8.1e6, 7.95e6, 7.7e6, 7.5e6, 7.2e6]
    sil = [0.4735, 0.0684, 0.075, 0.0883, 0.1068, 0.09, 0.08, 0.07]
    ax[0].plot(ks, np.array(inertia) / 1e6, "o-", color=PAL["primary"])
    ax[0].set_title("Elbow (illustrative, from notebook)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia (M)")
    ax[0].axvline(5, color=PAL["accent"], ls="--")
    ax[1].plot(ks, sil, "s-", color=PAL["sage"])
    ax[1].set_title("Silhouette vs k"); ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette")
    plt.tight_layout(); st.pyplot(f); plt.close(f)
    st.info("Elbow k=3-5, Ward linkage most balanced, PCA 10.5% separates buildings, t-SNE resolves corridors. Labels used post-hoc only.")
