"""UJI explorer visuals. Called from app.py with (df, waps)."""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
PAL = {"primary": "#1B3A4B", "accent": "#C67C4E", "sage": "#6AA99E"}
CAT8 = ["#1B3A4B", "#C67C4E", "#6AA99E", "#8AA8C8", "#E0C084", "#7A6C5D", "#4A7C7E", "#B7836B"]


def render(df, waps):
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="LONGITUDE", y="LATITUDE", color="BUILDINGID",
            color_discrete_sequence=["#1B3A4B", "#C67C4E", "#6AA99E"], title="Buildings separate in XY")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#F8F6F1")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(df, x="LONGITUDE", y="LATITUDE", color="FLOOR",
            color_discrete_sequence=CAT8, title="Floors overlap in XY (need classifier)")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#F8F6F1")
        st.plotly_chart(fig, use_container_width=True)
    f, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].hist(df["LONGITUDE"], bins=40, color=PAL["primary"], edgecolor="black", alpha=0.8)
    ax[0].set_title("Longitude"); ax[1].hist(df["LATITUDE"], bins=40, color=PAL["accent"], edgecolor="black", alpha=0.8)
    ax[1].set_title("Latitude"); plt.tight_layout(); st.pyplot(f)
    det = (df[waps] != 100).sum().sort_values(ascending=False)
    top5 = det.head(5).index.tolist()
    st.write("Top-5 most-seen WAPs:", top5)
    f, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    mask = df[top5[0]] != 100
    ax[0].scatter(df.loc[mask, top5[0]], df.loc[mask, "LONGITUDE"], s=4, alpha=0.4, color=PAL["primary"])
    ax[0].set_title(f"{top5[0]} vs LONGITUDE")
    vis = (df[waps] != 100).sum(axis=1)
    ax[1].hist(vis, bins=30, color=PAL["sage"], edgecolor="black")
    ax[1].set_title("Visible APs per sample"); plt.tight_layout(); st.pyplot(f)
