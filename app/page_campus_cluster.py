"""Campus EDA + clustering visuals."""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
CAT8 = ["#1B3A4B", "#C67C4E", "#6AA99E", "#8AA8C8", "#E0C084", "#7A6C5D", "#4A7C7E", "#B7836B"]


def campus_eda(df):
    import plotly.express as px
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df, x="longitude", y="latitude", color="block", title="Campus blocks in GPS space")
        fig.update_layout(plot_bgcolor="#FFF", paper_bgcolor="#F8F6F1")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.bar_chart(df["block"].value_counts())
        st.caption("Samples per block (G/H thin = collect more there).")
    st.bar_chart(df["floor"].value_counts().sort_index())
    st.caption("Floor balance. Top rooms:")
    st.dataframe(df["room"].value_counts().head(10))
    f, ax = plt.subplots(figsize=(8, 3.5))
    vis = (df[[c for c in df.columns if c.startswith("WAP")]] > -105).sum(axis=1)
    ax.hist(vis, bins=25, color="#6AA99E", edgecolor="black")
    ax.set_title("Campus visible APs per scan"); plt.tight_layout(); st.pyplot(f)


def clustering_tab():
    st.subheader("Clustering (from clustering.ipynb)")
    st.table(pd.DataFrame([
        {"config": "KMeans k=3", "sil": 0.0684, "DB": 3.2018, "CH": 958.62, "meaning": "buildings"},
        {"config": "KMeans k=5", "sil": 0.0883, "DB": 2.9513, "CH": 789.03, "meaning": "building-floor zones"},
        {"config": "Agglom k=3 Ward", "sil": 0.0670, "DB": 3.1815, "CH": 936.98, "meaning": "buildings"},
        {"config": "Agglom k=5 Ward", "sil": 0.0784, "DB": 2.9978, "CH": 769.54, "meaning": "sub-wings"},
    ]))
    st.info("Elbow k=3-5, Ward linkage most balanced, PCA 10.5% separates buildings, t-SNE resolves corridors. Labels used post-hoc only.")
