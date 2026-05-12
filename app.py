from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PROFILE_CSV = OUTPUT_DIR / "kmeans_segment_profile.csv"
CUSTOMER_CSV = OUTPUT_DIR / "customer_segments.csv"
METRICS_JSON = OUTPUT_DIR / "metrics_summary.json"
DASHBOARD_IMG = OUTPUT_DIR / "pipeline_new_results.png"
DBSCAN_PROFILE = OUTPUT_DIR / "dbscan_segment_profile.csv"
SCALER_PKL = OUTPUT_DIR / "scaler.pkl"
KMEANS_PKL = OUTPUT_DIR / "kmeans.pkl"

st.set_page_config(
    page_title="Data Mining Demo - Online Retail",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }

    .hero {
        background: linear-gradient(140deg, #10375c 0%, #135d66 45%, #3da35d 100%);
        border-radius: 18px;
        color: #f8f8f2;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(16, 55, 92, 0.20);
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        margin: 0;
        letter-spacing: 0.4px;
    }

    .hero p {
        margin: 0.3rem 0 0;
        opacity: 0.95;
    }

    .kpi-card {
        border-radius: 14px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        border: 1px solid rgba(16, 55, 92, 0.12);
        box-shadow: 0 8px 20px rgba(16, 55, 92, 0.08);
    }

    .kpi-title {
        font-size: 0.9rem;
        color: #546A7B;
        margin-bottom: 0.15rem;
    }

    .kpi-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #10375c;
    }

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #10375c;
        margin-top: 0.3rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_profile() -> pd.DataFrame:
    df = pd.read_csv(PROFILE_CSV)
    return df.sort_values("RevenueSharePct", ascending=False)


@st.cache_data
def load_customer() -> pd.DataFrame:
    return pd.read_csv(CUSTOMER_CSV)


@st.cache_data
def load_dbscan_profile() -> pd.DataFrame:
    if DBSCAN_PROFILE.exists():
        return pd.read_csv(DBSCAN_PROFILE)
    return pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models():
    if SCALER_PKL.exists() and KMEANS_PKL.exists():
        return joblib.load(SCALER_PKL), joblib.load(KMEANS_PKL)
    return None, None


def run_pipeline() -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT.parent / "pipeline_new.py")]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_pct(v: float) -> str:
    return f"{v:.1f}%"


st.markdown(
    """
    <div class="hero">
        <h1>Online Retail Mining Dashboard</h1>
        <p>KMeans + DBSCAN trên RFM, tập trung insight kinh doanh và hành động cụ thể.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not PROFILE_CSV.exists() or not METRICS_JSON.exists() or not CUSTOMER_CSV.exists():
    st.warning("Chưa có đầy đủ output. Hãy bấm 'Run/Rebuild Pipeline' để tạo dữ liệu cho dashboard.")

col_a, col_b = st.columns([1.1, 2.4])
with col_a:
    if st.button("Run/Rebuild Pipeline", width='stretch'):
        ok, logs = run_pipeline()
        if ok:
            st.success("Pipeline chạy thành công. Dữ liệu dashboard đã được cập nhật.")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Pipeline bị lỗi. Kiểm tra logs bên dưới.")
            st.code(logs, language="text")

with col_b:
    st.caption("Nguồn dữ liệu: Online Retail UCI | Mục tiêu: phân khúc khách hàng và phát hiện hành vi bất thường")

if PROFILE_CSV.exists() and METRICS_JSON.exists() and CUSTOMER_CSV.exists():
    profile_df = load_profile()
    customer_df = load_customer()
    metrics = load_metrics()
    scaler, kmeans_model = load_models()

    segment_options = ["All"] + profile_df["KM_Label"].tolist()
    db_profile = load_dbscan_profile()
    db_options = ["All"] + db_profile['Cluster'].astype(str).tolist() if not db_profile.empty else ["All"]

    filter_col_1, filter_col_2, filter_col_3 = st.columns([1.2, 1.2, 1.4])
    with filter_col_1:
        selected_segment = st.selectbox("Filter segment", options=segment_options, index=0)
    with filter_col_2:
        show_noise_only = st.checkbox("Chỉ xem DBSCAN noise", value=False)
    with filter_col_3:
        selected_db = st.selectbox("Filter DBSCAN cluster", options=db_options, index=0)
    with filter_col_3:
        min_monetary = st.slider(
            "Monetary tối thiểu",
            min_value=0,
            max_value=int(customer_df["Monetary"].max()),
            value=0,
            step=50,
        )

    filtered = customer_df.copy()
    if selected_segment != "All":
        filtered = filtered[filtered["KM_Label"] == selected_segment]
    if show_noise_only:
        filtered = filtered[filtered["Cluster_DB"] == -1]
    if selected_db != "All":
        # Cluster labels are strings in db_profile; map 'Noise' -> -1 else int
        if selected_db == 'Noise':
            filtered = filtered[filtered["Cluster_DB"] == -1]
        else:
            try:
                cid = int(selected_db)
                filtered = filtered[filtered["Cluster_DB"] == cid]
            except Exception:
                pass
    filtered = filtered[filtered["Monetary"] >= min_monetary]

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        kpi("Customers modeled", f"{metrics['dataset']['customers_modeled']:,}")
    with kpi_cols[1]:
        kpi("KMeans K", str(metrics["kmeans"]["k_opt"]))
    with kpi_cols[2]:
        kpi("KMeans Silhouette", f"{metrics['kmeans']['silhouette']:.3f}")
    with kpi_cols[3]:
        kpi("DBSCAN Clusters", str(metrics["dbscan"]["clusters"]))
    with kpi_cols[4]:
        kpi("DBSCAN Noise", f"{metrics['dbscan']['noise_points']}")

    top_left, top_right = st.columns([1.3, 1])
    with top_left:
        st.markdown('<div class="section-title">Customer vs Revenue Share theo Segment</div>', unsafe_allow_html=True)
        melt_df = profile_df[["KM_Label", "CustomerSharePct", "RevenueSharePct"]].melt(
            id_vars="KM_Label", var_name="Type", value_name="Percent"
        )
        fig_share = px.bar(
            melt_df,
            x="KM_Label",
            y="Percent",
            color="Type",
            barmode="group",
            color_discrete_map={
                "CustomerSharePct": "#2A9D8F",
                "RevenueSharePct": "#E76F51",
            },
        )
        fig_share.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="",
            xaxis_title="",
            yaxis_title="Percent",
        )
        st.plotly_chart(fig_share, width='stretch')

    # ── Stability Test Results ──────────────────────────────────────────
    with st.expander("🔍 **Model Stability Test** (K-Means Robustness)", expanded=False):
        if "stability_test" in metrics.get("kmeans", {}):
            stab = metrics["kmeans"]["stability_test"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Runs", f"{stab['n_runs']}")
            with col2:
                st.metric("Mean Silhouette", f"{stab['mean_silhouette']:.4f}")
            with col3:
                st.metric("Std Dev", f"{stab['std_silhouette']:.4f}")
            with col4:
                cv = stab['coefficient_of_variation']
                if cv < 0.05:
                    status = "✅ RẤT ỔNĐỊNH"
                elif cv < 0.10:
                    status = "✅ ỔNĐỊNH"
                else:
                    status = "⚠️ KHÔNG ỔNĐỊNH"
                st.metric("Coefficient of Variation", f"{cv:.4f}", delta=status)
            
            st.write("")
            st.write("**Diễn giải:**")
            st.write("""
            - **Stability Test** chạy K-Means 10 lần với random seed khác nhau
            - **Mean Silhouette**: giá trị Silhouette trung bình qua 10 lần chạy
            - **Std Dev**: độ lệch chuẩn của Silhouette (thấp hơn = ổn định hơn)
            - **Coefficient of Variation (CV)**: std/mean → đo độ ổn định tương đối
              - CV < 0.05: ✅ **RẤT ỔNĐỊNH** – K=3 không phụ thuộc seed, mạnh mẽ
              - CV < 0.10: ✅ **ỔNĐỊNH** – K=3 khá robust
              - CV ≥ 0.10: ⚠️ **KHÔNG ỔNĐỊNH** – kết quả dễ thay đổi
            """)
            
            # Vẽ stability trend
            stab_df = pd.DataFrame({
                "Run": list(range(1, len(stab['silhouette_scores'])+1)),
                "Silhouette": stab['silhouette_scores']
            })
            fig_stab = px.line(stab_df, x="Run", y="Silhouette", markers=True, 
                              title="Silhouette Score qua 10 Runs",
                              color_discrete_sequence=["#3A86FF"])
            fig_stab.add_hline(y=stab['mean_silhouette'], line_dash="dash", 
                              line_color="red", annotation_text=f"Mean: {stab['mean_silhouette']:.4f}")
            fig_stab.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_stab, use_container_width=True, key="stability_test_chart")
        else:
            st.info("Chưa có Stability Test data. Hãy chạy lại pipeline.")
    
    top_left, top_right = st.columns([1.3, 1])
    with top_right:
        st.markdown('<div class="section-title">Phân bổ KMeans Segment</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            profile_df,
            names="KM_Label",
            values="Count",
            color_discrete_sequence=["#0E9594", "#127475", "#F4B942", "#D95D39", "#1D3557"],
            hole=0.45,
        )
        fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, width='stretch')

    mid_left, mid_right = st.columns([1.6, 1])

    # ── Tính PCA + Silhouette một lần, dùng chung cho cả hai chart ──
    if scaler is not None and kmeans_model is not None:
        model_df = customer_df.copy()
        model_df["Frequency_log"] = np.log1p(model_df["Frequency"])
        model_df["Monetary_log"]  = np.log1p(model_df["Monetary"])
        X_model   = scaler.transform(model_df[["Recency", "Frequency_log", "Monetary_log"]])
        pca       = PCA(n_components=2, random_state=42)
        X_pca     = pca.fit_transform(X_model)
        cluster_ids = kmeans_model.predict(X_model)
        sil_vals    = silhouette_samples(X_model, cluster_ids)

        pca_df = model_df[["CustomerID", "KM_Label", "Cluster_DB"]].copy()
        pca_df["PCA1"]       = X_pca[:, 0]
        pca_df["PCA2"]       = X_pca[:, 1]
        pca_df["DBSCAN_Type"] = pca_df["Cluster_DB"].apply(lambda x: "Noise" if x == -1 else "Cluster")

        sil_df = pd.DataFrame({
            "Cluster":   cluster_ids,
            "Silhouette": sil_vals,
            "KM_Label":  model_df["KM_Label"].values,
        }).sort_values(["Cluster", "Silhouette"], ascending=[True, False])

        cluster_to_label = (
            sil_df[["Cluster", "KM_Label"]]
            .drop_duplicates("Cluster")
            .set_index("Cluster")["KM_Label"]
            .to_dict()
        )
        COLORS_SIL = ["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"]

    with mid_left:
        st.markdown('<div class="section-title">PCA 2D Projection (RFM)</div>', unsafe_allow_html=True)
        if scaler is None or kmeans_model is None:
            st.info("Chưa tìm thấy model. Hãy chạy lại pipeline.")
        elif filtered.empty:
            st.info("Không có dữ liệu theo bộ lọc hiện tại.")
        else:
            # Lọc pca_df theo filter hiện tại
            pca_filtered = pca_df[pca_df["CustomerID"].isin(filtered["CustomerID"])]
            pca_fig = px.scatter(
                pca_filtered,
                x="PCA1", y="PCA2",
                color="KM_Label",
                symbol="DBSCAN_Type",
                hover_data=["CustomerID", "Cluster_DB"],
                title=f"PC1={pca.explained_variance_ratio_[0]:.1%} | PC2={pca.explained_variance_ratio_[1]:.1%} | Total={sum(pca.explained_variance_ratio_):.1%}",
                opacity=0.75,
                color_discrete_sequence=["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"],
            )
            pca_fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                legend_title_text="Segment",
            )
            st.plotly_chart(pca_fig, width='stretch')

    with mid_right:
        st.markdown('<div class="section-title">Model Quality</div>', unsafe_allow_html=True)
        quality_df = pd.DataFrame(
            {
                "Model":  ["KMeans", "DBSCAN", "KMeans", "DBSCAN"],
                "Metric": ["Silhouette", "Silhouette", "Davies-Bouldin", "Davies-Bouldin"],
                "Value":  [
                    metrics["kmeans"]["silhouette"],
                    metrics["dbscan"]["silhouette"],
                    metrics["kmeans"]["davies_bouldin"],
                    metrics["dbscan"]["davies_bouldin"],
                ],
            }
        )
        fig_quality = px.bar(
            quality_df, x="Metric", y="Value", color="Model", barmode="group",
            color_discrete_map={"KMeans": "#3A86FF", "DBSCAN": "#FF006E"},
        )
        fig_quality.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_quality, width='stretch')

    # ── Silhouette chart full-width ──────────────────────────────────────
    if scaler is not None and kmeans_model is not None:
        st.markdown('<div class="section-title">Silhouette Distribution per Segment</div>', unsafe_allow_html=True)
        sil_fig = go.Figure()
        y_base = 0
        ytick_vals, ytick_texts = [], []
        for i, cluster_id in enumerate(sorted(sil_df["Cluster"].unique())):
            cluster_data = sil_df[sil_df["Cluster"] == cluster_id]
            seg_name = cluster_to_label.get(cluster_id, f"Cluster {cluster_id}")
            avg_sil  = cluster_data["Silhouette"].mean()
            y_positions = list(range(y_base, y_base + len(cluster_data)))
            ytick_vals.append(y_base + len(cluster_data) / 2)
            ytick_texts.append(f"<b>{seg_name}</b><br>(n={len(cluster_data)}, avg={avg_sil:.3f})")
            sil_fig.add_trace(go.Bar(
                x=cluster_data["Silhouette"],
                y=y_positions,
                orientation="h",
                name=seg_name,
                marker_color=COLORS_SIL[i % len(COLORS_SIL)],
                hovertemplate=f"<b>{seg_name}</b><br>Silhouette=%{{x:.3f}}<extra></extra>",
                opacity=0.82,
            ))
            y_base += len(cluster_data) + 20

        sil_fig.update_layout(
            barmode="overlay",
            height=380,
            margin=dict(l=160, r=10, t=10, b=10),
            xaxis_title="Silhouette coefficient",
            yaxis=dict(tickvals=ytick_vals, ticktext=ytick_texts, tickfont=dict(size=11)),
            legend_title_text="Segment",
        )
        sil_fig.add_vline(
            x=float(metrics["kmeans"]["silhouette"]),
            line_dash="dash", line_color="#D85A30",
            annotation_text=f"Avg={metrics['kmeans']['silhouette']:.3f}",
            annotation_position="top right",
        )
        st.plotly_chart(sil_fig, width='stretch')

    st.markdown('<div class="section-title">Actionable Segment Table</div>', unsafe_allow_html=True)
    action_df = profile_df[["KM_Label", "Count", "CustomerSharePct", "RevenueSharePct", "Action"]].copy()
    action_df["CustomerSharePct"] = action_df["CustomerSharePct"].map(format_pct)
    action_df["RevenueSharePct"] = action_df["RevenueSharePct"].map(format_pct)
    st.dataframe(action_df, width='stretch', hide_index=True)

    if not db_profile.empty:
        st.markdown('<div class="section-title">DBSCAN Cluster Table</div>', unsafe_allow_html=True)
        db_show = db_profile[['Cluster', 'Count', 'CustomerSharePct', 'RevenueSharePct', 'Recency', 'Frequency', 'Monetary']].copy()
        db_show['CustomerSharePct'] = db_show['CustomerSharePct'].map(format_pct)
        db_show['RevenueSharePct'] = db_show['RevenueSharePct'].map(format_pct)
        st.dataframe(db_show.reset_index(drop=True), width='stretch')