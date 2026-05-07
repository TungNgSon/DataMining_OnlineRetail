from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
PROFILE_CSV = OUTPUT_DIR / "kmeans_segment_profile.csv"
CUSTOMER_CSV = OUTPUT_DIR / "customer_segments.csv"
METRICS_JSON = OUTPUT_DIR / "metrics_summary.json"
DASHBOARD_IMG = OUTPUT_DIR / "pipeline_real_results.png"
DBSCAN_PROFILE = OUTPUT_DIR / "dbscan_segment_profile.csv"

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


def run_pipeline() -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "pipeline_real.py")]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
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

    segment_options = ["All"] + profile_df["KM_Label"].tolist()
    db_profile = load_dbscan_profile()
    db_options = ["All"] + db_profile['Cluster'].astype(str).tolist() if not db_profile.empty else ["All"]

    # Hiển thị nguồn thuật toán và danh sách cụm để người dùng biết số liệu từ đâu
    st.markdown('**Model Sources & Cluster Lists**')
    alg_col1, alg_col2 = st.columns([1, 1])
    with alg_col1:
        st.markdown('**K-Means (source)**')
        st.markdown(f"- K = **{metrics['kmeans']['k_opt']}**")
        st.markdown(f"- Silhouette = **{metrics['kmeans']['silhouette']:.3f}**")
        km_list = profile_df['KM_Label'].unique().tolist()
        st.markdown('**Clusters:** ' + ', '.join(km_list))
    with alg_col2:
        st.markdown('**DBSCAN (source)**')
        st.markdown(f"- eps = **{metrics['dbscan']['eps']}** | min_pts = **{metrics['dbscan']['min_pts']}**")
        st.markdown(f"- Clusters = **{metrics['dbscan']['clusters']}** | Noise = **{metrics['dbscan']['noise_points']}**")
        if not db_profile.empty:
            st.markdown('**Clusters:** ' + ', '.join(db_profile['Cluster'].astype(str).tolist()))
        else:
            st.markdown('*Chưa có DBSCAN profile*')

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
        if not db_profile.empty:
            st.markdown('<div class="section-title">DBSCAN Cluster Distribution</div>', unsafe_allow_html=True)
            fig_db_pie = px.pie(db_profile.reset_index(), names='Cluster', values='Count', hole=0.4,
                                color_discrete_sequence=["#534AB7", "#1D9E75", "#D85A30", "#185FA5", "#993556"])
            fig_db_pie.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_db_pie, width='stretch')

    mid_left, mid_right = st.columns([1.6, 1])
    with mid_left:
        st.markdown('<div class="section-title">Scatter: Recency vs Monetary</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.info("Không có dữ liệu theo bộ lọc hiện tại.")
        else:
            fig_scatter = px.scatter(
                filtered,
                x="Recency",
                y="Monetary",
                color="KM_Label",
                symbol=filtered["Cluster_DB"].apply(lambda x: "Noise" if x == -1 else "Cluster"),
                hover_data=["CustomerID", "Frequency", "Cluster_DB"],
                opacity=0.75,
                color_discrete_sequence=["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"],
            )
            fig_scatter.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                legend_title_text="Segment",
            )
            st.plotly_chart(fig_scatter, width='stretch')

    with mid_right:
        st.markdown('<div class="section-title">Model Quality</div>', unsafe_allow_html=True)
        quality_df = pd.DataFrame(
            {
                "Model": ["KMeans", "DBSCAN", "KMeans", "DBSCAN"],
                "Metric": ["Silhouette", "Silhouette", "Davies-Bouldin", "Davies-Bouldin"],
                "Value": [
                    metrics["kmeans"]["silhouette"],
                    metrics["dbscan"]["silhouette"],
                    metrics["kmeans"]["davies_bouldin"],
                    metrics["dbscan"]["davies_bouldin"],
                ],
            }
        )
        fig_quality = px.bar(
            quality_df,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            color_discrete_map={"KMeans": "#3A86FF", "DBSCAN": "#FF006E"},
        )
        fig_quality.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_quality, width='stretch')

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

    with st.expander("Xem dashboard ảnh gốc từ pipeline"):
        if DASHBOARD_IMG.exists():
            st.image(str(DASHBOARD_IMG), caption="Dashboard gốc xuất từ pipeline_real.py", width='stretch')
        else:
            st.info("Chưa có ảnh dashboard gốc.")
