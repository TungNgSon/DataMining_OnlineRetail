"""
Data Mining Pipeline - Online Retail Dataset (UCI - THẬT) [ENHANCED]
Tham khảo: BTL-KDL-KPDL.docx (nhóm 13)
Thuật toán: K-Means + DBSCAN

THÊM MỚI:
  1. PCA 2D visualization + Silhouette subplot
  2. Lưu model bằng joblib (scaler + KMeans → tái sử dụng sau)
  3. Stability Test: chạy KMeans 10 lần với seed khác nhau
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def select_k_by_silhouette(X, k_min=2, k_max=10, random_state=42):
    inertias, silhouettes, models = [], [], {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, init='k-means++', random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        models[k] = model
    best_k = int(np.argmax(silhouettes) + k_min)
    return best_k, inertias, silhouettes, models[best_k]


def pick_dbscan_eps(X, min_pts=5, eps_grid=None):
    if eps_grid is None:
        eps_grid = np.arange(0.15, 1.05, 0.05)

    best = {
        'eps': None,
        'score': -1,
        'labels': None,
        'n_clusters': 0,
        'noise_ratio': 1.0,
    }

    for eps in eps_grid:
        model = DBSCAN(eps=float(round(eps, 3)), min_samples=min_pts)
        labels = model.fit_predict(X)
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float((labels == -1).sum() / len(labels))

        if n_clusters < 2 or n_clusters > 8:
            continue
        if mask.sum() < 30:
            continue

        score = silhouette_score(X[mask], labels[mask])

        if (
            score > best['score']
            or (abs(score - best['score']) < 1e-8 and noise_ratio < best['noise_ratio'])
        ):
            best = {
                'eps': float(round(eps, 3)),
                'score': float(score),
                'labels': labels,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
            }

    return best


def stability_test(X, k, n_runs=10, random_state_start=42):
    """
    Stability Test: Chạy KMeans n_runs lần với seed khác nhau
    để đánh giá ổn định của clustering
    
    Return: {
        'silhouette_scores': list (độ dài n_runs),
        'mean': float,
        'std': float,
        'coefficient_of_variation': float (std/mean),
    }
    """
    silhouette_scores = []
    for i in range(n_runs):
        model = KMeans(n_clusters=k, init='k-means++', random_state=random_state_start+i, n_init=20)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouette_scores.append(sil)
    
    silhouette_scores = np.array(silhouette_scores)
    mean_sil = float(silhouette_scores.mean())
    std_sil = float(silhouette_scores.std())
    cv = float(std_sil / mean_sil) if mean_sil > 0 else 0.0
    
    return {
        'silhouette_scores': [float(x) for x in silhouette_scores],
        'mean': mean_sil,
        'std': std_sil,
        'coefficient_of_variation': cv,
    }


def label_kmeans_segments(stats_df):
    r_order = stats_df['Recency'].rank(ascending=True)
    f_order = stats_df['Frequency'].rank(ascending=True)
    m_order = stats_df['Monetary'].rank(ascending=True)

    segment_labels = {}
    for cid in stats_df.index:
        recency_good = r_order[cid] <= len(stats_df) * 0.35
        value_high = (f_order[cid] + m_order[cid]) >= (len(stats_df) * 1.25)
        recency_bad = r_order[cid] >= len(stats_df) * 0.7

        if recency_good and value_high:
            segment_labels[cid] = 'Champions'
        elif recency_bad and stats_df.loc[cid, 'Monetary'] >= stats_df['Monetary'].median():
            segment_labels[cid] = 'At Risk Big Spenders'
        elif recency_bad:
            segment_labels[cid] = 'Hibernating'
        else:
            segment_labels[cid] = 'Potential Loyalists'

    used, normalized = {}, {}
    for cid, lbl in segment_labels.items():
        if lbl not in used:
            used[lbl] = 1
            normalized[cid] = lbl
        else:
            used[lbl] += 1
            normalized[cid] = f"{lbl} ({used[lbl]})"
    return normalized


def business_action_from_segment(name):
    if 'Champion' in name:
        return 'Upsell premium bundle + loyalty rewards'
    if 'At Risk' in name:
        return 'Win-back campaign with limited-time offer'
    if 'Hibernating' in name:
        return 'Low-cost reactivation via email automation'
    return 'Nurture with cross-sell and onboarding series'


# ─────────────────────────────────────────────
# 0. CẤU HÌNH
# ─────────────────────────────────────────────
def resolve_data_file():
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    preferred_paths = [
        Path('Online_Retail.xlsx'),
        Path('Online Retail.xlsx'),
        Path('DataMining_OnlineRetail/Online_Retail.xlsx'),
        Path('DataMining_OnlineRetail/Online Retail.xlsx'),
        script_dir / 'Online_Retail.xlsx',
        script_dir / 'Online Retail.xlsx',
        script_dir / 'DataMining_OnlineRetail' / 'Online_Retail.xlsx',
        script_dir / 'DataMining_OnlineRetail' / 'Online Retail.xlsx',
        cwd / 'Online_Retail.xlsx',
        cwd / 'Online Retail.xlsx',
    ]

    for candidate in preferred_paths:
        if candidate.exists():
            return candidate

    search_patterns = [
        '**/Online_Retail.xlsx',
        '**/Online Retail.xlsx',
        '**/Online_Retail*.xlsx',
        '**/Online Retail*.xlsx',
    ]
    matches = []
    for pattern in search_patterns:
        matches.extend(Path('.').glob(pattern))

    matches = [path for path in matches if path.is_file()]
    unique_matches = []
    seen = set()
    for path in matches:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_matches.append(path)

    if len(unique_matches) == 1:
        return unique_matches[0]

    return None


DATA_FILE = resolve_data_file()

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
MIN_PTS = 5

# ─────────────────────────────────────────────
# 0. ĐỌC DỮ LIỆU THẬT
# ─────────────────────────────────────────────
print("=" * 60)
print("BƯỚC 1: ĐỌC DỮ LIỆU THẬT (Online Retail UCI)")
print("=" * 60)

if DATA_FILE is None or not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu: {(Path('Online_Retail.xlsx').resolve())}\n"
        "Hãy đặt file Online_Retail.xlsx cùng thư mục với script."
    )

df = pd.read_excel(DATA_FILE)
rows_raw = len(df)

print(f"Tổng số bản ghi ban đầu : {len(df):,}")
print(f"Số cột                  : {df.shape[1]}")
print(f"Các cột: {list(df.columns)}")
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nKhoảng thời gian: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"\nMô tả Quantity và UnitPrice:")
print(df[['Quantity','UnitPrice']].describe().round(2))

# ─────────────────────────────────────────────
# 1. TIỀN XỬ LÝ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
print("=" * 60)

before = len(df)
df.dropna(subset=['CustomerID', 'Description'], inplace=True)
print(f"Xóa hàng null (CustomerID/Description): {before:,} → {len(df):,} hàng")

before = len(df)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"Loại invoice hủy (bắt đầu 'C')       : {before:,} → {len(df):,} hàng")

before = len(df)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
print(f"Loại Quantity âm / UnitPrice <= 0     : {before:,} → {len(df):,} hàng")

df['CustomerID'] = df['CustomerID'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"\nSau tiền xử lý: {len(df):,} bản ghi sạch")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING: RFM
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 3: XÂY DỰNG MÔ HÌNH RFM")
print("=" * 60)

df['TotalCost'] = df['Quantity'] * df['UnitPrice']
now = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Ngày mốc tính Recency: {now.date()}")

recency_df = (df.groupby('CustomerID')['InvoiceDate']
              .max().reset_index()
              .rename(columns={'InvoiceDate': 'LastPurchaseDate'}))
recency_df['Recency'] = (now - recency_df['LastPurchaseDate']).dt.days

freq_df = (df.drop_duplicates(subset=['InvoiceNo','CustomerID'])
           .groupby('CustomerID')['InvoiceNo']
           .count().reset_index()
           .rename(columns={'InvoiceNo': 'Frequency'}))

monetary_df = (df.groupby('CustomerID')['TotalCost']
               .sum().reset_index()
               .rename(columns={'TotalCost': 'Monetary'}))

rfm = (recency_df[['CustomerID','Recency']]
       .merge(freq_df, on='CustomerID')
       .merge(monetary_df, on='CustomerID'))
rfm.set_index('CustomerID', inplace=True)

print(f"\nBảng RFM: {len(rfm):,} khách hàng")
print(rfm.describe().round(2))

# ─────────────────────────────────────────────
# 3. XỬ LÝ NGOẠI LAI (IQR)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 4: XỬ LÝ NGOẠI LAI (IQR)")
print("=" * 60)

rfm_clean = rfm.copy()
before = len(rfm_clean)

mask = pd.Series(True, index=rfm_clean.index)
for col in ['Recency', 'Frequency', 'Monetary']:
    Q1 = rfm_clean[col].quantile(0.25)
    Q3 = rfm_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lo = Q1 - 1.5 * IQR
    hi = Q3 + 1.5 * IQR
    col_mask = (rfm_clean[col] >= lo) & (rfm_clean[col] <= hi)
    removed = (~col_mask).sum()
    mask &= col_mask
    print(
        f"  {col:10s}: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f} "
        f"→ đánh dấu {removed} outliers [lo={lo:.1f}, hi={hi:.1f}]"
    )

rfm_clean = rfm_clean[mask].copy()

print(f"\nSau IQR: {before:,} → {len(rfm_clean):,} khách hàng")

# ─────────────────────────────────────────────
# 4. CHUẨN HÓA (StandardScaler)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 5: CHUẨN HÓA DỮ LIỆU (StandardScaler)")
print("=" * 60)

rfm_model = rfm_clean.copy()
rfm_model['Frequency_log'] = np.log1p(rfm_model['Frequency'])
rfm_model['Monetary_log'] = np.log1p(rfm_model['Monetary'])

scaler = StandardScaler()
X = scaler.fit_transform(rfm_model[['Recency', 'Frequency_log', 'Monetary_log']])
print(f"Mean sau chuẩn hóa : {X.mean(axis=0).round(4)}")
print(f"Std  sau chuẩn hóa : {X.std(axis=0).round(4)}")

# ─────────────────────────────────────────────
# 5. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 6: K-MEANS CLUSTERING")
print("=" * 60)

K_MIN, K_MAX = 2, 9
K_OPT, inertias, sil_scores_km, km_final = select_k_by_silhouette(
    X,
    k_min=K_MIN,
    k_max=K_MAX,
    random_state=RANDOM_STATE,
)
K_range = range(K_MIN, K_MAX + 1)

rfm_clean = rfm_clean.copy()
rfm_clean['Cluster_KM'] = km_final.labels_

sil_km = silhouette_score(X, km_final.labels_)
db_km  = davies_bouldin_score(X, km_final.labels_)
print(f"K tối ưu (Silhouette)     : {K_OPT}")
print(f"Silhouette Score (K-Means): {sil_km:.4f}")
print(f"Davies-Bouldin Index      : {db_km:.4f}")

stats_km = rfm_clean.groupby('Cluster_KM')[['Recency','Frequency','Monetary']].mean()
km_names = label_kmeans_segments(stats_km)

rfm_clean['KM_Label'] = rfm_clean['Cluster_KM'].map(km_names)

print("\nThống kê các cụm K-Means:")
km_stats = rfm_clean.groupby('KM_Label')[['Recency','Frequency','Monetary']].mean().round(1)
km_stats['Count'] = rfm_clean.groupby('KM_Label').size()
print(km_stats)

# ─────────────────────────────────────────────
# 5a. STABILITY TEST (NEW)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 6a: STABILITY TEST - Chạy KMeans 10 lần (seed khác nhau)")
print("=" * 60)

stability_result = stability_test(X, k=K_OPT, n_runs=10, random_state_start=RANDOM_STATE)
print(f"Mean Silhouette (10 runs)   : {stability_result['mean']:.4f}")
print(f"Std Dev (10 runs)           : {stability_result['std']:.4f}")
print(f"Coefficient of Variation    : {stability_result['coefficient_of_variation']:.4f}")
if stability_result['coefficient_of_variation'] < 0.05:
    print("→ ✅ Model RẤT ỔNĐỊNH (CV < 0.05) – K=3 không phụ thuộc seed")
elif stability_result['coefficient_of_variation'] < 0.10:
    print("→ ✅ Model ỔNĐỊNH (CV < 0.10) – K=3 khá robust")
else:
    print("→ ⚠️  Model không ổn định (CV ≥ 0.10) – cân nhắc xét lại")

# ─────────────────────────────────────────────
# 6. DBSCAN CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 7: DBSCAN CLUSTERING")
print("=" * 60)

nbrs = NearestNeighbors(n_neighbors=MIN_PTS).fit(X)
distances, _ = nbrs.kneighbors(X)
k_dist = np.sort(distances[:, MIN_PTS-1])[::-1]
diffs2 = np.diff(np.diff(k_dist))
elbow_idx = np.argmax(np.abs(diffs2)) + 1
eps_auto = round(float(k_dist[elbow_idx]), 3)

db_best = pick_dbscan_eps(X, min_pts=MIN_PTS)
if db_best['eps'] is None:
    eps_opt = eps_auto
    dbscan = DBSCAN(eps=eps_opt, min_samples=MIN_PTS)
    db_labels = dbscan.fit_predict(X)
else:
    eps_opt = db_best['eps']
    db_labels = db_best['labels']

print(f"MinPts                    : {MIN_PTS}")
print(f"Epsilon (K-distance auto) : {eps_auto}")
print(f"Epsilon (dùng cho DBSCAN) : {eps_opt}")

if db_best['eps'] is not None:
    print(f"Silhouette tốt nhất (DBSCAN): {db_best['score']:.4f}")

rfm_clean['Cluster_DB'] = db_labels

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db    = (db_labels == -1).sum()
print(f"Số cluster tìm được : {n_clusters_db}")
print(f"Số noise points     : {n_noise_db:,} ({n_noise_db/len(db_labels)*100:.1f}%)")

X_nn = X[db_labels != -1]
labels_nn = db_labels[db_labels != -1]
if len(set(labels_nn)) > 1:
    sil_db = silhouette_score(X_nn, labels_nn)
    db_db  = davies_bouldin_score(X_nn, labels_nn)
else:
    sil_db, db_db = 0.0, 0.0
print(f"Silhouette Score (DBSCAN, bỏ noise): {sil_db:.4f}")
print(f"Davies-Bouldin Index               : {db_db:.4f}")

print("\nThống kê DBSCAN clusters:")
for cid in sorted(rfm_clean['Cluster_DB'].unique()):
    grp = rfm_clean[rfm_clean['Cluster_DB'] == cid]
    tag = 'NOISE' if cid == -1 else f'C{cid}'
    print(f"  [{tag}] {len(grp):>5} KH | "
          f"R={grp['Recency'].mean():>6.1f}d | "
          f"F={grp['Frequency'].mean():>5.1f}x | "
          f"M=£{grp['Monetary'].mean():>8.1f}")

# ─────────────────────────────────────────────
# 7. LƯU MODEL JOBLIB
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 8: LƯU MODEL BẰNG JOBLIB")
print("=" * 60)

joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
joblib.dump(km_final, OUTPUT_DIR / 'kmeans.pkl')
print(f"✓ Đã lưu scaler.pkl")
print(f"✓ Đã lưu kmeans.pkl")

# ─────────────────────────────────────────────
# 8. PCA + SILHOUETTE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 9: HÌNH HỌC VÀ CHẤT LƯỢNG (PCA + SILHOUETTE)")
print("=" * 60)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")

silhouette_vals = silhouette_samples(X, km_final.labels_)
print(f"✓ Tính toán silhouette per sample")

# ─────────────────────────────────────────────
# 9. VẼ BIỂU ĐỒ
# ─────────────────────────────────────────────
COLORS_KM = ['#534AB7','#1D9E75','#D85A30']
COLORS_DB = ['#534AB7','#1D9E75','#D85A30','#185FA5','#993556']
NOISE_C   = '#E24B4A'
TITLE_KW  = dict(fontsize=12, fontweight='bold', pad=8, color='#2C2C2A')
LAB_KW    = dict(fontsize=10, color='#5F5E5A')

km_profile = (
    rfm_clean.groupby('KM_Label')
    .agg(
        Count=('Recency', 'size'),
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean'),
        Revenue=('Monetary', 'sum')
    )
    .sort_values('Revenue', ascending=False)
)
km_profile['CustomerSharePct'] = km_profile['Count'] / km_profile['Count'].sum() * 100
km_profile['RevenueSharePct'] = km_profile['Revenue'] / km_profile['Revenue'].sum() * 100
km_profile['Action'] = [business_action_from_segment(x) for x in km_profile.index]

rfm_z = (km_profile[['Recency', 'Frequency', 'Monetary']] - km_profile[['Recency', 'Frequency', 'Monetary']].mean())
rfm_z = rfm_z / km_profile[['Recency', 'Frequency', 'Monetary']].std(ddof=0).replace(0, 1)

fig = plt.figure(figsize=(22, 32))
fig.patch.set_facecolor('#F8F8F6')
gs = gridspec.GridSpec(8, 3, figure=fig, hspace=0.55, wspace=0.35)

# Plot 1: Elbow
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(list(K_range), inertias, 'o-', color='#534AB7', lw=2, ms=6)
ax1.axvline(K_OPT, color='#D85A30', ls='--', lw=1.5, label=f'K={K_OPT}')
ax1.set_title('Elbow Method – K-Means', **TITLE_KW)
ax1.set_xlabel('Số cụm K', **LAB_KW); ax1.set_ylabel('Inertia (SSE)', **LAB_KW)
ax1.legend(fontsize=9); ax1.set_facecolor('white')

# Plot 2: Silhouette vs K
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(list(K_range), sil_scores_km, 's-', color='#1D9E75', lw=2, ms=6)
ax2.axvline(K_OPT, color='#D85A30', ls='--', lw=1.5, label=f'K={K_OPT}')
ax2.set_title('Silhouette Score vs K', **TITLE_KW)
ax2.set_xlabel('Số cụm K', **LAB_KW); ax2.set_ylabel('Silhouette Score', **LAB_KW)
ax2.legend(fontsize=9); ax2.set_facecolor('white')

# Plot 3: K-Distance Graph
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(k_dist, color='#534AB7', lw=1.5)
ax3.axhline(eps_opt, color='#D85A30', ls='--', lw=1.5, label=f'ε={eps_opt}')
ax3.axvline(elbow_idx, color='#1D9E75', ls=':', lw=1.5)
ax3.set_title(f'K-Distance Graph – DBSCAN\n(MinPts={MIN_PTS})', **TITLE_KW)
ax3.set_xlabel('Điểm (sorted)', **LAB_KW)
ax3.set_ylabel(f'{MIN_PTS}-th neighbor dist', **LAB_KW)
ax3.legend(fontsize=9); ax3.set_facecolor('white')

# Plot 4: PCA 2D K-Means
ax4 = fig.add_subplot(gs[1, :2])
for i, lbl in enumerate(sorted(rfm_clean['KM_Label'].unique())):
    m = rfm_clean['KM_Label'] == lbl
    indices = rfm_clean[m].index
    pca_indices = [idx for idx in indices if idx in rfm_clean.index]
    pca_mask = np.array([idx in pca_indices for idx in rfm_clean.index])
    ax4.scatter(X_pca[pca_mask, 0], X_pca[pca_mask, 1],
                c=COLORS_KM[i % len(COLORS_KM)], s=30, alpha=0.60, label=lbl, edgecolors='none')
ax4.set_title(f'K-Means: PCA 2D Projection\n(PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})', **TITLE_KW)
ax4.set_xlabel('First Principal Component', **LAB_KW)
ax4.set_ylabel('Second Principal Component', **LAB_KW)
ax4.legend(fontsize=9); ax4.set_facecolor('white')

# Plot 5: Pie K-Means
ax5 = fig.add_subplot(gs[1, 2])
km_c = rfm_clean['KM_Label'].value_counts()
km_colors = [COLORS_KM[i % len(COLORS_KM)] for i in range(len(km_c))]
ax5.pie(km_c, labels=km_c.index, colors=km_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize':8})
ax5.set_title('K-Means: Tỷ lệ phân khúc', **TITLE_KW)

# Plot 6: Silhouette Plot per cluster
ax6 = fig.add_subplot(gs[2, :])
y_lower = 10
colors_sil = ['#534AB7','#1D9E75','#D85A30','#185FA5','#993556']
for i in range(K_OPT):
    cluster_silhouette_vals = silhouette_vals[km_final.labels_ == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = colors_sil[i % len(colors_sil)]
    ax6.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_vals,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax6.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10, fontweight='bold')
    y_lower = y_upper + 10

ax6.axvline(x=sil_km, color='#D85A30', linestyle='--', lw=2, label=f'Mean Silhouette = {sil_km:.3f}')
ax6.set_title('Silhouette Plot per Cluster (K-Means)', **TITLE_KW)
ax6.set_xlabel('Silhouette Coefficient', **LAB_KW)
ax6.set_ylabel('Cluster Label', **LAB_KW)
ax6.set_ylim([0, len(X) + (K_OPT + 1) * 10])
ax6.legend(fontsize=10)
ax6.set_facecolor('white')

# Plot 7: Heatmap z-score RFM
ax7 = fig.add_subplot(gs[3, :2])
hm = ax7.imshow(rfm_z.values, cmap='RdYlGn_r', aspect='auto', vmin=-2.5, vmax=2.5)
ax7.set_yticks(np.arange(len(rfm_z.index)))
ax7.set_yticklabels(rfm_z.index, fontsize=9)
ax7.set_xticks(np.arange(3))
ax7.set_xticklabels(['Recency', 'Frequency', 'Monetary'], fontsize=10)
ax7.set_title('K-Means Segment Profile (z-score)', **TITLE_KW)
for i in range(rfm_z.shape[0]):
    for j in range(rfm_z.shape[1]):
        ax7.text(j, i, f"{rfm_z.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8, color='#1F1F1F')
cbar = fig.colorbar(hm, ax=ax7, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
ax7.set_facecolor('white')

# Plot 8: Customer share vs Revenue share
ax8 = fig.add_subplot(gs[3, 2])
x = np.arange(len(km_profile.index))
bw = 0.38
ax8.bar(x - bw/2, km_profile['CustomerSharePct'], width=bw, color='#1D9E75', alpha=0.9, label='Customer %')
ax8.bar(x + bw/2, km_profile['RevenueSharePct'], width=bw, color='#D85A30', alpha=0.9, label='Revenue %')
ax8.set_xticks(x)
ax8.set_xticklabels(km_profile.index, rotation=30, ha='right', fontsize=8)
ax8.set_ylabel('Tỷ trọng (%)', **LAB_KW)
ax8.set_title('K-Means: Customer vs Revenue Share', **TITLE_KW)
ax8.legend(fontsize=8)
ax8.set_facecolor('white')

# Plot 9: Scatter DBSCAN
ax9 = fig.add_subplot(gs[4, :2])
noise_m = rfm_clean['Cluster_DB'] == -1
ax9.scatter(rfm_clean[noise_m]['Recency'], rfm_clean[noise_m]['Monetary'],
            c=NOISE_C, s=50, marker='x', alpha=0.9,
            label=f'Noise ({n_noise_db} KH – wholesalers/bất thường)', zorder=5, lw=1.2)
for cid in sorted(rfm_clean[~noise_m]['Cluster_DB'].unique()):
    m = rfm_clean['Cluster_DB'] == cid
    ax9.scatter(rfm_clean[m]['Recency'], rfm_clean[m]['Monetary'],
                c=COLORS_DB[cid % len(COLORS_DB)], s=25, alpha=0.55,
                label=f'Cluster {cid}', edgecolors='none')
ax9.set_title('DBSCAN: Recency vs Monetary\n(dấu ✕ đỏ = noise/outlier DBSCAN tự phát hiện)', **TITLE_KW)
ax9.set_xlabel('Recency (ngày)', **LAB_KW)
ax9.set_ylabel('Monetary (£)', **LAB_KW)
ax9.legend(fontsize=9); ax9.set_facecolor('white')

# Plot 10: Pie DBSCAN
ax10 = fig.add_subplot(gs[4, 2])
db_c  = rfm_clean['Cluster_DB'].value_counts().sort_index()
p_lbl = [f'C{i}' if i != -1 else 'Noise' for i in db_c.index]
p_col = [COLORS_DB[i % len(COLORS_DB)] if i != -1 else NOISE_C for i in db_c.index]
ax10.pie(db_c, labels=p_lbl, colors=p_col,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize':8})
ax10.set_title('DBSCAN: Cluster + Noise', **TITLE_KW)

# Plot 11: So sánh metrics
ax11 = fig.add_subplot(gs[5, :])
metrics = {
    'Silhouette Score\n(cao hơn = tốt hơn)': [sil_km, sil_db],
    'Davies-Bouldin Index\n(thấp hơn = tốt hơn)': [db_km, db_db],
}
bar_c = ['#534AB7','#1D9E75']
for i, (metric, vals) in enumerate(metrics.items()):
    for j, val in enumerate(vals):
        lbl = ['K-Means','DBSCAN'][j]
        ax11.bar(i*3 + j*1.1, val, 0.95,
                label=lbl if i == 0 else '_',
                color=bar_c[j], alpha=0.85)
        ax11.text(i*3 + j*1.1, val + 0.005, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2C2C2A')
ax11.set_xticks([0.55, 3.55])
ax11.set_xticklabels(list(metrics.keys()), fontsize=11)
ax11.set_ylabel('Giá trị', **LAB_KW)
ax11.set_title('So sánh độ đo: K-Means vs DBSCAN', **TITLE_KW)
ax11.legend(fontsize=10); ax11.set_facecolor('white')

# Plot 12: Action table
ax12 = fig.add_subplot(gs[6:, :])
ax12.axis('off')
table_cols = ['Count', 'CustomerSharePct', 'RevenueSharePct', 'Action']
table_data = km_profile[table_cols].copy()
table_data['CustomerSharePct'] = table_data['CustomerSharePct'].map(lambda v: f"{v:.1f}%")
table_data['RevenueSharePct'] = table_data['RevenueSharePct'].map(lambda v: f"{v:.1f}%")
table_data['Count'] = table_data['Count'].astype(int)

tbl = ax12.table(
    cellText=table_data.values,
    rowLabels=table_data.index,
    colLabels=['Count', 'Customer %', 'Revenue %', 'Recommended Action'],
    loc='center',
    cellLoc='left',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.4)
ax12.set_title('Actionable Insight theo Segment (K-Means)', **TITLE_KW)

fig.suptitle('Data Mining – Online Retail UCI (THẬT) [ENHANCED]\nCustomer Segmentation: K-Means vs DBSCAN (RFM Analysis)\n📊 Với PCA 2D + Silhouette Plot Chi Tiết + Joblib Models',
             fontsize=15, fontweight='bold', color='#2C2C2A', y=0.998)

plt.savefig(OUTPUT_DIR / 'pipeline_new_results.png',
            dpi=150, bbox_inches='tight', facecolor='#F8F8F6')
plt.close()
print("\n✓ Biểu đồ đã lưu (pipeline_new_results.png).")

# ─────────────────────────────────────────────
# 10. XUẤT DỮ LIỆU
# ─────────────────────────────────────────────
km_profile_export = km_profile.copy()
km_profile_export[['Recency', 'Frequency', 'Monetary', 'Revenue']] = km_profile_export[
    ['Recency', 'Frequency', 'Monetary', 'Revenue']
].round(2)
km_profile_export[['CustomerSharePct', 'RevenueSharePct']] = km_profile_export[
    ['CustomerSharePct', 'RevenueSharePct']
].round(2)
km_profile_export.to_csv(OUTPUT_DIR / 'kmeans_segment_profile.csv', encoding='utf-8-sig')
print("✓ Đã lưu file profile cụm K-Means.")

customer_export = rfm_clean.reset_index()[
    ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'KM_Label', 'Cluster_DB']
].copy()
customer_export.to_csv(OUTPUT_DIR / 'customer_segments.csv', index=False, encoding='utf-8-sig')
print("✓ Đã lưu file customer-level segmentation.")

metrics_payload = {
    'kmeans': {
        'k_opt': int(K_OPT),
        'silhouette': float(round(sil_km, 4)),
        'davies_bouldin': float(round(db_km, 4)),
        'stability_test': {
            'n_runs': int(10),
            'silhouette_scores': stability_result['silhouette_scores'],
            'mean_silhouette': float(round(stability_result['mean'], 4)),
            'std_silhouette': float(round(stability_result['std'], 4)),
            'coefficient_of_variation': float(round(stability_result['coefficient_of_variation'], 4)),
        }
    },
    'dbscan': {
        'min_pts': int(MIN_PTS),
        'eps': float(eps_opt),
        'clusters': int(n_clusters_db),
        'noise_points': int(n_noise_db),
        'silhouette': float(round(sil_db, 4)),
        'davies_bouldin': float(round(db_db, 4)),
    },
    'dataset': {
        'rows_raw': int(rows_raw),
        'rows_clean': int(len(df)),
        'customers_rfm': int(len(rfm)),
        'customers_modeled': int(len(rfm_clean)),
    },
}

with open(OUTPUT_DIR / 'metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
print("✓ Đã lưu file metrics summary.")

db_profile = (
    rfm_clean.reset_index()
    .groupby('Cluster_DB')
    .agg(
        Count=('CustomerID', 'size'),
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean'),
        Revenue=('Monetary', 'sum')
    )
    .sort_values('Count', ascending=False)
)
db_profile = db_profile.rename(index={-1: 'Noise'})
db_profile['Cluster'] = db_profile.index.astype(str)
db_profile['CustomerSharePct'] = db_profile['Count'] / db_profile['Count'].sum() * 100
db_profile['RevenueSharePct'] = db_profile['Revenue'] / db_profile['Revenue'].sum() * 100
db_profile.to_csv(OUTPUT_DIR / 'dbscan_segment_profile.csv', encoding='utf-8-sig')
print("✓ Đã lưu file DBSCAN profile.")

# ─────────────────────────────────────────────
# 11. INSIGHT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BƯỚC 10: INSIGHT PHÂN KHÚC KHÁCH HÀNG")
print("=" * 60)

print(f"\n── K-MEANS ({K_OPT} cụm) ──")
for lbl in km_profile.index:
    g = rfm_clean[rfm_clean['KM_Label'] == lbl]
    rev_share = km_profile.loc[lbl, 'RevenueSharePct']
    cus_share = km_profile.loc[lbl, 'CustomerSharePct']
    print(f"\n[{lbl}] — {len(g):,} KH ({len(g)/len(rfm_clean)*100:.1f}%)")
    print(f"  Recency  : {g['Recency'].mean():.0f} ngày")
    print(f"  Frequency: {g['Frequency'].mean():.1f} lần")
    print(f"  Monetary : £{g['Monetary'].mean():,.0f}")
    print(f"  Revenue share: {rev_share:.1f}% | Customer share: {cus_share:.1f}%")
    print(f"  Action: {business_action_from_segment(lbl)}")

print(f"\n── DBSCAN ({n_clusters_db} clusters + noise) ──")
for cid in sorted(rfm_clean['Cluster_DB'].unique()):
    g = rfm_clean[rfm_clean['Cluster_DB'] == cid]
    tag = 'NOISE – wholesalers/giao dịch bất thường' if cid == -1 else f'Cluster {cid}'
    print(f"\n[{tag}] — {len(g):,} KH ({len(g)/len(rfm_clean)*100:.1f}%)")
    print(f"  Recency  : {g['Recency'].mean():.0f} ngày")
    print(f"  Frequency: {g['Frequency'].mean():.1f} lần")
    print(f"  Monetary : £{g['Monetary'].mean():,.0f}")

print("\n── SO SÁNH ĐỘ ĐO ──")
print(f"{'Metric':<38} {'K-Means':>10} {'DBSCAN':>10}")
print("-" * 60)
print(f"{'Silhouette Score (cao = tốt)':<38} {sil_km:>10.4f} {sil_db:>10.4f}")
print(f"{'Davies-Bouldin Index (thấp = tốt)':<38} {db_km:>10.4f} {db_db:>10.4f}")
print(f"{'Số cluster':<38} {K_OPT:>10} {n_clusters_db:>10}")
print(f"{'Noise tự phát hiện':<38} {'N/A':>10} {n_noise_db:>10}")

print("\n── STABILITY TEST (K-Means) ──")
print(f"{'Mean Silhouette':<38} {stability_result['mean']:>10.4f}")
print(f"{'Std Dev':<38} {stability_result['std']:>10.4f}")
print(f"{'Coefficient of Variation':<38} {stability_result['coefficient_of_variation']:>10.4f}")
if stability_result['coefficient_of_variation'] < 0.05:
    print("✅ Model RẤT ỔNĐỊNH – K không phụ thuộc seed")
elif stability_result['coefficient_of_variation'] < 0.10:
    print("✅ Model ỔNĐỊNH – K khá robust")
else:
    print("⚠️  Model không ổn định – cân nhắc xét lại K")

print("\nKết quả đầu ra:")
print(f"  - Ảnh dashboard: {(OUTPUT_DIR / 'pipeline_new_results.png').resolve()}")
print(f"  - CSV profile  : {(OUTPUT_DIR / 'kmeans_segment_profile.csv').resolve()}")
print(f"  - CSV customer : {(OUTPUT_DIR / 'customer_segments.csv').resolve()}")
print(f"  - JSON metrics : {(OUTPUT_DIR / 'metrics_summary.json').resolve()}")
print(f"  - Joblib models:")
print(f"    • Scaler: {(OUTPUT_DIR / 'scaler.pkl').resolve()}")
print(f"    • KMeans: {(OUTPUT_DIR / 'kmeans.pkl').resolve()}")

print("\n" + "=" * 60)
print("✓ HOÀN THÀNH PIPELINE NEW (ENHANCED + STABILITY TEST)")
print("=" * 60)
