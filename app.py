import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Marketing Data Science Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình giao diện biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Tựa đề chính của Dashboard
st.title("📊 Dashboard Phân Tích Dữ Liệu & Dự Đoán Chi Tiêu Khách Hàng")
st.markdown("Ứng dụng tích hợp quy trình: **Tiền xử lý ➡️ Trực quan hóa ➡️ Phân cụm (K-Means) ➡️ Dự đoán (Linear Regression)**")

# ==========================================
# TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU (Dùng cache để tối ưu tốc độ)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    # Đọc file dữ liệu phân tách bằng tab
    df = pd.read_csv('marketing_campaign.csv', sep='\t')
    
    # 1. Xử lý giá trị trống ở cột Income bằng trung vị
    df['Income'] = df['Income'].fillna(df['Income'].median())
    
    # 2. Tạo đặc trưng mới (Feature Engineering)
    df['Age'] = 2026 - df['Year_Birth']
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Mnt'] = df[mnt_cols].sum(axis=1)
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    
    # 3. Lọc nhiễu Outliers
    df = df[(df['Age'] < 100) & (df['Income'] < 600000)]
    return df

# Tải dữ liệu vào ứng dụng
try:
    df = load_and_preprocess_data()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'marketing_campaign.csv'. Hãy đảm bảo file nằm cùng thư mục với đoạn code này!")
    st.stop()

# ==========================================
# THANH SIDEBAR (CÁC THÔNG SỐ CẤU HÌNH)
# ==========================================
st.sidebar.header("⚙️ Cấu Hình Mô Hình")
st.sidebar.markdown("Điều chỉnh các tham số dưới đây để cập nhật mô hình tự động:")

# Cho phép người dùng tùy chọn số cụm K-Means ngay trên giao diện
num_clusters = st.sidebar.slider("Số lượng cụm (K-Means):", min_value=2, max_value=6, value=3, step=1)

# Định nghĩa các biến đặc trưng
features = ['Income', 'Age', 'Recency', 'NumWebPurchases', 'NumStorePurchases']

# Hiển thị thông tin tổng quan ở Sidebar
st.sidebar.markdown("---")
st.sidebar.write(f"**Tổng số mẫu sau làm sạch:** {df.shape[0]}")
st.sidebar.write(f"**Số lượng đặc trưng sử dụng:** {len(features)}")

# ==========================================
# THIẾT KẾ DASHBOARD NHIỀU TAB
# ==========================================
# Khởi tạo 3 Tab chính
tab1, tab2, tab3 = st.tabs([
    "📈 1. Trực Quan Hóa Dữ Liệu", 
    "🎯 2. Phân Cụm Khách Hàng (K-Means)", 
    "🤖 3. Dự Đoán Chi Tiêu (Hồi Quy Tuyến Tính)"
])

# ------------------------------------------
# TAB 1: TRỰC QUAN HÓA DỮ LIỆU (5 BIỂU ĐỒ)
# ------------------------------------------
with tab1:
    st.header("🔍 Trực Quan Hóa & Phám Phá Dữ Liệu")
    st.write("Phần này hiển thị 5 biểu đồ phân tích hành vi và thuộc tính của khách hàng trước khi đưa vào thuật toán.")
    
    # Chia layout thành các cột để hiển thị biểu đồ cân đối
    col1, col2 = st.columns(2)
    
    with col1:
        # Biểu đồ 1: Phân phối thu nhập
        st.subheader("Biểu đồ 1: Phân Phối Thu Nhập")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['Income'], kde=True, color='skyblue', bins=30, ax=ax1)
        ax1.set_xlabel('Thu nhập (Income)')
        ax1.set_ylabel('Số lượng')
        st.pyplot(fig1)
        
        # Biểu đồ 3: Hôn nhân vs Chi tiêu
        st.subheader("Biểu đồ 3: Chi Tiêu Theo Tình Trạng Hôn Nhân")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='Marital_Status', y='Total_Mnt', palette='Set3', ax=ax3)
        plt.xticks(rotation=45)
        st.pyplot(fig3)

    with col2:
        # Biểu đồ 2: Thu nhập vs Chi tiêu
        st.subheader("Biểu đồ 2: Mối Quan Hệ Thu Nhập & Chi Tiêu")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='Income', y='Total_Mnt', alpha=0.6, color='purple', ax=ax2)
        ax2.set_xlabel('Thu nhập (Income)')
        ax2.set_ylabel('Tổng chi tiêu (Total_Mnt)')
        st.pyplot(fig2)
        
        # Biểu đồ 4: Số con cái vs Chi tiêu
        st.subheader("Biểu đồ 4: Chi Tiêu Trung Bình Theo Số Con Cái")
        fig4, ax4 = plt.subplots()
        sns.barplot(data=df, x='Total_Children', y='Total_Mnt', errorbar=None, palette='coolwarm', ax=ax4)
        ax4.set_xlabel('Tổng số con cái')
        ax4.set_ylabel('Tổng chi tiêu trung bình')
        st.pyplot(fig4)
        
    # Biểu đồ 5 nằm toàn chiều rộng ở dưới
    st.markdown("---")
    st.subheader("Biểu đồ 5: Ma Trận Tương Quan Giữa Các Biến Số")
    purchase_cols = ['Income', 'Age', 'Total_Mnt', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    corr_matrix = df[purchase_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5, ax=ax5)
    st.pyplot(fig5)


# ------------------------------------------
# TAB 2: PHÂN CỤM K-MEANS
# ------------------------------------------
with tab2:
    st.header("🎯 Phân Cụm Phân Khúc Khách Hàng")
    st.write("Sử dụng thuật toán **K-Means** để tự động gom nhóm khách hàng dựa trên Thu nhập, Tuổi tác, Tần suất mua sắm và Số lượng giao dịch.")
    
    # Thực hiện chuẩn hóa và tính toán K-Means cục bộ dựa trên số cụm chọn ở Slider
    X_data = df[features]
    y_data = df['Total_Mnt']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Hiển thị biểu đồ phân cụm
    st.subheader(f"Biểu đồ Phân Cụm Khách Hàng (K = {num_clusters})")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='Income', y='Total_Mnt', hue='Cluster', palette='Set1', alpha=0.8, ax=ax_cluster)
    ax_cluster.set_xlabel('Thu Nhập (Income)')
    ax_cluster.set_ylabel('Tổng Chi Tiêu (Total Mnt)')
    st.pyplot(fig_cluster)
    
    # Hiển thị bảng dữ liệu thống kê theo từng cụm
    st.subheader("📊 Thống kê đặc trưng trung bình của từng Cụm")
    cluster_summary = df.groupby('Cluster')[['Income', 'Age', 'Total_Mnt', 'NumWebPurchases', 'NumStorePurchases']].mean()
    st.dataframe(cluster_summary.style.format("{:.2f}"))


# ------------------------------------------
# TAB 3: HỒI QUY TUYẾN TÍNH THEO TỪNG CỤM
# ------------------------------------------
with tab3:
    st.header("🤖 Dự Đoán Tổng Chi Tiêu Bằng Hồi Quy Tuyến Tính")
    st.markdown("Xây dựng mô hình **Hồi quy tuyến tính riêng cho từng cụm** được tạo ra ở Tab 2 giúp cá nhân hóa dự đoán chính xác hơn.")
    
    # Chia tập dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(df[features + ['Cluster']], y_data, test_size=0.2, random_state=42)
    final_predictions = np.zeros(len(X_test))
    
    # Khởi tạo các cột để hiển thị kết quả R2 của từng cụm
    st.subheader("📉 Kết quả huấn luyện cục bộ của các cụm:")
    cols_metrics = st.columns(num_clusters)
    
    # Duyệt và huấn luyện hồi quy trên từng cụm
    for cluster_id in range(num_clusters):
        train_mask = X_train['Cluster'] == cluster_id
        X_train_cluster = X_train[train_mask].drop(columns=['Cluster'])
        y_train_cluster = y_train[train_mask]
        
        test_mask = X_test['Cluster'] == cluster_id
        X_test_cluster = X_test[test_mask].drop(columns=['Cluster'])
        
        if len(X_train_cluster) > 0 and len(X_test_cluster) > 0:
            lr_model = LinearRegression()
            lr_model.fit(X_train_cluster, y_train_cluster)
            
            cluster_preds = lr_model.predict(X_test_cluster)
            final_predictions[test_mask] = cluster_preds
            
            # Tính toán chỉ số R2 của riêng cụm này
            cluster_r2 = r2_score(y_test[test_mask], cluster_preds)
            
            # Hiển thị số liệu dạng thẻ (Card) trực quan
            with cols_metrics[cluster_id]:
                st.metric(label=f"Cụm {cluster_id} (R² Score)", value=f"{cluster_r2:.4f}")
                st.caption(f"Số mẫu Train: {len(X_train_cluster)}")

    # Đánh giá tổng thể mô hình kết hợp
    st.markdown("---")
    st.subheader("🏆 Đánh giá tổng thể hệ thống (K-Means + Linear Regression)")
    
    overall_r2 = r2_score(y_test, final_predictions)
    overall_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
    
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.metric(label="Tổng điểm R² (Độ giải thích dữ liệu toàn cục)", value=f"{overall_r2:.4f}")
    with meta_col2:
        st.metric(label="Lỗi RMSE (Căn sai số trung bình bình phương)", value=f"{overall_rmse:.2f}")
        
    # Vẽ biểu đồ So sánh Thực tế vs Dự đoán
    st.subheader("Biểu đồ So Sánh: Chi Tiêu Thực Tế vs Dự Đoán")
    fig_eval, ax_eval = plt.subplots(figsize=(10, 5))
    ax_eval.scatter(y_test, final_predictions, alpha=0.6, color='teal', edgecolors='w')
    ax_eval.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    ax_eval.set_xlabel('Chi Tiêu Thực Tế')
    ax_eval.set_ylabel('Chi Tiêu Dự Đoán')
    st.pyplot(fig_eval)
