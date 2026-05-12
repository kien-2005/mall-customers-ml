# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data.csv")

# Encode Gender
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

# =========================
# FEATURE & TARGET
# =========================

X = df.drop("Spending Score (1-100)", axis=1)
y = df["Spending Score (1-100)"]

# =========================
# SCALE DATA
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# KMEANS
# =========================

k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42
)

kmeans.fit(X_train)

train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)

# =========================
# TRAIN REGRESSION MODELS
# =========================

models = {}

for cluster in range(k):

    X_cluster = X_train[train_clusters == cluster]
    y_cluster = y_train[train_clusters == cluster]

    model = LinearRegression()
    model.fit(X_cluster, y_cluster)

    models[cluster] = model

# =========================
# EVALUATION
# =========================

y_pred = np.zeros(len(X_test))

for cluster in range(k):

    X_cluster_test = X_test[test_clusters == cluster]

    if len(X_cluster_test) > 0:

        y_pred[test_clusters == cluster] = models[
            cluster
        ].predict(X_cluster_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Mall Customers Prediction",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Nhập thông tin khách hàng")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider(
    "Age",
    18,
    70,
    30
)

income = st.sidebar.slider(
    "Annual Income (k$)",
    10,
    150,
    60
)

# =========================
# MAIN TITLE
# =========================

st.markdown(
    """
    <h1 style='text-align:center;'>
    🛍️ Dự đoán Spending Score
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align:center;color:gray;'>
    KMeans + Linear Regression
    </h3>
    """,
    unsafe_allow_html=True
)

st.write("")

# =========================
# PREDICT BUTTON
# =========================

if st.button("Dự đoán Spending Score 🎯"):

    # Encode gender
    gender_value = 1 if gender == "Male" else 0

    # Create input
    input_data = pd.DataFrame([[
        1,
        gender_value,
        age,
        income
    ]], columns=X.columns)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    # Predict spending score
    prediction = models[cluster].predict(input_scaled)[0]

    prediction = round(prediction, 2)

    st.success(
        f"Spending Score dự đoán: {prediction}"
    )

    st.info(
        f"Khách hàng thuộc cụm: {cluster}"
    )

# =========================
# MODEL METRICS
# =========================

st.write("")
st.write("")
st.subheader("Đánh giá mô hình")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MSE",
        round(mse, 2)
    )

with col2:
    st.metric(
        "R² Score",
        round(r2, 2)
    )

# =========================
# SHOW DATA
# =========================

st.write("")
st.subheader("Dữ liệu mẫu")

st.dataframe(df.head())
