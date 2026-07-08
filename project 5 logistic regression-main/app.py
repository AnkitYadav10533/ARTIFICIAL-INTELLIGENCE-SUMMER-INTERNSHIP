# -*- coding: utf-8 -*-
"""Logistic Regression Deployment on StreamLit
This app predicts whether a person will buy life insurance based on their age.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Purchase Predictor",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for modern visual design
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #2E8B57;
    }
    .metric-container {
        background-color: #f1f3f6;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e1e4e8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Title and Description
# -----------------------------------
st.markdown("# 🛡️ Life Insurance Purchase Predictor")
st.write("Train a Logistic Regression model on insurance customer data and predict purchase behavior.")
st.markdown("---")

# -----------------------------------
# Sidebar - Configuration & Parameters
# -----------------------------------
st.sidebar.header("⚙️ Model Configuration")

# User adjustable train/test size and random state
train_size = st.sidebar.slider(
    "Training Data Ratio",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.05,
    help="Fraction of the dataset used to train the model."
)

random_state = st.sidebar.number_input(
    "Random State (Seed)",
    min_value=1,
    max_value=1000,
    value=42,
    step=1,
    help="Seed for random split to ensure reproducible training."
)

# -----------------------------------
# Load & Process Dataset
# -----------------------------------
try:
    df = pd.read_csv("insurance_data.csv")
except Exception as e:
    st.error(f"Error loading insurance_data.csv: {e}")
    st.stop()

# Splitting features and target
X = df[['age']]
y = df['bought_insurance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=train_size, 
    random_state=random_state
)

# -----------------------------------
# Model Training
# -----------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# Calculate model scores
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# -----------------------------------
# Prediction Interface
# -----------------------------------
st.subheader("🔮 Make a Prediction")
st.write("Adjust the slider to predict purchase behavior for a specific age.")

# User input for age
input_age = st.slider(
    "Person's Age",
    min_value=10,
    max_value=85,
    value=35,
    step=1
)

# Model predictions
# Reshape input for prediction
test_val = np.array([[input_age]])
pred_prob = model.predict_proba(test_val)[0][1]
prediction = model.predict(test_val)[0]

st.markdown("### Prediction Result")

if prediction == 1:
    st.success("🛡️ **Predicted Outcome:** Will Buy Insurance")
else:
    st.error("❌ **Predicted Outcome:** Will NOT Buy Insurance")

# Progress bar and probability display
st.write(f"**Probability of buying:** `{pred_prob * 100:.2f}%`")
st.progress(float(pred_prob))

st.markdown("---")

# -----------------------------------
# Model Statistics & Details
# -----------------------------------
st.subheader("📊 Dataset & Model Statistics")

col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.markdown(
        f"""
        <div class="metric-container">
            <span style="font-size: 14px; color: #555;">Total Dataset Samples</span><br>
            <span style="font-size: 28px; font-weight: bold;">{len(df)} rows</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_stats2:
    st.markdown(
        f"""
        <div class="metric-container">
            <span style="font-size: 14px; color: #555;">Training Accuracy</span><br>
            <span style="font-size: 28px; font-weight: bold; color: #2E8B57;">{train_accuracy * 100:.1f}%</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_stats3:
    st.markdown(
        f"""
        <div class="metric-container">
            <span style="font-size: 14px; color: #555;">Testing Accuracy</span><br>
            <span style="font-size: 28px; font-weight: bold; color: #1f77b4;">{test_accuracy * 100:.1f}%</span>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Expander to see dataset and model details
with st.expander("🔍 View Raw Dataset & Model Parameters"):
    data_col, model_col = st.columns(2)
    with data_col:
        st.write("**Dataset Preview:**")
        st.dataframe(df, height=200)
    with model_col:
        # Coefficients
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        st.write("**Trained Parameters:**")
        st.markdown(f"- **Coefficient (m):** `{coef:.6f}`")
        st.markdown(f"- **Intercept (c):** `{intercept:.6f}`")
        st.write("**Notebook Approximation comparison:**")
        st.write("The Colab notebook lists approximations of:")
        st.code(f"z = 0.042 * age - 1.53")
        st.write(f"Based on your sidebar split, the current parameters are:")
        st.code(f"z = {coef:.3f} * age + ({intercept:.3f})")