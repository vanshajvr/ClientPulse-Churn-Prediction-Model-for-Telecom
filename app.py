import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ----------------------
# Load model + scaler
# ----------------------
scaler = joblib.load("scaler.pkl")   # saved during training
ann_model = tf.keras.models.load_model("churn_model.h5")

st.title("📊 ClientPulse: Customer Churn Prediction")

# ----------------------
# User inputs
# ----------------------
st.sidebar.header("Customer Information")

tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0, max_value=200, value=70)
total_charges = st.sidebar.number_input("Total Charges", min_value=0, max_value=10000, value=1000)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# ----------------------
# Build DataFrame (raw inputs)
# ----------------------
input_dict = {
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    # ➡️ add ALL other features you used in training
}
input_df = pd.DataFrame(input_dict)

# ----------------------
# Encode categorical features like in training
# ----------------------
input_encoded = pd.get_dummies(input_df)

# Align with training features (order + fill missing with 0)
input_encoded = input_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

# ----------------------
# Scale + Predict
# ----------------------
input_scaled = scaler.transform(input_encoded).astype(np.float32)
prediction = ann_model.predict(input_scaled)[0][0]  # sigmoid output = probability

# ----------------------
# Display Result
# ----------------------
st.subheader("Prediction Result:")
if prediction > 0.5:
    st.error(f"🚨 Customer is likely to churn! (Probability: {prediction:.2f})")
else:
    st.success(f"✅ Customer is likely to stay. (Probability: {prediction:.2f})")
