import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ----------------------
# Load model + scaler
# ----------------------
scaler = joblib.load("scaler.pkl")   # you must have saved this during training
ann_model = tf.keras.models.load_model("churn_model.h5")

st.title("📊 ClientPulse: Customer Churn Prediction")

# ----------------------
# User inputs
# ----------------------
st.sidebar.header("Customer Information")

# Example inputs (replace with your dataset’s features)
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0, max_value=200, value=70)
total_charges = st.sidebar.number_input("Total Charges", min_value=0, max_value=10000, value=1000)

# Add all the categorical fields you used in training (e.g. Contract, InternetService, etc.)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# ----------------------
# Build DataFrame
# ----------------------
# Match the exact columns you trained the ANN on
input_dict = {
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    # add other features here in the same order as training
}

input_df = pd.DataFrame(input_dict)

# ----------------------
# Align columns to training scaler
# ----------------------
try:
    input_df = input_df[scaler.feature_names_in_]   # enforce same column order
except:
    st.error("⚠️ Input features do not match training features. Please check column names.")
    st.stop()

# ----------------------
# Scale + Predict
# ----------------------
input_scaled = scaler.transform(input_df).astype(np.float32)
prediction = ann_model.predict(input_scaled)[0][0]  # assuming binary churn (sigmoid output)

# ----------------------
# Display Result
# ----------------------
st.subheader("Prediction Result:")
if prediction > 0.5:
    st.error(f"🚨 Customer is likely to churn! (Probability: {prediction:.2f})")
else:
    st.success(f"✅ Customer is likely to stay. (Probability: {prediction:.2f})")
