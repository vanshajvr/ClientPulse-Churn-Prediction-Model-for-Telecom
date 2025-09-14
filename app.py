import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model + scaler
model = load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("📊 ClientPulse: Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn or stay.")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 80, 30)
tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

gender_map = {"Male": 0, "Female": 1}
gender_num = gender_map[gender]

if st.button("Predict"):
    # Create dataframe
    input_df = pd.DataFrame({
        "gender": [gender_num],
        "age": [age],
        "tenure": [tenure],
        "monthly_charges": [monthly_charges]
    })

    # Scale features
    input_array = scaler.transform(input_df).astype(np.float32)

    # Predict
    prediction = model.predict(input_array)

    if prediction.shape[-1] == 1:  # Binary
        prob = float(prediction[0][0])
        label = "Churn" if prob > 0.5 else "Stay"
        st.success(f"✅ Prediction: **{label}** (Churn Probability: {prob:.2f})")
    else:
        predicted_class = prediction.argmax(axis=1)[0]
        st.success(f"✅ Predicted Class: {predicted_class}")

