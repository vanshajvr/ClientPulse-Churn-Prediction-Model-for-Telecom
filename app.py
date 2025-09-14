import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("churn_model.pkl")

st.title("📊 ClientPulse: Customer Churn Prediction")

st.write("Predict whether a customer is likely to churn.")

# Example input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 80, 30)
tenure = st.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

if st.button("Predict"):
    # Convert input to dataframe
    input_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "tenure": [tenure],
        "monthly_charges": [monthly_charges]
    })
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'Churn' if prediction==1 else 'Stay'} (Prob: {prob:.2f})")
