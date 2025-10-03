import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# -----------------------
# Load Model & Scaler
# -----------------------
# Use compile=False to avoid H5 deserialization issues
model = tf.keras.models.load_model("churn_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# -----------------------
# Hardcoded columns
# -----------------------
expected_columns = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain", "Gender_Male"
]

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="📊 Customer Churn Prediction", 
    page_icon="📈", 
    layout="wide"
)

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Fill customer details on the main panel to predict churn.")

# -----------------------
# Title & Description
# -----------------------
st.title("📊 Customer Churn Prediction")
st.markdown("### Predict whether a customer is likely to churn based on their details.")
st.write("This tool uses a trained **ANN model** to predict churn probability.")

# -----------------------
# Input Form
# -----------------------
st.subheader("📝 Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    CreditScore = st.number_input("💳 Credit Score", min_value=300, max_value=850, value=650)
    Age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
    Tenure = st.number_input("📅 Tenure (Years)", min_value=0, max_value=10, value=3)
    Balance = st.number_input("💰 Balance", min_value=0.0, max_value=250000.0, value=50000.0)
    NumOfProducts = st.number_input("📦 Number of Products", min_value=1, max_value=10, value=2)

with col2:
    HasCrCard = st.selectbox("💳 Has Credit Card?", ["Yes", "No"])
    IsActiveMember = st.selectbox("🏃‍♂️ Is Active Member?", ["Yes", "No"])
    EstimatedSalary = st.number_input("💵 Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)
    Geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("👤 Gender", ["Male", "Female"])

# -----------------------
# Prepare Input Data
# -----------------------
input_dict = {
    "CreditScore": CreditScore,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": 1 if HasCrCard=="Yes" else 0,
    "IsActiveMember": 1 if IsActiveMember=="Yes" else 0,
    "EstimatedSalary": EstimatedSalary,
    "Geography_Germany": 1 if Geography=="Germany" else 0,
    "Geography_Spain": 1 if Geography=="Spain" else 0,
    "Gender_Male": 1 if Gender=="Male" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# -----------------------
# Prediction
# -----------------------
if st.button("🚀 Predict Churn"):
    try:
        input_scaled = scaler.transform(input_df).astype(np.float32)
        prediction = model.predict(input_scaled)
        prob = float(prediction[0][0])

        # Result text
        if prob > 0.5:
            st.error(f"⚠️ High Risk: Customer is likely to **CHURN** with probability {prob:.2f}")
        else:
            st.success(f"✅ Safe: Customer is likely to **STAY** with probability {1-prob:.2f}")

        # Plotly gauge chart for probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={'suffix': "%"},
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prob>0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob*100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
