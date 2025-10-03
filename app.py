import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load model and scaler
model = tf.keras.models.load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

# 🎨 Page Config
st.set_page_config(page_title="📊 Telecom Churn Prediction", page_icon="📱", layout="wide")

# 💡 Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Fill customer details on the main panel to predict churn.")

st.title("📱 Telecom Customer Churn Prediction")
st.markdown("### Predict whether a customer is likely to churn based on their details.")
st.write("This tool uses a trained **ANN (Artificial Neural Network)** model to predict churn probability.")

# 📋 Input Form
st.subheader("📝 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("📅 Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("💵 Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("💰 Total Charges", min_value=0.0, max_value=10000.0, value=600.0)

with col2:
    contract = st.selectbox("📑 Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

# 🔄 Convert Input into DataFrame
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "InternetService": internet_service
}

input_df = pd.DataFrame([input_dict])

# Align columns with training features
input_df = pd.get_dummies(input_df).reindex(columns=expected_columns, fill_value=0)

# 🔮 Prediction Button
if st.button("🚀 Predict Churn"):
    try:
        input_scaled = scaler.transform(input_df).astype(np.float32)
        prediction = model.predict(input_scaled)
        prob = prediction[0][0]

        if prob > 0.5:
            st.error(f"⚠️ High Risk: Customer is likely to **CHURN** with probability {prob:.2f}")
        else:
            st.success(f"✅ Safe: Customer is likely to **STAY** with probability {1-prob:.2f}")

        # Show probability meter
        st.progress(int(prob * 100))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
