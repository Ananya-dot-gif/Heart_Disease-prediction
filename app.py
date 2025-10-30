# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's details to predict the likelihood of heart disease.")

# Input fields (replace these with your dataset feature names)
features = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_bs",
    "rest_ecg", "max_hr", "exercise_angina", "oldpeak", "st_slope"
]

user_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_data.append(value)

if st.button("Predict"):
    input_df = pd.DataFrame([user_data], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🩺 Prediction Result:")
    if prediction[0] == 1:
        st.error(f"The model predicts **Heart Disease** with probability {prob*100:.2f}%")
    else:
        st.success(f"The model predicts **No Heart Disease** with probability {(1-prob)*100:.2f}%")

st.markdown("---")
st.caption("Developed with Streamlit | Capstone Project 2")
