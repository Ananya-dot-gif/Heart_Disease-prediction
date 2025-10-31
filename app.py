# app.py
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load saved components
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.markdown("Predict the risk of **heart disease** using health parameters.")

st.write("---")

# Create input fields dynamically
input_data = {}
for col in feature_columns:
    input_data[col] = st.number_input(f"Enter value for {col}", min_value=0.0, format="%.2f")

# Predict button
if st.button("🔍 Predict"):
    try:
        # Prepare input
        input_values = np.array(list(input_data.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_values)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ No Heart Disease Detected (Confidence: {1 - probability:.2%})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.write("---")
st.caption("Developed as Capstone Project 2 | Heart Disease Prediction using ML")
