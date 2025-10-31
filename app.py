# app.py
import streamlit as st
import pickle
import json
import numpy as np
import os

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.markdown("Predict the risk of **heart disease** using health parameters.")
st.write("---")

# -------------------------------
# Check required files
# -------------------------------
required_files = ["best_model.pkl", "scaler.pkl", "feature_columns.json"]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Missing required file: `{file}`. Please upload it to the repository.")
        st.stop()

# -------------------------------
# Safe loading of model/scaler/features
# -------------------------------
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler. Please recheck files. Details: {e}")
    st.stop()

# -------------------------------
# Input UI for each feature
# -------------------------------
st.subheader("🩺 Enter Patient Details")
inputs = {}

for col in feature_columns:
    inputs[col] = st.number_input(f"{col}", min_value=0.0, format="%.2f")

# -------------------------------
# Predict button logic
# -------------------------------
if st.button("🔍 Predict"):
    try:
        # Convert inputs to numpy array
        input_values = np.array(list(inputs.values())).reshape(1, -1)

        # Check if scaler is valid
        if hasattr(scaler, "transform"):
            input_scaled = scaler.transform(input_values)
        else:
            # Fallback: if scaler is just a numpy array (incorrectly saved)
            st.warning("⚠️ Scaler file seems invalid. Using unscaled input temporarily.")
            input_scaled = input_values

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Check if model supports probability
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = float(prediction)  # fallback

        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ No Heart Disease Detected (Confidence: {1 - probability:.2%})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.write("---")
st.caption("Capstone Project 2 | Heart Disease Prediction using ML")
