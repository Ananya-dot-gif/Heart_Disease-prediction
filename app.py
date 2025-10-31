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

missing_files = [file for file in required_files if not os.path.exists(file)]
if missing_files:
    st.error(f"❌ Missing required files: {', '.join(missing_files)}")
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

# Check if model is valid
if not hasattr(model, "predict"):
    st.error("❌ The loaded model file is invalid. Please ensure 'best_model.pkl' is a trained ML model (not a string).")
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

        # Validate scaler
        if hasattr(scaler, "transform"):
            input_scaled = scaler.transform(input_values)
        else:
            st.warning("⚠️ Scaler file seems invalid. Using unscaled input temporarily.")
            input_scaled = input_values

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Confidence / probability (if available)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            probability = float(prediction)

        st.write("---")
        if prediction == 1:
            st.error(f"🚨 High Risk of Heart Disease\nConfidence: **{probability:.2%}**")
        else:
            st.success(f"💚 No Heart Disease Detected\nConfidence: **{1 - probability:.2%}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.write("---")
st.caption("Capstone Project 2 | Heart Disease Prediction using ML")
