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
# File Check
# -------------------------------
required_files = ["best_model.pkl", "scaler.pkl", "feature_columns.json"]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Missing file: `{file}`. Upload all model files to your repo.")
        st.stop()

# -------------------------------
# Load Model, Scaler, and Features
# -------------------------------
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    if not hasattr(model, "predict"):
        raise TypeError("Invalid model object. Must be an sklearn model.")
    if not hasattr(scaler, "transform"):
        raise TypeError("Invalid scaler object. Must be a StandardScaler.")
except Exception as e:
    st.error(f"⚠️ Model loading failed. Details: {e}")
    st.stop()

# -------------------------------
# Input Fields
# -------------------------------
st.subheader("🩺 Enter Patient Health Details")

user_inputs = []
for col in feature_columns:
    val = st.number_input(f"{col}", min_value=0.0, format="%.2f")
    user_inputs.append(val)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("🔍 Predict"):
    try:
        input_array = np.array(user_inputs).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_input)[0][1]
        else:
            prob = float(prediction)

        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease\n**Confidence:** {prob:.2%}")
        else:
            st.success(f"✅ No Heart Disease Detected\n**Confidence:** {(1 - prob):.2%}")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")

st.write("---")
st.caption("Capstone Project 2 | Heart Disease Prediction using ML")
