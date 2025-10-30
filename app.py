# app.py
# -----------------------------
# Streamlit Web App for Heart Disease Prediction
# -----------------------------

import streamlit as st
import pandas as pd
import pickle
import json

# -----------------------------
# Load Model, Scaler, and Columns
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's details below to predict heart disease risk.")

# -----------------------------
# Dynamic Input Fields
# -----------------------------
user_inputs = {}
for col in feature_columns:
    user_inputs[col] = st.number_input(f"Enter {col}:", value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_inputs])
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🩺 Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts **Heart Disease** (Risk: {prob*100:.2f}%)")
    else:
        st.success(f"✅ The model predicts **No Heart Disease** (Confidence: {(1-prob)*100:.2f}%)")

st.markdown("---")
st.caption("Capstone Project 2 | Developed with Streamlit")
