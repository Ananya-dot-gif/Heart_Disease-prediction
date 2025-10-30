import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load trained model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = json.load(open("feature_columns.json", "r"))

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the risk of heart disease:")

# Input fields dynamically
user_input = []
for col in feature_columns:
    val = st.number_input(f"{col}", min_value=0.0, max_value=500.0, value=0.0)
    user_input.append(val)

if st.button("Predict"):
    data = np.array(user_input).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    result = "🩺 Heart Disease Detected" if prediction[0] == 1 else "💚 No Heart Disease Detected"
    st.success(result)
