
import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load("best_xgboost_model.pkl")  # Load model here
scaler = joblib.load("scaler.pkl")

st.title("Sovereign Debt Crisis Predictor")
st.write("Enter the details below to check for a sovereign debt crisis.")

country = st.text_input("Enter Country Name")
year = st.number_input("Enter Year", min_value=2000, max_value=2024, step=1)

if st.button("🔍 Predict Crisis"):
    # Get prediction from the model (1 = Crisis, 0 = No Crisis)
    prediction = model.predict(input_data_scaled)[0]
    result = "🔴 Crisis" if prediction == 1 else "🟢 No Crisis"

    st.subheader(f"Prediction for {country} in {year}:")
    st.markdown(f"## {result}")
