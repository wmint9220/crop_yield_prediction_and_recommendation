import streamlit as st
import joblib
import numpy as np

st.title("🌱 Crop Recommendation System")

model = joblib.load("crop_recommendation_rf.pkl")
le = joblib.load("label_encoder.pkl")

# User inputs
N = st.number_input("Nitrogen (N)", 0, 200)
P = st.number_input("Phosphorus (P)", 0, 200)
K = st.number_input("Potassium (K)", 0, 200)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("Soil pH", 0.0, 14.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop = le.inverse_transform(prediction)
    st.success(f"🌾 Recommended Crop: {crop[0]}")
