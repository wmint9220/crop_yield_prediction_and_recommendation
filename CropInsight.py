import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="🌱 Smart Crop Recommender",
    page_icon="🌾",
    layout="centered"
)

# Custom styling (optional extra polish)
st.markdown("""
    <style>
    .main {
        background-color: #f0f8eb;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌱 Smart Crop Recommendation System")
st.markdown("Discover the **best crop** to grow based on your soil and climate conditions! 🌾💧")
st.divider()

# Load model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("crop_recommendation_rf.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_model()

# Input section with columns for better layout
st.subheader("🔍 Enter Your Farm Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90, help="Typical range: 20–100")
    P = st.number_input("Phosphorus (P)", 0, 200, 40, help="Typical range: 10–50")
    K = st.number_input("Potassium (K)", 0, 200, 45, help="Typical range: 20–60")
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, help="Ideal crop range: 10–35°C")

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, help="Typical: 40–80%")
    ph = st.number_input("Soil pH", 3.5, 10.0, 6.5, help="Most crops prefer 5.5–7.5")
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, help="Monthly average")

st.divider()

# Prediction button
if st.button("🌿 Get Crop Recommendation", use_container_width=True):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Validate pH range (common issue)
    if ph < 3.5 or ph > 10:
        st.warning("⚠️ Soil pH outside typical agricultural range (3.5–10.0). Results may be unreliable.")
    
    try:
        prediction = model.predict(input_data)
        crop = le.inverse_transform(prediction)[0]

        # Display result beautifully
        st.success(f"✅ **Recommended Crop:** {crop}")
        
        # Optional: Add fun emoji or image based on crop
        crop_images = {
            "rice": "🌾", "wheat": "🫘", "maize": "🌽", "chickpea": "🥜",
            "kidneybeans": "🫘", "pigeonpeas": "🌱", "mothbeans": "🌿",
            "mungbean": "🌱", "blackgram": "🫘", "lentil": "🌿",
            "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
            "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
            "apple": "🍎", "orange": "🍊", "papaya": "🍈",
            "coconut": "🥥", "cotton": "☁️", "jute": "🌿",
            "coffee": "☕"
        }
        emoji = crop_images.get(crop.lower(), "🌱")
        st.markdown(f"### {emoji} Happy Farming! 🌻")

    except Exception as e:
        st.error("❌ Something went wrong. Please check your inputs and try again.")
        st.write(e)

# Footer
st.markdown("---")
st.caption("💡 Powered by Machine Learning • Random Forest Model • Data from agricultural research")
