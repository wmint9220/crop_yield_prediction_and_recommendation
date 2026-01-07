import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =============================
# PAGE CONFIG & STYLING
# =============================
st.set_page_config(
    page_title="🌱 Smart Crop Recommender",
    page_icon="🌾",
    layout="wide"
)

# Custom background & button style
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
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE INITIALIZATION
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"  # can be: login, trend, prediction

# =============================
# LOGIN LOGIC
# =============================
def show_login():
    st.title("🔐 Smart Crop Recommender Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.page = "trend"  # auto redirect to Trend Visualization first
            st.experimental_rerun()
        else:
            st.error("❌ Invalid credentials")

# =============================
# LOAD MODEL & DATA
# =============================
@st.cache_resource
def load_model():
    model = joblib.load("crop_recommendation_rf.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crop_recommendation.csv")  # updated dataset name
        return df
    except FileNotFoundError:
        return None

# =============================
# TREND VISUALIZATION SECTION
# =============================
def show_trend():
    st.title("📊 Crop & Climate Trends")
    st.markdown("Explore patterns in crop cultivation based on soil and weather data.")
    
    df = load_data()
    if df is not None:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        
        st.subheader("🌡️ Average Temperature by Crop")
        temp_by_crop = df.groupby('label')['temperature'].mean().sort_values(ascending=False)
        st.bar_chart(temp_by_crop)
        
        st.subheader("💧 Rainfall Distribution")
        st.line_chart(df[['rainfall']].sample(100))
    else:
        st.info("📁 Dataset (`Crop_recommendation.csv`) not found. Add it to enable visualizations!")

# =============================
# PREDICTION SECTION
# =============================
def show_prediction():
    st.title("🌱 Smart Crop Recommendation System")
    st.markdown("Discover the **best crop** to grow based on your soil and climate conditions! 🌾💧")
    st.divider()
    
    model, le = load_model()
    
    st.subheader("🌿 Enter Your Farm Conditions")
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 90)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
        K = st.number_input("Potassium (K)", 0, 200, 45)
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    with col2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        ph = st.number_input("Soil pH", 3.5, 10.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    
    if st.button("✨ Predict Best Crop", use_container_width=True):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        crop = le.inverse_transform(prediction)[0]
        crop_emojis = {"rice":"🌾","wheat":"🌾","maize":"🌽","chickpea":"🫘",
                       "kidneybeans":"🫘","pigeonpeas":"🌱","mothbeans":"🌿",
                       "mungbean":"🌱","blackgram":"🫘","lentil":"🌿",
                       "pomegranate":"🍇","banana":"🍌","mango":"🥭",
                       "grapes":"🍇","watermelon":"🍉","muskmelon":"🍈",
                       "apple":"🍎","orange":"🍊","papaya":"🍈","coconut":"🥥",
                       "cotton":"☁️","jute":"🌿","coffee":"☕"}
        emoji = crop_emojis.get(crop.lower(), "🌱")
        st.success(f"✅ **Recommended Crop:** {crop}")
        st.markdown(f"### {emoji} Happy Farming! 🌻")

# =============================
# SIDEBAR NAVIGATION (AFTER LOGIN)
# =============================
if st.session_state.logged_in:
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to:", ["Trend Visualization", "Crop Prediction"])
    st.session_state.page = "trend" if choice == "Trend Visualization" else "prediction"

# =============================
# PAGE DISPLAY LOGIC
# =============================
if not st.session_state.logged_in:
    show_login()
elif st.session_state.page == "trend":
    show_trend()
elif st.session_state.page == "prediction":
    show_prediction()
