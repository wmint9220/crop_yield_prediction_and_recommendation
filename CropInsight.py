import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =============================
# PAGE CONFIG & STYLING (ENHANCED)
# =============================
st.set_page_config(
    page_title="🌱 Crop Insight",
    page_icon="🌾",
    layout="wide"
)

# ✅ Streamlit-Safe Enhanced Styling (Works in all modern Streamlit versions)
st.markdown("""
<style>
/* Main content background — soft earthy gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8f9f4 0%, #eef7e8 40%, #e0f2e5 100%);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f0f7eb !important;
    border-right: 1px solid #d4edda;
}

/* Titles & headers — rich green */
h1, h2, h3 {
    color: #1b5e20 !important;
    font-weight: 700 !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(to right, #2e7d32, #1b5e20);
    color: white;
    border-radius: 14px;
    height: 54px;
    font-weight: bold;
    font-size: 18px;
    width: 100%;
    border: none;
    box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(to right, #1b5e20, #0d4c1a);
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(27, 94, 32, 0.3);
}

/* Input fields */
input, textarea, select {
    border-radius: 8px !important;
    border: 1px solid #a5d6a7 !important;
    background-color: white !important;
}

/* Dataframe container */
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Divider & HR */
hr {
    border-color: #a5d6a7 !important;
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

def logout():
    st.session_state.logged_in = False
    st.session_state.page = "login"
    st.rerun()
    
# =============================
# LOAD MODEL & DATA
# =============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("crop_recommendation_rf.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, le
    except Exception as e:
        st.error(f"ModelError: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        return df
    except FileNotFoundError:
        return None
        
# =============================
# LOGIN LOGIC
# =============================
def show_login():
    st.title("🔐 Crop Insight Login")
    st.markdown("""
    <div style='background: linear-gradient(to right, #43a047, #2e7d32); padding: 12px 20px; border-radius: 12px; margin-bottom: 20px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <h4 style='margin:0; font-weight:600;'>🌱 Empowering Farmers with AI-Data Driven Crop Insights</h4>
    </div>
    """, unsafe_allow_html=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.page = "trend"
            st.experimental_rerun()
        else:
            st.error("❌ Invalid credentials")

# =============================
# TREND VISUALIZATION SECTION
# =============================
def show_trend():
    st.title("📊 Trend Visualization")
    
    # ✅ Header banner
    st.markdown("""
    <div style='background: linear-gradient(to right, #43a047, #2e7d32); padding: 12px 20px; border-radius: 12px; margin-bottom: 20px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <h4 style='margin:0; font-weight:600;'>🌾 Smart Crop Insights for Sustainable Farming</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Explore patterns in crop cultivation based on soil and weather data.")
    
    df = load_data()
    if df is not None:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        
        st.subheader("🌡️ Average Temperature by Crop")
        temp_by_crop = df.groupby('label')['temperature'].mean().sort_values(ascending=False)
        st.bar_chart(temp_by_crop)
        
        st.subheader("💧 Rainfall Distribution (Sample)")
        st.line_chart(df[['rainfall']].sample(min(100, len(df)), random_state=42))
    else:
        st.warning("📁 Dataset `Crop_recommendation.csv` not found. Place it in the same directory to enable visualizations!")

# =============================
# PREDICTION SECTION
# =============================
def show_prediction():
    st.title("🌱 Crop Recommendation System")
    
    # ✅ Header banner
    st.markdown("""
    <div style='background: linear-gradient(to right, #43a047, #2e7d32); padding: 12px 20px; border-radius: 12px; margin-bottom: 20px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
        <h4 style='margin:0; font-weight:600;'>🌾 Smart Crop Recommendation for Sustainable Farming</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Discover the **best crop** to grow based on your soil and climate conditions! 🌾💧")
    st.divider()
    
    model, le = load_model()
    if model is None or le is None:
        st.error("❌ Model files missing. Please ensure `crop_recommendation_rf.pkl` and `label_encoder.pkl` exist.")
        return
    
    st.subheader("🌿 Enter Your Farm Conditions")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.slider("Nitrogen (N)", 0, 150, 50)
            P = st.slider("Phosphorus (P)", 0, 150, 50)
            K = st.slider("Potassium (K)", 0, 150, 50)
            ph = st.number_input("Soil pH (0-14)", 0.0, 14.0, 6.5)
        with col2:
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
            hum = st.slider("Humidity (%)", 0, 100, 50)
            rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
        
        submit = st.form_submit_button("✨ Predict Best Crop")

    if submit:
        # Match order: N, P, K, temperature, humidity, ph, rainfall
        input_data = np.array([[N, P, K, temp, hum, ph, rain]])
        prediction = model.predict(input_data)
        crop = le.inverse_transform(prediction)[0]
        
        crop_emojis = {"rice":"🌾","wheat":"🌾","maize":"🌽","coffee":"☕","cotton":"☁️"} # truncated for brevity
        emoji = crop_emojis.get(crop.lower(), "🌱")

        # --- PREDICTION CARD ---
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Recommended Crop: <strong>{crop.upper()} {emoji}</strong></h3>
                <p>Based on your soil's NPK levels and local climate, <b>{crop}</b> is the most viable option for a high-yield harvest.</p>
            </div>
            """, unsafe_allow_html=True)
        st.balloons()
        
    # col1, col2 = st.columns(2)
    # with col1:
    #     N = st.number_input("Nitrogen (N)", 0, 200, 90)
    #     P = st.number_input("Phosphorus (P)", 0, 200, 40)
    #     K = st.number_input("Potassium (K)", 0, 200, 45)
    #     temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    # with col2:
    #     humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    #     ph = st.number_input("Soil pH", 3.5, 10.0, 6.5)
    #     rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
    
    # if st.button("✨ Predict Best Crop", use_container_width=True):
    #     input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    #     try:
    #         prediction = model.predict(input_data)
    #         crop = le.inverse_transform(prediction)[0]
    #         crop_emojis = {
    #             "rice":"🌾","wheat":"🌾","maize":"🌽","chickpea":"🫘",
    #             "kidneybeans":"🫘","pigeonpeas":"🌱","mothbeans":"🌿",
    #             "mungbean":"🌱","blackgram":"🫘","lentil":"🌿",
    #             "pomegranate":"🍇","banana":"🍌","mango":"🥭",
    #             "grapes":"🍇","watermelon":"🍉","muskmelon":"🍈",
    #             "apple":"🍎","orange":"🍊","papaya":"🍈","coconut":"🥥",
    #             "cotton":"☁️","jute":"🌿","coffee":"☕"
    #         }
    #         emoji = crop_emojis.get(crop.lower(), "🌱")
    #         st.success(f"✅ **Recommended Crop:** {crop} {emoji}")
    #         st.balloons()  # 🎉 optional fun!
    #         st.markdown(f"### {emoji} Happy Farming! 🌻")
    #     except Exception as e:
    #         st.error(f"Prediction error: {e}")


# =============================
# SIDEBAR NAVIGATION 
# =============================
if st.session_state.logged_in:
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to:", ["📊 Trend Visualization", "🌱 Crop Prediction"])
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        logout()

    if choice == "📊 Trend Visualization":
        show_trend()
    else:
        show_prediction()
else:
    show_login()
        
# =============================
# PAGE DISPLAY LOGIC
# =============================
if not st.session_state.logged_in:
    show_login()
elif st.session_state.page == "trend":
    show_trend()
elif st.session_state.page == "prediction":
    show_prediction()


