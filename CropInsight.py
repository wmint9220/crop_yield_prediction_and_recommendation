import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =============================
# PAGE CONFIG & STYLING
# =============================
st.set_page_config(
    page_title="🌱 Crop Insight | Smart Agriculture",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
/* Main content background - Earthy Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8f9f4 0%, #eef7e8 40%, #e0f2e5 100%);
}

/* Custom Card for Login and Results */
.prediction-card {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
    border-left: 10px solid #2e7d32;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    margin-top: 25px;
}
.prediction-card h2 {
    color: #1b5e20 !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] { 
    background-color: #f0f7eb !important; 
}

/* Button Styling */
.stButton > button {
    border-radius: 12px;
    font-weight: bold;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE & LOGOUT
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

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
    except:
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("Crop_recommendation.csv")
    except:
        return None

# =============================
# UI SECTIONS
# =============================
def show_login():
    st.title("🔐 Crop Insight Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.page = "trend"
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
        
        

def show_trend():
    st.title("📊 Agricultural Data Trends")
    
    st.info("""
    ### 📖 Data Interpretation Guide
    The following visualizations represent a multi-variate analysis of soil and climate conditions. 
    Using these tools, you can identify which environmental factors (like temperature or rainfall) 
    are the most critical drivers for successful crop cultivation in your specific region.
    """)
    
    df = load_data()
    if df is not None:
        st.subheader("🌡️ Optimal Temperature Ranges per Crop")
        st.write("This analysis identifies the thermal limits for various crop species, helping you plan for climate variability.")
        temp_by_crop = df.groupby('label')['temperature'].mean().sort_values(ascending=False)
        st.bar_chart(temp_by_crop)
    else:
        st.warning("⚠️ Data source file ('Crop_recommendation.csv') is currently missing.")


def show_prediction():
    st.title("🌱 Crop Recommendation System")
    st.success("Input your soil & climate parameters to get AI-based crop suggestions.")

    model, le = load_model()
    if model is None:
        st.error("🚨 Model files missing!")
        return

    st.subheader("Enter Farm Conditions")
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

    if st.button("✨ Predict Crop", use_container_width=True):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        try:
            prediction = model.predict(input_data)
            crop = le.inverse_transform(prediction)[0]

            crop_emojis = {
                "rice":"🌾","wheat":"🌾","maize":"🌽","chickpea":"🫘",
                "kidneybeans":"🫘","pigeonpeas":"🌱","mothbeans":"🌿",
                "mungbean":"🌱","blackgram":"🫘","lentil":"🌿",
                "pomegranate":"🍇","banana":"🍌","mango":"🥭",
                "grapes":"🍇","watermelon":"🍉","muskmelon":"🍈",
                "apple":"🍎","orange":"🍊","papaya":"🍈","coconut":"🥥",
                "cotton":"☁️","jute":"🌿","coffee":"☕"
            }
            emoji = crop_emojis.get(crop.lower(), "🌱")

            st.markdown(f"""
            <div class="prediction-card">
                <h3>Recommended Crop: <strong>{crop.upper()} {emoji}</strong></h3>
                <p>Based on your soil's NPK and climate, <b>{crop}</b> is the most suitable crop for a high-yield harvest.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =============================
# MAIN NAVIGATION
# =============================
if st.session_state.logged_in:
    st.sidebar.title("🧭 Navigation")
    st.sidebar.write(f"Logged in as: **Admin**")
    
    choice = st.sidebar.radio("Go to:", ["📊 Trend Visualization", "🌱 Crop Prediction"])
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()

    if "Trend" in choice:
        show_trend()
    else:
        show_prediction()
else:
    show_login()

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("© 2026 Crop Insight AI. Integrated Machine Learning for Sustainable Agriculture.")
