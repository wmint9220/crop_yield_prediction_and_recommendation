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
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8f9f4 0%, #eef7e8 40%, #e0f2e5 100%);
}
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
[data-testid="stSidebar"] { 
    background-color: #f0f7eb !important; 
}
.stButton > button {
    border-radius: 12px;
    font-weight: bold;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE INIT
# =============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "stage1_crop" not in st.session_state:
    st.session_state.stage1_crop = None
if "stage1_input" not in st.session_state:
    st.session_state.stage1_input = None

def logout():
    st.session_state.logged_in = False
    st.session_state.page = "login"
    st.session_state.stage1_crop = None
    st.session_state.stage1_input = None
    st.rerun()

# =============================
# LOAD MODELS & DATA
# =============================
@st.cache_resource
def load_stage1():
    try:
        model = joblib.load("crop_recommendation_rf.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, le
    except Exception as e:
        st.error(f"Stage 1 model load failed: {e}")
        return None, None

@st.cache_resource
def load_stage2():
    try:
        return joblib.load("xgboost_yield_model.pkl")  
    except Exception as e:
        st.error(f"Stage 2 model load failed: {e}")
        return None

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
    """)
    df = load_data()
    if df is not None:
        st.subheader("🌡️ Optimal Temperature Ranges per Crop")
        temp_by_crop = df.groupby('label')['temperature'].mean().sort_values(ascending=False)
        st.bar_chart(temp_by_crop)
    else:
        st.warning("⚠️ Data source file ('Crop_recommendation.csv') is missing.")

def show_prediction():
    st.title("🌱 Intelligent Crop Recommendation")
    
    st.success("""
    ### 🛠️ Recommendation Methodology
    Our AI model evaluates **seven distinct data points** to minimize risk and maximize harvest yield. 
    Please input your soil test results accurately. Nitrogen (N), Phosphorus (P), and Potassium (K) 
    are measured in kg/ha, while climate factors are based on seasonal averages.
    """)

    # Load models
    stage1_model, le = load_stage1()
    stage2_model = load_stage2()
    
    if stage1_model is None or le is None:
        st.error("🚨 Stage 1 model files missing.")
        return
    if stage2_model is None:
        st.error("🚨 Stage 2 model files missing.")
        return

    # Initialize session state for Stage 2
    if "stage2_continue" not in st.session_state:
        st.session_state.stage2_continue = False

    # -------------------- Stage 1 Input Form --------------------
    with st.form("stage1_form"):
        st.subheader("📝 Farm Environment Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### **Soil Chemical Properties**")
            N = st.slider("Nitrogen (N) Content", 0, 150, 50)
            P = st.slider("Phosphorus (P) Content", 0, 150, 50)
            K = st.slider("Potassium (K) Content", 0, 150, 50)
            ph = st.number_input("Soil pH Level (0.0 - 14.0)", 0.0, 14.0, 6.5)
        
        with col2:
            st.markdown("##### **Atmospheric Parameters**")
            temp = st.number_input("Ambient Temperature (°C)", 0.0, 50.0, 25.0)
            hum = st.slider("Relative Humidity (%)", 0, 100, 50)
            rain = st.number_input("Average Rainfall (mm)", 0.0, 1000.0, 100.0)
        
        st.markdown("---")
        stage1_submit = st.form_submit_button("✨ Stage 1: Recommend Crop")

    if stage1_submit:
        input_stage1 = np.array([[N, P, K, temp, hum, ph, rain]])
        crop_encoded = stage1_model.predict(input_stage1)[0]
        crop_name = le.inverse_transform([crop_encoded])[0]

        crop_emojis = {"rice":"🌾","wheat":"🌾","maize":"🌽","coffee":"☕","cotton":"☁️", "banana":"🍌"}
        emoji = crop_emojis.get(crop_name.lower(), "🌱")

        st.markdown(f"""
            <div class="prediction-card">
                <h2>Recommended Crop: <strong>{crop_name.upper()} {emoji}</strong></h2>
                <p>
                    Based on your input, <b>{crop_name}</b> is identified as the most suitable crop.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Only offer Stage 2 if crop is rice, maize, or cotton
        if crop_name.lower() in ["rice", "maize", "cotton"]:
            st.info("Do you want to continue to yield prediction?")
            col1, col2 = st.columns(2)
            if col1.button("Yes"):
                st.session_state.stage2_continue = True
            if col2.button("No"):
                st.session_state.stage2_continue = False
        else:
            st.info("Stage 2 yield prediction is only available for Rice, Maize, and Cotton.")

    # -------------------- Stage 2 Prediction --------------------
    if st.session_state.stage2_continue:
        with st.form("stage2_form"):
            st.subheader("🌱 Stage 2: Yield Prediction Additional Inputs")

            col1, col2 = st.columns(2)
            with col1:
                sunlight = st.number_input("Sunlight Hours (hours/day)", 0.0, 24.0, 6.0)
                soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 30)
                fertilizer = st.number_input("Fertilizer Used (kg/ha)", 0, 500, 50)
            with col2:
                irrigation = st.selectbox("Irrigation Type", ["Drip", "Sprinkler", "Flood", "Rainfed"])
                soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clay", "Silty"])
                pesticide = st.number_input("Pesticide Used (kg/ha)", 0, 200, 0)

            st.markdown("---")
            stage2_submit = st.form_submit_button("🌾 Predict Yield")

        if stage2_submit:
            # Prepare Stage 2 input DataFrame (match training features)
            stage2_input = pd.DataFrame([{
                "N": N,
                "P": P,
                "K": K,
                "temperature": temp,
                "humidity": hum,
                "ph": ph,
                "rainfall": rain,
                "Sunlight_Hours": sunlight,
                "Soil_Moisture": soil_moisture,
                "Fertilizer_Used": fertilizer,
                "Pesticide_Used": pesticide,
                "Crop_Type": crop_name,
                "Irrigation_Type": irrigation,
                "Soil_Type": soil_type
            }])

            # Predict yield
            yield_pred = stage2_model.predict(stage2_input)[0]

            # Add remark for crop
            crop_remarks = {
                "Rice": "Provides high nitrogen, ideal for rapid leafy growth. Prefer this for nitrogen-deficient soils as it supports vegetative growth.",
                "Maize": "Requires balanced nutrients, thrives in moderate rainfall. Good choice for high sunlight areas.",
                "Cotton": "Needs adequate potassium for fiber development. Suitable for warmer regions."
            }
            remark = crop_remarks.get(crop_name, "Ensure proper soil fertility and climate management for best yield.")

            st.markdown(f"""
                <div class="prediction-card">
                    <h2>Predicted Yield: <strong>{yield_pred:.2f} tons/hectare</strong></h2>
                    <p>{remark}</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()


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
