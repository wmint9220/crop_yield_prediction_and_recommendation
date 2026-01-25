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
# SESSION STATE
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
# LOAD MODELS
# =============================
@st.cache_resource
def load_stage1():
    return joblib.load("crop_recommendation_rf.pkl"), joblib.load("label_encoder.pkl")

@st.cache_resource
def load_stage2():
    return joblib.load("xgboost_yield_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

# =============================
# LOGIN PAGE
# =============================
def show_login():
    st.title("🔐 Crop Insight Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# =============================
# TREND PAGE
# =============================
def show_trend():
    st.title("📊 Agricultural Data Trends")
    df = load_data()
    if df is not None:
        temp_by_crop = df.groupby("label")["temperature"].mean()
        st.bar_chart(temp_by_crop)

# =============================
# PREDICTION PAGE
# =============================
def show_prediction():
    st.title("🌱 Intelligent Crop Recommendation")

    stage1_model, le = load_stage1()
    stage2_model = load_stage2()

    with st.form("stage1_form"):
        col1, col2 = st.columns(2)

        with col1:
            N = st.slider("Nitrogen (N)", 0, 150, 50)
            P = st.slider("Phosphorus (P)", 0, 150, 50)
            K = st.slider("Potassium (K)", 0, 150, 50)
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

        with col2:
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
            hum = st.slider("Humidity (%)", 0, 100, 50)
            rain = st.number_input("Rainfall (mm)", 0.0, 1000.0, 100.0)

        submit = st.form_submit_button("🌱 Recommend Crop")

    if not submit:
        return

    # -------- Stage 1 Prediction --------
    X1 = np.array([[N, P, K, temp, hum, ph, rain]])
    crop_encoded = stage1_model.predict(X1)[0]
    crop = le.inverse_transform([crop_encoded])[0]

    st.markdown(f"""
    <div class="prediction-card">
        <h2>Recommended Crop: <b>{crop.upper()}</b></h2>
    </div>
    """, unsafe_allow_html=True)

    # Save stage 1 result
    st.session_state["stage1"] = {
        "N": N, "P": P, "K": K,
        "Soil_pH": ph,
        "Temperature": temp,
        "Humidity": hum,
        "Rainfall": rain,
        "Crop_Type": crop
    }

    # -------- Stage 2 Conditions --------
    allowed_crops = ["Rice", "Maize", "Cotton"]

    if crop not in allowed_crops:
        st.warning(
            f"⚠️ Yield prediction is available only for **Rice, Maize, and Cotton**.\n\n"
            f"Selected crop **{crop}** is not supported."
        )
        return

    st.markdown("### 👉 Continue to Yield Prediction?")
    go_stage2 = st.radio(
        "Do you want to estimate yield?",
        ["No", "Yes"],
        horizontal=True
    )

    if go_stage2 == "No":
        st.info("You may stop here or change inputs.")
        return

    # -------- Stage 2 Inputs --------
    st.subheader("🌾 Stage 2: Yield Estimation")

    soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 40.0)
    sunlight = st.slider("Sunlight Hours", 0.0, 15.0, 8.0)
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", 0.0, 500.0, 50.0)
    pesticide = st.number_input("Pesticide Used (L/ha)", 0.0, 50.0, 5.0)

    soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clay", "Silty"])
    irrigation = st.selectbox("Irrigation Type", ["Drip", "Sprinkler", "Flood", "Rainfed"])

    if st.button("📈 Predict Yield"):
        stage1 = st.session_state["stage1"]

        X2 = pd.DataFrame([{
            "N": stage1["N"],
            "P": stage1["P"],
            "K": stage1["K"],
            "Soil_pH": stage1["Soil_pH"],
            "Temperature": stage1["Temperature"],
            "Humidity": stage1["Humidity"],
            "Rainfall": stage1["Rainfall"],
            "Soil_Moisture": soil_moisture,
            "Sunlight_Hours": sunlight,
            "Crop_Type": stage1["Crop_Type"],
            "Soil_Type": soil_type,
            "Irrigation_Type": irrigation,
            "Fertilizer_Used": fertilizer,
            "Pesticide_Used": pesticide
        }])

        yield_pred = stage2_model.predict(X2)[0]

        st.markdown(f"""
        <div class="prediction-card">
            <h2>🌱 Predicted Yield</h2>
            <h3>{yield_pred:.2f} tons / hectare</h3>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

# =============================
# MAIN NAVIGATION
# =============================
if st.session_state.logged_in:
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to:", ["📊 Trend Visualization", "🌱 Crop Prediction"])
    if st.sidebar.button("🚪 Logout"):
        logout()

    if "Trend" in choice:
        show_trend()
    else:
        show_prediction()
else:
    show_login()

st.caption("© 2026 Crop Insight AI. Integrated Machine Learning for Sustainable Agriculture.")
