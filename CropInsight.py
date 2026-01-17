import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="🌱 Crop Insight | Smart Agriculture",
    page_icon="🌾",
    layout="wide"
)

# =============================
# STYLING
# =============================
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

def logout():
    st.session_state.logged_in = False
    st.rerun()

# =============================
# LOAD MODELS
# =============================
@st.cache_resource
def load_stage1():
    try:
        model = joblib.load("crop_recommendation_rf.pkl")
        le = joblib.load("label_encoder.pkl")
        return model, le
    except:
        return None, None

@st.cache_resource
def load_stage2():
    try:
        return joblib.load("XGboost_yield_model.pkl")  # full pipeline
    except:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("Crop_recommendation.csv")
    except:
        return None

# =============================
# AGRONOMIC REMARKS
# =============================
def generate_remarks(N, P, K, ph, crop):
    remarks = []

    if N < 40:
        remarks.append("Low nitrogen detected. Increase nitrogen to support leafy and vegetative growth.")
    elif N > 100:
        remarks.append("High nitrogen availability detected. Suitable for crops with high vegetative demand.")

    if P < 30:
        remarks.append("Low phosphorus may limit root development and flowering.")
    if K < 40:
        remarks.append("Low potassium detected. Potassium improves stress tolerance and yield quality.")

    if ph < 5.5:
        remarks.append("Soil is acidic. Liming may improve nutrient availability.")
    elif ph > 7.5:
        remarks.append("Soil is alkaline. Organic matter can help balance soil pH.")

    remarks.append(f"{crop} is agronomically suitable under the given soil and climate conditions.")

    return remarks

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
        temp_by_crop = df.groupby("label")["temperature"].mean().sort_values()
        st.subheader("🌡️ Average Temperature by Crop")
        st.bar_chart(temp_by_crop)
    else:
        st.warning("Dataset not found.")

# =============================
# PREDICTION PAGE (STAGE 1 + 2)
# =============================
def show_prediction():
    st.title("🌱 Intelligent Crop & Yield Prediction")

    stage1_model, le = load_stage1()
    stage2_model = load_stage2()

    if stage1_model is None or stage2_model is None:
        st.error("🚨 Model files not found.")
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            N = st.slider("Nitrogen (N)", 0, 150, 50)
            P = st.slider("Phosphorus (P)", 0, 150, 50)
            K = st.slider("Potassium (K)", 0, 150, 50)
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

        with col2:
            temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
            hum = st.slider("Humidity (%)", 0, 100, 60)
            rain = st.number_input("Rainfall (mm)", 0.0, 1000.0, 120.0)

        submit = st.form_submit_button("🌾 Predict")

    if submit:
        # ---------- Stage 1 ----------
        stage1_input = np.array([[N, P, K, temp, hum, ph, rain]])
        crop_encoded = stage1_model.predict(stage1_input)
        crop = le.inverse_transform(crop_encoded)[0]

        # ---------- Stage 2 ----------
        stage2_input = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "ph": ph,
            "temperature": temp,
            "humidity": hum,
            "rainfall": rain,
            "Crop_Type": crop,
            "Soil_Type": "Loamy",
            "Irrigation_Type": "Rainfed",
            "Sunlight_Hours": 6.0,
            "Soil_Moisture": 45.0,
            "Fertilizer_Used": 120.0
        }])

        predicted_yield = stage2_model.predict(stage2_input)[0]

        remarks = generate_remarks(N, P, K, ph, crop)

        st.markdown(f"""
        <div class="prediction-card">
            <h2>🌱 Recommended Crop: <strong>{crop.upper()}</strong></h2>
            <h3>📈 Estimated Yield: <strong>{predicted_yield:.2f} ton/ha</strong></h3>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🧾 Agronomic Recommendations")
        for r in remarks:
            st.markdown(f"- ✅ {r}")

        st.balloons()

# =============================
# MAIN APP
# =============================
if st.session_state.logged_in:
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to:", ["📊 Trends", "🌱 Prediction"])
    if st.sidebar.button("🚪 Logout"):
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
st.caption("© 2026 Crop Insight AI | Two-Stage Machine Learning for Smart Agriculture")
