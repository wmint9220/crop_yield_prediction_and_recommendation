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
# CUSTOM STYLING (DARK BROWN THEME)
# =============================
st.markdown("""
<style>
/* Main background gradient - earthy dark brown */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f4ece3 0%, #e5d7c2 50%, #d8c3a5 100%);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f0e6da !important;
    border-right: 1px solid #c7b198;
}

/* Titles & headers */
h1, h2, h3 {
    color: #4e342e !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(to right, #6d4c41, #4e342e);
    color: white;
    border-radius: 14px;
    height: 50px;
    font-weight: bold;
    font-size: 16px;
    width: 100%;
    border: none;
    box-shadow: 0 4px 8px rgba(78, 52, 46, 0.3);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background: linear-gradient(to right, #4e342e, #3e2723);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(62, 39, 35, 0.4);
}

/* Input fields */
input, textarea, select {
    border-radius: 8px !important;
    border: 1px solid #a1887f !important;
    background-color: white !important;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Divider */
hr {
    border-color: #a1887f !important;
}

/* Prediction card */
.prediction-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    border-left: 5px solid #6d4c41;
    box-shadow: 0 6px 20px rgba(78, 52, 46, 0.12);
    margin: 20px 0;
    animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.prediction-card h3 {
    color: #4e342e !important;
    margin-bottom: 12px;
    font-size: 24px;
}
.prediction-card p {
    color: #5d4037;
    line-height: 1.6;
    margin: 0;
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
    st.experimental_rerun()

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
        st.error(f"Model load error: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        return df
    except FileNotFoundError:
        return None

# =============================
# LOGIN PAGE
# =============================
def show_login():
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("<h1 style='text-align:center;color:#4e342e;'>🌱 Crop Insight Portal</h1>", unsafe_allow_html=True)
        st.success("""
        ### **Smart Agriculture AI**
        Log in to access predictive crop recommendations and analyze soil & climate trends.
        """)

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
# TREND VISUALIZATION
# =============================
def show_trend():
    st.title("📊 Agricultural Trend Insights")
    st.info("Explore soil and climate patterns to optimize crop selection.")

    df = load_data()
    if df is not None:
        st.subheader("Dataset Sample")
        st.dataframe(df.head(10))

        st.subheader("🌡️ Avg Temperature per Crop")
        temp_by_crop = df.groupby('label')['temperature'].mean().sort_values(ascending=False)
        st.bar_chart(temp_by_crop)

        st.subheader("💧 Rainfall Distribution")
        st.line_chart(df[['rainfall']].sample(min(100, len(df)), random_state=42))
    else:
        st.warning("⚠️ Dataset not found. Place 'Crop_recommendation.csv' in the app directory.")

# =============================
# CROP PREDICTION
# =============================
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
    st.sidebar.write("Logged in as: **Admin**")
    choice = st.sidebar.radio("Go to:", ["📊 Trend Visualization", "🌱 Crop Prediction"])
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()

    if choice == "📊 Trend Visualization":
        show_trend()
    else:
        show_prediction()
else:
    show_login()

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("© 2026 Crop Insight AI | Sustainable Agriculture with Machine Learning")
