import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
 
def half_circle_gauge_card(value, max_value, feature, color, unit=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': unit, 'font': {'size': 22, 'color': 'black'}},
        title={'text': feature, 'font': {'size': 24, 'color': 'black'}, 'align': 'center'},
        gauge={
            'axis': {'range': [0, max_value], 'visible': True, 'tickcolor': 'black'},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': "#CFE8C1",  # light pistachio
            'borderwidth': 1,
        },
        domain={'x': [0, 1], 'y': [0, 1]}  # full square to center
    ))
    fig.update_layout(
        paper_bgcolor="#CFE8C1",
        plot_bgcolor="#CFE8C1",
        margin=dict(t=10, b=10, l=10, r=10),
        height=250
    )
    return fig



def show_trend():
    st.title("📊 Agricultural Data Trends")
    st.info("Select a crop to view its average soil nutrients and climate conditions.")

    df = load_data()
    if df is None:
        st.warning("⚠️ Data source file ('Crop_recommendation.csv') is missing.")
        return

    # Features, max values, units, colors
    features_row1 = ["N", "P", "K"]
    features_row2 = ["ph", "temperature", "humidity", "rainfall"]

    feature_max = {"N":150,"P":150,"K":150,"ph":14,"temperature":50,"humidity":100,"rainfall":300}
    feature_units = {"N":"","P":"","K":"","ph":"","temperature":"°C","humidity":"%","rainfall":"mm"}

    colors_row1 = ["#2ca02c","#ff7f0e","#1f77b4"]
    colors_row2 = ["#9467bd","#d62728","#8c564b","#e377c2"]

    # Crop filter
    selected_crop = st.selectbox("Select Crop", sorted(df["label"].unique()))
    crop_df = df[df["label"] == selected_crop]
    mean_values = crop_df[features_row1 + features_row2].mean().round(1)
    sample_count = crop_df.shape[0]
    st.caption(f"Based on {sample_count} samples")

    # Row 1: N, P, K
    st.subheader("🌱 Soil Nutrients")
    cols1 = st.columns(len(features_row1), gap="medium")
    for i, f in enumerate(features_row1):
        with cols1[i]:
            st.markdown(
                f"<div style='background-color:#CFE8C1;padding:10px;border-radius:15px;'>", unsafe_allow_html=True)
            fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row1[i], feature_units[f])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: pH, Temperature, Humidity, Rainfall
    st.subheader("🌤️ Climate & Soil pH")
    cols2 = st.columns(len(features_row2), gap="medium")
    for i, f in enumerate(features_row2):
        with cols2[i]:
            st.markdown(
                f"<div style='background-color:#CFE8C1;padding:10px;border-radius:15px;'>", unsafe_allow_html=True)
            fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row2[i], feature_units[f])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

def show_prediction():
    st.title("🌱 Intelligent Crop Recommendation")
    
    stage1_model, le = load_stage1()
    stage2_model = load_stage2()
    
    if stage1_model is None or le is None:
        st.error("🚨 Stage 1 model files missing.")
        return
    if stage2_model is None:
        st.warning("⚠️ Stage 2 model not loaded. You can still get crop recommendation.")

    with st.form("prediction_form"):
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
        submit = st.form_submit_button("✨ Analyze & Recommend")

    if submit:
        # ------------------------
        # Stage 1: Crop Recommendation
        # ------------------------
        input_stage1 = np.array([[N, P, K, temp, hum, ph, rain]])
        crop_encoded = stage1_model.predict(input_stage1)[0]
        crop_name = le.inverse_transform([crop_encoded])[0]
        
        # Save Stage 1 results in session
        st.session_state.stage1_crop = crop_name
        st.session_state.stage1_input = {"N": N, "P": P, "K": K, "temperature": temp, "humidity": hum, "ph": ph, "rainfall": rain}

        crop_emojis = {"rice":"🌾","wheat":"🌾","maize":"🌽","coffee":"☕","cotton":"☁️","banana":"🍌"}
        emoji = crop_emojis.get(crop_name.lower(), "🌱")

        st.markdown(f"""
            <div class="prediction-card">
                <h2>Recommended Crop: <strong>{crop_name.upper()} {emoji}</strong></h2>
                <p>Based on your input, <b>{crop_name}</b> is identified as the most suitable crop.</p>
            </div>
        """, unsafe_allow_html=True)

        # ------------------------
        # Stage 2: Yield Prediction Prompt
        # ------------------------
        allowed_crops = ["rice", "maize", "cotton"]
        if crop_name.strip().lower() in allowed_crops and stage2_model is not None:
            continue_stage2 = st.radio(
                "Do you want to predict yield for this crop?",
                ("No", "Yes")
            )

            if continue_stage2 == "Yes":
                # Use session state input
                stage2_input = st.session_state.stage1_input.copy()
                stage2_input["crop"] = crop_name

                stage2_input_df = pd.DataFrame([stage2_input])
                yield_pred = stage2_model.predict(stage2_input_df)[0]

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
