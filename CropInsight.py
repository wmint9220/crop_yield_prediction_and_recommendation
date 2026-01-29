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
    st.info("Explore optimal growing conditions for different crops based on historical data.")
    
    df = load_data()
    if df is None:
        st.warning("⚠️ Data source file ('Crop_recommendation.csv') is missing.")
        return

    # ----------------------------
    # Features, max values, units, colors
    # ----------------------------
    features_row1 = ["N", "P", "K"]
    features_row2 = ["ph", "temperature", "humidity", "rainfall"]
    
    feature_max = {"N":150,"P":150,"K":150,"ph":14,"temperature":50,"humidity":100,"rainfall":300}
    feature_units = {"N":"","P":"","K":"","ph":"","temperature":"°C","humidity":"%","rainfall":"mm"}
    feature_names = {
        "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
        "ph": "pH Level", "temperature": "Temperature", 
        "humidity": "Humidity", "rainfall": "Rainfall"
    }
    
    colors_row1 = ["#2ca02c","#ff7f0e","#1f77b4"]
    colors_row2 = ["#9467bd","#d62728","#8c564b","#e377c2"]

    # ----------------------------
    # Crop Selection with Multi-Column Layout
    # ----------------------------
    col_select, col_info = st.columns([2, 1])
    
    with col_select:
        selected_crop = st.selectbox(
            "Select Crop to Analyze", 
            sorted(df["label"].unique()),
            help="Choose a crop to view its optimal growing conditions"
        )
    
    crop_df = df[df["label"] == selected_crop]
    mean_values = crop_df[features_row1 + features_row2].mean().round(1)
    sample_count = crop_df.shape[0]
    
    with col_info:
        st.metric("📋 Data Samples", f"{sample_count:,}")
    
    crop_emojis = {
        "rice":"🌾", "wheat":"🌾", "maize":"🌽", "jute":"🌿",
        "cotton":"☁️", "coconut":"🥥", "papaya":"🍈", "orange":"🍊",
        "apple":"🍎", "muskmelon":"🍈", "watermelon":"🍉", "grapes":"🍇",
        "mango":"🥭", "banana":"🍌", "pomegranate":"💎", "lentil":"🫘",
        "blackgram":"⚫", "mungbean":"🟢", "mothbeans":"🫘", "pigeonpeas":"🫘",
        "kidneybeans":"🫘", "chickpea":"🫘", "coffee":"☕" }
    emoji = crop_emojis.get(selected_crop.lower(), "🌱")
    
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4B371C 0%, #3C280D 100%); 
                    padding: 25px; border-radius: 15px; color: white; margin: 20px 0;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
            <h2 style='margin: 0; color: white;'>{emoji} {selected_crop.upper()}</h2>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>Optimal growing conditions profile</p>
        </div>
    """, unsafe_allow_html=True)
    

    # ----------------------------
    # Row 1: N, P, K
    # ----------------------------
    st.markdown("---")
    st.subheader("🌱 Soil Nutrients (NPK)")
    
    cols1 = st.columns(len(features_row1), gap="medium")
    for i, f in enumerate(features_row1):
        with cols1[i]:
            st.markdown(
                f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
                unsafe_allow_html=True
            )
            fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row1[i], feature_units[f])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
                unsafe_allow_html=True
            )
            
            # Add interpretation
            percentage = (mean_values[f] / feature_max[f]) * 100
            if percentage < 30:
                status = "🔵 Low"
            elif percentage < 60:
                status = "🟢 Moderate"
            else:
                status = "🟠 High"
            
            st.markdown(
                f"<p style='text-align:center;color:#666;font-size:12px;margin-top:-10px;'>{status}</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # Row 2: pH, Temperature, Humidity, Rainfall
    # ----------------------------
    st.subheader("🌤️ Climate & Soil Conditions")
    
    cols2 = st.columns(len(features_row2), gap="medium")
    for i, f in enumerate(features_row2):
        with cols2[i]:
            st.markdown(
                f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
                unsafe_allow_html=True
            )
            fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row2[i], feature_units[f])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
                unsafe_allow_html=True
            )
            
            # Add interpretation for each parameter
            if f == "ph":
                if mean_values[f] < 6:
                    status = "🔴 Acidic"
                elif mean_values[f] <= 7.5:
                    status = "🟢 Neutral"
                else:
                    status = "🔵 Alkaline"
            elif f == "temperature":
                if mean_values[f] < 20:
                    status = "❄️ Cool"
                elif mean_values[f] <= 30:
                    status = "🌡️ Moderate"
                else:
                    status = "🔥 Warm"
            elif f == "humidity":
                if mean_values[f] < 40:
                    status = "🏜️ Dry"
                elif mean_values[f] <= 70:
                    status = "💧 Moderate"
                else:
                    status = "💦 Humid"
            else:  # rainfall
                if mean_values[f] < 100:
                    status = "🌵 Low"
                elif mean_values[f] <= 200:
                    status = "🌧️ Moderate"
                else:
                    status = "⛈️ High"
            
            st.markdown(
                f"<p style='text-align:center;color:#666;font-size:12px;margin-top:-10px;'>{status}</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------------------------
    # Crop Comparison Feature
    # ----------------------------
    st.markdown("---")
    st.subheader(" 🔬  Compare with Other Crops")
    
    with st.expander("🔍 View Crop Comparison"):
        compare_crops = st.multiselect(
            "Select crops to compare",
            [c for c in sorted(df["label"].unique()) if c != selected_crop],
            max_selections=3
        )
        
        if compare_crops:
            comparison_data = {"Crop": [selected_crop] + compare_crops}
            
            for feature in features_row1 + features_row2:
                comparison_data[feature_names[feature]] = [mean_values[feature]]
                for crop in compare_crops:
                    crop_mean = df[df["label"] == crop][feature].mean()
                    comparison_data[feature_names[feature]].append(round(crop_mean, 1))
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Visualization
            selected_feature = st.selectbox(
                "Select parameter to visualize",
                [feature_names[f] for f in features_row1 + features_row2]
            )
            
            import plotly.express as px
            fig = px.bar(
                comp_df, 
                x="Crop", 
                y=selected_feature,
                title=f"{selected_feature} Comparison",
                color="Crop",
                text=selected_feature
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    



# def show_trend():
#     st.title("📊 Agricultural Data Trends")
#     st.info("Select a crop to view its average soil nutrients and climate conditions.")

#     df = load_data()
#     if df is None:
#         st.warning("⚠️ Data source file ('Crop_recommendation.csv') is missing.")
#         return

#     # ----------------------------
#     # Features, max values, units, colors
#     # ----------------------------
#     features_row1 = ["N", "P", "K"]
#     features_row2 = ["ph", "temperature", "humidity", "rainfall"]

#     feature_max = {"N":150,"P":150,"K":150,"ph":14,"temperature":50,"humidity":100,"rainfall":300}
#     feature_units = {"N":"","P":"","K":"","ph":"","temperature":"°C","humidity":"%","rainfall":"mm"}

#     colors_row1 = ["#2ca02c","#ff7f0e","#1f77b4"]
#     colors_row2 = ["#9467bd","#d62728","#8c564b","#e377c2"]

#     # ----------------------------
#     # Crop filter
#     # ----------------------------
#     selected_crop = st.selectbox("Select Crop", sorted(df["label"].unique()))
#     crop_df = df[df["label"] == selected_crop]
#     mean_values = crop_df[features_row1 + features_row2].mean().round(1)
#     sample_count = crop_df.shape[0]
#     st.caption(f"Based on {sample_count} samples")

#     # ----------------------------
#     # Row 1: N, P, K
#     # ----------------------------
#     st.subheader("🌱 Soil Nutrients")
#     cols1 = st.columns(len(features_row1), gap="medium")
#     for i, f in enumerate(features_row1):
#         with cols1[i]:
#             st.markdown(
#                 f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
#                 unsafe_allow_html=True
#             )
#             fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row1[i], feature_units[f])
#             st.plotly_chart(fig, use_container_width=True)
#             st.markdown(
#                 f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
#                 unsafe_allow_html=True
#             )
#             st.markdown("</div>", unsafe_allow_html=True)

#     # ----------------------------
#     # Row 2: pH, Temperature, Humidity, Rainfall
#     # ----------------------------
#     st.subheader("🌤️ Climate & Soil pH")
#     cols2 = st.columns(len(features_row2), gap="medium")
#     for i, f in enumerate(features_row2):
#         with cols2[i]:
#             st.markdown(
#                 f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
#                 unsafe_allow_html=True
#             )
#             fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row2[i], feature_units[f])
#             st.plotly_chart(fig, use_container_width=True)
#             st.markdown(
#                 f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
#                 unsafe_allow_html=True
#             )
#             st.markdown("</div>", unsafe_allow_html=True)


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
        # Stage 1: Crop Recommendation
        input_stage1 = np.array([[N, P, K, temp, hum, ph, rain]])
        crop_encoded = stage1_model.predict(input_stage1)[0]
        crop_name = le.inverse_transform([crop_encoded])[0]
        
        st.session_state.stage1_crop = crop_name
        st.session_state.stage1_input = {"N": N, "P": P, "K": K, "temperature": temp, "humidity": hum, "ph": ph, "rainfall": rain}
        
        crop_emojis = {
        "rice":"🌾", "wheat":"🌾", "maize":"🌽", "jute":"🌿",
        "cotton":"☁️", "coconut":"🥥", "papaya":"🍈", "orange":"🍊",
        "apple":"🍎", "muskmelon":"🍈", "watermelon":"🍉", "grapes":"🍇",
        "mango":"🥭", "banana":"🍌", "pomegranate":"💎", "lentil":"🫘",
        "blackgram":"⚫", "mungbean":"🟢", "mothbeans":"🫘", "pigeonpeas":"🫘",
        "kidneybeans":"🫘", "chickpea":"🫘", "coffee":"☕" }
        
        emoji = crop_emojis.get(crop_name.lower(), "🌱")
    
        st.markdown(f"""
            <div class="prediction-card">
                <h2>Recommended Crop: <strong>{crop_name.upper()} {emoji}</strong></h2>
                <p>Based on your input, <b>{crop_name}</b> is identified as the most suitable crop. This recommendation takes into account the specific 
                soil pH and NPK balance required for this species to thrive under the current temperature and rainfall projections.</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Suggestion Improvement")
        df = load_data()
        crop_optimal = df[df["label"] == crop_name].mean(numeric_only=True)
        
        # Calculate indices
        thi = temp - (0.55 - 0.0055 * hum) * (temp - 14.4)
        sfi = (N + P + K) / 3
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🌡️ Temperature-Humidity Index", f"{thi:.1f}")
            
            # THI interpretation
            if thi < 15:
                thi_status = "❄️ **Cold Stress** - May slow crop growth"
                thi_color = "#3498db"
            elif 15 <= thi < 22:
                thi_status = "✅ **Optimal** - Ideal growing conditions"
                thi_color = "#2ecc71"
            elif 22 <= thi < 28:
                thi_status = "⚠️ **Warm** - Monitor water needs"
                thi_color = "#f39c12"
            else:
                thi_status = "🔥 **Heat Stress** - Risk to crop health"
                thi_color = "#e74c3c"
            
            st.markdown(f"<p style='color:{thi_color};font-size:14px;'>{thi_status}</p>", unsafe_allow_html=True)
            
            with st.expander("ℹ️ What is THI?"):
                st.write("""
                **Temperature-Humidity Index** measures environmental stress on crops.
                
                - **Below 15**: Cold stress conditions
                - **15-22**: Optimal comfort zone
                - **22-28**: Moderate heat stress
                - **Above 28**: Severe heat stress
                
                Higher humidity reduces heat stress effects.
                """)
        
        with col2:
            st.metric("🌱 Soil Fertility Index", f"{sfi:.1f}")
            
            # SFI interpretation
            if sfi < 30:
                sfi_status = "📉 **Low** - Needs fertilization"
                sfi_color = "#e74c3c"
            elif 30 <= sfi < 60:
                sfi_status = "📊 **Moderate** - Adequate nutrients"
                sfi_color = "#f39c12"
            elif 60 <= sfi < 90:
                sfi_status = "📈 **Good** - Well-balanced soil"
                sfi_color = "#2ecc71"
            else:
                sfi_status = "⚡ **Excellent** - Nutrient-rich soil"
                sfi_color = "#27ae60"
            
            st.markdown(f"<p style='color:{sfi_color};font-size:14px;'>{sfi_status}</p>", unsafe_allow_html=True)
            
            with st.expander("ℹ️ What is SFI?"):
                st.write("""
                **Soil Fertility Index** reflects overall nutrient availability (N+P+K average).
                
                - **0-30**: Low - Requires fertilizer input
                - **30-60**: Moderate - Baseline fertility
                - **60-90**: Good - Supports healthy growth
                - **90+**: Excellent - Premium soil quality
                
                Balance all three nutrients for best results.
                """)
        
        # st.markdown("##### 🎯 Match Score")
        
        # params = {
        #     "Nitrogen (N)": (N, crop_optimal["N"], 150),
        #     "Phosphorus (P)": (P, crop_optimal["P"], 150),
        #     "Potassium (K)": (K, crop_optimal["K"], 150),
        #     "pH Level": (ph, crop_optimal["ph"], 14),
        #     "Temperature": (temp, crop_optimal["temperature"], 50),
        #     "Humidity": (hum, crop_optimal["humidity"], 100),
        #     "Rainfall": (rain, crop_optimal["rainfall"], 300)
        # }
        
        # for param_name, (user_val, opt_val, max_val) in params.items():
        #     match_pct = 100 - abs((user_val - opt_val) / opt_val * 100)
        #     match_pct = max(0, min(100, match_pct))  # Clamp between 0-100
            
        #     # Color based on match
        #     if match_pct >= 90:
        #         color = "🟢"
        #         bar_color = "#28a745"
        #     elif match_pct >= 70:
        #         color = "🟡"
        #         bar_color = "#ffc107"
        #     else:
        #         color = "🔴"
        #         bar_color = "#dc3545"
            
        #     st.markdown(f"**{color} {param_name}**: Your: {user_val:.1f} | Optimal: {opt_val:.1f}")
        #     st.progress(match_pct / 100)
  
        st.markdown("##### 🎯 Parameter Match Score")
        params = {
            "Nitrogen (N)": (N, crop_optimal["N"], 150),
            "Phosphorus (P)": (P, crop_optimal["P"], 150),
            "Potassium (K)": (K, crop_optimal["K"], 150),
            "pH Level": (ph, crop_optimal["ph"], 14),
            "Temperature": (temp, crop_optimal["temperature"], 50),
            "Humidity": (hum, crop_optimal["humidity"], 100),
            "Rainfall": (rain, crop_optimal["rainfall"], 300)
        }
        
        # Create two columns for better layout
        col_left, col_right = st.columns(2)
        
        param_items = list(params.items())
        mid_point = len(param_items) // 2 + len(param_items) % 2
        
        with col_left:
            for param_name, (user_val, opt_val, max_val) in param_items[:mid_point]:
                match_pct = 100 - abs((user_val - opt_val) / opt_val * 100) if opt_val != 0 else 100
                match_pct = max(0, min(100, match_pct))
                
                if match_pct >= 90:
                    color = "🟢"
                elif match_pct >= 70:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"**{color} {param_name}**")
                st.progress(match_pct / 100)
                st.caption(f"Your: {user_val:.1f} | Optimal: {opt_val:.1f} | Match: {match_pct:.0f}%")
                st.markdown("<br>", unsafe_allow_html=True)
        
        with col_right:
            for param_name, (user_val, opt_val, max_val) in param_items[mid_point:]:
                match_pct = 100 - abs((user_val - opt_val) / opt_val * 100) if opt_val != 0 else 100
                match_pct = max(0, min(100, match_pct))
                
                if match_pct >= 90:
                    color = "🟢"
                elif match_pct >= 70:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"**{color} {param_name}**")
                st.progress(match_pct / 100)
                st.caption(f"Your: {user_val:.1f} | Optimal: {opt_val:.1f} | Match: {match_pct:.0f}%")
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Overall Match Summary
        overall_matches = []
        for param_name, (user_val, opt_val, max_val) in params.items():
            match_pct = 100 - abs((user_val - opt_val) / opt_val * 100) if opt_val != 0 else 100
            match_pct = max(0, min(100, match_pct))
            overall_matches.append(match_pct)
        
        avg_match = sum(overall_matches) / len(overall_matches)
        
        if avg_match >= 90:
            match_emoji = "🌟"
            match_text = "Excellent Match!"
            match_color = "#2ecc71"
        elif avg_match >= 75:
            match_emoji = "👍"
            match_text = "Good Match"
            match_color = "#27ae60"
        elif avg_match >= 60:
            match_emoji = "⚠️"
            match_text = "Fair Match"
            match_color = "#f39c12"
        else:
            match_emoji = "❗"
            match_text = "Needs Adjustment"
            match_color = "#e74c3c"

        st.markdown(f"""
            <div style='background-color:{match_color}22; padding:20px; border-radius:10px; text-align:center; border:2px solid {match_color};'>
                <h3 style='color:{match_color}; margin:0;'>{match_emoji} Overall Match: {avg_match:.1f}%</h3>
                <p style='margin:5px 0; font-size:16px;'><b>{match_text}</b></p>
                <p style='margin:0; font-size:14px; color:#666;'>
                    {'Your conditions are very close to optimal!' if avg_match >= 90 else
                     'Your conditions are good for this crop.' if avg_match >= 75 else
                     'Consider adjusting some parameters for better yield.' if avg_match >= 60 else
                     'Several parameters need adjustment for optimal growth.'}
                </p>
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
