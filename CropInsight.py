import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, Color
from reportlab.lib import colors
from datetime import datetime
import io
import os


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
            'bgcolor': "#CFE8C1", 
            'borderwidth': 1,
        },
        domain={'x': [0, 1], 'y': [0, 1]}  
    ))
    fig.update_layout(
        paper_bgcolor="#CFE8C1",
        plot_bgcolor="#CFE8C1",
        margin=dict(t=45, b=10, l=10, r=10),
        height=270
    )
    return fig

def create_crop_prediction_pdf(
    N, P, K, ph, temperature, humidity, rainfall,
    recommended_crop, thi, sfi, parameter_matches, overall_match,
    soil_moisture=None, soil_type=None, sunlight_hours=None,
    irrigation_type=None, fertilizer_used=None, pesticide_used=None,
    predicted_yield=None
):
    """Generate PDF report in memory and return BytesIO object"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch,
                           bottomMargin=0.75*inch, leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=26,
                                 textColor=HexColor('#2c5282'), spaceAfter=10,
                                 alignment=TA_CENTER, fontName='Helvetica-Bold')
    
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12,
                                    textColor=HexColor('#666666'), alignment=TA_CENTER, spaceAfter=30)
    
    section_header = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=18,
                                   textColor=HexColor('#2c5282'), spaceAfter=15,
                                   spaceBefore=20, fontName='Helvetica-Bold')
    
    subsection_header = ParagraphStyle('SubsectionHeader', parent=styles['Heading3'], fontSize=14,
                                      textColor=HexColor('#4a5568'), spaceAfter=10,
                                      spaceBefore=15, fontName='Helvetica-Bold')
    
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontSize=11,
                               leading=16, spaceAfter=12, alignment=TA_JUSTIFY)
    
    # Header
    story.append(Paragraph("🌱 Intelligent Crop Recommendation Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Input Parameters
    story.append(Paragraph("📝 Input Parameters", section_header))
    input_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Nitrogen (N)', f'{N}', 'ppm'],
        ['Phosphorus (P)', f'{P}', 'ppm'],
        ['Potassium (K)', f'{K}', 'ppm'],
        ['Soil pH Level', f'{ph:.1f}', 'pH'],
        ['Temperature', f'{temperature:.1f}', '°C'],
        ['Humidity', f'{humidity}', '%'],
        ['Rainfall', f'{rainfall:.1f}', 'mm']
    ]
    
    input_table = Table(input_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f7fafc')),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f7fafc')])
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommended Crop
    story.append(Paragraph("🌾 Recommended Crop", section_header))
    crop_emojis = {"rice":"🌾", "wheat":"🌾", "maize":"🌽", "jute":"🌿", "cotton":"☁️",
                   "coconut":"🥥", "papaya":"🍈", "orange":"🍊", "apple":"🍎",
                   "muskmelon":"🍈", "watermelon":"🍉", "grapes":"🍇", "mango":"🥭",
                   "banana":"🍌", "pomegranate":"💎", "lentil":"🫘", "blackgram":"⚫",
                   "mungbean":"🟢", "mothbeans":"🫘", "pigeonpeas":"🫘",
                   "kidneybeans":"🫘", "chickpea":"🫘", "coffee":"☕"}
    
    emoji = crop_emojis.get(recommended_crop.lower(), "🌱")
    story.append(Paragraph(f"<b>{emoji} {recommended_crop.upper()}</b>",
                          ParagraphStyle('CropName', parent=body_style, fontSize=16,
                                       textColor=HexColor('#2c5282'), alignment=TA_CENTER)))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"Based on your soil and environmental parameters, <b>{recommended_crop}</b> is identified as the most suitable crop.",
        body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Environmental Indices
    story.append(Paragraph("📊 Environmental Indices", subsection_header))
    
    if thi < 15:
        thi_status = "Cold Stress - May slow crop growth"
    elif 15 <= thi < 22:
        thi_status = "Optimal - Ideal growing conditions"
    elif 22 <= thi < 28:
        thi_status = "Warm - Monitor water needs"
    else:
        thi_status = "Heat Stress - Risk to crop health"
    
    if sfi < 30:
        sfi_status = "Low - Needs fertilization"
    elif 30 <= sfi < 60:
        sfi_status = "Moderate - Adequate nutrients"
    elif 60 <= sfi < 90:
        sfi_status = "Good - Well-balanced soil"
    else:
        sfi_status = "Excellent - Nutrient-rich soil"
    
    indices_data = [
        ['Index', 'Value', 'Status'],
        ['Temperature-Humidity Index (THI)', f'{thi:.1f}', thi_status],
        ['Soil Fertility Index (SFI)', f'{sfi:.1f}', sfi_status]
    ]
    
    indices_table = Table(indices_data, colWidths=[2.2*inch, 1.2*inch, 2.1*inch])
    indices_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0'))
    ]))
    story.append(indices_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Parameter Match Analysis
    story.append(Paragraph("🎯 Parameter Match Analysis", subsection_header))
    match_data = [['Parameter', 'Your Value', 'Optimal Value', 'Match %']]
    
    for param_name, (user_val, opt_val, match_pct) in parameter_matches.items():
        match_data.append([param_name, f'{user_val:.1f}', f'{opt_val:.1f}', f'{match_pct:.1f}%'])
    
    match_table = Table(match_data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.1*inch])
    match_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f7fafc')])
    ]))
    story.append(match_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Overall Match
    if overall_match >= 90:
        match_text = "Excellent Match! Your conditions are very close to optimal."
        match_color = HexColor('#2ecc71')
    elif overall_match >= 75:
        match_text = "Good Match - Your conditions are good for this crop."
        match_color = HexColor('#27ae60')
    elif overall_match >= 60:
        match_text = "Fair Match - Consider adjusting some parameters."
        match_color = HexColor('#f39c12')
    else:
        match_text = "Needs Adjustment - Several parameters need improvement."
        match_color = HexColor('#e74c3c')
    
    story.append(Paragraph(f"<b>Overall Match Score: {overall_match:.1f}%</b>",
                          ParagraphStyle('MatchScore', parent=body_style, fontSize=14,
                                       textColor=match_color, alignment=TA_CENTER)))
    story.append(Paragraph(match_text,
                          ParagraphStyle('MatchText', parent=body_style, fontSize=11, alignment=TA_CENTER)))
    
    # Stage 2: Yield Prediction (if available)
    if predicted_yield is not None:
        story.append(PageBreak())
        story.append(Paragraph("🌾 Yield Prediction Analysis", section_header))
        story.append(Paragraph("Additional Farm Parameters", subsection_header))
        
        stage2_data = [
            ['Parameter', 'Value', 'Unit'],
            ['Soil Moisture', f'{soil_moisture}', '%'],
            ['Soil Type', soil_type, '-'],
            ['Sunlight Hours', f'{sunlight_hours:.1f}', 'hours/day'],
            ['Irrigation Type', irrigation_type, '-'],
            ['Fertilizer Used', f'{fertilizer_used:.1f}', 'kg/hectare'],
            ['Pesticide Used', f'{pesticide_used:.1f}', 'kg/hectare']
        ]
        
        stage2_table = Table(stage2_data, colWidths=[2.5*inch, 1.5*inch, 1*inch])
        stage2_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5282')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f7fafc')])
        ]))
        story.append(stage2_table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph("🎯 Predicted Yield", subsection_header))
        story.append(Paragraph(f"<b>{predicted_yield:.2f} tons/hectare</b>",
                              ParagraphStyle('YieldValue', parent=body_style, fontSize=18,
                                           textColor=HexColor('#2c5282'), alignment=TA_CENTER)))
        story.append(Spacer(1, 0.2*inch))
        
        crop_remarks = {
            "rice": "Rice thrives with high nitrogen and consistent water management.",
            "maize": "Maize requires balanced NPK nutrients and adequate sunlight.",
            "cotton": "Cotton needs sufficient potassium for fiber quality."
        }
        remark = crop_remarks.get(recommended_crop.lower(), "Ensure proper soil fertility.")
        story.append(Paragraph(remark, body_style))
    
    # Footer
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(
        "⚠️ <i>Note: Predictions are based on historical data patterns.</i>",
        ParagraphStyle('Footer', parent=body_style, fontSize=9, textColor=HexColor('#666666'), alignment=TA_CENTER)))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

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


# def show_trend():
#     st.title("📊 Agricultural Data Trends")           
#     st.markdown("Welcome to the **Crop Insight**. This platform leverages historical soil and climate data to identify optimal growing conditions and crop alternatives.")
#     st.info("💡 The dashboard analyzes the relationship between soil nutrients, environmental factors, and pH levels. It is designed to support data-driven decision-making for sustainable farming.")
#     df = load_data()
#     if df is None:
#         st.warning("⚠️ Data source file ('Crop_recommendation.csv') is missing.")
#         return

#     # ----------------------------
#     # Features, max values, units, colors
#     # ----------------------------
#     features_row1 = ["N", "P", "K"]
#     features_row2 = ["ph", "temperature", "humidity", "rainfall"]
    
#     feature_max = {"N":150,"P":150,"K":200,"ph":14,"temperature":50,"humidity":100,"rainfall":300}
#     feature_units = {"N":"","P":"","K":"","ph":"","temperature":"°C","humidity":"%","rainfall":"mm"}
#     feature_names = {
#         "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
#         "ph": "pH Level", "temperature": "Temperature", 
#         "humidity": "Humidity", "rainfall": "Rainfall"
#     }
    
#     colors_row1 = ["#2ca02c","#ff7f0e","#1f77b4"]
#     colors_row2 = ["#9467bd","#d62728","#8c564b","#e377c2"]



#     # ----------------------------
#     # Data Insights Section 
#     # ----------------------------
#     with st.expander("📊 **About the Dataset**"):
#         st.markdown("Explore the underlying relationships and requirements across all crop types.")
        
#         # --- 1. FEATURE HEATMAP ---
#         st.subheader("🔥 Feature Heatmap Across All Crops")
#         st.markdown("""
#                     **What this shows:** This heatmap compares the **average requirements** of every crop side-by-side.
                    
#                     * **Vertical Axis:** The features (N, P, K, Temp, etc.).
#                     * **Horizontal Axis:** The different crop types.
#                     * **Colors:** 🟩 **Green** indicates high values, while 🟥 **Red** indicates low values.
                    
#                     **💡 Pro-Tip:** Use this to find "extreme" crops. For example, you can quickly spot which crops need the most rainfall or the highest Nitrogen levels compared to all others.
#         """)
        
#         # Create pivot table
#         heatmap_data = df.groupby("label")[features_row1 + features_row2].mean()
        
#         fig_heat = px.imshow(
#             heatmap_data.T,
#             labels=dict(x="Crop", y="Feature", color="Value"),
#             aspect="auto",
#             # Use .T to put crops on the horizontal axis
#             color_continuous_scale="RdYlGn"
#         )
#         fig_heat.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
#         st.plotly_chart(fig_heat, use_container_width=True)
        
#         st.divider() # Visual break between charts
    
#         # --- 2. CORRELATION MATRIX ---
#         st.subheader("🔗 Feature Correlations")
#         st.markdown("""
#                     **What this shows:** This matrix measures the **strength of the relationship** between two variables.
                    
#                     * **Scale:** Values range from **+1.0 to -1.0**.
#                     * **Positive (+):** As one feature increases, the other tends to increase (e.g., Phosphorus and Potassium often have high positive correlation).
#                     * **Negative (-):** As one increases, the other decreases.
#                     * **Near 0:** No linear relationship between the features.
                    
#                     **💡 Why it matters:** If two features are almost perfectly correlated (near 1.0), they provide the same information. This helps in **feature selection** for your machine learning model.
#         """)
        
#         corr_matrix = df[features_row1 + features_row2].corr()
        
#         fig_corr = px.imshow(
#             corr_matrix,
#             text_auto='.2f',
#             aspect="auto",
#             color_continuous_scale="RdBu_r",
#             labels=dict(color="Correlation")
#         )
#         fig_corr.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
#         st.plotly_chart(fig_corr, use_container_width=True)

#     col1, col2, col3, col4 = st.columns(4)
            
#     with col1:
#         st.metric("🌾 Total Crops", df["label"].nunique())
#     with col2:
#         st.metric("📋 Total Samples", len(df))
#     with col3:
#         st.metric("📊 Features", len(features_row1 + features_row2))
#     with col4:
#         avg_samples = len(df) / df["label"].nunique()
#         st.metric("📈 Avg Samples/Crop", f"{avg_samples:.0f}")

    
#     selected_crop = st.selectbox(
#             "Select Crop to Analyze", 
#             sorted(df["label"].unique()),
#             help="Choose a crop to view its optimal growing conditions"
#     )
    
#     crop_df = df[df["label"] == selected_crop]
#     mean_values = crop_df[features_row1 + features_row2].mean().round(1)

    
#     crop_emojis = {
#         "rice":"🌾", "wheat":"🌾", "maize":"🌽", "jute":"🌿",
#         "cotton":"☁️", "coconut":"🥥", "papaya":"🍈", "orange":"🍊",
#         "apple":"🍎", "muskmelon":"🍈", "watermelon":"🍉", "grapes":"🍇",
#         "mango":"🥭", "banana":"🍌", "pomegranate":"💎", "lentil":"🫘",
#         "blackgram":"⚫", "mungbean":"🟢", "mothbeans":"🫘", "pigeonpeas":"🫘",
#         "kidneybeans":"🫘", "chickpea":"🫘", "coffee":"☕"
#     }
#     emoji = crop_emojis.get(selected_crop.lower(), "🌱")
    
#     st.markdown(f"""
#         <div style='background: linear-gradient(135deg, #4B371C 0%, #3C280D 100%); 
#                     padding: 25px; border-radius: 15px; color: white; margin: 20px 0;
#                     box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
#             <h2 style='margin: 0; color: white;'>{emoji} {selected_crop.upper()}</h2>
#             <p style='margin: 10px 0 0 0; opacity: 0.9;'>Optimal growing conditions profile</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     # ----------------------------
#     # TABS: Overview and Comparison
#     # ----------------------------
#     tab1, tab2  = st.tabs(["📊 Crop Overview", "🔬 Crop Comparison"])

#     with tab1:
#         with st.expander("📅 **Growing Insights**", expanded=False):
#             # This text stays inside the expander but above the columns
#             st.markdown(f"""
#                 These ranges show the **minimum and maximum** values observed in the dataset for **{selected_crop}**. 
#                 They represent the tolerance limits of this crop.
#             """)
            
#             # Create the two columns inside the expander
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown(f"""
#                     **🌡️ Temperature Range:** {crop_df['temperature'].min():.1f}°C - {crop_df['temperature'].max():.1f}°C
                    
#                     **💧 Humidity Range:** {crop_df['humidity'].min():.1f}% - {crop_df['humidity'].max():.1f}%
#                 """)
                
#             with col2:
#                 st.markdown(f"""
#                     **🌧️ Rainfall Range:** {crop_df['rainfall'].min():.1f}mm - {crop_df['rainfall'].max():.1f}mm
                    
#                     **⚗️ pH Range:** {crop_df['ph'].min():.1f} - {crop_df['ph'].max():.1f}
#                 """)
#         # ----------------------------
#         # Row 1: N, P, K
#         # ----------------------------
#         st.subheader("🌱 Soil Nutrients (NPK)")
        
#         cols1 = st.columns(len(features_row1), gap="medium")
#         for i, f in enumerate(features_row1):
#             with cols1[i]:
#                 st.markdown(
#                     f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
#                     unsafe_allow_html=True
#                 )
#                 fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row1[i], feature_units[f])
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.markdown(
#                     f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
#                     unsafe_allow_html=True
#                 )
                
#                 # Add interpretation
#                 percentage = (mean_values[f] / feature_max[f]) * 100
#                 if percentage < 30:
#                     status = "🔵 Low"
#                 elif percentage < 60:
#                     status = "🟢 Moderate"
#                 else:
#                     status = "🟠 High"
                
#                 st.markdown(
#                     f"<p style='text-align:center;color:#666;font-size:12px;margin-top:-10px;'>{status}</p>",
#                     unsafe_allow_html=True
#                 )
#                 st.markdown("</div>", unsafe_allow_html=True)

#         # ----------------------------
#         # Row 2: pH, Temperature, Humidity, Rainfall
#         # ----------------------------
#         st.subheader("🌤️ Climate & Soil Conditions")
        
#         cols2 = st.columns(len(features_row2), gap="medium")
#         for i, f in enumerate(features_row2):
#             with cols2[i]:
#                 st.markdown(
#                     f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
#                     unsafe_allow_html=True
#                 )
#                 fig = half_circle_gauge_card(mean_values[f], feature_max[f], f, colors_row2[i], feature_units[f])
#                 st.plotly_chart(fig, use_container_width=True)
#                 st.markdown(
#                     f"<p style='text-align:center;font-weight:bold;color:black;'>{mean_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
#                     unsafe_allow_html=True
#                 )
                
#                 # Add interpretation for each parameter
#                 if f == "ph":
#                     if mean_values[f] < 6:
#                         status = "🔴 Acidic"
#                     elif mean_values[f] <= 7.5:
#                         status = "🟢 Neutral"
#                     else:
#                         status = "🔵 Alkaline"
#                 elif f == "temperature":
#                     if mean_values[f] < 20:
#                         status = "❄️ Cool"
#                     elif mean_values[f] <= 30:
#                         status = "🌡️ Moderate"
#                     else:
#                         status = "🔥 Warm"
#                 elif f == "humidity":
#                     if mean_values[f] < 40:
#                         status = "🏜️ Dry"
#                     elif mean_values[f] <= 70:
#                         status = "💧 Moderate"
#                     else:
#                         status = "💦 Humid"
#                 else:  # rainfall
#                     if mean_values[f] < 100:
#                         status = "🌵 Low"
#                     elif mean_values[f] <= 200:
#                         status = "🌧️ Moderate"
#                     else:
#                         status = "⛈️ High"
                
#                 st.markdown(
#                     f"<p style='text-align:center;color:#666;font-size:12px;margin-top:-10px;'>{status}</p>",
#                     unsafe_allow_html=True
#                 )
#                 st.markdown("</div>", unsafe_allow_html=True)
                
#         st.markdown("---")
#         st.subheader("📈 Distribution Analysis")
           
#         selected_param = st.selectbox(
#                "View distribution for:",
#                features_row1 + features_row2,
#                format_func=lambda x: feature_names[x]
#         )
           
#         fig = px.histogram(
#                crop_df, 
#                x=selected_param,
#                nbins=30,
#                title=f"{feature_names[selected_param]} Distribution for {selected_crop}",
#                color_discrete_sequence=['#4B371C']
#         )
#         fig.add_vline(x=mean_values[selected_param], line_dash="dash", 
#                          line_color="red", annotation_text="Mean")

#         with st.expander("🔍 **Understanding this Distribution**"):
#             st.markdown(f"""
#             **What this chart shows:**
#             This histogram displays how **{feature_names[selected_param]}** values are spread across all samples for **{selected_crop}**.
        
#             * **📊 Bars (Bins):** The height of each bar shows how many samples fall within that specific range.
#             * **🔴 Red Dashed Line:** This is the **Mean (Average)**. It shows the typical requirement for this crop.
            
#             **How to use this information:**
#             * **Narrow Cluster:** If the bars are tightly packed, the crop has very **strict** requirements. You must be precise with your inputs.
#             * **Wide Spread:** If the bars are spread out, the crop is **resilient** and can tolerate a wider range of conditions.
#             * **Gaps in Bars:** Large gaps might indicate that certain conditions are unsuitable for growth.
#             """)
        
#         st.plotly_chart(fig, use_container_width=True)

def show_trend():
    st.title("📊 Agricultural Data Trends")           
    st.markdown("Welcome to the **Crop Insight**. This platform leverages historical soil and climate data to identify optimal growing conditions and crop alternatives.")
    st.info("💡 The dashboard analyzes the relationship between soil nutrients, environmental factors, and pH levels. It is designed to support data-driven decision-making for sustainable farming.")
    df = load_data()
    if df is None:
        st.warning("⚠️ Data source file ('Crop_recommendation.csv') is missing.")
        return

    # ----------------------------
    # Features, max values, units, colors
    # ----------------------------
    features_row1 = ["N", "P", "K"]
    features_row2 = ["ph", "temperature", "humidity", "rainfall"]
    
    feature_max = {"N":150,"P":150,"K":200,"ph":14,"temperature":50,"humidity":100,"rainfall":300}
    feature_units = {"N":"","P":"","K":"","ph":"","temperature":"°C","humidity":"%","rainfall":"mm"}
    feature_names = {
        "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
        "ph": "pH Level", "temperature": "Temperature", 
        "humidity": "Humidity", "rainfall": "Rainfall"
    }
    
    colors_row1 = ["#2ca02c","#ff7f0e","#1f77b4"]
    colors_row2 = ["#9467bd","#d62728","#8c564b","#e377c2"]



    # ----------------------------
    # Data Insights Section 
    # ----------------------------
    with st.expander("📊 **About the Dataset**"):
        st.markdown("Explore the underlying relationships and requirements across all crop types.")
        
        # --- 1. FEATURE HEATMAP ---
        st.subheader("🔥 Feature Heatmap Across All Crops")
        st.markdown("""
                    **What this shows:** This heatmap compares the **average requirements** of every crop side-by-side.
                    
                    * **Vertical Axis:** The features (N, P, K, Temp, etc.).
                    * **Horizontal Axis:** The different crop types.
                    * **Colors:** 🟩 **Green** indicates high values, while 🟥 **Red** indicates low values.
                    
                    **💡 Pro-Tip:** Use this to find "extreme" crops. For example, you can quickly spot which crops need the most rainfall or the highest Nitrogen levels compared to all others.
        """)
        
        # Create pivot table
        heatmap_data = df.groupby("label")[features_row1 + features_row2].mean()
        
        fig_heat = px.imshow(
            heatmap_data.T,
            labels=dict(x="Crop", y="Feature", color="Value"),
            aspect="auto",
            # Use .T to put crops on the horizontal axis
            color_continuous_scale="RdYlGn"
        )
        fig_heat.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.divider() # Visual break between charts
    
        # --- 2. CORRELATION MATRIX ---
        st.subheader("🔗 Feature Correlations")
        st.markdown("""
                    **What this shows:** This matrix measures the **strength of the relationship** between two variables.
                    
                    * **Scale:** Values range from **+1.0 to -1.0**.
                    * **Positive (+):** As one feature increases, the other tends to increase (e.g., Phosphorus and Potassium often have high positive correlation).
                    * **Negative (-):** As one increases, the other decreases.
                    * **Near 0:** No linear relationship between the features.
                    
                    **💡 Why it matters:** If two features are almost perfectly correlated (near 1.0), they provide the same information. This helps in **feature selection** for your machine learning model.
        """)
        
        corr_matrix = df[features_row1 + features_row2].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
            
    with col1:
        st.metric("🌾 Total Crops", df["label"].nunique())
    with col2:
        st.metric("📋 Total Samples", len(df))
    with col3:
        st.metric("📊 Features", len(features_row1 + features_row2))
    with col4:
        avg_samples = len(df) / df["label"].nunique()
        st.metric("📈 Avg Samples/Crop", f"{avg_samples:.0f}")

    
    selected_crop = st.selectbox(
            "Select Crop to Analyze", 
            sorted(df["label"].unique()),
            help="Choose a crop to view its optimal growing conditions"
    )
    
    crop_df = df[df["label"] == selected_crop]

    
    crop_emojis = {
        "rice":"🌾", "wheat":"🌾", "maize":"🌽", "jute":"🌿",
        "cotton":"☁️", "coconut":"🥥", "papaya":"🍈", "orange":"🍊",
        "apple":"🍎", "muskmelon":"🍈", "watermelon":"🍉", "grapes":"🍇",
        "mango":"🥭", "banana":"🍌", "pomegranate":"💎", "lentil":"🫘",
        "blackgram":"⚫", "mungbean":"🟢", "mothbeans":"🫘", "pigeonpeas":"🫘",
        "kidneybeans":"🫘", "chickpea":"🫘", "coffee":"☕"
    }
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
    # TABS: Overview and Comparison
    # ----------------------------
    tab1, tab2  = st.tabs(["📊 Crop Overview", "🔬 Crop Comparison"])

    with tab1:
        # ----------------------------
        # CENTRAL TENDENCY SELECTION
        # ----------------------------
     
        st.markdown(f"### 📅 Growing Insights for **{selected_crop.title()}**")
        with st.container():
            st.markdown(f"""
                *These ranges show the **minimum and maximum** values observed in the dataset for **{selected_crop}**. 
                They represent the environmental tolerance limits of this crop.*
            """)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <strong>🌡️ Temperature Range:</strong> {crop_df['temperature'].min():.1f}°C - {crop_df['temperature'].max():.1f}°C<br><br>
                        <strong>💧 Humidity Range:</strong> {crop_df['humidity'].min():.1f}% - {crop_df['humidity'].max():.1f}%
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <strong>🌧️ Rainfall Range:</strong> {crop_df['rainfall'].min():.1f}mm - {crop_df['rainfall'].max():.1f}mm<br><br>
                        <strong>⚗️ pH Range:</strong> {crop_df['ph'].min():.1f} - {crop_df['ph'].max():.1f}
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        
        st.markdown("### 📊 Statistical Measure Selection")
        
        col_radio, col_expander = st.columns([1, 2])
        
        with col_radio:
            central_tendency = st.radio(
                "Choose calculation method:",
                ("Mean", "Median"),
                help="Select how to calculate central values for this crop"
            )
        
        with col_expander:
            with st.expander("ℹ️ **What's the difference?**"):
                st.markdown("""
                **Mean (Average)** 📊
                - **What it is:** Sum of all values divided by the count
                - **Best for:** Data without extreme outliers
                - **Example:** If values are [10, 12, 11, 13, 10] → Mean = 11.2
                
                **Median (Middle Value)** 📍
                - **What it is:** The middle value when data is sorted
                - **Best for:** Data with outliers or skewed distributions
                - **Example:** If values are [10, 12, 11, 13, 100] → Median = 12
                
                ---
                
                **🎯 When to use each:**
                
                | Situation | Recommended |
                |-----------|-------------|
                | Normal distribution | **Mean** |
                | Data with outliers | **Median** |
                | Extreme values present | **Median** |
                | Symmetric data | **Mean** |
                
                **💡 Agricultural Context:**
                - **Mean** shows the average requirement across all samples
                - **Median** shows the typical middle-ground requirement, less affected by unusual farming conditions or measurement errors
                
                **Example:** If 99 farms use 50mm rainfall but 1 farm recorded 300mm (possibly an error), the **median** would be 50mm while the **mean** would be skewed higher.
                """)
        
        st.markdown("---")
        
        # Calculate values based on selection
        if central_tendency == "Mean":
            calculated_values = crop_df[features_row1 + features_row2].mean().round(1)
            metric_label = "Average"
        else:  # Median
            calculated_values = crop_df[features_row1 + features_row2].median().round(1)
            metric_label = "Median"
        
        with st.expander("📅 **Growing Insights**", expanded=False):
            # This text stays inside the expander but above the columns
            st.markdown(f"""
                These ranges show the **minimum and maximum** values observed in the dataset for **{selected_crop}**. 
                They represent the tolerance limits of this crop.
            """)
            
            # Create the two columns inside the expander
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                    **🌡️ Temperature Range:** {crop_df['temperature'].min():.1f}°C - {crop_df['temperature'].max():.1f}°C
                    
                    **💧 Humidity Range:** {crop_df['humidity'].min():.1f}% - {crop_df['humidity'].max():.1f}%
                """)
                
            with col2:
                st.markdown(f"""
                    **🌧️ Rainfall Range:** {crop_df['rainfall'].min():.1f}mm - {crop_df['rainfall'].max():.1f}mm
                    
                    **⚗️ pH Range:** {crop_df['ph'].min():.1f} - {crop_df['ph'].max():.1f}
                """)
        
        # ----------------------------
        # Row 1: N, P, K
        # ----------------------------
        st.subheader(f"🌱 Soil Nutrients (NPK) - {metric_label} Values")
        
        cols1 = st.columns(len(features_row1), gap="medium")
        for i, f in enumerate(features_row1):
            with cols1[i]:
                st.markdown(
                    f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
                    unsafe_allow_html=True
                )
                fig = half_circle_gauge_card(calculated_values[f], feature_max[f], f, colors_row1[i], feature_units[f])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center;font-weight:bold;color:black;'>{calculated_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
                    unsafe_allow_html=True
                )
                
                # Add interpretation
                percentage = (calculated_values[f] / feature_max[f]) * 100
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
        st.subheader(f"🌤️ Climate & Soil Conditions - {metric_label} Values")
        
        cols2 = st.columns(len(features_row2), gap="medium")
        for i, f in enumerate(features_row2):
            with cols2[i]:
                st.markdown(
                    f"<div style='background-color:#CFE8C1; padding:15px; border-radius:18px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>",
                    unsafe_allow_html=True
                )
                fig = half_circle_gauge_card(calculated_values[f], feature_max[f], f, colors_row2[i], feature_units[f])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"<p style='text-align:center;font-weight:bold;color:black;'>{calculated_values[f]}{feature_units[f]} / {feature_max[f]}{feature_units[f]}</p>",
                    unsafe_allow_html=True
                )
                
                # Add interpretation for each parameter
                if f == "ph":
                    if calculated_values[f] < 6:
                        status = "🔴 Acidic"
                    elif calculated_values[f] <= 7.5:
                        status = "🟢 Neutral"
                    else:
                        status = "🔵 Alkaline"
                elif f == "temperature":
                    if calculated_values[f] < 20:
                        status = "❄️ Cool"
                    elif calculated_values[f] <= 30:
                        status = "🌡️ Moderate"
                    else:
                        status = "🔥 Warm"
                elif f == "humidity":
                    if calculated_values[f] < 40:
                        status = "🏜️ Dry"
                    elif calculated_values[f] <= 70:
                        status = "💧 Moderate"
                    else:
                        status = "💦 Humid"
                else:  # rainfall
                    if calculated_values[f] < 100:
                        status = "🌵 Low"
                    elif calculated_values[f] <= 200:
                        status = "🌧️ Moderate"
                    else:
                        status = "⛈️ High"
                
                st.markdown(
                    f"<p style='text-align:center;color:#666;font-size:12px;margin-top:-10px;'>{status}</p>",
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
        st.markdown("---")
        st.subheader("📈 Distribution Analysis")
           
        selected_param = st.selectbox(
               "View distribution for:",
               features_row1 + features_row2,
               format_func=lambda x: feature_names[x]
        )
        
        # Calculate both mean and median for the selected parameter
        param_mean = crop_df[selected_param].mean()
        param_median = crop_df[selected_param].median()
           
        fig = px.histogram(
               crop_df, 
               x=selected_param,
               nbins=30,
               title=f"{feature_names[selected_param]} Distribution for {selected_crop}",
               color_discrete_sequence=['#4B371C']
        )
        
        # Add both mean and median lines
        fig.add_vline(x=param_mean, line_dash="dash", 
                     line_color="red", annotation_text=f"Mean: {param_mean:.1f}")
        fig.add_vline(x=param_median, line_dash="dot", 
                     line_color="blue", annotation_text=f"Median: {param_median:.1f}")

        with st.expander("🔍 **Understanding this Distribution**"):
            st.markdown(f"""
            **What this chart shows:**
            This histogram displays how **{feature_names[selected_param]}** values are spread across all samples for **{selected_crop}**.
        
            * **📊 Bars (Bins):** The height of each bar shows how many samples fall within that specific range.
            * **🔴 Red Dashed Line:** This is the **Mean (Average)** = {param_mean:.1f}{feature_units[selected_param]}
            * **🔵 Blue Dotted Line:** This is the **Median (Middle Value)** = {param_median:.1f}{feature_units[selected_param]}
            
            **How to interpret the lines:**
            * **Lines close together:** Data is symmetrically distributed, both measures are reliable
            * **Lines far apart:** Data has outliers or is skewed, median is more reliable
            * **Mean > Median:** Data is skewed right (has high outliers)
            * **Mean < Median:** Data is skewed left (has low outliers)
            
            **How to use this information:**
            * **Narrow Cluster:** If the bars are tightly packed, the crop has very **strict** requirements. You must be precise with your inputs.
            * **Wide Spread:** If the bars are spread out, the crop is **resilient** and can tolerate a wider range of conditions.
            * **Gaps in Bars:** Large gaps might indicate that certain conditions are unsuitable for growth.
            """)
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(f"### Compare **{selected_crop.upper()}** {emoji} with Other Crops")
        st.info("Select up to 3 crops to compare their optimal growing conditions side-by-side.")
        
        compare_crops = st.multiselect(
            "🌾 Select crops to compare with " + selected_crop,
            [c for c in sorted(df["label"].unique()) if c != selected_crop],
            max_selections=3,
            help="Choose crops you want to compare"
        )
        
        if compare_crops:
            # Build comparison data
            comparison_data = {"Crop": [selected_crop] + compare_crops}
            
            for feature in features_row1 + features_row2:
                comparison_data[feature_names[feature]] = [mean_values[feature]]
                for crop in compare_crops:
                    crop_mean = df[df["label"] == crop][feature].mean()
                    comparison_data[feature_names[feature]].append(round(crop_mean, 1))
            
            comp_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.markdown("#### 📋 Comparison Table in Farm Environment Profile")
            st.dataframe(
                comp_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Crop": st.column_config.TextColumn("🌱 Crop", width="medium"),
                }
            )
            
            # Visualization section
            st.markdown("---")
            st.markdown("#### 🌡️ Farm Environment Visual Comparison")
            
            col_viz1, col_viz2 = st.columns([1, 2])
            
            with col_viz1:
                selected_feature = st.selectbox(
                    "Select parameter to visualize",
                    [feature_names[f] for f in features_row1 + features_row2],
                    help="Choose which parameter to compare visually"
                )
            
            with col_viz2:
                chart_type = st.radio(
                    "Chart Type",
                    ["Bar Chart", "Radar Chart"],
                    horizontal=True
                )
            
            if chart_type == "Bar Chart":
                fig = px.bar(
                    comp_df, 
                    x="Crop", 
                    y=selected_feature,
                    title=f"{selected_feature} Comparison Across Crops",
                    color="Crop",
                    text=selected_feature,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(
                    showlegend=False, 
                    height=450,
                    xaxis_title="Crop",
                    yaxis_title=selected_feature
                )
                with st.expander("📊 **Understanding the Bar Chart**"):
                    st.markdown("""
                        * This chart compares **Single Parameter Focus** requirements across the selected crops.
                        * Use this to see exact numerical differences for a specific metric. 
                        * It is the best way to determine which crop is the "most" or "least" demanding for a single nutrient.
                    """)
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Radar Chart
                # Normalize values for radar chart
                categories = [feature_names[f] for f in features_row1 + features_row2]
                
                fig = go.Figure()
                
                # Add trace for each crop
                for idx, crop in enumerate([selected_crop] + compare_crops):
                    crop_data = comp_df[comp_df["Crop"] == crop]
                    values = []
                    for feature in features_row1 + features_row2:
                        val = crop_data[feature_names[feature]].values[0]
                        max_val = feature_max[feature]
                        normalized = (val / max_val)
                        values.append(normalized)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=crop,
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickformat='.0%'
                        )
                    ),
                    showlegend=True,
                    title="Normalized Multi-Parameter Comparison",
                    height=500
                )
                with st.expander("📊 **Understanding the Rader Chart**"):
                    st.markdown("""
                            * This chart **normalizes** all values (0% to 100%) so you can compare temperature, pH, and nutrients on the same scale.
                            * **What to look for:** If the shapes of two crops overlap significantly, they share a similar "biological fingerprint" and likely grow well in the same regions.
                    """)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Key differences section
            st.markdown("---")
            st.markdown("#### 🔍 Key Differences")
            with st.expander("ℹ️ **How to interpret these differences**"):
                st.markdown(f"""
                This section highlights where the comparison crops **deviate significantly** from your primary choice, **{selected_crop.upper()}**.
                
                * **The 20% Rule:** We only display a difference if it is greater than **20%**. This filters out minor variations and focuses on the factors that truly change how you manage the field.
                * **🔺 (Red Up):** Indicates the comparison crop requires **significantly more** of this resource. You may need higher fertilizer input or more frequent irrigation.
                * **🔻 (Red Down):** Indicates the comparison crop requires **significantly less**. This could represent a more cost-effective or "hardy" alternative for your specific soil.
                
                **Why this matters:** If you see a **+80% Nitrogen** difference, switching to that crop would require a complete overhaul of your fertilization schedule.
                """)
                
                # Optional: Display the formula for technical clarity
                st.latex(r"Percentage\ Change = \frac{Comparison\ Mean - Selected\ Mean}{Selected\ Mean} \times 100")
                
            diff_cols = st.columns(len(compare_crops))
            for idx, crop in enumerate(compare_crops):
                with diff_cols[idx]:
                    crop_emoji = crop_emojis.get(crop.lower(), "🌱")
                    st.markdown(f"**{crop_emoji} {crop.upper()} vs {selected_crop.upper()}**")
                    
                    crop_data = df[df["label"] == crop][features_row1 + features_row2].mean()
                    
                    differences = []
                    for feature in features_row1 + features_row2:
                        diff = crop_data[feature] - mean_values[feature]
                        pct_diff = (diff / mean_values[feature] * 100) if mean_values[feature] != 0 else 0
                        
                        if abs(pct_diff) > 20:  # Only show significant differences
                            if diff > 0:
                                differences.append(f"🔺 {feature_names[feature]}: +{pct_diff:.0f}%")
                            else:
                                differences.append(f"🔻 {feature_names[feature]}: {pct_diff:.0f}%")
                    
                    if differences:
                        for diff in differences[:3]:  # Show top 3 differences
                            st.caption(diff)
                    else:
                        st.caption("Similar conditions")
        
        else:
            st.warning("👆 Select at least one crop to start comparing!")
            
        if compare_crops:
            st.markdown("#### 🎯 Similarity Analysis")
       
        for crop in compare_crops:
           # Calculate similarity score
           crop_data = df[df["label"] == crop][features_row1 + features_row2].mean()
           differences = []
           for feature in features_row1 + features_row2:
               diff = abs(crop_data[feature] - mean_values[feature])
               max_val = feature_max[feature]
               norm_diff = (diff / max_val)
               differences.append(norm_diff)
           
           similarity = (1 - sum(differences) / len(differences)) * 100
           
           crop_emoji = crop_emojis.get(crop.lower(), "🌱")
           st.progress(similarity / 100)
           st.caption(f"{crop_emoji} **{crop}** is {similarity:.1f}% similar to **{selected_crop}**")
                           


def show_prediction():
    st.title("🌱 Intelligent Crop Recommendation")
    
    stage1_model, le = load_stage1()
    stage2_model = load_stage2()
    
    if stage1_model is None or le is None:
        st.error("🚨 Stage 1 model files missing.")
        return
    if stage2_model is None:
        st.warning("⚠️ Stage 2 model not loaded. You can still get crop recommendation.")
    
    # m1, m2 = st.columns(2)
    # with m1:
    #     st.metric(label="Stage 1: Crop Recommendation", value="99.5%", delta="Accuracy")
    # with m2:
    #     st.metric(label="Stage 2: Yield Prediction", value="0.723", delta="R² Score")


    with st.expander("📖 **How to Use This System: A 2-Step Journey**"):
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Stage 1: Crop Recommendation", value="99.5%", delta="Accuracy")
        with m2:
            st.metric(label="Stage 2: Yield Prediction", value="0.723", delta="R² Score")
            
        st.markdown("### 🧬 Step 1: The Recommendation (22 Crops)")
        st.write("""
        Our **Stage 1 Intelligence** analyzes your soil and climate to pick the best crop for your land. 
        It can identify the perfect environment for **22 different crops**, including fruits, grains, 
        and legumes.
        """)
        
        st.divider()
        
        st.markdown("### 📈 Step 2: Advanced Yield Prediction (The Big Three)")
        st.write("""
        If the AI recommends **Rice, Maize, or Cotton**, you can unlock **Stage 2**. 
        
        Because these three crops are vital for global industry, we use a specialized **XGBoost Regressor** to predict exactly how many tons per hectare ($t/ha$) you will harvest based on your 
        farming inputs (Irrigation, Fertilizer, and Pesticides).
        """)
        

    
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
        # Show Stage 1 Model Performance
        with st.expander("🧠 **Stage 1 Model Performance** "):
            st.markdown("**Stage 1: Classification**")
            st.markdown("""
            - **Model:** Random Forest Classifier
            - **Accuracy:** 0.995 
            - **Insights:** This model excels at identifying the biological 'sweet spot' for 22 different crop varieties based on soil and climate input.
            """)
            st.caption("⚠️ Note: Predictions are based on historical data patterns and should be used as a decision-support tool, not a guarantee of harvest.")
        
        # Stage 1: Crop Recommendation
        input_stage1 = np.array([[N, P, K, temp, hum, ph, rain]])
        crop_encoded = stage1_model.predict(input_stage1)[0]
        crop_name = le.inverse_transform([crop_encoded])[0]
        
        st.session_state.stage1_crop = crop_name
        st.session_state.stage1_input = {"N": N, "P": P, "K": K, "temperature": temp, "humidity": hum, "ph": ph, "rainfall": rain}
        st.session_state.submitted = True
        
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

        st.balloons()
        st.subheader("📊 Suggestion Improvement")
        df = load_data()
        crop_data = df[df["label"] == crop_name]
        
        # Calculate mode (most frequently used values) for each numeric parameter
        crop_frequently_used = {}
        for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
            mode_values = crop_data[col].mode()
            crop_frequently_used[col] = mode_values[0] if len(mode_values) > 0 else crop_data[col].mean()
        
        # Calculate indices
        thi = temp - (0.55 - 0.0055 * hum) * (temp - 14.4)
        sfi = (N + P + K) / 3
        
        # Store in session state for PDF generation
        st.session_state.thi = thi
        st.session_state.sfi = sfi
        
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
        

  
        st.markdown("##### 🎯 Frequently Used Parameter Match Score")
        params = {
            "Nitrogen (N)": (N, crop_frequently_used["N"], 150),
            "Phosphorus (P)": (P, crop_frequently_used["P"], 150),
            "Potassium (K)": (K, crop_frequently_used["K"], 150),
            "pH Level": (ph, crop_frequently_used["ph"], 14),
            "Temperature": (temp, crop_frequently_used["temperature"], 50),
            "Humidity": (hum, crop_frequently_used["humidity"], 100),
            "Rainfall": (rain, crop_frequently_used["rainfall"], 300)
        }
        
        # Create two columns for better layout
        col_left, col_right = st.columns(2)
        
        param_items = list(params.items())
        mid_point = len(param_items) // 2 + len(param_items) % 2
        
        with col_left:
            for param_name, (user_val, freq_val, max_val) in param_items[:mid_point]:
                match_pct = 100 - abs((user_val - freq_val) / freq_val * 100) if freq_val != 0 else 100
                match_pct = max(0, min(100, match_pct))
                
                if match_pct >= 90:
                    color = "🟢"
                elif match_pct >= 70:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"**{color} {param_name}**")
                st.progress(match_pct / 100)
                st.caption(f"Your: {user_val:.1f} | Frequently Used: {freq_val:.1f} | Match: {match_pct:.0f}%")
                st.markdown("<br>", unsafe_allow_html=True)
        
        with col_right:
            for param_name, (user_val, freq_val, max_val) in param_items[mid_point:]:
                match_pct = 100 - abs((user_val - freq_val) / freq_val * 100) if freq_val != 0 else 100
                match_pct = max(0, min(100, match_pct))
                
                if match_pct >= 90:
                    color = "🟢"
                elif match_pct >= 70:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"**{color} {param_name}**")
                st.progress(match_pct / 100)
                st.caption(f"Your: {user_val:.1f} | Frequently Used: {freq_val:.1f} | Match: {match_pct:.0f}%")
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Overall Match Summary
        overall_matches = []
        param_matches_dict = {}  # For PDF generation
        
        for param_name, (user_val, freq_val, max_val) in params.items():
            match_pct = 100 - abs((user_val - freq_val) / freq_val * 100) if freq_val != 0 else 100
            match_pct = max(0, min(100, match_pct))
            overall_matches.append(match_pct)
            param_matches_dict[param_name] = (user_val, freq_val, match_pct)
        
        avg_match = sum(overall_matches) / len(overall_matches)
        
        # Store in session state for PDF generation
        st.session_state.overall_match = avg_match
        st.session_state.param_matches = param_matches_dict
        
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
                    {'Your conditions are very close to frequently used values!' if avg_match >= 90 else
                     'Your conditions are good for this crop.' if avg_match >= 75 else
                     'Consider adjusting some parameters for better yield.' if avg_match >= 60 else
                     'Several parameters need adjustment for optimal growth.'}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # ========================================
        # PDF DOWNLOAD BUTTON - STAGE 1
        # ========================================
        st.markdown("---")
        
        try:
            pdf_filename = f"crop_report_{crop_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            # Generate PDF in memory (returns BytesIO object)
            pdf_buffer = create_crop_prediction_pdf(
                N=N, P=P, K=K, ph=ph,
                temperature=temp, humidity=hum, rainfall=rain,
                recommended_crop=crop_name,
                thi=thi, sfi=sfi,
                parameter_matches=param_matches_dict,
                overall_match=avg_match
            )
            
            st.download_button(
                label="📄 Download Stage 1 Report (PDF)",
                data=pdf_buffer,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
                        
    # ========================================
    # Stage 2: Yield Prediction 
    # ========================================
    if st.session_state.get('submitted', False):
        crop_name = st.session_state.stage1_crop
        allowed_crops = ["rice", "maize", "cotton"]
        
        if crop_name.strip().lower() in allowed_crops and stage2_model is not None:
            st.markdown("---")
            
            # Initialize session state for stage 2 choice if not exists
            if 'stage2_choice' not in st.session_state:
                st.session_state.stage2_choice = "No"
            
            # Simple "Do you want to predict yield?" prompt
            st.markdown(f"""
                <div style='background-color:#f9faf0; padding:20px; border-radius:10px; border-left:5px solid #4caf50; margin: 20px 0;'>
                    <h3 style='margin:0; color:#2e7d32;'>🌾 Yield Prediction Available</h3>
                    <p style='margin:5px 0 0 0; color:#555;'>Would you like to predict the yield for <strong>{crop_name}</strong>?</p>
                </div>
            """, unsafe_allow_html=True)
            
            choice = st.radio(
                "Do you want to predict yield for this crop?",
                ("No", "Yes"),
                key="stage2_choice"
            )
            
            if st.session_state.stage2_choice == "Yes":
                # Show Stage 2 Model Performance
                with st.expander("🧠 **Stage 2 Model Performance** "):
                    st.markdown("**Stage 2: Regression**")
                    st.markdown(f"""
                    - **Model:** XGBoost Regressor
                    - **R-Squared:** 0.723 
                    - **Insights:** Predicting yield is complex due to external variables. An $R^2$ of 0.723 indicates the model explains 72% of the variance in crop tonnage.
                    """)
                    st.caption("⚠️ Note: Predictions are based on historical data patterns and should be used as a decision-support tool, not a guarantee of harvest.")
                
                with st.form("stage2_form"):
                    st.subheader("📋 Additional Farm Parameters")
                    st.caption("💡 Reused from Stage 1: N={}, P={}, K={}, pH={}, Temp={}°C, Humidity={}%, Rainfall={}mm".format(
                        st.session_state.stage1_input["N"],
                        st.session_state.stage1_input["P"],
                        st.session_state.stage1_input["K"],
                        st.session_state.stage1_input["ph"],
                        st.session_state.stage1_input["temperature"],
                        st.session_state.stage1_input["humidity"],
                        st.session_state.stage1_input["rainfall"]
                    ))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### **Soil & Environmental**")
                        
                        soil_moisture = st.slider(
                            "Soil Moisture (%)", 
                            0, 100, 50,
                            help="Current soil moisture content percentage"
                        )
                        
                        soil_type = st.selectbox(
                            "Soil Type",
                            ["Loamy", "Sandy", "Silt", "Clay"],
                            help="Primary soil composition type"
                        )
                        
                        sunlight_hours = st.number_input(
                            "Sunlight Hours (hours/day)", 
                            0.0, 24.0, 8.0, 0.5,
                            help="Average daily sunlight exposure"
                        )
                    
                    with col2:
                        st.markdown("##### **Farm Management**")
                        
                        irrigation_type = st.selectbox(
                            "Irrigation Type",
                            ["Drip", "Canal", "Rainfed", "Sprinkler"],
                            help="Primary irrigation method used"
                        )
                        
                        fertilizer_used = st.number_input(
                            "Fertilizer Used (kg/hectare)",
                            0.0, 500.0, 100.0, 10.0,
                            help="Amount of fertilizer applied"
                        )
                        
                        pesticide_used = st.number_input(
                            "Pesticide Used (kg/hectare)",
                            0.0, 50.0, 5.0, 0.5,
                            help="Amount of pesticide applied"
                        )
                    st.markdown("---")
                    submit_stage2 = st.form_submit_button("✨  Predict Yield")
                
                if submit_stage2:
                    # Combine Stage 1 and Stage 2 inputs
                    stage2_input = {
                        # From Stage 1 (lowercase)
                        "N": st.session_state.stage1_input["N"],
                        "P": st.session_state.stage1_input["P"],
                        "K": st.session_state.stage1_input["K"],
                        "ph": st.session_state.stage1_input["ph"],
                        "temperature": st.session_state.stage1_input["temperature"],
                        "humidity": st.session_state.stage1_input["humidity"],
                        "rainfall": st.session_state.stage1_input["rainfall"],
                        
                        # New Stage 2 inputs
                        "Soil_Moisture": soil_moisture,
                        "Sunlight_Hours": sunlight_hours,
                        "Fertilizer_Used": fertilizer_used,
                        "Pesticide_Used": pesticide_used,
                        "Soil_Type": soil_type,
                        "Irrigation_Type": irrigation_type,
                        "Crop_Type": crop_name
                    }
                    
                    # Convert to DataFrame
                    stage2_input_df = pd.DataFrame([stage2_input])
                    
                    # Make prediction
                    try:
                        yield_pred = stage2_model.predict(stage2_input_df)[0]
                        
                        # Crop-specific remarks
                        crop_remarks = {
                                "rice": "Rice thrives with high nitrogen and consistent water management. Your predicted yield reflects optimal flooded conditions and balanced nutrients.",
                                "maize": "Maize requires balanced NPK nutrients and adequate sunlight. Ensure proper spacing and weed control for maximum yield.",
                                "cotton": "Cotton needs sufficient potassium for fiber quality. Monitor for pests and ensure adequate irrigation during flowering stage."
                        }
                        remark = crop_remarks.get(crop_name.lower(), "Ensure proper soil fertility and climate management for best yield.")

                        # Display result
                        st.markdown(f"""
                        <div class="prediction-card" style="background: linear-gradient(135deg, #ffffff 0%, #fafcf7 100%); color: white;">
                            <h2 style="color: white;">🎯 Predicted Yield: <strong>{yield_pred:.2f} tons/hectare</strong></h2>
                            <p style="color: black; opacity: 0.95;">{remark}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        
                        # ========================================
                        # PDF DOWNLOAD BUTTON - STAGE 2 (Full Report)
                        # ========================================
                        st.markdown("---")
                        
                        try:
                            pdf_filename_full = f"crop_report_full_{crop_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            
                            # Generate PDF in memory (returns BytesIO object)
                            pdf_buffer_full = create_crop_prediction_pdf(
                                # Stage 1 inputs
                                N=st.session_state.stage1_input["N"],
                                P=st.session_state.stage1_input["P"],
                                K=st.session_state.stage1_input["K"],
                                ph=st.session_state.stage1_input["ph"],
                                temperature=st.session_state.stage1_input["temperature"],
                                humidity=st.session_state.stage1_input["humidity"],
                                rainfall=st.session_state.stage1_input["rainfall"],
                                # Stage 1 results
                                recommended_crop=crop_name,
                                thi=st.session_state.thi,
                                sfi=st.session_state.sfi,
                                parameter_matches=st.session_state.param_matches,
                                overall_match=st.session_state.overall_match,
                                # Stage 2 inputs
                                soil_moisture=soil_moisture,
                                soil_type=soil_type,
                                sunlight_hours=sunlight_hours,
                                irrigation_type=irrigation_type,
                                fertilizer_used=fertilizer_used,
                                pesticide_used=pesticide_used,
                                # Stage 2 results
                                predicted_yield=yield_pred
                            )
                            
                            st.download_button(
                                label="📄 Download Complete Report (PDF)",
                                data=pdf_buffer_full,
                                file_name=pdf_filename_full,
                                mime="application/pdf",
                                use_container_width=True,
                                type="primary"
                            )
                        except Exception as e:
                            st.error(f"Error generating full PDF report: {str(e)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error predicting yield: {str(e)}")
                        with st.expander("Debug Info"):
                            st.write("Input data:")
                            st.json(stage2_input)
        
        elif crop_name.strip().lower() not in allowed_crops:
            pass


                          
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
