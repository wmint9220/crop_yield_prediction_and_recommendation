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

CARD_COLORS = {
    "N":           ("#166534", "#dcfce7"),   # dark green  / light green bg
    "P":           ("#92400e", "#fef3c7"),   # dark amber  / light amber bg
    "K":           ("#1e3a8a", "#dbeafe"),   # dark blue   / light blue bg
    "ph":          ("#581c87", "#f3e8ff"),   # dark purple / light purple bg
    "temperature": ("#9f1239", "#ffe4e6"),   # dark rose   / light rose bg
    "humidity":    ("#134e4a", "#ccfbf1"),   # dark teal   / light teal bg
    "rainfall":    ("#312e81", "#e0e7ff"),   # dark indigo / light indigo bg
}

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


def get_npk_card(feature, value):
    """Return (status_label, text_color, bg_color, role, meaning, action) for N/P/K."""
    text_color, bg_color = CARD_COLORS[feature]

    if feature == "N":
        role = "The Growth Driver — fuels leafy, green, vegetative growth."
        if value < 45:
            return ("🔵 Low", text_color, bg_color, role,
                    "Very little Nitrogen needed. Over-fertilizing can burn the plant or cause excessive leaf growth at the expense of fruit/grain.",
                    "✅ Use minimal or no N-fertilizer.")
        elif value <= 90:
            return ("🟢 Moderate", text_color, bg_color, role,
                    "Balanced Nitrogen need. Standard fertilization schedules apply.",
                    "✅ Monitor leaf color — yellowing (chlorosis) signals deficiency.")
        else:
            return ("🟠 High", text_color, bg_color, role,
                    "Heavy feeder. Multiple N-fertilizer applications (split dosing) needed throughout the season.",
                    "✅ Use split-dose N-fertilizer at key growth stages.")

    elif feature == "P":
        role = "The Root & Flower Builder — drives root development, flowering, and seed formation."
        if value < 45:
            return ("🔵 Low", text_color, bg_color, role,
                    "Minimal phosphorus needed. Common in root/tuber crops. Excess P can lock out Zinc and Iron.",
                    "✅ Avoid over-application; background soil P is sufficient.")
        elif value <= 90:
            return ("🟢 Moderate", text_color, bg_color, role,
                    "Standard P requirements.",
                    "✅ Apply basal phosphate fertilizer before sowing or planting.")
        else:
            return ("🟠 High", text_color, bg_color, role,
                    "High demand, often seen in fruiting or flowering crops.",
                    "✅ Incorporate bone meal, superphosphate, or DAP at planting time.")

    else:  # K
        role = "The Quality & Stress Shield — improves fruit quality, disease resistance, and drought tolerance."
        if value < 60:
            return ("🔵 Low", text_color, bg_color, role,
                    "Low K demand. Background soil potassium levels are sufficient for this crop.",
                    "✅ No additional K application needed under normal conditions.")
        elif value <= 120:
            return ("🟢 Moderate", text_color, bg_color, role,
                    "Moderate K need.",
                    "✅ A single basal application of MOP or SOP at planting is typically sufficient.")
        else:
            return ("🟠 High", text_color, bg_color, role,
                    "High demand, especially in fruit crops. Impacts yield and sweetness.",
                    "✅ Regular K top-dressing throughout the growing season.")


def get_climate_card(feature, value):
    """Return (status_label, text_color, bg_color, role, meaning, action) for climate features."""
    text_color, bg_color = CARD_COLORS[feature]

    if feature == "ph":
        role = "Soil Acidity / Alkalinity"
        if value < 6:
            return ("🔴 Acidic", text_color, bg_color, role,
                    f"pH {value} — Acidic soil. Aluminum and Manganese may reach toxic levels. Phosphorus becomes unavailable to plants.",
                    "✅ Apply agricultural lime (calcium carbonate) to raise pH.")
        elif value <= 7.5:
            return ("🟢 Neutral", text_color, bg_color, role,
                    f"pH {value} — The sweet spot for most crops. Nutrient availability is at its peak and microbial activity is healthy.",
                    "✅ Maintain current soil management practices.")
        else:
            return ("🔵 Alkaline", text_color, bg_color, role,
                    f"pH {value} — Alkaline soil. Iron, Zinc, and Manganese become less available, risking deficiency symptoms.",
                    "✅ Apply sulfur or acidifying fertilizers (e.g. ammonium sulfate) to lower pH.")

    elif feature == "temperature":
        role = "Metabolic Activity Rate"
        if value < 20:
            return ("❄️ Cool", text_color, bg_color, role,
                    f"{value}°C — This crop prefers cool conditions. Warm-season crops will fail or produce poorly at this temperature.",
                    "✅ Ideal for highland or temperate-zone cultivation.")
        elif value <= 30:
            return ("🌡️ Moderate", text_color, bg_color, role,
                    f"{value}°C — Optimal range for the majority of tropical and subtropical crops.",
                    "✅ No special temperature management required.")
        else:
            return ("🔥 Warm", text_color, bg_color, role,
                    f"{value}°C — This crop thrives in heat. Cool-season crops planted here will bolt, wilt, or die.",
                    "✅ Ensure adequate water availability to offset high evapotranspiration.")

    elif feature == "humidity":
        role = "Atmospheric Moisture"
        if value < 40:
            return ("🏜️ Dry", text_color, bg_color, role,
                    f"{value}% — Low atmospheric moisture. The crop is drought-adapted. High-humidity crops planted here will suffer water stress.",
                    "✅ Supplemental irrigation may be necessary during dry periods.")
        elif value <= 70:
            return ("💧 Moderate", text_color, bg_color, role,
                    f"{value}% — A comfortable moisture level for most crops. Disease risk is manageable.",
                    "✅ Standard fungicide and pest schedules are sufficient.")
        else:
            return ("💦 Humid", text_color, bg_color, role,
                    f"{value}% — High moisture environment adapted to humid tropics. Fungal diseases (blight, mildew) are a major risk.",
                    "✅ Monitor closely and apply preventive fungicides regularly.")

    else:  # rainfall
        role = "Total Water Input"
        if value < 100:
            return ("🌵 Low", text_color, bg_color, role,
                    f"{value}mm — Drought-tolerant crop or short growing season. Low annual rainfall regions can support this crop.",
                    "✅ Minimal or no irrigation required.")
        elif value <= 200:
            return ("🌧️ Moderate", text_color, bg_color, role,
                    f"{value}mm — Standard rainfall requirement suitable for semi-arid to sub-humid regions.",
                    "✅ Supplemental irrigation during dry spells may improve yield.")
        else:
            return ("⛈️ High", text_color, bg_color, role,
                    f"{value}mm — This crop demands significant water. Waterlogging is a risk without proper drainage.",
                    "✅ Ensure good drainage; in low-rainfall areas heavy irrigation infrastructure is needed.")


def create_crop_prediction_pdf(
    N, P, K, ph, temperature, humidity, rainfall,
    recommended_crop, thi, sfi, parameter_matches, overall_match,
    soil_moisture=None, soil_type=None, sunlight_hours=None,
    irrigation_type=None, fertilizer_used=None, pesticide_used=None,
    predicted_yield=None
):
    """Generate PDF report in memory and return BytesIO object"""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        leftMargin=0.75*inch, rightMargin=0.75*inch
    )

    story = []
    styles = getSampleStyleSheet()
    
    # ------------------------------------------------------------------
    # Custom styles
    # ------------------------------------------------------------------
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=26, textColor=HexColor('#2c5282'),
        spaceAfter=10, alignment=TA_CENTER, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=12, textColor=HexColor('#666666'),
        alignment=TA_CENTER, spaceAfter=30
    )
    section_header = ParagraphStyle(
        'SectionHeader', parent=styles['Heading2'],
        fontSize=18, textColor=HexColor('#2c5282'),
        spaceAfter=15, spaceBefore=20, fontName='Helvetica-Bold'
    )
    subsection_header = ParagraphStyle(
        'SubsectionHeader', parent=styles['Heading3'],
        fontSize=14, textColor=HexColor('#4a5568'),
        spaceAfter=10, spaceBefore=15, fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'],
        fontSize=11, leading=16, spaceAfter=12, alignment=TA_JUSTIFY
    )
    center_style = ParagraphStyle(
        'Center', parent=styles['Normal'],
        fontSize=11, leading=16, spaceAfter=8, alignment=TA_CENTER
    )
    note_style = ParagraphStyle(
        'Note', parent=styles['Normal'],
        fontSize=9, textColor=HexColor('#666666'),
        alignment=TA_CENTER, spaceAfter=6
    )
    info_box_style = ParagraphStyle(
        'InfoBox', parent=styles['Normal'],
        fontSize=10, textColor=HexColor('#1a365d'),
        leading=15, spaceAfter=4
    )

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    story.append(Paragraph("Crop Insight Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))

    # ------------------------------------------------------------------
    # SECTION 1 — Input Parameters
    # ------------------------------------------------------------------
    story.append(Paragraph("1. Input Parameters", section_header))
    input_data = [
        ['Parameter', 'Your Value', 'Unit'],
        ['Nitrogen (N)',    f'{N}',              'ppm'],
        ['Phosphorus (P)', f'{P}',              'ppm'],
        ['Potassium (K)',  f'{K}',              'ppm'],
        ['Soil pH Level',  f'{ph:.1f}',         'pH'],
        ['Temperature',    f'{temperature:.1f}', '°C'],
        ['Humidity',       f'{humidity}',        '%'],
        ['Rainfall',       f'{rainfall:.1f}',    'mm'],
    ]
    input_table = Table(input_data, colWidths=[2.8*inch, 1.8*inch, 1.4*inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#2c5282')),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 12),
        ('BOTTOMPADDING', (0, 0), (-1,  0), 12),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND',    (0, 1), (-1, -1), HexColor('#f7fafc')),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, HexColor('#f7fafc')]),
        ('GRID',          (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ('FONTSIZE',      (0, 1), (-1, -1), 11),
        ('TOPPADDING',    (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 0.3*inch))

    # ------------------------------------------------------------------
    # SECTION 2 — Recommended Crop
    # ------------------------------------------------------------------
    story.append(Paragraph("2. Recommended Crop", section_header))
    crop_emojis = {
        "rice":"Rice", "wheat":"Wheat", "maize":"Maize", "jute":"Jute",
        "cotton":"Cotton", "coconut":"Coconut", "papaya":"Papaya",
        "orange":"Orange", "apple":"Apple", "muskmelon":"Muskmelon",
        "watermelon":"Watermelon", "grapes":"Grapes", "mango":"Mango",
        "banana":"Banana", "pomegranate":"Pomegranate", "lentil":"Lentil",
        "blackgram":"Blackgram", "mungbean":"Mungbean", "mothbeans":"Mothbeans",
        "pigeonpeas":"Pigeonpeas", "kidneybeans":"Kidneybeans",
        "chickpea":"Chickpea", "coffee":"Coffee",
    }
    story.append(Paragraph(
        f"<b>{recommended_crop.upper()}</b>",
        ParagraphStyle('CropName', parent=body_style, fontSize=20,
                       textColor=HexColor('#2c5282'), alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"Based on your soil and environmental parameters, <b>{recommended_crop}</b> "
        f"is identified as the most suitable crop. This recommendation takes into "
        f"account the soil pH, NPK balance, temperature, humidity, and rainfall "
        f"required for this species to thrive.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # ------------------------------------------------------------------
    # SECTION 3 — Environmental Indices
    # ------------------------------------------------------------------
    story.append(Paragraph("3. Environmental Indices", section_header))

    if thi < 15:
        thi_status = "Cold Stress — May slow crop growth"
    elif thi < 22:
        thi_status = "Optimal — Ideal growing conditions"
    elif thi < 28:
        thi_status = "Warm — Monitor water needs"
    else:
        thi_status = "Heat Stress — Risk to crop health"

    if sfi < 30:
        sfi_status = "Low — Needs fertilization"
    elif sfi < 60:
        sfi_status = "Moderate — Adequate nutrients"
    elif sfi < 90:
        sfi_status = "Good — Well-balanced soil"
    else:
        sfi_status = "Excellent — Nutrient-rich soil"

    indices_data = [
        ['Index', 'Value', 'Status'],
        ['Temperature-Humidity Index (THI)', f'{thi:.1f}', thi_status],
        ['Soil Fertility Index (SFI)',        f'{sfi:.1f}', sfi_status],
    ]
    indices_table = Table(indices_data, colWidths=[2.5*inch, 1.0*inch, 2.5*inch])
    indices_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#4a5568')),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 11),
        ('BOTTOMPADDING', (0, 0), (-1,  0), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN',         (1, 0), (1,  -1), 'CENTER'),
        ('BACKGROUND',    (0, 1), (-1, -1), colors.white),
        ('GRID',          (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ('FONTSIZE',      (0, 1), (-1, -1), 10),
        ('TOPPADDING',    (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(indices_table)
    story.append(Spacer(1, 0.3*inch))

    # ------------------------------------------------------------------
    # SECTION 4 — Parameter Match Analysis
    # ------------------------------------------------------------------
    story.append(Paragraph("4. Frequency Used Parameter Match Analysis", section_header))

    # Explain the scoring method used
    story.append(Paragraph(
        "How the Match % is calculated:",
        ParagraphStyle('SmallBold', parent=styles['Normal'],
                       fontSize=10, fontName='Helvetica-Bold',
                       textColor=HexColor('#4a5568'), spaceAfter=4)
    ))
    story.append(Paragraph(
        "Match % = (1 - |Your Value - Typical Value| / Parameter Range) x 100. "
        "The difference is divided by the full parameter scale range (e.g. N: 0-150, pH: 0-14) "
        "rather than by the typical value itself, so scores remain fair and comparable "
        "across all parameters regardless of their magnitude.",
        ParagraphStyle('FormulaNote', parent=styles['Normal'],
                       fontSize=9, textColor=HexColor('#555555'),
                       leading=13, spaceAfter=10)
    ))

    # Score legend
    legend_data = [
        ['Score',    'Meaning'],
        ['>= 80%',  'Very close to the typical value — great fit for this crop'],
        ['50-79%',  'Moderate deviation — crop may still grow, consider adjusting'],
        ['< 50%',   'Large deviation — this parameter may limit crop growth'],
    ]
    legend_table = Table(legend_data, colWidths=[1.1*inch, 4.9*inch])
    legend_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#4a5568')),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 10),
        ('BOTTOMPADDING', (0, 0), (-1,  0), 8),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('TOPPADDING',    (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [HexColor('#f0fff4'), HexColor('#fffff0'), HexColor('#fff5f5')]),
        ('GRID',          (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 0.2*inch))

    # Match scores table
    match_data = [['Parameter', 'Your Value', 'Typical Value', 'Match %', 'Diff']]
    PARAM_RANGES = {
        "Nitrogen (N)":   150,
        "Phosphorus (P)": 150,
        "Potassium (K)":  150,
        "pH Level":       14,
        "Temperature":    50,
        "Humidity":       100,
        "Rainfall":       300,
    }
    for param_name, (user_val, opt_val, match_pct) in parameter_matches.items():
        diff = user_val - opt_val
        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        # Colour-code match % text
        if match_pct >= 80:
            pct_label = f"{match_pct:.0f}%  Good"
        elif match_pct >= 50:
            pct_label = f"{match_pct:.0f}%  Fair"
        else:
            pct_label = f"{match_pct:.0f}%  Low"
        match_data.append([
            param_name,
            f'{user_val:.1f}',
            f'{opt_val:.1f}',
            f'{match_pct:.0f}%',
            diff_str,
        ])

    match_table = Table(match_data, colWidths=[2.0*inch, 1.1*inch, 1.2*inch, 0.9*inch, 0.8*inch])
    match_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#4a5568')),
        ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 11),
        ('BOTTOMPADDING', (0, 0), (-1,  0), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, HexColor('#f7fafc')]),
        ('GRID',          (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ('FONTSIZE',      (0, 1), (-1, -1), 10),
        ('TOPPADDING',    (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
    ]))
    story.append(match_table)
    story.append(Spacer(1, 0.2*inch))

    # Overall match banner
    if overall_match >= 90:
        match_summary = "Excellent Match — Your conditions are very close to the typical values."
        match_color   = HexColor('#2ecc71')
    elif overall_match >= 75:
        match_summary = "Good Match — Your conditions suit this crop well."
        match_color   = HexColor('#27ae60')
    elif overall_match >= 60:
        match_summary = "Fair Match — Consider adjusting some parameters for better yield."
        match_color   = HexColor('#f39c12')
    else:
        match_summary = "Needs Adjustment — Several parameters deviate significantly."
        match_color   = HexColor('#e74c3c')

    story.append(Paragraph(
        f"<b>Overall Match Score: {overall_match:.1f}%</b>",
        ParagraphStyle('MatchScore', parent=body_style, fontSize=14,
                       textColor=match_color, alignment=TA_CENTER)
    ))
    story.append(Paragraph(
        match_summary,
        ParagraphStyle('MatchText', parent=body_style, fontSize=11, alignment=TA_CENTER)
    ))

    # ------------------------------------------------------------------
    # SECTION 5 — Yield Prediction (Stage 2, optional)
    # ------------------------------------------------------------------
    if predicted_yield is not None:
        story.append(PageBreak())
        story.append(Paragraph("5. Yield Prediction Analysis", section_header))

        # Additional farm parameters table
        story.append(Paragraph("Additional Farm Parameters", subsection_header))
        stage2_data = [
            ['Parameter',        'Value',                    'Unit'],
            ['Soil Moisture',    f'{soil_moisture}',         '%'],
            ['Soil Type',         soil_type,                 '-'],
            ['Sunlight Hours',   f'{sunlight_hours:.1f}',   'hours/day'],
            ['Irrigation Type',   irrigation_type,           '-'],
            ['Fertilizer Used',  f'{fertilizer_used:.1f}',  'kg/hectare'],
            ['Pesticide Used',   f'{pesticide_used:.1f}',   'kg/hectare'],
        ]
        stage2_table = Table(stage2_data, colWidths=[2.5*inch, 1.8*inch, 1.7*inch])
        stage2_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#2c5282')),
            ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1,  0), 12),
            ('BOTTOMPADDING', (0, 0), (-1,  0), 12),
            ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND',    (0, 1), (-1, -1), HexColor('#f7fafc')),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, HexColor('#f7fafc')]),
            ('GRID',          (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
            ('FONTSIZE',      (0, 1), (-1, -1), 11),
            ('TOPPADDING',    (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(stage2_table)
        story.append(Spacer(1, 0.3*inch))

        # Predicted yield headline
        story.append(Paragraph("Predicted Yield Result", subsection_header))
        story.append(Paragraph(
            f"<b>{predicted_yield:.2f} t/ha  ({predicted_yield * 1000:,.0f} kg/ha)</b>",
            ParagraphStyle('YieldValue', parent=body_style, fontSize=20,
                           textColor=HexColor('#2c5282'), alignment=TA_CENTER)
        ))
        story.append(Spacer(1, 0.15*inch))

        # ── Unit explanation table ──────────────────────────────────────
        story.append(Paragraph("Understanding the Yield Units", subsection_header))
        units_data = [
            ['Unit',               'Equals',                                    'Real-World Reference'],
            ['1 Hectare (ha)',      '10,000 m²  |  2.47 acres',                 'About the size of a standard football pitch'],
            ['1 Metric Tonne (t)', '1,000 kg  |  2,204 lbs',                   'Approximately the weight of a small car'],
            ['Yield (t/ha)',        'Tonnes harvested per hectare of farmland',  f'This crop: {predicted_yield:.2f} t/ha'],
        ]
        units_table = Table(units_data, colWidths=[1.5*inch, 2.2*inch, 2.3*inch])
        units_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#1b5e20')),
            ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1,  0), 10),
            ('BOTTOMPADDING', (0, 0), (-1,  0), 10),
            ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [HexColor('#f1f8e9'), HexColor('#e8f5e9'), HexColor('#dcedc8')]),
            ('GRID',          (0, 0), (-1, -1), 0.5, HexColor('#c8e6c9')),
            ('FONTSIZE',      (0, 1), (-1, -1), 9),
            ('TOPPADDING',    (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
        ]))
        story.append(units_table)
        story.append(Spacer(1, 0.2*inch))

        # ── Scale-to-farm breakdown ─────────────────────────────────────
        story.append(Paragraph("Scale to Your Farm Size", subsection_header))
        farm_data = [
            ['Farm Size', 'Est. Total Harvest', 'Est. Total Harvest (kg)'],
            ['1 hectare',  f'{predicted_yield:.2f} t',      f'{predicted_yield * 1000:,.0f} kg'],
            ['2 hectares', f'{predicted_yield * 2:.2f} t',  f'{predicted_yield * 2000:,.0f} kg'],
            ['5 hectares', f'{predicted_yield * 5:.2f} t',  f'{predicted_yield * 5000:,.0f} kg'],
            ['10 hectares',f'{predicted_yield * 10:.2f} t', f'{predicted_yield * 10000:,.0f} kg'],
        ]
        farm_table = Table(farm_data, colWidths=[1.8*inch, 2.0*inch, 2.2*inch])
        farm_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1,  0), HexColor('#2c5282')),
            ('TEXTCOLOR',     (0, 0), (-1,  0), colors.white),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1,  0), 11),
            ('BOTTOMPADDING', (0, 0), (-1,  0), 10),
            ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, HexColor('#f0f4ff')]),
            ('GRID',          (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
            ('FONTSIZE',      (0, 1), (-1, -1), 10),
            ('TOPPADDING',    (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(farm_table)
        story.append(Spacer(1, 0.2*inch))

        # Crop-specific remark
        crop_remarks = {
            "rice":   "Rice thrives with high nitrogen and consistent water management. "
                      "Your predicted yield reflects optimal flooded conditions and balanced nutrients.",
            "maize":  "Maize requires balanced NPK nutrients and adequate sunlight. "
                      "Ensure proper spacing and weed control for maximum yield.",
            "cotton": "Cotton needs sufficient potassium for fiber quality. "
                      "Monitor for pests and ensure adequate irrigation during flowering stage.",
        }
        remark = crop_remarks.get(recommended_crop.lower(), "Ensure proper soil fertility and climate management for best yield.")
        story.append(Paragraph(remark, body_style))

    # ------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(
        "<i>Note: Predictions are based on historical data patterns and should be used "
        "as a decision-support tool, not a guarantee of harvest outcome.</i>",
        ParagraphStyle('Footer', parent=body_style, fontSize=9,
                       textColor=HexColor('#666666'), alignment=TA_CENTER)
    ))

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
        if username == "user" and password == "user123":
            st.session_state.logged_in = True
            st.session_state.page = "trend"
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

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
        
        st.subheader("🔥 Feature Heatmap Across All Crops")
        st.markdown("""
                    **What this shows:** This heatmap compares the **average requirements** of every crop side-by-side.
                    
                    * **Vertical Axis:** The features (N, P, K, Temp, etc.).
                    * **Horizontal Axis:** The different crop types.
                    * **Colors:** 🟩 **Green** indicates high values, while 🟥 **Red** indicates low values.
                    
                    **💡 Pro-Tip:** Use this to find "extreme" crops. For example, you can quickly spot which crops need the most rainfall or the highest Nitrogen levels compared to all others.
        """)
        
        heatmap_data = df.groupby("label")[features_row1 + features_row2].mean()
        fig_heat = px.imshow(
            heatmap_data.T,
            labels=dict(x="Crop", y="Feature", color="Value"),
            aspect="auto",
            color_continuous_scale="RdYlGn"
        )
        fig_heat.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.divider()
    
        st.subheader("🔗 Feature Correlations")
        st.markdown("""
                    **What this shows:** This matrix measures the **strength of the relationship** between two variables.
                    
                    * **Scale:** Values range from **+1.0 to -1.0**.
                    * **Positive (+):** As one feature increases, the other tends to increase.
                    * **Negative (-):** As one increases, the other decreases.
                    * **Near 0:** No linear relationship between the features.
                    
                    **💡 Why it matters:** Highly correlated features provide the same information — useful for feature selection in your ML model.
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
    
    tab1, tab2 = st.tabs(["📊 Crop Overview", "🔬 Crop Comparison"])

    with tab1:
        st.markdown(f"### 📅 Growing Insights")
        with st.container():
            st.markdown(f"""
                *These ranges show the **minimum and maximum** values observed in the dataset for **{selected_crop}**. 
                They represent the environmental tolerance limits of this crop.*
            """)
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
                - It shows the average requirement across all samples
                - **Best for:** Data without extreme outliers
                
                **Median (Middle Value)** 📍
                - It shows the typical middle-ground requirement, less affected by unusual farming conditions or measurement errors
                - **Best for:** Data with outliers or skewed distributions
            
                ---
                
                **🎯 When to use each:**
                
                | Situation | Recommended |
                |-----------|-------------|
                | Normal distribution | **Mean** |
                | Data with outliers | **Median** |
                | Extreme values present | **Median** |
                | Symmetric data | **Mean** |
                """)
        
        if central_tendency == "Mean":
            calculated_values = crop_df[features_row1 + features_row2].mean().round(1)
            metric_label = "Mean"
        else:
            calculated_values = crop_df[features_row1 + features_row2].median().round(1)
            metric_label = "Median"
        
        # ----------------------------
        # Row 1: N, P, K — Gauges + Cards
        # ----------------------------
        st.subheader(f"🌱 Soil Nutrients (NPK) — {metric_label} Values")

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
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        npk_labels = {"N": "🧪 Nitrogen (N)", "P": "🧫 Phosphorus (P)", "K": "💊 Potassium (K)"}
        npk_card_cols = st.columns(3, gap="medium")
        for i, f in enumerate(features_row1):
            status_label, text_color, bg_color, role, meaning, action = get_npk_card(f, calculated_values[f])
            with npk_card_cols[i]:
                st.markdown(f"""
                    <div style='
                        background-color:{bg_color};
                        border-left:6px solid {text_color};
                        border-radius:16px;
                        padding:20px 20px 16px 20px;
                        box-shadow:0 4px 14px rgba(0,0,0,0.09);
                        height:260px;
                        box-sizing:border-box;
                        overflow:hidden;
                        display:flex;
                        flex-direction:column;
                        justify-content:space-between;
                    '>
                        <div>
                            <div style='font-size:16px;font-weight:800;color:{text_color};margin-bottom:2px;'>
                                {npk_labels[f]}
                            </div>
                            <div style='font-size:11px;color:#555;font-style:italic;margin-bottom:10px;'>
                                {role}
                            </div>
                            <div style='
                                display:inline-flex;
                                align-items:center;
                                gap:6px;
                                background:{text_color};
                                color:white;
                                font-size:12px;
                                font-weight:700;
                                border-radius:24px;
                                padding:4px 14px;
                                margin-bottom:10px;
                            '>
                                {status_label}&nbsp;|&nbsp;{calculated_values[f]}
                            </div>
                            <div style='font-size:12.5px;color:#222;line-height:1.55;'>
                                {meaning}
                            </div>
                        </div>
                        <div style='
                            font-size:12px;
                            color:{text_color};
                            font-weight:700;
                            border-top:1px solid {text_color}33;
                            padding-top:8px;
                            margin-top:8px;
                        '>
                            {action}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ----------------------------
        # Row 2: pH, Temperature, Humidity, Rainfall — Gauges + Cards
        # ----------------------------
        st.subheader(f"🌤️ Climate & Soil Conditions — {metric_label} Values")

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
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        climate_labels = {
            "ph":          "⚗️ pH Level",
            "temperature": "🌡️ Temperature",
            "humidity":    "💧 Humidity",
            "rainfall":    "🌧️ Rainfall"
        }
        climate_card_cols = st.columns(4, gap="medium")
        for i, f in enumerate(features_row2):
            status_label, text_color, bg_color, role, meaning, action = get_climate_card(f, calculated_values[f])
            with climate_card_cols[i]:
                st.markdown(f"""
                    <div style='
                        background-color:{bg_color};
                        border-left:6px solid {text_color};
                        border-radius:16px;
                        padding:20px 18px 16px 18px;
                        box-shadow:0 4px 14px rgba(0,0,0,0.09);
                        height:260px;
                        box-sizing:border-box;
                        overflow:hidden;
                        display:flex;
                        flex-direction:column;
                        justify-content:space-between;
                    '>
                        <div>
                            <div style='font-size:16px;font-weight:800;color:{text_color};margin-bottom:2px;'>
                                {climate_labels[f]}
                            </div>
                            <div style='font-size:11px;color:#555;font-style:italic;margin-bottom:10px;'>
                                {role}
                            </div>
                            <div style='
                                display:inline-flex;
                                align-items:center;
                                gap:6px;
                                background:{text_color};
                                color:white;
                                font-size:12px;
                                font-weight:700;
                                border-radius:24px;
                                padding:4px 12px;
                                margin-bottom:10px;
                            '>
                                {status_label}&nbsp;|&nbsp;{calculated_values[f]}{feature_units[f]}
                            </div>
                            <div style='font-size:12px;color:#222;line-height:1.55;'>
                                {meaning}
                            </div>
                        </div>
                        <div style='
                            font-size:12px;
                            color:{text_color};
                            font-weight:700;
                            border-top:1px solid {text_color}33;
                            padding-top:8px;
                            margin-top:8px;
                        '>
                            {action}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 Distribution Analysis")
           
        selected_param = st.selectbox(
               "View distribution for:",
               features_row1 + features_row2,
               format_func=lambda x: feature_names[x]
        )
        
        param_mean = crop_df[selected_param].mean()
        param_median = crop_df[selected_param].median()
           
        fig = px.histogram(
               crop_df, 
               x=selected_param,
               nbins=30,
               title=f"{feature_names[selected_param]} Distribution for {selected_crop}",
               color_discrete_sequence=['#4B371C']
        )
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
            * **Narrow Cluster:** The crop has very **strict** requirements — be precise with inputs.
            * **Wide Spread:** The crop is **resilient** and tolerates a wider range of conditions.
            * **Gaps in Bars:** Certain conditions may be unsuitable for growth.
            """)
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown(f"### Compare **{selected_crop.upper()}** {emoji} with Other Crops")
        st.info("Select up to 3 crops to compare their optimal growing conditions side-by-side.")

        st.markdown("### 📊 Statistical Measure Selection")
        col_t2_radio, col_t2_exp = st.columns([1, 2])

        with col_t2_radio:
            central_tendency_tab2 = st.radio(
                "Choose calculation method:",
                ("Mean", "Median"),
                key="tab2_central_tendency",
                help="Select how to calculate central values for the comparison crops"
            )

        with col_t2_exp:
            with st.expander("ℹ️ **What's the difference?**"):
                st.markdown("""
                **Mean (Average)** 📊
                - Shows the average requirement across all samples for each crop
                - **Best for:** Data without extreme outliers

                **Median (Middle Value)** 📍
                - Shows the typical middle-ground requirement, less affected by unusual farming conditions or measurement errors
                - **Best for:** Data with outliers or skewed distributions

                ---

                **🎯 When to use each:**

                | Situation | Recommended |
                |-----------|-------------|
                | Normal distribution | **Mean** |
                | Data with outliers | **Median** |
                | Extreme values present | **Median** |
                | Symmetric data | **Mean** |

                > 💡 **Tip:** Changing this selector only affects the Crop Comparison tab. The Crop Overview tab has its own independent selector.
                """)

        if central_tendency_tab2 == "Mean":
            calculated_values_tab2 = crop_df[features_row1 + features_row2].mean().round(1)
            metric_label_tab2 = "Mean"
        else:
            calculated_values_tab2 = crop_df[features_row1 + features_row2].median().round(1)
            metric_label_tab2 = "Median"

        compare_crops = st.multiselect(
            "🌾 Select crops to compare with " + selected_crop,
            [c for c in sorted(df["label"].unique()) if c != selected_crop],
            max_selections=3,
            help="Choose crops you want to compare"
        )
        
        if compare_crops:
            comparison_data = {"Crop": [selected_crop] + compare_crops}
            for feature in features_row1 + features_row2:
                comparison_data[feature_names[feature]] = [calculated_values_tab2[feature]]
                for crop in compare_crops:
                    if central_tendency_tab2 == "Mean":
                        crop_value = df[df["label"] == crop][feature].mean()
                    else:
                        crop_value = df[df["label"] == crop][feature].median()
                    comparison_data[feature_names[feature]].append(round(crop_value, 1))
            
            comp_df = pd.DataFrame(comparison_data)
            
            st.markdown(f"#### 📋 Comparison Table in Farm Environment Profile ({metric_label_tab2})")
            st.dataframe(
                comp_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Crop": st.column_config.TextColumn("🌱 Crop", width="medium"),
                }
            )
            
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
                    title=f"{selected_feature} Comparison Across Crops ({metric_label_tab2})",
                    color="Crop",
                    text=selected_feature,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_traces(texttemplate='%{text}', textposition='outside')
                fig.update_layout(showlegend=False, height=450, xaxis_title="Crop", yaxis_title=selected_feature)
                with st.expander("📊 **Understanding the Bar Chart**"):
                    st.markdown("""
                        * This chart compares **Single Parameter Focus** requirements across the selected crops.
                        * Use this to see exact numerical differences for a specific metric. 
                        * It is the best way to determine which crop is the "most" or "least" demanding for a single nutrient.
                    """)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                categories = [feature_names[f] for f in features_row1 + features_row2]
                fig = go.Figure()
                for idx, crop in enumerate([selected_crop] + compare_crops):
                    crop_data = comp_df[comp_df["Crop"] == crop]
                    values = [(crop_data[feature_names[feature]].values[0] / feature_max[feature])
                              for feature in features_row1 + features_row2]
                    fig.add_trace(go.Scatterpolar(
                        r=values, theta=categories, fill='toself', name=crop, line=dict(width=2)
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%')),
                    showlegend=True,
                    title=f"Normalized Multi-Parameter Comparison ({metric_label_tab2})",
                    height=500
                )
                with st.expander("📊 **Understanding the Radar Chart**"):
                    st.markdown("""
                            * This chart **normalizes** all values (0% to 100%) so you can compare temperature, pH, and nutrients on the same scale.
                            * **What to look for:** If the shapes of two crops overlap significantly, they share a similar "biological fingerprint" and likely grow well in the same regions.
                    """)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🔍 Key Differences")
            with st.expander("ℹ️ **How to interpret these differences**"):
                st.markdown(f"""
                This section highlights where the comparison crops **deviate significantly** from your primary choice, **{selected_crop.upper()}**.
                
                * **The 20% Rule:** We only display a difference if it is greater than **20%**. This filters out minor variations and focuses on the factors that truly change how you manage the field.
                * **🔺 (Red Up):** Indicates the comparison crop requires **significantly more** of this resource.
                * **🔻 (Red Down):** Indicates the comparison crop requires **significantly less**.
                
                **Why this matters:** If you see a **+80% Nitrogen** difference, switching to that crop would require a complete overhaul of your fertilization schedule.
                """)
                st.latex(r"Percentage\ Change = \frac{Comparison\ Mean - Selected\ Mean}{Selected\ Mean} \times 100")
                
            diff_cols = st.columns(len(compare_crops))
            for idx, crop in enumerate(compare_crops):
                with diff_cols[idx]:
                    crop_emoji = crop_emojis.get(crop.lower(), "🌱")
                    st.markdown(f"**{crop_emoji} {crop.upper()} vs {selected_crop.upper()}**")
                    if central_tendency_tab2 == "Mean":
                        crop_data = df[df["label"] == crop][features_row1 + features_row2].mean()
                    else:
                        crop_data = df[df["label"] == crop][features_row1 + features_row2].median()
                    differences = []
                    for feature in features_row1 + features_row2:
                        diff = crop_data[feature] - calculated_values_tab2[feature]
                        pct_diff = (diff / calculated_values_tab2[feature] * 100) if calculated_values_tab2[feature] != 0 else 0
                        if abs(pct_diff) > 20:
                            if diff > 0:
                                differences.append(f"🔺 {feature_names[feature]}: +{pct_diff:.0f}%")
                            else:
                                differences.append(f"🔻 {feature_names[feature]}: {pct_diff:.0f}%")
                    if differences:
                        for diff in differences[:3]:
                            st.caption(diff)
                    else:
                        st.caption("Similar conditions")
            
            st.markdown("---")
            st.markdown("#### 🎯 Similarity Analysis")
            for crop in compare_crops:
               if central_tendency_tab2 == "Mean":
                   crop_data = df[df["label"] == crop][features_row1 + features_row2].mean()
               else:
                   crop_data = df[df["label"] == crop][features_row1 + features_row2].median()
               differences = []
               for feature in features_row1 + features_row2:
                   diff = abs(crop_data[feature] - calculated_values_tab2[feature])
                   norm_diff = diff / feature_max[feature]
                   differences.append(norm_diff)
               similarity = (1 - sum(differences) / len(differences)) * 100
               crop_emoji = crop_emojis.get(crop.lower(), "🌱")
               st.progress(similarity / 100)
               st.caption(f"{crop_emoji} **{crop}** is {similarity:.1f}% similar to **{selected_crop}**")
        
        else:
            st.warning("👆 Select at least one crop to start comparing!")

def show_prediction():
    st.title("🌱 Intelligent Crop Recommendation")
    
    stage1_model, le = load_stage1()
    stage2_model = load_stage2()
    
    if stage1_model is None or le is None:
        st.error("🚨 Stage 1 model files missing.")
        return
    if stage2_model is None:
        st.warning("⚠️ Stage 2 model not loaded. You can still get crop recommendation.")

    with st.expander("📖 **How to Use This System?**"):
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Stage 1: Crop Recommendation", value="99.5%", delta="Accuracy")
        with m2:
            st.metric(label="Stage 2: Yield Prediction", value="72.3%", delta="R² Score")
        st.caption("⚠️ Predictions are based on historical data and should be used as decision support, not a guarantee of results.")

        st.markdown("---")
        
        st.markdown("### 🌱 Stage 1: Crop Recommendation")
        st.markdown("""
            - Uses Random Forest model  
            - Recommends the most suitable crop based on soil and environmental conditions  
        """)
    
        st.markdown("**Step 1: Enter Input Data**")
        st.write("""
            - Fill in soil and environmental details (N, P, K, temperature, humidity, pH, rainfall)  
        """)
    
        st.markdown("**Step 2: Generate Recommendation**")
        st.write("""
            - Click the prediction button  
            - The system will suggest the most suitable crop  
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Stage 2: Yield Prediction")
        st.markdown("""
            - Available for selected crops (Rice, Maize, Cotton)  
            - Uses XGBoost model to estimate crop yield  
        """)
    
        st.markdown("**Step 3: Confirm & Proceed**")
        st.write("""
            - Review the recommended crop  
            - Click **\"Yes\"** to proceed  
        """)
    
        st.markdown("**Step 4: View Yield Result**")
        st.write("""
            - Enter additional farming details if required  
            - The system will display the predicted crop yield  
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
            "kidneybeans":"🫘", "chickpea":"🫘", "coffee":"☕"
        }
        
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

        # ── MATCH SCORE SECTION ───────────────────────────────────────────────
        st.markdown("##### 🎯 Frequently Used Parameter Match Score")

        PARAM_RANGES = {
            "Nitrogen (N)":   150,
            "Phosphorus (P)": 150,
            "Potassium (K)":  150,
            "pH Level":       14,
            "Temperature":    50,
            "Humidity":       100,
            "Rainfall":       300,
        }

        params = {
            "Nitrogen (N)":   (N,    crop_frequently_used["N"],           150),
            "Phosphorus (P)": (P,    crop_frequently_used["P"],           150),
            "Potassium (K)":  (K,    crop_frequently_used["K"],           150),
            "pH Level":       (ph,   crop_frequently_used["ph"],          14),
            "Temperature":    (temp, crop_frequently_used["temperature"],  50),
            "Humidity":       (hum,  crop_frequently_used["humidity"],    100),
            "Rainfall":       (rain, crop_frequently_used["rainfall"],    300),
        }

        with st.expander("ℹ️ **How is the Match Score calculated?**"):
            st.markdown("""
            Each bar shows how close **your input** is to the **most frequently recorded value**
            for this crop in the dataset.

            **Formula used:**
            """)
            st.latex(r"\text{Match \%} = \left(1 - \frac{|\text{Your Value} - \text{Typical Value}|}{\text{Parameter Range}}\right) \times 100")
            st.markdown("""
            | Score | Meaning |
            |-------|---------|
            | 🟢 ≥ 80% | Your value is very close to the typical value — great fit |
            | 🟡 50–79% | Moderate deviation — crop may still grow, but consider adjusting |
            | 🔴 < 50% | Large deviation — this parameter may limit growth |

            > **Why "Parameter Range" and not the typical value itself?**  
            > Dividing by a small typical value (e.g. pH mode = 5) would make tiny
            > differences look catastrophic. Using the full scale (e.g. pH 0–14) keeps
            > scores fair and comparable across all parameters.
            """)

        col_left, col_right = st.columns(2)
        param_items = list(params.items())
        mid_point = len(param_items) // 2 + len(param_items) % 2

        def render_param_bar(param_name, user_val, freq_val, scale_range):
            match_pct = (1 - abs(user_val - freq_val) / scale_range) * 100
            match_pct = max(0, min(100, match_pct))
            if match_pct >= 80:
                color = "🟢"
            elif match_pct >= 50:
                color = "🟡"
            else:
                color = "🔴"
            diff = user_val - freq_val
            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            st.markdown(f"**{color} {param_name}**")
            st.progress(match_pct / 100)
            st.caption(
                f"Your: **{user_val:.1f}** | Typical: **{freq_val:.1f}** | "
                f"Diff: {diff_str} | Match: **{match_pct:.0f}%**"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            return match_pct

        overall_matches = []
        param_matches_dict = {}

        with col_left:
            for param_name, (user_val, freq_val, scale_range) in param_items[:mid_point]:
                pct = render_param_bar(param_name, user_val, freq_val, scale_range)
                overall_matches.append(pct)
                param_matches_dict[param_name] = (user_val, freq_val, pct)

        with col_right:
            for param_name, (user_val, freq_val, scale_range) in param_items[mid_point:]:
                pct = render_param_bar(param_name, user_val, freq_val, scale_range)
                overall_matches.append(pct)
                param_matches_dict[param_name] = (user_val, freq_val, pct)

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

        # ── PDF DOWNLOAD — STAGE 1 ────────────────────────────────────────────
        st.markdown("---")
        try:
            pdf_filename = f"crop_report_{crop_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
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

    # ── STAGE 2: YIELD PREDICTION ─────────────────────────────────────────────
    # if st.session_state.get('submitted', False):
    #     crop_name = st.session_state.stage1_crop
    #     allowed_crops = ["rice", "maize", "cotton"]
        
    #     if crop_name.strip().lower() in allowed_crops and stage2_model is not None:
    
    if st.session_state.get('submitted', False):
    crop_name = st.session_state.get("stage1_crop", "")
    allowed_crops = ["rice", "maize", "cotton"]
    
        if isinstance(crop_name, str) and crop_name.strip().lower() in allowed_crops and stage2_model is not None:
            st.markdown("---")

            if 'stage2_choice' not in st.session_state:
                st.session_state.stage2_choice = "No"

            st.markdown(f"""
                <div style='background-color:#FFFFFF; padding:20px; border-radius:10px; border-left:5px solid #4caf50; margin: 20px 0;'>
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

                # ── Unit explanation expander ─────────────────────────────────
                with st.expander("🪵 **Understanding the Yield Units**"):
                    st.markdown("""
                        <div style='
                            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
                            border-radius: 10px;
                            padding: 16px 20px;
                        '>
                            <div style='display:flex; gap:24px; flex-wrap:wrap;'>
                                <div style='flex:1; min-width:160px;'>
                                    <div style='font-size:13px; font-weight:700; color:#1b5e20;'>🟩 1 Hectare (ha)</div>
                                    <div style='font-size:12.5px; color:#333; margin-top:4px; line-height:1.6;'>
                                        = 10,000 m² of land<br>
                                        ≈ the size of a standard football pitch<br>
                                        ≈ 2.47 acres
                                    </div>
                                </div>
                                <div style='flex:1; min-width:160px;'>
                                    <div style='font-size:13px; font-weight:700; color:#1b5e20;'>⚖️ 1 Metric Tonne (t)</div>
                                    <div style='font-size:12.5px; color:#333; margin-top:4px; line-height:1.6;'>
                                        = 1,000 kg of crop weight<br>
                                        ≈ 2,204 lbs<br>
                                        ≈ the weight of a small car
                                    </div>
                                </div>
                                <div style='flex:1; min-width:160px;'>
                                    <div style='font-size:13px; font-weight:700; color:#1b5e20;'>📦 Yield (t/ha)</div>
                                    <div style='font-size:12.5px; color:#333; margin-top:4px; line-height:1.6;'>
                                        = Metric tonnes harvested<br>per hectare of farmland<br>
                                        e.g. 3 t/ha → 3,000 kg per field of 10,000 m²
                                    </div>
                                </div>
                            </div>
                            <div style='margin-top:12px; font-size:12px; color:#555; font-style:italic;'>
                                💡 Example: If the model predicts <strong>4.5 t/ha</strong> and you farm <strong>3 hectares</strong>,
                                your estimated total harvest = 4.5 × 3 = <strong>13.5 metric tonnes (13,500 kg)</strong>.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                

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
                            "Soil Moisture (%)", 0, 100, 50,
                            help="Current soil moisture content percentage"
                        )
                        soil_type = st.selectbox(
                            "Soil Type", ["Loamy", "Sandy", "Silt", "Clay"],
                            help="Primary soil composition type"
                        )
                        sunlight_hours = st.number_input(
                            "Sunlight Hours (hours/day)", 0.0, 24.0, 8.0, 0.5,
                            help="Average daily sunlight exposure"
                        )
                    
                    with col2:
                        st.markdown("##### **Farm Management**")
                        irrigation_type = st.selectbox(
                            "Irrigation Type", ["Drip", "Canal", "Rainfed", "Sprinkler"],
                            help="Primary irrigation method used"
                        )
                        fertilizer_used = st.number_input(
                            "Fertilizer Used (kg/hectare)", 0.0, 500.0, 100.0, 10.0,
                            help="Amount of fertilizer applied per hectare of farmland"
                        )
                        pesticide_used = st.number_input(
                            "Pesticide Used (kg/hectare)", 0.0, 50.0, 5.0, 0.5,
                            help="Amount of pesticide applied per hectare of farmland"
                        )

                    st.markdown("---")
                    submit_stage2 = st.form_submit_button("✨  Predict Yield")
                
                if submit_stage2:
                    stage2_input = {
                        "N":               st.session_state.stage1_input["N"],
                        "P":               st.session_state.stage1_input["P"],
                        "K":               st.session_state.stage1_input["K"],
                        "ph":              st.session_state.stage1_input["ph"],
                        "temperature":     st.session_state.stage1_input["temperature"],
                        "humidity":        st.session_state.stage1_input["humidity"],
                        "rainfall":        st.session_state.stage1_input["rainfall"],
                        "Soil_Moisture":   soil_moisture,
                        "Sunlight_Hours":  sunlight_hours,
                        "Fertilizer_Used": fertilizer_used,
                        "Pesticide_Used":  pesticide_used,
                        "Soil_Type":       soil_type,
                        "Irrigation_Type": irrigation_type,
                        "Crop_Type":       crop_name,
                    }
                    
                    stage2_input_df = pd.DataFrame([stage2_input])
                    
                    try:
                        yield_pred = stage2_model.predict(stage2_input_df)[0]
                        
                        crop_remarks = {
                            "rice":   "Rice thrives with high nitrogen and consistent water management. Your predicted yield reflects optimal flooded conditions and balanced nutrients.",
                            "maize":  "Maize requires balanced NPK nutrients and adequate sunlight. Ensure proper spacing and weed control for maximum yield.",
                            "cotton": "Cotton needs sufficient potassium for fiber quality. Monitor for pests and ensure adequate irrigation during flowering stage."
                        }
                        remark = crop_remarks.get(crop_name.lower(), "Ensure proper soil fertility and climate management for best yield.")

                        total_kg = yield_pred * 1000

                        st.markdown(f"""
                            <div class="prediction-card" style="background: linear-gradient(135deg, #ffffff 0%, #fafcf7 100%); color: white;">
                                <h2 style="color: white;">🎯 Predicted Yield: <strong>{yield_pred:.2f} t/ha</strong></h2>
                                <p style="color: black; opacity: 0.95;">{remark}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        # ── NEW: Yield breakdown explanation card ─────────────
                        st.markdown(f"""
                            <div style='
                                background-color:#f3f8ff;
                                border-left:5px solid #1565c0;
                                border-radius:12px;
                                padding:18px 22px;
                                margin:16px 0;
                                box-shadow:0 2px 8px rgba(0,0,0,0.07);
                            '>
                                <div style='font-size:15px;font-weight:700;color:#1565c0;margin-bottom:12px;'>
                                    📊 What does {yield_pred:.2f} t/ha mean?
                                </div>
                                <div style='display:flex;gap:16px;flex-wrap:wrap;'>
                                    <div style='
                                        flex:1;min-width:140px;
                                        background:white;border-radius:10px;
                                        padding:12px 16px;text-align:center;
                                        box-shadow:0 1px 4px rgba(0,0,0,0.08);
                                    '>
                                        <div style='font-size:22px;font-weight:800;color:#1565c0;'>{yield_pred:.2f}</div>
                                        <div style='font-size:12px;color:#555;margin-top:2px;'>metric tonnes<br>per hectare</div>
                                    </div>
                                    <div style='
                                        flex:1;min-width:140px;
                                        background:white;border-radius:10px;
                                        padding:12px 16px;text-align:center;
                                        box-shadow:0 1px 4px rgba(0,0,0,0.08);
                                    '>
                                        <div style='font-size:22px;font-weight:800;color:#2e7d32;'>{total_kg:,.0f}</div>
                                        <div style='font-size:12px;color:#555;margin-top:2px;'>kilograms<br>per hectare</div>
                                    </div>
                                    <div style='
                                        flex:1;min-width:140px;
                                        background:white;border-radius:10px;
                                        padding:12px 16px;text-align:center;
                                        box-shadow:0 1px 4px rgba(0,0,0,0.08);
                                    '>
                                        <div style='font-size:22px;font-weight:800;color:#6a1b9a;'>{yield_pred * 10000:.0f}</div>
                                        <div style='font-size:12px;color:#555;margin-top:2px;'>square metres<br>= 1 hectare</div>
                                    </div>
                                </div>
                                <div style='
                                    margin-top:14px;
                                    background:#e3f2fd;
                                    border-radius:8px;
                                    padding:10px 14px;
                                    font-size:13px;
                                    color:#1a237e;
                                    line-height:1.6;
                                '>
                                    💡 <strong>Scale it to your farm:</strong>
                                    multiply the predicted yield by your farm size in hectares.<br>
                                    e.g. farming <strong>5 ha</strong> → estimated harvest =
                                    {yield_pred:.2f} × 5 = <strong>{yield_pred * 5:.2f} t
                                    ({yield_pred * 5 * 1000:,.0f} kg)</strong>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        # ── END yield breakdown card ──────────────────────────

                        st.balloons()
                        
                        # ── PDF DOWNLOAD — STAGE 2 ────────────────────────────
                        st.markdown("---")
                        try:
                            pdf_filename_full = f"crop_report_full_{crop_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            pdf_buffer_full = create_crop_prediction_pdf(
                                N=st.session_state.stage1_input["N"],
                                P=st.session_state.stage1_input["P"],
                                K=st.session_state.stage1_input["K"],
                                ph=st.session_state.stage1_input["ph"],
                                temperature=st.session_state.stage1_input["temperature"],
                                humidity=st.session_state.stage1_input["humidity"],
                                rainfall=st.session_state.stage1_input["rainfall"],
                                recommended_crop=crop_name,
                                thi=st.session_state.thi,
                                sfi=st.session_state.sfi,
                                parameter_matches=st.session_state.param_matches,
                                overall_match=st.session_state.overall_match,
                                soil_moisture=soil_moisture,
                                soil_type=soil_type,
                                sunlight_hours=sunlight_hours,
                                irrigation_type=irrigation_type,
                                fertilizer_used=fertilizer_used,
                                pesticide_used=pesticide_used,
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
        
        # elif crop_name.strip().lower() not in allowed_crops:
        #     pass
        elif isinstance(crop_name, str) and crop_name.strip().lower() not in allowed_crops:
            pass
                          
# =============================
# MAIN NAVIGATION
# =============================
if st.session_state.logged_in:
    st.sidebar.title("🧭 Navigation")
    st.sidebar.write(f"Logged in as: **User**")
    
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
