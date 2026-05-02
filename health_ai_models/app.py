"""
🏥 Agentic AI Health Coach — Streamlit UI
Early Disease Risk Prediction & Post-Surgery Recovery Assistant
"""
import os
import sys
import json
import csv
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# ── Project path setup ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="AI Health Coach",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
import base64

# Load background image if available
bg_css = ""
bg_image_path = os.path.join(BASE_DIR, "assets", "bg_stethoscope.png")
if not os.path.exists(bg_image_path):
    bg_image_path = os.path.join(BASE_DIR, "assets", "bg_stethoscope.jpg")
if os.path.exists(bg_image_path):
    with open(bg_image_path, "rb") as f:
        bg_data = base64.b64encode(f.read()).decode()
    img_mime = "image/png" if bg_image_path.endswith(".png") else "image/jpeg"
    bg_css = f"""
    .stApp {{
        background: linear-gradient(rgba(10, 25, 47, 0.7), rgba(10, 25, 47, 0.75)),
                    url("data:{img_mime};base64,{bg_data}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    bg_css = """
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #0d2137 50%, #112240 100%);
    }
    """

st.markdown("<style>" + bg_css + """
    /* Top toolbar/header bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: white !important; margin: 0; font-size: 2.2rem; }
    .main-header p { color: #e0e0ff; margin: 0.3rem 0 0 0; font-size: 1rem; }

    /* Risk badge styles */
    .risk-high { background: #ff4444; color: white; padding: 4px 14px; border-radius: 20px; font-weight: bold; }
    .risk-moderate { background: #ffaa00; color: #333; padding: 4px 14px; border-radius: 20px; font-weight: bold; }
    .risk-low { background: #00C851; color: white; padding: 4px 14px; border-radius: 20px; font-weight: bold; }

    /* Card style */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    /* Sidebar style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        min-width: 300px !important;
        width: 300px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        font-size: 1.3rem !important;
        padding: 0.7rem 1rem !important;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: background 0.2s;
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.8rem !important;
        color: white !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #aaa !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    /* Feature card on landing */
    .feature-card {
        background: rgba(10, 25, 47, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.15);
        height: 100%;
        backdrop-filter: blur(5px);
    }
    .feature-card h3 { color: #e0e0e0; margin-top: 0.5rem; }
    .feature-card p { color: #b0b0b0; font-size: 0.9rem; }

    /* Main content fills full width */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Input fields and form elements - match button style */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stNumberInput input,
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect > div > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stNumberInput"] > div > div,
    [data-testid="stTextInput"] input,
    [data-baseweb="input"] input,
    [data-baseweb="input"],
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    .stNumberInput > div > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    input[type="number"],
    input[type="text"],
    input {
        background: rgba(144, 164, 194, 0.2) !important;
        border: 1px solid rgba(144, 164, 194, 0.35) !important;
        color: white !important;
        height: 42px !important;
        min-height: 42px !important;
    }
    /* Remove the vertical separator line in select boxes */
    .stSelectbox div[data-baseweb="select"] > div > div {
        border-right: none !important;
        border-left: none !important;
    }
    .stSelectbox div[data-baseweb="select"] > div > div + div {
        border-left: none !important;
    }
    /* Hide any separator elements inside select */
    .stSelectbox [data-baseweb="select"] div[role="separator"],
    .stSelectbox [data-baseweb="select"] div[aria-hidden="true"]:not(svg) {
        display: none !important;
        width: 0 !important;
    }
    /* Force no internal borders on select children */
    .stSelectbox [data-baseweb="select"] > div * {
        border-left: none !important;
        border-right: none !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        border: 1px solid rgba(144, 164, 194, 0.35) !important;
        cursor: pointer !important;
    }
    .stSelectbox [data-baseweb="select"] > div * {
        cursor: pointer !important;
    }
    /* File uploader - single clean box */
    .stFileUploader {
        background: rgba(144, 164, 194, 0.2) !important;
        border: 1px solid rgba(144, 164, 194, 0.35) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: white !important;
    }
    .stFileUploader > div,
    .stFileUploader section,
    .stFileUploader section > button,
    .stFileUploader [data-testid="stFileUploaderDropzone"],
    .stFileUploader [data-testid="stFileUploaderDropzone"] * {
        background: transparent !important;
        border: none !important;
        color: white !important;
        height: auto !important;
        min-height: auto !important;
    }
    .stFileUploader button {
        background: rgba(144, 164, 194, 0.3) !important;
        border: 1px solid rgba(144, 164, 194, 0.4) !important;
        color: white !important;
    }
    /* Slider - no fixed height */
    .stSlider > div {
        background: transparent !important;
        border: none !important;
        height: auto !important;
        min-height: auto !important;
    }
    .stFormSubmitButton > button {
        background: rgba(144, 164, 194, 0.2) !important;
        border: 1px solid rgba(144, 164, 194, 0.35) !important;
        color: white !important;
        height: 42px !important;
        min-height: 42px !important;
    }
    /* Make all input containers same width */
    .stTextInput, .stNumberInput, .stSelectbox, .stMultiSelect {
        width: 100% !important;
    }
    /* Form submit instruction text - hide it */
    .stTextInput [data-testid="InputInstructions"],
    .stNumberInput [data-testid="InputInstructions"],
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    /* Also style the number input step buttons */
    .stNumberInput button,
    [data-testid="stNumberInput"] button {
        background: rgba(144, 164, 194, 0.25) !important;
        border: 1px solid rgba(144, 164, 194, 0.35) !important;
        color: white !important;
    }

    /* Buttons match same style as inputs */
    .stButton > button {
        background: rgba(144, 164, 194, 0.2) !important;
        border: 1px solid rgba(144, 164, 194, 0.35) !important;
        color: white !important;
    }
    .stButton > button:hover {
        background: rgba(144, 164, 194, 0.35) !important;
        border: 1px solid rgba(144, 164, 194, 0.5) !important;
    }

    /* Dropdown/select menu (the popup list when clicking select) */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [data-baseweb="menu"],
    [data-baseweb="menu"] *,
    [data-baseweb="listbox"],
    [data-baseweb="listbox"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    div[role="listbox"],
    div[role="listbox"] *,
    [data-baseweb="select"] ~ div,
    [data-baseweb="select"] ~ div * {
        background-color: #1a2940 !important;
        background: #1a2940 !important;
        color: white !important;
    }
    ul[role="listbox"] li:hover,
    [data-baseweb="menu"] li:hover,
    [data-baseweb="listbox"] li:hover {
        background: rgba(144, 164, 194, 0.3) !important;
        background-color: rgba(144, 164, 194, 0.3) !important;
    }
    /* Expander header styling */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details summary {
        background: transparent !important;
        background-color: transparent !important;
        color: white !important;
    }
    [data-testid="stExpander"] {
        background: transparent !important;
        background-color: transparent !important;
        border-color: rgba(144, 164, 194, 0.3) !important;
    }
    [data-testid="stExpander"] details {
        background: transparent !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
# Session state for page navigation
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Home"

NAV_OPTIONS = ["🏠 Home", "🫀 Cardiovascular", "🩸 Diabetes", "🫁 Pneumonia", "🏥 Recovery Assistant", "📝 Feedback"]

with st.sidebar:
    st.markdown("")
    st.markdown("# 🏥 AI Health Coach")
    st.markdown("")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Navigate",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state.nav_page),
        label_visibility="collapsed",
    )
    st.session_state.nav_page = page
    st.markdown("---")
    st.markdown("")
    st.markdown("")
    st.markdown("**Powered by**")
    st.markdown("Hybrid ML · LSTM · Gen AI")
    st.markdown("")
    st.markdown(f"© {datetime.now().year} · Senior Year Project")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: risk badge
# ═══════════════════════════════════════════════════════════════════════════
def risk_badge(level: str) -> str:
    level = level.lower()
    css = {"high": "risk-high", "moderate": "risk-moderate", "low": "risk-low"}.get(level, "risk-low")
    return f'<span class="{css}">{level.upper()} RISK</span>'


FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")


def save_feedback(module: str, patient_name: str, rating: int, comment: str):
    """Save feedback entry to CSV file."""
    fieldnames = ["module", "patient_name", "rating", "comment", "date", "time"]
    row = {
        "module": module,
        "patient_name": patient_name,
        "rating": rating,
        "comment": comment,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    write_header = not os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# 🏠 HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Agentic AI Health Coach</h1>
        <p>Early Disease Risk Prediction & Post-Surgery Recovery Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Welcome! Choose a module to get started:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h1>🫀</h1>
            <h3>Cardiovascular</h3>
            <p>Predict heart disease risk from vitals like blood pressure, cholesterol, and lifestyle factors.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="nav_cardio", width="stretch"):
            st.session_state.nav_page = "🫀 Cardiovascular"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h1>🩸</h1>
            <h3>Diabetes</h3>
            <p>Assess diabetes risk using 21 health indicators from the CDC BRFSS survey.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="nav_diabetes", width="stretch"):
            st.session_state.nav_page = "🩸 Diabetes"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h1>🫁</h1>
            <h3>Pneumonia</h3>
            <p>Detect pneumonia from chest X-ray images using deep learning (CNN).</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="nav_pneumonia", width="stretch"):
            st.session_state.nav_page = "🫁 Pneumonia"
            st.rerun()

    with col4:
        st.markdown("""
        <div class="feature-card">
            <h1>🏥</h1>
            <h3>Recovery Coach</h3>
            <p>Post-surgery recovery assistant powered by LSTM forecasting + Generative AI advice.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", key="nav_recovery", width="stretch"):
            st.session_state.nav_page = "🏥 Recovery Assistant"
            st.rerun()

    st.markdown("---")
    with st.expander("🏗️ System Architecture", expanded=False):
        st.markdown("""
        | Component | Technology |
        |-----------|-----------|
        | **Disease Prediction** | XGBoost / LightGBM classifiers |
        | **Pneumonia Detection** | ResNet-50 CNN on chest X-rays |
        | **Recovery Forecasting** | Hybrid LSTM (vitals + static features) |
        | **AI Coach** | Gemini 2.0 Flash / GPT-3.5 / Offline rules |
        | **Frontend** | Streamlit |
        | **Backend** | Python inference pipeline |
        """)


# ═══════════════════════════════════════════════════════════════════════════
# 🫀 CARDIOVASCULAR PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🫀 Cardiovascular":
    st.markdown("""
    <div class="main-header">
        <h1>🫀 Cardiovascular Disease Prediction</h1>
        <p>XGBoost / LightGBM model trained on 70,000 patient records</p>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input("👤 Patient Name", placeholder="Enter patient name")

    with st.form("cardio_form"):
        st.markdown("#### Patient Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=168)
            ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=60, max_value=250, value=120)
            cholesterol = st.selectbox("Cholesterol", options=[1, 2, 3],
                                       format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

        with c2:
            gender = st.selectbox("Gender", options=[1, 2],
                                   format_func=lambda x: {1: "Female", 2: "Male"}[x])
            weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
            ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80)
            gluc = st.selectbox("Glucose", options=[1, 2, 3],
                                 format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

        with c3:
            smoke = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            alco = st.selectbox("Alcohol intake?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            active = st.selectbox("Physically active?", [1, 0], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔍 Predict Risk", width="stretch")

    if submitted:
        with st.spinner("Running cardiovascular model..."):
            try:
                from inference.predict_cardio import predict_cardio
                result = predict_cardio({
                    "age": age_years * 365,  # model expects age in days
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "ap_hi": ap_hi,
                    "ap_lo": ap_lo,
                    "cholesterol": cholesterol,
                    "gluc": gluc,
                    "smoke": smoke,
                    "alco": alco,
                    "active": active,
                })
                st.session_state.cardio_result = result
                st.session_state.cardio_patient_name = patient_name
            except Exception as e:
                st.error(f"Model error: {e}")

    if "cardio_result" in st.session_state:
        result = st.session_state.cardio_result
        patient_name = st.session_state.get("cardio_patient_name", "")

        st.markdown("---")
        if patient_name:
            st.markdown(f"### Hi {patient_name}!! 👋")
        st.markdown("### 📊 Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", result["predicted_class"])
        r2.metric("Probability", f"{result['probability']:.1%}")
        r3.markdown(f"**Risk Level:** {risk_badge(result['risk_level'])}", unsafe_allow_html=True)

        # Probability bar
        st.progress(min(result["probability"], 1.0))

        if result["risk_level"] == "high":
            st.error("⚠️ High cardiovascular risk detected. Please consult a healthcare professional.")
        elif result["risk_level"] == "moderate":
            st.warning("⚡ Moderate risk. Consider lifestyle changes and regular check-ups.")
        else:
            st.success("✅ Low risk. Keep maintaining a healthy lifestyle!")

        # Feedback section
        st.markdown("---")
        with st.expander("📝 Leave Feedback"):
            with st.form("fb_cardio_form"):
                fb_rating = st.slider("Rate this report (1-10)", 1, 10, 8, key="fb_cardio_rating")
                fb_comment = st.text_area("Comments (optional)", key="fb_cardio_comment")
                fb_submitted = st.form_submit_button("Submit Feedback")
            if fb_submitted:
                save_feedback("Cardiovascular", patient_name or "Anonymous", fb_rating, fb_comment)
                st.success("✅ Feedback saved! Thank you.")


# ═══════════════════════════════════════════════════════════════════════════
# 🩸 DIABETES PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🩸 Diabetes":
    st.markdown("""
    <div class="main-header">
        <h1>🩸 Diabetes Risk Prediction</h1>
        <p>XGBoost / LightGBM model trained on CDC BRFSS 2015 data (253K records)</p>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input("👤 Patient Name", placeholder="Enter patient name")

    with st.form("diabetes_form"):
        st.markdown("#### Health Indicators (BRFSS Survey)")

        c1, c2, c3 = st.columns(3)

        with c1:
            HighBP = st.selectbox("High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            HighChol = st.selectbox("High Cholesterol?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            CholCheck = st.selectbox("Cholesterol Check (last 5y)?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            BMI = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.5)
            Smoker = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            Stroke = st.selectbox("Ever had a Stroke?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            HeartDiseaseorAttack = st.selectbox("Heart Disease/Attack?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with c2:
            PhysActivity = st.selectbox("Physical Activity (last 30d)?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            Fruits = st.selectbox("Consume Fruits daily?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            Veggies = st.selectbox("Consume Veggies daily?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            HvyAlcoholConsump = st.selectbox("Heavy Alcohol?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            AnyHealthcare = st.selectbox("Have Healthcare coverage?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            NoDocbcCost = st.selectbox("Couldn't see doctor (cost)?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)

        with c3:
            MentHlth = st.slider("Mental Health (bad days / 30)", 0, 30, 5)
            PhysHlth = st.slider("Physical Health (bad days / 30)", 0, 30, 5)
            DiffWalk = st.selectbox("Difficulty Walking?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            Sex = st.selectbox("Sex", [0, 1], format_func=lambda x: {0: "Female", 1: "Male"}[x])
            Age = st.slider("Age Category (1-13)", 1, 13, 9,
                            help="1=18-24, 2=25-29, ..., 9=60-64, ..., 13=80+")
            Education = st.slider("Education Level (1-6)", 1, 6, 5,
                                  help="1=Never attended, 6=College graduate")
            Income = st.slider("Income Level (1-8)", 1, 8, 6,
                               help="1=<$10k, 8=$75k+")

        submitted = st.form_submit_button("🔍 Predict Risk", width="stretch")

    if submitted:
        with st.spinner("Running diabetes model..."):
            try:
                from inference.predict_diabetes import predict_diabetes
                result = predict_diabetes({
                    "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck,
                    "BMI": BMI, "Smoker": Smoker, "Stroke": Stroke,
                    "HeartDiseaseorAttack": HeartDiseaseorAttack,
                    "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
                    "HvyAlcoholConsump": HvyAlcoholConsump, "AnyHealthcare": AnyHealthcare,
                    "NoDocbcCost": NoDocbcCost, "GenHlth": GenHlth,
                    "MentHlth": MentHlth, "PhysHlth": PhysHlth, "DiffWalk": DiffWalk,
                    "Sex": Sex, "Age": Age, "Education": Education, "Income": Income,
                })
                st.session_state.diabetes_result = result
                st.session_state.diabetes_patient_name = patient_name
            except Exception as e:
                st.error(f"Model error: {e}")

    if "diabetes_result" in st.session_state:
        result = st.session_state.diabetes_result
        patient_name = st.session_state.get("diabetes_patient_name", "")

        st.markdown("---")
        if patient_name:
            st.markdown(f"### Hi {patient_name}!! 👋")
        st.markdown("### 📊 Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", result["predicted_class"])
        r2.metric("Confidence", f"{result['confidence']:.1%}")
        r3.markdown(f"**Risk Level:** {risk_badge(result['risk_level'])}", unsafe_allow_html=True)

        # Class probabilities
        st.markdown("#### Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": list(result["probabilities"].keys()),
            "Probability": list(result["probabilities"].values()),
        })
        st.bar_chart(prob_df.set_index("Class"))

        if result["risk_level"] == "high":
            st.error("⚠️ High diabetes risk detected. Please consult a healthcare professional.")
        elif result["risk_level"] == "moderate":
            st.warning("⚡ Moderate risk. Consider dietary changes and regular screening.")
        else:
            st.success("✅ Low risk. Keep maintaining healthy habits!")

        # Feedback section
        st.markdown("---")
        with st.expander("📝 Leave Feedback"):
            with st.form("fb_diabetes_form"):
                fb_rating = st.slider("Rate this report (1-10)", 1, 10, 8, key="fb_diabetes_rating")
                fb_comment = st.text_area("Comments (optional)", key="fb_diabetes_comment")
                fb_submitted = st.form_submit_button("Submit Feedback")
            if fb_submitted:
                save_feedback("Diabetes", patient_name or "Anonymous", fb_rating, fb_comment)
                st.success("✅ Feedback saved! Thank you.")


# ═══════════════════════════════════════════════════════════════════════════
# 🫁 PNEUMONIA PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🫁 Pneumonia":
    st.markdown("""
    <div class="main-header">
        <h1>🫁 Pneumonia Detection from Chest X-Ray</h1>
        <p>CNN (ResNet / EfficientNet) trained on labeled chest X-ray images</p>
    </div>
    """, unsafe_allow_html=True)

    patient_name = st.text_input("👤 Patient Name", placeholder="Enter patient name")

    uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"], key="xray")

    if uploaded:
        from PIL import Image
        img = Image.open(uploaded)

        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(img, caption="Uploaded X-Ray", width="stretch")

        with col_res:
            if "pneumonia_result" not in st.session_state or st.session_state.get("pneumonia_uploaded_name") != uploaded.name:
                with st.spinner("Analyzing X-ray..."):
                    try:
                        # Save temp file for the model
                        temp_path = os.path.join(BASE_DIR, "temp_xray.png")
                        img.save(temp_path)

                        from inference.predict_pneumonia import predict_pneumonia
                        result = predict_pneumonia(temp_path)

                        os.remove(temp_path)
                        st.session_state.pneumonia_result = result
                        st.session_state.pneumonia_uploaded_name = uploaded.name
                        st.session_state.pneumonia_patient_name = patient_name
                    except Exception as e:
                        st.error(f"Model error: {e}")

            if "pneumonia_result" in st.session_state:
                result = st.session_state.pneumonia_result
                pn_name = st.session_state.get("pneumonia_patient_name", "") or patient_name

                if pn_name:
                    st.markdown(f"### Hi {pn_name}!! 👋")
                st.markdown("### 📊 Results")
                st.metric("Prediction", result["predicted_class"])
                st.metric("Pneumonia Probability", f"{result['probability']:.1%}")
                st.markdown(f"**Risk Level:** {risk_badge(result['risk_level'])}", unsafe_allow_html=True)
                st.caption(f"Model: {result['model_used']}")

                st.progress(min(result["probability"], 1.0))

                if result["risk_level"] == "high":
                    st.error("⚠️ High probability of pneumonia detected. Please seek medical attention.")
                elif result["risk_level"] == "moderate":
                    st.warning("⚡ Moderate probability. Further examination recommended.")
                else:
                    st.success("✅ X-ray appears normal.")

                # Feedback section
                st.markdown("---")
                with st.expander("📝 Leave Feedback"):
                    with st.form("fb_pneumonia_form"):
                        fb_rating = st.slider("Rate this report (1-10)", 1, 10, 8, key="fb_pneumonia_rating")
                        fb_comment = st.text_area("Comments (optional)", key="fb_pneumonia_comment")
                        fb_submitted = st.form_submit_button("Submit Feedback")
                    if fb_submitted:
                        save_feedback("Pneumonia", pn_name or "Anonymous", fb_rating, fb_comment)
                        st.success("✅ Feedback saved! Thank you.")

    else:
        st.info("👆 Upload a chest X-ray image (JPG/PNG) to get started.")
        st.markdown("#### Sample Images Available")
        st.markdown("You can test with images from the `data/chest_xray/test/` directory.")


# ═══════════════════════════════════════════════════════════════════════════
# 🏥 POST-SURGERY RECOVERY ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏥 Recovery Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Post-Surgery Recovery Assistant</h1>
        <p>LSTM Recovery Forecasting + Generative AI Coach</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs for workflow ──
    tab_profile, tab_vitals, tab_results = st.tabs(["👤 Patient Profile", "📋 Daily Vitals", "📊 Recovery Plan"])

    # Initialize session state for recovery
    if "recovery_patient" not in st.session_state:
        st.session_state.recovery_patient = None
    if "recovery_vitals" not in st.session_state:
        st.session_state.recovery_vitals = []

    # ── TAB 1: Patient Profile ──
    with tab_profile:
        st.markdown("#### Enter patient information (one-time setup)")

        with st.form("patient_profile"):
            pat_name = st.text_input("👤 Patient Name", placeholder="Enter patient name")
            p1, p2 = st.columns(2)
            with p1:
                pat_age = st.number_input("Age", 18, 100, 55)
                pat_gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x else "Female")
                pat_bmi = st.number_input("BMI", 15.0, 60.0, 27.0, step=0.5)
                pat_surgery = st.selectbox("Surgery Type", ["cardiac", "orthopedic", "abdominal", "neurological"])

            with p2:
                pat_diabetes = st.selectbox("Diabetes?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                pat_hypertension = st.selectbox("Hypertension?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                pat_smoking = st.selectbox("Smoking?", [0, 1], format_func=lambda x: "Yes" if x else "No")

            save_profile = st.form_submit_button("💾 Save Profile", width="stretch")

        if save_profile:
            st.session_state.recovery_patient = {
                "name": pat_name, "age": pat_age, "gender": pat_gender, "bmi": pat_bmi,
                "diabetes": pat_diabetes, "hypertension": pat_hypertension,
                "smoking": pat_smoking, "surgery_type": pat_surgery,
            }
            st.session_state.recovery_vitals = []
            st.success(f"✅ Profile saved! Surgery: {pat_surgery.title()}, Age: {pat_age}")

        if st.session_state.recovery_patient:
            p = st.session_state.recovery_patient
            st.markdown(f"""
            <div class="metric-card">
                <strong>Current Patient:</strong> {p['age']}y {'Male' if p['gender'] else 'Female'},
                BMI {p['bmi']}, {p['surgery_type'].title()} surgery<br/>
                Comorbidities: {'Diabetes ' if p['diabetes'] else ''}{'Hypertension ' if p['hypertension'] else ''}{'Smoking' if p['smoking'] else ''}
                {'None' if not (p['diabetes'] or p['hypertension'] or p['smoking']) else ''}
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Daily Vitals ──
    with tab_vitals:
        if not st.session_state.recovery_patient:
            st.warning("⚠️ Please fill in the Patient Profile first.")
        else:
            st.markdown(f"#### Day {len(st.session_state.recovery_vitals) + 1} Vitals Entry")

            with st.form("vitals_form"):
                v1, v2, v3 = st.columns(3)
                with v1:
                    pain = st.slider("Pain Level (0-10)", 0.0, 10.0, 5.0, 0.5)
                    temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
                    hr = st.number_input("Heart Rate (bpm)", 40, 180, 80)

                with v2:
                    bp_sys = st.number_input("BP Systolic (mmHg)", 70, 220, 120)
                    mobility = st.slider("Mobility Score (0-10)", 0.0, 10.0, 5.0, 0.5)
                    wound = st.slider("Wound Status (0=healed, 1=bad)", 0.0, 1.0, 0.5, 0.05)

                with v3:
                    sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
                    recovery = st.slider("Recovery Score (0-100)", 0, 100, 50)

                add_vitals = st.form_submit_button("➕ Add Day's Vitals", width="stretch")

            if add_vitals:
                st.session_state.recovery_vitals.append({
                    "pain_level": pain, "temperature": temp, "heart_rate": hr,
                    "bp_systolic": bp_sys, "mobility_score": mobility,
                    "wound_status": wound, "sleep_hours": sleep,
                    "recovery_score": recovery,
                })
                st.success(f"✅ Day {len(st.session_state.recovery_vitals)} vitals recorded.")

            # Show vitals history
            if st.session_state.recovery_vitals:
                st.markdown("#### 📈 Vitals History")
                vitals_df = pd.DataFrame(st.session_state.recovery_vitals)
                vitals_df.index = [f"Day {i+1}" for i in range(len(vitals_df))]
                st.dataframe(vitals_df, width="stretch")

                # Quick charts
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.line_chart(vitals_df[["pain_level", "mobility_score", "recovery_score"]])
                with chart_col2:
                    st.line_chart(vitals_df[["temperature", "heart_rate"]])

                st.info(f"📊 {len(st.session_state.recovery_vitals)} day(s) recorded. "
                        f"Need at least 7 days for LSTM prediction.")

                if st.button("🗑️ Clear All Vitals"):
                    st.session_state.recovery_vitals = []
                    st.rerun()

            # Quick-fill demo data
            st.markdown("---")
            demo_col1, demo_col2 = st.columns(2)
            with demo_col1:
                if st.button("📋 Load Recovering Patient (7 days)", width="stretch"):
                    st.session_state.recovery_vitals = [
                        {"pain_level": 7.0, "temperature": 37.8, "heart_rate": 92, "bp_systolic": 135, "mobility_score": 2.5, "wound_status": 0.8, "sleep_hours": 4.5, "recovery_score": 30},
                        {"pain_level": 6.5, "temperature": 37.6, "heart_rate": 90, "bp_systolic": 132, "mobility_score": 3.0, "wound_status": 0.75, "sleep_hours": 5.0, "recovery_score": 35},
                        {"pain_level": 6.0, "temperature": 37.5, "heart_rate": 88, "bp_systolic": 130, "mobility_score": 3.5, "wound_status": 0.7, "sleep_hours": 5.5, "recovery_score": 40},
                        {"pain_level": 5.5, "temperature": 37.3, "heart_rate": 85, "bp_systolic": 128, "mobility_score": 4.0, "wound_status": 0.6, "sleep_hours": 6.0, "recovery_score": 45},
                        {"pain_level": 5.0, "temperature": 37.2, "heart_rate": 84, "bp_systolic": 126, "mobility_score": 4.5, "wound_status": 0.55, "sleep_hours": 6.5, "recovery_score": 50},
                        {"pain_level": 4.5, "temperature": 37.1, "heart_rate": 82, "bp_systolic": 124, "mobility_score": 5.0, "wound_status": 0.5, "sleep_hours": 7.0, "recovery_score": 55},
                        {"pain_level": 4.0, "temperature": 37.0, "heart_rate": 80, "bp_systolic": 122, "mobility_score": 5.5, "wound_status": 0.45, "sleep_hours": 7.0, "recovery_score": 60},
                    ]
                    st.rerun()
            with demo_col2:
                if st.button("⚠️ Load Deteriorating Patient (7 days)", width="stretch"):
                    st.session_state.recovery_vitals = [
                        {"pain_level": 8.0, "temperature": 38.5, "heart_rate": 100, "bp_systolic": 150, "mobility_score": 1.5, "wound_status": 0.90, "sleep_hours": 3.0, "recovery_score": 20},
                        {"pain_level": 8.5, "temperature": 38.8, "heart_rate": 105, "bp_systolic": 155, "mobility_score": 1.2, "wound_status": 0.92, "sleep_hours": 2.5, "recovery_score": 18},
                        {"pain_level": 9.0, "temperature": 39.0, "heart_rate": 110, "bp_systolic": 160, "mobility_score": 1.0, "wound_status": 0.95, "sleep_hours": 2.0, "recovery_score": 15},
                        {"pain_level": 8.8, "temperature": 39.2, "heart_rate": 112, "bp_systolic": 158, "mobility_score": 0.8, "wound_status": 0.97, "sleep_hours": 2.0, "recovery_score": 12},
                        {"pain_level": 9.2, "temperature": 39.5, "heart_rate": 115, "bp_systolic": 162, "mobility_score": 0.5, "wound_status": 0.98, "sleep_hours": 1.5, "recovery_score": 10},
                        {"pain_level": 9.5, "temperature": 39.8, "heart_rate": 118, "bp_systolic": 165, "mobility_score": 0.3, "wound_status": 0.99, "sleep_hours": 1.0, "recovery_score": 8},
                        {"pain_level": 9.8, "temperature": 40.0, "heart_rate": 120, "bp_systolic": 170, "mobility_score": 0.2, "wound_status": 1.0, "sleep_hours": 0.5, "recovery_score": 5},
                    ]
                    # Also set a high-risk patient profile for this demo
                    st.session_state.recovery_patient = {
                        "name": "Demo Patient", "age": 65, "gender": 1, "bmi": 35.0,
                        "diabetes": 1, "hypertension": 1,
                        "smoking": 1, "surgery_type": "cardiac",
                    }
                    st.rerun()

    # ── TAB 3: Recovery Plan ──
    with tab_results:
        if not st.session_state.recovery_patient:
            st.warning("⚠️ Please fill in the Patient Profile first.")
        elif len(st.session_state.recovery_vitals) < 7:
            st.warning(f"⚠️ Need at least 7 days of vitals. Currently have {len(st.session_state.recovery_vitals)} day(s).")
            st.info("💡 Go to **Daily Vitals** tab to add data, or click **Load 7-Day Demo Data**.")
        else:
            # Optional API key
            with st.expander("🔑 AI Coach API Key (optional)"):
                api_provider = st.selectbox("Provider", ["offline", "gemini", "openai"])
                api_key = st.text_input("API Key", type="password",
                                        help="Leave empty for offline rule-based coach")

            if st.button("🚀 Generate Recovery Plan", width="stretch"):
                with st.spinner("Running LSTM model and generating recovery plan..."):
                    try:
                        # Set env vars for the coach if provided
                        if api_key and api_provider == "gemini":
                            os.environ["GEMINI_API_KEY"] = api_key
                        elif api_key and api_provider == "openai":
                            os.environ["OPENAI_API_KEY"] = api_key

                        from inference.predict_recovery import predict_recovery
                        result = predict_recovery(
                            st.session_state.recovery_vitals,
                            st.session_state.recovery_patient,
                        )

                        # Clean up env vars
                        os.environ.pop("GEMINI_API_KEY", None)
                        os.environ.pop("OPENAI_API_KEY", None)

                        st.session_state.recovery_result = result

                    except FileNotFoundError:
                        st.error("❌ Recovery model not found. Please train first:\n"
                                 "1. `python data/generate_synthetic_recovery.py`\n"
                                 "2. `python training/train_recovery_lstm.py`")
                    except Exception as e:
                        st.error(f"Error: {e}")

            if "recovery_result" in st.session_state:
                result = st.session_state.recovery_result

                # ── Display Summary ──
                st.markdown("---")
                rec_name = st.session_state.recovery_patient.get("name", "")
                if rec_name:
                    st.markdown(f"### Hi {rec_name}!! 👋")
                st.markdown("### 📊 Recovery Summary")
                summary = result["patient_summary"]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recovery Day", summary["day"])
                m2.metric("Recovery Score", f"{summary['current_recovery_score']}/100")
                m3.metric("Complication Risk", summary["complication_risk"])
                m4.markdown(
                    f"**Risk Level:** {risk_badge(summary['risk_level'])}",
                    unsafe_allow_html=True,
                )

                # Trend
                trend_icons = {"improving": "📈", "declining": "📉", "stable": "➡️"}
                st.markdown(f"**Trend:** {trend_icons.get(summary['trend'], '')} {summary['trend'].title()}")

                # 3-day forecast chart
                st.markdown("#### 🔮 3-Day Recovery Forecast")
                forecast = summary["recovery_forecast_3day"]
                forecast_df = pd.DataFrame({
                    "Day": [f"Day +{i+1}" for i in range(len(forecast))],
                    "Predicted Recovery Score": forecast,
                })
                st.bar_chart(forecast_df.set_index("Day"))

                # ── AI Coach Advice ──
                st.markdown("---")
                st.markdown("### 🤖 AI Recovery Coach Advice")
                st.markdown(result["advice"], unsafe_allow_html=True)

                # Feedback section
                st.markdown("---")
                with st.expander("📝 Leave Feedback"):
                    rec_fb_name = st.session_state.recovery_patient.get("name", "Anonymous")
                    with st.form("fb_recovery_form"):
                        fb_rating = st.slider("Rate this report (1-10)", 1, 10, 8, key="fb_recovery_rating")
                        fb_comment = st.text_area("Comments (optional)", key="fb_recovery_comment")
                        fb_submitted = st.form_submit_button("Submit Feedback")
                    if fb_submitted:
                        save_feedback("Recovery", rec_fb_name or "Anonymous", fb_rating, fb_comment)
                        st.success("✅ Feedback saved! Thank you.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📝 Feedback":
    st.markdown("""
    <div class="main-header">
        <h1>📝 Submit Feedback</h1>
        <p>Share your experience with our AI Health modules</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("feedback_page_form"):
        fb_name = st.text_input("👤 Your Name", placeholder="Enter your name")
        fb_module = st.selectbox("📚 Module", ["Cardiovascular", "Diabetes", "Pneumonia", "Recovery"])
        fb_rating = st.slider("⭐ Rating (1-10)", 1, 10, 8)
        fb_comment = st.text_area("💬 Comments", placeholder="Share your suggestions...")
        fb_submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

    if fb_submitted:
        if fb_name.strip():
            save_feedback(fb_module, fb_name.strip(), fb_rating, fb_comment)
            st.success("✅ Feedback submitted! Thank you.")
        else:
            st.warning("Please enter your name.")
