import os
import time
import requests
import streamlit as st
from PIL import Image

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Skin Cancer Screening",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Backend URL
# =========================
BACKEND_URL = os.getenv("BACKEND_URL", "").strip()
if not BACKEND_URL:
    try:
        BACKEND_URL = st.secrets["BACKEND_URL"].strip()
    except Exception:
        BACKEND_URL = "http://127.0.0.1:8000"

# =========================
# Helpers
# =========================
def format_percent(prob: float) -> str:
    percent = prob * 100
    if percent >= 99.995:
        return "99.99%+"
    if percent <= 0.005:
        return "<0.01%"
    return f"{percent:.2f}%"

def get_risk_color(risk_level: str) -> str:
    if risk_level == "Low Risk":
        return "#22c55e"
    elif risk_level == "Suspicious":
        return "#f59e0b"
    return "#ef4444"

# =========================
# Session state
# =========================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "show_result" not in st.session_state:
    st.session_state.show_result = False

# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    :root {
        --radius-xl: 28px;
        --radius-lg: 22px;
        --radius-md: 18px;
        --shadow: 0 18px 40px rgba(0, 0, 0, 0.12);
    }

    @media (prefers-color-scheme: light) {
        :root {
            --card-bg: rgba(255, 255, 255, 0.62);
            --card-border: rgba(15, 23, 42, 0.08);
            --title: #0f172a;
            --text: #334155;
            --muted: #64748b;
            --soft: #f8fafc;
            --accent: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            --hero: linear-gradient(135deg, rgba(255,255,255,0.78) 0%, rgba(241,245,249,0.72) 100%);
            --tech: rgba(255,255,255,0.72);
        }
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: rgba(15, 23, 42, 0.66);
            --card-border: rgba(148, 163, 184, 0.14);
            --title: #f8fafc;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --soft: #0f172a;
            --accent: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            --hero: linear-gradient(135deg, rgba(15,23,42,0.82) 0%, rgba(17,24,39,0.76) 100%);
            --tech: rgba(17,24,39,0.85);
        }
    }

    .hero-card {
        background: var(--hero);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-xl);
        padding: 32px 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: var(--shadow);
        text-align: center;
        margin-bottom: 22px;
    }

    .hero-title {
        color: var(--title);
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 10px;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        color: var(--text);
        font-size: 1.08rem;
        line-height: 1.75;
        max-width: 850px;
        margin: 0 auto;
    }

    .section-heading {
        color: var(--title);
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 10px;
        margin-bottom: 10px;
        letter-spacing: -0.02em;
    }

    .section-subtext {
        color: var(--muted);
        text-align: center;
        font-size: 1rem;
        margin-bottom: 10px;
    }

    .main-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-xl);
        padding: 28px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: var(--shadow);
        margin-bottom: 22px;
    }

    .result-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-xl);
        padding: 30px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: var(--shadow);
        margin-bottom: 20px;
    }

    .result-title {
        color: var(--title);
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 14px;
        letter-spacing: -0.02em;
    }

    .result-line {
        color: var(--text);
        font-size: 1.08rem;
        margin-bottom: 12px;
        line-height: 1.7;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.38rem 0.9rem;
        border-radius: 999px;
        font-size: 0.92rem;
        font-weight: 800;
        margin-left: 0.35rem;
        vertical-align: middle;
    }

    .mini-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-lg);
        padding: 22px;
        min-height: 132px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: var(--shadow);
    }

    .mini-label {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 10px;
    }

    .mini-value {
        color: var(--title);
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.15;
    }

    .tech-card {
        background: var(--tech);
        border: 1px solid var(--card-border);
        border-radius: var(--radius-lg);
        padding: 22px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: var(--shadow);
        margin-top: 12px;
    }

    .tech-title {
        color: var(--title);
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .tech-code {
        color: var(--text);
        font-family: Consolas, Monaco, monospace;
        font-size: 15px;
        line-height: 1.9;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    .preview-caption {
        color: var(--muted);
        text-align: center;
        font-size: 0.95rem;
        margin-top: 10px;
    }

    .scan-box {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 18px;
        box-shadow: var(--shadow);
        margin-top: 14px;
        margin-bottom: 10px;
    }

    .footer-note {
        color: var(--muted);
        text-align: center;
        font-size: 0.95rem;
        margin-top: 26px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        font-weight: 800;
        font-size: 1rem;
        border: none;
        background: var(--accent);
        color: white;
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.24);
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 34px rgba(37, 99, 235, 0.28);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    div[data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.14);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Section 1: Header
# =========================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🩺 Skin Cancer Screening System</div>
        <div class="hero-subtitle">
            Upload a skin lesion image and receive an AI-based screening result with
            prediction confidence, risk level, and recommendation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This tool is for educational and screening purposes only. "
    "It is not a confirmed medical diagnosis."
)

st.write("")

# =========================
# Section 2: Upload
# =========================
st.markdown('<div class="section-heading">Upload Image</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">Supported formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtext">On free hosting, the backend may take around 30–60 seconds to wake up on the first request.</div>', unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

analyze = False
reset = False

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Equal-distance 3-part layout
    left_col, center_col, right_col = st.columns([1.2, 1.7, 1.2], gap="large")

    with left_col:
        st.write("")
        st.write("")
        analyze = st.button("Analyze Image", use_container_width=True)

    with center_col:
        st.image(image, width=320)
        st.markdown('<div class="preview-caption">Image Preview</div>', unsafe_allow_html=True)

    with right_col:
        st.write("")
        st.write("")
        reset = st.button("Reset", use_container_width=True)

else:
    analyze = False
    reset = False

if reset:
    st.session_state.prediction_result = None
    st.session_state.error_message = None
    st.session_state.show_result = False
    st.rerun()

# =========================
# Analyze logic + AI scan effect
# =========================
if analyze and uploaded_file is not None:
    st.session_state.error_message = None
    st.session_state.prediction_result = None
    st.session_state.show_result = False

    scan_holder = st.empty()
    progress_holder = st.empty()

    try:
        scan_steps = [
            "Initializing AI scanner...",
            "Preprocessing uploaded lesion image...",
            "Extracting visual patterns...",
            "Running classification model...",
            "Generating screening report..."
        ]

        for i, step in enumerate(scan_steps, start=1):
            scan_holder.markdown(
                f'<div class="scan-box"><strong>{step}</strong></div>',
                unsafe_allow_html=True
            )
            progress_holder.progress(i * 18)
            time.sleep(0.35)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            timeout=180
        )

        progress_holder.progress(100)

        if response.status_code == 200:
            st.session_state.prediction_result = response.json()
            st.session_state.show_result = True
            scan_holder.markdown(
                '<div class="scan-box"><strong>Analysis completed successfully.</strong></div>',
                unsafe_allow_html=True
            )
        else:
            st.session_state.prediction_result = None
            st.session_state.error_message = f"API Error: {response.status_code}"
            st.session_state.show_result = False

    except requests.exceptions.Timeout:
        st.session_state.prediction_result = None
        st.session_state.error_message = (
            "The backend is likely waking up from cold start. "
            "Please wait about 30–60 seconds and click Analyze Image again."
        )
        st.session_state.show_result = False

    except requests.exceptions.ConnectionError:
        st.session_state.prediction_result = None
        st.session_state.error_message = (
            "Could not connect to backend. On free hosting, the backend may be asleep. "
            "Please wait a bit and try again."
        )
        st.session_state.show_result = False

    except Exception as e:
        st.session_state.prediction_result = None
        st.session_state.error_message = f"Something went wrong: {e}"
        st.session_state.show_result = False

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Error
# =========================
if st.session_state.error_message:
    st.warning(st.session_state.error_message)

# =========================
# Section 3: Screening Result
# =========================
if st.session_state.show_result and st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result

    predicted_class = result["predicted_class"]
    predicted_probability = result["predicted_probability"]
    benign_prob = result["benign_probability"]
    malignant_prob = result["malignant_probability"]
    risk_level = result["risk_level"]
    recommendation = result["recommendation"]

    risk_color = get_risk_color(risk_level)
    badge_bg = {
        "Low Risk": "rgba(34,197,94,0.16)",
        "Suspicious": "rgba(245,158,11,0.16)",
        "High Risk": "rgba(239,68,68,0.16)"
    }.get(risk_level, "rgba(148,163,184,0.16)")

    st.write("")
    st.markdown('<div class="section-heading">Screening Result</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{predicted_class}</div>
            <div class="result-line">
                <strong>Prediction Confidence:</strong> {format_percent(predicted_probability)}
            </div>
            <div class="result-line">
                <strong>Risk Level:</strong>
                <span class="risk-badge" style="background:{badge_bg}; color:{risk_color}; border:1px solid {risk_color};">
                    {risk_level}
                </span>
            </div>
            <div class="result-line" style="margin-top:1rem;">
                <strong>Recommendation:</strong> {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading" style="font-size:1.8rem;">Detailed Probabilities</div>', unsafe_allow_html=True)

    prob_col1, prob_col2 = st.columns(2, gap="large")

    with prob_col1:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">Benign Probability</div>
                <div class="mini-value">{format_percent(benign_prob)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(benign_prob))

    with prob_col2:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">Malignant Probability</div>
                <div class="mini-value">{format_percent(malignant_prob)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(float(malignant_prob))

    # Technical Output always visible
    result_text = f"""===== Prediction Result =====

Benign Probability: {benign_prob:.4f}
Malignant Probability: {malignant_prob:.4f}
Predicted Class: {predicted_class.lower()}
Risk Level: {risk_level}

Recommendation: {recommendation}"""

    st.markdown(
        f"""
        <div class="tech-card">
            <div class="tech-title">Technical Output</div>
            <div class="tech-code">{result_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Footer
# =========================
st.markdown(
    """
    <div class="footer-note">
        Built for AI-assisted skin lesion screening research and demonstration.
    </div>
    """,
    unsafe_allow_html=True
)