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
    layout="centered"
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

if "uploaded_preview" not in st.session_state:
    st.session_state.uploaded_preview = None

if "error_message" not in st.session_state:
    st.session_state.error_message = None

# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .center-title {
        text-align: center;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2.6rem;
    }

    .section-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 750;
        margin-top: 0.3rem;
        margin-bottom: 1rem;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1.02rem;
        line-height: 1.7;
        opacity: 0.88;
        margin-bottom: 0.4rem;
    }

    .glass-card {
        border-radius: 24px;
        padding: 26px 24px;
        margin-bottom: 22px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.14);
    }

    .subtle-text {
        text-align: center;
        font-size: 0.96rem;
        opacity: 0.78;
        margin-bottom: 0.7rem;
    }

    .preview-caption {
        text-align: center;
        font-size: 0.95rem;
        opacity: 0.75;
        margin-top: 0.35rem;
    }

    .result-card {
        border-radius: 22px;
        padding: 24px;
        margin-bottom: 18px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.14);
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.9rem;
    }

    .result-line {
        font-size: 1.05rem;
        margin-bottom: 0.7rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.36rem 0.85rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-left: 0.35rem;
    }

    .mini-card {
        border-radius: 18px;
        padding: 18px;
        min-height: 115px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .mini-label {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-bottom: 0.45rem;
    }

    .mini-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .footer-note {
        text-align: center;
        font-size: 0.95rem;
        opacity: 0.72;
        margin-top: 2rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 16px;
        padding: 0.85rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    /* LIGHT MODE */
    @media (prefers-color-scheme: light) {
        .glass-card, .result-card, .mini-card {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid rgba(15, 23, 42, 0.08);
        }

        .hero-title, .section-title, .result-title {
            color: #0f172a;
        }

        .hero-subtitle, .subtle-text, .preview-caption, .footer-note,
        .mini-label, .result-line {
            color: #334155;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.22);
        }
    }

    /* DARK MODE */
    @media (prefers-color-scheme: dark) {
        .glass-card, .result-card, .mini-card {
            background: rgba(15, 23, 42, 0.70);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }

        .hero-title, .section-title, .result-title,
        .hero-subtitle, .subtle-text, .preview-caption, .footer-note,
        .mini-label, .result-line {
            color: #e5e7eb;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.22);
        }
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
    <div class="glass-card">
        <div class="center-title hero-title">🩺 Skin Cancer Screening System</div>
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
st.markdown('<div class="center-title section-title">Upload Image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle-text">Supported formats: JPG, JPEG, PNG</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtle-text">On free hosting, the backend may take around 30–60 seconds to wake up on the first request.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

analyze = False
reset = False

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_preview = image

    action_left, action_center, action_right = st.columns([1.2, 2.2, 1.2], gap="large")

    with action_left:
        analyze = st.button("Analyze Image", use_container_width=True)

    with action_center:
        st.image(
            image,
            width=300
        )
        st.markdown('<div class="preview-caption">Image Preview</div>', unsafe_allow_html=True)

    with action_right:
        reset = st.button("Reset", use_container_width=True)

if reset:
    st.session_state.prediction_result = None
    st.session_state.uploaded_preview = None
    st.session_state.error_message = None
    st.rerun()

# =========================
# Analyze logic with AI scan effect
# =========================
if analyze and uploaded_file is not None:
    st.session_state.error_message = None
    st.session_state.prediction_result = None

    progress_text = st.empty()
    progress_bar = st.progress(0)

    scan_steps = [
        "Initializing AI scanner...",
        "Preprocessing uploaded image...",
        "Extracting lesion features...",
        "Running classification model...",
        "Preparing screening result..."
    ]

    try:
        for i, text in enumerate(scan_steps, start=1):
            progress_text.info(text)
            progress_bar.progress(i * 15)
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

        progress_bar.progress(100)
        progress_text.success("Analysis completed successfully.")

        if response.status_code == 200:
            st.session_state.prediction_result = response.json()
        else:
            st.session_state.prediction_result = None
            st.session_state.error_message = f"API Error: {response.status_code}"

    except requests.exceptions.Timeout:
        st.session_state.prediction_result = None
        st.session_state.error_message = (
            "The backend is likely waking up from cold start. "
            "Please wait about 30–60 seconds and click Analyze Image again."
        )
    except requests.exceptions.ConnectionError:
        st.session_state.prediction_result = None
        st.session_state.error_message = (
            "Could not connect to backend. On free hosting, the backend may be asleep. "
            "Please wait a bit and try again."
        )
    except Exception as e:
        st.session_state.prediction_result = None
        st.session_state.error_message = f"Something went wrong: {e}"

# =========================
# Error message
# =========================
if st.session_state.error_message:
    st.warning(st.session_state.error_message)

# =========================
# Section 3: Result
# =========================
result = st.session_state.prediction_result

if result is not None:
    st.write("")
    st.markdown('<div class="center-title section-title">Screening Result</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="center-title" style="font-size:1.6rem; margin-bottom:1rem;">Detailed Probabilities</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")

    with p1:
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

    with p2:
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

    with st.expander("Technical Output"):
        result_text = f"""===== Prediction Result =====

Benign Probability: {benign_prob:.4f}
Malignant Probability: {malignant_prob:.4f}
Predicted Class: {predicted_class.lower()}
Risk Level: {risk_level}

Recommendation: {recommendation}"""

        st.code(result_text, language=None)

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