import streamlit as st
import requests
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
# Styling
# =========================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #0f172a 50%, #1e293b 100%);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 24px;
        padding: 26px 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.28);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
    }

    .panel {
        background: rgba(15, 23, 42, 0.94);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.22);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        margin-bottom: 0.8rem;
    }

    .muted-text {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    .result-card {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border: 1px solid rgba(148,163,184,0.16);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }

    .result-line {
        font-size: 1.05rem;
        margin-bottom: 0.7rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-left: 0.4rem;
    }

    .mini-card {
        background: rgba(17, 24, 39, 0.92);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 18px;
        text-align: left;
        min-height: 110px;
    }

    .mini-label {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }

    .mini-value {
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .footer-note {
        text-align: center;
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 2rem;
    }

    .stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.9rem;
    font-weight: 700;
    font-size: 1rem;
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Session state
# =========================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "uploaded_preview" not in st.session_state:
    st.session_state.uploaded_preview = None

# =========================
# Hero section
# =========================
st.markdown(
    """
    <div class="hero">
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

# =========================
# Upload section
# =========================
with st.container():
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted-text">Supported formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

button_left, button_spacer, button_right = st.columns([1, 3, 1])

with button_left:
    analyze = st.button("Analyze Image")

with button_right:
    reset = st.button("Reset")

if reset:
    st.session_state.prediction_result = None
    st.session_state.uploaded_preview = None
    st.rerun()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_preview = image

    preview_left, preview_center, preview_right = st.columns([1, 2, 1])

    with preview_center:
        st.image(
            image,
            caption="Image Preview",
            width=300
        )

if analyze and uploaded_file is not None:
    with st.spinner("Analyzing image..."):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files,
                timeout=60
            )

            if response.status_code == 200:
                st.session_state.prediction_result = response.json()
                st.success("Analysis completed successfully.")
            else:
                st.error(f"API Error: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Please make sure the FastAPI server is running.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Result section
# =========================
result = st.session_state.prediction_result

if result is not None:
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

    st.markdown("## Screening Result")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title">{predicted_class}</div>
            <div class="result-line">
                <strong>Prediction Confidence:</strong> {format_percent(predicted_probability)}
            </div>
            <div class="result-line">
                <strong>Risk Level:</strong>
                <span class="badge" style="background:{badge_bg}; color:{risk_color}; border:1px solid {risk_color};">
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

    st.markdown("### Detailed Probabilities")
    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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