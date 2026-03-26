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
# Title and description
# =========================
st.title("🩺 Skin Cancer Screening System")
st.write(
    "Upload a skin lesion image to get an AI-based screening result."
)

st.info(
    "This tool is for educational and screening purposes only. "
    "It is not a confirmed medical diagnosis."
)

# =========================
# Upload image
# =========================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction completed!")

                    st.subheader("Result")
                    st.write(f"**Prediction:** {result['prediction'].capitalize()}")
                    st.write(f"**Risk Level:** {result['risk_level']}")

                    st.subheader("Probabilities")
                    st.write(f"**Benign Probability:** {result['benign_probability']}")
                    st.write(f"**Malignant Probability:** {result['malignant_probability']}")

                    st.subheader("Recommendation")
                    st.write(result["recommendation"])

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Please make sure FastAPI server is running.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")