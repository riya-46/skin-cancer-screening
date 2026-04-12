# Skin Cancer Detection (Benign vs Malignant)

## 📌 Project Overview
This project focuses on detecting skin cancer using deep learning (DenseNet121).  
It classifies images into:
- Benign
- Malignant

Additionally, the system is designed as a **risk-based screening tool** rather than a simple classifier.

---

## 🧠 Model Details
- Model: DenseNet121 (Transfer Learning)
- Framework: PyTorch
- Image Size: 224x224
- Loss: Weighted CrossEntropy
- Optimization: AdamW

---

## 📊 Dataset
Dataset used:
https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

- Train: 2637 images
- Test: 660 images
- Classes:
  - Benign
  - Malignant

---

## 📈 Results

- Accuracy: ~88-90%
- Malignant Recall: ~0.92+
- F1 Score: ~0.88

⚠️ Model is optimized for **high recall** (to avoid missing cancer cases)

---

## 🧪 Threshold-based Prediction

Instead of direct classification, model outputs risk:

- Low Risk
- Suspicious
- High Risk

---

## 🚀 Future Improvements
- Threshold tuning
- Clinical feature integration
- Grad-CAM explainability
- Better architectures (EfficientNet)

---

## ⚠️ Disclaimer
This project is for educational purposes only and not a medical diagnostic tool.

## Deployment

### Option 1: Render for backend + frontend
- This repo now includes [render.yaml](./render.yaml).
- Push the repo to GitHub.
- In Render, create a new Blueprint from the repository.
- Render will create:
  - `skin-cancer-backend`
  - `skin-cancer-frontend`

### Option 2: Render backend + Streamlit Community Cloud frontend
- Deploy the FastAPI backend on Render with:
  - Build command: `pip install -r requirements.txt`
  - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Deploy the frontend on Streamlit Community Cloud from `frontend/app.py`
- In Streamlit secrets, set:
  - `BACKEND_URL = "https://your-backend-service.onrender.com"`
- A sample secrets file is included at [`.streamlit/secrets.toml.example`](./.streamlit/secrets.toml.example).
