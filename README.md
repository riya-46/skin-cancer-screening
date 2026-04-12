# Skin Cancer Screening System

A Streamlit + FastAPI project for AI-assisted skin lesion screening using a DenseNet121-based PyTorch model.

The system is built for **close-up skin lesion images**, not for full-body dermatology diagnosis. It supports three outcomes:

- `Benign`
- `Malignant`
- `Invalid`

`Invalid` is used for non-lesion or unsuitable uploads such as full-arm rash photos, body-part images, blurry captures, or unrelated images.

## What The Project Does

- Accepts a skin image from the user
- Sends it to a FastAPI backend for inference
- Returns a screening summary with:
  - predicted class
  - confidence
  - risk level
  - recommendation
  - detailed probabilities
- Supports a curated **sample folder** for demo use

## Project Scope

This is a **skin lesion screening** project.

It is designed for:
- close-up lesion photos
- binary lesion risk assessment with invalid-image rejection

It is not designed for:
- general rash diagnosis
- full-body dermatology analysis
- clinical diagnosis

## Model

- Architecture: `DenseNet121`
- Framework: `PyTorch`
- Backend inference: `FastAPI`
- Frontend: `Streamlit`

The training pipeline supports:
- `benign`
- `malignant`
- `invalid`

## Dataset

Base lesion dataset source:
- https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

Additional real-world robustness was added through an `invalid` class containing non-lesion or unsuitable images.

## Current Behavior

The model is tuned as a **screening system**, so it prioritizes safety and may still produce false positives on difficult benign images. The `invalid` class helps reject clearly unsuitable uploads.

In practical terms:
- it works best on lesion-focused close-up images
- it may reject broad rash or body-part photos as `Invalid`
- it should be demonstrated as a screening aid, not as a medical diagnostic tool

## Demo Sample Images

For project showcase, curated demo images are available in:

- [sample_images/benign](./sample_images/benign)
- [sample_images/malignant](./sample_images/malignant)
- [sample_images/invalid](./sample_images/invalid)

The frontend includes a `Choose from sample folder` option that loads images only from `sample_images/`.

This is useful when sharing the deployed app with people who do not already have lesion images.

## Local Run

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the backend

```powershell
python -m uvicorn backend.main:app --reload
```

### 3. Run the frontend

```powershell
streamlit run frontend/app.py
```

## Deployment

### Option 1: Render for backend + frontend

This repo includes [render.yaml](./render.yaml).

Steps:
- push the repo to GitHub
- create a new Blueprint in Render
- Render will provision the frontend and backend services

### Option 2: Render backend + Streamlit Community Cloud frontend

Backend on Render:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

Frontend on Streamlit Community Cloud:
- main file: `frontend/app.py`
- add this secret:

```toml
BACKEND_URL = "https://your-backend-service.onrender.com"
```

Reference file:
- [`.streamlit/secrets.toml.example`](./.streamlit/secrets.toml.example)

## Repository Structure

```text
backend/
  main.py
  predictor.py
frontend/
  app.py
models/
  best_model.pth
  best_model_meta.json
training/
  train_model.py
sample_images/
  benign/
  malignant/
  invalid/
```

## Future Improvements

- better benign vs malignant precision
- more hard benign examples to reduce false positives
- more real-world invalid images
- calibration and uncertainty tuning
- lesion localization / segmentation before classification
- lighter deployment model for faster inference

## Disclaimer

This tool is for educational and screening purposes only. It is not a confirmed medical diagnosis.
