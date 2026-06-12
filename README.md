# Skin Cancer Screening System

An AI-assisted skin lesion screening project built with `Streamlit`, `FastAPI`, and a `DenseNet121` PyTorch model.

The system accepts a close-up skin lesion image, sends it to a backend inference API, and returns a screening summary with class prediction, confidence, risk level, recommendation, and class-wise probabilities.

## Live Deployment

- Frontend: https://skin-cancer-screening.streamlit.app/
- Backend API: https://skin-cancer-backend-reva.onrender.com

## Project Scope

This project is designed for close-up dermoscopic or lesion-focused skin images.

Supported output classes:

- `Benign`
- `Malignant`
- `Invalid`

The `Invalid` class is used for unsuitable inputs such as full body-part photos, broad rash images, blurry captures, normal casual skin photos, screenshots, or unrelated images.

This project is a screening aid for educational use. It is not a confirmed clinical diagnosis system.

## What Was Built

The complete project flow includes:

1. Dataset preparation for benign, malignant, and invalid image classes.
2. Dataset inspection utilities to check class counts, image sizes, color modes, and corrupted files.
3. A PyTorch training pipeline using transfer learning with `DenseNet121`.
4. Model evaluation using accuracy, precision, recall, F1-score, confusion matrix, classification report, and threshold analysis.
5. Saved model artifacts for production inference.
6. A FastAPI backend that loads the trained model and exposes a prediction endpoint.
7. A Streamlit frontend that allows image upload or sample image selection.
8. A styled screening report UI with risk labels, probability cards, recommendations, and disclaimer.
9. Deployment configuration for Render and Streamlit Community Cloud.
10. Cleanup of unused experiment files and generated cache files.

## Tech Stack

- Language: `Python`
- Model framework: `PyTorch`
- Model architecture: `DenseNet121`
- Image processing: `Pillow`, `torchvision.transforms`
- Metrics and splitting: `scikit-learn`
- Backend API: `FastAPI`
- API server: `Uvicorn`
- Frontend: `Streamlit`
- Dataset download support: `requests`, `gdown`
- Deployment: `Render`, `Streamlit Community Cloud`

## Repository Structure

```text
backend/
  __init__.py
  main.py
  predictor.py

frontend/
  app.py

models/
  best_model.pth
  best_model_meta.json

training/
  analyze_dataset.py
  predict_single.py
  train_model.py

sample_images/
  benign/
  malignant/
  invalid/

.streamlit/
  secrets.toml.example

render.yaml
requirements.txt
README.md
```

Ignored local folders such as `venv/`, `data/`, `.cache/`, and `__pycache__/` are not part of the committed application source.

## Dataset

Base lesion dataset source:

- https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

The dataset is expected in this structure:

```text
data/skin_cancer/
  train/
    benign/
    malignant/
    invalid/
  test/
    benign/
    malignant/
    invalid/
```

The main lesion classes come from dermoscopic images. Additional invalid examples were added so the model can reject non-lesion or unsuitable uploads.

## Dataset Analysis

The dataset can be inspected using:

```powershell
python training/analyze_dataset.py
```

This checks:

- total image count
- class-wise distribution
- image color modes
- image dimensions
- corrupted image files

This step helps verify the dataset before training.

## Model Training

The main training pipeline is:

```powershell
python training/train_model.py
```

You can use a custom dataset path:

```powershell
$env:SKIN_CANCER_DATA_DIR="G:\My Drive\skin_cancer_dataset"
python training/train_model.py
```

Training includes:

- fixed random seed for reproducibility
- automatic class discovery
- stratified train/validation split
- data augmentation
- ImageNet normalization
- weighted sampling for class imbalance
- class-weighted cross entropy loss
- label smoothing
- `AdamW` optimizer
- `ReduceLROnPlateau` scheduler
- early stopping
- threshold tuning for malignant detection
- confidence threshold analysis

The model uses ImageNet-pretrained `DenseNet121`. Its classifier is replaced with a custom output layer matching the detected classes.

## Saved Model Artifacts

After training, the best model is saved as:

```text
models/best_model.pth
```

Model metadata is saved as:

```text
models/best_model_meta.json
```

The metadata currently stores:

- class names
- class-to-index mapping
- image size
- recommended malignant threshold
- recommended confidence threshold
- validation malignant precision
- validation malignant recall
- validation malignant F1-score

## Current Model Metadata

Current saved model metadata:

```text
Classes: benign, malignant, invalid
Image size: 300
Recommended malignant threshold: 0.55
Recommended confidence threshold: 0.8
Validation malignant precision: 0.8853
Validation malignant recall: 0.9333
Validation malignant F1-score: 0.9087
```

## Single Image Prediction Test

For local model testing without running the web app:

```powershell
python training/predict_single.py
```

You can override the image path:

```powershell
$env:PREDICT_IMAGE_PATH="G:\My Drive\skin_cancer_dataset\test\benign\2.jpg"
python training/predict_single.py
```

## Backend API

The backend is implemented in `backend/main.py` and `backend/predictor.py`.

Run locally:

```powershell
python -m uvicorn backend.main:app --reload
```

API endpoints:

```text
GET  /
POST /predict
```

The `/predict` endpoint accepts an uploaded image file, preprocesses it, runs model inference, applies threshold logic, and returns JSON.

Example response fields:

```text
predicted_class
predicted_probability
benign_probability
malignant_probability
invalid_probability
risk_level
recommendation
is_valid_image
is_uncertain
```

## Prediction Logic

The backend performs these steps:

1. Read uploaded image bytes.
2. Open the image with `Pillow`.
3. Convert it to RGB.
4. Resize it to `300 x 300`.
5. Convert it to a tensor.
6. Normalize it using ImageNet mean and standard deviation.
7. Run the image through the trained DenseNet121 model.
8. Convert logits to probabilities using softmax.
9. Apply invalid-image and malignant-threshold rules.
10. Return risk level and recommendation.

Risk levels:

- `Low Risk`
- `Suspicious`
- `High Risk`
- `Invalid Image`

## Frontend App

The frontend is implemented in `frontend/app.py`.

Run locally:

```powershell
streamlit run frontend/app.py
```

The Streamlit app supports:

- image upload
- bundled sample image gallery
- selected image preview
- backend prediction request
- loading/progress states
- probability display
- risk badge
- recommendation output
- invalid-image warning
- uncertainty warning
- medical disclaimer

The frontend reads backend configuration from:

- `BACKEND_URL`
- `BACKEND_HOSTPORT`
- `.streamlit/secrets.toml`

## Sample Image Library

Bundled samples are stored in:

```text
sample_images/
  benign/
  malignant/
  invalid/
```

The app can also load a larger external image library using:

- `DEMO_LIBRARY_DIR`
- `DEMO_LIBRARY_DRIVE_FOLDER_URL`
- `DEMO_LIBRARY_DRIVE_FOLDER_ID`
- `DEMO_LIBRARY_DRIVE_FILE_ID`
- `DEMO_LIBRARY_DRIVE_URL`

Example local dataset library:

```powershell
$env:DEMO_LIBRARY_DIR="G:\My Drive\skin_cancer_dataset"
streamlit run frontend/app.py
```

Example Google Drive zip:

```powershell
$env:DEMO_LIBRARY_DRIVE_FILE_ID="your_public_google_drive_zip_file_id"
$env:DEMO_LIBRARY_LABEL="Skin Lesion Dataset"
streamlit run frontend/app.py
```

The Google Drive archive should contain either:

```text
benign/
malignant/
invalid/
```

or:

```text
train/
  benign/
  malignant/
  invalid/
test/
  benign/
  malignant/
  invalid/
```

On first load, the app downloads and extracts the archive into `.cache/demo_library/`.

## Local Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run backend:

```powershell
python -m uvicorn backend.main:app --reload
```

Run frontend in another terminal:

```powershell
streamlit run frontend/app.py
```

Default local backend URL:

```text
http://127.0.0.1:8000
```

## Deployment

This repository includes `render.yaml` for Render deployment.

Render services:

- `skin-cancer-backend`
- `skin-cancer-frontend`

Backend start command:

```powershell
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Frontend start command:

```powershell
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port $PORT
```

For Streamlit Community Cloud, set the backend URL in secrets:

```toml
BACKEND_URL = "https://your-backend-service.onrender.com"
```

Reference:

```text
.streamlit/secrets.toml.example
```

## End-To-End Flow

```text
Dataset prepared
-> Dataset analyzed
-> Images transformed and augmented
-> DenseNet121 trained with transfer learning
-> Validation metrics and thresholds calculated
-> Best model saved
-> Model metadata saved
-> FastAPI backend loads model once
-> Streamlit frontend accepts user image
-> Frontend sends image to /predict
-> Backend preprocesses image
-> Model returns probabilities
-> Threshold logic resolves final class
-> Risk level and recommendation generated
-> Frontend displays screening report
```

## Cleanup Performed

The project was cleaned by removing:

- the unused EfficientNet experiment script
- redundant `.gitkeep` files from non-empty sample image folders
- generated local cache and Python bytecode folders

The active production path now uses the DenseNet121 model, FastAPI backend, and Streamlit frontend.

## Future Improvements

- improve benign vs malignant precision
- add more hard benign examples to reduce false positives
- add more invalid real-world images
- calibrate model probabilities
- add uncertainty calibration
- add lesion localization or segmentation before classification
- optimize model size for faster deployment inference

## Medical Disclaimer

This tool is for educational and screening purposes only. It is not a confirmed medical diagnosis. Always consult a qualified healthcare professional for proper medical advice and diagnosis.
