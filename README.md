# 🔬 Skin Cancer Screening System

A Streamlit + FastAPI project for AI-assisted skin lesion screening using a DenseNet121-based PyTorch model.

🔗 Live Demo: [Skin Cancer Screening App](https://skin-cancer-screening.streamlit.app/)

💻 Tech Stack:  
Python • PyTorch • DenseNet121 • FastAPI • Streamlit

---

## 👩‍💻 Development Approach

This project was developed using an AI-assisted workflow along with feature planning, testing, model experimentation, and iterative improvements as part of my learning journey.

### My contribution included:

- Defining the project problem statement
- Planning application flow and prediction outputs
- Integrating frontend and backend workflow
- Understanding model behavior and outputs
- Testing and refining prediction behavior
- Deployment and project refinement
- Using prompt engineering to support implementation and learning

### About implementation

AI tools were used as development assistants for implementation support and explanations, while I focused on understanding concepts, integrating components, improving features, and learning practical AI application workflows.

The goal was not only to build a working system but also to understand how AI-assisted development can support learning and project building.

---

## 📚 Learning Note

This repository represents an AI-assisted learning project.

The focus was on:

- understanding machine learning workflows
- model integration
- frontend-backend communication
- API deployment
- practical AI application building

This project represents an ongoing learning journey through practical project building and AI-assisted workflows.

---

## 📌 Project Scope

This is a skin lesion screening project.

Designed for:

- close-up lesion photos
- binary lesion risk assessment
- invalid-image rejection

Not designed for:

- general rash diagnosis
- full-body dermatology analysis
- clinical diagnosis

---

## 🚀 What This Project Does

The system:

- Accepts a skin image from the user
- Sends image data to a FastAPI backend
- Performs model inference
- Returns a screening summary including:

  - predicted class
  - confidence
  - risk level
  - recommendation
  - detailed probabilities

- Supports demo sample images

---

## 🎯 Prediction Categories

Possible outcomes:

### Benign

Lower-risk lesion prediction

### Malignant

Higher-risk lesion prediction

### Invalid

Used for:

- non-lesion images
- blurry images
- full body photos
- unrelated uploads
- unsuitable inputs

---

## 🧠 Model Information

Architecture:

```text
DenseNet121
```

Framework:

```text
PyTorch
```

Backend:

```text
FastAPI
```

Frontend:

```text
Streamlit
```

Training pipeline supports:

- benign
- malignant
- invalid

---

## 📂 Dataset

Base dataset source:

https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

Additional robustness was introduced through an Invalid class containing unsuitable images.

---

## ⚠️ Current Model Behavior

The model is tuned as a screening system and prioritizes safety.

Current behavior:

- works best on lesion-focused close-up images
- may produce false positives on difficult benign cases
- may reject body-part images as Invalid
- intended as a screening aid only

---

## 📷 Demo Sample Images

Sample images available:

```text
sample_images/
├── benign/
├── malignant/
└── invalid/
```

The frontend supports:

**Choose from sample folder**

This helps users test the project without uploading personal images.

---

## ⚙️ Local Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run backend

```bash
python -m uvicorn backend.main:app --reload
```

### Run frontend

```bash
streamlit run frontend/app.py
```

---

## 🌐 Deployment

### Option 1

Render for frontend + backend

Steps:

- Push repository to GitHub
- Create Blueprint on Render
- Render provisions services automatically

---

### Option 2

Backend:

Render

Frontend:

Streamlit Community Cloud

Backend settings:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Frontend secret:

```env
BACKEND_URL=https://your-backend-service.onrender.com
```

---

## 📁 Repository Structure

```text
backend/
│   main.py
│   predictor.py
│
frontend/
│   app.py
│
models/
│   best_model.pth
│   best_model_meta.json
│
training/
│   train_model.py
│
sample_images/
│   ├── benign/
│   ├── malignant/
│   └── invalid/
```

---

## 🚀 Future Improvements

Planned improvements:

- improve benign vs malignant precision
- add more hard benign examples
- improve invalid image handling
- uncertainty calibration
- lesion localization before classification
- faster lightweight deployment model

---

## ⚠️ Disclaimer

This project is intended for educational and screening purposes only.

It is not a confirmed medical diagnosis system and should not replace professional medical advice.

---

## 📖 Learning Journey Note

This repository represents part of my learning journey in AI-assisted development.

The focus was understanding:

- model workflows
- deployment concepts
- backend integration
- practical AI application building
- real-world project structure
