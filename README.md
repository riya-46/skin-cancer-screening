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