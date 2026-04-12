import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================
# Config
# =========================
MODEL_PATH = "models/best_model.pth"
MODEL_META_PATH = "models/best_model_meta.json"
IMAGE_PATH = "C:/Users/LENOVO/skin-cancer-screening/data/skin_cancer/test/benign/2.jpg"
DEFAULT_IMG_SIZE = 300
DEFAULT_CLASS_NAMES = ["benign", "malignant"]
DEFAULT_MALIGNANT_THRESHOLD = 0.75
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
INVALID_CLASS_NAMES = {"invalid", "unknown", "not_skin_lesion"}


def load_metadata():
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
    except FileNotFoundError:
        meta = {}

    return {
        "class_names": meta.get("class_names", DEFAULT_CLASS_NAMES),
        "img_size": meta.get("img_size", DEFAULT_IMG_SIZE),
        "malignant_threshold": meta.get(
            "recommended_malignant_threshold",
            DEFAULT_MALIGNANT_THRESHOLD,
        ),
        "confidence_threshold": meta.get(
            "recommended_confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD,
        ),
    }


META = load_metadata()
CLASS_NAMES = META["class_names"]
IMG_SIZE = META["img_size"]
MALIGNANT_THRESHOLD = META["malignant_threshold"]
CONFIDENCE_THRESHOLD = META["confidence_threshold"]


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
print("Classes:", CLASS_NAMES)


# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# Load model
# =========================
model = models.densenet121(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.35),
    nn.Linear(model.classifier.in_features, len(CLASS_NAMES)),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()


def resolve_prediction(probs):
    confidence, predicted_idx = torch.max(probs, dim=0)
    predicted_class = CLASS_NAMES[int(predicted_idx.item())]
    malignant_prob = float(probs[CLASS_NAMES.index("malignant")].item())

    if predicted_class not in INVALID_CLASS_NAMES and malignant_prob >= MALIGNANT_THRESHOLD:
        predicted_class = "malignant"
        confidence = probs[CLASS_NAMES.index("malignant")]
    elif predicted_class == "malignant" and malignant_prob < MALIGNANT_THRESHOLD:
        predicted_class = "benign"
        confidence = probs[CLASS_NAMES.index("benign")]

    is_invalid = predicted_class in INVALID_CLASS_NAMES
    is_uncertain = (not is_invalid) and float(confidence.item()) < CONFIDENCE_THRESHOLD

    return predicted_class, float(confidence.item()), malignant_prob, is_invalid, is_uncertain


# =========================
# Prediction function
# =========================
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_class, confidence, malignant_prob, is_invalid, is_uncertain = resolve_prediction(probs)

    print("\n===== Prediction Result =====")
    print(f"Image: {image_path}")

    for idx, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name.title()} Probability: {probs[idx].item():.4f}")

    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction Confidence: {confidence:.4f}")

    if is_invalid:
        print("Risk Level: Invalid Image")
        print("Recommendation: Please upload a clear skin lesion image.")
        return

    if malignant_prob < 0.4:
        risk_level = "Low Risk"
    elif malignant_prob < 0.7:
        risk_level = "Suspicious"
    else:
        risk_level = "High Risk"

    print(f"Risk Level: {risk_level}")

    if is_uncertain:
        print("Recommendation: Model is uncertain. Please upload a clearer lesion image or seek clinical review.")
    elif predicted_class == "malignant":
        print("Recommendation: Dermatologist consultation advised.")
    else:
        print("Recommendation: Low-risk pattern detected, but monitor changes and consult a doctor if needed.")


# =========================
# Run
# =========================
predict_image(IMAGE_PATH)
