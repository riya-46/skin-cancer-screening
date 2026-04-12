import io
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================
# Config
# =========================
MODEL_PATH = "models/best_model.pth"
MODEL_META_PATH = "models/best_model_meta.json"
DEFAULT_IMG_SIZE = 300
DEFAULT_CLASS_NAMES = ["benign", "malignant"]
DEFAULT_MALIGNANT_THRESHOLD = 0.75
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
INVALID_CLASS_NAMES = {"invalid", "unknown", "not_skin_lesion"}


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_metadata():
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
    except FileNotFoundError:
        metadata = {}

    class_names = metadata.get("class_names", DEFAULT_CLASS_NAMES)
    return {
        "class_names": class_names,
        "img_size": metadata.get("img_size", DEFAULT_IMG_SIZE),
        "malignant_threshold": metadata.get(
            "recommended_malignant_threshold",
            DEFAULT_MALIGNANT_THRESHOLD,
        ),
        "confidence_threshold": metadata.get(
            "recommended_confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD,
        ),
    }


MODEL_META = load_model_metadata()
CLASS_NAMES = MODEL_META["class_names"]
IMG_SIZE = MODEL_META["img_size"]
MALIGNANT_THRESHOLD = MODEL_META["malignant_threshold"]
CONFIDENCE_THRESHOLD = MODEL_META["confidence_threshold"]


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
# Load model once
# =========================
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.classifier.in_features, len(CLASS_NAMES)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


model = load_model()


# =========================
# Helper logic
# =========================
def get_risk_level(malignant_prob: float, predicted_class: str) -> str:
    if predicted_class in INVALID_CLASS_NAMES:
        return "Invalid Image"
    if malignant_prob < 0.4:
        return "Low Risk"
    if malignant_prob < 0.7:
        return "Suspicious"
    return "High Risk"


def get_recommendation(predicted_class: str, confidence: float) -> str:
    if predicted_class in INVALID_CLASS_NAMES:
        return "Please upload a clear skin lesion image. Non-lesion or unrelated images are not valid for screening."
    if confidence < CONFIDENCE_THRESHOLD:
        return "The model is uncertain. Please upload a clearer lesion image or seek clinical review."
    if predicted_class == "malignant":
        return "Dermatologist consultation advised."
    return "Low-risk pattern detected, but monitor changes and consult a doctor if needed."


def get_probability_map(probs):
    probability_map = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        probability_map[f"{class_name}_probability"] = float(probs[idx].item())
    return probability_map


def resolve_prediction(probs):
    confidence, predicted_idx = torch.max(probs, dim=0)
    predicted_class = CLASS_NAMES[int(predicted_idx.item())]
    probability_map = get_probability_map(probs)
    malignant_prob = probability_map.get("malignant_probability", 0.0)

    if predicted_class not in INVALID_CLASS_NAMES and "malignant" in CLASS_NAMES:
        malignant_idx = CLASS_NAMES.index("malignant")
        if float(probs[malignant_idx].item()) >= MALIGNANT_THRESHOLD:
            predicted_class = "malignant"
            confidence = probs[malignant_idx]
        elif predicted_class == "malignant":
            predicted_class = "benign"
            confidence = probs[CLASS_NAMES.index("benign")]

    is_invalid = predicted_class in INVALID_CLASS_NAMES
    is_uncertain = (not is_invalid) and float(confidence.item()) < CONFIDENCE_THRESHOLD

    return {
        "predicted_class": predicted_class,
        "predicted_probability": float(confidence.item()),
        "malignant_probability": float(malignant_prob),
        "is_valid_image": not is_invalid,
        "is_uncertain": is_uncertain,
        "probability_map": probability_map,
    }


# =========================
# Main prediction function
# =========================
def predict_image_bytes(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    prediction = resolve_prediction(probs)
    predicted_class = prediction["predicted_class"]
    predicted_probability = prediction["predicted_probability"]
    malignant_prob = prediction["malignant_probability"]
    benign_prob = prediction["probability_map"].get("benign_probability", 0.0)
    invalid_prob = max(
        prediction["probability_map"].get("invalid_probability", 0.0),
        prediction["probability_map"].get("unknown_probability", 0.0),
        prediction["probability_map"].get("not_skin_lesion_probability", 0.0),
    )

    risk_level = get_risk_level(malignant_prob, predicted_class)
    recommendation = get_recommendation(predicted_class, predicted_probability)

    return {
        "predicted_class": predicted_class.replace("_", " ").title(),
        "predicted_probability": float(predicted_probability),
        "benign_probability": float(benign_prob),
        "malignant_probability": float(malignant_prob),
        "invalid_probability": float(invalid_prob),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "is_valid_image": prediction["is_valid_image"],
        "is_uncertain": prediction["is_uncertain"],
    }
