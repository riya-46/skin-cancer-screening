import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# =========================
# Config
# =========================
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 300
THRESHOLD = 0.5
CLASS_NAMES = ["benign", "malignant"]

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# =========================
# Load model once
# =========================
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# Risk logic
# =========================
def get_risk_level(malignant_prob: float) -> str:
    if malignant_prob < 0.4:
        return "Low Risk"
    elif malignant_prob < 0.7:
        return "Suspicious"
    else:
        return "High Risk"

def get_recommendation(predicted_class: str) -> str:
    if predicted_class == "malignant":
        return "Dermatologist consultation advised."
    return "Low-risk pattern detected, but monitor changes and consult a doctor if needed."

# =========================
# Main prediction function
# =========================
def predict_image_bytes(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    benign_prob = float(probs[0].item())
    malignant_prob = float(probs[1].item())

    predicted_label = 1 if malignant_prob >= THRESHOLD else 0
    predicted_class = CLASS_NAMES[predicted_label]

    risk_level = get_risk_level(malignant_prob)
    recommendation = get_recommendation(predicted_class)

    predicted_probability = benign_prob if predicted_class == "benign" else malignant_prob

    return {
        "predicted_class": predicted_class.capitalize(),
        "predicted_probability": float(predicted_probability),
        "benign_probability": float(benign_prob),
        "malignant_probability": float(malignant_prob),
        "risk_level": risk_level,
        "recommendation": recommendation,
    }