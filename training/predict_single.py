import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# =========================
# Config
# =========================
MODEL_PATH = "models/best_model.pth"
IMAGE_PATH = "C:/Users/LENOVO/skin-cancer-screening/data/skin_cancer/test/benign/2.jpg"
IMG_SIZE = 300
THRESHOLD = 0.5
CLASS_NAMES = ["benign", "malignant"]


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


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
# Load model
# =========================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()


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

    benign_prob = probs[0].item()
    malignant_prob = probs[1].item()

    predicted_label = 1 if malignant_prob >= THRESHOLD else 0
    predicted_class = CLASS_NAMES[predicted_label]

    # Risk logic
    if malignant_prob < 0.4:
        risk_level = "Low Risk"
    elif malignant_prob < 0.7:
        risk_level = "Suspicious"
    else:
        risk_level = "High Risk"

    print("\n===== Prediction Result =====")
    print(f"Image: {image_path}")
    print(f"Benign Probability: {benign_prob:.4f}")
    print(f"Malignant Probability: {malignant_prob:.4f}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Risk Level: {risk_level}")

    if predicted_class == "malignant":
        print("Recommendation: Dermatologist consultation advised.")
    else:
        print("Recommendation: Low-risk pattern detected, but monitor changes and consult a doctor if needed.")


# =========================
# Run
# =========================
predict_image(IMAGE_PATH)