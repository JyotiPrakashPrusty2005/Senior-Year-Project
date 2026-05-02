# Pneumonia prediction inference script (Chest X-Ray)
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# ── Load Model ──
MODEL_PATH = os.path.join(MODELS_DIR, "pneumonia_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

CLASS_NAMES = checkpoint["class_names"]  # ['NORMAL', 'PNEUMONIA']
IMG_SIZE = checkpoint["img_size"]
MODEL_NAME = checkpoint["model_name"]

# Rebuild the same architecture
if "ResNet-18" in MODEL_NAME:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASS_NAMES)))
elif "ResNet-50" in MODEL_NAME:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASS_NAMES)))
elif "EfficientNet" in MODEL_NAME:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# ── Image Transform ──
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_pneumonia(image_path: str):
    """
    Predict pneumonia from a chest X-ray image.

    Args:
        image_path: Path to the chest X-ray image file.

    Returns:
        dict with disease, prediction, probability, and risk_level.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

    pneumonia_prob = probs[0][1].item()  # index 1 = PNEUMONIA
    pred = int(pneumonia_prob > 0.5)

    if pneumonia_prob > 0.7:
        risk = "high"
    elif pneumonia_prob > 0.4:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "disease": "pneumonia",
        "prediction": pred,
        "predicted_class": CLASS_NAMES[pred],
        "probability": round(pneumonia_prob, 4),
        "risk_level": risk,
        "model_used": MODEL_NAME
    }


# ── Test with a single image ──
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = predict_pneumonia(img_path)
        print(result)
    else:
        print("Usage: python predict_pneumonia.py <path_to_xray_image>")
