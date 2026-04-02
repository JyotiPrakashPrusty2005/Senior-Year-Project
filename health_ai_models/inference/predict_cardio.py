# Cardiovascular prediction inference script
import os
import joblib
import numpy as np
import pandas as pd

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# ── Load Model & Scaler ──
model = joblib.load(os.path.join(MODELS_DIR, "cardio_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "cardio_scaler.pkl"))

# Feature order must match training (cardio_train.csv minus id and cardio)
FEATURE_NAMES = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active"
]


def predict_cardio(data: dict):
    """
    Predict cardiovascular disease risk from patient data.

    Args:
        data: dict with keys matching FEATURE_NAMES.

    Returns:
        dict with disease, prediction, probability, and risk_level.
    """
    features = pd.DataFrame([{f: data[f] for f in FEATURE_NAMES}])
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    cardio_prob = float(proba[1])
    pred = int(cardio_prob > 0.5)

    if cardio_prob > 0.7:
        risk = "high"
    elif cardio_prob > 0.4:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "disease": "cardiovascular",
        "prediction": pred,
        "predicted_class": "Positive" if pred == 1 else "Negative",
        "probability": round(cardio_prob, 4),
        "risk_level": risk,
    }


if __name__ == "__main__":
    sample = {
        "age": 18393,   # age in days (~50 years)
        "gender": 2,
        "height": 168,
        "weight": 62,
        "ap_hi": 110,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
    }
    result = predict_cardio(sample)
    print(result)
