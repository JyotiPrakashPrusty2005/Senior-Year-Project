# Stroke prediction inference script
import os
import joblib
import numpy as np
import pandas as pd

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# ── Load Model, Scaler & Encoders ──
model = joblib.load(os.path.join(MODELS_DIR, "stroke_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "stroke_scaler.pkl"))
label_encoders = joblib.load(os.path.join(MODELS_DIR, "stroke_encoders.pkl"))

# Feature order must match training (drops id and stroke)
FEATURE_NAMES = [
    "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status"
]

CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status"
]


def predict_stroke(data: dict):
    """
    Predict stroke risk from patient data.

    Args:
        data: dict with keys matching FEATURE_NAMES.
              Categorical values should be raw strings (e.g. "Male", "Yes").

    Returns:
        dict with disease, prediction, probability, and risk_level.
    """
    encoded = {}
    for f in FEATURE_NAMES:
        val = data[f]
        if f in CATEGORICAL_FEATURES:
            val = label_encoders[f].transform([str(val)])[0]
        encoded[f] = val

    features = pd.DataFrame([{f: encoded[f] for f in FEATURE_NAMES}])
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    stroke_prob = float(proba[1])
    pred = int(stroke_prob > 0.5)

    if stroke_prob > 0.7:
        risk = "high"
    elif stroke_prob > 0.4:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "disease": "stroke",
        "prediction": pred,
        "predicted_class": "Positive" if pred == 1 else "Negative",
        "probability": round(stroke_prob, 4),
        "risk_level": risk,
    }


if __name__ == "__main__":
    sample = {
        "gender": "Male",
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }
    result = predict_stroke(sample)
    print(result)