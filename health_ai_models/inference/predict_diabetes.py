# Diabetes prediction inference script
import os
import joblib
import numpy as np
import pandas as pd

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# ── Load Model & Scaler ──
model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))

# Feature order must match training
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

CLASS_NAMES = ["No Diabetes", "Prediabetes", "Diabetes"]


def predict_diabetes(data: dict):
    """
    Predict diabetes risk from patient health indicators.

    Args:
        data: dict with keys matching FEATURE_NAMES.

    Returns:
        dict with disease, prediction, predicted_class, probability, and risk_level.
    """
    features = pd.DataFrame([{f: data[f] for f in FEATURE_NAMES}])
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    # Risk based on diabetes probability (class 2)
    diabetes_prob = float(proba[2])
    if diabetes_prob > 0.7:
        risk = "high"
    elif diabetes_prob > 0.4:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "disease": "diabetes",
        "prediction": pred_class,
        "predicted_class": CLASS_NAMES[pred_class],
        "probabilities": {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(len(CLASS_NAMES))},
        "confidence": round(confidence, 4),
        "risk_level": risk,
    }


if __name__ == "__main__":
    sample = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 30,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
        "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 3, "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 0,
        "Sex": 1, "Age": 9, "Education": 5, "Income": 6,
    }
    result = predict_diabetes(sample)
    print(result)