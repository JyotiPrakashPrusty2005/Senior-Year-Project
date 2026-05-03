"""
Inference pipeline for the Post-Surgery Recovery Assistant.

Combines:
  1. LSTM model — predicts recovery trajectory and complication risk
  2. Recovery Coach — generates personalized advice via Generative AI

This is the unified inference endpoint that ties the hybrid ML + LSTM + GenAI together.
"""

import os
import sys
import numpy as np
import torch
import joblib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from training.train_recovery_lstm import RecoveryLSTM, VITALS_FEATURES, STATIC_FEATURES, SEQUENCE_LENGTH
from inference.recovery_coach import get_recovery_advice

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_recovery_model():
    """Load the trained LSTM model and preprocessing artifacts."""
    model_path = os.path.join(MODEL_DIR, "recovery_lstm_model.pth")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    model = RecoveryLSTM(
        num_vitals=config["num_vitals"],
        num_static=config["num_static"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        forecast_days=config["forecast_days"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    vitals_scaler = joblib.load(os.path.join(MODEL_DIR, "recovery_vitals_scaler.pkl"))
    static_scaler = joblib.load(os.path.join(MODEL_DIR, "recovery_static_scaler.pkl"))
    surgery_encoder = joblib.load(os.path.join(MODEL_DIR, "recovery_surgery_encoder.pkl"))

    return model, vitals_scaler, static_scaler, surgery_encoder, config


def predict_recovery(vitals_history, patient_info, model=None, vitals_scaler=None,
                     static_scaler=None, surgery_encoder=None, config=None,
                     api_key=None, provider="offline"):
    """
    Predict recovery trajectory and complication risk.

    Args:
        vitals_history: List of dicts, each with keys from VITALS_FEATURES.
                       Must have at least SEQUENCE_LENGTH (7) days.
                       Example: [{"pain_level": 7, "temperature": 37.5, ...}, ...]
        patient_info: Dict with patient demographics:
                     {"age": 55, "gender": 1, "bmi": 28.0, "diabetes": 0,
                      "hypertension": 1, "smoking": 0, "surgery_type": "cardiac"}
        api_key: Optional API key for AI coach (Gemini/OpenAI/Groq)
        provider: "gemini", "openai", "groq", or "offline"

    Returns:
        Dict with predictions and formatted advice
    """
    # Load model if not provided
    if model is None:
        model, vitals_scaler, static_scaler, surgery_encoder, config = load_recovery_model()

    # Prepare vitals sequence
    vitals_array = np.array([
        [day[feat] for feat in VITALS_FEATURES]
        for day in vitals_history[-SEQUENCE_LENGTH:]
    ])
    vitals_scaled = vitals_scaler.transform(vitals_array)

    # Prepare static features
    surgery_encoded = surgery_encoder.transform([patient_info["surgery_type"]])[0]
    static_array = np.array([[
        patient_info["age"], patient_info["gender"], patient_info["bmi"],
        patient_info["diabetes"], patient_info["hypertension"],
        patient_info["smoking"], surgery_encoded,
    ]])
    static_scaled = static_scaler.transform(static_array)

    # Run inference
    vitals_tensor = torch.FloatTensor(vitals_scaled).unsqueeze(0).to(DEVICE)
    static_tensor = torch.FloatTensor(static_scaled).to(DEVICE)

    with torch.no_grad():
        recovery_pred, complication_pred = model(vitals_tensor, static_tensor)

    recovery_forecast = recovery_pred.cpu().numpy()[0].tolist()
    complication_risk = complication_pred.cpu().item()

    # Determine risk level
    if complication_risk > 0.7:
        risk_level = "HIGH"
    elif complication_risk > 0.4:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    # Determine trend
    if len(recovery_forecast) >= 2:
        if recovery_forecast[-1] > recovery_forecast[0]:
            trend = "improving"
        elif recovery_forecast[-1] < recovery_forecast[0]:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "stable"

    lstm_predictions = {
        "recovery_forecast": [round(r, 1) for r in recovery_forecast],
        "complication_risk": complication_risk,
        "risk_level": risk_level,
        "trend": trend,
    }

    # Get the latest vitals for the coach
    latest_vitals = vitals_history[-1]
    patient_data = {
        **patient_info,
        **latest_vitals,
        "current_day": len(vitals_history),
        "recovery_score": latest_vitals.get("recovery_score", 50),
    }

    # Generate advice using the recovery coach
    # Use directly passed api_key/provider, fall back to env vars for CLI usage
    _api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key and not provider:
        if os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "offline"

    advice = get_recovery_advice(patient_data, lstm_predictions, api_key=_api_key, provider=provider)

    return {
        "predictions": lstm_predictions,
        "patient_summary": {
            "day": len(vitals_history),
            "current_recovery_score": latest_vitals.get("recovery_score", 50),
            "complication_risk": f"{complication_risk:.1%}",
            "risk_level": risk_level,
            "recovery_forecast_3day": lstm_predictions["recovery_forecast"],
            "trend": trend,
        },
        "advice": advice,
    }


# ─── Example Usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated 7-day vitals for a patient on day 10 after cardiac surgery
    example_vitals = [
        {"pain_level": 7.0, "temperature": 37.8, "heart_rate": 92, "bp_systolic": 135, "mobility_score": 2.5, "wound_status": 0.8, "sleep_hours": 4.5, "recovery_score": 30},
        {"pain_level": 6.5, "temperature": 37.6, "heart_rate": 90, "bp_systolic": 132, "mobility_score": 3.0, "wound_status": 0.75, "sleep_hours": 5.0, "recovery_score": 35},
        {"pain_level": 6.0, "temperature": 37.5, "heart_rate": 88, "bp_systolic": 130, "mobility_score": 3.5, "wound_status": 0.7, "sleep_hours": 5.5, "recovery_score": 40},
        {"pain_level": 5.5, "temperature": 37.3, "heart_rate": 85, "bp_systolic": 128, "mobility_score": 4.0, "wound_status": 0.6, "sleep_hours": 6.0, "recovery_score": 45},
        {"pain_level": 5.0, "temperature": 37.2, "heart_rate": 84, "bp_systolic": 126, "mobility_score": 4.5, "wound_status": 0.55, "sleep_hours": 6.5, "recovery_score": 50},
        {"pain_level": 4.5, "temperature": 37.1, "heart_rate": 82, "bp_systolic": 124, "mobility_score": 5.0, "wound_status": 0.5, "sleep_hours": 7.0, "recovery_score": 55},
        {"pain_level": 4.0, "temperature": 37.0, "heart_rate": 80, "bp_systolic": 122, "mobility_score": 5.5, "wound_status": 0.45, "sleep_hours": 7.0, "recovery_score": 60},
    ]

    patient_info = {
        "age": 58,
        "gender": 1,
        "bmi": 28.5,
        "diabetes": 0,
        "hypertension": 1,
        "smoking": 0,
        "surgery_type": "cardiac",
    }

    print("Loading model and generating recovery plan...")
    try:
        result = predict_recovery(example_vitals, patient_info)
        print("\n" + "=" * 60)
        print("PATIENT RECOVERY SUMMARY")
        print("=" * 60)
        for key, value in result["patient_summary"].items():
            print(f"  {key}: {value}")
        print("\n" + result["advice"])
    except FileNotFoundError:
        print("Model not found. Please run the training first:")
        print("  1. python data/generate_synthetic_recovery.py")
        print("  2. python training/train_recovery_lstm.py")
        print("\nShowing offline coach demo instead...\n")

        # Demo the coach without the LSTM model
        from inference.recovery_coach import get_recovery_advice

        patient_data = {
            **patient_info,
            **example_vitals[-1],
            "current_day": 10,
            "recovery_score": 60,
        }
        mock_predictions = {
            "recovery_forecast": [62.5, 65.0, 67.5],
            "complication_risk": 0.15,
            "risk_level": "LOW",
            "trend": "improving",
        }
        print(get_recovery_advice(patient_data, mock_predictions))
