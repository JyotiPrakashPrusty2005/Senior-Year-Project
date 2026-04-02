# JSON response utilities
def build_response(disease, prob):

    if prob > 0.7:
        risk = "high"
    elif prob > 0.4:
        risk = "moderate"
    else:
        risk = "low"

    return {
        "disease": disease,
        "probability": float(prob),
        "risk_level": risk
    }