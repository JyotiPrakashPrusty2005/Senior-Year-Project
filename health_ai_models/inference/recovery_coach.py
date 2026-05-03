"""
Post-Surgery Recovery Coach using Generative AI (Agentic AI component).

This module takes the LSTM predictions (recovery score forecast + complication risk)
and patient context, then uses a generative AI model to produce:
  1. Personalized recovery advice
  2. Exercise/mobility recommendations
  3. Dietary guidance
  4. Warning signs to watch for
  5. When to contact a doctor

This is the "Agentic AI" part — it reasons over multiple data sources
and takes action (generates tailored plans, escalates alerts).
"""

import os
import json
from datetime import datetime

# Try importing google genai (Gemini) — new package
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Try importing openai
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try importing groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ─── Recovery Knowledge Base ──────────────────────────────────────────────
# Structured medical knowledge for the AI coach to reference
RECOVERY_GUIDELINES = {
    "cardiac": {
        "typical_recovery_weeks": "6-12",
        "activity_phases": [
            "Week 1-2: Bed rest, gentle breathing exercises, short walks in room",
            "Week 3-4: Walking 5-10 minutes, light stretching, stair climbing (1 flight)",
            "Week 5-8: Walking 15-30 minutes, light household chores",
            "Week 9-12: Moderate exercise, return to most daily activities",
        ],
        "warning_signs": [
            "Chest pain or pressure", "Shortness of breath at rest",
            "Rapid or irregular heartbeat", "Fever above 38.3°C",
            "Redness, swelling, or drainage from incision site",
            "Sudden weight gain (>1kg/day)"
        ],
        "dietary": "Low sodium, heart-healthy fats, high fiber, limit caffeine",
    },
    "orthopedic": {
        "typical_recovery_weeks": "8-16",
        "activity_phases": [
            "Week 1-2: Ice, elevation, prescribed exercises, assistive devices",
            "Week 3-4: Gentle range of motion exercises, short walks",
            "Week 5-8: Progressive strengthening, balance exercises",
            "Week 9-16: Return to normal activities, sport-specific training",
        ],
        "warning_signs": [
            "Increasing pain not controlled by medication",
            "Numbness or tingling in extremities",
            "Fever above 38.3°C", "Calf pain or swelling (DVT risk)",
            "Wound drainage or foul smell",
        ],
        "dietary": "High protein for tissue repair, calcium, vitamin D, anti-inflammatory foods",
    },
    "abdominal": {
        "typical_recovery_weeks": "4-8",
        "activity_phases": [
            "Week 1-2: Walking short distances, avoid lifting >2kg",
            "Week 3-4: Gradual increase in walking, light activities",
            "Week 5-6: Resume most daily activities, avoid heavy lifting",
            "Week 7-8: Return to normal activities including exercise",
        ],
        "warning_signs": [
            "Severe abdominal pain", "Persistent nausea/vomiting",
            "Fever above 38.3°C", "No bowel movement for 3+ days",
            "Wound redness, warmth, or drainage",
        ],
        "dietary": "Start with clear liquids, progress to soft foods, high fiber gradually",
    },
    "neurological": {
        "typical_recovery_weeks": "8-24",
        "activity_phases": [
            "Week 1-2: Bed rest, cognitive rest, minimal stimulation",
            "Week 3-4: Short walks, light reading, limited screen time",
            "Week 5-8: Gradual return to mental activities, longer walks",
            "Week 9+: Progressive return to work/school, monitored exercise",
        ],
        "warning_signs": [
            "Severe headache", "Vision changes", "Confusion or disorientation",
            "Seizures", "Weakness on one side", "Fever above 38.3°C",
            "Clear fluid leaking from nose or ears",
        ],
        "dietary": "Anti-inflammatory diet, omega-3 fatty acids, adequate hydration, antioxidant-rich foods",
    },
}


def build_system_prompt(surgery_type):
    """Build the system prompt for the generative AI coach."""
    guidelines = RECOVERY_GUIDELINES.get(surgery_type, RECOVERY_GUIDELINES["abdominal"])

    return f"""You are an AI-powered post-surgery recovery coach. You provide personalized, 
evidence-based recovery guidance to patients after {surgery_type} surgery.

ROLE: You are a supportive, knowledgeable health assistant. You do NOT replace doctors.
Always remind patients to consult their healthcare provider for medical decisions.

SURGERY TYPE: {surgery_type}
TYPICAL RECOVERY: {guidelines['typical_recovery_weeks']} weeks

ACTIVITY PHASES:
{chr(10).join(f'- {phase}' for phase in guidelines['activity_phases'])}

KEY WARNING SIGNS (tell patient to contact doctor immediately):
{chr(10).join(f'- {sign}' for sign in guidelines['warning_signs'])}

DIETARY GUIDELINES: {guidelines['dietary']}

RESPONSE FORMAT:
1. **Recovery Assessment**: Brief analysis of current status
2. **Today's Recommendations**: Specific actionable advice for today
3. **Exercise Plan**: Appropriate exercises for current recovery phase
4. **Dietary Advice**: Meal suggestions aligned with recovery
5. **Warning Signs**: What to watch for based on current vitals
6. **Motivation**: Encouraging message

IMPORTANT RULES:
- Be empathetic and encouraging
- Give specific, actionable advice (not vague)
- If complication risk is HIGH, strongly recommend contacting doctor
- Adjust advice based on the patient's actual vitals and recovery score
- Consider patient's age, BMI, and comorbidities
- Never diagnose or prescribe medication"""


def build_patient_context(patient_data, lstm_predictions):
    """Build a patient context string from data and LSTM predictions."""
    context = f"""
PATIENT PROFILE:
- Age: {patient_data['age']} years
- Gender: {'Male' if patient_data['gender'] == 1 else 'Female'}
- BMI: {patient_data['bmi']}
- Diabetes: {'Yes' if patient_data['diabetes'] else 'No'}
- Hypertension: {'Yes' if patient_data['hypertension'] else 'No'}
- Smoking: {'Yes' if patient_data['smoking'] else 'No'}
- Surgery Type: {patient_data['surgery_type']}
- Current Recovery Day: {patient_data['current_day']}

TODAY'S VITALS:
- Pain Level: {patient_data['pain_level']}/10
- Temperature: {patient_data['temperature']}°C
- Heart Rate: {patient_data['heart_rate']} bpm
- Blood Pressure (Systolic): {patient_data['bp_systolic']} mmHg
- Mobility Score: {patient_data['mobility_score']}/10
- Wound Status: {'Healing well' if patient_data['wound_status'] < 0.3 else 'Still fresh' if patient_data['wound_status'] < 0.7 else 'Needs attention'}
- Sleep Hours: {patient_data['sleep_hours']} hours
- Current Recovery Score: {patient_data['recovery_score']}/100

AI MODEL PREDICTIONS:
- Predicted Recovery Score (next 3 days): {lstm_predictions['recovery_forecast']}
- Complication Risk: {lstm_predictions['complication_risk']:.1%}
- Risk Level: {lstm_predictions['risk_level']}
- Recovery Trend: {lstm_predictions['trend']}
"""
    return context


def generate_recovery_advice_gemini(patient_data, lstm_predictions, api_key):
    """Generate personalized recovery advice using Google Gemini."""
    if not GENAI_AVAILABLE:
        raise ImportError("google-genai package not installed. Run: pip install google-genai")

    import httpx
    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1"},
        httpx_client=httpx.Client(verify=False),
    )

    system_prompt = build_system_prompt(patient_data["surgery_type"])
    patient_context = build_patient_context(patient_data, lstm_predictions)

    prompt = f"""{system_prompt}

{patient_context}

Based on the patient's current vitals, recovery progress, and the AI model's predictions, 
provide a comprehensive personalized recovery plan for today."""

    # Try multiple models in case one is unavailable or has quota issues
    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Retry on quota errors or model-not-found errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "404" in error_str or "not found" in error_str.lower():
                continue
            raise  # Other errors, don't retry

    raise last_error


def generate_recovery_advice_openai(patient_data, lstm_predictions, api_key):
    """Generate personalized recovery advice using OpenAI."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = openai.OpenAI(api_key=api_key)

    system_prompt = build_system_prompt(patient_data["surgery_type"])
    patient_context = build_patient_context(patient_data, lstm_predictions)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{patient_context}\n\nProvide a comprehensive personalized recovery plan for today."},
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_recovery_advice_groq(patient_data, lstm_predictions, api_key):
    """Generate personalized recovery advice using Groq (free, fast inference)."""
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed. Run: pip install groq")

    import httpx
    client = Groq(api_key=api_key, http_client=httpx.Client(verify=False))

    system_prompt = build_system_prompt(patient_data["surgery_type"])
    patient_context = build_patient_context(patient_data, lstm_predictions)

    # Try multiple models in case one is unavailable
    models_to_try = ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"]
    last_error = None
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{patient_context}\n\nProvide a comprehensive personalized recovery plan for today."},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "model" in error_str.lower() or "not found" in error_str.lower() or "404" in error_str:
                continue
            raise
    raise last_error


def generate_recovery_advice_offline(patient_data, lstm_predictions):
    """
    Rule-based fallback when no API key is available.
    This provides structured advice based on recovery guidelines and model predictions.
    """
    surgery_type = patient_data["surgery_type"]
    guidelines = RECOVERY_GUIDELINES.get(surgery_type, RECOVERY_GUIDELINES["abdominal"])
    day = patient_data["current_day"]
    risk = lstm_predictions["complication_risk"]
    recovery_score = patient_data["recovery_score"]
    trend = lstm_predictions["trend"]

    # Determine recovery phase
    if day <= 7:
        phase = 0
    elif day <= 14:
        phase = 1
    elif day <= 21:
        phase = 2
    else:
        phase = 3
    phase = min(phase, len(guidelines["activity_phases"]) - 1)

    # Build advice
    advice = {
        "recovery_assessment": "",
        "todays_recommendations": [],
        "exercise_plan": [],
        "dietary_advice": "",
        "warning_signs": [],
        "motivation": "",
        "alert_doctor": False,
    }

    # Recovery assessment
    if recovery_score >= 75:
        advice["recovery_assessment"] = f"Excellent progress! Your recovery score is {recovery_score}/100. You're ahead of schedule."
    elif recovery_score >= 50:
        advice["recovery_assessment"] = f"Good progress. Your recovery score is {recovery_score}/100. You're on track for a {surgery_type} surgery recovery."
    elif recovery_score >= 25:
        advice["recovery_assessment"] = f"Your recovery score is {recovery_score}/100. Recovery is progressing but slower than expected. Focus on rest and following your plan."
    else:
        advice["recovery_assessment"] = f"Your recovery score is {recovery_score}/100. This is below expected. Please ensure you're following all medical instructions."

    # Trend analysis
    if trend == "improving":
        advice["recovery_assessment"] += " Your vitals show an improving trend."
    elif trend == "declining":
        advice["recovery_assessment"] += " ⚠️ Your vitals show a declining trend. Please monitor closely."
        advice["alert_doctor"] = True
    else:
        advice["recovery_assessment"] += " Your vitals are stable."

    # Complication risk
    if risk > 0.7:
        advice["alert_doctor"] = True
        advice["todays_recommendations"].append(
            "🚨 HIGH COMPLICATION RISK DETECTED. Please contact your healthcare provider immediately."
        )
    elif risk > 0.4:
        advice["todays_recommendations"].append(
            "⚠️ Moderate complication risk detected. Monitor your symptoms carefully and contact your doctor if anything worsens."
        )

    # Activity recommendations based on phase
    advice["todays_recommendations"].append(f"Current phase: {guidelines['activity_phases'][phase]}")

    # Vitals-based recommendations
    if patient_data["temperature"] > 38.0:
        advice["todays_recommendations"].append("Your temperature is elevated. Stay hydrated and monitor. Contact doctor if it exceeds 38.3°C.")
        advice["alert_doctor"] = patient_data["temperature"] > 38.3

    if patient_data["pain_level"] > 7:
        advice["todays_recommendations"].append("Your pain level is high. Ensure you're taking prescribed pain medication on schedule. Try repositioning and deep breathing.")

    if patient_data["sleep_hours"] < 5:
        advice["todays_recommendations"].append("You need more sleep for recovery. Try relaxation techniques before bed. Avoid screens 1 hour before sleep.")

    # Exercise plan
    if phase == 0:
        advice["exercise_plan"] = [
            "Deep breathing exercises: 10 slow breaths, 4 times today",
            "Ankle pumps: 10 repetitions each foot, every 2 hours",
            "Gentle bed-to-chair transfers with assistance",
        ]
    elif phase == 1:
        advice["exercise_plan"] = [
            "Walk in hallway/home: 5-10 minutes, 3 times today",
            "Seated arm raises: 10 repetitions, 2 sets",
            "Gentle stretching: 5 minutes morning and evening",
        ]
    elif phase == 2:
        advice["exercise_plan"] = [
            "Walk outdoors: 15-20 minutes, 2 times today",
            "Light resistance exercises with bodyweight",
            "Balance exercises: standing on one foot (with support), 30 seconds each",
        ]
    else:
        advice["exercise_plan"] = [
            "Walk 30 minutes at comfortable pace",
            "Light strengthening exercises as tolerated",
            "Gradual return to pre-surgery activity levels",
        ]

    # Dietary advice
    advice["dietary_advice"] = guidelines["dietary"]

    # Warning signs
    advice["warning_signs"] = guidelines["warning_signs"][:4]  # Top 4 for brevity

    # Motivation
    motivations = [
        f"Day {day} of recovery — every day you're getting stronger! 💪",
        f"You've made it through {day} days. Your body is healing and you're doing great!",
        f"Recovery is a marathon, not a sprint. Day {day} is another step forward.",
        f"Keep going! Your dedication to recovery on day {day} will pay off.",
    ]
    advice["motivation"] = motivations[day % len(motivations)]

    return advice


def format_advice_as_text(advice):
    """Format the advice dictionary as readable text."""
    if isinstance(advice, str):
        return advice  # Already formatted (from API)

    lines = []
    lines.append("### 🏥 Post-Surgery Recovery Coach — Daily Plan")

    lines.append(f"\n**📊 Recovery Assessment:**\n{advice['recovery_assessment']}")

    if advice.get("alert_doctor"):
        lines.append("\n🚨 **ALERT:** Please contact your healthcare provider today.")

    lines.append("\n**📋 Today's Recommendations:**")
    for rec in advice["todays_recommendations"]:
        lines.append(f"- {rec}")

    lines.append("\n**🏃 Exercise Plan:**")
    for ex in advice["exercise_plan"]:
        lines.append(f"- {ex}")

    lines.append(f"\n**🥗 Dietary Advice:**\n{advice['dietary_advice']}")

    lines.append("\n**⚠️ Watch for These Warning Signs:**")
    for sign in advice["warning_signs"]:
        lines.append(f"- {sign}")

    lines.append(f"\n💪 {advice['motivation']}")
    lines.append("\n---")
    lines.append("<small>⚕️ Disclaimer: This is AI-generated guidance. Always follow your doctor's specific instructions for your recovery.</small>")

    return "\n".join(lines)


def get_recovery_advice(patient_data, lstm_predictions, api_key=None, provider="gemini"):
    """
    Main entry point for the recovery coach.

    Args:
        patient_data: Dict with patient demographics and today's vitals
        lstm_predictions: Dict with LSTM model predictions
        api_key: Optional API key for Gemini/OpenAI/Groq
        provider: "gemini", "openai", "groq", or "offline"

    Returns:
        Formatted recovery advice string
    """
    error_msg = None

    if api_key and provider == "gemini":
        try:
            advice = generate_recovery_advice_gemini(patient_data, lstm_predictions, api_key)
            return f"*🤖 Powered by Gemini AI*\n\n{advice}"
        except Exception as e:
            error_msg = f"Gemini API error: {e}"

    if api_key and provider == "openai":
        try:
            advice = generate_recovery_advice_openai(patient_data, lstm_predictions, api_key)
            return f"*🤖 Powered by OpenAI*\n\n{advice}"
        except Exception as e:
            error_msg = f"OpenAI API error: {e}"

    if api_key and provider == "groq":
        try:
            advice = generate_recovery_advice_groq(patient_data, lstm_predictions, api_key)
            return f"*🤖 Powered by Groq (Llama 3.3)*\n\n{advice}"
        except Exception as e:
            error_msg = f"Groq API error: {e}"

    # Offline fallback with rule-based system
    advice = generate_recovery_advice_offline(patient_data, lstm_predictions)
    fallback_text = format_advice_as_text(advice)

    if error_msg:
        fallback_text = f"⚠️ **{error_msg}**\n\n*Showing offline rule-based advice instead:*\n\n{fallback_text}"

    return fallback_text
