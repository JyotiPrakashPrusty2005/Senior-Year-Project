"""
Generate synthetic post-surgery recovery data for training LSTM and complication models.

Simulates daily vitals for patients recovering from surgery over 30 days.
Each patient has a recovery trajectory that is either:
  - Normal recovery
  - Slow recovery
  - Complication (infection/readmission)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_PATIENTS = 1000
MAX_DAYS = 30

# Surgery types
SURGERY_TYPES = ["cardiac", "orthopedic", "abdominal", "neurological"]


def generate_patient_profile():
    """Generate a single patient's demographics and surgery info."""
    age = np.random.randint(25, 80)
    gender = np.random.choice([0, 1])  # 0=female, 1=male
    bmi = np.round(np.random.normal(27, 5), 1)
    bmi = np.clip(bmi, 16, 45)
    diabetes = np.random.choice([0, 1], p=[0.75, 0.25])
    hypertension = np.random.choice([0, 1], p=[0.7, 0.3])
    surgery_type = np.random.choice(SURGERY_TYPES)
    smoking = np.random.choice([0, 1], p=[0.7, 0.3])

    # Higher age, BMI, comorbidities increase complication probability
    risk_score = (
        (age - 25) / 55 * 0.3
        + (bmi - 16) / 29 * 0.15
        + diabetes * 0.2
        + hypertension * 0.15
        + smoking * 0.2
    )
    risk_score = np.clip(risk_score, 0, 1)

    # Determine outcome based on risk
    outcome_probs = [
        max(0.6 - risk_score * 0.4, 0.2),   # normal
        max(0.25, risk_score * 0.3),          # slow
        min(risk_score * 0.4, 0.4),           # complication
    ]
    total = sum(outcome_probs)
    outcome_probs = [p / total for p in outcome_probs]
    outcome = np.random.choice(["normal", "slow", "complication"], p=outcome_probs)

    return {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "surgery_type": surgery_type,
        "smoking": smoking,
        "outcome": outcome,
    }


def generate_daily_vitals(profile, num_days=MAX_DAYS):
    """Generate time-series daily vitals based on patient profile and outcome."""
    outcome = profile["outcome"]
    rows = []

    for day in range(1, num_days + 1):
        t = day / num_days  # normalized time [0, 1]

        if outcome == "normal":
            # Smooth recovery
            pain = max(1, 8 - 7 * t + np.random.normal(0, 0.5))
            temperature = 37.0 + max(0, 0.8 * (1 - t) + np.random.normal(0, 0.15))
            heart_rate = 90 - 15 * t + np.random.normal(0, 3)
            bp_systolic = 130 - 10 * t + np.random.normal(0, 5)
            mobility_score = min(10, 2 + 8 * t + np.random.normal(0, 0.5))
            wound_status = max(0, 1 - t * 1.2)  # 0=healed, 1=fresh
            sleep_hours = min(9, 4 + 5 * t + np.random.normal(0, 0.5))
            recovery_score = min(100, 20 + 80 * t + np.random.normal(0, 3))

        elif outcome == "slow":
            # Slower improvement
            pain = max(1, 8 - 4 * t + np.random.normal(0, 0.7))
            temperature = 37.0 + max(0, 1.0 * (1 - 0.5 * t) + np.random.normal(0, 0.2))
            heart_rate = 95 - 8 * t + np.random.normal(0, 4)
            bp_systolic = 135 - 5 * t + np.random.normal(0, 6)
            mobility_score = min(10, 1 + 5 * t + np.random.normal(0, 0.7))
            wound_status = max(0, 1 - t * 0.7)
            sleep_hours = min(8, 3 + 4 * t + np.random.normal(0, 0.7))
            recovery_score = min(100, 10 + 55 * t + np.random.normal(0, 5))

        else:  # complication
            # Worsening around day 5-15, then partial recovery or plateau
            complication_peak = 0.3  # peak at 30% of recovery
            decay = np.exp(-3 * (t - complication_peak) ** 2)

            pain = max(1, 5 + 4 * decay + np.random.normal(0, 0.8))
            temperature = 37.0 + 1.5 * decay + 0.5 + np.random.normal(0, 0.25)
            heart_rate = 95 + 20 * decay + np.random.normal(0, 5)
            bp_systolic = 140 + 15 * decay + np.random.normal(0, 7)
            mobility_score = max(0, min(10, 2 + 3 * t - 4 * decay + np.random.normal(0, 0.5)))
            wound_status = min(1, 0.5 + 0.5 * decay)
            sleep_hours = max(2, 4 - 2 * decay + 3 * t + np.random.normal(0, 0.5))
            recovery_score = max(0, min(100, 15 + 30 * t - 30 * decay + np.random.normal(0, 5)))

        # Clip values to realistic ranges
        pain = np.clip(round(pain, 1), 0, 10)
        temperature = np.clip(round(temperature, 1), 35.5, 40.5)
        heart_rate = np.clip(round(heart_rate), 50, 150)
        bp_systolic = np.clip(round(bp_systolic), 80, 200)
        mobility_score = np.clip(round(mobility_score, 1), 0, 10)
        wound_status = np.clip(round(wound_status, 2), 0, 1)
        sleep_hours = np.clip(round(sleep_hours, 1), 0, 14)
        recovery_score = np.clip(round(recovery_score, 1), 0, 100)

        # Complication flag (1 if complication outcome AND in the danger window)
        complication_flag = 1 if (outcome == "complication" and 4 <= day <= 18) else 0

        rows.append({
            "day": day,
            "pain_level": pain,
            "temperature": temperature,
            "heart_rate": heart_rate,
            "bp_systolic": bp_systolic,
            "mobility_score": mobility_score,
            "wound_status": wound_status,
            "sleep_hours": sleep_hours,
            "recovery_score": recovery_score,
            "complication_flag": complication_flag,
        })

    return rows


def main():
    all_records = []

    for patient_id in range(1, NUM_PATIENTS + 1):
        profile = generate_patient_profile()
        daily_vitals = generate_daily_vitals(profile)

        for day_record in daily_vitals:
            record = {
                "patient_id": patient_id,
                "age": profile["age"],
                "gender": profile["gender"],
                "bmi": profile["bmi"],
                "diabetes": profile["diabetes"],
                "hypertension": profile["hypertension"],
                "surgery_type": profile["surgery_type"],
                "smoking": profile["smoking"],
                "outcome": profile["outcome"],
                **day_record,
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)

    output_path = os.path.join(os.path.dirname(__file__), "synthetic_recovery_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records for {NUM_PATIENTS} patients -> {output_path}")
    print(f"\nOutcome distribution:")
    print(df.groupby("patient_id")["outcome"].first().value_counts())
    print(f"\nSample columns: {list(df.columns)}")
    print(f"\nFirst patient sample:\n{df[df.patient_id == 1].head(5)}")


if __name__ == "__main__":
    main()
