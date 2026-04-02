# Diabetes model training script
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Load data
print("Loading diabetes dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, "diabetes_012_health_indicators_BRFSS2015.csv"))

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Separate features and target
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"].astype(int)

print(f"\nDataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Class distribution:\n{pd.Series(y_train_resampled).value_counts().sort_index()}")

# Save scaler for inference
joblib.dump(scaler, os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))

# ── Define multiple models to compare ──
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, eval_metric='mlogloss', verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, verbose=-1),
}

# ── Train & evaluate each model ──
print("\n" + "="*60)
print("COMPARING MODELS ON DIABETES DATASET")
print("="*60)

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "model": model}
    print(f"  Accuracy: {acc:.4f}")

# ── Print comparison table ──
print("\n" + "="*60)
print(f"{'Model':<25} {'Accuracy':>10}")
print("-"*37)
for name, res in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
    print(f"{name:<25} {res['accuracy']:>10.4f}")

# ── Select best model ──
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_accuracy = results[best_name]["accuracy"]
best_model = results[best_name]["model"]

print(f"\n>>> Best model: {best_name} with Accuracy = {best_accuracy:.4f}")

# ── Detailed evaluation of the best model ──
print(f"\n{'='*60}")
print(f"DETAILED EVALUATION — {best_name}")
print(f"{'='*60}")

y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

print(f"Accuracy:           {accuracy:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")
print(f"F1 Score (macro):    {f1_macro:.4f}")
print(f"ROC-AUC (weighted):  {roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No Diabetes', 'Prediabetes', 'Diabetes'])}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── Save best model ──
joblib.dump(best_model, os.path.join(MODELS_DIR, "diabetes_model.pkl"))
print(f"\nBest model ({best_name}) saved to models/diabetes_model.pkl")
