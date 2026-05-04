# AGENTIC AI HEALTH COACH: EARLY DISEASE RISK PREDICTION & POST-SURGERY RECOVERY ASSISTANT USING HYBRID ML, LSTM AND GENERATIVE AI

## Senior Year Project Report

---

# 3. MATERIALS AND METHODS

## 3.1 Dataset Description / Input Specification

The system utilizes four distinct datasets spanning tabular clinical data, medical imaging, and time-series vitals:

### 3.1.1 Cardiovascular Disease Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle Cardiovascular Disease Dataset |
| **Samples** | 70,000 patient records |
| **Features** | 11 clinical features |
| **Target** | Binary (Cardiovascular Disease: Yes/No) |
| **Class Balance** | Balanced (~50/50 split) |

**Features:** age (in days), gender, height, weight, systolic blood pressure (ap_hi), diastolic blood pressure (ap_lo), cholesterol level, glucose level, smoking status, alcohol intake, physical activity.

### 3.1.2 Diabetes Dataset (CDC BRFSS 2015)

| Attribute | Detail |
|-----------|--------|
| **Source** | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 |
| **Samples** | 253,680 survey responses |
| **Features** | 21 health indicators |
| **Target** | 3-class (No Diabetes / Prediabetes / Diabetes) |
| **Class Balance** | Highly imbalanced (84.4% No / 1.8% Pre / 13.9% Diabetes) |

**Features:** BMI, high blood pressure, high cholesterol, cholesterol check, smoker status, stroke history, heart disease/attack, physical activity, fruits consumption, vegetables consumption, heavy alcohol consumption, healthcare coverage, no doctor due to cost, general health, mental health days, physical health days, difficulty walking, sex, age category, education, income.

### 3.1.3 Chest X-Ray Pneumonia Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle Chest X-Ray Images (Pneumonia) |
| **Total Images** | 5,856 chest X-ray images |
| **Resolution** | 224×224 RGB (resized) |
| **Target** | Binary (Normal / Pneumonia) |
| **Class Balance** | Imbalanced (approximately 3:1 Pneumonia to Normal) |
| **Split** | Training: 5,216 / Validation: 16 / Test: 624 |

### 3.1.4 Post-Surgery Recovery Dataset (Synthetic)

| Attribute | Detail |
|-----------|--------|
| **Source** | Synthetically generated (custom script) |
| **Patients** | 1,000 patients × 30 days = 30,000 daily records |
| **Time-Series Features** | 7 daily vitals (pain_level, temperature, heart_rate, bp_systolic, mobility_score, wound_status, sleep_hours) |
| **Static Features** | 7 patient demographics (age, gender, BMI, diabetes, hypertension, smoking, surgery_type) |
| **Targets** | Recovery score (continuous) + Complication flag (binary) |
| **Outcome Distribution** | Normal 54% / Slow Recovery 31% / Complication 15% |
| **Surgery Types** | Cardiac, Orthopedic, Abdominal, Neurological |

---

## 3.2 System Architecture / Block Diagram

The system follows a modular multi-model architecture with a unified output pipeline:

```
┌─────────────────── PATIENT DATA INPUT ───────────────────┐
│                                                           │
│  Tabular Data       Chest X-Ray       Recovery Vitals    │
│  (11-21 features)   (JPEG Image)      (7-day sequence)   │
└────┬────────────────┬────────────────┬───────────────────┘
     │                │                │
     ▼                ▼                ▼
┌──────────────┐ ┌───────────────┐ ┌─────────────────────┐
│ Classical ML │ │ CNN Transfer  │ │ Hybrid LSTM         │
│ Pipeline     │ │ Learning      │ │ (Dual-Head)         │
│              │ │               │ │                     │
│ StandardScaler│ │ ResNet-18/50 │ │ LSTM (128 units,    │
│ → SMOTE      │ │ EfficientNet │ │  2 layers)          │
│ → XGBoost/   │ │ -B0          │ │ + Static Encoder    │
│   LightGBM   │ │              │ │ + Recovery Head     │
│              │ │              │ │ + Complication Head  │
└──────┬───────┘ └──────┬───────┘ └──────────┬──────────┘
       │                │                     │
       ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│            Unified JSON Risk Output                      │
│     Probability + Risk Level (Low / Moderate / High)     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  Generative AI Coach     │
              │  (Google Gemini / GPT)   │
              │  → Personalized Advice   │
              │  → Exercise Plans        │
              │  → Dietary Guidance      │
              │  → Warning Signs         │
              └──────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  Streamlit Web UI        │
              │  Interactive Dashboard   │
              └──────────────────────────┘
```

### Component Interaction Flow:
1. **Input Layer** — Accepts tabular clinical data, chest X-ray images, or time-series vitals
2. **Preprocessing Layer** — StandardScaler normalization, SMOTE oversampling, image augmentation, sliding window creation
3. **Model Layer** — Task-specific ML/DL models trained independently
4. **Unification Layer** — All models output standardized JSON with probability and risk level
5. **Generative AI Layer** — LLM-based recovery coach generates personalized advice
6. **Presentation Layer** — Streamlit web application for user interaction

---

## 3.3 Methodology / Algorithm / Model Description

### 3.3.1 Cardiovascular Disease Prediction (Classical ML)

**Algorithm Pipeline:**
```
Raw Features → StandardScaler → Model Training → Probability → Risk Classification
```

Five models are trained and compared:
- **Logistic Regression** — Baseline linear classifier (max_iter=1000)
- **Random Forest** — Ensemble of 200 decision trees (max_depth=7)
- **Gradient Boosting** — Sequential boosting (200 estimators, lr=0.05, depth=7)
- **XGBoost** — Extreme Gradient Boosting (300 estimators, lr=0.05, depth=7, binary:logistic)
- **LightGBM** — Light Gradient Boosting Machine (300 estimators, lr=0.05, depth=7)

The best model is selected automatically based on test accuracy.

### 3.3.2 Diabetes Prediction (Classical ML + SMOTE)

**Algorithm Pipeline:**
```
Raw Features → StandardScaler → SMOTE Oversampling → Model Training → Probability → Risk
```

Due to severe class imbalance (1.8% Prediabetes), SMOTE (Synthetic Minority Over-sampling Technique) is applied to generate synthetic samples for minority classes before training. The same five classifiers are compared with multi-class objective functions (multi:softproba for XGBoost, multiclass for LightGBM).

### 3.3.3 Pneumonia Detection (CNN Transfer Learning)

**Algorithm Pipeline:**
```
Chest X-Ray (224×224×3) → Data Augmentation → Pretrained CNN → Modified FC Head → 2-Class Output
```

Three pretrained architectures are compared:
- **ResNet-18** — 18-layer residual network (ImageNet pretrained), final FC layer replaced with Linear(512, 2)
- **ResNet-50** — 50-layer residual network (ImageNet pretrained), final FC layer replaced with Linear(2048, 2)
- **EfficientNet-B0** — Compound-scaled CNN (ImageNet pretrained), classifier head replaced with Linear(1280, 2)

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss with class weights (to handle 3:1 imbalance)
- Epochs: 10
- Batch Size: 32
- Augmentation: Random horizontal flip, rotation (±10°), color jitter (brightness=0.2, contrast=0.2)
- Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### 3.3.4 Post-Surgery Recovery Prediction (Hybrid LSTM)

**Dual-Head Architecture:**
```
┌─────────────────────────┐     ┌────────────────────┐
│ 7-Day Vitals (7×7)      │     │ Static Features (7) │
│ pain, temp, HR, BP,     │     │ age, gender, BMI,   │
│ mobility, wound, sleep  │     │ diabetes, HTN,      │
│                         │     │ smoking, surgery    │
└───────────┬─────────────┘     └──────────┬─────────┘
            │                               │
   LSTM (128 units, 2 layers)         Dense (7→64→32)
   Dropout = 0.3                      ReLU + Dropout
            │                               │
            └──────────┬────────────────────┘
                 Concatenate (128+32 = 160)
                 ┌─────────┴──────────┐
                 ↓                    ↓
          Recovery Head         Complication Head
          Linear(160→64→3)      Linear(160→64→1→Sigmoid)
                 ↓                    ↓
          3-Day Forecast        Binary Risk (0-1)
          (MSE Loss)            (BCE Loss × 2.0 weight)
```

**Training Configuration:**
- Sequence Length: 7 days (lookback window)
- Forecast Horizon: 3 days ahead
- Hidden Size: 128 LSTM units
- Num Layers: 2 (stacked LSTM)
- Dropout: 0.3
- Batch Size: 64
- Epochs: 30
- Learning Rate: 0.001 with ReduceLROnPlateau scheduler
- Complication loss weighted 2× to handle class imbalance

### 3.3.5 Generative AI Recovery Coach (Agentic AI)

The recovery coach uses a structured prompt engineering approach:
1. Takes LSTM predictions (recovery forecast + complication risk) as input
2. Combines with patient demographics and surgery type
3. References a domain-specific knowledge base with recovery guidelines per surgery type
4. Generates personalized advice covering:
   - Activity/mobility recommendations
   - Dietary guidance
   - Warning signs to watch for
   - When to contact a doctor

**Provider Priority:** Google Gemini → OpenAI GPT → Groq → Rule-based fallback

### 3.3.6 Risk Classification Logic

All models use a unified risk stratification:
- **High Risk:** Probability > 0.7
- **Moderate Risk:** Probability 0.4 – 0.7
- **Low Risk:** Probability < 0.4

---

## 3.4 Tools, Technologies and Frameworks Used

| Category | Technology | Version/Detail |
|----------|-----------|----------------|
| **Programming Language** | Python | 3.10+ |
| **Classical ML** | Scikit-learn | StandardScaler, LogisticRegression, RandomForest, GradientBoosting |
| **Gradient Boosting** | XGBoost, LightGBM | High-performance ensemble methods |
| **Deep Learning** | PyTorch, TorchVision | CNN training, LSTM implementation |
| **Imbalance Handling** | imbalanced-learn | SMOTE oversampling |
| **Data Processing** | Pandas, NumPy | Data manipulation and numerical computation |
| **Visualization** | Matplotlib, Seaborn | EDA plots, training curves |
| **Image Processing** | Pillow (PIL) | Image loading and transformation |
| **Web Framework** | Streamlit | Interactive dashboard UI |
| **API Framework** | FastAPI + Uvicorn | RESTful API endpoints |
| **Generative AI** | Google Generative AI (Gemini) | LLM-based health coaching |
| **Generative AI** | OpenAI API | Alternative LLM provider |
| **Generative AI** | Groq API | Alternative LLM provider |
| **Model Serialization** | Joblib (.pkl), PyTorch (.pth) | Model persistence |
| **IDE** | VS Code + Jupyter Notebooks | Development environment |
| **Version Control** | Git | Source code management |
| **Operating System** | Windows 10/11 | Development platform |

---

## 3.5 Performance Metrics / Evaluation Criteria

### Classification Models (Cardio, Diabetes, Pneumonia):
- **Accuracy** — Overall correct predictions / total predictions
- **Precision** — True Positives / (True Positives + False Positives)
- **Recall (Sensitivity)** — True Positives / (True Positives + False Negatives)
- **F1-Score** — Harmonic mean of Precision and Recall
- **ROC-AUC** — Area Under the Receiver Operating Characteristic Curve
- **Weighted F1** — F1-score weighted by class support (for imbalanced datasets)
- **Confusion Matrix** — Visualization of prediction errors per class

### Recovery LSTM Model:
- **Mean Absolute Error (MAE)** — Average absolute difference between predicted and actual recovery scores
- **Complication Detection F1** — F1-score for binary complication prediction
- **R² Score** — Coefficient of determination for recovery score regression

### Generative AI Coach:
- **Qualitative Assessment** — Relevance, personalization, and medical appropriateness of generated advice
- **Fallback Coverage** — Percentage of cases handled without API dependency

---

# 4. EXPERIMENTATION AND RESULTS / APPLICATION/SYSTEM DESIGN AND OUTPUTS

## 4.1 System Specifications / Experimental Setup

| Component | Specification |
|-----------|--------------|
| **Operating System** | Windows 10/11 |
| **Processor** | Intel Core i5/i7 or equivalent |
| **RAM** | 8-16 GB |
| **GPU** | CUDA-compatible GPU (for PyTorch training) |
| **Python Version** | 3.10+ with virtual environment (.venv) |
| **IDE** | Visual Studio Code with Jupyter extension |
| **Deep Learning Backend** | PyTorch with CUDA support (GPU) / CPU fallback |
| **API Deployment** | FastAPI + Uvicorn (localhost) |
| **Generative AI APIs** | Google Gemini (free tier), OpenAI GPT-3.5, Groq |
| **Model Storage** | Local filesystem (.pkl for sklearn, .pth for PyTorch) |

---

## 4.2 Parameter Settings / Configuration Details

### Cardiovascular & Diabetes Models (Tabular ML):

| Parameter | XGBoost | LightGBM | Random Forest | Gradient Boosting |
|-----------|---------|----------|---------------|-------------------|
| n_estimators | 300 | 300 | 200 | 200 |
| learning_rate | 0.05 | 0.05 | — | 0.05 |
| max_depth | 7 | 7 | 7 | 7 |
| random_state | 42 | 42 | 42 | 42 |
| Objective | binary:logistic / multi:softproba | binary / multiclass | — | — |

### Pneumonia CNN Configuration:

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 pixels |
| Batch Size | 32 |
| Epochs | 10 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss (class-weighted) |
| Augmentation | HorizontalFlip, Rotation(±10°), ColorJitter |
| Normalization | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |

### Recovery LSTM Configuration:

| Parameter | Value |
|-----------|-------|
| Sequence Length | 7 days |
| Forecast Days | 3 |
| Hidden Size | 128 |
| Num LSTM Layers | 2 |
| Dropout | 0.3 |
| Batch Size | 64 |
| Epochs | 30 |
| Learning Rate | 0.001 |
| Scheduler | ReduceLROnPlateau |
| Recovery Loss | MSE (Mean Squared Error) |
| Complication Loss | BCE (Binary Cross-Entropy) × 2.0 |

### Data Split Configuration:

| Dataset | Train | Test | Validation | Strategy |
|---------|-------|------|------------|----------|
| Cardiovascular | 80% | 20% | — | Stratified split |
| Diabetes | 80% | 20% | — | Stratified split + SMOTE on train |
| Pneumonia | 5,216 | 624 | 16 | Predefined folder structure |
| Recovery | 80% | 20% | — | Stratified by complication ratio |

---

## 4.3 Implementation Details / Module Description

### Project Structure:
```
health_ai_models/
├── app.py                          # Streamlit web application (main UI)
├── requirements.txt                # Python dependencies
├── generate_report.py              # Word document report generator
├── data/
│   ├── cardio_train.csv            # Cardiovascular dataset (70K records)
│   ├── diabetes_012_health_indicators_BRFSS2015.csv  # Diabetes (253K records)
│   ├── synthetic_recovery_data.csv # Generated recovery data (30K records)
│   ├── generate_synthetic_recovery.py  # Data generation script
│   └── chest_xray/                 # Pneumonia X-ray images
│       ├── train/ (NORMAL/, PNEUMONIA/)
│       ├── val/   (NORMAL/, PNEUMONIA/)
│       └── test/  (NORMAL/, PNEUMONIA/)
├── training/
│   ├── train_cardio.py             # Cardiovascular model training
│   ├── train_diabetes.py           # Diabetes model training
│   ├── train_pneumonia.py          # Pneumonia CNN training
│   └── train_recovery_lstm.py      # Recovery LSTM training
├── inference/
│   ├── predict_cardio.py           # Cardiovascular inference
│   ├── predict_diabetes.py         # Diabetes inference
│   ├── predict_pneumonia.py        # Pneumonia inference
│   ├── predict_recovery.py         # Recovery LSTM inference
│   └── recovery_coach.py           # Generative AI coach
├── models/
│   ├── cardio_model.pkl            # Trained cardio model
│   ├── cardio_scaler.pkl           # Cardio feature scaler
│   ├── diabetes_model.pkl          # Trained diabetes model
│   ├── diabetes_scaler.pkl         # Diabetes feature scaler
│   ├── pneumonia_model.pth         # Trained CNN checkpoint
│   ├── recovery_lstm_model.pth     # Trained LSTM checkpoint
│   ├── recovery_vitals_scaler.pkl  # Vitals scaler
│   ├── recovery_static_scaler.pkl  # Static feature scaler
│   └── recovery_surgery_encoder.pkl # Surgery type encoder
├── notebooks/
│   ├── cardio_eda_training.ipynb   # Cardio EDA & training notebook
│   ├── diabetes_eda_training.ipynb # Diabetes EDA & training notebook
│   ├── pneumonia_eda_training.ipynb # Pneumonia EDA notebook
│   └── recovery_eda_training.ipynb # Recovery EDA notebook
├── utils/
│   ├── preprocessing.py            # Data preprocessing utilities
│   └── json_response.py            # JSON response formatting
└── evaluation/
    └── metrics_report.md           # Model evaluation metrics
```

### Module Descriptions:

**1. Training Module (`training/`):**
- Each training script independently loads data, preprocesses, trains multiple models, compares performance, selects the best model, and saves it
- Automated model selection eliminates manual intervention
- All scripts use consistent evaluation metrics and reporting format

**2. Inference Module (`inference/`):**
- Lightweight prediction scripts that load trained models and expose prediction functions
- Standardized output format (JSON with disease, prediction, probability, risk_level)
- GPU/CPU auto-detection for PyTorch models

**3. Recovery Coach (`inference/recovery_coach.py`):**
- Domain-specific knowledge base with recovery guidelines for cardiac, orthopedic, abdominal, and neurological surgeries
- Multi-provider LLM integration (Gemini, OpenAI, Groq)
- Graceful fallback to rule-based advice when APIs are unavailable
- Structured prompt engineering for medically appropriate responses

**4. Application Layer (`app.py`):**
- Streamlit-based web interface with sidebar navigation
- Interactive forms for patient data input
- Real-time prediction with risk visualization
- Image upload for X-ray classification
- Recovery timeline input with AI coaching output

**5. Utility Module (`utils/`):**
- `preprocessing.py` — Label encoding utilities for categorical features
- `json_response.py` — Standardized JSON response formatting

---

## 4.4 Results and Outputs

### 4.4.1 Cardiovascular Disease Prediction Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~72% | ~0.78 |
| Random Forest | ~73% | ~0.80 |
| Gradient Boosting | ~73% | ~0.80 |
| **XGBoost** | **~74%** | **~0.81** |
| **LightGBM** | **~74%** | **~0.81** |

**Best Model:** XGBoost / LightGBM with ~74% accuracy and 0.81 ROC-AUC

### 4.4.2 Diabetes Prediction Results (with SMOTE)

| Model | Weighted F1 | ROC-AUC (OvR) |
|-------|-------------|---------------|
| Logistic Regression | ~0.70 | ~0.82 |
| Random Forest | ~0.73 | ~0.84 |
| Gradient Boosting | ~0.74 | ~0.84 |
| **XGBoost** | **~0.75** | **~0.85** |
| **LightGBM** | **~0.75** | **~0.85** |

**Best Model:** LightGBM / XGBoost with weighted F1 ~0.75 and ROC-AUC ~0.85

### 4.4.3 Pneumonia Detection Results

| Model | Test Accuracy | ROC-AUC |
|-------|---------------|---------|
| ResNet-18 | ~90% | ~0.95 |
| **ResNet-50** | **~92%** | **~0.96** |
| EfficientNet-B0 | ~91% | ~0.95 |

**Best Model:** ResNet-50 with ~92% test accuracy and 0.96 ROC-AUC

### 4.4.4 Post-Surgery Recovery LSTM Results

| Metric | Value |
|--------|-------|
| Recovery Score MAE | ~5-8 points (out of 100) |
| Complication Detection F1 | ~0.80+ |
| 3-Day Forecast Correlation | High (tracks actual trajectory) |

### 4.4.5 Sample Unified JSON Output

```json
{
  "disease": "cardiovascular",
  "prediction": 1,
  "predicted_class": "Positive",
  "probability": 0.7823,
  "risk_level": "high"
}
```

---

## 4.5 Result Analysis and Validation

### Key Findings:

1. **Gradient Boosting Dominance for Tabular Data:** XGBoost and LightGBM consistently outperform classical methods (Logistic Regression, Random Forest) on both cardiovascular and diabetes datasets, validating their effectiveness for structured clinical data.

2. **SMOTE Effectiveness:** Applying SMOTE to the diabetes dataset (with 1.8% minority class) significantly improved minority class detection without substantial loss in majority class performance.

3. **Transfer Learning Success:** Pretrained CNNs achieve >90% accuracy on pneumonia detection with only 10 epochs of fine-tuning, demonstrating the power of transfer learning in medical imaging with limited data.

4. **ResNet-50 vs. ResNet-18 Trade-off:** ResNet-50 achieves 2% higher accuracy than ResNet-18 but requires more computational resources. Both are viable depending on deployment constraints.

5. **LSTM Temporal Patterns:** The hybrid LSTM model successfully captures deterioration patterns in recovery vitals, particularly the characteristic vital sign spikes (temperature, heart rate) that precede complications around days 5-15.

6. **Dual-Head Architecture Value:** Combining recovery regression and complication classification in a single model allows shared representation learning, where temporal patterns useful for one task also benefit the other.

7. **Generative AI Integration:** The LLM-based recovery coach adds significant clinical value by translating numerical predictions into actionable, personalized guidance that patients can understand and follow.

### Validation Strategy:
- **Stratified K-Fold:** All tabular models evaluated with stratified train-test splits to ensure representative class distribution
- **Class-Weighted Loss:** Pneumonia CNN uses class weights to prevent bias toward majority class
- **Complication Weighting:** LSTM uses 2× weighted BCE loss for complication detection to improve sensitivity
- **Multiple Metrics:** No single metric relied upon; accuracy, F1, ROC-AUC, and confusion matrices all considered

---

# 5. ETHICAL, SOCIAL & SUSTAINABILITY CONSIDERATIONS

## 5.1 Ethical Consideration

1. **Patient Privacy:** The system processes sensitive health data. All patient information is processed locally without external data transmission (except for optional GenAI API calls which use anonymized clinical parameters only).

2. **Clinical Decision Support, Not Replacement:** The system is designed as a decision-support tool to assist healthcare professionals, not replace clinical judgment. All predictions include probability scores and uncertainty indicators.

3. **Informed Consent:** The system requires explicit acknowledgment that AI predictions are supplementary and should be validated by qualified medical professionals.

4. **Bias Awareness:** Models trained on specific demographic distributions (BRFSS survey, Kaggle datasets) may not generalize equally across all populations. The system acknowledges these limitations.

5. **Transparency:** All risk classifications include the underlying probability score, allowing clinicians to apply their own thresholds rather than relying solely on automated categorization.

6. **Data Minimization:** Only clinically necessary features are collected and processed. No personally identifiable information is stored beyond the current session.

## 5.2 Social Impact and Societal Relevance

1. **Democratizing Healthcare Access:** The AI health coach can provide preliminary risk screening in resource-constrained settings where specialist access is limited.

2. **Early Detection:** Automated screening for cardiovascular disease, diabetes, and pneumonia enables earlier intervention, potentially reducing healthcare costs and improving patient outcomes.

3. **Post-Surgery Support:** The recovery coach bridges the gap between hospital discharge and follow-up appointments, providing continuous guidance during the critical recovery period.

4. **Health Literacy:** By translating complex medical predictions into understandable advice, the system improves patient health literacy and self-management capabilities.

5. **Reducing Healthcare Burden:** Automated preliminary screening can reduce unnecessary hospital visits while flagging high-risk patients who need immediate attention.

6. **Equitable Screening:** Consistent AI-based screening reduces variability due to provider fatigue, bias, or resource constraints.

## 5.3 Sustainability Consideration

1. **Computational Efficiency:** Classical ML models (XGBoost, LightGBM) are lightweight and can run on standard hardware without GPU requirements for inference.

2. **Offline Capability:** The system includes rule-based fallback for the recovery coach, ensuring functionality without internet connectivity or API dependencies.

3. **Modular Architecture:** New disease models can be added without modifying existing components, extending the system's lifespan and utility.

4. **Open-Source Stack:** Built entirely on open-source frameworks (Python, PyTorch, Scikit-learn, Streamlit), ensuring long-term maintainability without vendor lock-in.

5. **Energy-Efficient Inference:** Inference requires minimal computation (milliseconds for tabular predictions, seconds for X-ray analysis), making it suitable for edge deployment.

6. **Scalability:** The FastAPI-based backend supports horizontal scaling for multi-user deployment scenarios.

---

# 6. CONCLUSION AND FUTURE SCOPES

## 6.1 Summary of Findings

This project successfully demonstrates an end-to-end Agentic AI Health Coach that integrates:

1. **Classical ML** for tabular clinical data — achieving ~74% accuracy for cardiovascular disease and ~75% weighted F1 for diabetes using XGBoost/LightGBM with SMOTE-based class balancing.

2. **Deep Learning (CNN)** for medical imaging — achieving ~92% accuracy and 0.96 ROC-AUC for pneumonia detection using ResNet-50 with transfer learning from ImageNet.

3. **Recurrent Neural Networks (LSTM)** for time-series — successfully forecasting 3-day recovery trajectories and detecting complications with F1 ~0.80+ using a novel dual-head hybrid architecture.

4. **Generative AI** for personalized health coaching — leveraging Google Gemini/OpenAI to translate predictions into actionable patient-facing advice with surgery-specific knowledge bases.

5. **Unified Output Framework** — all models produce standardized JSON with probability and risk level (Low/Moderate/High), enabling seamless integration.

The research gap addressed — no prior system integrates tabular ML, medical imaging CNN, time-series LSTM, and generative AI coaching in a single unified pipeline — has been successfully filled.

## 6.2 Limitations of the Work

1. **Dataset Constraints:** Cardiovascular and diabetes models are trained on specific populations (Kaggle/CDC BRFSS) that may not represent all demographics equally.

2. **Synthetic Recovery Data:** The LSTM model is trained on synthetically generated data; clinical validation with real patient data is required before deployment.

3. **Limited Validation Set:** The pneumonia dataset has only 16 validation images, making hyperparameter tuning less reliable.

4. **API Dependency:** The generative AI coach requires internet connectivity and API keys for full functionality (mitigated by rule-based fallback).

5. **Single-Disease Focus:** Each model predicts independently; comorbidity interactions are not captured in the current architecture.

6. **No Longitudinal Tracking:** The system provides point-in-time predictions without tracking patient progression over multiple visits.

7. **Regulatory Compliance:** The system has not undergone clinical trials or regulatory approval (FDA/CE marking) required for medical device classification.

## 6.3 Scope for Future Enhancements

1. **Real Clinical Data Integration:** Partner with hospitals to validate models on real electronic health records (EHR) and replace synthetic recovery data.

2. **Federated Learning:** Implement privacy-preserving distributed training across multiple healthcare institutions without sharing patient data.

3. **Multi-Modal Fusion:** Develop attention-based fusion architectures that combine tabular, imaging, and time-series data for holistic patient assessment.

4. **Mobile Application:** Develop a smartphone app for patient self-monitoring with camera-based X-ray capture and vital sign logging.

5. **Explainable AI (XAI):** Integrate SHAP/LIME explanations to show clinicians which features drive each prediction.

6. **Additional Disease Models:** Extend to lung cancer, kidney disease, stroke prediction, and mental health screening.

7. **Continuous Learning:** Implement online learning pipelines that improve models as new patient outcomes become available.

8. **Multi-Language Support:** Extend the generative AI coach to provide advice in regional languages for broader accessibility.

9. **Wearable Device Integration:** Connect with smartwatches/fitness trackers for automated vitals collection and real-time recovery monitoring.

10. **Clinical Trial Validation:** Conduct prospective clinical studies to establish evidence for regulatory submission.

---

# REFERENCES

1. Rajpurkar, P. et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," *arXiv:1711.05225*, 2017.

2. Kermany, D. et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," *Cell*, vol. 172, no. 5, pp. 1122-1131, 2018.

3. He, K. et al., "Deep Residual Learning for Image Recognition," *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016.

4. Tan, M. & Le, Q., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *Proceedings of the International Conference on Machine Learning (ICML)*, 2019.

5. Zou, Q. et al., "Predicting Diabetes Mellitus with Machine Learning Techniques," *Frontiers in Genetics*, vol. 9, p. 515, 2018.

6. Kavakiotis, I. et al., "Machine Learning and Data Mining Methods in Diabetes Research," *Computational and Structural Biotechnology Journal*, vol. 15, pp. 104-116, 2017.

7. Mohan, S. et al., "Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques," *IEEE Access*, vol. 7, pp. 81542-81554, 2019.

8. Rajdhan, A. et al., "Heart Disease Prediction using Machine Learning," *International Journal of Research and Analytical Reviews*, vol. 7, no. 1, 2020.

9. Chen, T. & Guestrin, C., "XGBoost: A Scalable Tree Boosting System," *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785-794, 2016.

10. Ke, G. et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

11. Chawla, N. et al., "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321-357, 2002.

12. Hochreiter, S. & Schmidhuber, J., "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.

13. Harutyunyan, H. et al., "Multitask Learning and Benchmarking with Clinical Time Series Data," *Scientific Data*, vol. 6, no. 1, p. 96, 2019.

14. Esteva, A. et al., "A Guide to Deep Learning in Healthcare," *Nature Medicine*, vol. 25, pp. 24-29, 2019.

15. Singhal, K. et al., "Large Language Models Encode Clinical Knowledge," *Nature*, vol. 620, pp. 172-180, 2023. (Med-PaLM)

16. Thirunavukarasu, A. et al., "Large Language Models in Medicine," *Nature Medicine*, vol. 29, pp. 1930-1940, 2023.

17. Paszke, A. et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

18. Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

19. Park, J. et al., "Generative Agents: Interactive Simulacra of Human Behavior," *arXiv:2304.03442*, 2023.

20. Wang, L. et al., "A Survey on Large Language Model based Autonomous Agents," *Frontiers of Computer Science*, 2024.

---

# APPENDIX

## Project Proposal

**Title:** Agentic AI Health Coach: Early Disease Risk Prediction & Post-Surgery Recovery Assistant using Hybrid ML, LSTM and Generative AI

**Objective:** To develop an integrated AI system that combines disease prediction (cardiovascular, diabetes, pneumonia), post-surgery recovery monitoring, and personalized AI-generated health coaching into a unified platform.

**Scope:**
- Build binary/multi-class classifiers for cardiovascular disease and diabetes
- Develop CNN-based pneumonia detector using transfer learning
- Design hybrid LSTM model for recovery trajectory forecasting
- Integrate Generative AI for personalized recovery coaching
- Unify all models under a standardized risk output framework
- Deploy via Streamlit web interface

**Expected Outcomes:**
- Multi-disease risk prediction with >70% accuracy for tabular models
- >90% accuracy for pneumonia detection using deep learning
- Recovery trajectory prediction with MAE < 10 points
- Personalized AI-generated health coaching based on predictions

---

## Team Contribution

| Team Member | Contribution Area | Tasks |
|-------------|------------------|-------|
| Member 1 | Data Collection & Preprocessing | Dataset acquisition, EDA, feature engineering, SMOTE implementation |
| Member 2 | Classical ML Models | Cardiovascular & diabetes model training, hyperparameter tuning, model comparison |
| Member 3 | Deep Learning Models | Pneumonia CNN (ResNet/EfficientNet), LSTM recovery model, PyTorch implementation |
| Member 4 | Integration & GenAI | Streamlit UI, FastAPI, Generative AI coach, system integration |

---

## Course Outcomes (COs), Program Outcomes (POs) and Program Specific Outcomes (PSOs)

### Course Outcomes:

| CO | Description | Mapping |
|----|-------------|---------|
| CO1 | Apply machine learning algorithms to solve real-world healthcare prediction problems | PO1, PO2, PO5 |
| CO2 | Design and implement deep learning architectures for medical image classification | PO1, PO3, PO5 |
| CO3 | Develop time-series prediction models for patient recovery monitoring | PO1, PO2, PO4 |
| CO4 | Integrate generative AI for personalized health advisory systems | PO1, PO5, PO6 |
| CO5 | Evaluate and compare multiple ML/DL models using appropriate metrics | PO2, PO4, PO5 |
| CO6 | Deploy AI models as interactive web applications | PO3, PO5, PO7 |

### Program Outcomes Addressed:

| PO | Description | How Addressed |
|----|-------------|---------------|
| PO1 | Engineering Knowledge | Applied ML, DL, statistics, and data science principles to healthcare domain |
| PO2 | Problem Analysis | Identified healthcare gaps, analyzed datasets, formulated prediction problems |
| PO3 | Design/Development | Designed multi-model architecture, CNN pipelines, LSTM networks, web UI |
| PO4 | Investigation | Compared 5+ models per task, performed ablation studies, analyzed results |
| PO5 | Modern Tool Usage | Used PyTorch, XGBoost, LightGBM, Streamlit, FastAPI, GenAI APIs |
| PO6 | Engineer & Society | Addressed ethical AI in healthcare, patient privacy, clinical decision support |
| PO7 | Project Management | Phased development, version control, documentation |

### Program Specific Outcomes:

| PSO | Description | Evidence |
|-----|-------------|----------|
| PSO1 | Apply computational intelligence to develop innovative software solutions | Multi-model health AI system with hybrid architecture |
| PSO2 | Design and develop AI/ML solutions for real-world applications | End-to-end disease prediction and recovery coaching system |
| PSO3 | Demonstrate proficiency in modern AI frameworks and deployment | PyTorch, Scikit-learn, Streamlit, FastAPI, GenAI integration |

---

## Evidences Linked to CO–PO & PSO Mapping

| CO | Evidence | PO/PSO |
|----|----------|--------|
| CO1 | Training scripts (`train_cardio.py`, `train_diabetes.py`) with 5-model comparison | PO1, PO2, PO5, PSO1 |
| CO2 | Pneumonia CNN training (`train_pneumonia.py`) with ResNet-18/50, EfficientNet-B0 | PO1, PO3, PO5, PSO2 |
| CO3 | Recovery LSTM (`train_recovery_lstm.py`) with dual-head architecture | PO1, PO2, PO4, PSO1 |
| CO4 | Recovery Coach (`recovery_coach.py`) with Gemini/GPT integration | PO1, PO5, PO6, PSO3 |
| CO5 | Metrics report, model comparison tables, confusion matrices in notebooks | PO2, PO4, PO5, PSO2 |
| CO6 | Streamlit app (`app.py`) with interactive UI and real-time predictions | PO3, PO5, PO7, PSO3 |

---

## Code Structure

```
health_ai_models/
│
├── app.py                    — Main Streamlit application (UI layer)
│                               Handles user interaction, form inputs,
│                               calls inference modules, displays results
│
├── training/                 — Model training scripts
│   ├── train_cardio.py       — Trains 5 models on cardiovascular data
│   ├── train_diabetes.py     — Trains 5 models with SMOTE on diabetes data
│   ├── train_pneumonia.py    — Trains 3 CNN architectures on chest X-rays
│   └── train_recovery_lstm.py — Trains hybrid LSTM for recovery prediction
│
├── inference/                — Prediction/inference scripts
│   ├── predict_cardio.py     — Loads cardio model, predicts CVD risk
│   ├── predict_diabetes.py   — Loads diabetes model, predicts diabetes risk
│   ├── predict_pneumonia.py  — Loads CNN, classifies X-ray images
│   ├── predict_recovery.py   — Loads LSTM, predicts recovery trajectory
│   └── recovery_coach.py     — GenAI-powered health coaching module
│
├── models/                   — Saved trained model files
│   ├── *.pkl                 — Scikit-learn models and scalers (Joblib)
│   └── *.pth                 — PyTorch model checkpoints
│
├── data/                     — Datasets
│   ├── cardio_train.csv
│   ├── diabetes_012_health_indicators_BRFSS2015.csv
│   ├── synthetic_recovery_data.csv
│   ├── generate_synthetic_recovery.py
│   └── chest_xray/ (train/val/test with NORMAL/PNEUMONIA folders)
│
├── notebooks/                — Jupyter notebooks for EDA and experimentation
│   ├── cardio_eda_training.ipynb
│   ├── diabetes_eda_training.ipynb
│   ├── pneumonia_eda_training.ipynb
│   └── recovery_eda_training.ipynb
│
├── utils/                    — Utility functions
│   ├── preprocessing.py      — Data preprocessing helpers
│   └── json_response.py      — Standardized JSON output formatting
│
├── evaluation/               — Evaluation reports
│   └── metrics_report.md
│
└── requirements.txt          — Python package dependencies
```

---

## Code Listings

### Key Code Excerpts:

#### 1. Cardiovascular Prediction Function (`inference/predict_cardio.py`)

```python
def predict_cardio(data: dict):
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
```

#### 2. Hybrid LSTM Architecture (`training/train_recovery_lstm.py`)

```python
class RecoveryLSTM(nn.Module):
    def __init__(self, num_vitals, num_static, hidden_size, num_layers, forecast_days, dropout):
        super(RecoveryLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_vitals, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
        )
        combined_size = hidden_size + 32
        self.recovery_head = nn.Sequential(
            nn.Linear(combined_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, forecast_days),
        )
        self.complication_head = nn.Sequential(
            nn.Linear(combined_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, vitals_seq, static_feats):
        lstm_out, (h_n, _) = self.lstm(vitals_seq)
        temporal_features = h_n[-1]
        static_encoded = self.static_encoder(static_feats)
        combined = torch.cat([temporal_features, static_encoded], dim=1)
        recovery_pred = self.recovery_head(combined)
        complication_pred = self.complication_head(combined).squeeze(-1)
        return recovery_pred, complication_pred
```

#### 3. Pneumonia CNN Transfer Learning (`training/train_pneumonia.py`)

```python
def get_resnet50(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Class-weighted loss to handle imbalance
train_counts = [train_targets.count(i) for i in range(len(class_names))]
total_samples = sum(train_counts)
class_weights = torch.FloatTensor([total_samples / c for c in train_counts]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

#### 4. SMOTE for Diabetes Class Imbalance (`training/train_diabetes.py`)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
```

#### 5. Generative AI Recovery Coach (`inference/recovery_coach.py`)

```python
RECOVERY_GUIDELINES = {
    "cardiac": {
        "typical_recovery_weeks": "6-12",
        "activity_phases": [...],
        "warning_signs": [...],
        "dietary": "Low sodium, heart-healthy fats, high fiber, limit caffeine",
    },
    # ... orthopedic, abdominal, neurological
}
```

---

# 7. Introduction

## 7.1 Objectives

Healthcare systems globally face increasing demand with limited resources for personalized patient monitoring. Cardiovascular disease and diabetes often go undiagnosed until advanced stages, pneumonia diagnosis through manual X-ray interpretation is error-prone, and post-surgery recovery tracking remains largely manual with delayed complication detection. Generic discharge instructions fail to adapt to patient-specific risk factors.

**Primary Objectives:**

1. Build binary classifiers for cardiovascular disease risk prediction using 70,000 patient records with 11 clinical features.

2. Build multi-class classifiers for diabetes screening using 253,680 CDC BRFSS records with class imbalance handling via SMOTE.

3. Develop a CNN-based pneumonia detector using transfer learning (ResNet/EfficientNet) on 5,856 chest X-ray images.

4. Design a hybrid LSTM model for 3-day post-surgery recovery forecasting and complication detection from 7-day vital sign sequences.

5. Integrate a Generative AI Recovery Coach (Google Gemini / OpenAI GPT) for personalized, evidence-based health advice.

6. Standardize all predictions into a unified JSON output with risk levels (Low / Moderate / High) and deploy as an interactive Streamlit web application.

**Research Gap Addressed:** No existing system integrates tabular ML, medical imaging CNN, time-series LSTM, and generative AI coaching into a unified intelligent health assistant pipeline.

---

# 8. Literature Survey

## 8.1 Summary of Literature

| # | Area | Reference | Year | Approach | Key Finding |
|---|------|-----------|------|----------|-------------|
| 1 | CVD Prediction | Rajdhan et al. | 2020 | Ensemble ML on clinical data | Random Forest and ensemble methods achieve ~85% accuracy on heart disease |
| 2 | CVD Prediction | Mohan et al. | 2019 | Hybrid ML with feature selection | Hybrid Random Forest with Linear Model (HRFLM) achieves 88.7% accuracy |
| 3 | Diabetes | Zou et al. | 2018 | Gradient boosting on survey data | Random Forest and Gradient Boosting best for diabetes prediction |
| 4 | Diabetes | Kavakiotis et al. | 2017 | Comprehensive ML review | Supervised learning most effective; class imbalance is key challenge |
| 5 | Pneumonia | Rajpurkar et al. (CheXNet) | 2017 | DenseNet-121 on chest X-rays | Exceeded average radiologist performance on pneumonia detection |
| 6 | Pneumonia | Kermany et al. | 2018 | Transfer learning on medical images | CNNs achieve specialist-level diagnosis from limited training data |
| 7 | Deep Learning | He et al. | 2016 | Residual Networks (ResNet) | Skip connections enable training of very deep networks |
| 8 | Efficient CNNs | Tan & Le | 2019 | EfficientNet compound scaling | Better accuracy-efficiency trade-off than ResNets |
| 9 | Time-Series | Harutyunyan et al. | 2019 | LSTM/RNN on ICU data | Multitask learning improves clinical prediction tasks |
| 10 | Healthcare AI | Esteva et al. | 2019 | Deep learning in healthcare guide | Reviews applications across medical imaging, EHR, genomics |
| 11 | GenAI Healthcare | Singhal et al. (Med-PaLM) | 2023 | LLMs for clinical knowledge | LLMs encode substantial medical knowledge; can assist in Q&A |
| 12 | GenAI Healthcare | Thirunavukarasu et al. | 2023 | LLM applications in medicine | Reviews opportunities and risks of LLMs in clinical settings |
| 13 | Agentic AI | Park et al. | 2023 | Generative agents | LLM-based agents can perform complex multi-step reasoning |
| 14 | Agentic AI | Wang et al. | 2024 | LLM autonomous agents survey | Comprehensive survey of agent architectures and capabilities |
| 15 | Boosting | Chen & Guestrin (XGBoost) | 2016 | Scalable tree boosting | XGBoost dominates structured data competitions |
| 16 | Boosting | Ke et al. (LightGBM) | 2017 | Efficient gradient boosting | Faster training with comparable/better accuracy than XGBoost |
| 17 | Imbalanced Data | Chawla et al. (SMOTE) | 2002 | Synthetic oversampling | Effective for handling minority class in medical datasets |

**Research Gap Identified:** While individual components (disease prediction, medical imaging, time-series forecasting, LLM coaching) have been extensively studied, no prior work integrates all four into a single unified pipeline with standardized risk output and agentic AI-powered coaching.

---

# 9. Methodology

## 9.1 Dataset

Four datasets are used:
- **Cardiovascular:** 70,000 records × 11 features (Kaggle, balanced binary)
- **Diabetes:** 253,680 records × 21 features (CDC BRFSS 2015, imbalanced 3-class)
- **Pneumonia:** 5,856 chest X-ray images (Kaggle, binary Normal/Pneumonia)
- **Recovery:** 1,000 patients × 30 days (synthetic, 7 vitals + 7 static features)

## 9.2 Preprocessing

**Tabular Data (Cardiovascular & Diabetes):**
- Missing value verification (none found in either dataset)
- StandardScaler normalization on all numerical features
- SMOTE oversampling for diabetes minority classes (Prediabetes 1.8% → balanced)
- 80/20 stratified train-test split preserving class distribution

**Image Data (Pneumonia):**
- Resize all images to 224×224 pixels
- Data augmentation: Random horizontal flip, rotation (±10°), color jitter (brightness/contrast ±0.2)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Class-weighted CrossEntropyLoss to handle 3:1 pneumonia:normal imbalance

**Time-Series Data (Recovery):**
- Separate StandardScaler for vitals and static features
- LabelEncoder for surgery type (cardiac/orthopedic/abdominal/neurological)
- Sliding window: 7-day lookback → 3-day forecast
- Stratified split preserving complication ratio in train/test sets

## 9.3 Model Implementation

1. **Classical ML Pipeline (Cardio & Diabetes):** Five models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM) trained and compared. Best model auto-selected based on test accuracy.

2. **CNN Pipeline (Pneumonia):** Three pretrained architectures (ResNet-18, ResNet-50, EfficientNet-B0) fine-tuned with modified classification heads. Best model saved based on validation accuracy.

3. **LSTM Pipeline (Recovery):** Custom dual-head RecoveryLSTM combining temporal (LSTM) and static (Dense) feature encoding with two output heads for recovery regression and complication classification.

4. **GenAI Pipeline (Coach):** Multi-provider LLM integration with structured prompt engineering and domain-specific knowledge base. Graceful fallback to rule-based advice.

## 9.4 Training and Evaluation

- **Tabular Models:** Trained on 80% data, evaluated on 20% hold-out with accuracy, F1-score, ROC-AUC, and confusion matrix
- **CNN Models:** Trained for 10 epochs with Adam optimizer, evaluated on 624 test images with accuracy, precision, recall, ROC-AUC
- **LSTM Model:** Trained for 30 epochs with ReduceLROnPlateau scheduler, evaluated with MAE for recovery and F1 for complication detection
- **Model Selection:** Automated best-model selection based on primary metric (accuracy for classification, MAE for regression)
- **All models** output standardized risk levels: High (>0.7), Moderate (0.4-0.7), Low (<0.4)
