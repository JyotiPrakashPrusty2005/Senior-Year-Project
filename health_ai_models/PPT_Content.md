# PPT SLIDE CONTENT
## Agentic AI Health Coach: Early Disease Risk Prediction & Post-Surgery Recovery Assistant using Hybrid ML, LSTM and Generative AI

---

## SLIDE 1 — Title Slide

**Title:** Agentic AI Health Coach: Early Disease Risk Prediction & Post-Surgery Recovery Assistant using Hybrid ML, LSTM and Generative AI

**Subtitle:** Senior Year Project

**By:** [Your Name(s)]

**Department / University:** [Your Department & University]

**Date:** April 2026

---

## SLIDE 2 — Introduction

- Healthcare systems face increasing demand with limited resources for personalized patient monitoring
- AI and ML can assist clinicians by automating risk screening and post-operative recovery tracking
- This project builds an **end-to-end intelligent health assistant** that:
  - Predicts cardiovascular disease, diabetes, and pneumonia from patient data
  - Monitors post-surgery recovery using time-series vital signs
  - Generates personalized coaching advice via Generative AI (LLM)
- Bridges the gap between **predictive analytics** and **actionable patient guidance**

---

## SLIDE 3 — Problem Statement

- **Early detection gaps:** Cardiovascular disease and diabetes often go undiagnosed until advanced stages
- **Pneumonia misdiagnosis:** Manual X-ray interpretation is error-prone and time-consuming
- **Post-surgery monitoring:** Recovery tracking is largely manual with delayed complication detection
- **Lack of personalization:** Generic discharge instructions don't adapt to patient-specific risk factors
- **Need:** An integrated AI system that combines disease prediction, real-time recovery monitoring, and personalized AI-generated health coaching

---

## SLIDE 4 — Project Objectives

1. Build **binary classifiers** for cardiovascular disease risk prediction (70K patient records)
2. Build **multi-class classifiers** for diabetes screening (253K CDC BRFSS records) with class imbalance handling
3. Develop a **CNN-based pneumonia detector** using transfer learning on chest X-ray images (5,856 images)
4. Design a **hybrid LSTM model** for 3-day post-surgery recovery forecasting and complication detection
5. Integrate a **Generative AI Recovery Coach** (Google Gemini / OpenAI GPT) for personalized health advice
6. Standardize all predictions into a unified JSON output with risk levels (Low / Moderate / High)

---

## SLIDE 5 — Literature Survey

| Area | Key References | Approach |
|------|---------------|----------|
| CVD Prediction | Rajdhan et al. (2020), Mohan et al. (2019) | Ensemble ML on clinical tabular data |
| Diabetes Classification | Zou et al. (2018), Kavakiotis et al. (2017) | Gradient boosting on BRFSS survey data; SMOTE for imbalance |
| Pneumonia Detection | Rajpurkar et al. (CheXNet, 2017), Kermany et al. (2018) | Transfer Learning with ResNet / DenseNet on chest X-rays |
| Post-Surgical Recovery | Esteva et al. (2019), Harutyunyan et al. (2019) | LSTM / RNN on ICU time-series vitals |
| Generative AI in Healthcare | Singhal et al. (Med-PaLM, 2023), Thirunavukarasu et al. (2023) | LLMs for clinical Q&A and patient-facing advice |
| Agentic AI Systems | Park et al. (2023), Wang et al. (2024) | LLM agents with domain-specific knowledge bases |

**Research Gap:** No existing system integrates all four — tabular ML, medical imaging CNN, time-series LSTM, and generative AI coaching — into a unified pipeline.

---

## SLIDE 6 — System Architecture

```
┌─────────────── PATIENT DATA INPUT ───────────────┐
│                                                    │
│  Tabular Data    Chest X-Ray    Recovery Vitals   │
│  (11-21 features)  (JPEG)       (7-day sequence)  │
└────┬──────────────┬──────────────┬────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐  ┌───────────┐  ┌──────────────────┐
│ Classical│  │ CNN       │  │ Hybrid LSTM      │
│ ML       │  │ Transfer  │  │ (Dual-Head)      │
│ (XGBoost │  │ Learning  │  │ Recovery + Risk  │
│ LightGBM)│  │ (ResNet/  │  │                  │
│          │  │ EfficientNet)│ │                │
└────┬─────┘  └─────┬─────┘  └────┬─────────────┘
     │              │              │
     ▼              ▼              ▼
┌──────────────────────────────────────────────────┐
│          Unified JSON Risk Output                 │
│     Probability + Risk Level (Low/Mod/High)       │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  Generative AI Coach    │
         │  (Gemini / GPT / Rules) │
         │  → Personalized Advice  │
         └─────────────────────────┘
```

---

## SLIDE 7 — Dataset Description

| Dataset | Source | Samples | Features | Target | Balance |
|---------|--------|---------|----------|--------|---------|
| **Cardiovascular** | Kaggle | 70,000 | 11 (age, BP, cholesterol, lifestyle) | Binary (CVD: Yes/No) | Balanced (50/50) |
| **Diabetes (BRFSS 2015)** | CDC | 253,680 | 21 (BMI, BP, lifestyle, demographics) | 3-class (No/Pre/Diabetes) | Imbalanced (84%/2%/14%) |
| **Chest X-Ray** | Kaggle | 5,856 | 224×224 RGB images | Binary (Normal/Pneumonia) | Imbalanced (3:1 Pneumonia) |
| **Recovery (Synthetic)** | Generated | 30,000 | 7 vitals + 7 static per day × 30 days | Recovery score + Complication flag | Normal 54% / Slow 31% / Complication 15% |

---

## SLIDE 8 — Data Preprocessing

**Tabular Data (Cardio & Diabetes):**
- Missing value check (none found)
- StandardScaler normalization on all features
- Label encoding for categorical features
- SMOTE oversampling for diabetes minority classes (Prediabetes 1.8% → balanced)
- 80/20 stratified train-test split

**Chest X-Ray Images:**
- Resize to 224×224 pixels
- Data augmentation: Random horizontal flip, rotation (±10°), color jitter
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Class-weighted loss function to handle 3:1 imbalance

**Recovery Time-Series:**
- Separate StandardScaler for vitals and static features
- LabelEncoder for surgery type
- Sliding window: 7-day lookback → 3-day forecast
- Stratified split preserving complication ratio

---

## SLIDE 9 — Exploratory Data Analysis

**Cardiovascular:**
- Correlation heatmap → ap_hi, ap_lo, cholesterol, age most correlated with CVD
- Blood pressure distribution: hypertensive patients have 2× CVD risk
- Age converted from days → years shows risk increases after 50

**Diabetes:**
- BMI boxplot: Diabetic patients median BMI ~32 vs ~27 for non-diabetic
- Age category analysis: prevalence spikes after category 8 (55-59 years)
- Class distribution pie chart: 84.4% No / 1.8% Pre / 13.9% Diabetes

**Pneumonia:**
- Sample X-ray grid: Normal lungs (clear) vs Pneumonia (opacities/consolidation)
- Dataset split visualization: Training 5,216 / Validation 16 / Test 624
- Augmentation before/after comparison

**Recovery:**
- 3 recovery trajectory plots (Normal / Slow / Complication)
- Complication patients show vital sign spikes around days 5-15
- Risk score distribution by surgery type

---

## SLIDE 10 — Model Selection

| Task | Models Compared | Selection Criteria |
|------|----------------|-------------------|
| **Cardiovascular** | Logistic Regression, Random Forest, Gradient Boosting, **XGBoost**, **LightGBM** | Accuracy, ROC-AUC, F1-Score |
| **Diabetes** | Logistic Regression, Random Forest, Gradient Boosting, **XGBoost**, **LightGBM** | Weighted F1 (handles imbalance), ROC-AUC (OvR) |
| **Pneumonia** | **ResNet-18**, **ResNet-50**, **EfficientNet-B0** | Test Accuracy, ROC-AUC, Precision/Recall |
| **Recovery** | **Hybrid LSTM** (custom dual-head) | MAE (recovery), F1 (complication detection) |
| **Coaching** | **Google Gemini**, OpenAI GPT-3.5, Rule-based fallback | Quality of personalized advice |

Best model selected automatically based on validation metrics.

---

## SLIDE 11 — Model Architecture (Tabular — Cardio & Diabetes)

**Pipeline:**
```
Raw Features → StandardScaler → [SMOTE if imbalanced] → Model → Probability → Risk Level
```

**Hyperparameters (Best Models):**

| Parameter | XGBoost | LightGBM |
|-----------|---------|----------|
| n_estimators | 300 | 300 |
| learning_rate | 0.05 | 0.05 |
| max_depth | 7 | 7 |
| Objective | binary:logistic / multi:softproba | binary / multiclass |

**Risk Classification:**
- **High Risk:** Probability > 0.7
- **Moderate Risk:** Probability 0.4 – 0.7
- **Low Risk:** Probability < 0.4

---

## SLIDE 12 — Model Architecture (Pneumonia CNN)

**Transfer Learning Pipeline:**
```
Chest X-Ray (224×224×3) → Pretrained CNN (ImageNet) → Modified FC Head → 2-Class Output
```

| Component | Detail |
|-----------|--------|
| Input | 224×224 RGB image |
| Backbone | ResNet-18 / ResNet-50 / EfficientNet-B0 (ImageNet pretrained) |
| Final Layer | Replaced with Linear(in_features, 2) |
| Loss | CrossEntropyLoss with class weights |
| Optimizer | Adam (lr=0.001) |
| Epochs | 10 |
| Augmentation | Flip, Rotate ±10°, Color Jitter |

---

## SLIDE 13 — Model Architecture (Recovery LSTM)

```
┌─────────────────────┐     ┌─────────────────┐
│ 7-Day Vitals (7×7)  │     │ Static (7 feats) │
│ pain, temp, HR,     │     │ age, gender, bmi,│
│ BP, mobility,       │     │ diabetes, HTN,   │
│ wound, sleep        │     │ smoking, surgery │
└────────┬────────────┘     └────────┬─────────┘
         │                           │
    LSTM (128 units, 2 layers)  Dense (7→32)
    Dropout = 0.3
         │                           │
         └─────────┬─────────────────┘
              Concatenate (160)
              ┌─────┴──────┐
              ↓            ↓
       Recovery Head   Complication Head
       Linear→Dense    Linear→Dense→Sigmoid
              ↓            ↓
       3-Day Forecast  Binary Risk (0-1)
       (MSE Loss)      (BCE Loss × 2.0)
```

**Config:** Seq Length=7, Forecast=3, Batch=64, Epochs=30, LR=0.001 + ReduceLROnPlateau

---

## SLIDE 14 — Software Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.x |
| **Classical ML** | Scikit-learn, XGBoost, LightGBM |
| **Deep Learning** | PyTorch, TorchVision |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Image Processing** | Pillow (PIL) |
| **API Framework** | FastAPI + Uvicorn |
| **Generative AI** | Google Generative AI (Gemini), OpenAI API |
| **Serialization** | Joblib (.pkl), PyTorch Checkpoints (.pth) |
| **Notebooks** | Jupyter Notebook |

---

## SLIDE 15 — Platform Information

| Component | Specification |
|-----------|--------------|
| **OS** | Windows 10/11 |
| **IDE** | VS Code + Jupyter Notebooks |
| **Python** | 3.10+ with virtual environment |
| **GPU (Training)** | CUDA-compatible GPU (PyTorch with CUDA support) |
| **API Deployment** | FastAPI + Uvicorn (localhost / cloud) |
| **Gen AI APIs** | Google Gemini (free tier), OpenAI GPT-3.5 |
| **Version Control** | Git |
| **Model Storage** | Local filesystem (.pkl, .pth files) |

---

## SLIDE 16 — Preliminary Results

### Cardiovascular Disease Prediction
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~72% | ~0.78 |
| Random Forest | ~73% | ~0.80 |
| Gradient Boosting | ~73% | ~0.80 |
| **XGBoost** | **~74%** | **~0.81** |
| LightGBM | ~74% | ~0.81 |

### Diabetes Prediction (with SMOTE)
| Model | Weighted F1 | ROC-AUC (OvR) |
|-------|-------------|---------------|
| Logistic Regression | ~0.70 | ~0.82 |
| **LightGBM / XGBoost** | **~0.75** | **~0.85** |

### Pneumonia Detection
| Model | Test Accuracy | ROC-AUC |
|-------|---------------|---------|
| ResNet-18 | ~90% | ~0.95 |
| **ResNet-50** | **~92%** | **~0.96** |
| EfficientNet-B0 | ~91% | ~0.95 |

### Recovery LSTM
| Metric | Value |
|--------|-------|
| Recovery MAE | ~5-8 points (out of 100) |
| Complication Detection F1 | ~0.80+ |

*(Replace these with your actual numbers from training runs)*

---

## SLIDE 17 — Project Roadmap

| Phase | Timeline | Tasks | Status |
|-------|----------|-------|--------|
| **Phase 1: Data Collection & EDA** | Month 1-2 | Dataset acquisition, exploration, visualization | ✅ Completed |
| **Phase 2: Preprocessing & Feature Engineering** | Month 2-3 | Scaling, SMOTE, augmentation, sliding windows | ✅ Completed |
| **Phase 3: Model Training & Selection** | Month 3-5 | Train 5 classifiers + CNNs + LSTM, hyperparameter tuning | ✅ Completed |
| **Phase 4: Inference Pipeline** | Month 5-6 | Unified prediction scripts, JSON output, risk levels | ✅ Completed |
| **Phase 5: Generative AI Coach** | Month 6-7 | Recovery coach with Gemini/GPT integration, knowledge base | ✅ Completed |
| **Phase 6: API & Integration** | Month 7-8 | FastAPI endpoints, end-to-end testing | 🔄 In Progress |
| **Phase 7: Evaluation & Documentation** | Month 8-9 | Metrics report, Word doc generation, final evaluation | 🔄 In Progress |
| **Phase 8: Presentation & Defense** | Month 9 | PPT, demo, research article | 📋 Upcoming |

---

## SLIDE 18 — Research Article Progress Status

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | 📋 Draft | Summarizes multi-model health AI system |
| Introduction & Problem Statement | ✅ Complete | Healthcare AI gap + objectives |
| Literature Review | ✅ Complete | CVD, Diabetes, Pneumonia, LSTM, GenAI references |
| Methodology | ✅ Complete | All 4 model architectures + coach design |
| Dataset Description | ✅ Complete | 4 datasets documented |
| Experimental Setup | ✅ Complete | Hyperparameters, splits, metrics |
| Results & Discussion | 🔄 In Progress | Awaiting final metric consolidation |
| Conclusion & Future Work | 📋 Pending | To be written after final results |
| References | 🔄 In Progress | 20+ references collected |

**Target Journal/Conference:** [Specify your target venue]

---

## SLIDE 19 — References

1. Rajpurkar, P. et al., "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," *arXiv:1711.05225*, 2017.
2. Kermany, D. et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," *Cell*, 2018.
3. He, K. et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
4. Tan, M. & Le, Q., "EfficientNet: Rethinking Model Scaling for CNNs," *ICML*, 2019.
5. Zou, Q. et al., "Predicting Diabetes Mellitus with Machine Learning Techniques," *Frontiers in Genetics*, 2018.
6. Kavakiotis, I. et al., "Machine Learning and Data Mining Methods in Diabetes Research," *Computational and Structural Biotechnology Journal*, 2017.
7. Mohan, S. et al., "Effective Heart Disease Prediction Using Hybrid ML Techniques," *IEEE Access*, 2019.
8. Chen, T. & Guestrin, C., "XGBoost: A Scalable Tree Boosting System," *KDD*, 2016.
9. Ke, G. et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *NeurIPS*, 2017.
10. Chawla, N. et al., "SMOTE: Synthetic Minority Over-sampling Technique," *JAIR*, 2002.
11. Hochreiter, S. & Schmidhuber, J., "Long Short-Term Memory," *Neural Computation*, 1997.
12. Harutyunyan, H. et al., "Multitask Learning and Benchmarking with Clinical Time Series Data," *Scientific Data*, 2019.
13. Esteva, A. et al., "A Guide to Deep Learning in Healthcare," *Nature Medicine*, 2019.
14. Singhal, K. et al., "Large Language Models Encode Clinical Knowledge," *Nature*, 2023 (Med-PaLM).
15. Thirunavukarasu, A. et al., "Large Language Models in Medicine," *Nature Medicine*, 2023.
16. Paszke, A. et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," *NeurIPS*, 2019.
17. Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 2011.

---

## SLIDE 20 — Thank You / Q&A

**Thank You!**

**Questions?**

**Project Repository:** [GitHub Link]

**Contact:** [Email]
