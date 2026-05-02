# 8-Week Progress Report
## Agentic AI Health Coach: Early Disease Risk Prediction & Post-Surgery Recovery Assistant using Hybrid ML, LSTM and Generative AI

---

## WEEK 1 — Project Setup, Literature Survey & Data Acquisition

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Finalized project scope and objectives. Conducted literature survey on ML for healthcare (CVD prediction, diabetes classification, pneumonia detection via CNNs, LSTM for ICU/recovery monitoring, GenAI in medicine). Set up Python virtual environment, installed core libraries (scikit-learn, PyTorch, XGBoost, LightGBM, pandas, numpy). Acquired datasets — Kaggle Cardiovascular (70K records), CDC BRFSS 2015 Diabetes (253,680 records), Kaggle Chest X-Ray Pneumonia (5,856 images). |
| **Outcome/Deliverable** | - Project proposal document with 6 objectives defined. <br> - Literature review covering CheXNet (Rajpurkar 2017), SMOTE (Chawla 2002), XGBoost (Chen 2016), Med-PaLM (Singhal 2023). <br> - `requirements.txt` finalized with all dependencies. <br> - 3 datasets downloaded and organized under `data/` directory. |
| **Issues & Challenges Faced** | - Chest X-ray dataset was large (~1.2 GB), slow to download and organize. <br> - Validation split in chest X-ray dataset was very small (only 16 images), which required rethinking the validation strategy. <br> - CDC BRFSS dataset had 253K+ rows — initial loading was slow without optimized dtypes. |
| **Corrective Action & Learning** | - Used structured folder hierarchy (`data/chest_xray/train/NORMAL`, `PNEUMONIA`, etc.) for PyTorch ImageFolder compatibility. <br> - Learned that small validation sets need to be merged with training data and re-split manually. <br> - Applied dtype optimization when loading large CSVs with pandas. |
| **Plans for Next Week** | Begin Exploratory Data Analysis (EDA) for Cardiovascular and Diabetes datasets. Inspect feature distributions, class balance, correlations, and missing values. |

---

## WEEK 2 — EDA for Cardiovascular & Diabetes Datasets

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Performed comprehensive EDA on the Cardiovascular dataset (70K records, 11 features) and Diabetes dataset (253,680 records, 21 features). Created Jupyter notebooks with visualizations — distribution plots, correlation heatmaps, box plots, class balance analysis. Identified key patterns and data quality issues. |
| **Outcome/Deliverable** | - `notebooks/cardio_eda_training.ipynb` — EDA section completed with: target distribution (50/50 balanced), BP analysis (ap_hi/ap_lo correlation with CVD), cholesterol & glucose impact, age vs risk trends, correlation heatmap. <br> - `notebooks/diabetes_eda_training.ipynb` — EDA section completed with: 3-class distribution (84.4% / 1.8% / 13.9%), BMI boxplots by class (diabetic median BMI ~32 vs ~27), age category analysis (risk spikes after 55-59). <br> - Identified severe class imbalance in diabetes (Prediabetes = only 1.8%). |
| **Issues & Challenges Faced** | - Cardiovascular dataset had age in days (not years), causing confusion in initial analysis. <br> - Blood pressure features (ap_hi, ap_lo) contained physiologically impossible outlier values (e.g., negative or >300 mmHg). <br> - Diabetes dataset has extreme class imbalance — Prediabetes class at 1.8% would be nearly impossible for models to learn. |
| **Corrective Action & Learning** | - Converted age from days to years for meaningful analysis. <br> - Applied physiological range filtering for blood pressure values. <br> - Decided to apply SMOTE (Synthetic Minority Over-sampling Technique) during training to handle diabetes class imbalance. <br> - Learned that macro F1 is more appropriate than accuracy for imbalanced multi-class problems. |
| **Plans for Next Week** | Perform EDA on Pneumonia X-ray dataset. Design and generate synthetic recovery data for the LSTM model. |

---

## WEEK 3 — Pneumonia EDA & Synthetic Recovery Data Generation

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Completed EDA on Chest X-ray pneumonia dataset — visualized sample images, analyzed class distribution, studied image resolution variations. Designed and implemented synthetic recovery data generation script simulating 1,000 patients over 30 post-surgery days with 3 distinct recovery trajectories. |
| **Outcome/Deliverable** | - `notebooks/pneumonia_eda_training.ipynb` — EDA section: sample X-ray grids (Normal vs Pneumonia), dataset split analysis (Train 5,216 / Val 16 / Test 624), class imbalance ratio 3:1 (Pneumonia:Normal). <br> - `data/generate_synthetic_recovery.py` — Script generating 30,000 records (1,000 patients × 30 days) with realistic vital sign trajectories. <br> - `data/synthetic_recovery_data.csv` — Generated dataset with 7 daily vitals, 7 static features, recovery_score, complication_flag, and recovery_outcome (Normal 54% / Slow 31% / Complication 15%). |
| **Issues & Challenges Faced** | - X-ray image resolutions varied widely (300px to 2000px+), requiring standardized resizing. <br> - Designing realistic synthetic recovery trajectories was challenging — had to model different complication onset patterns (days 5-15), fever spikes, pain trajectories per surgery type. <br> - Ensuring complication prevalence (~15%) was clinically realistic required research into post-surgical complication rates. |
| **Corrective Action & Learning** | - Standardized all images to 224×224 (compatible with ResNet/EfficientNet input). <br> - Modeled complication risk as a function of age, BMI, comorbidities, and smoking — weighted contributions: age(0.3), BMI(0.15), diabetes(0.2), hypertension(0.15), smoking(0.2). <br> - Learned that synthetic data generation requires careful domain knowledge to produce clinically plausible trajectories. |
| **Plans for Next Week** | Implement preprocessing pipelines (StandardScaler, SMOTE, image augmentation). Begin training tabular models for Cardiovascular disease prediction. |

---

## WEEK 4 — Preprocessing Pipelines & Cardiovascular Model Training

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Built reusable preprocessing utilities (`utils/preprocessing.py`, `utils/json_response.py`). Implemented StandardScaler normalization, label encoding, and stratified train-test splits. Trained 5 ML models for cardiovascular disease prediction — Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM. Evaluated and selected best model. |
| **Outcome/Deliverable** | - `utils/preprocessing.py` — Label encoding utility for categorical features. <br> - `utils/json_response.py` — Standardized JSON output with risk levels (Low/Moderate/High based on probability thresholds 0.4/0.7). <br> - `training/train_cardio.py` — 5-model comparison pipeline. <br> - **Best Cardio Model Results:** Accuracy = **0.7353**, Precision = **0.7529**, Recall = **0.7000**, F1 = **0.7255**, ROC-AUC = **0.7998**. <br> - Saved: `models/cardio_model.pkl`, `models/cardio_scaler.pkl`. |
| **Issues & Challenges Faced** | - Logistic Regression baseline was competitive (~72%) — ensemble models only marginally better. <br> - Hyperparameter tuning across 5 models was time-consuming. <br> - ROC-AUC of ~0.80 suggests some features have limited discriminative power for CVD. |
| **Corrective Action & Learning** | - Used consistent hyperparameters across boosting models (n_estimators=200-300, learning_rate=0.05, max_depth=7) for fair comparison. <br> - Automated best-model selection based on highest ROC-AUC. <br> - Learned that cardiovascular prediction from basic clinical features has a natural ceiling (~74-76% accuracy) — consistent with published literature. |
| **Plans for Next Week** | Train diabetes multi-class models with SMOTE handling. Build inference pipeline for cardiovascular predictions. |

---

## WEEK 5 — Diabetes Model Training & Cardio/Diabetes Inference Pipelines

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Applied SMOTE to balance the 3-class diabetes dataset. Trained 5 ML models (same set as cardio) on the balanced training set. Built inference scripts for both cardiovascular and diabetes predictions with standardized JSON output. |
| **Outcome/Deliverable** | - `training/train_diabetes.py` — SMOTE + 5-model comparison for multi-class classification. <br> - **Best Diabetes Model Results:** Accuracy = **0.8462**, Weighted F1 = **0.8188**, Macro F1 = **0.4202**, ROC-AUC (weighted OvR) = **0.8177**. Per-class: Diabetes recall ~0.99 (excellent), Prediabetes F1 ~0.32 (challenging). <br> - Saved: `models/diabetes_model.pkl`, `models/diabetes_scaler.pkl`. <br> - `inference/predict_cardio.py` — Takes 11 patient features → probability + risk level. <br> - `inference/predict_diabetes.py` — Takes 21 features → 3-class probabilities + risk level. |
| **Issues & Challenges Faced** | - Even after SMOTE, Prediabetes class (Class 1) prediction remains poor (F1 ~0.32) — this class is inherently ambiguous between No Diabetes and Diabetes. <br> - Macro F1 (0.42) is much lower than Weighted F1 (0.82) — indicating poor minority class performance. <br> - SMOTE on 253K samples with 3 classes was computationally expensive. |
| **Corrective Action & Learning** | - Accepted that Prediabetes is clinically a transitional state and inherently hard to classify — consistent with medical literature. <br> - Used weighted F1 and ROC-AUC (OvR) as primary selection metrics instead of accuracy. <br> - Learned that SMOTE helps recall for minority classes but can reduce precision — trade-off must be evaluated per use case. |
| **Plans for Next Week** | Train CNN models for pneumonia detection using transfer learning (ResNet-18, ResNet-50, EfficientNet-B0). |

---

## WEEK 6 — Pneumonia CNN Training & Inference

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Implemented transfer learning pipeline for pneumonia detection on chest X-rays. Trained 3 CNN architectures — ResNet-18, ResNet-50, EfficientNet-B0 — all pretrained on ImageNet with modified final classification layers. Applied data augmentation (horizontal flip, rotation ±10°, color jitter) and class-weighted cross-entropy loss to handle 3:1 imbalance. |
| **Outcome/Deliverable** | - `training/train_pneumonia.py` — Multi-architecture CNN training with class weights. <br> - **Best Pneumonia Model (ResNet-50):** Test Accuracy = **0.9022**, Best Val Accuracy = **0.9668**, Precision = **0.8730**, Recall = **0.9872**, F1 = **0.9266**, ROC-AUC = **0.9731**. <br> - Saved: `models/pneumonia_model.pth` (checkpoint with model_state_dict, model_name, class_names, accuracy, ROC-AUC). <br> - `inference/predict_pneumonia.py` — Takes chest X-ray image path → NORMAL/PNEUMONIA prediction + probability + risk level. Auto-detects architecture from saved checkpoint. |
| **Issues & Challenges Faced** | - Original validation split had only 16 images (8 per class) — unusable for reliable validation. <br> - Training was GPU-intensive; each architecture took significant time per epoch. <br> - High recall (0.9872) but lower precision (0.8730) — model tends to over-predict pneumonia (fewer false negatives but more false positives). |
| **Corrective Action & Learning** | - Merged original train and validation sets, then performed a proper 85/15 split for training. <br> - Used CUDA GPU acceleration where available; implemented batch size of 32 to balance memory and speed. <br> - In medical imaging, high recall (sensitivity) is preferred over high precision — missing pneumonia is more dangerous than a false alarm. <br> - Learned that transfer learning from ImageNet achieves >90% accuracy even with ~5K medical images. |
| **Plans for Next Week** | Design and train the hybrid LSTM model for post-surgery recovery prediction. |

---

## WEEK 7 — Recovery LSTM Training & Inference Pipeline

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Designed and trained a hybrid dual-head LSTM model for post-surgery recovery. The model takes 7-day vital sign sequences + static patient demographics and outputs: (1) 3-day recovery score forecast (regression), and (2) complication risk probability (binary classification). Built sliding window data pipeline, separate scalers for vitals and static features, and surgery type encoder. |
| **Outcome/Deliverable** | - `training/train_recovery_lstm.py` — Hybrid LSTM: 2 LSTM layers (128 units), dropout 0.3, static encoder (7→64→32), concatenation layer (160 features), dual output heads. Combined loss = MSE + 2.0×BCE. <br> - Saved: `models/recovery_lstm_model.pth`, `models/recovery_vitals_scaler.pkl`, `models/recovery_static_scaler.pkl`, `models/recovery_surgery_encoder.pkl`. <br> - `inference/predict_recovery.py` — Takes 7-day vitals history + patient info → 3-day recovery forecast, complication risk, trend (improving/declining/stable), risk level. <br> - Complication detection accuracy: **>80%**, recovery score MAE in clinically acceptable range. |
| **Issues & Challenges Faced** | - Sliding window creation (7-day lookback → 3-day forecast) required careful handling of patient boundaries to avoid data leakage across patients. <br> - Complication class is only ~15% of data — model initially biased toward predicting "no complication." <br> - Dual-head loss balancing — recovery MSE loss dominated, making complication head undertrained. |
| **Corrective Action & Learning** | - Implemented per-patient sliding window generation ensuring no cross-patient sequences. <br> - Applied 2.0× weight to BCE complication loss to emphasize risk detection. <br> - Used ReduceLROnPlateau scheduler and gradient clipping (max_norm=1.0) for stable training. <br> - Learned that multi-task learning requires careful loss weighting to prevent one head from dominating. |
| **Plans for Next Week** | Build the Generative AI Recovery Coach. Integrate LSTM predictions with LLM-based personalized advice generation. Create system architecture documentation. |

---

## WEEK 8 — Generative AI Recovery Coach, System Architecture & Documentation

| Header | Details |
|--------|---------|
| **Activity/Milestone Description** | Built the Agentic AI Recovery Coach integrating LSTM predictions with Generative AI (Google Gemini / OpenAI GPT) for personalized post-surgery advice. Created surgery-type-specific knowledge bases (Cardiac, Orthopedic, Abdominal, Neurological) with recovery phases, warning signs, dietary guidelines, and exercise recommendations. Designed system architecture (FastAPI backend layers). Prepared project documentation (PPT content, report generator, metrics report). |
| **Outcome/Deliverable** | - `inference/recovery_coach.py` — Multi-provider AI coach (Gemini 2.0 Flash primary, GPT-3.5 fallback, rule-based offline fallback). Dynamically builds context-specific prompts including patient profile, vitals, LSTM predictions, and surgery-type guidelines. Outputs: Recovery Assessment, Recommendations, Exercise Plan, Dietary Advice, Warning Signs, Motivation. <br> - `system_architecture.html` — System architecture diagrams (data flow, request/response sequences). <br> - `generate_report.py` — Automated Word document generation with all project sections. <br> - `PPT_Content.md` — 20-slide presentation content. <br> - `evaluation/metrics_report.md` — Evaluation metrics documentation. |
| **Issues & Challenges Faced** | - Google Gemini API rate limits on free tier caused intermittent failures during testing. <br> - Prompt engineering required iteration — initial prompts produced generic advice that wasn't surgery-type specific enough. <br> - Ensuring the offline rule-based fallback provided comparable quality to LLM outputs was challenging. <br> - System architecture integration required designing how LSTM predictions flow into the GenAI prompt seamlessly. |
| **Corrective Action & Learning** | - Implemented 3-tier fallback: Gemini → OpenAI → Rule-based, with automatic provider switching on failure. <br> - Added surgery-type-specific knowledge bases directly into the system prompt so the LLM has domain context for every response. <br> - Risk-based alerts built into the rule-based coach (fever >38.3°C → contact doctor, pain >7/10 → medication review, poor sleep <5h → relaxation techniques). <br> - Learned that agentic AI systems benefit greatly from structured domain knowledge injection rather than relying solely on LLM general knowledge. |
| **Plans for Next Week** | Build Streamlit frontend UI. Deploy FastAPI backend with all model endpoints. Conduct end-to-end integration testing. Finalize evaluation report with consolidated metrics. |

---

## CUMULATIVE PROGRESS SUMMARY (After 8 Weeks)

| Component | Status |
|-----------|--------|
| Data Collection & Acquisition (4 datasets) | ✅ Complete |
| Exploratory Data Analysis (3 notebooks) | ✅ Complete |
| Preprocessing Pipelines (Scaler, SMOTE, Augmentation, Sliding Window) | ✅ Complete |
| Cardiovascular Model (73.5% acc, 0.80 AUC) | ✅ Complete |
| Diabetes Model (84.6% acc, 0.82 AUC) | ✅ Complete |
| Pneumonia CNN (90.2% acc, 0.97 AUC) | ✅ Complete |
| Recovery LSTM (Dual-Head Hybrid) | ✅ Complete |
| Inference Pipelines (5 scripts) | ✅ Complete |
| Generative AI Recovery Coach | ✅ Complete |
| System Architecture Design | ✅ Complete |
| Documentation (PPT, Report, Metrics) | ✅ Complete |
| Frontend (Streamlit) | ❌ Not Started |
| Backend Deployment (FastAPI) | ❌ Not Started |
| End-to-End Integration Testing | ❌ Not Started |
