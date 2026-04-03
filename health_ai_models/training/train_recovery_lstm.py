"""
Train an LSTM model for post-surgery recovery prediction.

The model takes a sequence of daily vitals (7-day window) and predicts:
  1. Recovery score for the next 3 days
  2. Complication risk (binary classification)

This gives us both the LSTM component and hybrid ML approach for the project.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

# ─── Config ───────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 7       # Look back 7 days
FORECAST_DAYS = 3         # Predict 3 days ahead
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_recovery_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Features used as time-series input
VITALS_FEATURES = [
    "pain_level", "temperature", "heart_rate", "bp_systolic",
    "mobility_score", "wound_status", "sleep_hours"
]

# Static features (patient demographics)
STATIC_FEATURES = [
    "age", "gender", "bmi", "diabetes", "hypertension", "smoking", "surgery_type_encoded"
]


# ─── Dataset ──────────────────────────────────────────────────────────────
class RecoveryDataset(Dataset):
    """Dataset that creates sliding window sequences from patient time-series."""

    def __init__(self, sequences, static_feats, targets_recovery, targets_complication):
        self.sequences = torch.FloatTensor(sequences)
        self.static_feats = torch.FloatTensor(static_feats)
        self.targets_recovery = torch.FloatTensor(targets_recovery)
        self.targets_complication = torch.FloatTensor(targets_complication)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.static_feats[idx],
            self.targets_recovery[idx],
            self.targets_complication[idx],
        )


# ─── Model ────────────────────────────────────────────────────────────────
class RecoveryLSTM(nn.Module):
    """
    Hybrid LSTM model that combines:
    - LSTM for temporal vitals patterns
    - Dense layers for static patient features
    - Two output heads: recovery score regression + complication classification
    """

    def __init__(self, num_vitals, num_static, hidden_size, num_layers, forecast_days, dropout):
        super(RecoveryLSTM, self).__init__()

        # LSTM for time-series vital signs
        self.lstm = nn.LSTM(
            input_size=num_vitals,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combined features -> output heads
        combined_size = hidden_size + 32

        # Head 1: Recovery score prediction (regression, next N days)
        self.recovery_head = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_days),  # predict recovery score for each future day
        )

        # Head 2: Complication risk (binary classification)
        self.complication_head = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, vitals_seq, static_feats):
        # LSTM processes the vitals sequence
        lstm_out, (h_n, _) = self.lstm(vitals_seq)
        # Use the last hidden state
        temporal_features = h_n[-1]  # shape: (batch, hidden_size)

        # Encode static features
        static_encoded = self.static_encoder(static_feats)

        # Combine temporal + static
        combined = torch.cat([temporal_features, static_encoded], dim=1)

        # Two output heads
        recovery_pred = self.recovery_head(combined)
        complication_pred = self.complication_head(combined).squeeze(-1)

        return recovery_pred, complication_pred


# ─── Data Preparation ─────────────────────────────────────────────────────
def prepare_data():
    """Load data, create sequences, and prepare train/test splits."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Encode surgery type
    le = LabelEncoder()
    df["surgery_type_encoded"] = le.fit_transform(df["surgery_type"])

    # Scale vitals features
    vitals_scaler = StandardScaler()
    df[VITALS_FEATURES] = vitals_scaler.fit_transform(df[VITALS_FEATURES])

    # Scale static features
    static_scaler = StandardScaler()
    # Get one row per patient for fitting
    static_df = df.groupby("patient_id")[STATIC_FEATURES].first()
    static_scaler.fit(static_df)

    # Create sequences
    sequences = []
    static_feats_list = []
    targets_recovery = []
    targets_complication = []

    patient_ids = df["patient_id"].unique()
    print(f"Processing {len(patient_ids)} patients...")

    for pid in patient_ids:
        patient_data = df[df["patient_id"] == pid].sort_values("day")
        vitals = patient_data[VITALS_FEATURES].values
        recovery_scores = patient_data["recovery_score"].values
        complication_flags = patient_data["complication_flag"].values
        static = patient_data[STATIC_FEATURES].iloc[0].values

        # Scale static features
        static_scaled = static_scaler.transform(static.reshape(1, -1))[0]

        # Create sliding windows
        for i in range(len(vitals) - SEQUENCE_LENGTH - FORECAST_DAYS + 1):
            seq = vitals[i: i + SEQUENCE_LENGTH]
            future_recovery = recovery_scores[i + SEQUENCE_LENGTH: i + SEQUENCE_LENGTH + FORECAST_DAYS]
            # Complication in the forecast window
            future_complication = max(complication_flags[i + SEQUENCE_LENGTH: i + SEQUENCE_LENGTH + FORECAST_DAYS])

            sequences.append(seq)
            static_feats_list.append(static_scaled)
            targets_recovery.append(future_recovery)
            targets_complication.append(future_complication)

    sequences = np.array(sequences)
    static_feats_arr = np.array(static_feats_list)
    targets_recovery = np.array(targets_recovery)
    targets_complication = np.array(targets_complication)

    print(f"Total samples: {len(sequences)}")
    print(f"Complication rate: {targets_complication.mean():.2%}")

    # Train/test split (by index, preserving patient grouping is ideal but we simplify)
    indices = np.arange(len(sequences))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=targets_complication)

    train_dataset = RecoveryDataset(
        sequences[train_idx], static_feats_arr[train_idx],
        targets_recovery[train_idx], targets_complication[train_idx]
    )
    test_dataset = RecoveryDataset(
        sequences[test_idx], static_feats_arr[test_idx],
        targets_recovery[test_idx], targets_complication[test_idx]
    )

    # Save scalers and encoder
    joblib.dump(vitals_scaler, os.path.join(MODEL_DIR, "recovery_vitals_scaler.pkl"))
    joblib.dump(static_scaler, os.path.join(MODEL_DIR, "recovery_static_scaler.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "recovery_surgery_encoder.pkl"))

    return train_dataset, test_dataset


# ─── Training ─────────────────────────────────────────────────────────────
def train_model(train_dataset, test_dataset):
    """Train the hybrid LSTM model."""
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RecoveryLSTM(
        num_vitals=len(VITALS_FEATURES),
        num_static=len(STATIC_FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        forecast_days=FORECAST_DAYS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # Loss functions
    recovery_loss_fn = nn.MSELoss()
    complication_loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\nTraining on {DEVICE}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        train_loss_total = 0
        for vitals_seq, static, target_recovery, target_complication in train_loader:
            vitals_seq = vitals_seq.to(DEVICE)
            static = static.to(DEVICE)
            target_recovery = target_recovery.to(DEVICE)
            target_complication = target_complication.to(DEVICE)

            optimizer.zero_grad()
            recovery_pred, complication_pred = model(vitals_seq, static)

            loss_recovery = recovery_loss_fn(recovery_pred, target_recovery)
            loss_complication = complication_loss_fn(complication_pred, target_complication)
            loss = loss_recovery + 2.0 * loss_complication  # weight complication loss higher

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss_total = 0
        correct_complications = 0
        total_complications = 0

        with torch.no_grad():
            for vitals_seq, static, target_recovery, target_complication in test_loader:
                vitals_seq = vitals_seq.to(DEVICE)
                static = static.to(DEVICE)
                target_recovery = target_recovery.to(DEVICE)
                target_complication = target_complication.to(DEVICE)

                recovery_pred, complication_pred = model(vitals_seq, static)

                loss_recovery = recovery_loss_fn(recovery_pred, target_recovery)
                loss_complication = complication_loss_fn(complication_pred, target_complication)
                val_loss_total += (loss_recovery + 2.0 * loss_complication).item()

                # Complication accuracy
                predicted = (complication_pred > 0.5).float()
                correct_complications += (predicted == target_complication).sum().item()
                total_complications += target_complication.size(0)

        avg_val_loss = val_loss_total / len(test_loader)
        complication_acc = correct_complications / total_complications
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Complication Acc: {complication_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save best model
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_vitals": len(VITALS_FEATURES),
                    "num_static": len(STATIC_FEATURES),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "forecast_days": FORECAST_DAYS,
                    "dropout": DROPOUT,
                    "sequence_length": SEQUENCE_LENGTH,
                    "vitals_features": VITALS_FEATURES,
                    "static_features": STATIC_FEATURES,
                },
                "epoch": best_epoch,
                "val_loss": best_val_loss,
                "complication_acc": complication_acc,
            }
            torch.save(checkpoint, os.path.join(MODEL_DIR, "recovery_lstm_model.pth"))

    print(f"\nBest model saved at epoch {best_epoch} with val loss: {best_val_loss:.4f}")

    # ── Final evaluation ──
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    evaluate(model, test_loader)

    return model


def evaluate(model, test_loader):
    """Evaluate the model on the test set."""
    from sklearn.metrics import classification_report, mean_absolute_error

    model.eval()
    all_recovery_true = []
    all_recovery_pred = []
    all_complication_true = []
    all_complication_pred = []

    with torch.no_grad():
        for vitals_seq, static, target_recovery, target_complication in test_loader:
            vitals_seq = vitals_seq.to(DEVICE)
            static = static.to(DEVICE)

            recovery_pred, complication_pred = model(vitals_seq, static)

            all_recovery_true.append(target_recovery.numpy())
            all_recovery_pred.append(recovery_pred.cpu().numpy())
            all_complication_true.extend(target_complication.numpy().tolist())
            all_complication_pred.extend((complication_pred > 0.5).cpu().numpy().tolist())

    all_recovery_true = np.concatenate(all_recovery_true)
    all_recovery_pred = np.concatenate(all_recovery_pred)

    # Recovery score MAE
    mae = mean_absolute_error(all_recovery_true.flatten(), all_recovery_pred.flatten())
    print(f"Recovery Score MAE: {mae:.2f}")

    # Complication classification report
    print("\nComplication Detection Report:")
    print(classification_report(
        all_complication_true, all_complication_pred,
        target_names=["No Complication", "Complication"]
    ))


# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_dataset, test_dataset = prepare_data()
    model = train_model(train_dataset, test_dataset)
    print("\nTraining complete! Files saved in models/ directory.")
