# Pneumonia detection model training script (Chest X-Ray Classification)
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "chest_xray")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# ── Hyperparameters ──
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ── Data Transforms ──
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Load Datasets ──
print("\nLoading chest X-ray datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=test_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA']
print(f"Classes: {class_names}")
print(f"Train: {len(train_dataset)} images")
print(f"Val:   {len(val_dataset)} images")
print(f"Test:  {len(test_dataset)} images")

# ── Class distribution ──
train_targets = [label for _, label in train_dataset.samples]
print(f"\nTrain class distribution:")
for i, name in enumerate(class_names):
    count = train_targets.count(i)
    print(f"  {name}: {count}")

# ── Define Models ──
def get_resnet18(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_resnet50(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_efficientnet_b0(num_classes=2):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ── Training Function ──
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # ── Train phase ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"  Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

    return best_val_acc, train_losses, val_losses, train_accs, val_accs

# ── Evaluation Function ──
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ── Compare Models ──
print("\n" + "="*60)
print("COMPARING MODELS ON CHEST X-RAY DATASET")
print("="*60)

model_configs = {
    "ResNet-18": get_resnet18,
    "ResNet-50": get_resnet50,
    "EfficientNet-B0": get_efficientnet_b0,
}

# Handle class imbalance with weighted loss
train_counts = [train_targets.count(i) for i in range(len(class_names))]
total_samples = sum(train_counts)
class_weights = torch.FloatTensor([total_samples / c for c in train_counts]).to(DEVICE)

results = {}
for name, model_fn in model_configs.items():
    print(f"\nTraining {name}...")
    model = model_fn().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    results[name] = {
        "best_val_acc": best_val_acc,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")

# ── Print comparison table ──
print("\n" + "="*60)
print(f"{'Model':<25} {'Best Val Acc':>12}")
print("-"*37)
for name, res in sorted(results.items(), key=lambda x: x[1]["best_val_acc"], reverse=True):
    print(f"{name:<25} {res['best_val_acc']:>12.4f}")

# ── Select best model ──
best_name = max(results, key=lambda k: results[k]["best_val_acc"])
best_model = results[best_name]["model"]
best_val_acc = results[best_name]["best_val_acc"]

print(f"\n>>> Best model: {best_name} with Val Accuracy = {best_val_acc:.4f}")

# ── Detailed evaluation on test set ──
print(f"\n{'='*60}")
print(f"DETAILED EVALUATION ON TEST SET — {best_name}")
print(f"{'='*60}")

y_true, y_pred, y_probs = evaluate_model(best_model, test_loader)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_probs)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=class_names)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

# ── Save best model ──
model_path = os.path.join(MODELS_DIR, "pneumonia_model.pth")
torch.save({
    "model_name": best_name,
    "model_state_dict": best_model.state_dict(),
    "class_names": class_names,
    "img_size": IMG_SIZE,
    "accuracy": accuracy,
    "roc_auc": roc_auc,
}, model_path)

print(f"\nBest model ({best_name}) saved to {model_path}")
print(f"Test Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}")
