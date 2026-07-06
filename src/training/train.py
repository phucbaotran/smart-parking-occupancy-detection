# =========================================================
# File name: train.py
# Project: Smart Parking Occupancy Detection
# Description: Train Simple CNN model using CNRPark+EXT dataset
# =========================================================


# *********************** Supporting libraries
from pathlib import Path
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# *********************** Global configuration
PROJECT_ROOT = Path(r"D:\smart-parking-occupancy-detection")
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from data.parking_dataset import ParkingDataset
from models.simple_cnn import SimpleCNN


TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"

MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

MODEL_PATH = MODEL_DIR / "best_simple_cnn.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0

# Use smaller number first for CPU testing.
# After the code runs well, change these to None for full training.
MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# *********************** Supporting functions
def createOutputFolders():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def checkRequiredFiles():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train CSV not found: {TRAIN_CSV}")

    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")


def getDataTransforms():
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def createSubset(dataset, max_samples):
    if max_samples is None:
        return dataset

    sample_count = min(max_samples, len(dataset))
    indices = list(range(sample_count))

    return Subset(dataset, indices)


def createDataLoaders():
    train_transform, val_transform = getDataTransforms()

    train_dataset = ParkingDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = ParkingDataset(VAL_CSV, transform=val_transform)

    train_dataset = createSubset(train_dataset, MAX_TRAIN_SAMPLES)
    val_dataset = createSubset(val_dataset, MAX_VAL_SAMPLES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader


def saveTrainingCurve(history):
    history_df = pd.DataFrame(history)

    history_csv_path = REPORT_DIR / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(FIGURE_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(FIGURE_DIR / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTraining outputs saved:")
    print(f"- History CSV    : {history_csv_path}")
    print(f"- Loss curve     : {FIGURE_DIR / 'loss_curve.png'}")
    print(f"- Accuracy curve : {FIGURE_DIR / 'accuracy_curve.png'}")


# *********************** Processing functions
def trainOneEpoch(model, dataloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, 1)

        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        current_accuracy = correct_predictions / total_samples

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{current_accuracy:.4f}"
        })

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validateModel(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def trainCNN():
    print("=" * 80)
    print("Training CNN Model for Smart Parking Occupancy Detection")
    print("=" * 80)

    createOutputFolders()
    checkRequiredFiles()

    print(f"Device      : {DEVICE}")
    print(f"Train CSV   : {TRAIN_CSV}")
    print(f"Val CSV     : {VAL_CSV}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    train_loader, val_loader = createDataLoaders()

    print("\nDataset loaded:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    model = SimpleCNN(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    history = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 80)

        train_loss, train_accuracy = trainOneEpoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validateModel(
            model,
            val_loader,
            criterion
        )

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Accuracy  : {val_accuracy:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_accuracy,
                "epoch": epoch,
                "model_name": "SimpleCNN"
            }, MODEL_PATH)

            print(f"Best model saved: {MODEL_PATH}")

    total_time = time.time() - start_time

    saveTrainingCurve(history)

    print("\n" + "=" * 80)
    print("Training completed successfully.")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Total training time     : {total_time / 60:.2f} minutes")
    print(f"Best model path         : {MODEL_PATH}")


# *********************** Main function
def main():
    trainCNN()


if __name__ == "__main__":
    main()