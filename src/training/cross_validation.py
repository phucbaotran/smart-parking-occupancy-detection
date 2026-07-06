# =========================================================
# File name: cross_validation.py
# Project: Smart Parking Occupancy Detection
# Description: Full 5-Fold Cross Validation for CNRPark+EXT Dataset
# =========================================================


# *********************** Supporting libraries
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# *********************** Global configuration

# Project root path
PROJECT_ROOT = Path(r"D:\smart-parking-occupancy-detection")

# Add src folder to Python path
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# Import project files
from data.parking_dataset import ParkingDataset
from models.simple_cnn import SimpleCNN


# Dataset name for saving full 5-fold results
# This avoids overwriting the previous 25,000-sample experiment.
DATASET_NAME = "cnrpark_ext_full"

# Processed CSV files created by prepare_dataset.py
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"
TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"

# Output folders
SPLIT_DIR = PROJECT_ROOT / "data" / "splits" / DATASET_NAME
MODEL_DIR = PROJECT_ROOT / "models" / DATASET_NAME
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / DATASET_NAME

# Training configuration
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0

# Cross-validation configuration
N_SPLITS = 5
RANDOM_STATE = 42

# FULL DATASET MODE
# None means the script will use all 144,965 processed images.
MAX_SAMPLES = None

# Resume mode
# If a fold result already exists in cross_validation_results.csv,
# the script will skip that completed fold when restarted.
RESUME_IF_RESULTS_EXIST = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# *********************** Supporting functions

def setRandomSeed(seed):
    """Set random seed for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def createOutputFolders():
    """Create folders for fold CSV files, models, and reports."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def checkRequiredFiles():
    """Check whether required CSV files exist."""
    required_files = [
        TRAI                                                      N_CSV,
        VAL_CSV,
        TEST_CSV
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    print("All required processed CSV files found.")


def loadAllProcessedData():
    """Load train, validation, and test CSV files, then combine them."""
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    all_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    print("\nAll processed data loaded.")
    print(f"Train samples      : {len(train_df)}")
    print(f"Validation samples : {len(val_df)}")
    print(f"Test samples       : {len(test_df)}")
    print(f"Total samples      : {len(all_df)}")

    print("\nLabel distribution:")
    print(all_df["label"].value_counts())

    return all_df


def createSubset(dataframe):
    """Use full dataset or create a subset depending on MAX_SAMPLES."""
    if MAX_SAMPLES is None:
        print("\nUsing FULL dataset for 5-fold cross validation.")
        return dataframe.reset_index(drop=True)

    if MAX_SAMPLES >= len(dataframe):
        print("\nMAX_SAMPLES is larger than dataset size. Using full dataset.")
        return dataframe.reset_index(drop=True)

    sampled_df = dataframe.groupby("label", group_keys=False).apply(
        lambda x: x.sample(
            n=int(MAX_SAMPLES * len(x) / len(dataframe)),
            random_state=RANDOM_STATE
        )
    )

    sampled_df = sampled_df.reset_index(drop=True)

    print(f"\nUsing subset for 5-fold cross validation: {len(sampled_df)} samples")

    print("\nSubset label distribution:")
    print(sampled_df["label"].value_counts())

    return sampled_df


def getDataTransforms():
    """Create image transformations for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
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


def saveFoldCsvFiles(dataframe, train_index, val_index, fold_number):
    """Save train and validation CSV files for one fold."""
    train_df = dataframe.iloc[train_index].reset_index(drop=True)
    val_df = dataframe.iloc[val_index].reset_index(drop=True)

    train_csv_path = SPLIT_DIR / f"fold_{fold_number}_train.csv"
    val_csv_path = SPLIT_DIR / f"fold_{fold_number}_val.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    print(f"\nFold {fold_number} CSV files saved.")
    print(f"Train CSV: {train_csv_path}")
    print(f"Val CSV  : {val_csv_path}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples  : {len(val_df)}")

    return train_csv_path, val_csv_path


def createDataLoaders(train_csv_path, val_csv_path):
    """Create PyTorch DataLoaders for one fold."""
    train_transform, val_transform = getDataTransforms()

    train_dataset = ParkingDataset(
        csv_path=train_csv_path,
        transform=train_transform
    )

    val_dataset = ParkingDataset(
        csv_path=val_csv_path,
        transform=val_transform
    )

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


def getCompletedFolds(results_csv_path):
    """Read completed folds from the result CSV file."""
    if not RESUME_IF_RESULTS_EXIST:
        return set()

    if not results_csv_path.exists():
        return set()

    try:
        results_df = pd.read_csv(results_csv_path)

        if "fold" not in results_df.columns:
            return set()

        completed_folds = set(results_df["fold"].astype(int).tolist())
        return completed_folds

    except Exception:
        return set()


def saveIntermediateResults(fold_results, results_csv_path, summary_csv_path):
    """Save fold results and summary after each completed fold."""
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_csv_path, index=False)

    summary_df = results_df[["accuracy", "precision", "recall", "f1_score"]].agg(
        ["mean", "std"]
    )

    summary_df.to_csv(summary_csv_path)

    return results_df, summary_df


# *********************** Processing functions

def trainOneEpoch(model, train_loader, criterion, optimizer):
    """Train the CNN model for one epoch."""
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def validateModel(model, val_loader, criterion):
    """Validate the CNN model and calculate evaluation metrics."""
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = total_loss / len(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return average_loss, accuracy, precision, recall, f1


def trainSingleFold(fold_number, train_csv_path, val_csv_path):
    """Train and validate the model for one fold."""
    print("\n" + "=" * 80)
    print(f"Starting Fold {fold_number}/{N_SPLITS}")
    print("=" * 80)

    train_loader, val_loader = createDataLoaders(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path
    )

    model = SimpleCNN(num_classes=2).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_f1 = -1.0
    best_metrics = None

    model_path = MODEL_DIR / f"fold_{fold_number}_simple_cnn.pth"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nFold {fold_number} | Epoch {epoch}/{EPOCHS}")

        train_loss, train_accuracy = trainOneEpoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        val_loss, val_accuracy, val_precision, val_recall, val_f1 = validateModel(
            model=model,
            val_loader=val_loader,
            criterion=criterion
        )

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Accuracy  : {val_accuracy:.4f}")
        print(f"Precision : {val_precision:.4f}")
        print(f"Recall    : {val_recall:.4f}")
        print(f"F1-score  : {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1

            best_metrics = {
                "fold": fold_number,
                "best_epoch": epoch,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1
            }

            torch.save({
                "fold": fold_number,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1
            }, model_path)

            print(f"Best model saved: {model_path}")

    return best_metrics


def runFiveFoldCrossValidation():
    """Run full 5-fold cross validation on CNRPark+EXT."""
    setRandomSeed(RANDOM_STATE)
    createOutputFolders()
    checkRequiredFiles()

    results_csv_path = REPORT_DIR / "cross_validation_results.csv"
    summary_csv_path = REPORT_DIR / "cross_validation_summary.csv"

    print("=" * 80)
    print("FULL 5-Fold Cross Validation for CNRPark+EXT")
    print("=" * 80)
    print(f"Device          : {DEVICE}")
    print(f"Epochs per fold : {EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Max samples     : {MAX_SAMPLES}")
    print(f"Dataset name    : {DATASET_NAME}")

    all_df = loadAllProcessedData()
    all_df = createSubset(all_df)

    labels = all_df["label"]

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    completed_folds = getCompletedFolds(results_csv_path)

    if completed_folds:
        print(f"\nResume mode detected. Completed folds: {sorted(completed_folds)}")

    fold_results = []

    if results_csv_path.exists() and RESUME_IF_RESULTS_EXIST:
        previous_results = pd.read_csv(results_csv_path)
        fold_results = previous_results.to_dict("records")

    start_time = time.time()

    for fold_number, (train_index, val_index) in enumerate(
        skf.split(all_df, labels),
        start=1
    ):
        if fold_number in completed_folds:
            print("\n" + "=" * 80)
            print(f"Skipping Fold {fold_number}/{N_SPLITS} because it already exists in results.")
            print("=" * 80)
            continue

        train_csv_path, val_csv_path = saveFoldCsvFiles(
            dataframe=all_df,
            train_index=train_index,
            val_index=val_index,
            fold_number=fold_number
        )

        fold_metrics = trainSingleFold(
            fold_number=fold_number,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path
        )

        fold_results.append(fold_metrics)

        results_df, summary_df = saveIntermediateResults(
            fold_results=fold_results,
            results_csv_path=results_csv_path,
            summary_csv_path=summary_csv_path
        )

        print("\nIntermediate fold results:")
        print(results_df)

        print("\nIntermediate average results:")
        print(summary_df)

    results_df, summary_df = saveIntermediateResults(
        fold_results=fold_results,
        results_csv_path=results_csv_path,
        summary_csv_path=summary_csv_path
    )

    total_time = (time.time() - start_time) / 60

    print("\n" + "=" * 80)
    print("FULL 5-Fold Cross Validation Completed")
    print("=" * 80)

    print("\nFinal fold results:")
    print(results_df)

    print("\nFinal average results:")
    print(summary_df)

    print(f"\nTotal time for this run: {total_time:.2f} minutes")
    print(f"Results saved to: {results_csv_path}")
    print(f"Summary saved to: {summary_csv_path}")


# *********************** Main function

def main():
    runFiveFoldCrossValidation()


if __name__ == "__main__":
    main()