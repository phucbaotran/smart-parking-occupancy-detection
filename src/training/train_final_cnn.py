# =========================================================
# File: src/training/train_final_cnn.py
# Train the final SimpleCNN on all CNRPark+EXT images.
# This script does not create or use a validation/test split.
# =========================================================

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# =========================================================
# 1. PROJECT PATHS AND FINAL SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.parking_dataset import ParkingDataset
from models.simple_cnn import SimpleCNN


FULL_DATA_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_cnrpark_full.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_cnrpark_cnn.pth"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_cnrpark_cnn_training_history.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_cnrpark_cnn_training_summary.json"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "final_cnrpark_cnn_training_history.png"
)


# Same configuration as the completed 5-fold experiment.
SEED = 42
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_WORKERS = 0

IMAGE_SIZE = (150, 150)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["free", "occupied"]


# Counts already checked from train.csv + val.csv + test.csv.
EXPECTED_SAMPLES = 144_965

EXPECTED_COUNTS = {
    "free": 65_684,
    "occupied": 79_281,
}


# =========================================================
# 2. CHECKS AND REPRODUCIBILITY
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_output_paths() -> None:
    """
    Stop before accidentally overwriting an earlier final run.
    """

    output_paths = [
        MODEL_PATH,
        HISTORY_PATH,
        SUMMARY_PATH,
        FIGURE_PATH,
    ]

    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing:
        existing_text = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise FileExistsError(
            "Archive these existing final outputs "
            "before running again:\n"
            f"{existing_text}"
        )


def load_and_check_dataframe() -> pd.DataFrame:
    """
    Read and validate the full CNRPark+EXT CSV.
    """

    if not FULL_DATA_CSV.exists():
        raise FileNotFoundError(
            "Full CNRPark+EXT CSV was not found: "
            f"{FULL_DATA_CSV}"
        )

    dataframe = pd.read_csv(FULL_DATA_CSV)

    required_columns = {
        "image_path",
        "label",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing CSV columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe[
        ["image_path", "label"]
    ].isna().any().any():

        raise ValueError(
            "The full CSV contains missing paths or labels."
        )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_labels = (
        set(dataframe["label"].unique())
        - set(CLASS_NAMES)
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid labels: {sorted(invalid_labels)}"
        )

    duplicate_count = int(
        dataframe["image_path"].duplicated().sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate image paths."
        )

    label_counts = {
        label: int(count)
        for label, count
        in dataframe["label"].value_counts().items()
    }

    if len(dataframe) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES:,} samples, "
            f"but found {len(dataframe):,}."
        )

    if label_counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Expected label counts {EXPECTED_COUNTS}, "
            f"but found {label_counts}."
        )

    return dataframe.reset_index(drop=True)


def check_image_files(
    dataframe: pd.DataFrame,
) -> None:
    """
    Stop before training if an image path is broken.
    """

    missing_paths = []

    progress_bar = tqdm(
        dataframe["image_path"],
        desc="Checking image files",
        unit="image",
    )

    for value in progress_bar:
        image_path = Path(str(value))

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not image_path.is_file():
            missing_paths.append(str(image_path))

            if len(missing_paths) == 10:
                break

    if missing_paths:
        missing_text = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )

        raise FileNotFoundError(
            "Some training images were not found:\n"
            f"{missing_text}"
        )


# =========================================================
# 3. DATA AND TRAINING
# =========================================================

def get_training_transform():
    """
    Same training transform used in the 5-fold experiment.
    """

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": (
                f"{correct_predictions / total_samples:.4f}"
            ),
        })

    epoch_loss = (
        running_loss / total_samples
    )

    epoch_accuracy = (
        correct_predictions / total_samples
    )

    return epoch_loss, epoch_accuracy


# =========================================================
# 4. OUTPUT FIGURE
# =========================================================

def save_training_figure(
    history_dataframe: pd.DataFrame,
) -> None:

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].plot(
        history_dataframe["epoch"],
        history_dataframe["train_loss"],
        marker="o",
    )

    axes[0].set_title(
        "Final CNN Training Loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history_dataframe["epoch"],
        history_dataframe["train_accuracy"],
        marker="o",
        color="tab:green",
    )

    axes[1].set_title(
        "Final CNN Training Accuracy"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(
        "SimpleCNN on Full CNRPark+EXT Dataset"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# =========================================================
# 5. MAIN
# =========================================================

def main() -> None:
    set_seed(SEED)
    check_output_paths()

    dataframe = load_and_check_dataframe()

    # Check all image paths before beginning the long run.
    check_image_files(dataframe)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = ParkingDataset(
        csv_path=FULL_DATA_CSV,
        transform=get_training_transform(),
        root_dir=PROJECT_ROOT,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    model = SimpleCNN(
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    label_counts = (
        dataframe["label"].value_counts()
    )

    print("=" * 80)
    print(
        "FINAL SIMPLECNN TRAINING "
        "ON FULL CNRPARK+EXT"
    )
    print("=" * 80)

    print(f"Dataset CSV      : {FULL_DATA_CSV}")
    print(f"Training samples : {len(dataframe):,}")
    print(f"Free             : {int(label_counts['free']):,}")
    print(f"Occupied         : {int(label_counts['occupied']):,}")
    print(f"Device           : {device}")

    if device == "cuda":
        print(
            "GPU              : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Epochs           : {EPOCHS}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print("Validation/Test  : not used")
    print("=" * 80)

    training_history = []
    training_start = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_start = time.time()

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        epoch_time_minutes = (
            time.time() - epoch_start
        ) / 60

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "epoch_time_minutes": (
                epoch_time_minutes
            ),
        }

        training_history.append(
            epoch_result
        )

        # Keep records for every completed epoch.
        pd.DataFrame(
            training_history
        ).to_csv(
            HISTORY_PATH,
            index=False,
        )

        print(
            f"Train loss       : {train_loss:.4f}"
        )

        print(
            "Train accuracy   : "
            f"{train_accuracy:.4f}"
        )

        print(
            "Epoch time       : "
            f"{epoch_time_minutes:.2f} minutes"
        )

    total_time_minutes = (
        time.time() - training_start
    ) / 60

    history_dataframe = pd.DataFrame(
        training_history
    )

    final_epoch = training_history[-1]

    label_distribution = {
        label: int(count)
        for label, count
        in label_counts.items()
    }

    # -----------------------------------------------------
    # Save final model
    # -----------------------------------------------------

    checkpoint = {
        "dataset": "CNRPark+EXT",
        "training_type": (
            "full_dataset_final_training"
        ),
        "model_name": "SimpleCNN",
        "epoch": EPOCHS,
        "model_state_dict": (
            model.state_dict()
        ),
        "class_names": CLASS_NAMES,
        "label_map": {
            "free": 0,
            "occupied": 1,
        },
        "image_size": list(IMAGE_SIZE),
        "normalization_mean": MEAN,
        "normalization_std": STD,
        "dataset_csv": str(
            FULL_DATA_CSV
        ),
        "total_training_samples": (
            len(dataframe)
        ),
        "label_distribution": (
            label_distribution
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": (
            "CrossEntropyLoss"
        ),
        "random_state": SEED,
        "training_history": (
            training_history
        ),
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    # -----------------------------------------------------
    # Save summary report
    # -----------------------------------------------------

    summary = {
        "dataset": "CNRPark+EXT",
        "training_type": (
            "full_dataset_final_training"
        ),
        "model_name": "SimpleCNN",
        "total_training_samples": (
            len(dataframe)
        ),
        "label_distribution": (
            label_distribution
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": device,
        "final_training_loss": (
            final_epoch["train_loss"]
        ),
        "final_training_accuracy": (
            final_epoch["train_accuracy"]
        ),
        "total_training_time_minutes": (
            total_time_minutes
        ),
        "independent_test_performed": False,
        "note": (
            "Training accuracy is training history, "
            "not an independent test result."
        ),
        "model_path": str(MODEL_PATH),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    save_training_figure(
        history_dataframe
    )

    print("\n" + "=" * 80)
    print("FINAL CNN TRAINING COMPLETED")
    print("=" * 80)

    print(f"Model   : {MODEL_PATH}")
    print(f"History : {HISTORY_PATH}")
    print(f"Summary : {SUMMARY_PATH}")
    print(f"Figure  : {FIGURE_PATH}")

    print(
        "Time    : "
        f"{total_time_minutes:.2f} minutes"
    )

    print(
        "No independent test was performed "
        "by this script."
    )


if __name__ == "__main__":
    main()