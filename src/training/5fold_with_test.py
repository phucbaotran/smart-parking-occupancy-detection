# =========================================================
# File name: train_5fold_with_test.py
# Project: Smart Parking Occupancy Detection
# Description:
#   Shared 5-fold Train -> Validation -> Test pipeline
#   for CNRPark+EXT and PKLot.
#
# Flow for each outer fold:
#   1. Hold out one fold as Test.
#   2. Split the remaining four folds into Train and Validation.
#   3. Train SimpleCNN on Train.
#   4. Select the best epoch using Validation accuracy.
#   5. Load the best model and evaluate it once on Test.
#
# Examples:
#   python src\training\train_5fold_with_test.py --dataset cnrpark
#   python src\training\train_5fold_with_test.py --dataset pklot --max-samples 150000
#   python src\training\train_5fold_with_test.py --dataset cnrpark --fold 1
#   python src\training\train_5fold_with_test.py --dataset cnrpark --resume
# =========================================================

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

# Resolve project root automatically:
# src/training/train_5fold_with_test.py -> project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from data.parking_dataset import ParkingDataset
from models.simple_cnn import SimpleCNN


DATASET_CSV_MAP = {
    "cnrpark": PROJECT_ROOT / "data" / "processed" / "final_cnrpark_full.csv",
    "pklot": PROJECT_ROOT / "data" / "processed" / "pklot" / "pklot_all.csv",
}

SPLIT_ROOT = PROJECT_ROOT / "data" / "splits" / "5fold_train_valid_test"
MODEL_ROOT = PROJECT_ROOT / "models" / "experiments" / "5fold_train_valid_test"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "5fold_train_valid_test"

RANDOM_STATE = 42
CLASS_NAMES = ["Free", "Occupied"]
VALID_LABELS = {"free", "occupied"}


# =========================================================
# 2. ARGUMENTS AND REPRODUCIBILITY
# =========================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 5-fold Train -> Validation -> Test evaluation "
            "for CNRPark+EXT or PKLot."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cnrpark", "pklot"],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per fold. Default: 5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size. Default: 16.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate. Default: 0.001.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help=(
            "Fraction of the outer-training portion used for Validation. "
            "Default: 0.20."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional stratified sample limit. Recommended for PKLot, "
            "for example: 150000."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows for stability.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Run only one fold for a trial/debug run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds already recorded in the results CSV.",
    )

    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 3. DATASET LOADING AND SPLITTING
# =========================================================

def validate_dataset_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    sample = pd.read_csv(csv_path, nrows=5)
    required_columns = {"image_path", "label"}
    missing_columns = required_columns.difference(sample.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {csv_path}: {sorted(missing_columns)}"
        )


def load_dataset_dataframe(
    csv_path: Path,
    max_samples: int | None,
) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)

    dataframe = dataframe.dropna(
        subset=["image_path", "label"]
    ).reset_index(drop=True)

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_labels = set(dataframe["label"].unique()) - VALID_LABELS

    if invalid_labels:
        raise ValueError(
            f"Unsupported labels found: {sorted(invalid_labels)}. "
            f"Expected: {sorted(VALID_LABELS)}"
        )

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be greater than 0.")

        if max_samples < len(dataframe):
            sampled_dataframe, _ = train_test_split(
                dataframe,
                train_size=max_samples,
                stratify=dataframe["label"],
                random_state=RANDOM_STATE,
            )
            dataframe = sampled_dataframe.reset_index(drop=True)

    return dataframe


def create_fold_csv_files(
    dataframe: pd.DataFrame,
    outer_train_indices: np.ndarray,
    test_indices: np.ndarray,
    dataset_name: str,
    fold_number: int,
    validation_size: float,
) -> Tuple[Path, Path, Path, Dict[str, int]]:
    """
    One outer fold:
      - test_indices become the independent Test fold.
      - outer_train_indices are split into Train and Validation.
    """
    outer_train_dataframe = dataframe.iloc[
        outer_train_indices
    ].reset_index(drop=True)

    test_dataframe = dataframe.iloc[
        test_indices
    ].reset_index(drop=True)

    train_dataframe, validation_dataframe = train_test_split(
        outer_train_dataframe,
        test_size=validation_size,
        stratify=outer_train_dataframe["label"],
        random_state=RANDOM_STATE + fold_number,
    )

    train_dataframe = train_dataframe.reset_index(drop=True)
    validation_dataframe = validation_dataframe.reset_index(drop=True)

    fold_split_dir = SPLIT_ROOT / dataset_name / f"fold_{fold_number}"
    fold_split_dir.mkdir(parents=True, exist_ok=True)

    train_csv = fold_split_dir / "train.csv"
    validation_csv = fold_split_dir / "validation.csv"
    test_csv = fold_split_dir / "test.csv"

    train_dataframe.to_csv(train_csv, index=False)
    validation_dataframe.to_csv(validation_csv, index=False)
    test_dataframe.to_csv(test_csv, index=False)

    split_sizes = {
        "train_samples": len(train_dataframe),
        "validation_samples": len(validation_dataframe),
        "test_samples": len(test_dataframe),
    }

    return train_csv, validation_csv, test_csv, split_sizes


# =========================================================
# 4. TRANSFORMS AND DATALOADERS
# =========================================================

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, evaluation_transform


def create_data_loaders(
    train_csv: Path,
    validation_csv: Path,
    test_csv: Path,
    batch_size: int,
    num_workers: int,
    device: str,
):
    train_transform, evaluation_transform = get_data_transforms()

    train_dataset = ParkingDataset(
        train_csv,
        transform=train_transform,
    )
    validation_dataset = ParkingDataset(
        validation_csv,
        transform=evaluation_transform,
    )
    test_dataset = ParkingDataset(
        test_csv,
        transform=evaluation_transform,
    )

    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader


# =========================================================
# 5. TRAINING, VALIDATION, AND TESTING
# =========================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct_predictions / total_samples:.4f}",
        })

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    validation_loss = running_loss / total_samples
    validation_accuracy = correct_predictions / total_samples

    return validation_loss, validation_accuracy


def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray, pd.DataFrame]:
    """Evaluate the selected best model once on the Test fold."""
    model.eval()

    true_labels: List[int] = []
    predicted_labels: List[int] = []
    occupied_probabilities: List[float] = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Testing",
            leave=False,
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            probabilities = softmax(outputs)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(
                predictions.cpu().numpy().tolist()
            )
            occupied_probabilities.extend(
                probabilities[:, 1].cpu().numpy().tolist()
            )

    metrics = {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "f1_score": f1_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    predictions_dataframe = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "occupied_probability": occupied_probabilities,
    })

    return metrics, matrix, predictions_dataframe


# =========================================================
# 6. PLOTS
# =========================================================

def save_training_curves(
    history_dataframe: pd.DataFrame,
    fold_figure_dir: Path,
    dataset_name: str,
    fold_number: int,
) -> None:
    accuracy_path = (
        fold_figure_dir
        / f"{dataset_name}_fold_{fold_number}_train_validation_accuracy.png"
    )

    loss_path = (
        fold_figure_dir
        / f"{dataset_name}_fold_{fold_number}_train_validation_loss.png"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["train_accuracy"],
        marker="o",
        label="Training Accuracy",
    )
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["validation_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"{dataset_name.upper()} Fold {fold_number}: "
        "Training and Validation Accuracy"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["train_loss"],
        marker="o",
        label="Training Loss",
    )
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["validation_loss"],
        marker="o",
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{dataset_name.upper()} Fold {fold_number}: "
        "Training and Validation Loss"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()


def save_confusion_matrix(
    matrix: np.ndarray,
    fold_figure_dir: Path,
    dataset_name: str,
    fold_number: int,
) -> None:
    output_path = (
        fold_figure_dir
        / f"{dataset_name}_fold_{fold_number}_test_confusion_matrix.png"
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title(
        f"{dataset_name.upper()} Fold {fold_number}: "
        "Test Confusion Matrix"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_summary_plots(
    results_dataframe: pd.DataFrame,
    dataset_figure_dir: Path,
    dataset_name: str,
) -> None:
    metrics = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]

    x_positions = np.arange(len(results_dataframe))
    bar_width = 0.18

    plt.figure(figsize=(11, 6))

    for metric_index, metric_name in enumerate(metrics):
        plt.bar(
            x_positions + metric_index * bar_width,
            results_dataframe[metric_name],
            width=bar_width,
            label=(
                metric_name
                .replace("test_", "")
                .replace("_", " ")
                .title()
            ),
        )

    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title(
        f"{dataset_name.upper()}: Test Metrics Across 5 Folds"
    )
    plt.xticks(
        x_positions + bar_width * 1.5,
        [f"F{fold}" for fold in results_dataframe["fold"]],
    )
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    metrics_bar_path = (
        dataset_figure_dir
        / f"{dataset_name}_5fold_test_metrics_bar_chart.png"
    )

    plt.savefig(metrics_bar_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [
            results_dataframe["test_accuracy"],
            results_dataframe["test_precision"],
            results_dataframe["test_recall"],
            results_dataframe["test_f1_score"],
        ],
        tick_labels=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
        ],
    )
    plt.ylabel("Score")
    plt.title(
        f"{dataset_name.upper()}: Distribution of Test Metrics"
    )
    plt.ylim(0.0, 1.0)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    boxplot_path = (
        dataset_figure_dir
        / f"{dataset_name}_5fold_test_metrics_boxplot.png"
    )

    plt.savefig(boxplot_path, dpi=300)
    plt.close()


# =========================================================
# 7. REPORT HELPERS
# =========================================================

def save_experiment_configuration(
    configuration: Dict,
    dataset_report_dir: Path,
    dataset_name: str,
) -> None:
    config_path = (
        dataset_report_dir
        / f"{dataset_name}_5fold_experiment_config.json"
    )

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4)


def create_summary_dataframe(
    results_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]

    summary_rows = []

    for metric_name in metric_columns:
        summary_rows.append({
            "metric": metric_name,
            "mean": results_dataframe[metric_name].mean(),
            "std": results_dataframe[metric_name].std(ddof=1),
            "minimum": results_dataframe[metric_name].min(),
            "maximum": results_dataframe[metric_name].max(),
        })

    return pd.DataFrame(summary_rows)


# =========================================================
# 8. MAIN 5-FOLD EXPERIMENT
# =========================================================

def run_five_fold_experiment(args: argparse.Namespace) -> None:
    set_random_seeds(RANDOM_STATE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = args.dataset
    dataset_csv = DATASET_CSV_MAP[dataset_name]

    validate_dataset_csv(dataset_csv)

    dataframe = load_dataset_dataframe(
        dataset_csv,
        args.max_samples,
    )

    dataset_model_dir = MODEL_ROOT / dataset_name
    dataset_output_dir = OUTPUT_ROOT / dataset_name
    dataset_figure_dir = dataset_output_dir / "figures"
    dataset_report_dir = dataset_output_dir / "reports"

    dataset_model_dir.mkdir(parents=True, exist_ok=True)
    dataset_figure_dir.mkdir(parents=True, exist_ok=True)
    dataset_report_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = (
        dataset_report_dir
        / f"{dataset_name}_5fold_test_results.csv"
    )

    summary_csv_path = (
        dataset_report_dir
        / f"{dataset_name}_5fold_test_summary.csv"
    )

    experiment_configuration = {
        "dataset": dataset_name,
        "dataset_csv": str(dataset_csv),
        "total_samples_used": len(dataframe),
        "label_distribution": (
            dataframe["label"].value_counts().to_dict()
        ),
        "outer_folds": 5,
        "inner_validation_size": args.validation_size,
        "epochs_per_fold": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "max_samples": args.max_samples,
        "random_state": RANDOM_STATE,
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device == "cuda"
            else None
        ),
    }

    save_experiment_configuration(
        experiment_configuration,
        dataset_report_dir,
        dataset_name,
    )

    print("=" * 90)
    print("5-Fold Train -> Validation -> Test Experiment")
    print("=" * 90)
    print(f"Dataset              : {dataset_name}")
    print(f"Source CSV           : {dataset_csv}")
    print(f"Samples used         : {len(dataframe)}")
    print("Label distribution   :")
    print(dataframe["label"].value_counts())
    print(f"Device               : {device}")

    if device == "cuda":
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")

    print(f"Epochs per fold      : {args.epochs}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Learning rate        : {args.learning_rate}")
    print(f"Validation proportion: {args.validation_size:.2f}")
    print("=" * 90)

    completed_results: List[Dict] = []

    if args.resume and results_csv_path.exists():
        previous_results = pd.read_csv(results_csv_path)
        completed_results = previous_results.to_dict("records")

        print(
            "\nResume enabled. Completed folds:",
            sorted(previous_results["fold"].astype(int).tolist()),
        )

    completed_fold_numbers = {
        int(result["fold"])
        for result in completed_results
    }

    outer_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    labels = dataframe["label"].values
    total_experiment_start = time.time()

    for fold_number, (outer_train_indices, test_indices) in enumerate(
        outer_splitter.split(dataframe, labels),
        start=1,
    ):
        if args.fold is not None and fold_number != args.fold:
            continue

        if args.resume and fold_number in completed_fold_numbers:
            print(f"\nFold {fold_number} already completed. Skipping.")
            continue

        print("\n" + "=" * 90)
        print(f"FOLD {fold_number}/5")
        print("=" * 90)

        (
            train_csv,
            validation_csv,
            test_csv,
            split_sizes,
        ) = create_fold_csv_files(
            dataframe=dataframe,
            outer_train_indices=outer_train_indices,
            test_indices=test_indices,
            dataset_name=dataset_name,
            fold_number=fold_number,
            validation_size=args.validation_size,
        )

        print(f"Train samples     : {split_sizes['train_samples']}")
        print(f"Validation samples: {split_sizes['validation_samples']}")
        print(f"Test samples      : {split_sizes['test_samples']}")

        (
            train_loader,
            validation_loader,
            test_loader,
        ) = create_data_loaders(
            train_csv=train_csv,
            validation_csv=validation_csv,
            test_csv=test_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        fold_model_dir = dataset_model_dir / f"fold_{fold_number}"
        fold_figure_dir = dataset_figure_dir / f"fold_{fold_number}"
        fold_report_dir = dataset_report_dir / f"fold_{fold_number}"

        fold_model_dir.mkdir(parents=True, exist_ok=True)
        fold_figure_dir.mkdir(parents=True, exist_ok=True)
        fold_report_dir.mkdir(parents=True, exist_ok=True)

        best_model_path = (
            fold_model_dir
            / f"{dataset_name}_fold_{fold_number}_best_model.pth"
        )

        history_csv_path = (
            fold_report_dir
            / f"{dataset_name}_fold_{fold_number}_training_history.csv"
        )

        test_metrics_csv_path = (
            fold_report_dir
            / f"{dataset_name}_fold_{fold_number}_test_metrics.csv"
        )

        predictions_csv_path = (
            fold_report_dir
            / f"{dataset_name}_fold_{fold_number}_test_predictions.csv"
        )

        model = SimpleCNN(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )

        best_validation_accuracy = -1.0
        best_validation_loss = float("inf")
        best_epoch = 0
        training_history: List[Dict] = []

        fold_start = time.time()

        for epoch in range(1, args.epochs + 1):
            print("\n" + "-" * 90)
            print(
                f"Dataset: {dataset_name.upper()} | "
                f"Fold: {fold_number}/5 | "
                f"Epoch: {epoch}/{args.epochs}"
            )
            print("-" * 90)

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            validation_loss, validation_accuracy = validate_model(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.4f}"
            )
            print(
                f"Validation Loss: {validation_loss:.4f} | "
                f"Validation Accuracy: {validation_accuracy:.4f}"
            )

            training_history.append({
                "fold": fold_number,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            })

            validation_improved = (
                validation_accuracy > best_validation_accuracy
            )

            validation_tied_with_lower_loss = (
                np.isclose(
                    validation_accuracy,
                    best_validation_accuracy,
                )
                and validation_loss < best_validation_loss
            )

            if (
                validation_improved
                or validation_tied_with_lower_loss
            ):
                best_validation_accuracy = validation_accuracy
                best_validation_loss = validation_loss
                best_epoch = epoch

                torch.save({
                    "dataset": dataset_name,
                    "fold": fold_number,
                    "epoch": best_epoch,
                    "model_name": "SimpleCNN",
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": best_validation_accuracy,
                    "validation_loss": best_validation_loss,
                    "class_names": ["free", "occupied"],
                }, best_model_path)

                print(f"Best model saved: {best_model_path}")

        history_dataframe = pd.DataFrame(training_history)
        history_dataframe.to_csv(
            history_csv_path,
            index=False,
        )

        save_training_curves(
            history_dataframe=history_dataframe,
            fold_figure_dir=fold_figure_dir,
            dataset_name=dataset_name,
            fold_number=fold_number,
        )

        print(
            "\nLoading best validation model "
            "for independent Test evaluation..."
        )

        checkpoint = torch.load(
            best_model_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_metrics, test_matrix, predictions_dataframe = test_model(
            model=model,
            dataloader=test_loader,
            device=device,
        )

        save_confusion_matrix(
            matrix=test_matrix,
            fold_figure_dir=fold_figure_dir,
            dataset_name=dataset_name,
            fold_number=fold_number,
        )

        predictions_dataframe.to_csv(
            predictions_csv_path,
            index=False,
        )

        fold_time_minutes = (
            time.time() - fold_start
        ) / 60

        fold_result = {
            "dataset": dataset_name,
            "fold": fold_number,
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_loss": best_validation_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "train_samples": split_sizes["train_samples"],
            "validation_samples": split_sizes["validation_samples"],
            "test_samples": split_sizes["test_samples"],
            "fold_time_minutes": fold_time_minutes,
            "best_model_path": str(best_model_path),
        }

        pd.DataFrame([fold_result]).to_csv(
            test_metrics_csv_path,
            index=False,
        )

        completed_results = [
            result
            for result in completed_results
            if int(result["fold"]) != fold_number
        ]
        completed_results.append(fold_result)
        completed_results = sorted(
            completed_results,
            key=lambda result: int(result["fold"]),
        )

        results_dataframe = pd.DataFrame(completed_results)
        results_dataframe.to_csv(
            results_csv_path,
            index=False,
        )

        print("\nIndependent Test Results")
        print(f"Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall   : {test_metrics['recall']:.4f}")
        print(f"F1-score : {test_metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(test_matrix)
        print(f"Fold time: {fold_time_minutes:.2f} minutes")

        del model
        del optimizer
        del train_loader
        del validation_loader
        del test_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results_csv_path.exists():
        print("\nNo fold results were produced.")
        return

    final_results_dataframe = pd.read_csv(results_csv_path)
    final_results_dataframe = final_results_dataframe.sort_values(
        by="fold"
    ).reset_index(drop=True)

    if len(final_results_dataframe) == 5:
        summary_dataframe = create_summary_dataframe(
            final_results_dataframe
        )

        summary_dataframe.to_csv(
            summary_csv_path,
            index=False,
        )

        save_summary_plots(
            results_dataframe=final_results_dataframe,
            dataset_figure_dir=dataset_figure_dir,
            dataset_name=dataset_name,
        )

        total_time_minutes = (
            time.time() - total_experiment_start
        ) / 60

        print("\n" + "=" * 90)
        print("5-FOLD TRAIN -> VALIDATION -> TEST COMPLETED")
        print("=" * 90)
        print(final_results_dataframe[[
            "fold",
            "best_epoch",
            "best_validation_accuracy",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1_score",
        ]])

        print("\nSummary:")
        print(summary_dataframe)

        print(f"\nTotal current run time: {total_time_minutes:.2f} minutes")
        print(f"Fold results : {results_csv_path}")
        print(f"Summary      : {summary_csv_path}")
        print(f"Models       : {dataset_model_dir}")
        print(f"Figures      : {dataset_figure_dir}")
        print(f"Reports      : {dataset_report_dir}")

    else:
        print(
            f"\nCompleted folds currently available: "
            f"{len(final_results_dataframe)}/5"
        )
        print(
            "Run again with --resume to continue the remaining folds."
        )


# =========================================================
# 9. MAIN FUNCTION
# =========================================================

def main() -> None:
    args = parse_arguments()

    if not 0.0 < args.validation_size < 1.0:
        raise ValueError(
            "--validation-size must be between 0 and 1."
        )

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    run_five_fold_experiment(args)


if __name__ == "__main__":
    main()
