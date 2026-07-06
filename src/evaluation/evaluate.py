# =========================================================
# File name: evaluate.py
# Project: Smart Parking Occupancy Detection
# Description: Evaluate trained CNN model on test dataset
# =========================================================


# *********************** Supporting libraries
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm


# *********************** Global configuration
PROJECT_ROOT = Path(r"D:\smart-parking-occupancy-detection")
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from data.parking_dataset import ParkingDataset
from models.simple_cnn import SimpleCNN


TEST_CSV = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_simple_cnn.pth"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["free", "occupied"]


# *********************** Supporting functions
def createOutputFolders():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def checkRequiredFiles():
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


def getTestTransform():
    test_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return test_transform


def loadModel():
    model = SimpleCNN(num_classes=2).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print("Model loaded successfully.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Saved validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    print(f"Saved epoch: {checkpoint.get('epoch', 'N/A')}")

    return model


def saveConfusionMatrix(conf_matrix):
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center"
            )

    output_path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def saveEvaluationReport(metrics_dict, report_text):
    metrics_path = REPORT_DIR / "evaluation_metrics.csv"
    report_path = REPORT_DIR / "classification_report.txt"

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(metrics_path, index=False)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    print(f"Evaluation metrics saved to: {metrics_path}")
    print(f"Classification report saved to: {report_path}")


# *********************** Processing functions
def evaluateModel():
    print("=" * 80)
    print("Evaluating CNN Model on Test Dataset")
    print("=" * 80)

    createOutputFolders()
    checkRequiredFiles()

    print(f"Device: {DEVICE}")
    print(f"Test CSV: {TEST_CSV}")

    test_transform = getTestTransform()

    test_dataset = ParkingDataset(TEST_CSV, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches : {len(test_loader)}")

    model = loadModel()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")

        for images, labels in progress_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="binary")
    recall = recall_score(all_labels, all_predictions, average="binary")
    f1 = f1_score(all_labels, all_predictions, average="binary")

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES
    )

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(report_text)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "model_path": str(MODEL_PATH),
        "test_samples": len(test_dataset)
    }

    saveConfusionMatrix(conf_matrix)
    saveEvaluationReport(metrics_dict, report_text)

    print("\nEvaluation completed successfully.")


# *********************** Main function
def main():
    evaluateModel()


if __name__ == "__main__":
    main()