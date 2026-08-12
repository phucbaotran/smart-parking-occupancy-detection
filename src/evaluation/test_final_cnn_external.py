# =========================================================
# File: src/evaluation/test_final_cnn_external.py
# External qualitative testing for the final SimpleCNN
# =========================================================


# =========================================================
# 1. LIBRARIES, PATHS, AND SETTINGS
# =========================================================

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from models.simple_cnn import SimpleCNN


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_cnrpark_cnn.pth"
)

TEST_ROOT = (
    PROJECT_ROOT
    / "demo"
    / "input"
    / "cnn_external_test"
)

CSV_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_cnn_external_predictions.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "final_cnn_external_summary.json"
)

FIGURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "final_cnn_external_examples.png"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

DEFAULT_CLASS_NAMES = [
    "free",
    "occupied",
]

DEFAULT_LABEL_MAP = {
    "free": 0,
    "occupied": 1,
}

DEFAULT_IMAGE_SIZE = (
    150,
    150,
)

DEFAULT_MEAN = [
    0.485,
    0.456,
    0.406,
]

DEFAULT_STD = [
    0.229,
    0.224,
    0.225,
]


# =========================================================
# 2. MODEL AND DATA FUNCTIONS
# =========================================================

def load_checkpoint(device):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Final CNN model not found:\n{MODEL_PATH}"
        )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
        )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
    ):
        raise ValueError(
            "The checkpoint does not contain "
            "'model_state_dict'."
        )

    class_names = checkpoint.get(
        "class_names",
        DEFAULT_CLASS_NAMES,
    )

    label_map = checkpoint.get(
        "label_map",
        DEFAULT_LABEL_MAP,
    )

    image_size = tuple(
        checkpoint.get(
            "image_size",
            DEFAULT_IMAGE_SIZE,
        )
    )

    normalization_mean = checkpoint.get(
        "normalization_mean",
        DEFAULT_MEAN,
    )

    normalization_std = checkpoint.get(
        "normalization_std",
        DEFAULT_STD,
    )

    model = SimpleCNN(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "class_names": class_names,
        "label_map": label_map,
        "image_size": image_size,
        "mean": normalization_mean,
        "std": normalization_std,
    }


def create_transform(
    image_size,
    mean,
    std,
):
    """
    Evaluation transform without augmentation.
    """

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])


def collect_test_images(
    label_map,
):
    test_images = []

    for actual_label in label_map:
        class_folder = (
            TEST_ROOT / actual_label
        )

        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Test folder not found:\n{class_folder}"
            )

        image_paths = sorted([
            path
            for path in class_folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ])

        for image_path in image_paths:
            test_images.append({
                "image_path": image_path,
                "actual_label": actual_label,
            })

    if not test_images:
        raise ValueError(
            f"No external test images found in:\n{TEST_ROOT}"
        )

    return test_images


# =========================================================
# 3. INFERENCE AND OUTPUT FUNCTIONS
# =========================================================

def predict_images(
    model,
    test_images,
    image_transform,
    class_names,
    device,
):
    results = []

    with torch.no_grad():
        for item in test_images:
            image_path = item["image_path"]
            actual_label = item["actual_label"]

            with Image.open(image_path) as image_file:
                rgb_image = image_file.convert("RGB")

            input_tensor = (
                image_transform(rgb_image)
                .unsqueeze(0)
                .to(device)
            )

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[0]

            predicted_index = int(
                probabilities.argmax().item()
            )

            predicted_label = (
                class_names[predicted_index]
            )

            confidence = float(
                probabilities[
                    predicted_index
                ].item()
            )

            free_probability = float(
                probabilities[0].item()
            )

            occupied_probability = float(
                probabilities[1].item()
            )

            is_correct = (
                predicted_label
                == actual_label
            )

            result = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "free_probability": free_probability,
                "occupied_probability": (
                    occupied_probability
                ),
                "correct": is_correct,
            }

            results.append(result)

            status = (
                "CORRECT"
                if is_correct
                else "INCORRECT"
            )

            print(
                f"{image_path.name:<25} "
                f"Actual: {actual_label:<9} "
                f"Predicted: {predicted_label:<9} "
                f"Confidence: {confidence * 100:6.2f}% "
                f"{status}"
            )

    return results


def save_predictions_csv(results):
    CSV_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    field_names = [
        "image_name",
        "image_path",
        "actual_label",
        "predicted_label",
        "confidence",
        "free_probability",
        "occupied_probability",
        "correct",
    ]

    with open(
        CSV_OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(results)


def save_summary(results, device):
    total_images = len(results)

    correct_predictions = sum(
        result["correct"]
        for result in results
    )

    accuracy = (
        correct_predictions / total_images
    )

    free_count = sum(
        result["actual_label"] == "free"
        for result in results
    )

    occupied_count = sum(
        result["actual_label"] == "occupied"
        for result in results
    )

    summary = {
        "evaluation_type": (
            "small_external_qualitative_test"
        ),
        "model": str(MODEL_PATH),
        "dataset_folder": str(TEST_ROOT),
        "device": str(device),
        "total_images": total_images,
        "free_images": free_count,
        "occupied_images": occupied_count,
        "correct_predictions": (
            correct_predictions
        ),
        "incorrect_predictions": (
            total_images
            - correct_predictions
        ),
        "observed_accuracy": accuracy,
        "independent_benchmark": False,
        "note": (
            "This is a small manually selected external "
            "test and should not be treated as the main "
            "quantitative evaluation."
        ),
    }

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SUMMARY_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as summary_file:

        json.dump(
            summary,
            summary_file,
            indent=4,
        )

    return summary


def save_prediction_figure(results):
    total_images = len(results)

    columns = min(
        4,
        total_images,
    )

    rows = (
        total_images
        + columns
        - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * 4,
            rows * 3.5,
        ),
    )

    if total_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for axis, result in zip(
        axes,
        results,
    ):
        image_path = Path(
            result["image_path"]
        )

        with Image.open(image_path) as image_file:
            display_image = (
                image_file
                .convert("RGB")
                .copy()
            )

        axis.imshow(display_image)
        axis.axis("off")

        actual_label = (
            result["actual_label"].capitalize()
        )

        predicted_label = (
            result[
                "predicted_label"
            ].capitalize()
        )

        confidence_percent = (
            result["confidence"] * 100
        )

        title_color = (
            "green"
            if result["correct"]
            else "red"
        )

        axis.set_title(
            f"Actual: {actual_label}\n"
            f"Predicted: {predicted_label} "
            f"({confidence_percent:.1f}%)",
            color=title_color,
            fontsize=10,
        )

    for axis in axes[total_images:]:
        axis.axis("off")

    figure.suptitle(
        "Final CNN External Inference Examples",
        fontsize=14,
    )

    figure.tight_layout()

    FIGURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# =========================================================
# 4. MAIN
# =========================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 90)
    print("FINAL CNN EXTERNAL INFERENCE")
    print("=" * 90)
    print(f"Model  : {MODEL_PATH}")
    print(f"Images : {TEST_ROOT}")
    print(f"Device : {device}")
    print("=" * 90)

    model_information = load_checkpoint(
        device=device
    )

    image_transform = create_transform(
        image_size=(
            model_information["image_size"]
        ),
        mean=model_information["mean"],
        std=model_information["std"],
    )

    test_images = collect_test_images(
        label_map=(
            model_information["label_map"]
        )
    )

    results = predict_images(
        model=model_information["model"],
        test_images=test_images,
        image_transform=image_transform,
        class_names=(
            model_information["class_names"]
        ),
        device=device,
    )

    save_predictions_csv(results)

    summary = save_summary(
        results=results,
        device=device,
    )

    save_prediction_figure(results)

    observed_accuracy = (
        summary["observed_accuracy"]
        * 100
    )

    print("\n" + "=" * 90)
    print("EXTERNAL INFERENCE COMPLETED")
    print("=" * 90)
    print(
        f"Total images  : "
        f"{summary['total_images']}"
    )
    print(
        f"Correct       : "
        f"{summary['correct_predictions']}"
    )
    print(
        f"Incorrect     : "
        f"{summary['incorrect_predictions']}"
    )
    print(
        f"Observed acc. : "
        f"{observed_accuracy:.2f}%"
    )
    print(f"CSV           : {CSV_OUTPUT_PATH}")
    print(f"Summary       : {SUMMARY_OUTPUT_PATH}")
    print(f"Figure        : {FIGURE_OUTPUT_PATH}")
    print()
    print(
        "Note: this is a small external qualitative "
        "test, not the main CNN benchmark."
    )


if __name__ == "__main__":
    main()