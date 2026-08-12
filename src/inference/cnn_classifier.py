# =========================================================
# File name: cnn_classifier.py
# Project: Smart Parking Occupancy Detection
# Description:
#   Reusable CNN classifier for cropped parking-slot images.
#
# Supported input types:
#   1. Image file path
#   2. PIL Image
#   3. OpenCV NumPy image
#
# Output:
#   Free / Occupied prediction with confidence scores
# =========================================================


# *********************** Supporting libraries
import sys
import argparse
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms


# *********************** Project configuration

# File location:
# project/src/inference/cnn_classifier.py
#
# Therefore:
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Import the original CNN architecture.
from models.simple_cnn import (
    SimpleCNN,
    getNumberOfClasses
)


# *********************** Default configuration

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "final_cnrpark_cnn.pth"
)

IMAGE_SIZE = (150, 150)

NORMALIZATION_MEAN = [
    0.485,
    0.456,
    0.406
]

NORMALIZATION_STD = [
    0.229,
    0.224,
    0.225
]

CLASS_NAMES = {
    0: "free",
    1: "occupied"
}


# *********************** Supporting functions

def selectDevice(device=None):
    """
    Select the device used for inference.

    Parameters
    ----------
    device:
        None, "cpu", "cuda", or a torch.device object.

    Returns
    -------
    torch.device
    """

    if device is None:
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    selected_device = torch.device(device)

    if (
        selected_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    return selected_device


def getInferenceTransform():
    """
    Return the preprocessing pipeline used for CNN inference.

    This must match the preprocessing used during training.

    RandomHorizontalFlip and ColorJitter are excluded because
    random augmentation must not be used during inference.
    """

    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=NORMALIZATION_MEAN,
            std=NORMALIZATION_STD
        )
    ])

    return image_transform


def convertOpenCvToPil(image_bgr):
    """
    Convert an OpenCV NumPy image into a PIL RGB image.

    OpenCV reads images using BGR channel order.
    PIL and the CNN expect RGB channel order.
    """

    if image_bgr is None:
        raise ValueError(
            "OpenCV image is None. "
            "Check whether the image was read successfully."
        )

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(
            "OpenCV input must be a NumPy array."
        )

    if image_bgr.size == 0:
        raise ValueError(
            "OpenCV image is empty."
        )

    # Convert non-uint8 arrays into uint8 images.
    if image_bgr.dtype != np.uint8:
        image_array = image_bgr.astype(
            np.float32
        )

        # Handle floating-point images in range 0–1.
        if image_array.max() <= 1.0:
            image_array = image_array * 255.0

        image_bgr = np.clip(
            image_array,
            0,
            255
        ).astype(np.uint8)

    # Grayscale image
    if image_bgr.ndim == 2:
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_GRAY2RGB
        )

    # BGR image
    elif (
        image_bgr.ndim == 3
        and image_bgr.shape[2] == 3
    ):
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

    # BGRA image
    elif (
        image_bgr.ndim == 3
        and image_bgr.shape[2] == 4
    ):
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGRA2RGB
        )

    else:
        raise ValueError(
            "Unsupported OpenCV image shape: "
            f"{image_bgr.shape}"
        )

    return Image.fromarray(image_rgb)


# *********************** Processing class

class CNNClassifier:
    """
    Reusable parking occupancy CNN classifier.

    Input:
        Cropped parking-slot image.

    Supported input types:
        - str or Path
        - PIL.Image.Image
        - OpenCV NumPy array in BGR format

    Output example:
        {
            "class_id": 1,
            "label": "occupied",
            "confidence": 0.9821,
            "free_probability": 0.0179,
            "occupied_probability": 0.9821
        }
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        device=None
    ):
        """
        Load the CNN model once when the classifier is created.
        """

        self.model_path = Path(model_path)

        self.device = selectDevice(device)

        self.image_transform = (
            getInferenceTransform()
        )

        self.model = self.loadModel()

    def loadModel(self):
        """
        Create the SimpleCNN architecture and load its weights.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"CNN model not found: {self.model_path}"
            )

        number_of_classes = (
            getNumberOfClasses()
        )

        model = SimpleCNN(
            num_classes=number_of_classes
        )

        # The model is a trusted checkpoint created locally
        # by this project.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # Support the checkpoint dictionary format:
        #
        # {
        #     "model_state_dict": ...,
        #     "optimizer_state_dict": ...,
        #     ...
        # }
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint[
                "model_state_dict"
            ]

        # Also support a checkpoint containing only state_dict.
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError(
                "Unsupported CNN checkpoint format."
            )

        model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(self.device)

        # Disable Dropout and use BatchNorm inference behavior.
        model.eval()

        return model

    def prepareImage(
        self,
        image: Union[
            str,
            Path,
            Image.Image,
            np.ndarray
        ]
    ):
        """
        Convert an input image into a CNN-ready tensor.
        """

        # -------------------------------------------------
        # Input is an image file path
        # -------------------------------------------------
        if isinstance(image, (str, Path)):
            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Input image not found: {image_path}"
                )

            with Image.open(
                image_path
            ) as opened_image:
                pil_image = opened_image.convert(
                    "RGB"
                )

        # -------------------------------------------------
        # Input is already a PIL image
        # -------------------------------------------------
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")

        # -------------------------------------------------
        # Input is an OpenCV NumPy image
        # -------------------------------------------------
        elif isinstance(image, np.ndarray):
            pil_image = convertOpenCvToPil(
                image
            )

        else:
            raise TypeError(
                "Input image must be one of the following: "
                "file path, PIL image, or OpenCV NumPy array."
            )

        image_tensor = self.image_transform(
            pil_image
        )

        # Add batch dimension:
        #
        # [3, 150, 150]
        #       ↓
        # [1, 3, 150, 150]
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(
            self.device
        )

        return image_tensor

    def predict(self, image):
        """
        Predict whether one cropped parking slot is free
        or occupied.
        """

        image_tensor = self.prepareImage(
            image
        )

        with torch.inference_mode():
            outputs = self.model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class_id = int(
                probabilities
                .argmax(dim=1)
                .item()
            )

            free_probability = float(
                probabilities[0, 0].item()
            )

            occupied_probability = float(
                probabilities[0, 1].item()
            )

            confidence_score = float(
                probabilities[
                    0,
                    predicted_class_id
                ].item()
            )

        predicted_label = CLASS_NAMES[
            predicted_class_id
        ]

        result = {
            "class_id":
                predicted_class_id,

            "label":
                predicted_label,

            "confidence":
                confidence_score,

            "free_probability":
                free_probability,

            "occupied_probability":
                occupied_probability
        }

        return result


# *********************** Main function

def main():
    """
    Test CNNClassifier using one cropped parking-slot image.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test the reusable parking occupancy "
            "CNN classifier."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help=(
            "Path to one cropped parking-slot image."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the final CNN checkpoint."
    )

    args = parser.parse_args()

    classifier = CNNClassifier(
        model_path=args.model
    )

    result = classifier.predict(
        args.image
    )

    print("=" * 80)
    print("CNN Parking Occupancy Classification")
    print("=" * 80)
    print(f"Image      : {args.image}")
    print(f"Model      : {classifier.model_path}")
    print(f"Device     : {classifier.device}")
    print(f"Prediction : {result['label']}")
    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )
    print(
        f"Free       : "
        f"{result['free_probability']:.4f}"
    )
    print(
        f"Occupied   : "
        f"{result['occupied_probability']:.4f}"
    )


if __name__ == "__main__":
    main()