# =========================================================
# File name: parking_dataset.py
# Project: Smart Parking Occupancy Detection
# Description: Custom PyTorch Dataset for parking occupancy
# =========================================================


# *********************** Supporting libraries
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# *********************** Constants
REQUIRED_COLUMNS = {
    "image_path",
    "label"
}

LABEL_MAP = {
    "free": 0,
    "occupied": 1
}


# *********************** Supporting functions
def getLabelMap():
    """
    Return the label mapping used by the CNN.

    Class 0: free
    Class 1: occupied
    """

    # Return a copy so other files cannot accidentally
    # modify the original mapping.
    return LABEL_MAP.copy()


# *********************** Processing functions
class ParkingDataset(Dataset):
    """
    Custom PyTorch Dataset for parking occupancy classification.

    Expected CSV columns:
        image_path
        label

    Label mapping:
        free     -> 0
        occupied -> 1
    """

    def __init__(
        self,
        csv_path,
        transform=None,
        root_dir=None
    ):
        """
        Parameters
        ----------
        csv_path:
            Path to the CSV file.

        transform:
            PyTorch image transformation pipeline.

        root_dir:
            Optional root folder used when image paths in the
            CSV are relative paths.
        """

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_path}"
            )

        self.dataframe = pd.read_csv(self.csv_path)

        missing_columns = (
            REQUIRED_COLUMNS
            - set(self.dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if self.dataframe.empty:
            raise ValueError(
                f"CSV contains no samples: {self.csv_path}"
            )

        self.transform = transform
        self.label_map = getLabelMap()

        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else None
        )

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """

        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Read and return one image-label pair.
        """

        row = self.dataframe.iloc[index]

        # -----------------------------------------------------
        # Resolve image path
        # -----------------------------------------------------
        image_path = Path(
            str(row["image_path"])
        )

        if (
            not image_path.is_absolute()
            and self.root_dir is not None
        ):
            image_path = self.root_dir / image_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found at row {index}: "
                f"{image_path}"
            )

        # -----------------------------------------------------
        # Convert text label to numerical label
        # -----------------------------------------------------
        label_name = (
            str(row["label"])
            .strip()
            .lower()
        )

        if label_name not in self.label_map:
            raise ValueError(
                f"Invalid label '{label_name}' "
                f"at row {index}. "
                f"Expected one of: "
                f"{list(self.label_map.keys())}"
            )

        label = self.label_map[label_name]

        # -----------------------------------------------------
        # Read image
        # -----------------------------------------------------
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")

        # -----------------------------------------------------
        # Apply training or inference transform
        # -----------------------------------------------------
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# *********************** Main function
def main():
    """
    Display the dataset configuration.
    """

    print("ParkingDataset file is ready.")
    print("Required CSV columns:", REQUIRED_COLUMNS)
    print("Label mapping:", getLabelMap())


if __name__ == "__main__":
    main()