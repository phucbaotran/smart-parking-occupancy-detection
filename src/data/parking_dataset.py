# =========================================================
# File name: parking_dataset.py
# Project: Smart Parking Occupancy Detection
# Description: Custom PyTorch Dataset for CNRPark+EXT
# =========================================================


# *********************** Supporting libraries
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# *********************** Supporting functions
def getLabelMap():
    """
    Convert text label into numerical label.

    free     -> 0
    occupied -> 1
    """

    label_map = {
        "free": 0,
        "occupied": 1
    }

    return label_map


# *********************** Processing functions
class ParkingDataset(Dataset):
    """
    Custom PyTorch Dataset for parking occupancy classification.

    This class reads:
    - image_path
    - label

    from train.csv, val.csv, or test.csv.
    """

    def __init__(self, csv_path, transform=None):
        self.dataframe = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = getLabelMap()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])
        label_name = row["label"]

        label = self.label_map[label_name]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# *********************** Main function
def main():
    print("ParkingDataset file is ready.")

    label_map = getLabelMap()
    print("Label mapping:")
    print(label_map)


if __name__ == "__main__":
    main()