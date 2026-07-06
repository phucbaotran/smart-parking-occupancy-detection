from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(r"D:\smart-parking-occupancy-detection")

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "cnrpark_ext"
CSV_PATH = DATASET_DIR / "CNRPark+EXT.csv"
IMAGE_ROOT = DATASET_DIR / "CNR-EXT-Patches-150x150" / "PATCHES"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# =========================================================
# 2. FIND ALL IMAGES
# =========================================================

def find_all_images(image_root):
    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(image_root.rglob(f"*{ext}"))

    return image_paths


# =========================================================
# 3. CREATE IMAGE MAP
# =========================================================

def create_image_map(image_paths):
    image_map = {}

    for path in image_paths:
        image_map[path.name] = path

    return image_map


# =========================================================
# 4. EXTRACT FILENAME FROM CSV
# =========================================================

def extract_filename(image_url):
    image_url = str(image_url).replace("\\", "/")
    return image_url.split("/")[-1]


# =========================================================
# 5. PREPARE DATAFRAME
# =========================================================

def prepare_dataframe(df, image_map):
    df = df.copy()

    df["filename"] = df["image_url"].apply(extract_filename)

    df["image_path"] = df["filename"].apply(
        lambda filename: str(image_map.get(filename, ""))
    )

    before_count = len(df)
    df = df[df["image_path"] != ""].copy()
    after_count = len(df)

    print(f"CSV rows before matching : {before_count}")
    print(f"Rows matched with images : {after_count}")
    print(f"Rows removed            : {before_count - after_count}")

    df["label"] = df["occupancy"].apply(
        lambda value: "occupied" if int(value) == 1 else "free"
    )

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nWeather distribution:")
    print(df["weather"].value_counts())

    print("\nCamera distribution:")
    print(df["camera"].value_counts())

    selected_columns = [
        "image_path",
        "filename",
        "label",
        "occupancy",
        "camera",
        "datetime",
        "weather",
        "slot_id",
        "day",
        "hour",
        "minute",
        "month",
        "year",
    ]

    return df[selected_columns]


# =========================================================
# 6. CREATE TRAIN / VAL / TEST SPLIT
# =========================================================

def create_splits(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    print("\nVal label distribution:")
    print(val_df["label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts())

    return train_df, val_df, test_df


# =========================================================
# 7. SAVE CSV FILES
# =========================================================

def save_splits(train_df, val_df, test_df):
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved files:")
    print(train_path)
    print(val_path)
    print(test_path)


# =========================================================
# 8. MAIN
# =========================================================

def main():
    print("=" * 80)
    print("Preparing CNRPark+EXT Dataset")
    print("=" * 80)

    print(f"CSV path  : {CSV_PATH}")
    print(f"Image root: {IMAGE_ROOT}")

    df = pd.read_csv(CSV_PATH, sep=None, engine="python")

    print(f"\nCSV rows: {len(df)}")

    image_paths = find_all_images(IMAGE_ROOT)

    print(f"Images found: {len(image_paths)}")

    image_map = create_image_map(image_paths)

    clean_df = prepare_dataframe(df, image_map)

    train_df, val_df, test_df = create_splits(clean_df)

    save_splits(train_df, val_df, test_df)

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()